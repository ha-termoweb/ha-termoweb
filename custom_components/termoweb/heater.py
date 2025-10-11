"""Shared helpers and base entities for TermoWeb heaters."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from typing import Any, Final, cast

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.util import dt as dt_util

from .boost import (
    ALLOWED_BOOST_MINUTES,
    coerce_boost_bool,
    coerce_boost_minutes,
    coerce_boost_remaining_minutes,
    supports_boost,
)
from .const import DOMAIN, signal_ws_data
from .inventory import (
    HEATER_NODE_TYPES,
    Inventory,
    Node,
    build_node_inventory,
    heater_platform_details_from_inventory,
    normalize_node_addr,
    normalize_node_type,
    resolve_record_inventory,
)

_LOGGER = logging.getLogger(__name__)


_BOOST_RUNTIME_KEY: Final = "boost_runtime"
DEFAULT_BOOST_DURATION: Final = 60
_HASS_UNSET: Final[HomeAssistant | None] = cast(HomeAssistant | None, object())


@dataclass(frozen=True, slots=True)
class BoostButtonMetadata:
    """Metadata describing an accumulator boost helper button."""

    minutes: int | None
    unique_suffix: str
    label: str
    icon: str


_BOOST_HOUR_ICON_SUFFIXES: Final[dict[int, str]] = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
}


def format_boost_duration_label(minutes: int) -> str:
    """Return a human-readable label for boost durations."""

    if minutes <= 0:
        return "0 minutes"
    if minutes % 60:
        return f"{minutes} minutes"
    hours = minutes // 60
    suffix = "hour" if hours == 1 else "hours"
    return f"{hours} {suffix}"


def _build_boost_button_metadata() -> tuple[BoostButtonMetadata, ...]:
    """Return the configured metadata describing boost helper buttons."""

    entries: list[BoostButtonMetadata] = []
    for minutes in ALLOWED_BOOST_MINUTES:
        hours, remainder = divmod(minutes, 60)
        icon_suffix = _BOOST_HOUR_ICON_SUFFIXES.get(hours) if remainder == 0 else None
        icon = (
            f"mdi:clock-time-{icon_suffix}-outline"
            if icon_suffix is not None
            else "mdi:clock-outline"
        )
        entries.append(
            BoostButtonMetadata(
                minutes,
                str(minutes),
                f"Boost {format_boost_duration_label(minutes)}",
                icon,
            )
        )
    entries.append(BoostButtonMetadata(None, "cancel", "Cancel boost", "mdi:timer-off"))
    return tuple(entries)


BOOST_BUTTON_METADATA: Final[tuple[BoostButtonMetadata, ...]] = _build_boost_button_metadata()
BOOST_DURATION_OPTIONS: Final[tuple[int, ...]] = ALLOWED_BOOST_MINUTES



@dataclass(frozen=True, slots=True)
class HeaterPlatformDetails:
    """Immutable heater platform metadata resolved from inventory."""

    inventory: Inventory
    address_map: dict[str, list[str]]
    resolve_name: Callable[[str, str], str]

    def __iter__(self) -> Iterator[Any]:
        """Provide tuple-style iteration compatibility."""

        yield self.nodes_by_type
        yield self.address_map
        yield self.resolve_name

    @property
    def nodes_by_type(self) -> dict[str, list[Node]]:
        """Return nodes grouped by type from the immutable inventory."""

        return self.inventory.nodes_by_type

    @property
    def addrs_by_type(self) -> dict[str, list[str]]:
        """Return heater addresses grouped by node type."""

        return self.address_map

    def addresses_for(self, node_type: str) -> list[str]:
        """Return immutable heater addresses for ``node_type``."""

        return list(self.address_map.get(node_type, ()))


def _boost_runtime_store(
    entry_data: MutableMapping[str, Any] | None,
    *,
    create: bool,
) -> dict[str, dict[str, int]]:
    """Return the mutable boost runtime store for ``entry_data``."""

    if not isinstance(entry_data, MutableMapping):
        return {}

    store = entry_data.get(_BOOST_RUNTIME_KEY)
    if isinstance(store, dict):
        return store

    if not create:
        return {}

    new_store: dict[str, dict[str, int]] = {}
    entry_data[_BOOST_RUNTIME_KEY] = new_store
    return new_store


def get_boost_runtime_minutes(
    hass: HomeAssistant,
    entry_id: str,
    node_type: str,
    addr: str,
) -> int | None:
    """Return the stored boost runtime for the specified node."""

    domain_data = hass.data.get(DOMAIN)
    if not isinstance(domain_data, MutableMapping):
        return None

    entry_data = domain_data.get(entry_id)
    if not isinstance(entry_data, MutableMapping):
        return None

    node_type_norm = normalize_node_type(
        node_type,
        use_default_when_falsey=True,
    )
    addr_norm = normalize_node_addr(
        addr,
        use_default_when_falsey=True,
    )
    if not node_type_norm or not addr_norm:
        return None

    store = _boost_runtime_store(entry_data, create=False)
    bucket = store.get(node_type_norm)
    if not isinstance(bucket, MutableMapping):
        return None

    stored = bucket.get(addr_norm)
    minutes = coerce_boost_minutes(stored)
    if minutes is None:
        return None

    return minutes


def set_boost_runtime_minutes(
    hass: HomeAssistant,
    entry_id: str,
    node_type: str,
    addr: str,
    minutes: int | None,
) -> None:
    """Persist ``minutes`` as the preferred boost runtime for ``node``."""

    domain_data = hass.data.get(DOMAIN)
    if not isinstance(domain_data, MutableMapping):
        return

    entry_data = domain_data.get(entry_id)
    if not isinstance(entry_data, MutableMapping):
        return

    node_type_norm = normalize_node_type(
        node_type,
        use_default_when_falsey=True,
    )
    addr_norm = normalize_node_addr(
        addr,
        use_default_when_falsey=True,
    )
    if not node_type_norm or not addr_norm:
        return

    store = _boost_runtime_store(entry_data, create=True)

    if minutes is None:
        bucket = store.get(node_type_norm)
        if isinstance(bucket, MutableMapping):
            bucket.pop(addr_norm, None)
            if not bucket:
                store.pop(node_type_norm, None)
        return

    validated = coerce_boost_minutes(minutes)
    if validated is None:
        return

    bucket = store.setdefault(node_type_norm, {})
    bucket[addr_norm] = validated


def resolve_boost_runtime_minutes(
    hass: HomeAssistant,
    entry_id: str,
    node_type: str,
    addr: str,
    *,
    default: int = DEFAULT_BOOST_DURATION,
) -> int:
    """Return the preferred boost runtime or ``default`` when unset."""

    stored = get_boost_runtime_minutes(hass, entry_id, node_type, addr)
    if stored is not None:
        return stored
    return default


def iter_boost_button_metadata() -> Iterator[BoostButtonMetadata]:
    """Yield the metadata describing boost helper buttons."""

    yield from BOOST_BUTTON_METADATA


def iter_boostable_heater_nodes(
    details: HeaterPlatformDetails,
    *,
    node_types: Iterable[str] | None = None,
    accumulators_only: bool = False,
) -> Iterator[tuple[str, Node, str, str]]:
    """Yield heater nodes that expose boost functionality."""

    filtered_types: Iterable[str] | None = node_types

    if accumulators_only:
        accumulator_types: tuple[str, ...] = ("acm",)
        if filtered_types is None:
            filtered_types = accumulator_types
        else:
            filtered_types = [
                node_type
                for node_type in filtered_types
                if node_type in accumulator_types
            ]
            if not filtered_types:
                return

    for node_type, node, addr_str, base_name in iter_heater_nodes(
        details,
        node_types=filtered_types,
    ):
        if supports_boost(node):
            yield node_type, node, addr_str, base_name


@dataclass(slots=True)
class BoostState:
    """Derived boost metadata for a heater node."""

    active: bool | None
    minutes_remaining: int | None
    end_datetime: datetime | None
    end_iso: str | None
    end_label: str | None


# ruff: noqa: C901
def derive_boost_state(
    settings: Mapping[str, Any] | None, coordinator: Any
) -> BoostState:
    """Return derived boost metadata for ``settings`` using ``coordinator``."""

    source = settings if isinstance(settings, Mapping) else {}

    boost_active = coerce_boost_bool(source.get("boost_active"))
    if boost_active is None:
        boost_active = coerce_boost_bool(source.get("boost"))
    if boost_active is None:
        mode = source.get("mode")
        if isinstance(mode, str):
            boost_active = mode.strip().lower() == "boost"
        else:
            boost_active = False

    boost_day: Any = source.get("boost_end_day")
    boost_minute: Any = source.get("boost_end_min")
    raw_end = source.get("boost_end")
    if isinstance(raw_end, Mapping):
        if boost_day is None:
            boost_day = raw_end.get("day")
        if boost_minute is None:
            boost_minute = raw_end.get("minute")

    boost_end_dt: datetime | None = None
    derived_dt = source.get("boost_end_datetime")
    if isinstance(derived_dt, datetime):
        boost_end_dt = derived_dt

    boost_minutes: int | None = coerce_boost_minutes(source.get("boost_minutes_delta"))
    resolver = getattr(coordinator, "resolve_boost_end", None)
    if (
        callable(resolver)
        and boost_day is not None
        and boost_minute is not None
        and (boost_end_dt is None or boost_minutes is None)
    ):
        try:
            resolved_dt, resolved_minutes = resolver(boost_day, boost_minute)
        except Exception:  # noqa: BLE001 - defensive
            resolved_dt = None
            resolved_minutes = None
        if boost_end_dt is None:
            boost_end_dt = resolved_dt
        if boost_minutes is None:
            boost_minutes = resolved_minutes

    if boost_minutes is None:
        boost_minutes = coerce_boost_remaining_minutes(source.get("boost_remaining"))

    if boost_minutes is None and boost_end_dt is not None:
        delta_seconds = (boost_end_dt - dt_util.now()).total_seconds()
        boost_minutes = int(max(0.0, delta_seconds) // 60)

    if boost_minutes is not None and boost_minutes <= 0:
        boost_minutes = None

    boost_end_iso: str | None = None
    if boost_end_dt is not None:
        try:
            boost_end_iso = boost_end_dt.isoformat()
        except Exception:  # noqa: BLE001 - defensive
            boost_end_iso = None
    elif isinstance(raw_end, str):
        boost_end_iso = raw_end
    elif isinstance(raw_end, Mapping):
        day = raw_end.get("day")
        minute = raw_end.get("minute")
        if callable(resolver) and day is not None and minute is not None:
            try:
                derived_dt, _ = resolver(day, minute)
            except Exception:  # noqa: BLE001 - defensive
                derived_dt = None
            if derived_dt is not None:
                boost_end_dt = derived_dt
                boost_end_iso = derived_dt.isoformat()

    if boost_end_iso is None and boost_minutes is not None:
        try:
            boost_end_dt = dt_util.now() + timedelta(minutes=boost_minutes)
            boost_end_iso = boost_end_dt.isoformat()
        except Exception:  # noqa: BLE001 - defensive
            boost_end_dt = None
            boost_end_iso = None

    if boost_end_dt is None and isinstance(boost_end_iso, str):
        parsed: datetime | None = None
        parser = getattr(dt_util, "parse_datetime", None)
        if callable(parser):
            parsed = parser(boost_end_iso)  # pragma: no cover - defensive
        if parsed is None:
            try:
                parsed = datetime.fromisoformat(boost_end_iso)
            except ValueError:  # pragma: no cover - defensive
                parsed = None
        if parsed is not None:
            boost_end_dt = parsed

    placeholder_iso = boost_end_iso.strip() if isinstance(boost_end_iso, str) else None
    placeholder_detected = False
    if (boost_end_dt is not None and boost_end_dt.year <= 1971) or (placeholder_iso and placeholder_iso.startswith("1970-")):
        placeholder_detected = True

    if placeholder_detected:
        boost_end_dt = None
        boost_end_iso = None

    end_label: str | None = None
    if boost_active is False and boost_end_dt is None and boost_end_iso is None:
        end_label = "Never"

    return BoostState(
        active=boost_active,
        minutes_remaining=boost_minutes,
        end_datetime=boost_end_dt,
        end_iso=boost_end_iso,
        end_label=end_label,
    )


# ruff: enable=C901


class DispatcherSubscriptionHelper:
    """Manage dispatcher subscriptions tied to an entity lifecycle."""

    def __init__(self, owner: CoordinatorEntity) -> None:
        """Initialise the helper for the provided entity."""

        self._owner = owner
        self._unsub: Callable[[], None] | None = None

    def subscribe(
        self,
        hass: HomeAssistant,
        signal: str,
        handler: Callable[[dict], None],
    ) -> None:
        """Subscribe to a dispatcher signal and register clean-up."""

        self.unsubscribe()

        try:
            unsubscribe = async_dispatcher_connect(hass, signal, handler)
        except Exception:  # pragma: no cover - defensive
            _LOGGER.exception(
                "Failed to subscribe to dispatcher signal %s for %s",
                signal,
                getattr(self._owner, "_attr_unique_id", self._owner),
            )
            return

        self._owner.async_on_remove(self.unsubscribe)
        self._unsub = unsubscribe

    def unsubscribe(self) -> None:
        """Remove the dispatcher subscription if it exists."""

        unsubscribe = self._unsub
        if unsubscribe is None:
            return

        self._unsub = None
        try:
            unsubscribe()
        except Exception:  # pragma: no cover - defensive
            _LOGGER.exception(
                "Failed to remove dispatcher listener for %s",
                getattr(self._owner, "_attr_unique_id", self._owner),
            )

    @property
    def is_connected(self) -> bool:
        """Return True when the dispatcher listener is active."""

        return self._unsub is not None


def _iter_nodes(nodes: Any) -> Iterable[Node]:
    """Yield :class:`Node` instances from ``nodes`` when possible."""

    if isinstance(nodes, Iterable) and not isinstance(nodes, dict):
        candidate = list(nodes)
        if candidate and all(isinstance(node, Node) for node in candidate):
            yield from candidate
            return
        nodes = candidate

    try:
        inventory = build_node_inventory(nodes)
    except ValueError:  # pragma: no cover - defensive
        inventory = []

    yield from inventory


def iter_heater_nodes(
    details: HeaterPlatformDetails,
    *,
    node_types: Iterable[str] | None = None,
) -> Iterator[tuple[str, Node, str, str]]:
    """Yield heater node metadata for supported node types."""

    if not isinstance(details, HeaterPlatformDetails):
        return

    resolver = details.resolve_name
    nodes_by_type = details.nodes_by_type

    requested_types: list[str]
    if node_types is None:
        requested_types = list(HEATER_NODE_TYPES)
    elif isinstance(node_types, (str, bytes)):
        candidate = normalize_node_type(node_types, use_default_when_falsey=True)
        requested_types = [candidate] if candidate else []
    else:
        requested_types = [
            normalize_node_type(node_type, use_default_when_falsey=True)
            for node_type in node_types
        ]

    for desired_type in requested_types:
        if not desired_type:
            continue
        bucket = nodes_by_type.get(desired_type, [])
        if not bucket:
            continue
        for node in bucket:
            raw_addr = getattr(node, "addr", "")
            if raw_addr is None:
                continue
            addr_str = normalize_node_addr(raw_addr)
            if not addr_str or addr_str.lower() == "none":
                continue
            resolved_name = resolver(desired_type, addr_str)
            yield desired_type, node, addr_str, resolved_name


def iter_heater_maps(
    coordinator_cache: Mapping[str, Any] | None,
    *,
    map_key: str,
    node_types: Iterable[str] | None = None,
    inventory: Inventory | HeaterPlatformDetails | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield unique heater map dictionaries for ``map_key``."""

    if not isinstance(map_key, str) or not map_key:
        return

    cache = coordinator_cache if isinstance(coordinator_cache, Mapping) else {}
    seen: set[int] = set()

    if node_types is None:
        if isinstance(inventory, HeaterPlatformDetails):
            candidate_types = [
                key for key, values in inventory.addrs_by_type.items() if values
            ]
            desired_types = candidate_types or list(HEATER_NODE_TYPES)
        elif isinstance(inventory, Inventory):
            forward_map, _ = inventory.heater_address_map
            candidate_types = [
                key for key, values in forward_map.items() if values
            ]
            desired_types = candidate_types or list(HEATER_NODE_TYPES)
        else:
            desired_types = list(HEATER_NODE_TYPES)
    else:
        if isinstance(node_types, (str, bytes)):
            node_iter: Iterable[Any]
            node_iter = [node_types]
        else:
            node_iter = node_types
        desired_types = [
            normalize_node_type(candidate, use_default_when_falsey=True)
            for candidate in node_iter
        ]

    settings_map = cache.get("settings")
    if isinstance(settings_map, Mapping) and map_key == "settings":
        for desired in desired_types:
            if not desired:
                continue
            canonical = normalize_node_type(desired, use_default_when_falsey=True)
            lookup_keys: tuple[str, ...]
            if canonical and canonical != desired:
                lookup_keys = (canonical, desired)
            elif canonical:
                lookup_keys = (canonical,)
            else:
                lookup_keys = (desired,)
            for key in lookup_keys:
                section = settings_map.get(key)
                if not isinstance(section, Mapping):
                    continue
                candidate = section if isinstance(section, dict) else dict(section)
                ident = id(candidate)
                if ident in seen:
                    continue
                seen.add(ident)
                yield candidate
                break

    for desired in desired_types:
        if not desired:
            continue
        canonical = normalize_node_type(desired, use_default_when_falsey=True)
        lookup_keys: tuple[str, ...]
        if canonical and canonical != desired:
            lookup_keys = (canonical, desired)
        elif canonical:
            lookup_keys = (canonical,)
        else:
            lookup_keys = (desired,)
        for key in lookup_keys:
            section = cache.get(key)
            if not isinstance(section, Mapping):
                continue
            candidate = section.get(map_key)
            if not isinstance(candidate, Mapping):
                continue
            mapping = candidate if isinstance(candidate, dict) else dict(candidate)
            ident = id(mapping)
            if ident in seen:
                continue
            seen.add(ident)
            yield mapping
            break


def log_skipped_nodes(
    platform_name: str,
    inventory: Inventory | HeaterPlatformDetails | None,
    *,
    logger: logging.Logger | None = None,
    skipped_types: Iterable[str] = ("pmo", "thm"),
) -> None:
    """Log skipped TermoWeb nodes for a given platform."""

    log = logger or _LOGGER
    platform = str(platform_name or "").strip()
    if platform and not platform.lower().endswith("platform"):
        platform = f"{platform} platform"
    elif not platform:
        platform = "platform"

    if isinstance(inventory, HeaterPlatformDetails):
        resolved_inventory = inventory.inventory
    elif isinstance(inventory, Inventory):
        resolved_inventory = inventory
    else:
        resolved_inventory = None

    if resolved_inventory is None:
        return

    nodes_by_type = resolved_inventory.nodes_by_type

    for node_type in skipped_types:
        canonical = normalize_node_type(node_type, use_default_when_falsey=True)
        if not canonical:
            continue
        bucket = nodes_by_type.get(canonical, [])
        if not bucket:
            continue
        addrs = ", ".join(
            sorted(
                filter(
                    None,
                    (
                        normalize_node_addr(getattr(node, "addr", ""))
                        for node in bucket
                    ),
                )
            )
        )
        log.debug(
            "Skipping TermoWeb %s nodes for %s: %s",
            canonical,
            platform,
            addrs or "<no-addr>",
        )


def build_heater_name_map(
    nodes: Any, default_factory: Callable[[str], str]
) -> dict[Any, Any]:
    """Return a mapping of heater node identifiers to friendly names.

    The mapping exposes multiple lookup styles:

    * ``(node_type, addr)`` -> resolved friendly name
    * ``"htr"`` -> legacy heater-only mapping of ``addr`` -> friendly name
    * ``"by_type"`` -> ``{node_type: {addr: name}}`` for all heater nodes
    """

    by_type: dict[str, dict[str, str]] = {}
    by_node: dict[tuple[str, str], str] = {}

    for node in _iter_nodes(nodes):
        node_type = normalize_node_type(getattr(node, "type", ""))
        if node_type not in HEATER_NODE_TYPES:
            continue

        addr = normalize_node_addr(getattr(node, "addr", ""))
        if not addr or addr.lower() == "none":
            continue

        default_name = default_factory(addr)
        raw_name = getattr(node, "name", "")
        resolved = default_name
        if isinstance(raw_name, str):
            candidate = raw_name.strip()
            if candidate:
                resolved = candidate

        by_node[(node_type, addr)] = resolved
        bucket = by_type.setdefault(node_type, {})
        bucket[addr] = resolved

    result: dict[Any, Any] = {"htr": dict(by_type.get("htr", {}))}
    if by_type:
        result["by_type"] = {k: dict(v) for k, v in by_type.items()}

    result.update(by_node)

    return result


def _extract_inventory(entry_data: Mapping[str, Any] | None) -> Inventory | None:
    """Return the shared inventory stored alongside a config entry."""

    if not isinstance(entry_data, Mapping):
        return None

    candidate = entry_data.get("inventory")
    if isinstance(candidate, Inventory):
        return candidate

    hass = entry_data.get("hass")
    entry_id = entry_data.get("entry_id")
    domain_data: Mapping[str, Any] | None = None
    if hass is not None and isinstance(entry_id, str) and entry_id:
        hass_data = getattr(hass, "data", None)
        if isinstance(hass_data, Mapping):
            domain_bucket = hass_data.get(DOMAIN)
            if isinstance(domain_bucket, Mapping):
                domain_data = domain_bucket.get(entry_id)
    if isinstance(domain_data, Mapping):
        candidate = domain_data.get("inventory")
        if isinstance(candidate, Inventory):
            return candidate

    coordinator = entry_data.get("coordinator")
    candidate = getattr(coordinator, "inventory", None)
    if isinstance(candidate, Inventory):
        return candidate

    return None


def resolve_entry_inventory(
    entry_data: Mapping[str, Any] | None,
) -> Inventory | None:
    """Return the shared inventory stored alongside a config entry."""

    return _extract_inventory(entry_data)


def heater_platform_details_for_entry(
    entry_data: Mapping[str, Any] | None,
    *,
    default_name_simple: Callable[[str], str],
) -> HeaterPlatformDetails:
    """Return heater platform metadata derived from ``entry_data``."""

    inventory = _extract_inventory(entry_data)
    if inventory is None:
        dev_id: str | None = None
        if isinstance(entry_data, Mapping):
            dev_id = entry_data.get("dev_id")  # type: ignore[assignment]
        _LOGGER.error(
            "TermoWeb heater setup missing inventory for device %s",
            (dev_id or "<unknown>") if isinstance(dev_id, str) and dev_id else "<unknown>",
        )
        raise ValueError("TermoWeb inventory unavailable for heater platform")

    _, addrs_by_type, resolve_name = heater_platform_details_from_inventory(
        inventory,
        default_name_simple=default_name_simple,
    )
    return HeaterPlatformDetails(
        inventory=inventory,
        address_map=addrs_by_type,
        resolve_name=resolve_name,
    )


class HeaterNodeBase(CoordinatorEntity):
    """Base entity implementing common TermoWeb heater behaviour."""

    def __init__(
        self,
        coordinator: Any,
        entry_id: str,
        dev_id: str,
        addr: str,
        name: str,
        unique_id: str | None = None,
        *,
        device_name: str | None = None,
        node_type: str | None = None,
    ) -> None:
        """Initialise a heater entity tied to a TermoWeb device."""
        super().__init__(coordinator)
        self._hass: HomeAssistant | None = cast(HomeAssistant | None, _HASS_UNSET)
        self._entry_id = entry_id
        self._dev_id = dev_id
        self._addr = normalize_node_addr(addr)
        self._attr_name = name
        resolved_type = (
            normalize_node_type(
                node_type,
                default="htr",
                use_default_when_falsey=True,
            )
            or "htr"
        )
        self._node_type = resolved_type
        self._attr_unique_id = (
            unique_id or f"{DOMAIN}:{dev_id}:{resolved_type}:{self._addr}"
        )
        self._device_name = device_name or name
        self._ws_subscription = DispatcherSubscriptionHelper(self)

    async def async_added_to_hass(self) -> None:
        """Subscribe to websocket updates once the entity is added to hass."""
        coordinator = getattr(self, "coordinator", None)
        if hasattr(coordinator, "async_add_listener"):
            await super().async_added_to_hass()
        else:
            setattr(self, "_async_unsub_coordinator_update", None)
        hass = self._hass_for_runtime()
        if hass is None:
            return

        signal = signal_ws_data(self._entry_id)
        self._ws_subscription.subscribe(hass, signal, self._handle_ws_message)

    async def async_will_remove_from_hass(self) -> None:
        """Tidy up websocket listeners before the entity is removed."""
        self._ws_subscription.unsubscribe()
        coordinator = getattr(self, "coordinator", None)
        if hasattr(coordinator, "async_add_listener"):
            await super().async_will_remove_from_hass()
        else:
            unsub = getattr(self, "_async_unsub_coordinator_update", None)
            if callable(unsub):
                try:
                    unsub()
                finally:
                    setattr(self, "_async_unsub_coordinator_update", None)

    @callback
    def _handle_ws_message(self, payload: dict) -> None:
        """Process websocket payloads addressed to this heater."""
        if not self._payload_matches_heater(payload):
            return
        self._handle_ws_event(payload)

    def _payload_matches_heater(self, payload: dict) -> bool:
        """Return True when the websocket payload targets this heater."""
        if payload.get("dev_id") != self._dev_id:
            return False
        payload_type = payload.get("node_type")
        if payload_type is not None:
            payload_type_str = normalize_node_type(payload_type)
            if payload_type_str and payload_type_str != self._node_type:
                return False
        addr = payload.get("addr")
        if addr is None:
            return True

        payload_addr = normalize_node_addr(addr)
        if not payload_addr:
            return not self._addr

        return payload_addr == self._addr

    @callback
    def _handle_ws_event(self, _payload: dict) -> None:
        """Schedule a state refresh after a websocket update."""

        if getattr(self, "_removed", False):
            return

        callback = getattr(self, "schedule_update_ha_state", None)
        if not callable(callback):
            return

        hass = self._hass_for_runtime()
        loop = getattr(hass, "loop", None)
        if loop and loop.is_closed():
            return
        if loop is None and not hasattr(callback, "call_count"):
            return
        callback()

    @property
    def should_poll(self) -> bool:
        """Home Assistant should not poll heater entities."""
        return False

    @property
    def available(self) -> bool:
        """Return whether the backing device exposes heater data."""
        return self._device_available(self._device_record())

    def _device_available(self, device_entry: dict[str, Any] | None) -> bool:
        """Return True when ``device_entry`` provides heater metadata for this node."""

        if not isinstance(device_entry, Mapping):
            return False

        node_type = getattr(self, "_node_type", "htr")

        if self._extract_device_addresses(device_entry, node_type):
            return True

        resolution = resolve_record_inventory(device_entry)
        inventory = resolution.inventory if resolution is not None else None
        supporting_metadata = isinstance(inventory, Inventory)

        settings = device_entry.get("settings")
        if isinstance(settings, Mapping):
            node_settings = settings.get(node_type)
            if isinstance(node_settings, Mapping) and node_settings:
                return True

        return supporting_metadata

    def _device_record(self) -> dict[str, Any] | None:
        """Return the coordinator cache entry for this device."""
        data = getattr(self.coordinator, "data", {}) or {}
        getter = getattr(data, "get", None)

        if callable(getter):
            record = getter(self._dev_id)
        elif isinstance(data, dict):
            record = dict.get(data, self._dev_id)
        else:
            return None

        return record if isinstance(record, dict) else None

    def _heater_section(self) -> dict[str, Any]:
        """Return the heater-specific metadata cached for this entity."""

        record = self._device_record()
        if not isinstance(record, Mapping):
            return {}

        node_type = getattr(self, "_node_type", "htr")
        addresses = self._extract_device_addresses(record, node_type)

        settings = {}
        cached_settings = record.get("settings")
        if isinstance(cached_settings, Mapping):
            node_settings = cached_settings.get(node_type)
            if isinstance(node_settings, Mapping):
                settings = node_settings

        if not addresses and settings:
            addresses = list(settings)

        return {"addrs": addresses, "settings": settings}

    def heater_settings(self) -> dict[str, Any] | None:
        """Return the cached settings for this heater, if available."""
        section = self._heater_section()
        settings_map = section.get("settings")
        if not isinstance(settings_map, Mapping):
            return None
        settings = settings_map.get(self._addr)
        return settings if isinstance(settings, dict) else None

    @staticmethod
    def _normalise_addresses(addresses: Iterable[Any]) -> list[str]:
        """Return a list of normalised addresses from ``addresses``."""

        normalised: list[str] = []
        seen: set[str] = set()

        for candidate in addresses:
            addr = normalize_node_addr(candidate, use_default_when_falsey=True)
            if not addr or addr in seen:
                continue
            seen.add(addr)
            normalised.append(addr)

        return normalised

    def _extract_device_addresses(
        self, device_entry: Mapping[str, Any], node_type: str
    ) -> list[str]:
        """Return all known addresses for ``node_type`` from ``device_entry``."""

        addresses: list[str] = []
        seen: set[str] = set()

        def _add(candidates: Iterable[Any]) -> None:
            for addr in self._normalise_addresses(candidates):
                if addr in seen:
                    continue
                seen.add(addr)
                addresses.append(addr)

        resolution = resolve_record_inventory(device_entry)
        inventory = resolution.inventory if resolution is not None else None
        if isinstance(inventory, Inventory):
            forward_map, _ = inventory.heater_address_map
            _add(forward_map.get(node_type, ()))
            if not addresses:
                node_bucket = inventory.nodes_by_type.get(node_type, [])
                _add(getattr(node, "addr", "") for node in node_bucket)

        return addresses

    def _hass_for_runtime(self) -> HomeAssistant | None:
        """Return the best-effort Home Assistant instance for runtime access."""

        hass_attr = getattr(self, "_hass", _HASS_UNSET)
        if hass_attr is not _HASS_UNSET:
            return hass_attr
        coordinator_hass = getattr(self.coordinator, "hass", None)
        return cast(HomeAssistant | None, coordinator_hass)

    @property
    def hass(self) -> HomeAssistant | None:
        """Return the Home Assistant instance, falling back to the coordinator."""

        hass_attr = getattr(self, "_hass", _HASS_UNSET)
        if hass_attr is not _HASS_UNSET:
            return hass_attr
        coordinator_hass = getattr(self.coordinator, "hass", None)
        return cast(HomeAssistant | None, coordinator_hass)

    @hass.setter
    def hass(self, value: HomeAssistant | None) -> None:
        """Store the Home Assistant reference for runtime access."""

        if value is None:
            self._hass = None
            return
        self._hass = value

    def boost_state(self) -> BoostState:
        """Return derived boost metadata for this heater."""

        settings = self.heater_settings() or {}
        return derive_boost_state(settings, self.coordinator)

    def _client(self) -> Any:
        """Return the REST client used for write operations."""
        hass = self._hass_for_runtime()
        if hass is None:
            return None
        hass_data = getattr(hass, "data", None)
        if not isinstance(hass_data, dict):
            return None
        entry_bucket = hass_data.get(DOMAIN, {})
        if not isinstance(entry_bucket, dict):
            return None
        entry_data = entry_bucket.get(self._entry_id, {})
        if not isinstance(entry_data, dict):
            return None
        return entry_data.get("client")

    def _units(self) -> str:
        """Return the configured temperature units for this heater."""
        settings = self.heater_settings() or {}
        units = (settings.get("units") or "C").upper()
        return "C" if units not in {"C", "F"} else units

    @property
    def device_info(self) -> DeviceInfo:
        """Expose Home Assistant device metadata for the heater."""
        model = "Accumulator" if self._node_type == "acm" else "Heater"
        return DeviceInfo(
            identifiers=cast(
                set[tuple[str, str]], {(DOMAIN, self._dev_id, self._addr)}
            ),
            name=self._device_name,
            manufacturer="TermoWeb",
            model=model,
            via_device=(DOMAIN, self._dev_id),
        )
