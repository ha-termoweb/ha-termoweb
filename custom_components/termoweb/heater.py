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
    coerce_boost_bool,
    coerce_boost_minutes,
    coerce_boost_remaining_minutes,
    iter_inventory_heater_metadata,
    supports_boost,
)
from .const import DOMAIN, signal_ws_data
from .inventory import (
    Inventory,
    Node,
    normalize_node_addr,
    normalize_node_type,
    resolve_record_inventory,
)
from .utils import float_or_none

_LOGGER = logging.getLogger(__name__)


_BOOST_RUNTIME_KEY: Final = "boost_runtime"
_BOOST_TEMPERATURE_KEY: Final = "boost_temperature"
_CLIMATE_ENTITY_KEY: Final = "climate_entities"
DEFAULT_BOOST_DURATION: Final = 60
DEFAULT_BOOST_TEMPERATURE: Final = 20.0
_HASS_UNSET: Final[HomeAssistant | None] = cast(HomeAssistant | None, object())


@dataclass(frozen=True, slots=True)
class BoostButtonMetadata:
    """Metadata describing an accumulator boost helper button."""

    minutes: int | None
    unique_suffix: str
    label: str
    icon: str
    action: str = "start"


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

    return (
        BoostButtonMetadata(
            None,
            "start",
            "Start boost",
            "mdi:flash-outline",
            action="start",
        ),
        BoostButtonMetadata(
            None,
            "cancel",
            "Cancel boost",
            "mdi:timer-off",
            action="cancel",
        ),
    )


BOOST_BUTTON_METADATA: Final[tuple[BoostButtonMetadata, ...]] = _build_boost_button_metadata()



@dataclass(frozen=True, slots=True)
class HeaterPlatformDetails:
    """Immutable heater platform metadata resolved from inventory."""

    inventory: Inventory
    default_name_simple: Callable[[str], str]

    def __iter__(self) -> Iterator[Any]:
        """Provide tuple-style iteration compatibility."""

        yield self.nodes_by_type
        yield self.addrs_by_type
        yield self.resolve_name

    @property
    def nodes_by_type(self) -> dict[str, list[Node]]:
        """Return nodes grouped by type from the immutable inventory."""

        return self.inventory.nodes_by_type

    @property
    def addrs_by_type(self) -> dict[str, list[str]]:
        """Return heater addresses grouped by node type."""

        forward_map, _ = self.inventory.heater_address_map
        return forward_map

    def addresses_for(self, node_type: str) -> list[str]:
        """Return immutable heater addresses for ``node_type``."""

        return list(self.addrs_by_type.get(node_type, ()))

    def resolve_name(self, node_type: str, addr: str) -> str:
        """Resolve the friendly name for ``(node_type, addr)``."""

        return self.inventory.resolve_heater_name(
            node_type,
            addr,
            default_factory=self.default_name_simple,
        )

    def iter_metadata(self) -> Iterator[tuple[str, Node, str, str]]:
        """Yield heater metadata derived from the inventory."""

        for metadata in iter_inventory_heater_metadata(
            self.inventory,
            default_name_simple=self.default_name_simple,
        ):
            if hasattr(metadata, "node_type"):
                yield metadata.node_type, metadata.node, metadata.addr, metadata.name
                continue

            if not isinstance(metadata, tuple) or len(metadata) != 4:
                _LOGGER.debug(
                    "Skipping unexpected heater metadata payload: %r",
                    metadata,
                )
                continue

            node_type, addr, name, node = metadata
            yield node_type, node, addr, name


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


def _boost_temperature_store(
    entry_data: MutableMapping[str, Any] | None,
    *,
    create: bool,
) -> dict[str, dict[str, float]]:
    """Return the mutable boost temperature store for ``entry_data``."""

    if not isinstance(entry_data, MutableMapping):
        return {}

    store = entry_data.get(_BOOST_TEMPERATURE_KEY)
    if isinstance(store, dict):
        return store

    if not create:
        return {}

    new_store: dict[str, dict[str, float]] = {}
    entry_data[_BOOST_TEMPERATURE_KEY] = new_store
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


def get_boost_temperature(
    hass: HomeAssistant,
    entry_id: str,
    node_type: str,
    addr: str,
) -> float | None:
    """Return the stored boost temperature for the specified node."""

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

    store = _boost_temperature_store(entry_data, create=False)
    bucket = store.get(node_type_norm)
    if not isinstance(bucket, MutableMapping):
        return None

    return float_or_none(bucket.get(addr_norm))


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


def set_boost_temperature(
    hass: HomeAssistant,
    entry_id: str,
    node_type: str,
    addr: str,
    temperature: float,
) -> None:
    """Persist ``temperature`` as the preferred boost setpoint."""

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

    store = _boost_temperature_store(entry_data, create=True)
    bucket = store.setdefault(node_type_norm, {})
    if not isinstance(bucket, MutableMapping):
        bucket = {}
        store[node_type_norm] = bucket

    bucket[addr_norm] = float(temperature)


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


def resolve_boost_temperature(
    hass: HomeAssistant,
    entry_id: str,
    node_type: str,
    addr: str,
    *,
    default: float | None = None,
) -> float | None:
    """Return the preferred boost temperature for ``node`` or ``default``."""

    stored = get_boost_temperature(hass, entry_id, node_type, addr)
    if stored is not None:
        return stored
    return default


def _climate_entity_store(
    entry_data: MutableMapping[str, Any] | None,
    *,
    create: bool,
) -> dict[str, dict[str, str]]:
    """Return the mutable climate entity ID store for ``entry_data``."""

    if not isinstance(entry_data, MutableMapping):
        return {}

    store = entry_data.get(_CLIMATE_ENTITY_KEY)
    if isinstance(store, dict):
        return store

    if not create:
        return {}

    new_store: dict[str, dict[str, str]] = {}
    entry_data[_CLIMATE_ENTITY_KEY] = new_store
    return new_store


def register_climate_entity_id(
    hass: HomeAssistant,
    entry_id: str,
    node_type: str,
    addr: str,
    entity_id: str | None,
) -> None:
    """Record the climate entity ID for ``(entry_id, node_type, addr)``."""

    if not entity_id:
        return

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

    store = _climate_entity_store(entry_data, create=True)
    bucket = store.setdefault(node_type_norm, {})
    if not isinstance(bucket, MutableMapping):
        bucket = {}
        store[node_type_norm] = bucket

    bucket[addr_norm] = str(entity_id)


def clear_climate_entity_id(
    hass: HomeAssistant,
    entry_id: str,
    node_type: str,
    addr: str,
) -> None:
    """Remove the recorded climate entity ID for ``(entry_id, node_type, addr)``."""

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

    store = _climate_entity_store(entry_data, create=False)
    bucket = store.get(node_type_norm)
    if not isinstance(bucket, MutableMapping):
        return

    bucket.pop(addr_norm, None)


def resolve_climate_entity_id(
    hass: HomeAssistant,
    entry_id: str,
    node_type: str,
    addr: str,
) -> str | None:
    """Return the recorded climate entity ID for ``(entry_id, node_type, addr)``."""

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

    store = _climate_entity_store(entry_data, create=False)
    bucket = store.get(node_type_norm)
    if not isinstance(bucket, MutableMapping):
        return None

    entity_id = bucket.get(addr_norm)
    return str(entity_id) if isinstance(entity_id, str) else None


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

    if isinstance(details, HeaterPlatformDetails):
        metadata_iter = details.iter_metadata()
    elif isinstance(details, Inventory):  # pragma: no cover - compatibility shim
        metadata_iter = (
            (meta.node_type, meta.node, meta.addr, meta.name)
            for meta in iter_inventory_heater_metadata(details)
        )
    else:
        return

    if node_types is None:
        filter_types: set[str] | None = None
    elif isinstance(node_types, (str, bytes)):
        normalized = normalize_node_type(node_types, use_default_when_falsey=True)
        filter_types = {normalized} if normalized else set()
    else:
        filter_types = {
            normalize_node_type(candidate, use_default_when_falsey=True)
            for candidate in node_types
        }
        filter_types.discard("")

    for node_type, node, addr_str, base_name in metadata_iter:
        if filter_types is not None and node_type not in filter_types:
            continue
        if accumulators_only and node_type != "acm":
            continue
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

    addresses_by_type = resolved_inventory.addresses_by_type

    for node_type in skipped_types:
        canonical = normalize_node_type(node_type, use_default_when_falsey=True)
        if not canonical:
            continue
        addresses = addresses_by_type.get(canonical, [])
        if not addresses:
            continue
        addrs = ", ".join(sorted(addresses))
        log.debug(
            "Skipping TermoWeb %s nodes for %s: %s",
            canonical,
            platform,
            addrs or "<no-addr>",
        )


def resolve_entry_inventory(
    entry_data: Mapping[str, Any] | None,
) -> Inventory | None:
    """Return the shared inventory stored alongside a config entry."""

    if not isinstance(entry_data, Mapping):
        return None

    candidate = entry_data.get("inventory")
    if isinstance(candidate, Inventory):
        return candidate

    coordinator = entry_data.get("coordinator")
    candidate = getattr(coordinator, "inventory", None)
    if isinstance(candidate, Inventory):
        return candidate

    return None


def heater_platform_details_for_entry(
    entry_data: Mapping[str, Any] | None,
    *,
    default_name_simple: Callable[[str], str],
) -> HeaterPlatformDetails:
    """Return heater platform metadata derived from ``entry_data``."""

    inventory = resolve_entry_inventory(entry_data)
    if inventory is None:
        dev_id: str | None = None
        if isinstance(entry_data, Mapping):
            dev_id = entry_data.get("dev_id")  # type: ignore[assignment]
        _LOGGER.error(
            "TermoWeb heater setup missing inventory for device %s",
            (dev_id or "<unknown>") if isinstance(dev_id, str) and dev_id else "<unknown>",
        )
        raise ValueError("TermoWeb inventory unavailable for heater platform")

    return HeaterPlatformDetails(
        inventory=inventory,
        default_name_simple=default_name_simple,
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
        inventory: Inventory | None = None,
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
        self._inventory: Inventory | None = inventory

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

        inventory = self._resolve_inventory()
        if not isinstance(inventory, Inventory):
            return False

        node_type = getattr(self, "_node_type", "htr")
        forward_map, _ = inventory.heater_address_map
        return self._addr in forward_map.get(node_type, [])

    def _device_record(self) -> dict[str, Any] | None:
        """Return the coordinator cache entry for this device."""
        data = getattr(self.coordinator, "data", {}) or {}
        getter = getattr(data, "get", None)

        try:
            if callable(getter):
                record = getter(self._dev_id)
            elif isinstance(data, dict):
                record = dict.get(data, self._dev_id)
            else:
                return None
        except Exception:  # pragma: no cover - defensive
            _LOGGER.debug(
                "Failed to resolve device record for %s", self._dev_id, exc_info=True
            )
            return None

        return record if isinstance(record, dict) else None

    def _resolve_inventory(self) -> Inventory | None:
        """Return the cached inventory for this entity, if available."""

        inventory = getattr(self, "_inventory", None)
        if isinstance(inventory, Inventory):
            return inventory

        coordinator_inventory = getattr(self.coordinator, "inventory", None)
        if isinstance(coordinator_inventory, Inventory):
            self._inventory = coordinator_inventory
            return coordinator_inventory

        record = self._device_record()
        if isinstance(record, Mapping):
            resolution = resolve_record_inventory(record)
            candidate = resolution.inventory if resolution is not None else None
            if isinstance(candidate, Inventory):
                self._inventory = candidate
                return candidate

        return None

    def _heater_section(self) -> dict[str, Any]:
        """Return the heater-specific metadata cached for this entity."""

        record = self._device_record()
        if not isinstance(record, Mapping):
            return {}

        node_type = getattr(self, "_node_type", "htr")
        addresses: list[str] = []
        inventory = self._resolve_inventory()
        if isinstance(inventory, Inventory):
            forward_map, _ = inventory.heater_address_map
            addresses = list(forward_map.get(node_type, []))

        settings = {}
        cached_settings = record.get("settings")
        if isinstance(cached_settings, Mapping):
            node_settings = cached_settings.get(node_type)
            if isinstance(node_settings, Mapping):
                settings = node_settings

        if not addresses and settings:
            addresses = [addr for addr in settings if isinstance(addr, str)]

        return {"addrs": addresses, "settings": settings}

    def heater_settings(self) -> dict[str, Any] | None:
        """Return the cached settings for this heater, if available."""
        section = self._heater_section()
        settings_map = section.get("settings")
        if not isinstance(settings_map, Mapping):
            return None
        settings = settings_map.get(self._addr)
        return settings if isinstance(settings, dict) else None

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
