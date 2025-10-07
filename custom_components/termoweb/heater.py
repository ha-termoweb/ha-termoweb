"""Shared helpers and base entities for TermoWeb heaters."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from typing import Any, Final

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import dispatcher
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.util import dt as dt_util

from .const import DOMAIN, signal_ws_data
from .installation import InstallationSnapshot, ensure_snapshot
from .nodes import (
    HEATER_NODE_TYPES,
    Node,
    build_heater_address_map,
    build_node_inventory,
    ensure_node_inventory,
    normalize_node_addr,
    normalize_node_type,
)

_LOGGER = logging.getLogger(__name__)


_BOOST_RUNTIME_KEY: Final = "boost_runtime"
BOOST_DURATION_OPTIONS: Final[tuple[int, ...]] = (30, 60, 120)
DEFAULT_BOOST_DURATION: Final = 60


def _coerce_boost_remaining_minutes(value: Any) -> int | None:
    """Return ``value`` as a positive integer minute count when possible."""

    if value is None or isinstance(value, bool):
        return None

    candidate: int | None
    try:
        if isinstance(value, (int, float)):
            candidate = int(value)
        else:
            text = str(value).strip()
            if not text:
                return None
            candidate = int(float(text))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None

    if candidate is None or candidate <= 0:
        return None

    return candidate


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
    minutes = _coerce_boost_minutes(stored)
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

    validated = _coerce_boost_minutes(minutes)
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


def _coerce_boost_bool(value: Any) -> bool | None:
    """Return ``value`` as a boolean when possible."""

    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
    try:
        text = str(value).strip().lower()
    except Exception:  # noqa: BLE001 - defensive
        return None
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off"}:
        return False
    return None


def _coerce_boost_minutes(value: Any) -> int | None:
    """Return ``value`` as positive minutes when possible."""

    if value is None or isinstance(value, bool):
        return None
    try:
        if isinstance(value, (int, float)):
            minutes = int(value)
        else:
            text = str(value).strip()
            if not text:
                return None
            minutes = int(float(text))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None
    return minutes if minutes > 0 else None


@dataclass(slots=True)
class BoostState:
    """Derived boost metadata for a heater node."""

    active: bool | None
    minutes_remaining: int | None
    end_datetime: datetime | None
    end_iso: str | None


def derive_boost_state(
    settings: Mapping[str, Any] | None, coordinator: Any
) -> BoostState:
    """Return derived boost metadata for ``settings`` using ``coordinator``."""

    source = settings if isinstance(settings, Mapping) else {}

    boost_active = _coerce_boost_bool(source.get("boost_active"))
    if boost_active is None:
        boost_active = _coerce_boost_bool(source.get("boost"))
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
    boost_minutes: int | None = None
    resolver = getattr(coordinator, "resolve_boost_end", None)
    if (
        callable(resolver)
        and boost_day is not None
        and boost_minute is not None
    ):
        try:
            boost_end_dt, boost_minutes = resolver(boost_day, boost_minute)
        except Exception:  # noqa: BLE001 - defensive
            boost_end_dt = None
            boost_minutes = None

    if boost_minutes is None:
        boost_minutes = _coerce_boost_remaining_minutes(source.get("boost_remaining"))

    if boost_minutes is None and boost_end_dt is not None:
        delta_seconds = (boost_end_dt - dt_util.now()).total_seconds()
        boost_minutes = int(max(0.0, delta_seconds) // 60)

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
        except Exception:  # pragma: no cover - defensive
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

    return BoostState(
        active=boost_active,
        minutes_remaining=boost_minutes,
        end_datetime=boost_end_dt,
        end_iso=boost_end_iso,
    )


class DispatcherSubscriptionHelper:
    """Manage dispatcher subscriptions tied to an entity lifecycle."""

    def __init__(self, owner: CoordinatorEntity) -> None:
        """Initialise the helper for the provided entity."""

        self._owner = owner
        self._unsub: dispatcher.DispatcherHandle | None = None

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
    nodes_by_type: Mapping[str, Iterable[Node] | None],
    resolve_name: Callable[[str, str], str],
    *,
    node_types: Iterable[str] | None = None,
) -> Iterator[tuple[str, Node, str, str]]:
    """Yield heater node metadata for supported node types."""

    types = list(node_types or HEATER_NODE_TYPES)
    for node_type in types:
        if not node_type:
            continue
        nodes = nodes_by_type.get(node_type)
        if not nodes:
            continue
        if isinstance(nodes, Mapping):
            iterable: Iterable[Any] = nodes.values()
        elif isinstance(nodes, (str, bytes)):
            iterable = ()
        elif isinstance(nodes, Iterable):
            iterable = nodes
        else:
            iterable = (nodes,)
        for node in iterable:
            raw_addr = getattr(node, "addr", "")
            if raw_addr is None:
                continue
            addr_str = normalize_node_addr(raw_addr)
            if not addr_str:
                continue
            if addr_str.lower() == "none":
                continue
            resolved_name = resolve_name(node_type, addr_str)
            yield node_type, node, addr_str, resolved_name


def iter_heater_maps(
    coordinator_cache: Mapping[str, Any] | None,
    *,
    map_key: str,
    node_types: Iterable[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield unique heater map dictionaries for ``map_key``."""

    if not isinstance(map_key, str) or not map_key:
        return

    seen: set[int] = set()
    cache = coordinator_cache if isinstance(coordinator_cache, Mapping) else {}

    if node_types is None:
        types = list(HEATER_NODE_TYPES)
    elif isinstance(node_types, str):
        types = [node_types]
    elif isinstance(node_types, bytes):  # pragma: no cover - defensive
        try:  # pragma: no cover - defensive
            decoded = node_types.decode()
        except Exception:  # pragma: no cover - defensive
            decoded = node_types.decode(errors="ignore")
        types = [decoded]  # pragma: no cover - defensive
    else:
        types = list(node_types)

    nodes_by_type = cache.get("nodes_by_type")
    if isinstance(nodes_by_type, Mapping):
        for node_type in types:
            if not node_type:
                continue
            section = nodes_by_type.get(node_type)
            if not isinstance(section, Mapping):
                continue
            candidate = section.get(map_key)
            if isinstance(candidate, dict):
                ident = id(candidate)
                if ident in seen:
                    continue
                seen.add(ident)
                yield candidate

    legacy = cache.get("htr")
    if isinstance(legacy, Mapping):
        candidate = legacy.get(map_key)
        if isinstance(candidate, dict):
            ident = id(candidate)
            if ident not in seen:
                seen.add(ident)
                yield candidate


def log_skipped_nodes(
    platform_name: str,
    nodes_by_type: Mapping[str, Iterable[Node] | None],
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

    for node_type in skipped_types:
        nodes = nodes_by_type.get(node_type)
        if not nodes:
            continue

        addrs = ", ".join(
            sorted(
                filter(
                    None,
                    (
                        normalize_node_addr(getattr(node, "addr", ""))
                        for node in _iter_nodes(nodes)
                    ),
                )
            )
        )
        log.debug(
            "Skipping TermoWeb %s nodes for %s: %s",
            node_type,
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


def prepare_heater_platform_data(
    entry_data: dict[str, Any],
    *,
    default_name_simple: Callable[[str], str],
) -> tuple[
    list[Node],
    dict[str, list[Node]],
    dict[str, list[str]],
    Callable[[str, str], str],
]:
    """Return node metadata and name resolution helpers for a config entry."""

    snapshot = ensure_snapshot(entry_data)
    if isinstance(snapshot, InstallationSnapshot):
        inventory = snapshot.inventory
        nodes_by_type_raw = snapshot.nodes_by_type
        nodes_by_type = {
            node_type: list(nodes) for node_type, nodes in nodes_by_type_raw.items()
        }
        explicit_names = snapshot.explicit_heater_names
        type_to_addresses, _reverse_lookup = snapshot.heater_address_map
        addrs_by_type = {
            node_type: list(type_to_addresses.get(node_type, []))
            for node_type in HEATER_NODE_TYPES
        }
        name_map = snapshot.heater_name_map(default_name_simple)
    else:
        nodes = entry_data.get("nodes")
        inventory = ensure_node_inventory(entry_data, nodes=nodes)

        nodes_by_type = defaultdict(list)
        explicit_names = set()
        for node in inventory:
            node_type = normalize_node_type(getattr(node, "type", ""))
            if not node_type:
                continue
            nodes_by_type[node_type].append(node)
            addr = normalize_node_addr(getattr(node, "addr", ""))
            if addr and getattr(node, "name", "").strip():
                explicit_names.add((node_type, addr))

        type_to_addresses, _reverse_lookup = build_heater_address_map(inventory)

        addrs_by_type = {
            node_type: list(type_to_addresses.get(node_type, []))
            for node_type in HEATER_NODE_TYPES
        }

        name_map = build_heater_name_map(inventory, default_name_simple)
    names_by_type: dict[str, dict[str, str]] = name_map.get("by_type", {})
    legacy_names: dict[str, str] = name_map.get("htr", {})

    def _default_name(addr: str, node_type: str | None = None) -> str:
        if (node_type or "").lower() == "acm":
            return f"Accumulator {addr}"
        return default_name_simple(addr)

    def resolve_name(node_type: str, addr: str) -> str:
        """Resolve the friendly name for ``addr`` of the given node type."""

        node_type_norm = normalize_node_type(
            node_type,
            use_default_when_falsey=True,
        )
        addr_str = normalize_node_addr(
            addr,
            use_default_when_falsey=True,
        )
        default_simple = default_name_simple(addr_str)

        def _candidate(value: Any) -> str | None:
            if not isinstance(value, str) or not value:
                return None
            if (
                node_type_norm == "acm"
                and value == default_simple
                and (node_type_norm, addr_str) not in explicit_names
            ):
                return None
            return value

        per_type = names_by_type.get(node_type_norm, {})
        for candidate_value in (
            per_type.get(addr_str),
            name_map.get((node_type_norm, addr_str)),
            legacy_names.get(addr_str),
        ):
            candidate = _candidate(candidate_value)
            if candidate:
                return candidate

        return _default_name(addr_str, node_type_norm)

    return inventory, dict(nodes_by_type), addrs_by_type, resolve_name


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
        await super().async_added_to_hass()
        if self.hass is None:
            return

        signal = signal_ws_data(self._entry_id)
        self._ws_subscription.subscribe(self.hass, signal, self._handle_ws_message)

    async def async_will_remove_from_hass(self) -> None:
        """Tidy up websocket listeners before the entity is removed."""
        self._ws_subscription.unsubscribe()
        await super().async_will_remove_from_hass()

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

        self.schedule_update_ha_state()

    @property
    def should_poll(self) -> bool:
        """Home Assistant should not poll heater entities."""
        return False

    @property
    def available(self) -> bool:
        """Return whether the backing device exposes heater data."""
        return self._device_available(self._device_record())

    def _device_available(self, device_entry: dict[str, Any] | None) -> bool:
        """Return True when the device entry contains node data."""
        if not isinstance(device_entry, dict):
            return False
        return device_entry.get("nodes") is not None

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
        """Return the heater-specific portion of the coordinator data."""
        record = self._device_record()
        if record is None:
            return {}
        node_type = getattr(self, "_node_type", "htr")
        if node_type == "htr":
            legacy = record.get("htr")
            if isinstance(legacy, dict):
                return legacy

        by_type = record.get("nodes_by_type")
        if isinstance(by_type, dict):
            section = by_type.get(node_type)
            if isinstance(section, dict):
                return section

        heaters = record.get("htr")
        return heaters if isinstance(heaters, dict) else {}

    def heater_settings(self) -> dict[str, Any] | None:
        """Return the cached settings for this heater, if available."""
        settings_map = self._heater_section().get("settings")
        if not isinstance(settings_map, dict):
            return None
        settings = settings_map.get(self._addr)
        return settings if isinstance(settings, dict) else None

    def boost_state(self) -> BoostState:
        """Return derived boost metadata for this heater."""

        settings = self.heater_settings() or {}
        return derive_boost_state(settings, self.coordinator)

    def _client(self) -> Any:
        """Return the REST client used for write operations."""
        hass_data = self.hass.data.get(DOMAIN, {}).get(self._entry_id, {})
        return hass_data.get("client")

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
            identifiers={(DOMAIN, self._dev_id, self._addr)},
            name=self._device_name,
            manufacturer="TermoWeb",
            model=model,
            via_device=(DOMAIN, self._dev_id),
        )
