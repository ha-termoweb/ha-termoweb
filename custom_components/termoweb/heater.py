"""Shared helpers and base entities for TermoWeb heaters."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
import logging
from typing import Any

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import dispatcher
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN, signal_ws_data
from .nodes import Node, build_node_inventory
from .utils import HEATER_NODE_TYPES, build_heater_address_map, ensure_node_inventory

_LOGGER = logging.getLogger(__name__)


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
                str(getattr(node, "addr", "")).strip() for node in _iter_nodes(nodes)
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
        node_type = str(getattr(node, "type", "")).strip().lower()
        if node_type not in HEATER_NODE_TYPES:
            continue

        addr = str(getattr(node, "addr", "")).strip()
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

    nodes = entry_data.get("nodes")
    inventory = ensure_node_inventory(entry_data, nodes=nodes)

    nodes_by_type: dict[str, list[Node]] = defaultdict(list)
    explicit_names: set[tuple[str, str]] = set()
    for node in inventory:
        node_type = str(getattr(node, "type", "")).strip().lower()
        if not node_type:
            continue
        nodes_by_type[node_type].append(node)
        addr = str(getattr(node, "addr", "")).strip()
        if addr and getattr(node, "name", "").strip():
            explicit_names.add((node_type, addr))

    type_to_addresses, _reverse_lookup = build_heater_address_map(inventory)

    addrs_by_type: dict[str, list[str]] = {
        node_type: list(type_to_addresses.get(node_type, []))
        for node_type in HEATER_NODE_TYPES
    }

    name_map = build_heater_name_map(nodes, default_name_simple)
    names_by_type: dict[str, dict[str, str]] = name_map.get("by_type", {})
    legacy_names: dict[str, str] = name_map.get("htr", {})

    def _default_name(addr: str, node_type: str | None = None) -> str:
        if (node_type or "").lower() == "acm":
            return f"Accumulator {addr}"
        return default_name_simple(addr)

    def resolve_name(node_type: str, addr: str) -> str:
        """Resolve the friendly name for ``addr`` of the given node type."""

        node_type_norm = str(node_type or "").strip().lower()
        addr_str = str(addr or "").strip()
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
        self._addr = str(addr)
        self._attr_name = name
        resolved_type = str(node_type or "htr").strip().lower() or "htr"
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
            try:
                payload_type_str = str(payload_type).strip().lower()
            except Exception:  # pragma: no cover - defensive  # noqa: BLE001
                payload_type_str = ""
            if payload_type_str and payload_type_str != self._node_type:
                return False
        addr = payload.get("addr")
        if addr is None:
            return True
        return str(addr) == self._addr

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
