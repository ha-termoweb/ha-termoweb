"""Shared helpers and base entities for TermoWeb heaters."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import logging
from typing import Any

from homeassistant.core import callback
from homeassistant.helpers import dispatcher
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN, signal_ws_data
from .nodes import Node, build_node_inventory
from .utils import HEATER_NODE_TYPES

_LOGGER = logging.getLogger(__name__)


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

    for key, value in by_node.items():
        result[key] = value

    return result


class HeaterNodeBase(CoordinatorEntity):
    """Base entity implementing common TermoWeb heater behaviour."""

    _unsub_ws: dispatcher.DispatcherHandle | None

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
        self._unsub_ws = None

    async def async_added_to_hass(self) -> None:
        """Subscribe to websocket updates once the entity is added to hass."""
        await super().async_added_to_hass()
        if self.hass is None:
            return

        signal = signal_ws_data(self._entry_id)
        self._unsub_ws = async_dispatcher_connect(
            self.hass, signal, self._handle_ws_message
        )
        self.async_on_remove(self._remove_ws_listener)

    async def async_will_remove_from_hass(self) -> None:
        """Tidy up websocket listeners before the entity is removed."""
        self._remove_ws_listener()
        await super().async_will_remove_from_hass()

    def _remove_ws_listener(self) -> None:
        """Disconnect the websocket listener if it is registered."""
        if self._unsub_ws is None:
            return
        try:
            self._unsub_ws()
        except Exception:  # pragma: no cover - defensive
            _LOGGER.exception(
                "Failed to remove WS listener for dev=%s addr=%s",
                self._dev_id,
                self._addr,
            )
        finally:
            self._unsub_ws = None

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
            except Exception:  # pragma: no cover - defensive
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

