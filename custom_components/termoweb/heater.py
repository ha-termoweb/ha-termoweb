"""Shared helpers and base entities for TermoWeb heaters."""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any

from homeassistant.core import callback
from homeassistant.helpers import dispatcher
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN, signal_ws_data

_LOGGER = logging.getLogger(__name__)


def build_heater_name_map(
    nodes: Any, default_factory: Callable[[str], str]
) -> dict[str, str]:
    """Return a mapping of heater address -> friendly name.

    The TermoWeb API returns node metadata via ``/mgr/nodes``.  Both the
    climate and sensor platforms previously duplicated the logic that parsed
    that structure.  The helper tolerates malformed payloads and ensures we
    keep backwards compatible naming semantics by falling back to the
    ``default_factory`` when necessary.
    """

    name_map: dict[str, str] = {}

    node_list = None
    if isinstance(nodes, dict):
        node_list = nodes.get("nodes")

    if isinstance(node_list, list):
        for node in node_list:
            if not isinstance(node, dict):
                continue
            if (node.get("type") or "").lower() != "htr":
                continue

            addr = node.get("addr")
            if addr is None:
                continue

            addr_str = str(addr)
            raw_name = node.get("name")
            default_name = default_factory(addr_str)
            if isinstance(raw_name, str):
                candidate = raw_name.strip()
                name_map[addr_str] = candidate or default_name
            else:
                name_map[addr_str] = default_name

    return name_map


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
    ) -> None:
        """Initialise a heater entity tied to a TermoWeb device."""
        super().__init__(coordinator)
        self._entry_id = entry_id
        self._dev_id = dev_id
        self._addr = str(addr)
        self._attr_name = name
        self._attr_unique_id = unique_id or f"{DOMAIN}:{dev_id}:htr:{self._addr}"
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
        return DeviceInfo(
            identifiers={(DOMAIN, self._dev_id, self._addr)},
            name=self._device_name,
            manufacturer="TermoWeb",
            model="Heater",
            via_device=(DOMAIN, self._dev_id),
        )

