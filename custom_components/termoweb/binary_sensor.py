"""Binary sensor entities for TermoWeb gateway connectivity."""

from __future__ import annotations

from typing import Any

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.core import callback
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN, signal_ws_status
from .coordinator import StateCoordinator
from .heater import DispatcherSubscriptionHelper
from .utils import build_gateway_device_info


async def async_setup_entry(hass, entry, async_add_entities):
    """Set up one connectivity binary sensor per TermoWeb hub (dev_id)."""
    data = hass.data[DOMAIN][entry.entry_id]
    coord: StateCoordinator = data["coordinator"]
    dev_id = data["dev_id"]
    ent = GatewayOnlineBinarySensor(coord, entry.entry_id, dev_id)
    async_add_entities([ent])


class GatewayOnlineBinarySensor(
    CoordinatorEntity[StateCoordinator], BinarySensorEntity
):
    """Connectivity sensor for the TermoWeb hub (gateway)."""

    _attr_device_class = BinarySensorDeviceClass.CONNECTIVITY
    _attr_should_poll = False

    def __init__(
        self, coordinator: StateCoordinator, entry_id: str, dev_id: str
    ) -> None:
        """Initialise the connectivity binary sensor."""
        super().__init__(coordinator)
        self._entry_id = entry_id
        self._dev_id = str(dev_id)
        self._attr_name = "TermoWeb Gateway Online"
        self._attr_unique_id = f"{self._dev_id}_online"
        self._ws_subscription = DispatcherSubscriptionHelper(self)

    async def async_added_to_hass(self) -> None:
        """Subscribe to websocket status updates when added to hass."""
        await super().async_added_to_hass()
        if self.hass is None:
            return

        signal = signal_ws_status(self._entry_id)
        self._ws_subscription.subscribe(self.hass, signal, self._on_ws_status)

    async def async_will_remove_from_hass(self) -> None:
        """Unsubscribe from websocket updates before removal."""
        self._ws_subscription.unsubscribe()
        await super().async_will_remove_from_hass()

    def _ws_state(self) -> dict[str, Any]:
        """Return the latest websocket status payload for this device."""
        rec = self.hass.data.get(DOMAIN, {}).get(self._entry_id, {}) or {}
        return (rec.get("ws_state") or {}).get(self._dev_id, {})

    @property
    def is_on(self) -> bool:
        """Return True when the integration reports the gateway is online."""
        data = (self.coordinator.data or {}).get(self._dev_id, {}) or {}
        return bool(data.get("connected"))

    @property
    def device_info(self) -> DeviceInfo:
        """Return Home Assistant device metadata for the gateway."""
        return build_gateway_device_info(self.hass, self._entry_id, self._dev_id)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return additional gateway diagnostics and websocket state."""
        data = (self.coordinator.data or {}).get(self._dev_id, {}) or {}
        ws = self._ws_state()
        return {
            "dev_id": self._dev_id,
            "name": data.get("name"),
            "connected": data.get("connected"),
            "ws_status": ws.get("status"),
            "ws_last_event_at": ws.get("last_event_at"),
            "ws_healthy_minutes": ws.get("healthy_minutes"),
            "raw": data.get("raw") or {},
        }

    @callback
    def _on_ws_status(self, payload: dict) -> None:
        """Handle websocket status broadcasts from the integration."""
        if payload.get("dev_id") != self._dev_id:
            return
        self.schedule_update_ha_state()
