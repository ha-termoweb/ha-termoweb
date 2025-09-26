from __future__ import annotations

from typing import Any

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.core import callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN, signal_ws_status
from .coordinator import StateCoordinator


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
        super().__init__(coordinator)
        self._entry_id = entry_id
        self._dev_id = str(dev_id)
        self._attr_name = "TermoWeb Gateway Online"
        self._attr_unique_id = f"{self._dev_id}_online"
        self._unsub_ws = None

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        self._unsub_ws = async_dispatcher_connect(
            self.hass, signal_ws_status(self._entry_id), self._on_ws_status
        )
        self.async_on_remove(lambda: self._unsub_ws() if self._unsub_ws else None)

    def _ws_state(self) -> dict[str, Any]:
        rec = self.hass.data.get(DOMAIN, {}).get(self._entry_id, {}) or {}
        return (rec.get("ws_state") or {}).get(self._dev_id, {})

    @property
    def is_on(self) -> bool:
        data = (self.coordinator.data or {}).get(self._dev_id, {}) or {}
        return bool(data.get("connected"))

    @property
    def device_info(self) -> DeviceInfo:
        data = (self.coordinator.data or {}).get(self._dev_id, {}) or {}
        version = (self.hass.data.get(DOMAIN, {}).get(self._entry_id, {}) or {}).get(
            "version"
        )
        model = (data.get("raw") or {}).get("model") or "Gateway/Controller"
        return DeviceInfo(
            identifiers={(DOMAIN, self._dev_id)},
            name="TermoWeb Gateway",
            manufacturer="TermoWeb",
            model=str(model),
            sw_version=str(version) if version is not None else None,
            configuration_url="https://control.termoweb.net",
        )

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
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
        if payload.get("dev_id") != self._dev_id:
            return
        self.schedule_update_ha_state()
