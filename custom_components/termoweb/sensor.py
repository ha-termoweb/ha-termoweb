from __future__ import annotations

import logging
from typing import Any, Optional

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.const import UnitOfTemperature
from homeassistant.core import callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN, signal_ws_data

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass, entry, async_add_entities):
    """Set up temperature sensors for each heater node."""
    data = hass.data[DOMAIN][entry.entry_id]
    coordinator = data["coordinator"]

    added: set[str] = set()

    async def build_and_add() -> None:
        new_entities: list[SensorEntity] = []
        data_now = coordinator.data or {}
        for dev_id, dev in data_now.items():
            nodes = dev.get("nodes") or {}
            node_list = nodes.get("nodes") if isinstance(nodes, dict) else None
            if isinstance(node_list, list):
                for node in node_list:
                    if not isinstance(node, dict):
                        continue
                    ntype = (node.get("type") or "").lower()
                    if ntype != "htr":
                        continue
                    addr = str(node.get("addr"))
                    base_name = (node.get("name") or f"Node {addr}").strip() or f"Node {addr}"
                    unique_id = f"{DOMAIN}:{dev_id}:htr:{addr}:temp"
                    if unique_id in added:
                        continue
                    ent_name = f"{base_name} Temperature"
                    new_entities.append(
                        TermoWebHeaterTemp(
                            coordinator, entry.entry_id, dev_id, addr, ent_name, unique_id
                        )
                    )
                    added.add(unique_id)

        if new_entities:
            _LOGGER.debug("Adding %d TermoWeb sensors", len(new_entities))
            async_add_entities(new_entities)

    await build_and_add()

    def _on_coordinator_update() -> None:
        hass.async_create_task(build_and_add())

    coordinator.async_add_listener(_on_coordinator_update)


class TermoWebHeaterTemp(CoordinatorEntity, SensorEntity):
    """Temperature sensor for a single heater node (read-only mtemp)."""

    _attr_device_class = SensorDeviceClass.TEMPERATURE
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS

    def __init__(self, coordinator, entry_id: str, dev_id: str, addr: str, name: str, unique_id: str) -> None:
        super().__init__(coordinator)
        self._entry_id = entry_id
        self._dev_id = dev_id
        self._addr = addr
        self._attr_name = name
        self._attr_unique_id = unique_id
        self._unsub_ws = None

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        self._unsub_ws = async_dispatcher_connect(
            self.hass, signal_ws_data(self._entry_id), self._on_ws_data
        )
        self.async_on_remove(lambda: self._unsub_ws() if self._unsub_ws else None)

    @property
    def device_info(self) -> DeviceInfo:
        # Attach to the existing hub device (like climate + button)
        return DeviceInfo(identifiers={(DOMAIN, self._dev_id)})

    def _settings(self) -> dict[str, Any] | None:
        d = (self.coordinator.data or {}).get(self._dev_id, {})
        htr = d.get("htr") or {}
        settings = (htr.get("settings") or {}).get(self._addr)
        return settings if isinstance(settings, dict) else None

    @staticmethod
    def _f(val: Any) -> Optional[float]:
        try:
            if val is None:
                return None
            if isinstance(val, (int, float)):
                return float(val)
            s = str(val).strip()
            return float(s) if s else None
        except Exception:
            return None

    @callback
    def _on_ws_data(self, payload: dict) -> None:
        if payload.get("dev_id") != self._dev_id:
            return
        addr = payload.get("addr")
        if addr is not None and str(addr) != self._addr:
            return
        # Thread-safe state update
        self.schedule_update_ha_state()

    @property
    def available(self) -> bool:
        d = (self.coordinator.data or {}).get(self._dev_id, {})
        return d.get("nodes") is not None

    @property
    def native_value(self) -> Optional[float]:
        s = self._settings() or {}
        return self._f(s.get("mtemp"))

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        s = self._settings() or {}
        return {
            "dev_id": self._dev_id,
            "addr": self._addr,
            "units": s.get("units"),
        }


