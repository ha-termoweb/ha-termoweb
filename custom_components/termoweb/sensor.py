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
from .coordinator import TermoWebHeaterEnergyCoordinator

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass, entry, async_add_entities):
    """Set up sensors for each heater node."""
    data = hass.data[DOMAIN][entry.entry_id]
    coordinator = data["coordinator"]

    energy_coordinator: TermoWebHeaterEnergyCoordinator | None = data.get(
        "energy_coordinator"
    )
    if energy_coordinator is None:
        energy_coordinator = TermoWebHeaterEnergyCoordinator(hass, data["client"])
        data["energy_coordinator"] = energy_coordinator
        await energy_coordinator.async_config_entry_first_refresh()

    added: set[str] = set()

    async def build_and_add() -> None:
        new_entities: list[SensorEntity] = []
        data_now = coordinator.data or {}
        energy_now = energy_coordinator.data or {}

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

        dev_ids = set(data_now.keys()) | set(energy_now.keys())
        for dev_id in dev_ids:
            nodes = (data_now.get(dev_id, {}).get("nodes") or {})
            node_list = nodes.get("nodes") if isinstance(nodes, dict) else None
            name_map: dict[str, str] = {}
            if isinstance(node_list, list):
                for node in node_list:
                    if not isinstance(node, dict):
                        continue
                    if (node.get("type") or "").lower() != "htr":
                        continue
                    addr = str(node.get("addr"))
                    base_name = (node.get("name") or f"Node {addr}").strip() or f"Node {addr}"
                    name_map[addr] = base_name

            addrs: set[str] = set(name_map.keys())
            htr_main = (data_now.get(dev_id, {}).get("htr") or {}).get("addrs") or []
            addrs.update(str(a) for a in htr_main)
            htr_energy = (energy_now.get(dev_id, {}).get("htr") or {})
            energy_map = htr_energy.get("energy") or {}
            power_map = htr_energy.get("power") or {}
            addrs.update(str(a) for a in energy_map.keys())
            addrs.update(str(a) for a in power_map.keys())

            for addr in addrs:
                base_name = name_map.get(addr, f"Node {addr}")
                unique_id = f"{DOMAIN}:{dev_id}:htr:{addr}:energy"
                if unique_id not in added:
                    ent_name = f"{base_name} Energy"
                    new_entities.append(
                        TermoWebHeaterEnergyTotal(
                            energy_coordinator,
                            entry.entry_id,
                            dev_id,
                            addr,
                            ent_name,
                            unique_id,
                        )
                    )
                    added.add(unique_id)

                unique_id = f"{DOMAIN}:{dev_id}:htr:{addr}:power"
                if unique_id not in added:
                    ent_name = f"{base_name} Power"
                    new_entities.append(
                        TermoWebHeaterPower(
                            energy_coordinator,
                            entry.entry_id,
                            dev_id,
                            addr,
                            ent_name,
                            unique_id,
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
    energy_coordinator.async_add_listener(_on_coordinator_update)


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


class TermoWebHeaterEnergyTotal(CoordinatorEntity, SensorEntity):
    """Total energy consumption sensor for a heater."""

    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_state_class = SensorStateClass.TOTAL_INCREASING
    _attr_native_unit_of_measurement = "kWh"

    def __init__(
        self,
        coordinator: TermoWebHeaterEnergyCoordinator,
        entry_id: str,
        dev_id: str,
        addr: str,
        name: str,
        unique_id: str,
    ) -> None:
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
        return DeviceInfo(identifiers={(DOMAIN, self._dev_id)})

    @callback
    def _on_ws_data(self, payload: dict) -> None:
        if payload.get("dev_id") != self._dev_id:
            return
        addr = payload.get("addr")
        if addr is not None and str(addr) != self._addr:
            return
        self.schedule_update_ha_state()

    @property
    def available(self) -> bool:
        d = (self.coordinator.data or {}).get(self._dev_id)
        return d is not None

    @property
    def native_value(self) -> Optional[float]:
        d = (self.coordinator.data or {}).get(self._dev_id, {})
        energy = (d.get("htr") or {}).get("energy") or {}
        val = energy.get(self._addr)
        if val is None:
            return None
        try:
            return float(val) / 1000
        except (TypeError, ValueError):
            return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        return {"dev_id": self._dev_id, "addr": self._addr}


class TermoWebHeaterPower(CoordinatorEntity, SensorEntity):
    """Power sensor for a heater."""

    _attr_device_class = SensorDeviceClass.POWER
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = "W"

    def __init__(
        self,
        coordinator: TermoWebHeaterEnergyCoordinator,
        entry_id: str,
        dev_id: str,
        addr: str,
        name: str,
        unique_id: str,
    ) -> None:
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
        return DeviceInfo(identifiers={(DOMAIN, self._dev_id)})

    @callback
    def _on_ws_data(self, payload: dict) -> None:
        if payload.get("dev_id") != self._dev_id:
            return
        addr = payload.get("addr")
        if addr is not None and str(addr) != self._addr:
            return
        self.schedule_update_ha_state()

    @property
    def available(self) -> bool:
        d = (self.coordinator.data or {}).get(self._dev_id)
        return d is not None

    @property
    def native_value(self) -> Optional[float]:
        d = (self.coordinator.data or {}).get(self._dev_id, {})
        power = (d.get("htr") or {}).get("power") or {}
        val = power.get(self._addr)
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        return {"dev_id": self._dev_id, "addr": self._addr}


