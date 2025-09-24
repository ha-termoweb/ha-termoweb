from __future__ import annotations

import logging
import math
from typing import Any

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
from .utils import float_or_none

_WH_TO_KWH = 1 / 1000.0

_LOGGER = logging.getLogger(__name__)


def _looks_like_integer_string(value: str) -> bool:
    """Return True if the string looks like an integer number."""

    stripped = value.strip()
    if not stripped:
        return False
    if stripped[0] in "+-":
        stripped = stripped[1:]
    return stripped.isdigit()


def _normalise_energy_value(coordinator: Any, raw: Any) -> float | None:
    """Try to coerce a raw energy reading into kWh."""

    if isinstance(raw, bool):
        return None

    try:
        numeric = float(raw)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(numeric):
        return None

    scale_attr = getattr(coordinator, "_termoweb_energy_scale", None)
    scale: float | None = None
    if isinstance(scale_attr, (int, float)):
        if math.isfinite(scale_attr) and scale_attr > 0:
            scale = float(scale_attr)
    elif isinstance(scale_attr, str):
        lowered = scale_attr.strip().lower()
        if lowered in {"kwh", "kilowatthour", "kilowatt-hour"}:
            scale = 1.0
        elif lowered in {"wh", "watt-hour", "watthour"}:
            scale = _WH_TO_KWH
        else:
            try:
                parsed = float(lowered)
            except ValueError:
                parsed = None
            if parsed and math.isfinite(parsed) and parsed > 0:
                scale = parsed

    if scale is None:
        if isinstance(coordinator, TermoWebHeaterEnergyCoordinator):
            scale = 1.0
        elif isinstance(raw, int):
            scale = _WH_TO_KWH
        elif isinstance(raw, str) and _looks_like_integer_string(raw):
            scale = _WH_TO_KWH
        else:
            scale = 1.0

    return numeric * scale


async def async_setup_entry(hass, entry, async_add_entities):
    """Set up sensors for each heater node."""
    data = hass.data[DOMAIN][entry.entry_id]
    coordinator = data["coordinator"]
    dev_id = data["dev_id"]
    nodes = data["nodes"]
    addrs: list[str] = data["htr_addrs"]

    energy_coordinator: TermoWebHeaterEnergyCoordinator | None = data.get(
        "energy_coordinator",
    )
    if energy_coordinator is None:
        energy_coordinator = TermoWebHeaterEnergyCoordinator(
            hass, data["client"], dev_id, addrs
        )
        data["energy_coordinator"] = energy_coordinator
        await energy_coordinator.async_config_entry_first_refresh()

    name_map: dict[str, str] = {}
    node_list = nodes.get("nodes") if isinstance(nodes, dict) else None
    if isinstance(node_list, list):
        for node in node_list:
            if isinstance(node, dict) and (node.get("type") or "").lower() == "htr":
                addr = str(node.get("addr"))
                base = (node.get("name") or f"Node {addr}").strip() or f"Node {addr}"
                name_map[addr] = base

    new_entities: list[SensorEntity] = []
    for addr in addrs:
        base_name = name_map.get(addr, f"Node {addr}")
        uid_temp = f"{DOMAIN}:{dev_id}:htr:{addr}:temp"
        new_entities.append(
            TermoWebHeaterTemp(
                coordinator,
                entry.entry_id,
                dev_id,
                addr,
                f"{base_name} Temperature",
                uid_temp,
                base_name,
            )
        )
        uid_energy = f"{DOMAIN}:{dev_id}:htr:{addr}:energy"
        new_entities.append(
            TermoWebHeaterEnergyTotal(
                energy_coordinator,
                entry.entry_id,
                dev_id,
                addr,
                f"{base_name} Energy",
                uid_energy,
                base_name,
            )
        )
        uid_power = f"{DOMAIN}:{dev_id}:htr:{addr}:power"
        new_entities.append(
            TermoWebHeaterPower(
                energy_coordinator,
                entry.entry_id,
                dev_id,
                addr,
                f"{base_name} Power",
                uid_power,
                base_name,
            )
        )

    uid_total = f"{DOMAIN}:{dev_id}:energy_total"
    new_entities.append(
        TermoWebTotalEnergy(

            energy_coordinator,
            entry.entry_id,
            dev_id,
            "Total Energy",
            uid_total,
        )
    )

    if new_entities:
        _LOGGER.debug("Adding %d TermoWeb sensors", len(new_entities))
        async_add_entities(new_entities)


class TermoWebHeaterTemp(CoordinatorEntity, SensorEntity):
    """Temperature sensor for a single heater node (read-only mtemp)."""

    _attr_device_class = SensorDeviceClass.TEMPERATURE
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS

    def __init__(
        self,
        coordinator,
        entry_id: str,
        dev_id: str,
        addr: str,
        name: str,
        unique_id: str,
        device_name: str,
    ) -> None:
        super().__init__(coordinator)
        self._entry_id = entry_id
        self._dev_id = dev_id
        self._addr = addr
        self._attr_name = name
        self._attr_unique_id = unique_id
        self._device_name = device_name
        self._unsub_ws = None

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        self._unsub_ws = async_dispatcher_connect(
            self.hass, signal_ws_data(self._entry_id), self._on_ws_data
        )
        self.async_on_remove(lambda: self._unsub_ws() if self._unsub_ws else None)

    @property
    def device_info(self) -> DeviceInfo:
        return DeviceInfo(
            identifiers={(DOMAIN, self._dev_id, self._addr)},
            name=self._device_name,
            manufacturer="TermoWeb",
            model="Heater",
            via_device=(DOMAIN, self._dev_id),
        )

    def _settings(self) -> dict[str, Any] | None:
        d = (self.coordinator.data or {}).get(self._dev_id, {})
        htr = d.get("htr") or {}
        settings = (htr.get("settings") or {}).get(self._addr)
        return settings if isinstance(settings, dict) else None


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
    def native_value(self) -> float | None:
        s = self._settings() or {}
        return float_or_none(s.get("mtemp"))

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
        device_name: str,
    ) -> None:
        super().__init__(coordinator)
        self._entry_id = entry_id
        self._dev_id = dev_id
        self._addr = addr
        self._attr_name = name
        self._attr_unique_id = unique_id
        self._device_name = device_name
        self._unsub_ws = None

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        self._unsub_ws = async_dispatcher_connect(
            self.hass, signal_ws_data(self._entry_id), self._on_ws_data
        )
        self.async_on_remove(lambda: self._unsub_ws() if self._unsub_ws else None)

    @property
    def device_info(self) -> DeviceInfo:
        return DeviceInfo(
            identifiers={(DOMAIN, self._dev_id, self._addr)},
            name=self._device_name,
            manufacturer="TermoWeb",
            model="Heater",
            via_device=(DOMAIN, self._dev_id),
        )

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
    def native_value(self) -> float | None:
        d = (self.coordinator.data or {}).get(self._dev_id, {})
        energy = (d.get("htr") or {}).get("energy") or {}
        val = energy.get(self._addr)
        if val is None:
            return None
        return _normalise_energy_value(self.coordinator, val)

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
        device_name: str,
    ) -> None:
        super().__init__(coordinator)
        self._entry_id = entry_id
        self._dev_id = dev_id
        self._addr = addr
        self._attr_name = name
        self._attr_unique_id = unique_id
        self._device_name = device_name
        self._unsub_ws = None

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        self._unsub_ws = async_dispatcher_connect(
            self.hass, signal_ws_data(self._entry_id), self._on_ws_data
        )
        self.async_on_remove(lambda: self._unsub_ws() if self._unsub_ws else None)

    @property
    def device_info(self) -> DeviceInfo:
        return DeviceInfo(
            identifiers={(DOMAIN, self._dev_id, self._addr)},
            name=self._device_name,
            manufacturer="TermoWeb",
            model="Heater",
            via_device=(DOMAIN, self._dev_id),
        )

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
    def native_value(self) -> float | None:
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


class TermoWebTotalEnergy(CoordinatorEntity, SensorEntity):
    """Total energy consumption across all heaters."""

    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_state_class = SensorStateClass.TOTAL_INCREASING
    _attr_native_unit_of_measurement = "kWh"

    def __init__(
        self,
        coordinator: TermoWebHeaterEnergyCoordinator,
        entry_id: str,
        dev_id: str,
        name: str,
        unique_id: str,
    ) -> None:
        super().__init__(coordinator)
        self._entry_id = entry_id
        self._dev_id = dev_id
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
        self.schedule_update_ha_state()

    @property
    def available(self) -> bool:
        d = (self.coordinator.data or {}).get(self._dev_id)
        return d is not None

    @property
    def native_value(self) -> float | None:
        d = (self.coordinator.data or {}).get(self._dev_id, {})
        energy = (d.get("htr") or {}).get("energy") or {}
        total = 0.0
        found = False
        for val in energy.values():
            normalised = _normalise_energy_value(self.coordinator, val)
            if normalised is None:
                continue
            total += normalised
            found = True
        if not found:
            return None
        return total

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        return {"dev_id": self._dev_id}


