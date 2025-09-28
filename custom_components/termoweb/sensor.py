"""Sensor platform entities for TermoWeb heaters and gateways."""

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
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN, signal_ws_data
from .coordinator import EnergyStateCoordinator
from .heater import (
    DispatcherSubscriptionHelper,
    HeaterNodeBase,
    log_skipped_nodes,
    prepare_heater_platform_data,
)
from .utils import HEATER_NODE_TYPES, build_gateway_device_info, float_or_none

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
        if isinstance(coordinator, EnergyStateCoordinator):
            scale = 1.0
        elif isinstance(raw, int) or (
            isinstance(raw, str) and _looks_like_integer_string(raw)
        ):
            scale = _WH_TO_KWH
        else:
            scale = 1.0

    return numeric * scale


async def async_setup_entry(hass, entry, async_add_entities):
    """Set up sensors for each heater node."""
    data = hass.data[DOMAIN][entry.entry_id]
    coordinator = data["coordinator"]
    dev_id = data["dev_id"]
    inventory, nodes_by_type, addrs_by_type, resolve_name = (
        prepare_heater_platform_data(
            data,
            default_name_simple=lambda addr: f"Node {addr}",
        )
    )

    energy_coordinator: EnergyStateCoordinator | None = data.get(
        "energy_coordinator",
    )
    if energy_coordinator is None:
        energy_coordinator = EnergyStateCoordinator(
            hass, data["client"], dev_id, addrs_by_type
        )
        data["energy_coordinator"] = energy_coordinator
        await energy_coordinator.async_config_entry_first_refresh()
    else:
        energy_coordinator.update_addresses(addrs_by_type)

    new_entities: list[SensorEntity] = []
    for node_type in HEATER_NODE_TYPES:
        nodes_for_type = nodes_by_type.get(node_type, [])
        if not nodes_for_type:
            continue
        for node in nodes_for_type:
            addr_str = str(getattr(node, "addr", "")).strip()
            if not addr_str:
                continue
            base_name = resolve_name(node_type, addr_str)
            uid_prefix = f"{DOMAIN}:{dev_id}:{node_type}:{addr_str}"
            new_entities.extend(
                _create_heater_sensors(
                    coordinator,
                    energy_coordinator,
                    entry.entry_id,
                    dev_id,
                    addr_str,
                    base_name,
                    uid_prefix,
                    node_type=node_type,
                )
            )

    log_skipped_nodes("sensor", nodes_by_type, logger=_LOGGER)

    uid_total = f"{DOMAIN}:{dev_id}:energy_total"
    new_entities.append(
        InstallationTotalEnergySensor(
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


class HeaterTemperatureSensor(HeaterNodeBase, SensorEntity):
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
        *,
        node_type: str | None = None,
    ) -> None:
        """Initialise the heater temperature sensor entity."""
        super().__init__(
            coordinator,
            entry_id,
            dev_id,
            addr,
            name,
            unique_id,
            device_name=device_name,
            node_type=node_type,
        )

    @property
    def native_value(self) -> float | None:
        """Return the latest temperature reported by the heater."""
        s = self.heater_settings() or {}
        return float_or_none(s.get("mtemp"))

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return metadata describing the heater temperature source."""
        s = self.heater_settings() or {}
        return {
            "dev_id": self._dev_id,
            "addr": self._addr,
            "units": s.get("units"),
        }


class HeaterEnergyBase(HeaterNodeBase, SensorEntity):
    """Base helper for heater measurement sensors such as power and energy."""

    _metric_key: str

    def __init__(
        self,
        coordinator: EnergyStateCoordinator,
        entry_id: str,
        dev_id: str,
        addr: str,
        name: str,
        unique_id: str,
        device_name: str,
        *,
        node_type: str | None = None,
    ) -> None:
        """Initialise a heater energy-derived sensor entity."""
        super().__init__(
            coordinator,
            entry_id,
            dev_id,
            addr,
            name,
            unique_id,
            device_name=device_name,
            node_type=node_type,
        )

    def _device_available(self, device_entry: dict[str, Any] | None) -> bool:
        """Return True when the heater has a device entry."""
        return isinstance(device_entry, dict)

    def _metric_section(self) -> dict[str, Any]:
        """Return the dictionary with the requested metric values."""
        heater_section = self._heater_section()
        metric = heater_section.get(self._metric_key)
        return metric if isinstance(metric, dict) else {}

    def _raw_native_value(self) -> Any:
        """Return the raw metric value for this heater address."""
        return self._metric_section().get(self._addr)

    def _coerce_native_value(self, raw: Any) -> float | None:
        """Convert the raw metric value into a float."""
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    @property
    def native_value(self) -> float | None:
        """Return the processed metric value for Home Assistant."""
        raw = self._raw_native_value()
        if raw is None:
            return None
        return self._coerce_native_value(raw)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return identifiers that locate the heater metric."""
        return {"dev_id": self._dev_id, "addr": self._addr}


class HeaterEnergyTotalSensor(HeaterEnergyBase):
    """Total energy consumption sensor for a heater."""

    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_state_class = SensorStateClass.TOTAL_INCREASING
    _attr_native_unit_of_measurement = "kWh"
    _metric_key = "energy"

    def _coerce_native_value(self, raw: Any) -> float | None:  # type: ignore[override]
        """Normalise the raw energy metric into kWh."""
        return _normalise_energy_value(self.coordinator, raw)


class HeaterPowerSensor(HeaterEnergyBase):
    """Power sensor for a heater."""

    _attr_device_class = SensorDeviceClass.POWER
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = "W"
    _metric_key = "power"


def _create_heater_sensors(
    coordinator: Any,
    energy_coordinator: EnergyStateCoordinator,
    entry_id: str,
    dev_id: str,
    addr: str,
    base_name: str,
    uid_prefix: str,
    *,
    node_type: str | None = None,
    temperature_cls: type[HeaterTemperatureSensor] = HeaterTemperatureSensor,
    energy_cls: type[HeaterEnergyTotalSensor] = HeaterEnergyTotalSensor,
    power_cls: type[HeaterPowerSensor] = HeaterPowerSensor,
) -> tuple[
    HeaterTemperatureSensor,
    HeaterEnergyTotalSensor,
    HeaterPowerSensor,
]:
    """Create the three heater node sensors for the given node."""

    temperature = temperature_cls(
        coordinator,
        entry_id,
        dev_id,
        addr,
        f"{base_name} Temperature",
        f"{uid_prefix}:temp",
        base_name,
        node_type=node_type,
    )
    energy = energy_cls(
        energy_coordinator,
        entry_id,
        dev_id,
        addr,
        f"{base_name} Energy",
        f"{uid_prefix}:energy",
        base_name,
        node_type=node_type,
    )
    power = power_cls(
        energy_coordinator,
        entry_id,
        dev_id,
        addr,
        f"{base_name} Power",
        f"{uid_prefix}:power",
        base_name,
        node_type=node_type,
    )

    return (temperature, energy, power)


class InstallationTotalEnergySensor(CoordinatorEntity, SensorEntity):
    """Total energy consumption across all heaters."""

    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_state_class = SensorStateClass.TOTAL_INCREASING
    _attr_native_unit_of_measurement = "kWh"

    def __init__(
        self,
        coordinator: EnergyStateCoordinator,
        entry_id: str,
        dev_id: str,
        name: str,
        unique_id: str,
    ) -> None:
        """Initialise the installation-wide energy sensor."""
        super().__init__(coordinator)
        self._entry_id = entry_id
        self._dev_id = dev_id
        self._attr_name = name
        self._attr_unique_id = unique_id
        self._ws_subscription = DispatcherSubscriptionHelper(self)

    async def async_added_to_hass(self) -> None:
        """Register websocket callbacks once the entity is added."""
        await super().async_added_to_hass()
        if self.hass is None:
            return
        self._ws_subscription.subscribe(
            self.hass, signal_ws_data(self._entry_id), self._on_ws_data
        )

    async def async_will_remove_from_hass(self) -> None:
        """Tidy up websocket listeners prior to entity removal."""

        self._ws_subscription.unsubscribe()
        await super().async_will_remove_from_hass()

    @property
    def device_info(self) -> DeviceInfo:
        """Return the Home Assistant device metadata for the gateway."""
        return build_gateway_device_info(self.hass, self._entry_id, self._dev_id)

    @callback
    def _on_ws_data(self, payload: dict) -> None:
        """Handle websocket payloads that may update the totals."""
        if payload.get("dev_id") != self._dev_id:
            return
        self.schedule_update_ha_state()

    @property
    def available(self) -> bool:
        """Return True if the latest coordinator data contains totals."""
        d = (self.coordinator.data or {}).get(self._dev_id)
        return d is not None

    @property
    def native_value(self) -> float | None:
        """Return the summed energy usage across all heaters."""
        d = (self.coordinator.data or {}).get(self._dev_id, {})
        sections: list[dict[str, Any]] = []
        seen: set[int] = set()
        nodes_by_type = d.get("nodes_by_type")
        if isinstance(nodes_by_type, dict):
            for node_type in HEATER_NODE_TYPES:
                section = nodes_by_type.get(node_type)
                if not isinstance(section, dict):
                    continue
                energy_map = section.get("energy")
                if isinstance(energy_map, dict):
                    sections.append(energy_map)
                    seen.add(id(energy_map))
        legacy = (d.get("htr") or {}).get("energy")
        if isinstance(legacy, dict) and id(legacy) not in seen:
            sections.append(legacy)

        total = 0.0
        found = False
        for energy_map in sections:
            for val in energy_map.values():
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
        """Return identifiers describing the aggregated energy value."""
        return {"dev_id": self._dev_id}
