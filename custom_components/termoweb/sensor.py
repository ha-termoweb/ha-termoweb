"""Sensor platform entities for TermoWeb heaters and gateways."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import logging
from typing import Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.const import STATE_UNKNOWN, UnitOfTime
from homeassistant.core import callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity import DeviceInfo, EntityCategory
from homeassistant.helpers.typing import StateType
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from custom_components.termoweb.backend.factory import backend_capabilities
from custom_components.termoweb.coerce import as_float, as_int, as_percentage
from custom_components.termoweb.const import signal_radio_frames
from custom_components.termoweb.coordinator import EnergyStateCoordinator
from custom_components.termoweb.domain.ids import HEATING_NODE_TYPES
from custom_components.termoweb.domain.view import DomainStateView
from custom_components.termoweb.entity import (
    HeaterNodeBase,
    HeaterPlatformDetails,
    heater_platform_details_for_entry,
    iter_boostable_heater_nodes,
    log_skipped_nodes,
)
from custom_components.termoweb.identifiers import (
    build_gateway_entity_unique_id,
    build_heater_energy_unique_id,
    build_heater_unique_id,
    build_installation_entity_unique_id,
    build_power_monitor_energy_unique_id,
    build_power_monitor_power_unique_id,
    thermostat_fallback_name,
)
from custom_components.termoweb.inventory import (
    Inventory,
    PowerMonitorNode,
    normalize_node_addr,
    normalize_node_type,
)
from custom_components.termoweb.runtime import require_runtime
from custom_components.termoweb.utils import (
    build_gateway_device_info,
    build_installation_device_info,
    build_power_monitor_device_info,
)

_LOGGER = logging.getLogger(__name__)


def _power_monitor_display_name(node: PowerMonitorNode, addr: str) -> str:
    """Return the display name for a power monitor address."""

    return node.name.strip() or f"Power Monitor {addr}"


async def async_setup_entry(hass, entry, async_add_entities):
    """Set up sensors for each heater node."""
    runtime = require_runtime(hass, entry.entry_id)
    coordinator = runtime.coordinator
    dev_id = runtime.dev_id
    capabilities = backend_capabilities(runtime.brand)
    domain_view = coordinator.domain_view

    def default_name(addr: str) -> str:
        """Return the fallback name for heater nodes, as every platform does."""

        return f"Heater {addr}"

    heater_details = heater_platform_details_for_entry(
        runtime,
        default_name_simple=default_name,
    )
    inventory = heater_details.inventory

    energy_coordinator = runtime.energy_coordinator

    power_monitor_entities: list[SensorEntity] = []
    discovered_power_monitors = False
    for metadata in inventory.iter_nodes_metadata(node_types=("pmo",)):
        discovered_power_monitors = True
        display_name = _power_monitor_display_name(metadata.node, metadata.addr)
        energy_unique_id = build_power_monitor_energy_unique_id(dev_id, metadata.addr)
        power_unique_id = build_power_monitor_power_unique_id(dev_id, metadata.addr)
        power_monitor_entities.append(
            PowerMonitorEnergySensor(
                energy_coordinator,
                entry.entry_id,
                dev_id,
                metadata.addr,
                energy_unique_id,
                device_name=display_name,
                inventory=heater_details.inventory,
                domain_view=domain_view,
            )
        )
        power_monitor_entities.append(
            PowerMonitorPowerSensor(
                energy_coordinator,
                entry.entry_id,
                dev_id,
                metadata.addr,
                power_unique_id,
                device_name=display_name,
                inventory=heater_details.inventory,
                domain_view=domain_view,
            )
        )

    if not discovered_power_monitors:
        _LOGGER.debug(
            "No TermoWeb power monitors discovered for %s; skipping power sensors",
            dev_id,
        )

    new_entities: list[SensorEntity] = []
    for node_type, _node, addr_str, base_name in heater_details.iter_metadata():
        canonical_type = normalize_node_type(
            node_type,
            use_default_when_falsey=True,
        )
        addr = normalize_node_addr(
            addr_str,
            use_default_when_falsey=True,
        )
        if canonical_type == "thm":
            heater_fallback = default_name(addr)
            if base_name == heater_fallback:
                base_name = thermostat_fallback_name(addr)

        new_entities.extend(
            _create_heater_sensors(
                coordinator,
                energy_coordinator,
                domain_view,
                entry.entry_id,
                dev_id,
                addr,
                base_name,
                node_type=canonical_type,
                inventory=heater_details.inventory,
                include_energy=capabilities.energy,
            )
        )

        if canonical_type == "thm":
            battery_unique_id = build_heater_unique_id(
                dev_id,
                canonical_type,
                addr,
                suffix=":battery",
            )
            new_entities.append(
                ThermostatBatterySensor(
                    coordinator,
                    entry.entry_id,
                    dev_id,
                    addr,
                    unique_id=battery_unique_id,
                    device_name=base_name,
                    node_type=canonical_type,
                    inventory=heater_details.inventory,
                )
            )

    for node_type, _node, addr_str, base_name in iter_boostable_heater_nodes(
        heater_details,
    ):
        new_entities.extend(
            _create_boost_sensors(
                coordinator,
                entry.entry_id,
                dev_id,
                addr_str,
                base_name,
                node_type=node_type,
                inventory=heater_details.inventory,
            )
        )

    new_entities.extend(power_monitor_entities)

    log_skipped_nodes(
        "sensor",
        heater_details,
        logger=_LOGGER,
        skipped_types=("thm",),
    )

    if capabilities.energy:
        uid_total = build_installation_entity_unique_id(dev_id, "energy_total")
        new_entities.append(
            InstallationTotalEnergySensor(
                energy_coordinator,
                entry.entry_id,
                dev_id,
                uid_total,
                heater_details,
                domain_view,
            )
        )

    if capabilities.geo_data:
        new_entities.append(
            InstallationInfoSensor(
                coordinator,
                entry.entry_id,
                dev_id,
            )
        )

    if capabilities.frame_monitor:
        new_entities.append(RadioFramesSensor(entry.entry_id, dev_id))

    _LOGGER.debug("Adding %d TermoWeb sensors", len(new_entities))
    async_add_entities(new_entities)


class HeaterTemperatureSensor(HeaterNodeBase, SensorEntity):
    """Temperature sensor for a single heater node (read-only mtemp)."""

    _attr_device_class = SensorDeviceClass.TEMPERATURE
    _attr_has_entity_name = True
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_translation_key = "heater_temperature"

    def __init__(
        self,
        coordinator,
        entry_id: str,
        dev_id: str,
        addr: str,
        unique_id: str,
        device_name: str,
        *,
        node_type: str | None = None,
        inventory: Inventory | None = None,
    ) -> None:
        """Initialise the heater temperature sensor entity."""
        super().__init__(
            coordinator,
            entry_id,
            dev_id,
            addr,
            None,
            unique_id,
            device_name=device_name,
            node_type=node_type,
            inventory=inventory,
        )

    @property
    def native_unit_of_measurement(self) -> str:
        """Return the unit the device reports temperatures in."""
        return self._temperature_unit()

    @property
    def native_value(self) -> float | None:
        """Return the latest temperature reported by the heater."""
        state = self.heater_state()
        return as_float(getattr(state, "mtemp", None))

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return metadata describing the heater temperature source."""
        state = self.heater_state()
        return {
            "dev_id": self._dev_id,
            "addr": self._addr,
            "units": getattr(state, "units", None),
        }


class ThermostatBatterySensor(HeaterNodeBase, SensorEntity):
    """Battery level sensor for battery-powered thermostat nodes."""

    _attr_device_class = SensorDeviceClass.BATTERY
    _attr_native_unit_of_measurement = "%"
    _attr_state_class = SensorStateClass.MEASUREMENT

    def __init__(
        self,
        coordinator,
        entry_id: str,
        dev_id: str,
        addr: str,
        *,
        unique_id: str,
        device_name: str,
        node_type: str | None = None,
        inventory: Inventory | None = None,
    ) -> None:
        """Initialise the thermostat battery sensor entity."""

        super().__init__(
            coordinator,
            entry_id,
            dev_id,
            addr,
            None,
            unique_id,
            device_name=device_name,
            node_type=node_type,
            inventory=inventory,
        )
        self._attr_name = f"{device_name} Battery"

    @staticmethod
    def _coerce_level(value: Any) -> int | None:
        """Return a clamped 0–5 battery level from ``value`` when possible."""

        level = as_int(value)
        return None if level is None else max(0, min(5, level))

    @property
    def native_value(self) -> int | None:
        """Return the thermostat battery percentage as 0–100."""

        state = self.heater_state()
        raw_level = getattr(state, "batt_level", None)
        level = self._coerce_level(raw_level)
        if level is None:
            return None
        return level * 20

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return additional thermostat battery metadata."""

        state = self.heater_state()
        level = self._coerce_level(getattr(state, "batt_level", None))
        return {
            "dev_id": self._dev_id,
            "addr": self._addr,
            "batt_level_steps": level,
        }


class AccumulatorChargeSensorBase(HeaterNodeBase, SensorEntity):
    """Base helper exposing accumulator charge metadata sensors."""

    _metric_key: str

    def _raw_value(self) -> Any:
        """Return the raw setting backing this accumulator charge sensor."""

        state = self.accumulator_state()
        return getattr(state, self._metric_key, None) if state is not None else None

    @property
    def native_value(self) -> StateType:
        """Return the processed accumulator charge metric."""

        return self._coerce_value(self._raw_value())

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return identifiers that locate the accumulator charge metric."""

        return {"dev_id": self._dev_id, "addr": self._addr}


class AccumulatorChargingSensor(AccumulatorChargeSensorBase):
    """Boolean sensor indicating whether the accumulator is charging."""

    _attr_has_entity_name = True
    _attr_translation_key = "accumulator_charging"
    _metric_key = "charging"

    def _coerce_value(self, raw: Any) -> StateType:  # type: ignore[override]
        """Return a canonical boolean charging state when available."""

        return raw


class AccumulatorChargePercentageSensor(AccumulatorChargeSensorBase):
    """Base helper converting accumulator charge percentages."""

    _attr_has_entity_name = True
    _attr_native_unit_of_measurement = "%"
    _attr_state_class = SensorStateClass.MEASUREMENT

    def _coerce_value(self, raw: Any) -> StateType:  # type: ignore[override]
        """Return the accumulator charge percentage as an integer."""

        return as_percentage(raw)


class AccumulatorCurrentChargeSensor(AccumulatorChargePercentageSensor):
    """Sensor exposing the accumulator's current charge percentage."""

    _attr_translation_key = "accumulator_current_charge"
    _metric_key = "current_charge_per"


class AccumulatorTargetChargeSensor(AccumulatorChargePercentageSensor):
    """Sensor exposing the accumulator's target charge percentage."""

    _attr_translation_key = "accumulator_target_charge"
    _metric_key = "target_charge_per"


class HeaterEnergyBase(HeaterNodeBase, SensorEntity):
    """Base helper for heater measurement sensors such as power and energy."""

    _metric_key: str

    def __init__(
        self,
        coordinator: EnergyStateCoordinator,
        domain_view: DomainStateView | None,
        entry_id: str,
        dev_id: str,
        addr: str,
        unique_id: str,
        device_name: str,
        *,
        node_type: str | None = None,
        inventory: Inventory | None = None,
    ) -> None:
        """Initialise a heater energy-derived sensor entity."""
        super().__init__(
            coordinator,
            entry_id,
            dev_id,
            addr,
            None,
            unique_id,
            device_name=device_name,
            node_type=node_type,
            inventory=inventory,
        )
        self._domain_view = domain_view

    def _metric_entry(self) -> Any:
        """Return the energy metrics for this heater from the domain view."""

        return self._domain_view.get_energy_metric(self._node_type, self._addr)

    def _raw_native_value(self) -> Any:
        """Return the raw metric value for this heater address."""
        metrics = self._metric_entry()
        if metrics is None:
            return None
        if self._metric_key == "energy":
            return metrics.energy_kwh
        return metrics.power_w

    def _coerce_native_value(self, raw: Any) -> float | None:
        """Convert the raw metric value into a float."""
        return as_float(raw)

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
    _attr_has_entity_name = True
    _attr_state_class = SensorStateClass.TOTAL_INCREASING
    _attr_native_unit_of_measurement = "kWh"
    _attr_translation_key = "heater_energy_total"
    _metric_key = "energy"


class HeaterPowerSensor(HeaterEnergyBase):
    """Power sensor for a heater."""

    _attr_device_class = SensorDeviceClass.POWER
    _attr_has_entity_name = True
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = "W"
    _attr_translation_key = "heater_power"
    _metric_key = "power"


class HeaterBoostMinutesRemainingSensor(HeaterNodeBase, SensorEntity):
    """Sensor exposing the remaining minutes for the active boost."""

    _attr_device_class = SensorDeviceClass.DURATION
    _attr_has_entity_name = True
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = UnitOfTime.MINUTES
    _attr_translation_key = "boost_minutes_remaining"

    def __init__(
        self,
        coordinator,
        entry_id: str,
        dev_id: str,
        addr: str,
        name: str | None,
        unique_id: str,
        *,
        device_name: str | None = None,
        node_type: str | None = None,
        inventory: Inventory | None = None,
    ) -> None:
        """Initialise the boost duration helper sensor."""

        resolved_device_name = device_name or name
        super().__init__(
            coordinator,
            entry_id,
            dev_id,
            addr,
            name,
            unique_id,
            device_name=resolved_device_name,
            node_type=node_type,
            inventory=inventory,
        )

    @property
    def native_value(self) -> int | None:
        """Return the remaining boost duration in minutes."""

        state = self.boost_state()
        return state.minutes_remaining

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return metadata about the boost session."""

        state = self.boost_state()
        return {
            "dev_id": self._dev_id,
            "addr": self._addr,
            "boost_active": state.active,
            "boost_end": state.end_iso,
            "boost_end_label": state.end_label,
        }


class HeaterBoostEndSensor(HeaterNodeBase, SensorEntity):
    """Sensor exposing the expected end timestamp for the active boost."""

    _attr_device_class = SensorDeviceClass.TIMESTAMP
    _attr_has_entity_name = True
    _attr_translation_key = "boost_end"

    def __init__(
        self,
        coordinator,
        entry_id: str,
        dev_id: str,
        addr: str,
        name: str | None,
        unique_id: str,
        *,
        device_name: str | None = None,
        node_type: str | None = None,
        inventory: Inventory | None = None,
    ) -> None:
        """Initialise the boost end timestamp sensor."""

        resolved_device_name = device_name or name
        super().__init__(
            coordinator,
            entry_id,
            dev_id,
            addr,
            name,
            unique_id,
            device_name=resolved_device_name,
            node_type=node_type,
            inventory=inventory,
        )

    @property
    def native_value(self) -> datetime | None:
        """Return the boost end timestamp."""

        state = self.boost_state()
        return state.end_datetime

    @property
    def state(self) -> StateType:  # type: ignore[override]
        """Return the Home Assistant state value for the boost end sensor."""

        ha_state = super().state
        state = self.boost_state()
        if ha_state in (STATE_UNKNOWN, None):
            end_dt = state.end_datetime
            if end_dt is not None:
                return end_dt.isoformat()
            if state.end_label:
                return state.end_label
        return ha_state

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return metadata about the boost session."""

        state = self.boost_state()
        return {
            "dev_id": self._dev_id,
            "addr": self._addr,
            "boost_active": state.active,
            "boost_minutes_remaining": state.minutes_remaining,
            "boost_end_label": state.end_label,
        }


def _create_heater_sensors(
    coordinator: Any,
    energy_coordinator: EnergyStateCoordinator,
    domain_view: DomainStateView | None,
    entry_id: str,
    dev_id: str,
    addr: str,
    base_name: str,
    *,
    node_type: str | None = None,
    inventory: Inventory | None = None,
    include_energy: bool = True,
    temperature_cls: type[HeaterTemperatureSensor] = HeaterTemperatureSensor,
    energy_cls: type[HeaterEnergyTotalSensor] = HeaterEnergyTotalSensor,
    power_cls: type[HeaterPowerSensor] = HeaterPowerSensor,
) -> tuple[SensorEntity, ...]:
    """Create heater node sensors for ``addr`` including energy when available."""

    canonical_type = normalize_node_type(
        node_type,
        use_default_when_falsey=True,
    )
    canonical_addr = normalize_node_addr(addr, use_default_when_falsey=True)

    target_type = canonical_type or "htr"
    temperature_unique_id = build_heater_unique_id(
        dev_id,
        target_type,
        canonical_addr,
        suffix=":temp",
    )

    sensors: list[SensorEntity] = [
        temperature_cls(
            coordinator,
            entry_id,
            dev_id,
            canonical_addr,
            temperature_unique_id,
            device_name=base_name,
            node_type=target_type,
            inventory=inventory,
        )
    ]

    if target_type == "acm":
        charging_unique_id = build_heater_unique_id(
            dev_id,
            target_type,
            canonical_addr,
            suffix=":charging",
        )
        current_charge_unique_id = build_heater_unique_id(
            dev_id,
            target_type,
            canonical_addr,
            suffix=":current_charge_per",
        )
        target_charge_unique_id = build_heater_unique_id(
            dev_id,
            target_type,
            canonical_addr,
            suffix=":target_charge_per",
        )
        sensors.extend(
            (
                AccumulatorChargingSensor(
                    coordinator,
                    entry_id,
                    dev_id,
                    canonical_addr,
                    None,
                    charging_unique_id,
                    device_name=base_name,
                    node_type=target_type,
                    inventory=inventory,
                ),
                AccumulatorCurrentChargeSensor(
                    coordinator,
                    entry_id,
                    dev_id,
                    canonical_addr,
                    None,
                    current_charge_unique_id,
                    device_name=base_name,
                    node_type=target_type,
                    inventory=inventory,
                ),
                AccumulatorTargetChargeSensor(
                    coordinator,
                    entry_id,
                    dev_id,
                    canonical_addr,
                    None,
                    target_charge_unique_id,
                    device_name=base_name,
                    node_type=target_type,
                    inventory=inventory,
                ),
            )
        )

    if target_type != "thm" and include_energy:
        energy_unique_id = build_heater_energy_unique_id(
            dev_id,
            target_type,
            canonical_addr,
        )
        power_unique_id = build_heater_unique_id(
            dev_id, target_type, canonical_addr, suffix="power"
        )
        sensors.extend(
            (
                energy_cls(
                    energy_coordinator,
                    domain_view,
                    entry_id,
                    dev_id,
                    canonical_addr,
                    energy_unique_id,
                    device_name=base_name,
                    node_type=target_type,
                    inventory=inventory,
                ),
                power_cls(
                    energy_coordinator,
                    domain_view,
                    entry_id,
                    dev_id,
                    canonical_addr,
                    power_unique_id,
                    device_name=base_name,
                    node_type=target_type,
                    inventory=inventory,
                ),
            )
        )

    return tuple(sensors)


def _create_boost_sensors(
    coordinator: Any,
    entry_id: str,
    dev_id: str,
    addr: str,
    base_name: str,
    *,
    node_type: str | None = None,
    inventory: Inventory | None = None,
    minutes_cls: type[
        HeaterBoostMinutesRemainingSensor
    ] = HeaterBoostMinutesRemainingSensor,
    end_cls: type[HeaterBoostEndSensor] = HeaterBoostEndSensor,
) -> tuple[
    HeaterBoostMinutesRemainingSensor,
    HeaterBoostEndSensor,
]:
    """Create the boost-related sensors for a heater node."""

    uid_type = node_type or "htr"
    minutes = minutes_cls(
        coordinator,
        entry_id,
        dev_id,
        addr,
        name=None,
        unique_id=build_heater_unique_id(
            dev_id, uid_type, addr, suffix="boost_minutes_remaining"
        ),
        device_name=base_name,
        node_type=node_type,
        inventory=inventory,
    )
    end = end_cls(
        coordinator,
        entry_id,
        dev_id,
        addr,
        name=None,
        unique_id=build_heater_unique_id(dev_id, uid_type, addr, suffix="boost_end"),
        device_name=base_name,
        node_type=node_type,
        inventory=inventory,
    )

    return (minutes, end)


class PowerMonitorSensorBase(CoordinatorEntity, SensorEntity):
    """Base helper exposing shared behaviour for power monitor sensors."""

    _metric_key: str

    def __init__(
        self,
        coordinator: Any,
        entry_id: str,
        dev_id: str,
        addr: str,
        unique_id: str,
        device_name: str,
        *,
        inventory: Inventory,
        domain_view: DomainStateView | None = None,
    ) -> None:
        """Initialise the power monitor sensor base entity."""

        super().__init__(coordinator)
        self._entry_id = entry_id
        self._dev_id = dev_id
        normalized_addr = normalize_node_addr(addr, use_default_when_falsey=True)
        self._addr = normalized_addr or str(addr)
        self._attr_unique_id = unique_id
        self._device_name = device_name
        self._inventory = inventory
        self._domain_view = domain_view

    def _metric_entry(self) -> Any:
        """Return the energy metrics for this power monitor from the domain view."""

        return self._domain_view.get_energy_metric("pmo", self._addr)

    def _coerce_native_value(self, raw: Any) -> float | None:
        """Convert a metric payload value to ``float`` if possible."""

        try:
            return float(raw)
        except TypeError, ValueError:
            return None

    @property
    def should_poll(self) -> bool:
        """Coordinator updates push new data for power monitors."""

        return False

    @property
    def available(self) -> bool:
        """Return True when the last update succeeded and the monitor is tracked."""

        if not super().available:
            return False
        metrics = self._metric_entry()
        if metrics is not None:
            return True
        return self._inventory.has_node("pmo", self._addr)

    @property
    def native_value(self) -> float | None:
        """Return the processed metric value for Home Assistant."""

        metrics = self._metric_entry()
        if metrics is None:
            return None
        attr = "energy_kwh" if self._metric_key == "energy" else "power_w"
        value = getattr(metrics, attr, None)
        return self._coerce_native_value(value)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return identifiers for the power monitor metric."""

        return {"dev_id": self._dev_id, "addr": self._addr}

    @property
    def device_info(self) -> DeviceInfo:
        """Return the Home Assistant device metadata for the power monitor."""

        return build_power_monitor_device_info(
            self.hass,
            self._entry_id,
            self._dev_id,
            self._addr,
            name=self._device_name,
        )


class PowerMonitorEnergySensor(PowerMonitorSensorBase):
    """Energy consumption sensor for a power monitor."""

    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_has_entity_name = True
    _attr_state_class = SensorStateClass.TOTAL_INCREASING
    _attr_native_unit_of_measurement = "kWh"
    _attr_translation_key = "power_monitor_energy"
    _metric_key = "energy"


class PowerMonitorPowerSensor(PowerMonitorSensorBase):
    """Power sensor for a power monitor."""

    _attr_device_class = SensorDeviceClass.POWER
    _attr_has_entity_name = True
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = "W"
    _attr_translation_key = "power_monitor_power"
    _metric_key = "power"


class InstallationTotalEnergySensor(CoordinatorEntity, SensorEntity):
    """Total energy consumption across all heaters."""

    _attr_has_entity_name = True
    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_state_class = SensorStateClass.TOTAL_INCREASING
    _attr_native_unit_of_measurement = "kWh"
    _attr_translation_key = "installation_total_energy"

    def __init__(
        self,
        coordinator: EnergyStateCoordinator,
        entry_id: str,
        dev_id: str,
        unique_id: str,
        details: HeaterPlatformDetails,
        domain_view: DomainStateView | None,
    ) -> None:
        """Initialise the installation-wide energy sensor."""
        super().__init__(coordinator)
        self._entry_id = entry_id
        self._dev_id = dev_id
        self._attr_unique_id = unique_id
        self._details = details
        self._domain_view = domain_view

    @property
    def device_info(self) -> DeviceInfo:
        """Return the Home Assistant device metadata for the installation."""
        return build_installation_device_info(self.hass, self._entry_id, self._dev_id)

    @property
    def available(self) -> bool:
        """Return True if the last update succeeded and energy totals exist."""
        if not super().available:
            return False
        return self._domain_view.get_energy_snapshot() is not None

    @property
    def native_value(self) -> float | None:
        """Return the summed heater energy, or None unless every heater reports."""
        view = self._domain_view
        total = 0.0
        found = False
        for node_type, addrs in self._details.addrs_by_type.items():
            # The installation total sums heating nodes; thermostats meter no energy.
            if node_type not in HEATING_NODE_TYPES:
                continue
            metrics_by_addr = view.get_energy_metrics_for_type(node_type)
            for addr in addrs:
                metric = metrics_by_addr.get(addr)
                normalised = None if metric is None else as_float(metric.energy_kwh)
                if normalised is None:
                    return None
                total += normalised
                found = True
        if not found:
            return None
        return total


class InstallationInfoSensor(CoordinatorEntity, SensorEntity):
    """Diagnostic sensor exposing installation metadata and geo location."""

    _attr_has_entity_name = True
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_translation_key = "installation_info"

    def __init__(
        self,
        coordinator: Any,
        entry_id: str,
        dev_id: str,
    ) -> None:
        """Initialise the installation info diagnostic sensor."""
        super().__init__(coordinator)
        self._entry_id = entry_id
        self._dev_id = dev_id
        self._attr_unique_id = build_installation_entity_unique_id(dev_id, "info")

    @property
    def device_info(self) -> DeviceInfo:
        """Return the Home Assistant device metadata for the installation."""
        return build_installation_device_info(self.hass, self._entry_id, self._dev_id)

    @property
    def native_value(self) -> str | None:
        """Return a summary location string as the sensor state."""
        geo_data = self.coordinator.device_metadata.geo_data
        if geo_data is None:
            return None
        parts = [
            part
            for part in (geo_data.city, geo_data.state, geo_data.country)
            if part is not None
        ]
        return ", ".join(parts) if parts else None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return geo location and timezone details for the installation."""
        geo_data = self.coordinator.device_metadata.geo_data
        if geo_data is None:
            return {}
        fields = (
            ("country", geo_data.country),
            ("state", geo_data.state),
            ("city", geo_data.city),
            ("timezone", geo_data.tz_code),
            ("zip", geo_data.zip),
        )
        return {key: value for key, value in fields if value is not None}


class RadioFramesSensor(SensorEntity):
    """Frames a listen-only radio gateway has heard since Home Assistant started."""

    _attr_has_entity_name = True
    _attr_translation_key = "frames_heard"
    _attr_should_poll = False
    _attr_state_class = SensorStateClass.TOTAL_INCREASING
    _attr_native_unit_of_measurement = "frames"

    def __init__(self, entry_id: str, dev_id: str) -> None:
        """Start at zero frames; the radio monitor pushes every new count."""
        self._entry_id = entry_id
        self._dev_id = str(dev_id)
        self._attr_unique_id = build_gateway_entity_unique_id(
            self._dev_id, "frames_heard"
        )
        self._attr_native_value = 0
        self._attr_extra_state_attributes = {"last_frame": None}

    @property
    def device_info(self) -> DeviceInfo:
        """Return Home Assistant device metadata for the gateway."""
        return build_gateway_device_info(self.hass, self._entry_id, self._dev_id)

    async def async_added_to_hass(self) -> None:
        """Follow the radio monitor's frame count."""
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass, signal_radio_frames(self._entry_id), self._handle_frames
            )
        )

    @callback
    def _handle_frames(self, payload: Mapping[str, Any]) -> None:
        """Show the new frame count and the time of the last frame."""
        self._attr_native_value = payload.get("frames", 0)
        self._attr_extra_state_attributes = {"last_frame": payload.get("last_frame")}
        self.async_write_ha_state()
