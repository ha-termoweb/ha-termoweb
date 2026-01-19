"""Sensor platform entities for TermoWeb heaters and gateways."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import logging
import math
from typing import Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.const import STATE_UNKNOWN, UnitOfTemperature, UnitOfTime
from homeassistant.core import callback
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.typing import StateType
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from custom_components.termoweb.const import DOMAIN, signal_ws_data
from custom_components.termoweb.coordinator import EnergyStateCoordinator
from custom_components.termoweb.domain.energy import coerce_snapshot
from custom_components.termoweb.entities.entity import GatewayDispatcherEntity
from custom_components.termoweb.entities.heater import (
    HeaterNodeBase,
    HeaterPlatformDetails,
    heater_platform_details_for_entry,
    iter_boostable_heater_nodes,
    log_skipped_nodes,
)
from custom_components.termoweb.i18n import (
    async_get_fallback_translations,
    attach_fallbacks,
    format_fallback,
)
from custom_components.termoweb.identifiers import (
    build_heater_energy_unique_id,
    build_heater_entity_unique_id,
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
    build_power_monitor_device_info,
    float_or_none,
)

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


def _power_monitor_display_name(
    node: PowerMonitorNode, addr: str, fallbacks: Mapping[str, str] | None
) -> str:
    """Return the display name for a power monitor address."""

    raw_name = getattr(node, "name", None)
    trimmed = raw_name.strip() if isinstance(raw_name, str) else None
    if trimmed:
        return trimmed

    default_factory = getattr(node, "default_name", None)
    if callable(default_factory):
        try:
            fallback = default_factory()
        except Exception:  # pragma: no cover - defensive fallback  # noqa: BLE001
            fallback = None
        if isinstance(fallback, str):
            fallback_trimmed = fallback.strip()
            if fallback_trimmed:
                return fallback_trimmed

    return format_fallback(
        fallbacks,
        "fallbacks.power_monitor_name",
        "Power Monitor {addr}",
        addr=addr,
    )


async def async_setup_entry(hass, entry, async_add_entities):
    """Set up sensors for each heater node."""
    runtime = require_runtime(hass, entry.entry_id)
    coordinator = runtime.coordinator
    dev_id = runtime.dev_id

    fallbacks = await async_get_fallback_translations(hass, runtime)
    attach_fallbacks(coordinator, fallbacks)

    def default_name(addr: str) -> str:
        """Return a placeholder name for heater nodes."""

        return format_fallback(
            fallbacks,
            "fallbacks.node_name",
            "Node {addr}",
            addr=addr,
        )

    heater_details = heater_platform_details_for_entry(
        runtime,
        default_name_simple=default_name,
    )
    inventory = heater_details.inventory

    energy_coordinator = runtime.energy_coordinator
    if not isinstance(energy_coordinator, EnergyStateCoordinator):
        if not hasattr(energy_coordinator, "update_addresses"):
            energy_coordinator = EnergyStateCoordinator(
                hass, runtime.client, dev_id, inventory
            )
            runtime.energy_coordinator = energy_coordinator
            await energy_coordinator.async_config_entry_first_refresh()

    energy_coordinator.update_addresses(inventory)
    attach_fallbacks(energy_coordinator, fallbacks)

    power_monitor_entities: list[SensorEntity] = []
    discovered_power_monitors = False
    for metadata in inventory.iter_nodes_metadata(node_types=("pmo",)):
        node = metadata.node
        if not isinstance(node, PowerMonitorNode):
            continue
        discovered_power_monitors = True
        display_name = _power_monitor_display_name(node, metadata.addr, fallbacks)
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
        if not canonical_type or not addr:
            continue
        if canonical_type == "thm":
            heater_fallback = default_name(addr)
            thermostat_default = format_fallback(
                fallbacks,
                "fallbacks.thermostat_name",
                thermostat_fallback_name(addr),
                addr=addr,
            )
            if base_name == heater_fallback:
                base_name = thermostat_default

        new_entities.extend(
            _create_heater_sensors(
                coordinator,
                energy_coordinator,
                entry.entry_id,
                dev_id,
                addr,
                base_name,
                node_type=canonical_type,
                inventory=heater_details.inventory,
            )
        )

        if canonical_type == "thm":
            battery_unique_id = build_heater_entity_unique_id(
                dev_id,
                canonical_type,
                addr,
                ":battery",
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
        energy_unique_id = build_heater_energy_unique_id(dev_id, node_type, addr_str)
        uid_prefix = energy_unique_id.rsplit(":", 1)[0]
        new_entities.extend(
            _create_boost_sensors(
                coordinator,
                entry.entry_id,
                dev_id,
                addr_str,
                base_name,
                uid_prefix,
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

    uid_total = f"{DOMAIN}:{dev_id}:energy_total"
    new_entities.append(
        InstallationTotalEnergySensor(
            energy_coordinator,
            entry.entry_id,
            dev_id,
            uid_total,
            heater_details,
        )
    )

    if new_entities:
        _LOGGER.debug("Adding %d TermoWeb sensors", len(new_entities))
        async_add_entities(new_entities)


class HeaterTemperatureSensor(HeaterNodeBase, SensorEntity):
    """Temperature sensor for a single heater node (read-only mtemp)."""

    _attr_device_class = SensorDeviceClass.TEMPERATURE
    _attr_has_entity_name = True
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS
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
    def native_value(self) -> float | None:
        """Return the latest temperature reported by the heater."""
        state = self.heater_state()
        return float_or_none(getattr(state, "mtemp", None))

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

    _attr_device_class = getattr(SensorDeviceClass, "BATTERY", None)
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
        if device_name:
            self._attr_name = f"{device_name} Battery"
        else:
            self._attr_name = "Thermostat Battery"

    @staticmethod
    def _coerce_level(value: Any) -> int | None:
        """Return a clamped 0–5 battery level from ``value`` when possible."""

        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            try:
                numeric = float(str(value).strip())
            except (TypeError, ValueError):
                return None
        if math.isnan(numeric):
            return None
        return max(0, min(5, int(numeric)))

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

    def _coerce_value(self, raw: Any) -> StateType:
        """Convert a raw accumulator charge setting into a Home Assistant state."""

        return raw

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

        if isinstance(raw, bool):
            return raw
        if raw is None:
            return None
        if isinstance(raw, (int, float)):
            return bool(raw)
        return None


class AccumulatorChargePercentageSensor(AccumulatorChargeSensorBase):
    """Base helper converting accumulator charge percentages."""

    _attr_has_entity_name = True
    _attr_native_unit_of_measurement = "%"
    _attr_state_class = SensorStateClass.MEASUREMENT

    def _coerce_value(self, raw: Any) -> StateType:  # type: ignore[override]
        """Return the accumulator charge percentage as an integer."""

        numeric = float_or_none(raw)
        if numeric is None or not math.isfinite(numeric):
            return None
        return max(0, min(100, int(numeric)))


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

    def _device_available(self) -> bool:
        """Return True when inventory and coordinator expose this heater node."""

        if not super()._device_available():
            return False

        coordinator = getattr(self, "coordinator", None)
        coordinator_available = getattr(coordinator, "last_update_success", True)
        return bool(coordinator_available)

    def _metric_entry(self) -> Any:
        """Return the cached metric entry for this heater."""

        coordinator = getattr(self, "coordinator", None)
        getter = getattr(coordinator, "metric_for", None)
        if callable(getter):
            return getter(self._node_type, self._addr)
        return None

    def _raw_native_value(self) -> Any:
        """Return the raw metric value for this heater address."""
        metrics = self._metric_entry()
        if metrics is None:
            return None
        if self._metric_key == "energy":
            return getattr(metrics, "energy_kwh", None)
        if self._metric_key == "power":
            return getattr(metrics, "power_w", None)
        return None

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
    _attr_has_entity_name = True
    _attr_state_class = SensorStateClass.TOTAL_INCREASING
    _attr_native_unit_of_measurement = "kWh"
    _attr_translation_key = "heater_energy_total"
    _metric_key = "energy"

    def _coerce_native_value(self, raw: Any) -> float | None:  # type: ignore[override]
        """Normalise the raw energy metric into kWh."""
        return _normalise_energy_value(self.coordinator, raw)


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

    _attr_device_class = getattr(SensorDeviceClass, "DURATION", "duration")
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

    _attr_device_class = getattr(SensorDeviceClass, "TIMESTAMP", "timestamp")
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
                try:
                    return end_dt.isoformat()
                except (AttributeError, TypeError, ValueError):
                    return ha_state
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
    entry_id: str,
    dev_id: str,
    addr: str,
    base_name: str,
    *,
    node_type: str | None = None,
    inventory: Inventory | None = None,
    temperature_cls: type[HeaterTemperatureSensor] = HeaterTemperatureSensor,
    energy_cls: type[HeaterEnergyTotalSensor] = HeaterEnergyTotalSensor,
    power_cls: type[HeaterPowerSensor] = HeaterPowerSensor,
) -> tuple[SensorEntity, ...]:
    """Create heater node sensors for ``addr`` including energy when available."""

    canonical_type = normalize_node_type(
        node_type,
        use_default_when_falsey=True,
    )
    canonical_addr = normalize_node_addr(
        addr,
        use_default_when_falsey=True,
    ) or normalize_node_addr(addr)
    if not canonical_addr:
        canonical_addr = str(addr)

    target_type = canonical_type or "htr"
    temperature_unique_id = build_heater_entity_unique_id(
        dev_id,
        target_type,
        canonical_addr,
        ":temp",
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
        charging_unique_id = build_heater_entity_unique_id(
            dev_id,
            target_type,
            canonical_addr,
            ":charging",
        )
        current_charge_unique_id = build_heater_entity_unique_id(
            dev_id,
            target_type,
            canonical_addr,
            ":current_charge_per",
        )
        target_charge_unique_id = build_heater_entity_unique_id(
            dev_id,
            target_type,
            canonical_addr,
            ":target_charge_per",
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

    if target_type != "thm":
        energy_unique_id = build_heater_energy_unique_id(
            dev_id,
            target_type,
            canonical_addr,
        )
        power_unique_id = f"{energy_unique_id.rsplit(':', 1)[0]}:power"
        sensors.extend(
            (
                energy_cls(
                    energy_coordinator,
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
    uid_prefix: str,
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

    boost_prefix = f"{uid_prefix}:boost"
    minutes = minutes_cls(
        coordinator,
        entry_id,
        dev_id,
        addr,
        name=None,
        unique_id=f"{boost_prefix}:minutes_remaining",
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
        unique_id=f"{boost_prefix}:end",
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
    ) -> None:
        """Initialise the power monitor sensor base entity."""

        super().__init__(coordinator)
        self._entry_id = entry_id
        self._dev_id = dev_id
        normalized_addr = normalize_node_addr(addr, use_default_when_falsey=True)
        self._addr = normalized_addr or str(addr)
        self._attr_unique_id = unique_id
        self._device_name = device_name
        self._inventory: Inventory | None = inventory

    def _resolve_inventory(self) -> Inventory | None:
        """Return the immutable inventory backing this sensor."""

        inventory = getattr(self, "_inventory", None)
        if isinstance(inventory, Inventory):
            return inventory
        coordinator_inventory = getattr(self.coordinator, "inventory", None)
        if isinstance(coordinator_inventory, Inventory):
            self._inventory = coordinator_inventory
            return coordinator_inventory
        return None

    def _metric_entry(self) -> Any:
        """Return the cached metric entry for this power monitor."""

        getter = getattr(self.coordinator, "metric_for", None)
        if callable(getter):
            return getter("pmo", self._addr)
        return None

    def _coerce_native_value(self, raw: Any) -> float | None:
        """Convert a metric payload value to ``float`` if possible."""

        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    @property
    def should_poll(self) -> bool:
        """Coordinator updates push new data for power monitors."""

        return False

    @property
    def available(self) -> bool:
        """Return True when the coordinator tracks this power monitor."""

        metrics = self._metric_entry()
        if metrics is not None:
            return True
        inventory = self._resolve_inventory()
        if not isinstance(inventory, Inventory):
            return False
        return inventory.has_node("pmo", self._addr)

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

    def _coerce_native_value(self, raw: Any) -> float | None:  # type: ignore[override]
        """Normalise the raw energy metric into kWh."""

        return _normalise_energy_value(self.coordinator, raw)


class PowerMonitorPowerSensor(PowerMonitorSensorBase):
    """Power sensor for a power monitor."""

    _attr_device_class = SensorDeviceClass.POWER
    _attr_has_entity_name = True
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = "W"
    _attr_translation_key = "power_monitor_power"
    _metric_key = "power"


class InstallationTotalEnergySensor(
    GatewayDispatcherEntity, CoordinatorEntity, SensorEntity
):
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
    ) -> None:
        """Initialise the installation-wide energy sensor."""
        super().__init__(coordinator)
        self._entry_id = entry_id
        self._dev_id = dev_id
        self._attr_unique_id = unique_id
        self._details = details

    @property
    def gateway_signal(self) -> str:
        """Return the dispatcher signal for gateway websocket data."""

        return signal_ws_data(self._entry_id)

    @property
    def device_info(self) -> DeviceInfo:
        """Return the Home Assistant device metadata for the gateway."""
        return build_gateway_device_info(self.hass, self._entry_id, self._dev_id)

    @callback
    def _handle_gateway_dispatcher(self, payload: dict[str, Any]) -> None:
        """Handle websocket payloads that may update the totals."""
        if payload.get("dev_id") != self._dev_id:
            return
        self.schedule_update_ha_state()

    @property
    def available(self) -> bool:
        """Return True if the latest coordinator data contains totals."""
        snapshot = coerce_snapshot(getattr(self, "coordinator", None).data)
        return snapshot is not None and snapshot.dev_id == self._dev_id

    @property
    def native_value(self) -> float | None:
        """Return the summed energy usage across all heaters."""
        snapshot = coerce_snapshot(getattr(self, "coordinator", None).data)
        if snapshot is None or snapshot.dev_id != self._dev_id:
            return None
        total = 0.0
        found = False
        for node_type, addrs in self._details.addrs_by_type.items():
            metrics_by_addr = snapshot.metrics_for_type(node_type)
            if not metrics_by_addr:
                continue
            for addr in addrs:
                metric = metrics_by_addr.get(addr)
                if metric is None:
                    continue
                normalised = _normalise_energy_value(
                    self.coordinator, metric.energy_kwh
                )
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
