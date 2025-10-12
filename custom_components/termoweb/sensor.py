"""Sensor platform entities for TermoWeb heaters and gateways."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
import logging
import math
from typing import Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.const import STATE_UNKNOWN, UnitOfTemperature

try:  # pragma: no cover - fallback for older Home Assistant stubs
    from homeassistant.const import UnitOfTime
except ImportError:  # pragma: no cover - fallback for older Home Assistant stubs
    class UnitOfTime:  # type: ignore[override]
        """Fallback UnitOfTime namespace with minute granularity."""

        MINUTES = "min"
from homeassistant.core import callback
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.typing import StateType
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN, signal_ws_data
from .coordinator import EnergyStateCoordinator
from .entity import GatewayDispatcherEntity
from .heater import (
    HeaterNodeBase,
    HeaterPlatformDetails,
    heater_platform_details_for_entry,
    iter_boostable_heater_nodes,
    log_skipped_nodes,
)
from .identifiers import (
    build_heater_energy_unique_id,
    build_power_monitor_energy_unique_id,
    build_power_monitor_power_unique_id,
)
from .inventory import Inventory, PowerMonitorNode, normalize_node_addr
from .utils import (
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


def _power_monitor_display_name(node: PowerMonitorNode, addr: str) -> str:
    """Return the display name for a power monitor address."""

    raw_name = getattr(node, "name", None)
    trimmed = raw_name.strip() if isinstance(raw_name, str) else None
    if trimmed:
        return trimmed

    default_factory = getattr(node, "default_name", None)
    if callable(default_factory):
        try:
            fallback = default_factory()
        except Exception:  # pragma: no cover - defensive fallback
            fallback = None
        if isinstance(fallback, str):
            fallback_trimmed = fallback.strip()
            if fallback_trimmed:
                return fallback_trimmed

    return f"Power Monitor {addr}"


async def async_setup_entry(hass, entry, async_add_entities):
    """Set up sensors for each heater node."""
    data = hass.data[DOMAIN][entry.entry_id]
    coordinator = data["coordinator"]
    dev_id = data["dev_id"]

    def default_name(addr: str) -> str:
        """Return a placeholder name for heater nodes."""

        return f"Node {addr}"
    heater_details = heater_platform_details_for_entry(
        data,
        default_name_simple=default_name,
    )
    inventory = heater_details.inventory
    addrs_by_type = heater_details.addrs_by_type

    energy_coordinator: EnergyStateCoordinator | None = data.get(
        "energy_coordinator",
    )

    if energy_coordinator is None:
        energy_coordinator = EnergyStateCoordinator(
            hass, data["client"], dev_id, inventory
        )
        data["energy_coordinator"] = energy_coordinator
        await energy_coordinator.async_config_entry_first_refresh()
    else:
        energy_coordinator.update_addresses(inventory)

    new_entities: list[SensorEntity] = []
    for node_type, _node, addr_str, base_name in heater_details.iter_metadata():
        energy_unique_id = build_heater_energy_unique_id(dev_id, node_type, addr_str)
        uid_prefix = energy_unique_id.rsplit(":", 1)[0]
        new_entities.extend(
            _create_heater_sensors(
                coordinator,
                energy_coordinator,
                entry.entry_id,
                dev_id,
                addr_str,
                base_name,
                uid_prefix,
                energy_unique_id,
                node_type=node_type,
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

    for addr_str, base_name in sorted(power_monitor_entries):
        energy_unique_id = build_power_monitor_energy_unique_id(dev_id, addr_str)
        power_unique_id = build_power_monitor_power_unique_id(dev_id, addr_str)
        new_entities.append(
            PowerMonitorEnergySensor(
                energy_coordinator,
                entry.entry_id,
                dev_id,
                addr_str,
                f"{base_name} Energy",
                energy_unique_id,
                base_name,
            )
        )
        new_entities.append(
            PowerMonitorPowerSensor(
                energy_coordinator,
                entry.entry_id,
                dev_id,
                addr_str,
                f"{base_name} Power",
                power_unique_id,
                base_name,
            )
        )

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
            "Total Energy",
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
        inventory: Inventory | None = None,
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
            inventory=inventory,
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
        inventory: Inventory | None = None,
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
            inventory=inventory,
        )

    def _device_available(self, device_entry: dict[str, Any] | None) -> bool:
        """Return True when the heater has a device entry."""
        return isinstance(device_entry, dict)

    def _metric_section(self) -> dict[str, Any]:
        """Return the dictionary with the requested metric values."""
        heater_section = self._heater_section()
        metric = heater_section.get(self._metric_key)
        if not isinstance(metric, Mapping):
            device_entry = self._device_record()
            if isinstance(device_entry, Mapping):
                node_section = device_entry.get(self._node_type)
                if isinstance(node_section, Mapping):
                    metric = node_section.get(self._metric_key)
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


class HeaterBoostMinutesRemainingSensor(HeaterNodeBase, SensorEntity):
    """Sensor exposing the remaining minutes for the active boost."""

    _attr_device_class = getattr(SensorDeviceClass, "DURATION", "duration")
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = UnitOfTime.MINUTES

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
    uid_prefix: str,
    energy_unique_id: str,
    *,
    node_type: str | None = None,
    inventory: Inventory | None = None,
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
        inventory=inventory,
    )
    energy = energy_cls(
        energy_coordinator,
        entry_id,
        dev_id,
        addr,
        f"{base_name} Energy",
        energy_unique_id,
        base_name,
        node_type=node_type,
        inventory=inventory,
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
        inventory=inventory,
    )

    return (temperature, energy, power)


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
    minutes_cls: type[HeaterBoostMinutesRemainingSensor] = HeaterBoostMinutesRemainingSensor,
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
        f"{base_name} Boost Minutes Remaining",
        f"{boost_prefix}:minutes_remaining",
        device_name=base_name,
        node_type=node_type,
        inventory=inventory,
    )
    end = end_cls(
        coordinator,
        entry_id,
        dev_id,
        addr,
        f"{base_name} Boost End",
        f"{boost_prefix}:end",
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
        name: str,
        unique_id: str,
        device_name: str,
    ) -> None:
        """Initialise the power monitor sensor base entity."""

        super().__init__(coordinator)
        self._entry_id = entry_id
        self._dev_id = dev_id
        normalized_addr = normalize_node_addr(addr, use_default_when_falsey=True)
        self._addr = normalized_addr or str(addr)
        self._attr_name = name
        self._attr_unique_id = unique_id
        self._device_name = device_name

    def _device_record(self) -> Mapping[str, Any] | None:
        """Return the cached coordinator data for this device."""

        data = getattr(self.coordinator, "data", None)
        if not isinstance(data, Mapping):
            return None
        record = data.get(self._dev_id)
        return record if isinstance(record, Mapping) else None

    def _metric_bucket(self) -> Mapping[str, Any]:
        """Return the metric bucket for this power monitor."""

        record = self._device_record()
        if not isinstance(record, Mapping):
            return {}

        direct_bucket = record.get("pmo")
        if isinstance(direct_bucket, Mapping):
            direct_metric = direct_bucket.get(self._metric_key)
            if isinstance(direct_metric, Mapping):
                return direct_metric

        return {}

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

        bucket = self._metric_bucket()
        if self._addr in bucket:
            return True
        addresses: Iterable[str] = ()
        accessor = getattr(self.coordinator, "addresses_for_type", None)
        if callable(accessor):
            try:
                addresses = accessor("pmo")
            except Exception:  # pragma: no cover - defensive safeguard
                addresses = ()
        if not addresses:
            address_map = getattr(self.coordinator, "_addresses_by_type", None)
            if isinstance(address_map, Mapping):
                addresses = address_map.get("pmo", ())
        return self._addr in set(addresses)

    @property
    def native_value(self) -> float | None:
        """Return the processed metric value for Home Assistant."""

        bucket = self._metric_bucket()
        raw = bucket.get(self._addr)
        if raw is None:
            return None
        return self._coerce_native_value(raw)

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
    _attr_state_class = SensorStateClass.TOTAL_INCREASING
    _attr_native_unit_of_measurement = "kWh"
    _metric_key = "energy"

    def _coerce_native_value(self, raw: Any) -> float | None:  # type: ignore[override]
        """Normalise the raw energy metric into kWh."""

        return _normalise_energy_value(self.coordinator, raw)


class PowerMonitorPowerSensor(PowerMonitorSensorBase):
    """Power sensor for a power monitor."""

    _attr_device_class = SensorDeviceClass.POWER
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = "W"
    _metric_key = "power"
class InstallationTotalEnergySensor(
    GatewayDispatcherEntity, CoordinatorEntity, SensorEntity
):
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
        details: HeaterPlatformDetails,
    ) -> None:
        """Initialise the installation-wide energy sensor."""
        super().__init__(coordinator)
        self._entry_id = entry_id
        self._dev_id = dev_id
        self._attr_name = name
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
        d = (self.coordinator.data or {}).get(self._dev_id)
        return d is not None

    @property
    def native_value(self) -> float | None:
        """Return the summed energy usage across all heaters."""
        data = (self.coordinator.data or {}).get(self._dev_id)
        if not isinstance(data, Mapping):
            return None
        total = 0.0
        found = False
        for node_type in self._details.addrs_by_type:
            section = data.get(node_type)
            if not isinstance(section, Mapping):
                continue
            energy_map = section.get("energy")
            if not isinstance(energy_map, Mapping):
                continue
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
