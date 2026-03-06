from collections.abc import Mapping
from typing import Any

import pytest

from conftest import FakeCoordinator, _install_stubs, build_coordinator_device_state

_install_stubs()

from custom_components.termoweb.inventory import Inventory, build_node_inventory
from custom_components.termoweb.sensor import (
    AccumulatorChargingSensor,
    AccumulatorCurrentChargeSensor,
    AccumulatorTargetChargeSensor,
    ThermostatBatterySensor,
)
from homeassistant.core import HomeAssistant


@pytest.fixture
def thermostat_inventory(
    inventory_builder,
) -> Inventory:
    """Return helper inventory for thermostat sensor tests."""

    dev_id = "dev-thm"
    raw_nodes = {"nodes": [{"type": "thm", "addr": "T1"}]}
    node_list = build_node_inventory(raw_nodes)
    return inventory_builder(dev_id, raw_nodes, node_list)


@pytest.fixture
def accumulator_inventory(
    inventory_builder,
) -> Inventory:
    """Return helper inventory for accumulator sensor tests."""

    dev_id = "dev-acm"
    raw_nodes = {"nodes": [{"type": "acm", "addr": "A1"}]}
    node_list = build_node_inventory(raw_nodes)
    return inventory_builder(dev_id, raw_nodes, node_list)


def _make_coordinator(
    inventory: Inventory,
    *,
    payload: Mapping[str, Mapping[str, Any]] | None = None,
) -> FakeCoordinator:
    """Construct a fake coordinator with thermostat settings."""

    hass = HomeAssistant()
    dev_id = "dev-thm"
    nodes = {"nodes": [{"type": "thm", "addr": "T1"}]}
    settings = payload or {"thm": {"T1": {}}}
    record = build_coordinator_device_state(nodes=nodes, settings=settings)
    return FakeCoordinator(
        hass,
        dev_id=dev_id,
        dev=record,
        nodes=nodes,
        inventory=inventory,
        data={dev_id: record},
    )


def _make_accumulator_coordinator(
    inventory: Inventory,
    *,
    payload: Mapping[str, Mapping[str, Any]] | None = None,
) -> FakeCoordinator:
    """Construct a fake coordinator with accumulator settings."""

    hass = HomeAssistant()
    dev_id = "dev-acm"
    nodes = {"nodes": [{"type": "acm", "addr": "A1"}]}
    settings = payload or {"acm": {"A1": {}}}
    record = build_coordinator_device_state(nodes=nodes, settings=settings)
    return FakeCoordinator(
        hass,
        dev_id=dev_id,
        dev=record,
        nodes=nodes,
        inventory=inventory,
        data={dev_id: record},
    )


def test_thermostat_battery_sensor_reports_percentage(
    thermostat_inventory: Inventory,
) -> None:
    coordinator = _make_coordinator(
        thermostat_inventory,
        payload={"thm": {"T1": {"batt_level": 4}}},
    )

    sensor = ThermostatBatterySensor(
        coordinator,
        "entry-thm",
        "dev-thm",
        "T1",
        unique_id="uid-thm",
        device_name="Thermostat T1",
        node_type="thm",
        inventory=thermostat_inventory,
    )

    assert sensor.native_value == 80
    assert sensor.extra_state_attributes["batt_level_steps"] == 4


def test_thermostat_battery_sensor_handles_invalid_level(
    thermostat_inventory: Inventory,
) -> None:
    coordinator = _make_coordinator(
        thermostat_inventory,
        payload={"thm": {"T1": {"batt_level": "invalid"}}},
    )

    sensor = ThermostatBatterySensor(
        coordinator,
        "entry-thm",
        "dev-thm",
        "T1",
        unique_id="uid-invalid",
        device_name="Thermostat T1",
        node_type="thm",
        inventory=thermostat_inventory,
    )

    assert sensor.native_value is None
    assert sensor.extra_state_attributes["batt_level_steps"] is None


def test_accumulator_charging_sensor_reports_boolean(
    accumulator_inventory: Inventory,
) -> None:
    coordinator = _make_accumulator_coordinator(
        accumulator_inventory,
        payload={"acm": {"A1": {"charging": True}}},
    )

    sensor = AccumulatorChargingSensor(
        coordinator,
        "entry-acm",
        "dev-acm",
        "A1",
        None,
        "uid-acm-charging",
        device_name="Accumulator A1",
        node_type="acm",
        inventory=accumulator_inventory,
    )

    assert sensor.native_value is True
    assert sensor.extra_state_attributes == {"dev_id": "dev-acm", "addr": "A1"}


def test_accumulator_charging_sensor_handles_invalid_state(
    accumulator_inventory: Inventory,
) -> None:
    coordinator = _make_accumulator_coordinator(
        accumulator_inventory,
        payload={"acm": {"A1": {"charging": "unexpected"}}},
    )

    sensor = AccumulatorChargingSensor(
        coordinator,
        "entry-acm",
        "dev-acm",
        "A1",
        None,
        "uid-acm-charging-invalid",
        device_name="Accumulator A1",
        node_type="acm",
        inventory=accumulator_inventory,
    )

    assert sensor.native_value is None


def test_accumulator_charge_percentage_sensors_clamp_values(
    accumulator_inventory: Inventory,
) -> None:
    coordinator = _make_accumulator_coordinator(
        accumulator_inventory,
        payload={
            "acm": {
                "A1": {
                    "current_charge_per": 150.5,
                    "target_charge_per": -12,
                }
            }
        },
    )

    current_sensor = AccumulatorCurrentChargeSensor(
        coordinator,
        "entry-acm",
        "dev-acm",
        "A1",
        None,
        "uid-acm-current",
        device_name="Accumulator A1",
        node_type="acm",
        inventory=accumulator_inventory,
    )
    target_sensor = AccumulatorTargetChargeSensor(
        coordinator,
        "entry-acm",
        "dev-acm",
        "A1",
        None,
        "uid-acm-target",
        device_name="Accumulator A1",
        node_type="acm",
        inventory=accumulator_inventory,
    )

    assert current_sensor.native_value == 100
    assert target_sensor.native_value == 0


def test_accumulator_charge_percentage_handles_missing_values(
    accumulator_inventory: Inventory,
) -> None:
    coordinator = _make_accumulator_coordinator(
        accumulator_inventory,
        payload={"acm": {"A1": {"current_charge_per": None}}},
    )

    current_sensor = AccumulatorCurrentChargeSensor(
        coordinator,
        "entry-acm",
        "dev-acm",
        "A1",
        None,
        "uid-acm-current-missing",
        device_name="Accumulator A1",
        node_type="acm",
        inventory=accumulator_inventory,
    )

    assert current_sensor.native_value is None


# ---------------------------------------------------------------------------
# Coverage expansion: _looks_like_integer_string
# ---------------------------------------------------------------------------

from custom_components.termoweb.entities.sensor import (
    _looks_like_integer_string,
    _normalise_energy_value,
    _power_monitor_display_name,
    _create_heater_sensors,
    _create_boost_sensors,
)
from custom_components.termoweb.sensor import (
    HeaterTemperatureSensor,
    HeaterEnergyTotalSensor,
    HeaterPowerSensor,
    HeaterBoostMinutesRemainingSensor,
    HeaterBoostEndSensor,
    PowerMonitorEnergySensor,
    PowerMonitorPowerSensor,
    InstallationTotalEnergySensor,
    InstallationInfoSensor,
)
from custom_components.termoweb.inventory import (
    Inventory,
    build_node_inventory,
    PowerMonitorNode,
)
from custom_components.termoweb.domain.ids import NodeId, NodeType
from custom_components.termoweb.domain.state import DomainStateStore
from custom_components.termoweb.domain.energy import EnergyNodeMetrics, EnergySnapshot
from custom_components.termoweb.domain.view import DomainStateView
from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.entities.heater import HeaterPlatformDetails
from types import SimpleNamespace
import math
import importlib


def test_looks_like_integer_string_empty():
    """Empty or whitespace-only strings should return False."""
    assert _looks_like_integer_string("") is False
    assert _looks_like_integer_string("   ") is False


def test_looks_like_integer_string_with_sign_prefix():
    """Signed integer strings should be recognized."""
    assert _looks_like_integer_string("+42") is True
    assert _looks_like_integer_string("-7") is True
    assert _looks_like_integer_string("+ ") is False
    assert _looks_like_integer_string("-") is False


def test_looks_like_integer_string_plain_digit():
    """Plain digit strings are integers."""
    assert _looks_like_integer_string("100") is True
    assert _looks_like_integer_string(" 99 ") is True


def test_looks_like_integer_string_non_digit():
    """Non-digit strings should return False."""
    assert _looks_like_integer_string("12.5") is False
    assert _looks_like_integer_string("abc") is False


# ---------------------------------------------------------------------------
# Coverage expansion: _normalise_energy_value edge cases
# ---------------------------------------------------------------------------


def test_normalise_energy_value_infinity_returns_none():
    """Infinity should be rejected."""
    assert _normalise_energy_value(object(), float("inf")) is None
    assert _normalise_energy_value(object(), float("-inf")) is None
    assert _normalise_energy_value(object(), float("nan")) is None


def test_normalise_energy_value_string_scale_parse_failure():
    """An unparseable string scale attribute with a non-standard value."""
    coordinator = SimpleNamespace(_termoweb_energy_scale="garbage-value")
    # Falls through to heuristic: float value -> scale=1.0
    result = _normalise_energy_value(coordinator, 5.5)
    assert result == 5.5


def test_normalise_energy_value_string_scale_numeric_parse():
    """A string scale that is parseable but not a known keyword."""
    coordinator = SimpleNamespace(_termoweb_energy_scale="0")
    # parsed=0, which is not > 0, so falls through to heuristic
    result = _normalise_energy_value(coordinator, 100)
    # int raw -> _looks_like_integer_string won't be called; but isinstance(raw, int)
    # -> scale = _WH_TO_KWH (0.001)
    assert result == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Coverage expansion: _power_monitor_display_name
# ---------------------------------------------------------------------------


def test_power_monitor_display_name_uses_node_name():
    """Display name should come from node.name when available."""
    node = SimpleNamespace(name="  Kitchen Monitor  ")
    result = _power_monitor_display_name(node, "01", None)
    assert result == "Kitchen Monitor"


def test_power_monitor_display_name_uses_default_name():
    """Display name should fall back to default_name factory."""
    node = SimpleNamespace(name="", default_name=lambda: "  Default Monitor  ")
    result = _power_monitor_display_name(node, "01", None)
    assert result == "Default Monitor"


def test_power_monitor_display_name_fallback():
    """Display name should use format_fallback when no name is available."""
    node = SimpleNamespace(name=None)
    result = _power_monitor_display_name(node, "01", None)
    assert "01" in result


def test_power_monitor_display_name_default_name_empty():
    """Display name should fall back when default_name returns empty."""
    node = SimpleNamespace(name="", default_name=lambda: "")
    result = _power_monitor_display_name(node, "42", None)
    assert "42" in result


# ---------------------------------------------------------------------------
# Coverage expansion: HeaterTemperatureSensor
# ---------------------------------------------------------------------------


def test_heater_temperature_sensor_native_value(thermostat_inventory):
    """Temperature sensor should return mtemp from heater state."""
    coordinator = _make_coordinator(
        thermostat_inventory,
        payload={"thm": {"T1": {"mtemp": 22.5}}},
    )

    sensor = HeaterTemperatureSensor(
        coordinator,
        "entry-thm",
        "dev-thm",
        "T1",
        "uid-temp",
        device_name="Thermostat T1",
        node_type="thm",
        inventory=thermostat_inventory,
    )

    assert sensor.native_value == 22.5
    attrs = sensor.extra_state_attributes
    assert attrs["dev_id"] == "dev-thm"
    assert attrs["addr"] == "T1"


def test_heater_temperature_sensor_missing_mtemp(thermostat_inventory):
    """Temperature sensor should return None when mtemp is missing."""
    coordinator = _make_coordinator(
        thermostat_inventory,
        payload={"thm": {"T1": {}}},
    )

    sensor = HeaterTemperatureSensor(
        coordinator,
        "entry-thm",
        "dev-thm",
        "T1",
        "uid-temp-none",
        device_name="Thermostat T1",
        node_type="thm",
        inventory=thermostat_inventory,
    )

    assert sensor.native_value is None


# ---------------------------------------------------------------------------
# Coverage expansion: ThermostatBatterySensor._coerce_level edge cases
# ---------------------------------------------------------------------------


def test_thermostat_battery_coerce_level_bool():
    """Boolean True should coerce to 1, giving 20%."""
    coordinator = _make_coordinator(
        Inventory("dev-thm", build_node_inventory({"nodes": [{"type": "thm", "addr": "T1"}]})),
        payload={"thm": {"T1": {"batt_level": True}}},
    )
    sensor = ThermostatBatterySensor(
        coordinator,
        "entry-thm",
        "dev-thm",
        "T1",
        unique_id="uid-bool",
        device_name="Thermostat T1",
        node_type="thm",
        inventory=coordinator.inventory,
    )
    assert sensor.native_value == 20  # True -> 1 -> 1*20


def test_thermostat_battery_coerce_level_string_numeric():
    """String numeric values that fail initial float() should try str().strip()."""
    # We test with a stringifiable non-string type that fails float()
    assert ThermostatBatterySensor._coerce_level(float("nan")) is None


def test_thermostat_battery_coerce_level_string_fallback():
    """Coerce level should try str(value).strip() as fallback."""
    # An object whose str representation is a number
    class Tricky:
        def __float__(self):
            raise TypeError("nope")
        def __str__(self):
            return " 3 "

    assert ThermostatBatterySensor._coerce_level(Tricky()) == 3


def test_thermostat_battery_coerce_level_string_fallback_fails():
    """Coerce level should return None when all conversions fail."""
    class Unconvertible:
        def __float__(self):
            raise TypeError("nope")
        def __str__(self):
            return "not-a-number"

    assert ThermostatBatterySensor._coerce_level(Unconvertible()) is None


def test_thermostat_battery_sensor_no_device_name():
    """Battery sensor should use fallback name when device_name is empty."""
    inv = Inventory("dev-thm", build_node_inventory({"nodes": [{"type": "thm", "addr": "T1"}]}))
    coordinator = _make_coordinator(inv, payload={"thm": {"T1": {"batt_level": 3}}})
    sensor = ThermostatBatterySensor(
        coordinator,
        "entry-thm",
        "dev-thm",
        "T1",
        unique_id="uid-no-name",
        device_name="",
        node_type="thm",
        inventory=inv,
    )
    assert sensor._attr_name == "Thermostat Battery"


# ---------------------------------------------------------------------------
# Coverage expansion: AccumulatorChargingSensor coerce edge cases
# ---------------------------------------------------------------------------


def test_accumulator_charging_sensor_coerce_int_value(accumulator_inventory):
    """Integer non-zero value should be coerced to True."""
    coordinator = _make_accumulator_coordinator(
        accumulator_inventory,
        payload={"acm": {"A1": {"charging": 1}}},
    )
    sensor = AccumulatorChargingSensor(
        coordinator, "entry-acm", "dev-acm", "A1", None,
        "uid-charging-int", device_name="Acc", node_type="acm",
        inventory=accumulator_inventory,
    )
    assert sensor.native_value is True


def test_accumulator_charging_sensor_coerce_zero(accumulator_inventory):
    """Zero int should be coerced to False."""
    coordinator = _make_accumulator_coordinator(
        accumulator_inventory,
        payload={"acm": {"A1": {"charging": 0}}},
    )
    sensor = AccumulatorChargingSensor(
        coordinator, "entry-acm", "dev-acm", "A1", None,
        "uid-charging-zero", device_name="Acc", node_type="acm",
        inventory=accumulator_inventory,
    )
    assert sensor.native_value is False


def test_accumulator_charging_sensor_coerce_none(accumulator_inventory):
    """None value should return None."""
    coordinator = _make_accumulator_coordinator(
        accumulator_inventory,
        payload={"acm": {"A1": {"charging": None}}},
    )
    sensor = AccumulatorChargingSensor(
        coordinator, "entry-acm", "dev-acm", "A1", None,
        "uid-charging-none", device_name="Acc", node_type="acm",
        inventory=accumulator_inventory,
    )
    assert sensor.native_value is None


# ---------------------------------------------------------------------------
# Coverage expansion: HeaterEnergyBase methods
# ---------------------------------------------------------------------------


def test_heater_energy_base_device_available():
    """HeaterEnergyBase should check inventory and coordinator."""
    energy_module = importlib.import_module("custom_components.termoweb.domain.energy")
    ids_module = importlib.import_module("custom_components.termoweb.domain.ids")
    state_module = importlib.import_module("custom_components.termoweb.domain.state")
    view_module = importlib.import_module("custom_components.termoweb.domain.view")

    dev_id = "dev-e"
    raw_nodes = {"nodes": [{"type": "htr", "addr": "01"}]}
    inventory = Inventory(dev_id, build_node_inventory(raw_nodes))

    coordinator = SimpleNamespace(
        last_update_success=True,
        inventory=inventory,
        data={},
    )

    node_id = ids_module.NodeId(ids_module.NodeType.HEATER, "01")
    store = state_module.DomainStateStore([node_id])
    view = view_module.DomainStateView(dev_id, store)

    sensor = HeaterEnergyTotalSensor(
        coordinator,
        view,
        "entry-1",
        dev_id,
        "01",
        "uid-energy-1",
        device_name="Heater",
        node_type="htr",
        inventory=inventory,
    )
    assert sensor.available is True

    # When coordinator reports failure, should be unavailable
    coordinator.last_update_success = False
    assert sensor.available is False


def test_heater_energy_base_raw_native_value_energy():
    """HeaterEnergyBase should read energy_kwh from domain view."""
    sensor_module = importlib.import_module("custom_components.termoweb.sensor")
    energy_module = importlib.import_module("custom_components.termoweb.domain.energy")
    ids_module = importlib.import_module("custom_components.termoweb.domain.ids")
    state_module = importlib.import_module("custom_components.termoweb.domain.state")
    view_module = importlib.import_module("custom_components.termoweb.domain.view")

    dev_id = "dev-e"
    node_id = ids_module.NodeId(ids_module.NodeType.HEATER, "01")
    store = state_module.DomainStateStore([node_id])
    snapshot = energy_module.EnergySnapshot(
        dev_id=dev_id,
        metrics={
            node_id: energy_module.EnergyNodeMetrics(
                energy_kwh=3.5, power_w=150.0, source="test", ts=0.0,
            )
        },
        updated_at=0.0,
        ws_deadline=None,
    )
    store.set_energy_snapshot(snapshot)
    view = view_module.DomainStateView(dev_id, store)

    inventory = Inventory(dev_id, build_node_inventory({"nodes": [{"type": "htr", "addr": "01"}]}))
    coordinator = SimpleNamespace(last_update_success=True, inventory=inventory)

    energy_sensor = sensor_module.HeaterEnergyTotalSensor(
        coordinator, view, "entry-1", dev_id, "01",
        "uid-energy", device_name="Heater", node_type="htr", inventory=inventory,
    )
    assert energy_sensor.native_value == 3.5

    power_sensor = sensor_module.HeaterPowerSensor(
        coordinator, view, "entry-1", dev_id, "01",
        "uid-power", device_name="Heater", node_type="htr", inventory=inventory,
    )
    assert power_sensor.native_value == 150.0


def test_heater_energy_base_no_domain_view():
    """HeaterEnergyBase should return None without a domain view."""
    inventory = Inventory("dev-e", build_node_inventory({"nodes": [{"type": "htr", "addr": "01"}]}))
    coordinator = SimpleNamespace(last_update_success=True, inventory=inventory)

    sensor = HeaterEnergyTotalSensor(
        coordinator, None, "entry-1", "dev-e", "01",
        "uid-energy-none", device_name="Heater", node_type="htr", inventory=inventory,
    )
    assert sensor.native_value is None
    assert sensor.extra_state_attributes == {"dev_id": "dev-e", "addr": "01"}


def test_heater_energy_base_coerce_invalid():
    """_coerce_native_value should return None for non-numeric values."""
    inventory = Inventory("dev-e", build_node_inventory({"nodes": [{"type": "htr", "addr": "01"}]}))
    coordinator = SimpleNamespace(last_update_success=True, inventory=inventory)

    sensor = HeaterPowerSensor(
        coordinator, None, "entry-1", "dev-e", "01",
        "uid-coerce", device_name="Heater", node_type="htr", inventory=inventory,
    )
    assert sensor._coerce_native_value("not-a-number") is None
    assert sensor._coerce_native_value(None) is None


# ---------------------------------------------------------------------------
# Coverage expansion: _create_heater_sensors for accumulator type
# ---------------------------------------------------------------------------


def test_create_heater_sensors_accumulator_type():
    """Accumulator type should create extra charging sensors."""
    dev_id = "dev-acm"
    inventory = Inventory(dev_id, build_node_inventory({"nodes": [{"type": "acm", "addr": "A1"}]}))
    coordinator = SimpleNamespace(data={}, inventory=inventory)
    energy_coordinator = SimpleNamespace(data={}, inventory=inventory, last_update_success=True)

    sensors = _create_heater_sensors(
        coordinator, energy_coordinator, None,
        "entry-1", dev_id, "A1", "Accumulator A1",
        node_type="acm", inventory=inventory,
    )

    # For acm: temp + charging + current_charge + target_charge + energy + power = 6
    assert len(sensors) == 6
    type_names = [type(s).__name__ for s in sensors]
    assert "AccumulatorChargingSensor" in type_names
    assert "AccumulatorCurrentChargeSensor" in type_names
    assert "AccumulatorTargetChargeSensor" in type_names


def test_create_heater_sensors_thermostat_no_energy():
    """Thermostat type should not get energy or power sensors."""
    dev_id = "dev-thm"
    inventory = Inventory(dev_id, build_node_inventory({"nodes": [{"type": "thm", "addr": "T1"}]}))
    coordinator = SimpleNamespace(data={}, inventory=inventory)
    energy_coordinator = SimpleNamespace(data={}, inventory=inventory, last_update_success=True)

    sensors = _create_heater_sensors(
        coordinator, energy_coordinator, None,
        "entry-1", dev_id, "T1", "Thermostat T1",
        node_type="thm", inventory=inventory,
    )

    # For thm: only temp sensor (no energy/power)
    assert len(sensors) == 1
    assert type(sensors[0]).__name__ == "HeaterTemperatureSensor"


# ---------------------------------------------------------------------------
# Coverage expansion: _create_boost_sensors
# ---------------------------------------------------------------------------


def test_create_boost_sensors():
    """Boost sensor factory should create minutes and end sensors."""
    inventory = Inventory("dev", build_node_inventory({"nodes": [{"type": "acm", "addr": "1"}]}))
    coordinator = SimpleNamespace(data={}, inventory=inventory)

    minutes, end = _create_boost_sensors(
        coordinator, "entry-1", "dev", "1", "Accumulator",
        f"{DOMAIN}:dev:acm:1:energy",
        node_type="acm", inventory=inventory,
    )

    assert isinstance(minutes, HeaterBoostMinutesRemainingSensor)
    assert isinstance(end, HeaterBoostEndSensor)
    assert "boost:minutes_remaining" in minutes._attr_unique_id
    assert "boost:end" in end._attr_unique_id


# ---------------------------------------------------------------------------
# Coverage expansion: PowerMonitorSensorBase
# ---------------------------------------------------------------------------


def test_power_monitor_sensor_base_should_poll():
    """Power monitor sensors should not poll."""
    inventory = Inventory("dev", [PowerMonitorNode(name="Mon", addr="01")])
    coordinator = SimpleNamespace(data={})
    sensor = PowerMonitorPowerSensor(
        coordinator, "entry-1", "dev", "01", "uid-1",
        device_name="Monitor", inventory=inventory,
    )
    assert sensor.should_poll is False


def test_power_monitor_sensor_base_extra_state_attributes():
    """Power monitor extra attrs should expose dev_id and addr."""
    inventory = Inventory("dev", [PowerMonitorNode(name="Mon", addr="01")])
    coordinator = SimpleNamespace(data={})
    sensor = PowerMonitorPowerSensor(
        coordinator, "entry-1", "dev", "01", "uid-1",
        device_name="Monitor", inventory=inventory,
    )
    attrs = sensor.extra_state_attributes
    assert attrs == {"dev_id": "dev", "addr": "01"}


def test_power_monitor_sensor_base_native_value_none_without_metrics():
    """Power monitor should return None when no metrics are available."""
    inventory = Inventory("dev", [PowerMonitorNode(name="Mon", addr="01")])
    coordinator = SimpleNamespace(data={})
    sensor = PowerMonitorPowerSensor(
        coordinator, "entry-1", "dev", "01", "uid-1",
        device_name="Monitor", inventory=inventory,
    )
    assert sensor.native_value is None


def test_power_monitor_sensor_coerce_native_value_failure():
    """Power monitor _coerce_native_value should return None for bad input."""
    inventory = Inventory("dev", [PowerMonitorNode(name="Mon", addr="01")])
    coordinator = SimpleNamespace(data={})
    sensor = PowerMonitorPowerSensor(
        coordinator, "entry-1", "dev", "01", "uid-1",
        device_name="Monitor", inventory=inventory,
    )
    assert sensor._coerce_native_value("not-a-number") is None
    assert sensor._coerce_native_value(None) is None


def test_power_monitor_resolve_inventory_from_coordinator():
    """Resolve inventory should fall back to coordinator inventory."""
    inventory = Inventory("dev", [PowerMonitorNode(name="Mon", addr="01")])
    coordinator = SimpleNamespace(data={}, inventory=inventory)
    sensor = PowerMonitorPowerSensor(
        coordinator, "entry-1", "dev", "01", "uid-1",
        device_name="Monitor", inventory=inventory,
    )
    # Clear the direct inventory reference to test fallback
    sensor._inventory = None
    resolved = sensor._resolve_inventory()
    assert resolved is inventory


def test_power_monitor_resolve_inventory_none():
    """Resolve inventory should return None when nothing is available."""
    coordinator = SimpleNamespace(data={}, inventory=None)
    sensor = PowerMonitorPowerSensor(
        coordinator, "entry-1", "dev", "01", "uid-1",
        device_name="Monitor", inventory=Inventory("dev", []),
    )
    sensor._inventory = None
    resolved = sensor._resolve_inventory()
    assert resolved is None


def test_power_monitor_available_no_inventory():
    """Power monitor should be unavailable when no metrics and no inventory."""
    coordinator = SimpleNamespace(data={}, inventory=None)
    sensor = PowerMonitorPowerSensor(
        coordinator, "entry-1", "dev", "01", "uid-1",
        device_name="Monitor", inventory=Inventory("dev", []),
    )
    sensor._inventory = None
    assert sensor.available is False


def test_power_monitor_energy_sensor_uses_normalise():
    """Energy sensor should use _normalise_energy_value for coercion."""
    sensor_module = importlib.import_module("custom_components.termoweb.sensor")
    ids_module = importlib.import_module("custom_components.termoweb.domain.ids")
    energy_module = importlib.import_module("custom_components.termoweb.domain.energy")
    state_module = importlib.import_module("custom_components.termoweb.domain.state")
    view_module = importlib.import_module("custom_components.termoweb.domain.view")

    dev_id = "dev-pmo"
    node_id = ids_module.NodeId(ids_module.NodeType.POWER_MONITOR, "01")
    store = state_module.DomainStateStore([node_id])
    snapshot = energy_module.EnergySnapshot(
        dev_id=dev_id,
        metrics={
            node_id: energy_module.EnergyNodeMetrics(
                energy_kwh=2.0, power_w=100.0, source="test", ts=0.0,
            )
        },
        updated_at=0.0,
        ws_deadline=None,
    )
    store.set_energy_snapshot(snapshot)
    view = view_module.DomainStateView(dev_id, store)
    inventory = Inventory(dev_id, [PowerMonitorNode(name="Mon", addr="01")])
    coordinator = SimpleNamespace(data={})

    sensor = sensor_module.PowerMonitorEnergySensor(
        coordinator, "entry-1", dev_id, "01", "uid-energy",
        device_name="Monitor", inventory=inventory, domain_view=view,
    )
    assert sensor.native_value == 2.0


# ---------------------------------------------------------------------------
# Coverage expansion: InstallationTotalEnergySensor
# ---------------------------------------------------------------------------


def test_installation_total_energy_sensor_sums_metrics():
    """Installation total should sum energy across all heater types."""
    sensor_module = importlib.import_module("custom_components.termoweb.sensor")
    heater_entities = importlib.import_module("custom_components.termoweb.entities.heater")
    ids_module = importlib.import_module("custom_components.termoweb.domain.ids")
    energy_module = importlib.import_module("custom_components.termoweb.domain.energy")
    state_module = importlib.import_module("custom_components.termoweb.domain.state")
    view_module = importlib.import_module("custom_components.termoweb.domain.view")

    dev_id = "dev-total"
    node1 = ids_module.NodeId(ids_module.NodeType.HEATER, "01")
    node2 = ids_module.NodeId(ids_module.NodeType.HEATER, "02")
    store = state_module.DomainStateStore([node1, node2])
    snapshot = energy_module.EnergySnapshot(
        dev_id=dev_id,
        metrics={
            node1: energy_module.EnergyNodeMetrics(
                energy_kwh=1.5, power_w=50.0, source="test", ts=0.0,
            ),
            node2: energy_module.EnergyNodeMetrics(
                energy_kwh=2.5, power_w=100.0, source="test", ts=0.0,
            ),
        },
        updated_at=0.0,
        ws_deadline=None,
    )
    store.set_energy_snapshot(snapshot)
    view = view_module.DomainStateView(dev_id, store)

    raw_nodes = {"nodes": [
        {"type": "htr", "addr": "01"},
        {"type": "htr", "addr": "02"},
    ]}
    inventory = Inventory(dev_id, build_node_inventory(raw_nodes))
    details = heater_entities.HeaterPlatformDetails(
        inventory=inventory,
        default_name_simple=lambda addr: f"Heater {addr}",
    )

    coordinator = SimpleNamespace(data={}, inventory=inventory)

    sensor = sensor_module.InstallationTotalEnergySensor(
        coordinator, "entry-1", dev_id, "uid-total", details, view,
    )

    assert sensor.available is True
    assert sensor.native_value == pytest.approx(4.0)


def test_installation_total_energy_sensor_no_view():
    """Installation total should be unavailable without a domain view."""
    inventory = Inventory("dev", build_node_inventory({"nodes": [{"type": "htr", "addr": "01"}]}))
    details = HeaterPlatformDetails(
        inventory=inventory,
        default_name_simple=lambda addr: f"Heater {addr}",
    )
    coordinator = SimpleNamespace(data={})

    sensor = InstallationTotalEnergySensor(
        coordinator, "entry-1", "dev", "uid-total-none", details, None,
    )
    assert sensor.available is False
    assert sensor.native_value is None


def test_installation_total_energy_sensor_no_metrics():
    """Installation total should return None when no metrics match."""
    sensor_module = importlib.import_module("custom_components.termoweb.sensor")
    heater_entities = importlib.import_module("custom_components.termoweb.entities.heater")
    ids_module = importlib.import_module("custom_components.termoweb.domain.ids")
    state_module = importlib.import_module("custom_components.termoweb.domain.state")
    view_module = importlib.import_module("custom_components.termoweb.domain.view")
    energy_module = importlib.import_module("custom_components.termoweb.domain.energy")

    dev_id = "dev-empty"
    # Create store with a node but empty energy snapshot with no metrics
    node1 = ids_module.NodeId(ids_module.NodeType.HEATER, "01")
    store = state_module.DomainStateStore([node1])
    snapshot = energy_module.EnergySnapshot(
        dev_id=dev_id, metrics={}, updated_at=0.0, ws_deadline=None,
    )
    store.set_energy_snapshot(snapshot)
    view = view_module.DomainStateView(dev_id, store)

    inventory = Inventory(dev_id, build_node_inventory({"nodes": [{"type": "htr", "addr": "01"}]}))
    details = heater_entities.HeaterPlatformDetails(
        inventory=inventory,
        default_name_simple=lambda addr: f"Heater {addr}",
    )
    coordinator = SimpleNamespace(data={})

    sensor = sensor_module.InstallationTotalEnergySensor(
        coordinator, "entry-1", dev_id, "uid-no-metrics", details, view,
    )
    assert sensor.available is True
    assert sensor.native_value is None


# ---------------------------------------------------------------------------
# Coverage expansion: InstallationInfoSensor
# ---------------------------------------------------------------------------


def test_installation_info_sensor_with_geo_data():
    """InstallationInfoSensor should display geo location info."""
    from conftest import build_entry_runtime

    hass = HomeAssistant()
    entry_id = "entry-info"
    dev_id = "dev-info"

    # Build a coordinator with geo_data
    geo = SimpleNamespace(city="Oslo", state="Viken", country="Norway", tz_code="CET", zip="0001")
    device_metadata = SimpleNamespace(geo_data=geo)
    coordinator = SimpleNamespace(
        data={}, inventory=None, device_metadata=device_metadata,
    )
    build_entry_runtime(
        hass=hass, entry_id=entry_id, dev_id=dev_id,
        coordinator=coordinator,
    )

    sensor = InstallationInfoSensor(coordinator, entry_id, dev_id)
    sensor.hass = hass

    assert sensor.native_value == "Oslo, Viken, Norway"
    attrs = sensor.extra_state_attributes
    assert attrs["city"] == "Oslo"
    assert attrs["state"] == "Viken"
    assert attrs["country"] == "Norway"
    assert attrs["timezone"] == "CET"
    assert attrs["zip"] == "0001"


def test_installation_info_sensor_no_runtime():
    """InstallationInfoSensor should handle missing runtime gracefully."""
    hass = HomeAssistant()
    coordinator = SimpleNamespace(data={})

    sensor = InstallationInfoSensor(coordinator, "entry-missing", "dev-missing")
    sensor.hass = hass

    assert sensor.native_value is None
    assert sensor.extra_state_attributes == {}


def test_installation_info_sensor_no_geo_data():
    """InstallationInfoSensor should handle missing geo_data."""
    from conftest import build_entry_runtime

    hass = HomeAssistant()
    entry_id = "entry-no-geo"
    dev_id = "dev-no-geo"

    coordinator = SimpleNamespace(
        data={}, inventory=None, device_metadata=SimpleNamespace(geo_data=None),
    )
    build_entry_runtime(
        hass=hass, entry_id=entry_id, dev_id=dev_id,
        coordinator=coordinator,
    )

    sensor = InstallationInfoSensor(coordinator, entry_id, dev_id)
    sensor.hass = hass

    assert sensor.native_value is None
    assert sensor.extra_state_attributes == {}
