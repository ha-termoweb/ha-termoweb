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
