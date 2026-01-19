"""Tests for sensor energy normalisation helper."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import MagicMock

import pytest

from conftest import FakeCoordinator, build_coordinator_device_state
from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.coordinator import EnergyStateCoordinator
from custom_components.termoweb.inventory import (
    Inventory,
    PowerMonitorNode,
    build_node_inventory,
)
from custom_components.termoweb.sensor import _normalise_energy_value
from custom_components.termoweb.entities import sensor as entities_sensor_module
from homeassistant.core import HomeAssistant


class _ScaleStub(SimpleNamespace):
    """Coordinator stub that exposes a custom energy scale."""


def _energy_coordinator_factory() -> EnergyStateCoordinator:
    module = importlib.import_module("custom_components.termoweb.sensor")
    energy_cls = module._normalise_energy_value.__globals__["EnergyStateCoordinator"]
    instance = object.__new__(energy_cls)
    setattr(instance, "_termoweb_energy_scale", "kWh")
    return instance


@pytest.mark.parametrize(
    ("coordinator_factory", "raw", "expected"),
    [
        pytest.param(lambda: object(), True, None, id="bool_true"),
        pytest.param(lambda: object(), False, None, id="bool_false"),
        pytest.param(
            _energy_coordinator_factory,
            5,
            5.0,
            id="energy_coordinator_defaults_to_kwh",
        ),
        pytest.param(
            lambda: object(),
            "1200",
            1.2,
            id="integer_string_interpreted_as_wh",
        ),
        pytest.param(
            lambda: _ScaleStub(_termoweb_energy_scale=0.5),
            4,
            2.0,
            id="numeric_scale_attribute",
        ),
        pytest.param(
            lambda: _ScaleStub(_termoweb_energy_scale="2"),
            4,
            8.0,
            id="string_numeric_scale_attribute",
        ),
        pytest.param(
            lambda: _ScaleStub(_termoweb_energy_scale="wh"),
            500,
            0.5,
            id="textual_scale_wh",
        ),
        pytest.param(
            lambda: object(),
            "not-a-number",
            None,
            id="invalid_string_returns_none",
        ),
    ],
)
def test_normalise_energy_value(
    coordinator_factory: Callable[[], object], raw: object, expected: float | None
) -> None:
    """Ensure ``_normalise_energy_value`` handles diverse inputs."""

    coordinator = coordinator_factory()
    assert _normalise_energy_value(coordinator, raw) == expected


@pytest.mark.asyncio
async def test_async_setup_entry_handles_missing_power_monitors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sensor setup should tolerate inventories without power monitors."""

    module = importlib.import_module("custom_components.termoweb.sensor")
    hass = HomeAssistant()
    hass.data = {DOMAIN: {}}
    entry = SimpleNamespace(entry_id="entry-1")
    raw_nodes = [{"type": "htr", "addr": "1", "name": "Heater"}]
    node_inventory = build_node_inventory({"nodes": raw_nodes})
    inventory = Inventory("dev-1", node_inventory)
    iter_calls: list[tuple[Any, Any]] = []

    def _iter_nodes_metadata_stub(
        self,
        *,
        node_types: tuple[str, ...] | None = None,
        default_name_simple: Callable[[str], str] | None = None,
    ) -> object:
        iter_calls.append((node_types, default_name_simple))
        return iter(())

    monkeypatch.setattr(
        Inventory,
        "iter_nodes_metadata",
        _iter_nodes_metadata_stub,
    )

    heater_details = SimpleNamespace(
        inventory=inventory,
        iter_metadata=lambda: iter([("htr", object(), "1", "Heater 1")]),
    )
    monkeypatch.setattr(
        module,
        "heater_platform_details_for_entry",
        lambda _data, **_kwargs: heater_details,
    )
    monkeypatch.setattr(
        entities_sensor_module,
        "heater_platform_details_for_entry",
        lambda _data, **_kwargs: heater_details,
    )
    monkeypatch.setattr(module, "iter_boostable_heater_nodes", lambda _details: [])
    monkeypatch.setattr(
        entities_sensor_module,
        "iter_boostable_heater_nodes",
        lambda _details: [],
    )
    monkeypatch.setattr(module, "log_skipped_nodes", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        entities_sensor_module,
        "log_skipped_nodes",
        lambda *args, **kwargs: None,
    )

    def _unexpected_power_monitor(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("power monitor sensors should not be created")

    monkeypatch.setattr(module, "PowerMonitorEnergySensor", _unexpected_power_monitor)
    monkeypatch.setattr(module, "PowerMonitorPowerSensor", _unexpected_power_monitor)
    monkeypatch.setattr(
        entities_sensor_module,
        "PowerMonitorEnergySensor",
        _unexpected_power_monitor,
    )
    monkeypatch.setattr(
        entities_sensor_module,
        "PowerMonitorPowerSensor",
        _unexpected_power_monitor,
    )

    def _create_heater_sensors_stub(
        *_args: object, **_kwargs: object
    ) -> tuple[str, ...]:
        return ("temp-sensor", "energy-sensor", "power-sensor")

    monkeypatch.setattr(module, "_create_heater_sensors", _create_heater_sensors_stub)
    monkeypatch.setattr(
        entities_sensor_module,
        "_create_heater_sensors",
        _create_heater_sensors_stub,
    )
    monkeypatch.setattr(module, "_create_boost_sensors", lambda *a, **k: ())
    monkeypatch.setattr(
        entities_sensor_module, "_create_boost_sensors", lambda *a, **k: ()
    )

    class _DummyTotalEnergy:
        """Placeholder installation energy sensor for setup tests."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            return

    monkeypatch.setattr(module, "InstallationTotalEnergySensor", _DummyTotalEnergy)
    monkeypatch.setattr(
        entities_sensor_module, "InstallationTotalEnergySensor", _DummyTotalEnergy
    )

    energy_coordinator = SimpleNamespace(update_addresses=MagicMock())
    data_record = {
        "coordinator": object(),
        "dev_id": "dev-1",
        "client": object(),
        "energy_coordinator": energy_coordinator,
    }
    hass.data[DOMAIN][entry.entry_id] = data_record

    added_entities: list[object] = []

    def _async_add_entities(entities: list[object]) -> None:
        added_entities.extend(entities)

    await module.async_setup_entry(hass, entry, _async_add_entities)

    energy_coordinator.update_addresses.assert_called_once_with(inventory)
    assert iter_calls == [(("pmo",), None)]
    assert added_entities[:3] == [
        "temp-sensor",
        "energy-sensor",
        "power-sensor",
    ]
    assert any(isinstance(entity, _DummyTotalEnergy) for entity in added_entities)


def test_power_monitor_available_uses_inventory_has_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Power monitor availability should delegate to inventory.has_node."""

    module = importlib.import_module("custom_components.termoweb.sensor")
    monkeypatch.setattr(module, "Inventory", Inventory)

    coordinator = SimpleNamespace(
        data={"dev-1": {"pmo": {}}},
        async_add_listener=lambda *_args, **_kwargs: None,
        async_remove_listener=lambda *_args, **_kwargs: None,
        inventory=None,
    )
    inventory = Inventory(
        "dev-1",
        [PowerMonitorNode(name="Monitor", addr="01")],
    )

    calls: list[tuple[str, str]] = []

    def _fake_has_node(self, node_type: object, addr: object, *, _calls=calls) -> bool:
        _calls.append((node_type, addr))
        return True

    monkeypatch.setattr(Inventory, "has_node", _fake_has_node)

    sensor = module.PowerMonitorPowerSensor(
        coordinator,
        "entry-1",
        "dev-1",
        "01",
        "uid-1",
        device_name="Monitor",
        inventory=inventory,
    )

    setattr(sensor, "_inventory", inventory)
    setattr(sensor.coordinator, "inventory", inventory)
    resolved = sensor._resolve_inventory()
    assert resolved is inventory

    calls.clear()
    result = sensor.available
    assert calls == [("pmo", "01")]
    assert result is True


@pytest.mark.asyncio
async def test_heater_energy_sensor_availability() -> None:
    """Energy sensors should rely on inventory presence and coordinator health."""

    module = importlib.import_module("custom_components.termoweb.sensor")
    hass = HomeAssistant()
    hass.data = {DOMAIN: {}}
    dev_id = "dev-energy"
    raw_nodes = {"nodes": [{"type": "htr", "addr": "01", "name": "Heater"}]}
    node_inventory = list(build_node_inventory(raw_nodes))
    inventory = Inventory(dev_id, node_inventory)
    device_state = build_coordinator_device_state(
        nodes=raw_nodes,
        settings={"htr": {"01": {}}},
    )

    class _EnergyCoordinator(FakeCoordinator):
        """Coordinator stub exposing inventory updates for energy polling."""

        def update_addresses(self, updated_inventory: Inventory) -> None:
            """Store the latest immutable inventory reference."""

            self.inventory = updated_inventory

    energy_coordinator = _EnergyCoordinator(
        hass,
        dev_id=dev_id,
        dev=device_state,
        nodes=raw_nodes,
        inventory=inventory,
        data={dev_id: device_state},
    )
    energy_coordinator.last_update_success = True
    coordinator = FakeCoordinator(
        hass,
        dev_id=dev_id,
        dev=device_state,
        nodes=raw_nodes,
        inventory=inventory,
        data={dev_id: device_state},
    )
    entry_id = "entry-energy"
    entry = SimpleNamespace(entry_id=entry_id)
    hass.data[DOMAIN][entry.entry_id] = {
        "coordinator": coordinator,
        "dev_id": dev_id,
        "client": object(),
        "inventory": inventory,
        "energy_coordinator": energy_coordinator,
    }

    added_entities: list[Any] = []

    def _async_add_entities(entities: list[Any]) -> None:
        added_entities.extend(entities)

    await module.async_setup_entry(hass, entry, _async_add_entities)

    energy_entities = [
        entity
        for entity in added_entities
        if isinstance(entity, module.HeaterEnergyTotalSensor)
    ]
    assert energy_entities

    energy_sensor = energy_entities[0]
    energy_sensor.hass = hass
    await energy_sensor.async_added_to_hass()

    energy_coordinator.data = {}
    assert energy_sensor.available is True

    energy_coordinator.last_update_success = False
    assert energy_sensor.available is False
