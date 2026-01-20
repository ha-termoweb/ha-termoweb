"""Tests for sensor energy normalisation helper."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from conftest import (
    FakeCoordinator,
    build_coordinator_device_state,
    build_entry_runtime,
)
from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.coordinator import EnergyStateCoordinator
from custom_components.termoweb.sensor import _normalise_energy_value
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
    entities_sensor_module = importlib.import_module(
        "custom_components.termoweb.entities.sensor"
    )
    inventory_module = importlib.import_module("custom_components.termoweb.inventory")
    hass = HomeAssistant()
    hass.data = {DOMAIN: {}}
    entry = SimpleNamespace(entry_id="entry-1")
    raw_nodes = [{"type": "htr", "addr": "1", "name": "Heater"}]
    node_inventory = inventory_module.build_node_inventory({"nodes": raw_nodes})
    inventory = inventory_module.Inventory("dev-1", node_inventory)
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
        inventory_module.Inventory,
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

    energy_coordinator = EnergyStateCoordinator(
        hass,
        SimpleNamespace(),
        "dev-1",
        inventory,
    )
    build_entry_runtime(
        hass=hass,
        entry_id=entry.entry_id,
        dev_id="dev-1",
        coordinator=object(),
        client=object(),
        energy_coordinator=energy_coordinator,
        inventory=inventory,
    )

    added_entities: list[object] = []

    def _async_add_entities(entities: list[object]) -> None:
        added_entities.extend(entities)

    await module.async_setup_entry(hass, entry, _async_add_entities)

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
    inventory_module = importlib.import_module("custom_components.termoweb.inventory")
    monkeypatch.setattr(module, "Inventory", inventory_module.Inventory)

    coordinator = SimpleNamespace(
        data={"dev-1": {"pmo": {}}},
        async_add_listener=lambda *_args, **_kwargs: None,
        async_remove_listener=lambda *_args, **_kwargs: None,
        inventory=None,
    )
    inventory = inventory_module.Inventory(
        "dev-1",
        [inventory_module.PowerMonitorNode(name="Monitor", addr="01")],
    )

    calls: list[tuple[str, str]] = []

    def _fake_has_node(self, node_type: object, addr: object, *, _calls=calls) -> bool:
        _calls.append((node_type, addr))
        return True

    monkeypatch.setattr(inventory_module.Inventory, "has_node", _fake_has_node)

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
    inventory_module = importlib.import_module("custom_components.termoweb.inventory")
    hass = HomeAssistant()
    hass.data = {DOMAIN: {}}
    dev_id = "dev-energy"
    raw_nodes = {"nodes": [{"type": "htr", "addr": "01", "name": "Heater"}]}
    node_inventory = list(inventory_module.build_node_inventory(raw_nodes))
    inventory = inventory_module.Inventory(dev_id, node_inventory)
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
    build_entry_runtime(
        hass=hass,
        entry_id=entry.entry_id,
        dev_id=dev_id,
        coordinator=coordinator,
        client=object(),
        inventory=inventory,
        energy_coordinator=energy_coordinator,
        config_entry=entry,
    )

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
    assert energy_sensor.available is True

    energy_coordinator.last_update_success = False
    assert energy_sensor.available is False


def test_energy_sensors_read_domain_view_without_coordinator_data() -> None:
    """Energy sensors should read metrics from the domain view only."""

    module = importlib.import_module("custom_components.termoweb.sensor")
    energy_module = importlib.import_module("custom_components.termoweb.domain.energy")
    ids_module = importlib.import_module("custom_components.termoweb.domain.ids")
    inventory_module = importlib.import_module("custom_components.termoweb.inventory")
    state_module = importlib.import_module("custom_components.termoweb.domain.state")
    view_module = importlib.import_module("custom_components.termoweb.domain.view")

    dev_id = "dev-energy"
    node_id = ids_module.NodeId(ids_module.NodeType.HEATER, "01")
    store = state_module.DomainStateStore([node_id])
    snapshot = energy_module.EnergySnapshot(
        dev_id=dev_id,
        metrics={
            node_id: energy_module.EnergyNodeMetrics(
                energy_kwh=2.5,
                power_w=125.0,
                source="test",
                ts=0.0,
            )
        },
        updated_at=0.0,
        ws_deadline=None,
    )
    store.set_energy_snapshot(snapshot)
    view = view_module.DomainStateView(dev_id, store)

    raw_nodes = {"nodes": [{"type": "htr", "addr": "01", "name": "Heater"}]}
    node_inventory = list(inventory_module.build_node_inventory(raw_nodes))
    inventory = inventory_module.Inventory(dev_id, node_inventory)

    class _Coordinator:
        """Coordinator stub that forbids data access."""

        last_update_success = True

        @property
        def data(self) -> None:
            """Fail the test when coordinator data is accessed."""

            raise AssertionError("Energy sensors must not access coordinator.data")

    sensor = module.HeaterEnergyTotalSensor(
        _Coordinator(),
        view,
        "entry-1",
        dev_id,
        "01",
        "uid-1",
        device_name="Heater",
        node_type="htr",
        inventory=inventory,
    )

    assert sensor.native_value == 2.5


def test_power_monitor_sensors_read_domain_view_metrics() -> None:
    """Power monitor sensors should read metrics from the domain view."""

    module = importlib.import_module("custom_components.termoweb.sensor")
    energy_module = importlib.import_module("custom_components.termoweb.domain.energy")
    ids_module = importlib.import_module("custom_components.termoweb.domain.ids")
    inventory_module = importlib.import_module("custom_components.termoweb.inventory")
    state_module = importlib.import_module("custom_components.termoweb.domain.state")
    view_module = importlib.import_module("custom_components.termoweb.domain.view")

    dev_id = "dev-power"
    node_id = ids_module.NodeId(ids_module.NodeType.POWER_MONITOR, "01")
    store = state_module.DomainStateStore([node_id])
    snapshot = energy_module.EnergySnapshot(
        dev_id=dev_id,
        metrics={
            node_id: energy_module.EnergyNodeMetrics(
                energy_kwh=1.0,
                power_w=50.0,
                source="test",
                ts=0.0,
            )
        },
        updated_at=0.0,
        ws_deadline=None,
    )
    store.set_energy_snapshot(snapshot)
    view = view_module.DomainStateView(dev_id, store)

    inventory = inventory_module.Inventory(
        dev_id,
        [inventory_module.PowerMonitorNode(name="Monitor", addr="01")],
    )

    class _Coordinator:
        """Coordinator stub that forbids data access."""

        @property
        def data(self) -> None:
            """Fail the test when coordinator data is accessed."""

            raise AssertionError(
                "Power monitor sensors must not access coordinator.data"
            )

    sensor = module.PowerMonitorPowerSensor(
        _Coordinator(),
        "entry-1",
        dev_id,
        "01",
        "uid-1",
        device_name="Monitor",
        inventory=inventory,
        domain_view=view,
    )

    assert sensor.available is True
    assert sensor.native_value == 50.0
