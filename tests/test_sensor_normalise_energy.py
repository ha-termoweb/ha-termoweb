"""Tests for sensor energy normalisation helper."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Callable
from unittest.mock import MagicMock

import pytest

from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.coordinator import EnergyStateCoordinator
from custom_components.termoweb.inventory import Inventory, build_node_inventory
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
    hass = HomeAssistant()
    hass.data = {DOMAIN: {}}
    entry = SimpleNamespace(entry_id="entry-1")
    raw_nodes = [{"type": "htr", "addr": "1", "name": "Heater"}]
    node_inventory = build_node_inventory({"nodes": raw_nodes})
    inventory = Inventory("dev-1", {"nodes": raw_nodes}, node_inventory)
    assert "pmo" not in inventory.nodes_by_type

    heater_details = SimpleNamespace(
        inventory=inventory,
        iter_metadata=lambda: iter([("htr", object(), "1", "Heater 1")]),
    )
    monkeypatch.setattr(
        module,
        "heater_platform_details_for_entry",
        lambda _data, **_kwargs: heater_details,
    )
    monkeypatch.setattr(module, "iter_boostable_heater_nodes", lambda _details: [])
    monkeypatch.setattr(module, "log_skipped_nodes", lambda *args, **kwargs: None)

    def _unexpected_power_monitor(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("power monitor sensors should not be created")

    monkeypatch.setattr(module, "PowerMonitorEnergySensor", _unexpected_power_monitor)
    monkeypatch.setattr(module, "PowerMonitorPowerSensor", _unexpected_power_monitor)

    def _create_heater_sensors_stub(*_args: object, **_kwargs: object) -> tuple[str, ...]:
        return ("temp-sensor", "energy-sensor", "power-sensor")

    monkeypatch.setattr(module, "_create_heater_sensors", _create_heater_sensors_stub)
    monkeypatch.setattr(module, "_create_boost_sensors", lambda *a, **k: ())

    class _DummyTotalEnergy:
        """Placeholder installation energy sensor for setup tests."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            return

    monkeypatch.setattr(module, "InstallationTotalEnergySensor", _DummyTotalEnergy)

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
    assert added_entities[:3] == [
        "temp-sensor",
        "energy-sensor",
        "power-sensor",
    ]
    assert any(isinstance(entity, _DummyTotalEnergy) for entity in added_entities)
