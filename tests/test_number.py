from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from conftest import FakeCoordinator, _install_stubs, build_entry_runtime

from custom_components.termoweb.inventory import build_node_inventory
import custom_components.termoweb.heater as heater_module

_install_stubs()

import custom_components.termoweb.number as number_module
from custom_components.termoweb.entities import number as entities_number_module
from custom_components.termoweb.const import DOMAIN
from homeassistant.core import HomeAssistant

AccumulatorBoostDurationNumber = number_module.AccumulatorBoostDurationNumber
AccumulatorBoostTemperatureNumber = number_module.AccumulatorBoostTemperatureNumber
async_setup_entry = number_module.async_setup_entry


def _patch_number_attr(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: object,
    *,
    raising: bool | None = None,
) -> None:
    """Patch a number module attribute across shim + entity modules."""

    if raising is None:
        monkeypatch.setattr(number_module, name, value)
        monkeypatch.setattr(entities_number_module, name, value)
    else:
        monkeypatch.setattr(number_module, name, value, raising=raising)
        monkeypatch.setattr(entities_number_module, name, value, raising=raising)


def _make_duration_entity() -> AccumulatorBoostDurationNumber:
    """Create a duration number instance for direct method testing."""

    hass = HomeAssistant()
    coordinator = FakeCoordinator(hass, dev_id="dev-number-test")
    return AccumulatorBoostDurationNumber(
        coordinator,
        "entry-number-test",
        "dev-number-test",
        "01",
        "Accumulator 1",
        "test-duration-uid",
        node_type="acm",
    )


def _make_temperature_entity() -> AccumulatorBoostTemperatureNumber:
    """Create a temperature number instance for direct method testing."""

    hass = HomeAssistant()
    coordinator = FakeCoordinator(hass, dev_id="dev-number-test")
    return AccumulatorBoostTemperatureNumber(
        coordinator,
        "entry-number-test",
        "dev-number-test",
        "02",
        "Accumulator 2",
        "test-temperature-uid",
        node_type="acm",
    )


@pytest.mark.asyncio
async def test_duration_async_added_to_hass_uses_last_state_without_device_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restore boost minutes from the previous state when the device has not reported one."""

    entity = _make_duration_entity()
    entity.async_write_ha_state = MagicMock()

    monkeypatch.setattr(
        number_module.HeaterNodeBase,
        "async_added_to_hass",
        AsyncMock(),
    )
    monkeypatch.setattr(
        number_module.RestoreEntity,
        "async_added_to_hass",
        AsyncMock(),
    )

    entity.async_get_last_state = AsyncMock(
        return_value=type("state", (), {"state": "3"})(),
    )

    await entity.async_added_to_hass()

    assert entity.native_value == 3.0


@pytest.mark.asyncio
async def test_duration_async_added_to_hass_uses_settings_when_state_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restore boost minutes from the device settings when state is missing."""

    entity = _make_duration_entity()
    entity.async_write_ha_state = MagicMock()
    entity.accumulator_state = MagicMock(return_value=SimpleNamespace(boost_time=240))

    monkeypatch.setattr(
        number_module.HeaterNodeBase,
        "async_added_to_hass",
        AsyncMock(),
    )
    monkeypatch.setattr(
        number_module.RestoreEntity,
        "async_added_to_hass",
        AsyncMock(),
    )

    entity.async_get_last_state = AsyncMock(return_value=None)

    await entity.async_added_to_hass()

    assert entity.native_value == 4.0


@pytest.mark.asyncio
async def test_temperature_async_added_to_hass_uses_last_state_without_device_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restore boost temperature from the previous state when the device has not reported one."""

    entity = _make_temperature_entity()
    entity.async_write_ha_state = MagicMock()

    monkeypatch.setattr(
        number_module.HeaterNodeBase,
        "async_added_to_hass",
        AsyncMock(),
    )
    monkeypatch.setattr(
        number_module.RestoreEntity,
        "async_added_to_hass",
        AsyncMock(),
    )

    entity.async_get_last_state = AsyncMock(
        return_value=type("state", (), {"state": "21.25"})(),
    )

    await entity.async_added_to_hass()

    assert entity.native_value == 21.3


@pytest.mark.asyncio
async def test_temperature_async_added_to_hass_uses_settings_when_state_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restore boost temperature from the device settings when state is missing."""

    entity = _make_temperature_entity()
    entity.async_write_ha_state = MagicMock()
    entity.accumulator_state = MagicMock(return_value=SimpleNamespace(boost_temp=24.4))

    monkeypatch.setattr(
        number_module.HeaterNodeBase,
        "async_added_to_hass",
        AsyncMock(),
    )
    monkeypatch.setattr(
        number_module.RestoreEntity,
        "async_added_to_hass",
        AsyncMock(),
    )

    entity.async_get_last_state = AsyncMock(return_value=None)

    await entity.async_added_to_hass()

    assert entity.native_value == 24.4


@pytest.mark.asyncio
async def test_async_setup_entry_creates_number_entities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure the number platform sets up duration and temperature sliders."""

    hass = HomeAssistant()
    entry_id = "entry-setup-test"
    dev_id = "dev-setup-test"

    raw_nodes = [
        {
            "addr": "02",
            "name": "Accumulator 2",
            "type": "acm",
        }
    ]
    payload = {"nodes": raw_nodes}
    node_inventory = build_node_inventory(raw_nodes)
    InventoryType = heater_module.Inventory
    inventory = InventoryType(dev_id, node_inventory)
    coordinator = FakeCoordinator(hass, dev_id=dev_id)

    heater_details = heater_module.HeaterPlatformDetails(
        inventory=inventory,
        default_name_simple=lambda addr: f"Heater {addr}",
    )

    _patch_number_attr(
        monkeypatch,
        "boostable_accumulator_details_for_entry",
        lambda *_args, **_kwargs: (
            heater_details,
            [("acm", "02", "Accumulator 2")],
        ),
    )

    build_entry_runtime(
        hass=hass,
        entry_id=entry_id,
        dev_id=dev_id,
        inventory=inventory,
        coordinator=coordinator,
    )

    calls: list[
        list[AccumulatorBoostDurationNumber | AccumulatorBoostTemperatureNumber]
    ] = []

    def fake_add(
        entities: list[
            AccumulatorBoostDurationNumber | AccumulatorBoostTemperatureNumber
        ],
    ) -> None:
        calls.append(entities)

    await async_setup_entry(
        hass,
        type("entry", (), {"entry_id": entry_id})(),
        fake_add,
    )

    assert calls, "async_add_entities should receive number entities"
    created = calls[0]
    assert any(isinstance(entity, AccumulatorBoostDurationNumber) for entity in created)
    assert any(
        isinstance(entity, AccumulatorBoostTemperatureNumber) for entity in created
    )

    for entity in created:
        assert getattr(entity, "_attr_has_entity_name", None) is True
        assert getattr(entity, "_attr_entity_category", None) is not None
        assert getattr(entity, "entity_id", None) is None


@pytest.mark.asyncio
async def test_async_setup_entry_radio_gets_local_priority_and_limit() -> None:
    """The radio backend's local power manager brings priority and limit numbers."""

    hass = HomeAssistant()
    entry_id = "entry-radio-number"
    dev_id = "dev-radio-number"
    raw_nodes = [{"addr": "6", "name": "Heater 6", "type": "htr"}]
    inventory = heater_module.Inventory(dev_id, build_node_inventory(raw_nodes))
    build_entry_runtime(
        hass=hass,
        entry_id=entry_id,
        dev_id=dev_id,
        inventory=inventory,
        coordinator=FakeCoordinator(hass, dev_id=dev_id),
        brand="radio",
    )
    calls: list[list[object]] = []

    await async_setup_entry(
        hass,
        type("entry", (), {"entry_id": entry_id})(),
        calls.append,
    )

    assert [type(e).__name__ for e in calls[0]] == [
        "HeaterPriorityNumber",
        "PowerLimitNumber",
    ]
