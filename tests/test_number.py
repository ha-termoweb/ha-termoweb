from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from conftest import FakeCoordinator, _install_stubs

from custom_components.termoweb.inventory import build_node_inventory
import custom_components.termoweb.heater as heater_module

_install_stubs()

import custom_components.termoweb.number as number_module
from custom_components.termoweb.const import DOMAIN
from homeassistant.core import HomeAssistant

AccumulatorBoostDurationNumber = number_module.AccumulatorBoostDurationNumber
AccumulatorBoostTemperatureNumber = number_module.AccumulatorBoostTemperatureNumber
async_setup_entry = number_module.async_setup_entry


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
async def test_duration_async_added_to_hass_prefers_stored_minutes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure stored minutes override restored state and persist to hass."""

    entity = _make_duration_entity()
    entity.async_write_ha_state = MagicMock()

    stored_minutes = 180
    get_mock = MagicMock(return_value=stored_minutes)
    set_mock = MagicMock()

    monkeypatch.setattr(number_module, "get_boost_runtime_minutes", get_mock)
    monkeypatch.setattr(number_module, "set_boost_runtime_minutes", set_mock)
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

    entity.async_get_last_state = AsyncMock()

    await entity.async_added_to_hass()

    hass = entity.hass
    assert hass is not None
    get_mock.assert_called_once_with(
        hass,
        entity._entry_id,
        entity._node_type,
        entity._addr,
    )
    set_mock.assert_called_once_with(
        hass,
        entity._entry_id,
        entity._node_type,
        entity._addr,
        stored_minutes,
    )
    assert entity.native_value == stored_minutes / 60
    assert entity.extra_state_attributes == {"preferred_minutes": stored_minutes}


@pytest.mark.asyncio
async def test_duration_async_added_to_hass_uses_last_state_when_cache_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restore boost minutes from the previous state when cache is empty."""

    entity = _make_duration_entity()
    entity.async_write_ha_state = MagicMock()

    monkeypatch.setattr(
        number_module, "get_boost_runtime_minutes", MagicMock(return_value=None)
    )
    set_mock = MagicMock()
    monkeypatch.setattr(number_module, "set_boost_runtime_minutes", set_mock)
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

    hass = entity.hass
    assert hass is not None
    set_mock.assert_called_once_with(
        hass,
        entity._entry_id,
        entity._node_type,
        entity._addr,
        180,
    )
    assert entity.native_value == 3.0


@pytest.mark.asyncio
async def test_duration_async_added_to_hass_uses_settings_when_state_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restore boost minutes from cached settings when state is missing."""

    entity = _make_duration_entity()
    entity.async_write_ha_state = MagicMock()
    entity.heater_settings = MagicMock(return_value={"boost_time": 240})

    monkeypatch.setattr(
        number_module, "get_boost_runtime_minutes", MagicMock(return_value=None)
    )
    set_mock = MagicMock()
    monkeypatch.setattr(number_module, "set_boost_runtime_minutes", set_mock)
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

    hass = entity.hass
    assert hass is not None
    set_mock.assert_called_once_with(
        hass,
        entity._entry_id,
        entity._node_type,
        entity._addr,
        240,
    )
    assert entity.native_value == 4.0


@pytest.mark.asyncio
async def test_duration_async_set_native_value_persists_valid_and_rejects_invalid(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify valid slider updates persist and invalid inputs leave state untouched."""

    entity = _make_duration_entity()
    entity.async_write_ha_state = MagicMock()

    calls: list[tuple[HomeAssistant, str, str, str, int]] = []

    def fake_set(
        hass: HomeAssistant,
        entry_id: str,
        node_type: str,
        addr: str,
        minutes: int,
    ) -> None:
        calls.append((hass, entry_id, node_type, addr, minutes))

    monkeypatch.setattr(number_module, "set_boost_runtime_minutes", fake_set)

    await entity.async_set_native_value(2.0)

    hass = entity.hass
    assert hass is not None
    assert calls == [
        (hass, entity._entry_id, entity._node_type, entity._addr, 120),
    ]
    assert entity.native_value == 2.0

    caplog.set_level("ERROR")
    await entity.async_set_native_value(0.5)

    assert calls == [
        (hass, entity._entry_id, entity._node_type, entity._addr, 120),
    ]
    assert "Invalid boost duration" in caplog.text


@pytest.mark.asyncio
async def test_temperature_async_added_to_hass_prefers_stored_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure stored temperatures override restored state and persist to hass."""

    entity = _make_temperature_entity()
    entity.async_write_ha_state = MagicMock()

    stored_temperature = 21.5
    get_mock = MagicMock(return_value=stored_temperature)
    set_mock = MagicMock()

    monkeypatch.setattr(number_module, "get_boost_temperature", get_mock)
    monkeypatch.setattr(number_module, "set_boost_temperature", set_mock)
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

    entity.async_get_last_state = AsyncMock()

    await entity.async_added_to_hass()

    hass = entity.hass
    assert hass is not None
    get_mock.assert_called_once_with(
        hass,
        entity._entry_id,
        entity._node_type,
        entity._addr,
    )
    set_mock.assert_called_once_with(
        hass,
        entity._entry_id,
        entity._node_type,
        entity._addr,
        stored_temperature,
    )
    assert entity.native_value == stored_temperature
    assert entity.extra_state_attributes == {
        "preferred_temperature": stored_temperature,
    }


@pytest.mark.asyncio
async def test_temperature_async_added_to_hass_uses_last_state_when_cache_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restore boost temperature from the previous state when cache is empty."""

    entity = _make_temperature_entity()
    entity.async_write_ha_state = MagicMock()

    monkeypatch.setattr(
        number_module, "get_boost_temperature", MagicMock(return_value=None)
    )
    set_mock = MagicMock()
    monkeypatch.setattr(number_module, "set_boost_temperature", set_mock)
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

    hass = entity.hass
    assert hass is not None
    set_mock.assert_called_once_with(
        hass,
        entity._entry_id,
        entity._node_type,
        entity._addr,
        21.3,
    )
    assert entity.native_value == 21.3


@pytest.mark.asyncio
async def test_temperature_async_added_to_hass_uses_settings_when_state_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restore boost temperature from cached settings when state is missing."""

    entity = _make_temperature_entity()
    entity.async_write_ha_state = MagicMock()
    entity.heater_settings = MagicMock(return_value={"boost_temp": 24.4})

    monkeypatch.setattr(
        number_module, "get_boost_temperature", MagicMock(return_value=None)
    )
    set_mock = MagicMock()
    monkeypatch.setattr(number_module, "set_boost_temperature", set_mock)
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

    hass = entity.hass
    assert hass is not None
    set_mock.assert_called_once_with(
        hass,
        entity._entry_id,
        entity._node_type,
        entity._addr,
        24.4,
    )
    assert entity.native_value == 24.4


@pytest.mark.asyncio
async def test_temperature_async_set_native_value_calls_service(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify slider updates call the preset service and persist the value."""

    entity = _make_temperature_entity()
    hass = entity.hass
    assert hass is not None

    hass.services = type("svc", (), {"async_call": AsyncMock()})()
    entity.async_write_ha_state = MagicMock()

    set_mock = MagicMock()
    monkeypatch.setattr(number_module, "set_boost_temperature", set_mock)
    monkeypatch.setattr(
        number_module,
        "resolve_climate_entity_id",
        lambda *_: "climate.accumulator_2",
    )

    await entity.async_set_native_value(23.25)

    hass.services.async_call.assert_awaited_once_with(
        DOMAIN,
        "set_acm_preset",
        {"entity_id": "climate.accumulator_2", "temperature": 23.3},
        blocking=True,
    )
    set_mock.assert_called_once_with(
        hass,
        entity._entry_id,
        entity._node_type,
        entity._addr,
        23.3,
    )
    entity.async_write_ha_state.assert_called()

    caplog.set_level("ERROR")
    await entity.async_set_native_value(50.0)
    assert "Invalid boost temperature" in caplog.text
    assert hass.services.async_call.await_count == 1


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

    monkeypatch.setattr(
        number_module,
        "boostable_accumulator_details_for_entry",
        lambda *_args, **_kwargs: (
            heater_details,
            [("acm", "02", "Accumulator 2")],
        ),
    )

    hass.data.setdefault(DOMAIN, {})[entry_id] = {
        "coordinator": coordinator,
        "dev_id": dev_id,
    }

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
