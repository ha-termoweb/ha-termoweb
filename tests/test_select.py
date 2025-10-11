from __future__ import annotations

import asyncio
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from conftest import FakeCoordinator, _install_stubs, build_coordinator_device_state

_install_stubs()

import custom_components.termoweb.select as select_module
from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.heater import (
    BOOST_DURATION_OPTIONS,
    DEFAULT_BOOST_DURATION,
    format_boost_duration_label,
    get_boost_runtime_minutes,
    set_boost_runtime_minutes,
)
from custom_components.termoweb.identifiers import build_heater_entity_unique_id
from homeassistant.core import HomeAssistant

AccumulatorBoostDurationSelect = select_module.AccumulatorBoostDurationSelect
async_setup_entry = select_module.async_setup_entry


def _make_select_entity() -> AccumulatorBoostDurationSelect:
    """Create a selector instance for direct method testing."""

    hass = HomeAssistant()
    coordinator = FakeCoordinator(hass, dev_id="dev-select-test")
    return AccumulatorBoostDurationSelect(
        coordinator,
        "entry-select-test",
        "dev-select-test",
        "01",
        "Accumulator 1",
        "test-uid",
        node_type="acm",
    )




@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (format_boost_duration_label(60), 60),
        (" 2 hours ", 120),
        ("060", 60),
        ("120.0", 120),
        (60, 60),
        (60.0, 60),
        ("600", 600),
    ],
)
def test_option_to_minutes_accepts_valid_values(
    value: object, expected: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure valid inputs resolve to supported boost durations."""

    calls: list[object] = []
    real_coerce = select_module.coerce_boost_minutes

    def fake_coerce(candidate: object) -> int | None:
        calls.append(candidate)
        return real_coerce(candidate)

    monkeypatch.setattr(select_module, "coerce_boost_minutes", fake_coerce)

    entity = _make_select_entity()
    assert entity._option_to_minutes(value) == expected
    if not (isinstance(value, str) and value.strip() in entity._OPTION_MAP):
        assert calls == [value]


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "invalid",
        0,
        -60,
        45,
        "45",
        "11 hours",
        True,
    ],
)
def test_option_to_minutes_rejects_invalid_values(value: object) -> None:
    """Ensure invalid values are rejected by the coercion helper."""

    entity = _make_select_entity()
    assert entity._option_to_minutes(value) is None


@pytest.mark.asyncio
async def test_async_added_to_hass_prefers_stored_minutes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure stored minutes override restored state and persist to hass."""

    entity = _make_select_entity()
    entity.async_write_ha_state = MagicMock()

    stored_minutes = 180
    get_mock = MagicMock(return_value=stored_minutes)
    set_mock = MagicMock()

    monkeypatch.setattr(select_module, "get_boost_runtime_minutes", get_mock)
    monkeypatch.setattr(select_module, "set_boost_runtime_minutes", set_mock)
    monkeypatch.setattr(
        select_module.HeaterNodeBase,
        "async_added_to_hass",
        AsyncMock(),
    )
    monkeypatch.setattr(
        select_module.RestoreEntity,
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
    entity.async_get_last_state.assert_not_called()
    set_mock.assert_called_once_with(
        hass,
        entity._entry_id,
        entity._node_type,
        entity._addr,
        stored_minutes,
    )
    assert entity._attr_current_option == format_boost_duration_label(stored_minutes)
    assert entity.extra_state_attributes == {"preferred_minutes": stored_minutes}


@pytest.mark.asyncio
async def test_async_select_option_persists_valid_and_rejects_invalid(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify valid selections persist and invalid inputs leave state untouched."""

    entity = _make_select_entity()
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

    monkeypatch.setattr(select_module, "set_boost_runtime_minutes", fake_set)

    valid_option = format_boost_duration_label(120)
    await entity.async_select_option(valid_option)

    hass = entity.hass
    assert hass is not None
    assert calls == [
        (hass, entity._entry_id, entity._node_type, entity._addr, 120),
    ]
    assert entity._attr_current_option == valid_option

    caplog.set_level("ERROR")
    await entity.async_select_option("invalid option")

    assert calls == [
        (hass, entity._entry_id, entity._node_type, entity._addr, 120),
    ]
    assert entity._attr_current_option == valid_option
    assert "Invalid boost duration option" in caplog.text


def test_current_minutes_falls_back_to_resolved_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure ``_current_minutes`` uses resolver when no option is active."""

    entity = _make_select_entity()
    entity._attr_current_option = None

    resolved_minutes = 240
    resolver = MagicMock(return_value=resolved_minutes)
    monkeypatch.setattr(
        select_module,
        "resolve_boost_runtime_minutes",
        resolver,
    )

    hass = entity.hass
    assert hass is not None
    assert entity._current_minutes() == resolved_minutes
    resolver.assert_called_once_with(
        hass,
        entity._entry_id,
        entity._node_type,
        entity._addr,
    )



