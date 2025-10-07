from __future__ import annotations

import asyncio
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from conftest import FakeCoordinator, _install_stubs

_install_stubs()

import custom_components.termoweb.select as select_module
from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.heater import (
    DEFAULT_BOOST_DURATION,
    get_boost_runtime_minutes,
    iter_boost_button_metadata,
    set_boost_runtime_minutes,
)
from homeassistant.core import HomeAssistant

AccumulatorBoostDurationSelect = select_module.AccumulatorBoostDurationSelect
async_setup_entry = select_module.async_setup_entry


def test_select_setup_and_selection(
    monkeypatch: pytest.MonkeyPatch,
    heater_node_factory,
    boost_runtime_store,
    heater_hass_data,
) -> None:
    """Verify setup creates entities and persists selections."""

    async def _run() -> None:
        hass = HomeAssistant()
        entry = types.SimpleNamespace(entry_id="entry-select")
        dev_id = "dev-select"
        node = heater_node_factory("07")
        settings = {"boost_time": 120, "prog": [0] * 168}
        record = {
            "nodes": {},
            "nodes_by_type": {"acm": {"settings": {node.addr: settings}}},
            "htr": {"settings": {}},
        }
        coordinator = FakeCoordinator(
            hass,
            dev_id=dev_id,
            dev=record,
            data={dev_id: record},
        )
        heater_hass_data(
            hass,
            entry.entry_id,
            dev_id,
            coordinator,
            boost_runtime=boost_runtime_store("acm", node.addr, 30),
        )

        def fake_prepare(entry_data, default_name_simple):
            return ([node], {"acm": [node]}, {"acm": [node.addr]}, lambda *_: "Accumulator 7")

        monkeypatch.setattr(select_module, "prepare_heater_platform_data", fake_prepare)

        added: list[AccumulatorBoostDurationSelect] = []

        def _add_entities(entities):
            for entity in entities:
                entity.hass = hass
                entity.async_write_ha_state = MagicMock()
                added.append(entity)

        await async_setup_entry(hass, entry, _add_entities)

        assert len(added) == 1
        entity = added[0]
        assert isinstance(entity, AccumulatorBoostDurationSelect)
        assert entity.unique_id == f"{DOMAIN}:{dev_id}:acm:{node.addr}:boost_duration"
        expected_options = [
            str(item.minutes)
            for item in iter_boost_button_metadata()
            if item.minutes is not None
        ]
        options = getattr(entity, "options", getattr(entity, "_attr_options", []))
        assert options == expected_options

        await entity.async_added_to_hass()
        assert entity.async_write_ha_state.called
        assert entity.current_option == "30"
        assert (
            get_boost_runtime_minutes(hass, entry.entry_id, "acm", node.addr) == 30
        )

        entity.async_write_ha_state.reset_mock()
        await entity.async_select_option("120")
        entity.async_write_ha_state.assert_called_once()
        assert entity.current_option == "120"
        assert (
            get_boost_runtime_minutes(hass, entry.entry_id, "acm", node.addr) == 120
        )

        entity.async_write_ha_state.reset_mock()
        await entity.async_select_option("invalid")
        entity.async_write_ha_state.assert_not_called()
        assert entity.current_option == "120"

    asyncio.run(_run())


def test_select_restores_last_state(
    monkeypatch: pytest.MonkeyPatch,
    heater_node_factory,
    heater_hass_data,
) -> None:
    """Ensure stored state restoration takes precedence over defaults."""

    async def _run() -> None:
        hass = HomeAssistant()
        entry = types.SimpleNamespace(entry_id="entry-restore")
        dev_id = "dev-restore"
        node = heater_node_factory("09")
        record = {
            "nodes": {},
            "nodes_by_type": {"acm": {"settings": {node.addr: {"boost_time": 90}}}},
            "htr": {"settings": {}},
        }
        coordinator = FakeCoordinator(
            hass,
            dev_id=dev_id,
            dev=record,
            data={dev_id: record},
        )
        heater_hass_data(
            hass,
            entry.entry_id,
            dev_id,
            coordinator,
            boost_runtime={},
        )

        def fake_prepare(entry_data, default_name_simple):
            return ([node], {"acm": [node]}, {"acm": [node.addr]}, lambda *_: "Accumulator 9")

        monkeypatch.setattr(select_module, "prepare_heater_platform_data", fake_prepare)

        added: list[AccumulatorBoostDurationSelect] = []

        def _add_entities(entities):
            for entity in entities:
                entity.hass = hass
                entity.async_get_last_state = AsyncMock(
                    return_value=types.SimpleNamespace(state="60")
                )
                entity.async_write_ha_state = MagicMock()
                added.append(entity)

        await async_setup_entry(hass, entry, _add_entities)
        assert len(added) == 1
        entity = added[0]

        await entity.async_added_to_hass()
        assert entity.current_option == "60"
        assert (
            get_boost_runtime_minutes(hass, entry.entry_id, "acm", node.addr) == 60
        )
        assert entity.extra_state_attributes == {"preferred_minutes": 60}

        # Clearing stored value should fall back to defaults
        hass.data[DOMAIN][entry.entry_id]["boost_runtime"].clear()
        entity2 = AccumulatorBoostDurationSelect(
            coordinator,
            entry.entry_id,
            dev_id,
            node.addr,
            "Accumulator 9",
            f"{DOMAIN}:{dev_id}:acm:{node.addr}:boost_duration",
            node_type="acm",
        )
        entity2.hass = hass
        entity2.async_write_ha_state = MagicMock()
        entity2.async_get_last_state = AsyncMock(return_value=None)
        await entity2.async_added_to_hass()
        assert entity2.current_option == str(DEFAULT_BOOST_DURATION)
        assert (
            get_boost_runtime_minutes(hass, entry.entry_id, "acm", node.addr)
            == DEFAULT_BOOST_DURATION
        )

    asyncio.run(_run())


def test_select_filters_nodes_and_handles_fallback(
    monkeypatch: pytest.MonkeyPatch,
    heater_node_factory,
    heater_hass_data,
) -> None:
    """Verify setup skips unsupported nodes and applies default persistence."""

    async def _run() -> None:
        hass = HomeAssistant()
        entry = types.SimpleNamespace(entry_id="entry-fallback")
        dev_id = "dev-fallback"

        non_acm = heater_node_factory("01", node_type="htr")
        disabled_acm = heater_node_factory("02", supports_boost=False)
        enabled_acm = heater_node_factory("03")

        record = {
            "nodes": {},
            "nodes_by_type": {"acm": {"settings": {enabled_acm.addr: {}}}},
            "htr": {"settings": {}},
        }
        coordinator = FakeCoordinator(
            hass,
            dev_id=dev_id,
            dev=record,
            data={dev_id: record},
        )
        heater_hass_data(
            hass,
            entry.entry_id,
            dev_id,
            coordinator,
            boost_runtime={},
        )

        def fake_prepare(entry_data, default_name_simple):
            return (
                [non_acm, disabled_acm, enabled_acm],
                {"htr": [non_acm], "acm": [disabled_acm, enabled_acm]},
                {"acm": [disabled_acm.addr, enabled_acm.addr]},
                lambda *_: "Accumulator 3",
            )

        monkeypatch.setattr(select_module, "prepare_heater_platform_data", fake_prepare)

        added: list[AccumulatorBoostDurationSelect] = []

        def _add_entities(entities):
            for entity in entities:
                entity.hass = hass
                entity.async_write_ha_state = MagicMock()
                added.append(entity)

        await async_setup_entry(hass, entry, _add_entities)

        assert len(added) == 1
        entity = added[0]
        assert isinstance(entity, AccumulatorBoostDurationSelect)

        # Numeric string with decimals should normalise to a supported option.
        assert entity._option_to_minutes("60.0") == 60

        set_boost_runtime_minutes(hass, entry.entry_id, "acm", entity._addr, 45)
        entity._attr_current_option = "invalid"
        entity.hass = hass
        assert entity._current_minutes() == 45

        # Force fallback branch when option mapping is missing.
        entity._REVERSE_OPTION_MAP = {}
        entity._apply_minutes(75, persist=False)
        assert entity.current_option == str(DEFAULT_BOOST_DURATION)
        del entity._REVERSE_OPTION_MAP

        entity.heater_settings = lambda: {"boost_time": "120"}
        assert entity._initial_minutes_from_settings() == 120
        assert entity._option_to_minutes("999") is None
        assert entity._validate_minutes(None) == DEFAULT_BOOST_DURATION

        entity._attr_current_option = "invalid"
        entity.hass = None
        assert entity._current_minutes() == DEFAULT_BOOST_DURATION

    asyncio.run(_run())
