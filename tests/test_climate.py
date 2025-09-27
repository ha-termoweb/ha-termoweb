from __future__ import annotations

import asyncio
import datetime as dt
import logging
import time
from collections import deque
from collections.abc import Coroutine
import types
from typing import Any, Deque, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from conftest import FakeCoordinator, _install_stubs

_install_stubs()

from custom_components.termoweb import climate as climate_module
from custom_components.termoweb.const import BRAND_TERMOWEB, DOMAIN, signal_ws_data
from custom_components.termoweb.nodes import HeaterNode
from homeassistant.components.climate import HVACAction, HVACMode
from homeassistant.const import ATTR_TEMPERATURE
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import entity_platform as entity_platform_module
from homeassistant.helpers.entity_platform import EntityPlatform
from homeassistant.helpers import dispatcher as dispatcher_module
from homeassistant.util import dt as dt_util

HeaterClimateEntity = climate_module.HeaterClimateEntity
async_setup_entry = climate_module.async_setup_entry


def _reset_environment() -> None:
    _install_stubs()
    entity_platform_module._set_current_platform(EntityPlatform())
    dispatcher_module._dispatch_map = {}
    dt_util.NOW = dt.datetime(2024, 1, 1, 0, 0, tzinfo=dt.timezone.utc)
    FakeCoordinator.instances.clear()


def _make_coordinator(
    hass: HomeAssistant,
    dev_id: str,
    record: dict[str, Any],
    *,
    client: Any | None = None,
    node_inventory: list[Any] | None = None,
) -> FakeCoordinator:
    return FakeCoordinator(
        hass,
        client=client,
        dev_id=dev_id,
        dev=record,
        nodes=record.get("nodes", {}),
        node_inventory=node_inventory,
        data={dev_id: record},
    )


# -------------------- Helpers for tests --------------------


def test_termoweb_heater_is_heater_node() -> None:
    _reset_environment()
    hass = HomeAssistant()
    dev_id = "dev"
    coordinator_data = {dev_id: {"htr": {"settings": {}}, "nodes": {}}}
    coordinator = _make_coordinator(hass, dev_id, coordinator_data[dev_id])

    heater = HeaterClimateEntity(
        coordinator,
        "entry",
        "dev",
        "1",
        " Living Room ",
    )

    assert isinstance(heater, HeaterNode)
    assert heater.type == "htr"
    assert heater.addr == "1"
    assert heater.name == "Living Room"


def test_async_setup_entry_creates_entities() -> None:
    async def _run() -> None:
        _reset_environment()
        hass = HomeAssistant()
        entry_id = "entry"
        dev_id = "dev1"
        nodes = {
            "nodes": [
                {"type": "htr", "addr": "A1", "name": " Living Room "},
                {"type": "HTR", "addr": "B2"},
                {"type": "acm", "addr": "C3", "name": " Basement Accumulator "},
                {"type": "other", "addr": "X"},
            ]
        }
        coordinator_data = {
            dev_id: {
                "nodes": nodes,
                "htr": {"settings": {"A1": {}, "B2": {}}, "addrs": ["A1", "B2"]},
                "nodes_by_type": {
                    "htr": {
                        "settings": {"A1": {}, "B2": {}},
                        "addrs": ["A1", "B2"],
                    },
                    "acm": {
                        "settings": {"C3": {"units": "C"}},
                        "addrs": ["C3"],
                    },
                },
                "version": "3.1.4",
            }
        }
        coordinator = _make_coordinator(
            hass,
            dev_id,
            coordinator_data[dev_id],
            client=AsyncMock(),
            node_inventory=coordinator_data[dev_id].get("node_inventory"),
        )

        hass.data = {
            DOMAIN: {
                entry_id: {
                    "coordinator": coordinator,
                    "dev_id": dev_id,
                    "client": AsyncMock(),
                    "nodes": nodes,
                    "node_inventory": climate_module.build_node_inventory(nodes),
                    "version": "3.1.4",
                    "brand": BRAND_TERMOWEB,
                }
            }
        }

        added: list[HeaterClimateEntity] = []

        def _async_add_entities(entities: list[HeaterClimateEntity]) -> None:
            added.extend(entities)

        platform = EntityPlatform()
        entity_platform_module._set_current_platform(platform)

        entry = types.SimpleNamespace(entry_id=entry_id)
        await async_setup_entry(hass, entry, _async_add_entities)

        assert len(added) == 3
        entities_by_addr = {entity._addr: entity for entity in added}
        assert set(entities_by_addr) == {"A1", "B2", "C3"}
        assert isinstance(entities_by_addr["A1"], HeaterClimateEntity)
        assert isinstance(entities_by_addr["B2"], HeaterClimateEntity)
        acc = entities_by_addr["C3"]
        assert isinstance(acc, climate_module.AccumulatorClimateEntity)
        assert acc.available
        names = {entity._addr: entity._attr_name for entity in added}
        assert names["A1"] == "Living Room"
        assert names["B2"] == "Heater B2"
        assert names["C3"] == "Basement Accumulator"

        registered = [name for name, _, _ in platform.registered]
        assert registered == ["set_schedule", "set_preset_temperatures"]

        for entity in added:
            info = entity.device_info
            assert info["identifiers"] == {(DOMAIN, dev_id, entity._addr)}
            assert info["manufacturer"] == "TermoWeb"
            expected_model = "Accumulator"
            if getattr(entity, "_node_type", "htr") != "acm":
                expected_model = "Heater"
            assert info["model"] == expected_model
            assert info["via_device"] == (DOMAIN, dev_id)

        schedule_name, _, schedule_handler = platform.registered[0]
        preset_name, _, preset_handler = platform.registered[1]
        assert schedule_name == "set_schedule"
        assert preset_name == "set_preset_temperatures"

        schedule_prog = [0] * 168
        first = entities_by_addr["A1"]
        first.async_set_schedule = AsyncMock()
        await schedule_handler(first, ServiceCall({"prog": schedule_prog}))
        first.async_set_schedule.assert_awaited_once_with(schedule_prog)

        first.async_set_preset_temperatures = AsyncMock()
        await preset_handler(first, ServiceCall({"ptemp": [18.0, 19.0, 20.0]}))
        first.async_set_preset_temperatures.assert_awaited_once_with(
            ptemp=[18.0, 19.0, 20.0]
        )

        second = entities_by_addr["B2"]
        second.async_set_preset_temperatures = AsyncMock()
        await preset_handler(
            second,
            ServiceCall({"cold": 15.0, "night": 18.0, "day": 20.0}),
        )
        second.async_set_preset_temperatures.assert_awaited_once_with(
            cold=15.0, night=18.0, day=20.0
        )

    asyncio.run(_run())


def test_async_setup_entry_default_names_and_invalid_nodes() -> None:
    async def _run() -> None:
        _reset_environment()
        hass = HomeAssistant()
        entry_id = "entry-default"
        dev_id = "dev-default"
        raw_nodes = {
            "nodes": [
                {"type": "htr", "addr": "1"},
                {"type": "acm", "addr": "2"},
                {"type": "pmo", "addr": "P1"},
            ]
        }
        inventory = climate_module.build_node_inventory(raw_nodes)
        inventory.append(types.SimpleNamespace(type="  ", addr="extra"))
        inventory.append(types.SimpleNamespace(type="htr", addr=" "))

        coordinator_data = {
            dev_id: {
                "nodes": {},
                "htr": {"settings": {}},
                "nodes_by_type": {},
            }
        }
        coordinator = _make_coordinator(
            hass,
            dev_id,
            coordinator_data[dev_id],
            client=AsyncMock(),
            node_inventory=inventory,
        )

        hass.data = {
            DOMAIN: {
                entry_id: {
                    "coordinator": coordinator,
                    "dev_id": dev_id,
                    "client": AsyncMock(),
                    "nodes": {},
                    "node_inventory": inventory,
                }
            }
        }

        added: list[HeaterClimateEntity] = []

        def _add_entities(entities: list[HeaterClimateEntity]) -> None:
            added.extend(entities)

        entity_platform_module._set_current_platform(EntityPlatform())

        entry = types.SimpleNamespace(entry_id=entry_id)
        await async_setup_entry(hass, entry, _add_entities)

        names = sorted(entity._attr_name for entity in added)
        assert names == ["Accumulator 2", "Heater 1"]
        assert all(entity._addr in {"1", "2"} for entity in added)

    asyncio.run(_run())


def test_async_setup_entry_creates_accumulator_entity() -> None:
    async def _run() -> None:
        _reset_environment()
        hass = HomeAssistant()
        entry_id = "entry-acm"
        dev_id = "dev-acm"
        nodes = {"nodes": [{"type": "acm", "addr": "7", "name": "Store"}]}
        settings = {
            "mode": "manual",
            "state": "idle",
            "mtemp": "19.0",
            "stemp": "21.0",
            "ptemp": ["18.0", "19.0", "20.0"],
            "prog": [0, 1, 2] * 56,
            "units": "C",
        }
        coordinator_data = {
            dev_id: {
                "nodes": nodes,
                "nodes_by_type": {
                    "acm": {"addrs": ["7"], "settings": {"7": dict(settings)}}
                },
                "htr": {"settings": {}},
            }
        }
        coordinator = _make_coordinator(
            hass,
            dev_id,
            coordinator_data[dev_id],
            client=AsyncMock(),
            node_inventory=list(
                coordinator_data[dev_id]["nodes_by_type"]["acm"]["settings"].keys()
            ),
        )

        client = AsyncMock()
        client.set_node_settings = AsyncMock()
        client.set_htr_settings = AsyncMock()

        hass.data = {
            DOMAIN: {
                entry_id: {
                    "coordinator": coordinator,
                    "dev_id": dev_id,
                    "client": client,
                    "nodes": nodes,
                    "node_inventory": climate_module.build_node_inventory(nodes),
                }
            }
        }

        added: list[climate_module.HeaterClimateEntity] = []

        def _async_add_entities(
            entities: list[climate_module.HeaterClimateEntity],
        ) -> None:
            added.extend(entities)

        entry = types.SimpleNamespace(entry_id=entry_id)
        await async_setup_entry(hass, entry, _async_add_entities)

        assert len(added) == 1
        acc = added[0]
        assert isinstance(acc, climate_module.AccumulatorClimateEntity)
        assert acc._attr_unique_id == f"{DOMAIN}:{dev_id}:acm:7:climate"
        assert acc.available
        assert acc.device_info["model"] == "Accumulator"

        prog = [0, 1, 2] * 56
        await acc.async_set_schedule(list(prog))
        call = client.set_node_settings.await_args
        assert call.args == (dev_id, ("acm", "7"))
        assert call.kwargs["prog"] == list(prog)
        assert call.kwargs["units"] == "C"
        client.set_node_settings.reset_mock()

        await acc.async_set_preset_temperatures(ptemp=[18.5, 19.5, 20.5])
        call = client.set_node_settings.await_args
        assert call.kwargs["ptemp"] == [18.5, 19.5, 20.5]
        assert call.kwargs["units"] == "C"
        assert client.set_htr_settings.await_count == 0

    asyncio.run(_run())


def test_async_write_settings_without_client_returns_false() -> None:
    async def _run() -> None:
        _reset_environment()
        hass = HomeAssistant()
        dev_id = "dev-missing-client"
        coordinator = _make_coordinator(
            hass,
            dev_id,
            {"htr": {"settings": {}}, "nodes": {}},
        )
        hass.data = {
            DOMAIN: {
                "entry": {
                    "coordinator": coordinator,
                    "dev_id": dev_id,
                    "client": None,
                }
            }
        }

        heater = HeaterClimateEntity(coordinator, "entry", dev_id, "1", "Heater 1")
        heater.hass = hass

        success = await heater._async_write_settings(
            log_context="test", mode="auto", stemp=20.0
        )
        assert success is False

    asyncio.run(_run())


def test_async_setup_entry_rebuilds_inventory_when_missing() -> None:
    async def _run() -> None:
        _reset_environment()
        hass = HomeAssistant()
        entry = types.SimpleNamespace(entry_id="entry-missing")
        dev_id = "dev-missing"
        nodes = {
            "nodes": [
                {"type": "htr", "addr": "11", "name": " First "},
                {"type": "HTR", "addr": "22"},
            ]
        }

        coordinator = _make_coordinator(
            hass,
            dev_id,
            {"nodes": nodes, "htr": {"settings": {"11": {}, "22": {}}}},
            client=AsyncMock(),
        )

        record: dict[str, Any] = {
            "coordinator": coordinator,
            "dev_id": dev_id,
            "client": AsyncMock(),
            "nodes": nodes,
        }
        hass.data = {DOMAIN: {entry.entry_id: record}}

        added: list[HeaterClimateEntity] = []

        def _async_add_entities(entities: list[HeaterClimateEntity]) -> None:
            added.extend(entities)

        await async_setup_entry(hass, entry, _async_add_entities)

        assert len(added) == 2
        stored_inventory = hass.data[DOMAIN][entry.entry_id]["node_inventory"]
        assert [node.addr for node in stored_inventory] == ["11", "22"]

    asyncio.run(_run())


def test_refresh_fallback_skips_when_hass_inactive(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _run() -> None:
        _reset_environment()

        hass = HomeAssistant()
        hass.is_stopping = True
        hass.is_running = True
        entry_id = "entry"
        dev_id = "dev"
        addr = "A"
        coordinator = _make_coordinator(
            hass,
            dev_id,
            {"nodes": {"nodes": []}, "htr": {"settings": {addr: {}}}},
            client=AsyncMock(),
        )

        hass.data = {
            DOMAIN: {
                entry_id: {
                    "client": AsyncMock(),
                    "coordinator": coordinator,
                    "dev_id": dev_id,
                    "version": "1",
                    "ws_state": {},
                }
            }
        }

        heater = HeaterClimateEntity(coordinator, entry_id, dev_id, addr, "Heater")
        await heater.async_added_to_hass()

        async def fast_sleep(_delay: float) -> None:
            return None

        monkeypatch.setattr(climate_module.asyncio, "sleep", fast_sleep)

        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            heater._schedule_refresh_fallback()
            task = heater._refresh_fallback
            assert task is not None
            await task
        coordinator.async_refresh_heater.assert_not_awaited()
        assert heater._refresh_fallback is None
        assert "hass stopping" in caplog.text

        hass.is_stopping = False
        hass.is_running = False
        coordinator.async_refresh_heater.reset_mock()

        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            heater._schedule_refresh_fallback()
            task = heater._refresh_fallback
            assert task is not None
            await task
        coordinator.async_refresh_heater.assert_not_awaited()
        assert heater._refresh_fallback is None
        assert "hass not running" in caplog.text

    asyncio.run(_run())


def test_heater_additional_cancelled_edges(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _run() -> None:
        _reset_environment()
        hass = HomeAssistant()
        entry_id = "entry"
        dev_id = "dev"
        addr = "A"
        base_prog: list[int] = [0, 1, 2] * 56
        settings = {
            "mode": "manual",
            "state": "heating",
            "mtemp": "19.0",
            "stemp": "21.0",
            "ptemp": ["18.0", "19.0", "20.0"],
            "prog": list(base_prog),
            "units": "C",
        }
        coordinator = _make_coordinator(
            hass,
            dev_id,
            {"nodes": {"nodes": []}, "htr": {"settings": {addr: settings}}},
            client=AsyncMock(),
        )
        client = AsyncMock()
        client.set_htr_settings = AsyncMock()

        hass.data = {
            DOMAIN: {
                entry_id: {
                    "client": client,
                    "coordinator": coordinator,
                    "dev_id": dev_id,
                    "ws_state": {},
                    "version": "1",
                }
            }
        }

        heater = HeaterClimateEntity(coordinator, entry_id, dev_id, addr, "Heater")
        await heater.async_added_to_hass()

        class SentinelCancelled(Exception):
            pass

        monkeypatch.setattr(climate_module.asyncio, "CancelledError", SentinelCancelled)

        async def fast_sleep(_delay: float) -> None:
            return None

        monkeypatch.setattr(climate_module.asyncio, "sleep", fast_sleep)

        orig_float = climate_module.float_or_none

        def raising_float(_value: Any) -> float | None:
            raise SentinelCancelled()

        climate_module.float_or_none = raising_float
        with pytest.raises(SentinelCancelled):
            _ = heater.extra_state_attributes
        climate_module.float_or_none = orig_float

        prog = list(base_prog)
        orig_write = heater.async_write_ha_state

        def raising_write() -> None:
            raise SentinelCancelled()

        heater.async_write_ha_state = raising_write
        with pytest.raises(SentinelCancelled):
            await heater.async_set_schedule(prog)
        heater.async_write_ha_state = orig_write

        caplog.clear()
        with caplog.at_level(logging.ERROR):
            await heater.async_set_preset_temperatures()
        assert "Preset temperatures require" in caplog.text

        heater.async_write_ha_state = raising_write
        with pytest.raises(SentinelCancelled):
            await heater.async_set_preset_temperatures(ptemp=[18.0, 19.0, 20.0])
        heater.async_write_ha_state = orig_write

        client.set_htr_settings.side_effect = SentinelCancelled()
        heater._pending_mode = HVACMode.AUTO
        heater._pending_stemp = 20.5
        with pytest.raises(SentinelCancelled):
            await heater._write_after_debounce()
        client.set_htr_settings.side_effect = None

        class BadFloat:
            def __float__(self) -> float:
                raise SentinelCancelled()

        heater._pending_mode = HVACMode.HEAT
        heater._pending_stemp = BadFloat()
        with pytest.raises(SentinelCancelled):
            await heater._write_after_debounce()

        coordinator.async_refresh_heater = AsyncMock(
            side_effect=SentinelCancelled()
        )
        heater._refresh_fallback = None
        heater._schedule_refresh_fallback()
        assert heater._refresh_fallback is not None
        with pytest.raises(SentinelCancelled):
            await heater._refresh_fallback

    asyncio.run(_run())


def test_heater_properties_and_ws_update() -> None:
    async def _run() -> None:
        _reset_environment()
        hass = HomeAssistant()
        entry_id = "entry"
        dev_id = "dev1"
        addr = "A1"
        prog: list[int] = [2] + [1] * 167
        settings = {
            "mode": "manual",
            "state": "heating",
            "mtemp": "19.5",
            "stemp": "21.0",
            "ptemp": ["15.0", "18.0", "21.0"],
            "prog": prog,
            "units": "C",
            "max_power": 1200,
        }
        coordinator_data = {
            dev_id: {
                "nodes": {"nodes": []},
                "htr": {"settings": {addr: settings}},
                "version": "2.0.0",
            }
        }
        coordinator = _make_coordinator(
            hass,
            dev_id,
            coordinator_data[dev_id],
            client=AsyncMock(),
        )
        hass.data = {
            DOMAIN: {
                entry_id: {
                    "client": AsyncMock(),
                    "coordinator": coordinator,
                    "dev_id": dev_id,
                    "version": "2.0.0",
                }
            }
        }

        dt_util.NOW = dt.datetime(2024, 1, 1, 0, 0, tzinfo=dt.timezone.utc)

        heater = HeaterClimateEntity(coordinator, entry_id, dev_id, addr, "Living")
        assert heater.hass is hass

        await heater.async_added_to_hass()

        info = heater.device_info
        assert info["identifiers"] == {(DOMAIN, dev_id, addr)}
        assert info["manufacturer"] == "TermoWeb"
        assert info["model"] == "Heater"
        assert info["via_device"] == (DOMAIN, dev_id)

        assert heater.should_poll is False
        assert heater.available is True
        assert heater.hvac_mode == HVACMode.HEAT
        assert heater.hvac_action == HVACAction.HEATING
        assert heater.current_temperature == pytest.approx(19.5)
        assert heater.target_temperature == pytest.approx(21.0)
        assert heater.min_temp == 5.0
        assert heater.max_temp == 30.0
        assert heater.icon == "mdi:radiator"

        attrs = heater.extra_state_attributes
        assert attrs["dev_id"] == dev_id
        assert attrs["addr"] == addr
        assert attrs["units"] == "C"
        assert attrs["max_power"] == 1200
        assert attrs["ptemp"] == ["15.0", "18.0", "21.0"]
        assert attrs["prog"] == prog
        assert attrs["program_slot"] == "day"
        assert attrs["program_setpoint"] == pytest.approx(21.0)

        assert heater._unsub_ws is not None
        heater.schedule_update_ha_state = MagicMock()
        heater._handle_ws_message({"dev_id": dev_id, "addr": addr})
        heater.schedule_update_ha_state.assert_called_once()

        heater.schedule_update_ha_state.reset_mock()
        heater._handle_ws_message({"dev_id": "other", "addr": addr})
        heater._handle_ws_message({"dev_id": dev_id, "addr": "B2"})
        heater.schedule_update_ha_state.assert_not_called()

        heater._handle_ws_message({"dev_id": dev_id})
        heater.schedule_update_ha_state.assert_called_once()

        heater.schedule_update_ha_state.reset_mock()
        heater._handle_ws_message({"dev_id": dev_id, "addr": addr})
        heater.schedule_update_ha_state.assert_called_once()

        async def _pending() -> None:
            await asyncio.sleep(3600)

        task = asyncio.create_task(_pending())
        heater._refresh_fallback = task
        heater._handle_ws_message(
            {"dev_id": dev_id, "addr": addr, "kind": "htr_settings"}
        )
        await asyncio.sleep(0)
        assert task.cancelled()
        assert heater._refresh_fallback is None
        with pytest.raises(asyncio.CancelledError):
            await task

        original_now = dt_util.NOW
        try:
            dt_util.NOW = dt.datetime(2024, 1, 1, 1, 0, tzinfo=dt.timezone.utc)
            attrs = heater.extra_state_attributes
            assert attrs["program_slot"] == "night"
            assert attrs["program_setpoint"] == pytest.approx(18.0)
        finally:
            dt_util.NOW = original_now

        coordinator.data[dev_id]["nodes"] = None
        assert heater.available is False
        coordinator.data[dev_id]["nodes"] = {"nodes": []}
        assert heater.available is True

        settings["mode"] = "auto"
        settings["state"] = "idle"
        assert heater.hvac_mode == HVACMode.AUTO
        assert heater.hvac_action == HVACAction.IDLE
        assert heater.icon == "mdi:radiator-disabled"

        settings["mode"] = "off"
        settings["state"] = "idle"
        settings["prog"] = [1]
        settings["ptemp"] = None
        attrs = heater.extra_state_attributes
        assert heater.hvac_mode == HVACMode.OFF
        assert heater.hvac_action == HVACAction.OFF
        assert heater.icon == "mdi:radiator-off"
        assert "program_slot" not in attrs
        assert "program_setpoint" not in attrs

        settings["ptemp"] = ["15.0", "18.0", "21.0"]
        settings["prog"] = ["bad"] + [1] * 167
        settings["mode"] = "eco"
        settings["state"] = "heating"
        attrs = heater.extra_state_attributes
        assert "program_slot" not in attrs
        assert "program_setpoint" not in attrs
        assert heater.hvac_mode == HVACMode.HEAT
        assert heater.icon == "mdi:radiator"

        settings["state"] = None
        assert heater.hvac_action is None
        assert heater.icon == "mdi:radiator"

        settings["mtemp"] = "invalid"
        settings["stemp"] = ""
        assert heater.current_temperature is None
        assert heater.target_temperature is None

    asyncio.run(_run())


def test_heater_write_paths_and_errors(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def _run() -> None:
        _reset_environment()
        from homeassistant.const import ATTR_TEMPERATURE
        hass = HomeAssistant()
        entry_id = "entry"
        dev_id = "dev1"
        addr = "A1"
        base_prog: list[int] = [0, 1, 2] * 56
        settings = {
            "mode": "manual",
            "state": "heating",
            "mtemp": "19.5",
            "stemp": "21.0",
            "ptemp": ["15.0", "18.0", "21.0"],
            "prog": list(base_prog),
            "units": "C",
            "max_power": 1200,
        }
        coordinator_data = {
            dev_id: {
                "nodes": {"nodes": []},
                "htr": {"settings": {addr: settings}},
                "version": "5.0.0",
            }
        }

        coordinator = _make_coordinator(
            hass,
            dev_id,
            coordinator_data[dev_id],
            client=AsyncMock(),
        )
        client = AsyncMock()
        hass.data = {
            DOMAIN: {
                entry_id: {
                    "client": client,
                    "coordinator": coordinator,
                    "dev_id": dev_id,
                    "version": "5.0.0",
                    "ws_state": {
                        dev_id: {"status": "disconnected", "last_event_at": None}
                    },
                }
            }
        }

        heater = HeaterClimateEntity(coordinator, entry_id, dev_id, addr, "Heater")
        await heater.async_added_to_hass()

        fallback_waiters: Deque[asyncio.Future[None]] = deque()
        write_waiters: Deque[asyncio.Future[None]] = deque()
        write_block = False
        real_sleep = asyncio.sleep

        async def fake_sleep(delay: float) -> None:
            if delay == climate_module._WRITE_DEBOUNCE:
                if write_block:
                    loop = asyncio.get_running_loop()
                    fut: asyncio.Future[None] = loop.create_future()
                    write_waiters.append(fut)
                    await fut
                    return None
                return None
            if delay == climate_module._WS_ECHO_FALLBACK_REFRESH:
                loop = asyncio.get_running_loop()
                fut: asyncio.Future[None] = loop.create_future()
                fallback_waiters.append(fut)
                await fut
                return None
            await real_sleep(delay)
            return None

        real_create_task = asyncio.create_task
        created_tasks: list[asyncio.Task[Any]] = []

        def track_create_task(
            coro: Coroutine[Any, Any, Any], *, name: str | None = None
        ) -> asyncio.Task[Any]:
            task = real_create_task(coro, name=name)
            created_tasks.append(task)
            return task

        async def _pop_waiter() -> asyncio.Future[None]:
            for _ in range(10):
                if fallback_waiters:
                    return fallback_waiters.popleft()
                await real_sleep(0)
            raise AssertionError("fallback waiter not created")

        async def _pop_write_waiter() -> asyncio.Future[None]:
            for _ in range(10):
                if write_waiters:
                    return write_waiters.popleft()
                await real_sleep(0)
            raise AssertionError("write waiter not created")

        async def _complete_fallback_once() -> None:
            waiter = await _pop_waiter()
            task = heater._refresh_fallback
            assert task is not None
            assert coordinator.async_refresh_heater.await_count == 0
            waiter.set_result(None)
            await task
            coordinator.async_refresh_heater.assert_awaited_once_with(("htr", addr))
            coordinator.async_refresh_heater.reset_mock()
            assert heater._refresh_fallback is None

        class RaisingMapping:
            def __init__(self, real: dict[str, Any]) -> None:
                self._real = real
                self._calls = 0

            def get(self, *args: Any, **kwargs: Any) -> Any:
                self._calls += 1
                if self._calls >= 2:
                    raise RuntimeError("boom mapping")
                return self._real.get(*args, **kwargs)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)
        monkeypatch.setattr(asyncio, "create_task", track_create_task)

        caplog.set_level(logging.DEBUG)

        # -------------------- async_set_schedule (valid) --------------------
        await heater.async_set_schedule(list(base_prog))
        call = client.set_htr_settings.await_args
        assert call.args == (dev_id, addr)
        assert call.kwargs["prog"] == list(base_prog)
        assert call.kwargs["units"] == "C"

        settings_after = coordinator.data[dev_id]["htr"]["settings"][addr]
        assert settings_after["prog"] == list(base_prog)

        assert heater._refresh_fallback is not None
        await _complete_fallback_once()

        client.set_htr_settings.reset_mock()

        # -------------------- async_set_schedule (invalid length/value) -----
        caplog.clear()
        await heater.async_set_schedule([0, 1])
        assert client.set_htr_settings.await_count == 0
        assert "Invalid prog length" in caplog.text
        assert not fallback_waiters

        caplog.clear()
        client.set_htr_settings.reset_mock()
        bad_prog = list(base_prog)
        bad_prog[5] = 7
        await heater.async_set_schedule(bad_prog)
        assert client.set_htr_settings.await_count == 0
        assert "Invalid prog for dev" in caplog.text
        assert not fallback_waiters

        # -------------------- async_set_schedule (API error) ----------------
        caplog.clear()
        client.set_htr_settings.reset_mock()
        client.set_htr_settings.side_effect = RuntimeError("boom schedule")
        prev_prog = list(settings_after["prog"])
        prev_fallback = heater._refresh_fallback
        await heater.async_set_schedule(list(base_prog))
        assert client.set_htr_settings.await_count == 1
        assert settings_after["prog"] == prev_prog
        assert heater._refresh_fallback is prev_fallback
        assert not fallback_waiters
        assert "Schedule write failed" in caplog.text
        client.set_htr_settings.side_effect = None
        client.set_htr_settings.reset_mock()

        # -------------------- async_set_schedule (optimistic failure) -------
        caplog.clear()
        client.set_htr_settings.reset_mock()
        old_data = coordinator.data
        coordinator.data = RaisingMapping(old_data)
        await heater.async_set_schedule(list(base_prog))
        assert client.set_htr_settings.await_count == 1
        assert settings_after["prog"] == prev_prog
        assert "Optimistic update failed" in caplog.text
        waiter = await _pop_waiter()
        task = heater._refresh_fallback
        assert task is not None
        coordinator.data = old_data
        waiter.set_result(None)
        await task
        coordinator.async_refresh_heater.assert_awaited_once_with(("htr", addr))
        coordinator.async_refresh_heater.reset_mock()
        assert heater._refresh_fallback is None
        client.set_htr_settings.reset_mock()

        # -------------------- async_set_preset_temperatures (valid forms) ---
        caplog.clear()
        preset_payload = [18.5, 19.5, 20.5]
        await heater.async_set_preset_temperatures(ptemp=preset_payload)
        call = client.set_htr_settings.await_args
        assert call.args == (dev_id, addr)
        assert call.kwargs["ptemp"] == preset_payload
        assert call.kwargs["units"] == "C"
        assert settings_after["ptemp"] == ["18.5", "19.5", "20.5"]
        await _complete_fallback_once()
        client.set_htr_settings.reset_mock()

        caplog.clear()
        await heater.async_set_preset_temperatures(cold=16.5, night=17.5, day=18.5)
        call = client.set_htr_settings.await_args
        assert call.kwargs["ptemp"] == [16.5, 17.5, 18.5]
        assert settings_after["ptemp"] == ["16.5", "17.5", "18.5"]
        await _complete_fallback_once()
        client.set_htr_settings.reset_mock()

        # -------------------- async_set_preset_temperatures (invalid) -------
        caplog.clear()
        await heater.async_set_preset_temperatures(ptemp=[18.0, 19.0])
        assert client.set_htr_settings.await_count == 0
        assert "Invalid ptemp length" in caplog.text
        assert not fallback_waiters

        caplog.clear()
        client.set_htr_settings.reset_mock()
        await heater.async_set_preset_temperatures(ptemp=["bad", "bad", "bad"])
        assert client.set_htr_settings.await_count == 0
        assert "Invalid ptemp values" in caplog.text
        assert not fallback_waiters

        # -------------------- async_set_preset_temperatures (API error) -----
        caplog.clear()
        client.set_htr_settings.reset_mock()
        client.set_htr_settings.side_effect = RuntimeError("boom preset")
        prev_ptemp = list(settings_after["ptemp"])
        prev_fallback = heater._refresh_fallback
        await heater.async_set_preset_temperatures(ptemp=[19.1, 20.1, 21.1])
        assert client.set_htr_settings.await_count == 1
        assert settings_after["ptemp"] == prev_ptemp
        assert heater._refresh_fallback is prev_fallback
        assert not fallback_waiters
        assert "Preset write failed" in caplog.text
        client.set_htr_settings.side_effect = None
        client.set_htr_settings.reset_mock()

        # -------------------- async_set_preset_temperatures (optimistic failure) -
        caplog.clear()
        client.set_htr_settings.reset_mock()
        old_data = coordinator.data
        coordinator.data = RaisingMapping(old_data)
        await heater.async_set_preset_temperatures(ptemp=[19.2, 20.2, 21.2])
        assert client.set_htr_settings.await_count == 1
        assert settings_after["ptemp"] == prev_ptemp
        assert "Optimistic update failed" in caplog.text
        waiter = await _pop_waiter()
        task = heater._refresh_fallback
        assert task is not None
        coordinator.data = old_data
        waiter.set_result(None)
        await task
        coordinator.async_refresh_heater.assert_awaited_once_with(("htr", addr))
        coordinator.async_refresh_heater.reset_mock()
        assert heater._refresh_fallback is None
        client.set_htr_settings.reset_mock()

        # -------------------- async_set_temperature (valid + clamps) -------
        client.set_htr_settings.reset_mock()
        await heater.async_set_temperature(**{ATTR_TEMPERATURE: 35.6})
        assert heater._write_task is not None
        await heater._write_task
        call = client.set_htr_settings.await_args
        assert call.args == (dev_id, addr)
        assert call.kwargs["mode"] == "manual"
        assert call.kwargs["stemp"] == pytest.approx(30.0)
        assert call.kwargs["units"] == "C"
        assert settings_after["stemp"] == "30.0"
        await _complete_fallback_once()
        client.set_htr_settings.reset_mock()

        await heater.async_set_temperature(**{ATTR_TEMPERATURE: 2.0})
        assert heater._write_task is not None
        await heater._write_task
        call = client.set_htr_settings.await_args
        assert call.kwargs["stemp"] == pytest.approx(5.0)
        assert settings_after["stemp"] == "5.0"
        await _complete_fallback_once()
        client.set_htr_settings.reset_mock()

        caplog.clear()
        await heater.async_set_temperature(**{ATTR_TEMPERATURE: "bad"})
        assert client.set_htr_settings.await_count == 0
        assert (heater._write_task is None) or heater._write_task.done()
        assert "Invalid temperature payload" in caplog.text
        assert not fallback_waiters

        # -------------------- async_set_hvac_mode --------------------------
        client.set_htr_settings.reset_mock()
        await heater.async_set_hvac_mode(HVACMode.AUTO)
        assert heater._write_task is not None
        await heater._write_task
        call = client.set_htr_settings.await_args
        assert call.kwargs["mode"] == "auto"
        assert call.kwargs["stemp"] is None
        assert settings_after["mode"] == "auto"
        assert settings_after["stemp"] == "5.0"
        await _complete_fallback_once()
        client.set_htr_settings.reset_mock()

        await heater.async_set_hvac_mode(HVACMode.OFF)
        assert heater._write_task is not None
        await heater._write_task
        call = client.set_htr_settings.await_args
        assert call.kwargs["mode"] == "off"
        assert settings_after["mode"] == "off"
        assert settings_after["stemp"] == "5.0"
        await _complete_fallback_once()
        client.set_htr_settings.reset_mock()

        await heater.async_set_hvac_mode(HVACMode.HEAT)
        assert heater._write_task is not None
        await heater._write_task
        call = client.set_htr_settings.await_args
        assert call.kwargs["mode"] == "manual"
        assert call.kwargs["stemp"] == pytest.approx(5.0)
        assert settings_after["mode"] == "manual"
        assert settings_after["stemp"] == "5.0"
        await _complete_fallback_once()
        client.set_htr_settings.reset_mock()

        caplog.clear()
        await heater.async_set_hvac_mode(cast(HVACMode, "eco"))
        assert client.set_htr_settings.await_count == 0
        assert "Unsupported hvac_mode" in caplog.text
        assert not fallback_waiters

        # -------------------- _ensure_write_task and debounce -------------
        client.set_htr_settings.reset_mock()
        write_block = True
        pre_fallback = heater._refresh_fallback
        heater._pending_mode = None
        heater._pending_stemp = None
        await heater._ensure_write_task()
        first_task = heater._write_task
        assert first_task is not None
        await heater._ensure_write_task()
        assert heater._write_task is first_task
        write_waiter = await _pop_write_waiter()
        write_block = False
        write_waiter.set_result(None)
        await first_task
        assert client.set_htr_settings.await_count == 0
        assert heater._refresh_fallback is pre_fallback
        assert not write_waiters

        # -------------------- _write_after_debounce error path -------------
        caplog.clear()
        client.set_htr_settings.reset_mock()
        client.set_htr_settings.side_effect = RuntimeError("write boom")
        heater._pending_mode = HVACMode.AUTO
        heater._pending_stemp = None
        heater._refresh_fallback = None
        await heater._ensure_write_task()
        assert heater._write_task is not None
        await heater._write_task
        assert "Mode/setpoint write failed" in caplog.text
        assert heater._refresh_fallback is None
        assert not fallback_waiters
        client.set_htr_settings.side_effect = None
        client.set_htr_settings.reset_mock()

        # -------------------- _schedule_refresh_fallback behaviour --------
        heater._schedule_refresh_fallback()
        task_a = heater._refresh_fallback
        waiter_a = await _pop_waiter()
        heater._schedule_refresh_fallback()
        task_b = heater._refresh_fallback
        waiter_b = await _pop_waiter()

        assert task_a is not None and task_b is not None and task_a is not task_b
        with pytest.raises(asyncio.CancelledError):
            await task_a
        if not waiter_a.done():
            waiter_a.cancel()

        assert coordinator.async_refresh_heater.await_count == 0

        waiter_b.set_result(None)
        await task_b
        coordinator.async_refresh_heater.assert_awaited_once_with(("htr", addr))
        coordinator.async_refresh_heater.reset_mock()

        caplog.clear()
        coordinator.async_refresh_heater.side_effect = RuntimeError("refresh boom")
        heater._schedule_refresh_fallback()
        waiter_err = await _pop_waiter()
        task_err = heater._refresh_fallback
        assert task_err is not None
        waiter_err.set_result(None)
        await task_err
        assert "Refresh fallback failed" in caplog.text
        coordinator.async_refresh_heater.side_effect = None
        coordinator.async_refresh_heater.reset_mock()
        assert not fallback_waiters

        # -------------------- WS healthy suppresses fallback --------------
        hass.data[DOMAIN][entry_id]["ws_state"][dev_id] = {
            "status": "healthy",
            "last_event_at": time.time(),
        }
        client.set_htr_settings.reset_mock()
        await heater.async_set_temperature(**{ATTR_TEMPERATURE: 22.5})
        assert heater._write_task is not None
        await heater._write_task
        assert client.set_htr_settings.await_count == 1
        assert heater._refresh_fallback is None
        assert not fallback_waiters
        client.set_htr_settings.reset_mock()

        # -------------------- WS down restores fallback -------------------
        hass.data[DOMAIN][entry_id]["ws_state"][dev_id] = {
            "status": "disconnected",
            "last_event_at": None,
        }
        await heater.async_set_temperature(**{ATTR_TEMPERATURE: 23.5})
        assert heater._write_task is not None
        await heater._write_task
        assert heater._refresh_fallback is not None
        await _complete_fallback_once()
        client.set_htr_settings.reset_mock()

        assert created_tasks, "Expected background tasks to be created"

    asyncio.run(_run())

def test_heater_cancellation_and_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        _reset_environment()
        hass = HomeAssistant()
        entry_id = "entry"
        dev_id = "dev"
        addr = "A"
        base_prog: list[int] = [0, 1, 2] * 56
        settings = {
            "mode": "manual",
            "state": "heating",
            "mtemp": "19.0",
            "stemp": "21.0",
            "ptemp": ["18.0", "19.0", "20.0"],
            "prog": list(base_prog),
            "units": "C",
        }
        coordinator = _make_coordinator(
            hass,
            dev_id,
            {"nodes": {"nodes": []}, "htr": {"settings": {addr: settings}}},
            client=AsyncMock(),
        )
        client = AsyncMock()
        client.set_htr_settings = AsyncMock()

        hass.data = {
            DOMAIN: {
                entry_id: {
                    "client": client,
                    "coordinator": coordinator,
                    "dev_id": dev_id,
                    "version": "1",
                    "ws_state": {},
                }
            }
        }

        heater = HeaterClimateEntity(coordinator, entry_id, dev_id, addr, "Heater")
        await heater.async_added_to_hass()
        orig_cancelled = climate_module.asyncio.CancelledError

        class CancelList(list):
            def __getitem__(self, idx):
                raise ValueError("cancel slot")

        monkeypatch.setattr(climate_module.asyncio, "CancelledError", ValueError)
        settings["prog"] = CancelList(list(base_prog))
        with pytest.raises(ValueError):
            heater._current_prog_slot(settings)
        settings["prog"] = list(base_prog)

        class BadPTList(list):
            def __getitem__(self, idx):
                raise RuntimeError("bad ptemp")

        settings["ptemp"] = BadPTList(["18", "19", "20"])
        attrs = heater.extra_state_attributes
        assert "program_setpoint" not in attrs
        settings["ptemp"] = ["18.0", "19.0", "20.0"]

        class CancelInt(int):
            def __int__(self) -> int:
                raise ValueError("cancel prog")

        prog_cancel = list(base_prog)
        prog_cancel[0] = CancelInt(0)
        with pytest.raises(ValueError):
            await heater.async_set_schedule(prog_cancel)

        client.set_htr_settings.reset_mock()
        client.set_htr_settings.side_effect = ValueError("api cancel")
        with pytest.raises(ValueError):
            await heater.async_set_schedule(list(base_prog))
        client.set_htr_settings.side_effect = None

        class CancelMapping(dict):
            def __init__(self, real: dict[str, Any]) -> None:
                super().__init__(real)

            def get(self, *_args: Any, **_kwargs: Any) -> Any:
                raise ValueError("optimistic cancel")

        original_data = coordinator.data
        coordinator.data = CancelMapping(original_data)
        with pytest.raises(ValueError):
            await heater.async_set_schedule(list(base_prog))
        coordinator.data = original_data

        monkeypatch.setattr(climate_module.asyncio, "CancelledError", KeyError)
        with pytest.raises(KeyError):
            await heater.async_set_preset_temperatures()

        monkeypatch.setattr(climate_module.asyncio, "CancelledError", ValueError)

        class CancelFloat:
            def __float__(self) -> float:
                raise ValueError("cancel float")

        with pytest.raises(ValueError):
            await heater.async_set_preset_temperatures(ptemp=[CancelFloat(), 19.0, 20.0])

        client.set_htr_settings.reset_mock()
        client.set_htr_settings.side_effect = ValueError("preset cancel")
        with pytest.raises(ValueError):
            await heater.async_set_preset_temperatures(ptemp=[18.0, 19.0, 20.0])
        client.set_htr_settings.side_effect = None

        coordinator.data = CancelMapping(original_data)
        with pytest.raises(ValueError):
            await heater.async_set_preset_temperatures(ptemp=[18.0, 19.0, 20.0])
        coordinator.data = original_data

        monkeypatch.setattr(climate_module.asyncio, "CancelledError", orig_cancelled)

        async def fast_sleep(_delay: float) -> None:
            return None

        monkeypatch.setattr(climate_module.asyncio, "sleep", fast_sleep)

        client.set_htr_settings.reset_mock()
        heater._pending_mode = None
        heater._pending_stemp = 22.0
        await heater._write_after_debounce()
        call = client.set_htr_settings.await_args
        assert call.kwargs["mode"] == "manual"
        assert call.kwargs["stemp"] == 22.0

        client.set_htr_settings.reset_mock()
        heater._pending_mode = HVACMode.HEAT
        heater._pending_stemp = None
        await heater._write_after_debounce()
        call = client.set_htr_settings.await_args
        assert call.kwargs["mode"] == "manual"
        assert call.kwargs["stemp"] == 22.0

        client.set_htr_settings.reset_mock()
        client.set_htr_settings.side_effect = ValueError("write cancel")
        heater._pending_mode = HVACMode.AUTO
        heater._pending_stemp = 19.5
        await heater._write_after_debounce()
        call = client.set_htr_settings.await_args
        assert call.kwargs["mode"] == "manual"
        assert call.kwargs["stemp"] == 19.5
        client.set_htr_settings.side_effect = None

        class BadFloat:
            def __float__(self) -> float:
                raise RuntimeError("bad float")

        heater._pending_mode = HVACMode.HEAT
        heater._pending_stemp = BadFloat()
        await heater._write_after_debounce()

        writer = MagicMock(side_effect=ValueError("optimistic fail"))
        monkeypatch.setattr(heater, "async_write_ha_state", writer)
        heater._pending_mode = HVACMode.AUTO
        heater._pending_stemp = 20.0
        await heater._write_after_debounce()
        assert writer.call_count == 1

        async def failing_refresh() -> None:
            raise ValueError("fallback cancel")

        coordinator.async_refresh_heater = AsyncMock(side_effect=failing_refresh)
        heater._schedule_refresh_fallback()
        task = heater._refresh_fallback
        assert task is not None
        await task
        coordinator.async_refresh_heater.assert_awaited_once_with(("htr", addr))
        assert heater._refresh_fallback is None

    asyncio.run(_run())


def test_heater_cancelled_paths_propagate(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _run() -> None:
        _reset_environment()
        from homeassistant.components.climate import HVACMode

        hass = HomeAssistant()
        entry_id = "entry"
        dev_id = "dev"
        addr = "A"
        base_prog: list[int] = [0, 1, 2] * 56
        settings = {
            "mode": "manual",
            "state": "heating",
            "mtemp": "19.0",
            "stemp": "21.0",
            "ptemp": ["18.0", "19.0", "20.0"],
            "prog": list(base_prog),
            "units": "C",
        }
        coordinator = _make_coordinator(
            hass,
            dev_id,
            {"nodes": {"nodes": []}, "htr": {"settings": {addr: settings}}},
            client=AsyncMock(),
        )
        client = AsyncMock()
        client.set_htr_settings = AsyncMock()

        hass.data = {
            DOMAIN: {
                entry_id: {
                    "client": client,
                    "coordinator": coordinator,
                    "dev_id": dev_id,
                    "ws_state": {},
                    "version": "1",
                }
            }
        }

        heater = HeaterClimateEntity(coordinator, entry_id, dev_id, addr, "Heater")
        await heater.async_added_to_hass()

        orig_float = climate_module.float_or_none

        def raising_float(_value: Any) -> float | None:
            raise asyncio.CancelledError()

        monkeypatch.setattr(climate_module, "float_or_none", raising_float)
        with pytest.raises(asyncio.CancelledError):
            _ = heater.extra_state_attributes
        monkeypatch.setattr(climate_module, "float_or_none", orig_float)

        class CancelInt(int):
            def __int__(self) -> int:
                raise asyncio.CancelledError()

        prog_cancel = list(base_prog)
        prog_cancel[0] = CancelInt(0)
        with pytest.raises(asyncio.CancelledError):
            await heater.async_set_schedule(prog_cancel)

        caplog.clear()
        with caplog.at_level(logging.ERROR):
            await heater.async_set_preset_temperatures()
        assert "Preset temperatures require" in caplog.text

        class CancelSettings(dict):
            def get(self, *_args: Any, **_kwargs: Any) -> Any:
                raise asyncio.CancelledError()

        original_settings = coordinator.data[dev_id]["htr"]["settings"]
        coordinator.data[dev_id]["htr"]["settings"] = CancelSettings(original_settings)
        with pytest.raises(asyncio.CancelledError):
            await heater.async_set_preset_temperatures(ptemp=[18.0, 19.0, 20.0])
        coordinator.data[dev_id]["htr"]["settings"] = original_settings

        client.set_htr_settings.side_effect = asyncio.CancelledError()
        heater._pending_mode = HVACMode.AUTO
        heater._pending_stemp = 21.0
        with pytest.raises(asyncio.CancelledError):
            await heater._write_after_debounce()

        client.set_htr_settings.side_effect = None

        class CancelFloat:
            def __float__(self) -> float:
                raise asyncio.CancelledError()

        heater._pending_mode = HVACMode.HEAT
        heater._pending_stemp = CancelFloat()
        with pytest.raises(asyncio.CancelledError):
            await heater._write_after_debounce()

        class CancelMapping(dict):
            def get(self, *_args: Any, **_kwargs: Any) -> Any:
                raise asyncio.CancelledError()

        original_data = coordinator.data
        coordinator.data = CancelMapping(original_data)
        heater._pending_mode = HVACMode.AUTO
        heater._pending_stemp = 20.0
        with pytest.raises(asyncio.CancelledError):
            await heater._write_after_debounce()
        coordinator.data = original_data

        async def fast_sleep(_delay: float) -> None:
            return None

        monkeypatch.setattr(climate_module.asyncio, "sleep", fast_sleep)
        coordinator.async_refresh_heater = AsyncMock(
            side_effect=asyncio.CancelledError()
        )
        heater._schedule_refresh_fallback()
        task = heater._refresh_fallback
        assert task is not None
        with pytest.raises(asyncio.CancelledError):
            await task
        assert heater._refresh_fallback is None

    asyncio.run(_run())
