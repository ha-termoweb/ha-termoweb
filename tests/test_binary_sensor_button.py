# ruff: noqa: D100, D101, D102, D103, D105, D107, INP001
from __future__ import annotations

import asyncio
import logging
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from homeassistant.components.button import ButtonEntity
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_send

import custom_components.termoweb.binary_sensor as binary_sensor_module
import custom_components.termoweb.button as button_module
from custom_components.termoweb.const import DOMAIN, signal_ws_status
from custom_components.termoweb.utils import build_gateway_device_info

GatewayOnlineBinarySensor = binary_sensor_module.GatewayOnlineBinarySensor
async_setup_binary_sensor_entry = binary_sensor_module.async_setup_entry
StateRefreshButton = button_module.StateRefreshButton
async_setup_button_entry = button_module.async_setup_entry
AccumulatorBoostButton = button_module.AccumulatorBoostButton
AccumulatorBoostCancelButton = button_module.AccumulatorBoostCancelButton


def test_binary_sensor_setup_and_dispatch() -> None:
    async def _run() -> None:
        hass = HomeAssistant()
        entry = types.SimpleNamespace(entry_id="entry-1")
        dev_id = "device-123"

        coordinator = types.SimpleNamespace(
            hass=hass,
            data={
                dev_id: {
                    "name": "Living Room",  # attributes
                    "connected": True,
                    "raw": {"model": "TW-GW"},
                }
            },
        )

        hass.data = {
            DOMAIN: {
                entry.entry_id: {
                    "coordinator": coordinator,
                    "dev_id": dev_id,
                    "version": "2.1.0",
                    "ws_state": {
                        dev_id: {
                            "status": "healthy",
                            "last_event_at": "2024-05-01T12:00:00Z",
                            "healthy_minutes": 42,
                        }
                    },
                }
            }
        }

        added: list = []

        def _add_entities(entities):
            added.extend(entities)

        guard_coordinator = types.SimpleNamespace(hass=None, data={})
        guard_entity = GatewayOnlineBinarySensor(
            guard_coordinator,
            "guard-entry",
            "guard-device",
        )
        await guard_entity.async_added_to_hass()
        assert not guard_entity._ws_subscription.is_connected  # pylint: disable=protected-access

        await async_setup_binary_sensor_entry(hass, entry, _add_entities)

        assert len(added) == 1
        entity = added[0]
        assert isinstance(entity, GatewayOnlineBinarySensor)

        entity.hass = hass
        await entity.async_added_to_hass()

        assert entity.is_on is True
        assert entity._ws_subscription.is_connected  # pylint: disable=protected-access

        info = entity.device_info
        expected_info = build_gateway_device_info(hass, entry.entry_id, dev_id)
        assert info == expected_info

        attrs = entity.extra_state_attributes
        assert attrs == {
            "dev_id": dev_id,
            "name": "Living Room",
            "connected": True,
            "ws_status": "healthy",
            "ws_last_event_at": "2024-05-01T12:00:00Z",
            "ws_healthy_minutes": 42,
            "raw": {"model": "TW-GW"},
        }

        entity.schedule_update_ha_state = MagicMock()
        async_dispatcher_send(
            hass, signal_ws_status(entry.entry_id), {"dev_id": "other"}
        )
        entity.schedule_update_ha_state.assert_not_called()
        async_dispatcher_send(
            hass, signal_ws_status(entry.entry_id), {"dev_id": dev_id}
        )
        entity.schedule_update_ha_state.assert_called_once_with()

        await entity.async_will_remove_from_hass()
        assert not entity._ws_subscription.is_connected  # pylint: disable=protected-access

    asyncio.run(_run())


def test_refresh_button_device_info_and_press() -> None:
    async def _run() -> None:
        hass = HomeAssistant()
        entry = types.SimpleNamespace(entry_id="entry-button")
        dev_id = "device-123"
        coordinator = types.SimpleNamespace(
            hass=hass,
            async_request_refresh=AsyncMock(),
        )

        hass.data = {
            DOMAIN: {entry.entry_id: {"coordinator": coordinator, "dev_id": dev_id}}
        }

        added: list = []
        seen_ids: set[str] = set()
        call_sizes: list[int] = []

        def _add_entities(entities):
            call_sizes.append(len(entities))
            for entity in entities:
                uid = getattr(entity, "unique_id", None)
                if uid is None:
                    uid = getattr(entity, "_attr_unique_id", None)
                if uid in seen_ids:
                    continue
                seen_ids.add(str(uid))
                entity.hass = hass
                added.append(entity)

        await async_setup_button_entry(hass, entry, _add_entities)
        assert call_sizes == [1]
        assert len(added) == 1

        button_entity = added[0]
        assert isinstance(button_entity, StateRefreshButton)

        await async_setup_button_entry(hass, entry, _add_entities)
        assert call_sizes == [1, 1]
        assert len(added) == 1
        assert len(seen_ids) == 1

        info = button_entity.device_info
        expected_info = build_gateway_device_info(
            hass,
            entry.entry_id,
            dev_id,
        )
        assert info == expected_info

        await button_entity.async_press()
        coordinator.async_request_refresh.assert_awaited_once()


def test_button_setup_adds_accumulator_entities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        hass = HomeAssistant()
        entry = types.SimpleNamespace(entry_id="entry-boost")
        dev_id = "device-boost"
        coordinator = types.SimpleNamespace(hass=hass, data={})

        hass.data = {
            DOMAIN: {
                entry.entry_id: {
                    "coordinator": coordinator,
                    "dev_id": dev_id,
                }
            }
        }

        acm_node = types.SimpleNamespace(addr="5", supports_boost=lambda: True)
        acm_skip = types.SimpleNamespace(addr="6", supports_boost=lambda: False)
        htr_node = types.SimpleNamespace(addr="3")

        def fake_prepare(entry_data, *, default_name_simple):  # type: ignore[unused-argument]
            return (
                [],
                {"acm": [acm_node, acm_skip], "htr": [htr_node]},
                {},
                lambda node_type, addr: f"{node_type.upper()} {addr}",
            )

        monkeypatch.setattr(button_module, "prepare_heater_platform_data", fake_prepare)

        added: list = []

        def _add_entities(entities):
            for entity in entities:
                entity.hass = hass
            added.extend(entities)

        await async_setup_button_entry(hass, entry, _add_entities)

        assert len(added) == 5
        assert isinstance(added[0], StateRefreshButton)

        boost_entities = added[1:]
        assert all(isinstance(entity, ButtonEntity) for entity in boost_entities)
        names = [getattr(entity, "_attr_name", None) for entity in boost_entities]
        assert names == [
            "Boost 30 minutes",
            "Boost 60 minutes",
            "Boost 120 minutes",
            "Cancel boost",
        ]
        icons = [
            getattr(entity, "icon", getattr(entity, "_attr_icon", None))
            for entity in boost_entities
        ]
        assert icons == [
            "mdi:timer-play",
            "mdi:timer-play",
            "mdi:timer-play",
            "mdi:timer-off",
        ]

    asyncio.run(_run())


def test_accumulator_boost_button_triggers_service() -> None:
    async def _run() -> None:
        hass = HomeAssistant()
        entry_id = "entry-trigger"
        dev_id = "device-trigger"
        coordinator = types.SimpleNamespace(hass=hass, data={})
        hass.services = types.SimpleNamespace(async_call=AsyncMock())

        button = AccumulatorBoostButton(
            coordinator,
            entry_id,
            dev_id,
            "2",
            "Living Room",
            "uid-boost-60",
            minutes=60,
            node_type="acm",
        )
        button.hass = hass

        assert button.translation_placeholders == {"minutes": "60"}

        await button.async_press()

        hass.services.async_call.assert_awaited_once_with(
            DOMAIN,
            button_module._SERVICE_REQUEST_ACCUMULATOR_BOOST,
            {
                "entry_id": entry_id,
                "dev_id": dev_id,
                "node_type": "acm",
                "addr": "2",
                "minutes": 60,
            },
            blocking=True,
        )

    asyncio.run(_run())


def test_accumulator_boost_cancel_button_triggers_service_without_minutes() -> None:
    async def _run() -> None:
        hass = HomeAssistant()
        entry_id = "entry-cancel"
        dev_id = "device-cancel"
        coordinator = types.SimpleNamespace(hass=hass, data={})
        hass.services = types.SimpleNamespace(async_call=AsyncMock())

        button = AccumulatorBoostCancelButton(
            coordinator,
            entry_id,
            dev_id,
            "4",
            "Bedroom",
            "uid-cancel",
            node_type="acm",
        )
        button.hass = hass

        await button.async_press()

        hass.services.async_call.assert_awaited_once_with(
            DOMAIN,
            button_module._SERVICE_REQUEST_ACCUMULATOR_BOOST,
            {
                "entry_id": entry_id,
                "dev_id": dev_id,
                "node_type": "acm",
                "addr": "4",
            },
            blocking=True,
        )

    asyncio.run(_run())


def test_accumulator_boost_button_handles_missing_hass() -> None:
    async def _run() -> None:
        hass = HomeAssistant()
        coordinator = types.SimpleNamespace(hass=hass, data={})
        hass.services = types.SimpleNamespace(async_call=AsyncMock())

        button = AccumulatorBoostButton(
            coordinator,
            "entry-no-hass",
            "device-no-hass",
            "8",
            "Kitchen",
            "uid-no-hass",
            minutes=30,
            node_type="acm",
        )
        button.hass = None

        await button.async_press()

        hass.services.async_call.assert_not_called()

    asyncio.run(_run())


def test_accumulator_boost_button_logs_service_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def _run() -> None:
        caplog.set_level(logging.ERROR)
        hass = HomeAssistant()
        entry_id = "entry-errors"
        dev_id = "device-errors"
        coordinator = types.SimpleNamespace(hass=hass, data={})
        hass.services = types.SimpleNamespace(async_call=AsyncMock())

        button = AccumulatorBoostButton(
            coordinator,
            entry_id,
            dev_id,
            "10",
            "Office",
            "uid-errors",
            minutes=120,
            node_type="acm",
        )
        button.hass = hass

        hass.services.async_call.side_effect = button_module.ServiceNotFound(
            "termoweb", "boost"
        )
        await button.async_press()
        assert "Boost helper service unavailable" in caplog.text

        hass.services.async_call.reset_mock()
        hass.services.async_call.side_effect = button_module.HomeAssistantError("boom")
        await button.async_press()
        assert "Boost helper service failed" in caplog.text

    asyncio.run(_run())


def test_state_refresh_button_direct_press_and_info() -> None:
    async def _run() -> None:
        hass = HomeAssistant()
        coordinator = types.SimpleNamespace(
            hass=hass,
            async_request_refresh=AsyncMock(),
        )

        button = StateRefreshButton(coordinator, "entry-direct", "device-direct")
        button.hass = hass

        info = button.device_info
        expected = build_gateway_device_info(hass, "entry-direct", "device-direct")
        assert info == expected

        await button.async_press()
        coordinator.async_request_refresh.assert_awaited_once()

    asyncio.run(_run())


def test_binary_sensor_setup_adds_boost_entities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        hass = HomeAssistant()
        entry = types.SimpleNamespace(entry_id="entry-boost-binary")
        dev_id = "device-boost"
        coordinator = types.SimpleNamespace(hass=hass, data={})

        hass.data = {
            DOMAIN: {entry.entry_id: {"coordinator": coordinator, "dev_id": dev_id}}
        }

        boost_node = types.SimpleNamespace(addr="4", supports_boost=lambda: True)
        skip_node = types.SimpleNamespace(addr="5", supports_boost=False)

        def fake_prepare(entry_data, *, default_name_simple):  # type: ignore[unused-argument]
            return (
                [],
                {"acm": [boost_node, skip_node]},
                {},
                lambda node_type, addr: f"{node_type.upper()} {addr}",
            )

        monkeypatch.setattr(
            binary_sensor_module, "prepare_heater_platform_data", fake_prepare
        )

        added: list = []

        def _add_entities(entities: list) -> None:
            added.extend(entities)

        await async_setup_binary_sensor_entry(hass, entry, _add_entities)

        assert len(added) == 2
        gateway, boost = added
        assert isinstance(gateway, GatewayOnlineBinarySensor)
        assert isinstance(boost, binary_sensor_module.HeaterBoostActiveBinarySensor)
        assert boost._attr_name == "ACM 4 Boost Active"  # pylint: disable=protected-access

    asyncio.run(_run())
