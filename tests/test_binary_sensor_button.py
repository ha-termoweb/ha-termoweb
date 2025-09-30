# ruff: noqa: D100, D101, D102, D103, D105, D107, INP001
from __future__ import annotations

import asyncio
import types
from unittest.mock import AsyncMock, MagicMock

from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_send

import custom_components.termoweb.binary_sensor as binary_sensor_module
import custom_components.termoweb.button as button_module
from custom_components.termoweb.const import DOMAIN, signal_ws_status
from custom_components.termoweb.utils import build_gateway_device_info

GatewayOnlineBinarySensor = (
    binary_sensor_module.GatewayOnlineBinarySensor
)
async_setup_binary_sensor_entry = binary_sensor_module.async_setup_entry
StateRefreshButton = button_module.StateRefreshButton
async_setup_button_entry = button_module.async_setup_entry


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
        async_dispatcher_send(hass, signal_ws_status(entry.entry_id), {"dev_id": "other"})
        entity.schedule_update_ha_state.assert_not_called()
        async_dispatcher_send(hass, signal_ws_status(entry.entry_id), {"dev_id": dev_id})
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

        hass.data = {DOMAIN: {entry.entry_id: {"coordinator": coordinator, "dev_id": dev_id}}}

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

    asyncio.run(_run())
