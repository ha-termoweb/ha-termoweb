"""Accumulator boost numbers write to the device directly, on real Home Assistant."""

from __future__ import annotations

import asyncio
from collections.abc import Generator
import time
from typing import Any
from unittest.mock import AsyncMock, patch

from aiohttp import ClientError
from homeassistant.const import ATTR_ENTITY_ID
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.termoweb.backend.rest_client import RESTClient
from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.entities import heater as heater_module
from custom_components.termoweb.entities.heater import (
    get_boost_runtime_minutes,
    get_boost_temperature,
)

from .conftest import FakeCloud

ACM = "climate.store"
BOOST_TEMP = "number.store_boost_temperature"
BOOST_DURATION = "number.store_boost_duration"
ACM_SETTINGS = {
    "mode": "auto",
    "state": "off",
    "stemp": "19.0",
    "units": "C",
    "boost_time": 120,
    "boost_temp": "22.0",
}


@pytest.fixture
def extra_options(cloud: FakeCloud) -> Generator[AsyncMock]:
    """Serve one accumulator (addr 2) and patch the boost-defaults write."""
    cloud.get_nodes.return_value = {
        "nodes": [{"type": "acm", "addr": 2, "name": "Store"}]
    }

    async def _get(_dev_id: str, _node: tuple[str, Any]) -> dict[str, Any]:
        return dict(ACM_SETTINGS)

    cloud.get_node_settings.side_effect = _get
    write = AsyncMock(return_value=None)
    with patch.object(RESTClient, "set_acm_extra_options", write):
        yield write


async def _setup(hass: HomeAssistant, entry: MockConfigEntry) -> None:
    """Add the entry to hass and set it up."""
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()


async def _set(hass: HomeAssistant, entity_id: str, value: float) -> None:
    """Set a number value and let follow-up work finish."""
    await hass.services.async_call(
        "number", "set_value", {ATTR_ENTITY_ID: entity_id, "value": value}, True
    )
    for _ in range(3):
        await asyncio.sleep(0)
        await hass.async_block_till_done()


async def test_boost_duration_writes_boost_time(
    hass: HomeAssistant, extra_options: AsyncMock, config_entry: MockConfigEntry
) -> None:
    """Setting the boost duration writes boost_time to the device."""
    await _setup(hass, config_entry)

    await _set(hass, BOOST_DURATION, 3)

    extra_options.assert_awaited_once()
    assert extra_options.await_args.kwargs["boost_time"] == 180
    assert extra_options.await_args.kwargs["boost_temp"] is None
    assert float(hass.states.get(BOOST_DURATION).state) == 3.0
    assert get_boost_runtime_minutes(hass, config_entry.entry_id, "acm", "2") == 180


async def test_boost_duration_not_kept_when_write_fails(
    hass: HomeAssistant, extra_options: AsyncMock, config_entry: MockConfigEntry
) -> None:
    """A rejected boost_time write raises and keeps the previous value."""
    await _setup(hass, config_entry)
    before = hass.states.get(BOOST_DURATION).state
    stored = get_boost_runtime_minutes(hass, config_entry.entry_id, "acm", "2")
    extra_options.side_effect = ClientError("rejected")

    with pytest.raises(HomeAssistantError, match="Boost preset write"):
        await _set(hass, BOOST_DURATION, 5)

    assert hass.states.get(BOOST_DURATION).state == before
    assert get_boost_runtime_minutes(hass, config_entry.entry_id, "acm", "2") == stored


async def test_boost_duration_follows_device_value(
    hass: HomeAssistant, extra_options: AsyncMock, config_entry: MockConfigEntry
) -> None:
    """A boost_time change reported by the device shows on the number."""
    await _setup(hass, config_entry)
    assert float(hass.states.get(BOOST_DURATION).state) == 2.0
    coordinator = hass.data[DOMAIN][config_entry.entry_id].coordinator

    def _mutate(state: Any) -> None:
        state.boost_time = 240

    coordinator.apply_entity_patch("acm", "2", _mutate)
    await hass.async_block_till_done()

    assert float(hass.states.get(BOOST_DURATION).state) == 4.0
    assert hass.states.get(BOOST_DURATION).attributes["preferred_minutes"] == 240


async def test_boost_temperature_works_without_climate_entity(
    hass: HomeAssistant, extra_options: AsyncMock, config_entry: MockConfigEntry
) -> None:
    """The boost temperature writes to the device even if the climate is disabled."""
    await _setup(hass, config_entry)
    registry = er.async_get(hass)
    registry.async_update_entity(ACM, disabled_by=er.RegistryEntryDisabler.USER)
    assert await hass.config_entries.async_reload(config_entry.entry_id)
    await hass.async_block_till_done()
    assert hass.states.get(ACM) is None

    await _set(hass, BOOST_TEMP, 24)

    assert extra_options.await_args.kwargs["boost_temp"] == 24.0
    assert extra_options.await_args.kwargs["boost_time"] is None
    assert float(hass.states.get(BOOST_TEMP).state) == 24.0
    assert get_boost_temperature(hass, config_entry.entry_id, "acm", "2") == 24.0


async def test_boost_buttons_listen_to_coordinator_once(
    hass: HomeAssistant, extra_options: AsyncMock, config_entry: MockConfigEntry
) -> None:
    """Each boost button registers a single coordinator listener."""
    await _setup(hass, config_entry)
    coordinator = hass.data[DOMAIN][config_entry.entry_id].coordinator
    buttons = [
        entity
        for entity in hass.data["button"].entities
        if type(entity).__name__.startswith("AccumulatorBoost")
    ]
    assert buttons

    for button in buttons:
        listeners = [
            callback
            for callback, _context in coordinator._listeners.values()  # noqa: SLF001
            if getattr(callback, "__self__", None) is button
        ]
        assert len(listeners) == 1, button.entity_id


def _set_ws(hass: HomeAssistant, entry: MockConfigEntry, *, healthy: bool) -> None:
    """Report the WebSocket as healthy (recent payload) or disconnected."""
    coordinator = hass.data[DOMAIN][entry.entry_id].coordinator
    now = time.time()
    coordinator.update_gateway_connection(
        status="healthy" if healthy else "disconnected",
        connected=healthy,
        last_event_at=now if healthy else None,
        healthy_since=now if healthy else None,
        healthy_minutes=1.0 if healthy else None,
        last_payload_at=now if healthy else None,
        last_heartbeat_at=now if healthy else None,
        payload_stale=not healthy,
        payload_stale_after=None,
        idle_restart_pending=False,
    )


@pytest.mark.parametrize(
    ("entity_id", "value"), [(BOOST_DURATION, 3), (BOOST_TEMP, 24)]
)
@pytest.mark.parametrize("healthy", [True, False])
async def test_boost_write_refreshes_node_only_when_ws_down(
    hass: HomeAssistant,
    extra_options: AsyncMock,
    config_entry: MockConfigEntry,
    entity_id: str,
    value: float,
    healthy: bool,
) -> None:
    """A boost write refreshes its node once after the delay, only if WS is down."""
    await _setup(hass, config_entry)
    _set_ws(hass, config_entry, healthy=healthy)
    coordinator = hass.data[DOMAIN][config_entry.entry_id].coordinator
    refresh = AsyncMock(return_value=None)

    with (
        patch.object(heater_module, "WS_ECHO_FALLBACK_REFRESH", 0.05),
        patch.object(coordinator, "async_refresh_heater", refresh),
    ):
        await _set(hass, entity_id, value)
        extra_options.assert_awaited_once()
        refresh.assert_not_awaited()
        await asyncio.sleep(0.1)
        await hass.async_block_till_done()

    if healthy:
        refresh.assert_not_awaited()
    else:
        refresh.assert_awaited_once_with(("acm", "2"))
