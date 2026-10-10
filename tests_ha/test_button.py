"""Refresh, accumulator boost and display flash buttons on real Home Assistant."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

from homeassistant.const import ATTR_ENTITY_ID, STATE_UNAVAILABLE
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, entity_registry as er
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.termoweb.backend.termoweb import TermoWebBackend
from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.domain import NodeId, NodeSettingsDelta, NodeType

from .conftest import DEV_ID, FakeCloud
from .fakes.sensors import entity_ids, serve_nodes, setup_entry

REFRESH = "button.termoweb_gateway_force_refresh"
START = "button.store_start_boost"
CANCEL = "button.store_cancel_boost"
NODES = [
    {"type": "htr", "addr": 1, "name": "Living room"},
    {"type": "acm", "addr": 2, "name": "Store"},
    {"type": "thm", "addr": 4, "name": "Hall"},
]


async def _press(hass: HomeAssistant, entity_id: str) -> None:
    """Press ``entity_id`` through the button service."""
    await hass.services.async_call(
        "button", "press", {ATTR_ENTITY_ID: entity_id}, blocking=True
    )


def _set_boost(entry: MockConfigEntry, active: bool) -> None:
    """Push a websocket delta switching the accumulator's boost flag."""
    entry.runtime_data.coordinator.handle_ws_deltas(
        DEV_ID,
        [
            NodeSettingsDelta(
                node_id=NodeId(NodeType.ACCUMULATOR, "2"),
                changes={"boost_active": active},
            )
        ],
    )


async def test_button_set(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """Refresh on the gateway, boost on accumulators, flash on heating nodes."""
    serve_nodes(cloud, NODES)
    await setup_entry(hass, config_entry)

    assert entity_ids(hass, config_entry, "button") == {
        REFRESH,
        START,
        CANCEL,
        "button.living_room_flash_display",
        "button.store_flash_display",
    }
    registry = er.async_get(hass)
    devices = dr.async_get(hass)
    gateway = devices.async_get(registry.async_get(REFRESH).device_id)
    assert (DOMAIN, DEV_ID) in gateway.identifiers
    store = devices.async_get(registry.async_get(START).device_id)
    assert (store.name, store.model) == ("Store", "Accumulator")
    heater = devices.async_get(
        registry.async_get("button.living_room_flash_display").device_id
    )
    assert heater.model == "Heater"


async def test_refresh_button_polls_the_cloud(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """Pressing refresh re-reads the node settings."""
    serve_nodes(cloud, NODES[:1])
    await setup_entry(hass, config_entry)
    calls = cloud.get_node_settings.await_count

    await _press(hass, REFRESH)
    await hass.async_block_till_done()

    assert cloud.get_node_settings.await_count > calls


async def test_flash_button_selects_the_node(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """Pressing flash calls the backend select endpoint for that node."""
    serve_nodes(cloud, NODES[:1])
    await setup_entry(hass, config_entry)
    select = AsyncMock(return_value=None)
    with patch.object(TermoWebBackend, "set_node_display_select", select):
        await _press(hass, "button.living_room_flash_display")

    select.assert_awaited_once_with(DEV_ID, ("htr", "1"), select=True)


async def test_flash_button_failure_raises(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """A failed flash request is reported to the user."""
    serve_nodes(cloud, NODES[:1])
    await setup_entry(hass, config_entry)
    select = AsyncMock(side_effect=RuntimeError("network"))
    with (
        patch.object(TermoWebBackend, "set_node_display_select", select),
        pytest.raises(HomeAssistantError, match="Unable to flash"),
    ):
        await _press(hass, "button.living_room_flash_display")


@pytest.mark.parametrize(
    ("settings", "expected_stemp"),
    [
        ({"boost_time": 180, "boost_temp": "23.5", "stemp": "19.0"}, 23.5),
        ({"boost_time": 180, "stemp": "19.0"}, 19.0),
    ],
)
async def test_boost_start_uses_device_boost_settings(
    hass: HomeAssistant,
    cloud: FakeCloud,
    config_entry: MockConfigEntry,
    settings: dict[str, Any],
    expected_stemp: float,
) -> None:
    """Start uses the device boost time and temperature, falling back to stemp."""
    serve_nodes(cloud, NODES[1:2], {("acm", "2"): settings})
    await setup_entry(hass, config_entry)
    boost = AsyncMock(return_value=None)
    with patch.object(TermoWebBackend, "set_acm_boost_state", boost):
        await _press(hass, START)

    boost.assert_awaited_once_with(
        DEV_ID, "2", boost=True, boost_time=180, stemp=expected_stemp, units="C"
    )


async def test_boost_start_without_setpoint_raises(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """Without a boost temperature or setpoint, start fails and sends nothing."""
    serve_nodes(cloud, NODES[1:2], {("acm", "2"): {"stemp": None}})
    await setup_entry(hass, config_entry)
    boost = AsyncMock(return_value=None)
    with (
        patch.object(TermoWebBackend, "set_acm_boost_state", boost),
        pytest.raises(HomeAssistantError),
    ):
        await _press(hass, START)

    boost.assert_not_awaited()


async def test_cancel_button_is_available_only_during_boost(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """Cancel is offered while a boost runs and cancels it on the device."""
    serve_nodes(cloud, NODES[1:2])
    await setup_entry(hass, config_entry)
    assert hass.states.get(CANCEL).state == STATE_UNAVAILABLE

    _set_boost(config_entry, True)
    await hass.async_block_till_done()
    assert hass.states.get(CANCEL).state != STATE_UNAVAILABLE

    boost = AsyncMock(return_value=None)
    with patch.object(TermoWebBackend, "set_acm_boost_state", boost):
        await _press(hass, CANCEL)
    boost.assert_awaited_once_with(DEV_ID, "2", boost=False)

    _set_boost(config_entry, False)
    await hass.async_block_till_done()
    assert hass.states.get(CANCEL).state == STATE_UNAVAILABLE


@pytest.mark.parametrize("button", [START, CANCEL])
async def test_boost_backend_failure_raises(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry, button: str
) -> None:
    """A backend failure on start or cancel is reported to the user."""
    serve_nodes(cloud, NODES[1:2], {("acm", "2"): {"boost_temp": "22.0"}})
    await setup_entry(hass, config_entry)
    _set_boost(config_entry, True)
    await hass.async_block_till_done()
    boost = AsyncMock(side_effect=RuntimeError("network"))
    with (
        patch.object(TermoWebBackend, "set_acm_boost_state", boost),
        pytest.raises(HomeAssistantError, match="accumulator boost"),
    ):
        await _press(hass, button)

    boost.assert_awaited_once()
