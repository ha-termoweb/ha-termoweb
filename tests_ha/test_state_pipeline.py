"""Power limit and boost settings flow through DomainStateStore, on real HA."""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, patch

from aiohttp import ClientError
from homeassistant.const import ATTR_ENTITY_ID, STATE_UNAVAILABLE
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.termoweb.backend.rest_client import RESTClient
from custom_components.termoweb.backend.termoweb import TermoWebBackend
from custom_components.termoweb.backend.termoweb_ws import TermoWebWSClient
from custom_components.termoweb.domain import NodeId, NodeSettingsDelta, NodeType

from .conftest import DEV_ID, FakeCloud

ACM = "climate.store"
BOOST_TEMP = "number.store_boost_temperature"
BOOST_DURATION = "number.store_boost_duration"
POWER_LIMIT_PATH = f"/api/v2/devs/{DEV_ID}/htr_system/power_limit"
ACM_SETTINGS = {
    "mode": "auto",
    "state": "off",
    "stemp": "19.0",
    "units": "C",
    "boost_time": 120,
    "boost_temp": "22.0",
}


async def _setup(hass: HomeAssistant, entry: MockConfigEntry) -> None:
    """Add the entry to hass and set it up."""
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()


async def _settle(hass: HomeAssistant) -> None:
    """Let follow-up tasks finish."""
    for _ in range(3):
        await asyncio.sleep(0)
        await hass.async_block_till_done()


def _power_limit_entity(hass: HomeAssistant, entry: MockConfigEntry) -> str:
    """Return the entity_id of the power limit number."""
    return next(
        e.entity_id
        for e in er.async_entries_for_config_entry(er.async_get(hass), entry.entry_id)
        if e.unique_id.endswith("power_limit")
    )


def _ws_client(hass: HomeAssistant, entry: MockConfigEntry) -> TermoWebWSClient:
    """Return a real (not started) TermoWeb websocket client bound to the entry."""
    runtime = entry.runtime_data
    return TermoWebWSClient(
        hass,
        entry_id=entry.entry_id,
        dev_id=DEV_ID,
        api_client=runtime.client,
        coordinator=runtime.coordinator,
        inventory=runtime.inventory,
    )


async def test_ws_power_limit_push_updates_number(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """A WebSocket power_limit push lands in the store and on the number."""
    cloud.get_power_limit.return_value = 1000
    await _setup(hass, config_entry)
    number = _power_limit_entity(hass, config_entry)
    assert hass.states.get(number).state == "1000"

    client = _ws_client(hass, config_entry)
    client._handle_legacy_data_batch(  # noqa: SLF001
        [{"path": POWER_LIMIT_PATH, "body": {"power_limit": "2500"}}]
    )
    await _settle(hass)

    assert hass.states.get(number).state == "2500"
    coordinator = config_entry.runtime_data.coordinator
    assert coordinator.domain_view.get_power_limit() == 2500
    assert not hasattr(config_entry.runtime_data, "power_limit")


async def test_ws_power_limit_invalid_value_is_ignored(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """A malformed WebSocket power_limit keeps the last known value."""
    cloud.get_power_limit.return_value = 1000
    await _setup(hass, config_entry)
    number = _power_limit_entity(hass, config_entry)

    client = _ws_client(hass, config_entry)
    client._handle_legacy_data_batch(  # noqa: SLF001
        [{"path": POWER_LIMIT_PATH, "body": {"power_limit": "lots"}}]
    )
    await _settle(hass)

    assert hass.states.get(number).state == "1000"


async def test_rest_poll_power_limit_updates_number(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """The REST poll stores the power limit; the number follows it."""
    await _setup(hass, config_entry)
    number = _power_limit_entity(hass, config_entry)
    assert hass.states.get(number).state == STATE_UNAVAILABLE

    cloud.get_power_limit.return_value = 1500
    await config_entry.runtime_data.coordinator.async_refresh()
    await _settle(hass)

    assert hass.states.get(number).state == "1500"


async def test_rest_poll_power_limit_failure_keeps_value(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """A failing power limit read does not fail the poll or clear the value."""
    cloud.get_power_limit.return_value = 1500
    await _setup(hass, config_entry)
    number = _power_limit_entity(hass, config_entry)

    cloud.get_power_limit.side_effect = ClientError("down")
    coordinator = config_entry.runtime_data.coordinator
    await coordinator.async_refresh()
    await _settle(hass)

    assert coordinator.last_update_success
    assert hass.states.get(number).state == "1500"


async def test_power_limit_write_goes_through_backend(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """Writing the number calls the backend, then stores the value optimistically."""
    cloud.get_power_limit.return_value = 1000
    rest_write = AsyncMock(return_value=None)
    with patch.object(RESTClient, "set_power_limit", rest_write):
        await _setup(hass, config_entry)
        number = _power_limit_entity(hass, config_entry)
        backend_write = AsyncMock(
            wraps=config_entry.runtime_data.backend.set_power_limit
        )
        with patch.object(
            config_entry.runtime_data.backend, "set_power_limit", backend_write
        ):
            await hass.services.async_call(
                "number", "set_value", {ATTR_ENTITY_ID: number, "value": 3000}, True
            )
            await _settle(hass)

    backend_write.assert_awaited_once_with(DEV_ID, power_limit=3000)
    rest_write.assert_awaited_once_with(DEV_ID, power_limit=3000)
    assert hass.states.get(number).state == "3000"
    coordinator = config_entry.runtime_data.coordinator
    assert coordinator.domain_view.get_power_limit() == 3000


async def test_power_limit_write_failure_keeps_value(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """A rejected power limit write raises and keeps the stored value."""
    cloud.get_power_limit.return_value = 1000
    with patch.object(
        RESTClient, "set_power_limit", AsyncMock(side_effect=ClientError("no"))
    ):
        await _setup(hass, config_entry)
        number = _power_limit_entity(hass, config_entry)
        with pytest.raises(HomeAssistantError, match="Power limit write"):
            await hass.services.async_call(
                "number", "set_value", {ATTR_ENTITY_ID: number, "value": 3000}, True
            )

    assert hass.states.get(number).state == "1000"


@pytest.fixture
def accumulator(cloud: FakeCloud) -> Generator[AsyncMock]:
    """Serve one accumulator (addr 2) and patch the boost-defaults write."""
    cloud.get_nodes.return_value = {
        "nodes": [{"type": "acm", "addr": 2, "name": "Store"}]
    }
    settings = dict(ACM_SETTINGS)

    async def _get(_dev_id: str, _node: tuple[str, Any]) -> dict[str, Any]:
        return dict(settings)

    cloud.get_node_settings.side_effect = _get
    write = AsyncMock(return_value=None)
    write.settings = settings
    with patch.object(RESTClient, "set_acm_extra_options", write):
        yield write


async def test_boost_settings_come_from_device_state(
    hass: HomeAssistant, accumulator: AsyncMock, config_entry: MockConfigEntry
) -> None:
    """The boost numbers and the climate preset read the device's boost settings."""
    await _setup(hass, config_entry)

    assert float(hass.states.get(BOOST_DURATION).state) == 2.0
    assert float(hass.states.get(BOOST_TEMP).state) == 22.0
    assert hass.states.get(ACM).attributes["preferred_boost_minutes"] == 120
    runtime = config_entry.runtime_data
    assert not hasattr(runtime, "boost_runtime")
    assert not hasattr(runtime, "boost_temperature")


async def test_ws_boost_delta_updates_numbers_and_climate(
    hass: HomeAssistant, accumulator: AsyncMock, config_entry: MockConfigEntry
) -> None:
    """A WebSocket delta changing boost_time/boost_temp reaches every consumer."""
    await _setup(hass, config_entry)
    coordinator = config_entry.runtime_data.coordinator

    coordinator.handle_ws_deltas(
        DEV_ID,
        [
            NodeSettingsDelta(
                node_id=NodeId(NodeType.ACCUMULATOR, "2"),
                changes={"boost_time": 300, "boost_temp": "23.5"},
            )
        ],
    )
    await _settle(hass)

    assert float(hass.states.get(BOOST_DURATION).state) == 5.0
    assert float(hass.states.get(BOOST_TEMP).state) == 23.5
    assert hass.states.get(BOOST_TEMP).attributes["preferred_temperature"] == 23.5
    assert hass.states.get(ACM).attributes["preferred_boost_minutes"] == 300


async def test_boost_duration_write_survives_poll_and_reload(
    hass: HomeAssistant, accumulator: AsyncMock, config_entry: MockConfigEntry
) -> None:
    """A boost duration write is used by the climate, a poll and a reload."""
    await _setup(hass, config_entry)

    await hass.services.async_call(
        "number", "set_value", {ATTR_ENTITY_ID: BOOST_DURATION, "value": 3}, True
    )
    await _settle(hass)
    assert hass.states.get(ACM).attributes["preferred_boost_minutes"] == 180

    # The device accepted the write: later reads report the new boost_time.
    accumulator.settings["boost_time"] = 180
    await config_entry.runtime_data.coordinator.async_refresh()
    await _settle(hass)
    assert float(hass.states.get(BOOST_DURATION).state) == 3.0

    assert await hass.config_entries.async_reload(config_entry.entry_id)
    await hass.async_block_till_done()
    assert float(hass.states.get(BOOST_DURATION).state) == 3.0
    assert hass.states.get(ACM).attributes["preferred_boost_minutes"] == 180


async def test_boost_start_button_uses_device_boost_time(
    hass: HomeAssistant, accumulator: AsyncMock, config_entry: MockConfigEntry
) -> None:
    """The boost start button runs for the device's boost_time."""
    await _setup(hass, config_entry)
    start = next(
        e.entity_id
        for e in er.async_entries_for_config_entry(
            er.async_get(hass), config_entry.entry_id
        )
        if e.domain == "button" and e.unique_id.endswith("start")
    )
    boost = AsyncMock(return_value=None)
    with patch.object(TermoWebBackend, "set_acm_boost_state", boost):
        await hass.services.async_call("button", "press", {ATTR_ENTITY_ID: start}, True)
        await _settle(hass)

    boost.assert_awaited_once()
    assert boost.await_args.kwargs["boost_time"] == 120
