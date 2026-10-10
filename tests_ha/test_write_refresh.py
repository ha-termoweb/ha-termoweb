"""Entity writes rely on the WebSocket echo instead of polling every node."""

from __future__ import annotations

import asyncio
from collections.abc import Generator
import time
from typing import Any
from unittest.mock import AsyncMock, patch

from homeassistant.const import ATTR_ENTITY_ID
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.termoweb.backend.ducaheat import (
    DucaheatBackend,
    DucaheatRESTClient,
)
from custom_components.termoweb.backend.rest_client import RESTClient
from custom_components.termoweb.const import BRAND_DUCAHEAT, CONF_BRAND, DOMAIN
from custom_components.termoweb.entities import (
    climate as climate_module,
    heater as heater_module,
)

from .conftest import PASSWORD, USERNAME, FakeCloud

HTR = "climate.living_room"
PRIORITY = "number.living_room_priority"
SETTINGS = {
    "mode": "manual",
    "state": "off",
    "stemp": "21.0",
    "units": "C",
    "lock": False,
    "priority": 1,
}


class Ducaheat:
    """Fake Ducaheat cloud with two heaters (addr 1 and 3)."""

    def __init__(self, cloud: FakeCloud) -> None:
        """Serve two heaters and record reads and writes."""
        cloud.get_nodes.return_value = {
            "nodes": [
                {"type": "htr", "addr": 1, "name": "Living room"},
                {"type": "htr", "addr": 3, "name": "Bedroom"},
            ]
        }
        self.reads: list[str] = []
        self.cloud = cloud
        self.settings = AsyncMock(return_value=None)
        self.lock = AsyncMock(return_value=None)
        self.priority = AsyncMock(return_value=None)

    async def get_node_settings(
        self, _dev_id: str, node: tuple[str, Any]
    ) -> dict[str, Any]:
        """Record which node was read by REST."""
        self.reads.append(str(node[1]))
        return dict(SETTINGS)


@pytest.fixture
def ducaheat(cloud: FakeCloud) -> Generator[Ducaheat]:
    """Patch the Ducaheat REST client reads and writes."""
    fake = Ducaheat(cloud)
    with (
        patch.object(DucaheatRESTClient, "get_node_settings", fake.get_node_settings),
        patch.object(DucaheatRESTClient, "get_node_samples", cloud.get_node_samples),
        patch.object(DucaheatRESTClient, "set_node_settings", fake.settings),
        patch.object(DucaheatRESTClient, "set_node_lock", fake.lock),
        patch.object(DucaheatRESTClient, "set_node_priority", fake.priority),
        patch.object(
            DucaheatBackend,
            "create_ws_client",
            lambda _self, hass, *a, **kw: cloud.create_ws_client(hass, *a, **kw),
        ),
        patch.object(climate_module, "_WRITE_DEBOUNCE", 0),
        patch.object(heater_module, "WS_ECHO_FALLBACK_REFRESH", 0),
    ):
        yield fake


async def _setup(hass: HomeAssistant) -> MockConfigEntry:
    """Set up a Ducaheat entry and return it."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        unique_id=USERNAME,
        data={"username": USERNAME, "password": PASSWORD, CONF_BRAND: BRAND_DUCAHEAT},
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return entry


def _set_ws(hass: HomeAssistant, entry: MockConfigEntry, *, healthy: bool) -> None:
    """Report the WebSocket as healthy (recent payload) or disconnected."""
    coordinator = entry.runtime_data.coordinator
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


def _lock_entity_id(hass: HomeAssistant, entry: MockConfigEntry) -> str:
    """Return the child-lock entity of the living-room heater."""
    return next(
        e.entity_id
        for e in er.async_entries_for_config_entry(er.async_get(hass), entry.entry_id)
        if e.domain == "lock" and e.unique_id.endswith(":1:child_lock")
    )


async def _settle(hass: HomeAssistant) -> None:
    """Let write tasks and any fallback refresh run."""
    for _ in range(5):
        await asyncio.sleep(0)
        await hass.async_block_till_done()


async def _write(hass: HomeAssistant, entry: MockConfigEntry, target: str) -> None:
    """Perform one entity write of the given kind on heater 1."""
    if target == "climate":
        data = {ATTR_ENTITY_ID: HTR, "temperature": 23}
        await hass.services.async_call(
            "climate", "set_temperature", data, blocking=True
        )
    elif target == "lock":
        data = {ATTR_ENTITY_ID: _lock_entity_id(hass, entry)}
        await hass.services.async_call("lock", "lock", data, blocking=True)
    else:
        data = {ATTR_ENTITY_ID: PRIORITY, "value": 5}
        await hass.services.async_call("number", "set_value", data, blocking=True)
    await _settle(hass)


WRITES = ["climate", "lock", "priority"]


@pytest.mark.parametrize("target", WRITES)
async def test_write_with_healthy_ws_reads_nothing(
    hass: HomeAssistant, ducaheat: Ducaheat, target: str
) -> None:
    """With the WebSocket healthy a write triggers no REST read at all."""
    entry = await _setup(hass)
    _set_ws(hass, entry, healthy=True)
    ducaheat.reads.clear()

    await _write(hass, entry, target)

    assert (
        ducaheat.settings.await_count
        + ducaheat.lock.await_count
        + (ducaheat.priority.await_count)
        == 1
    )
    assert ducaheat.reads == []


@pytest.mark.parametrize("target", WRITES)
async def test_write_with_ws_down_refreshes_only_that_node(
    hass: HomeAssistant, ducaheat: Ducaheat, target: str
) -> None:
    """With the WebSocket down a write refreshes only the written node, once."""
    entry = await _setup(hass)
    _set_ws(hass, entry, healthy=False)
    ducaheat.reads.clear()

    await _write(hass, entry, target)

    assert ducaheat.reads == ["1"]


async def test_lock_write_is_optimistic(
    hass: HomeAssistant, ducaheat: Ducaheat
) -> None:
    """The lock shows the written state at once, without waiting for a read."""
    entry = await _setup(hass)
    _set_ws(hass, entry, healthy=True)
    lock = _lock_entity_id(hass, entry)
    assert hass.states.get(lock).state == "unlocked"
    ducaheat.reads.clear()

    await _write(hass, entry, "lock")
    assert hass.states.get(lock).state == "locked"

    await hass.services.async_call("lock", "unlock", {ATTR_ENTITY_ID: lock}, True)
    await _settle(hass)
    assert hass.states.get(lock).state == "unlocked"
    assert ducaheat.reads == []


async def test_priority_write_is_optimistic(
    hass: HomeAssistant, ducaheat: Ducaheat
) -> None:
    """The priority number shows the written value at once."""
    entry = await _setup(hass)
    _set_ws(hass, entry, healthy=True)

    await _write(hass, entry, "priority")

    assert hass.states.get(PRIORITY).state == "5"


async def test_fallback_cancelled_when_entity_removed(
    hass: HomeAssistant, ducaheat: Ducaheat
) -> None:
    """A pending fallback refresh does not run after the entry is unloaded."""
    entry = await _setup(hass)
    _set_ws(hass, entry, healthy=False)
    ducaheat.reads.clear()

    with patch.object(heater_module, "WS_ECHO_FALLBACK_REFRESH", 3600):
        await _write(hass, entry, "lock")
        assert await hass.config_entries.async_unload(entry.entry_id)
        await hass.async_block_till_done()

    assert ducaheat.reads == []


async def test_power_limit_write_reads_nothing(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """Writing the power limit updates the number without any REST read."""
    cloud.get_power_limit.return_value = 1000
    set_limit = AsyncMock(return_value=None)
    with patch.object(RESTClient, "set_power_limit", set_limit):
        config_entry.add_to_hass(hass)
        assert await hass.config_entries.async_setup(config_entry.entry_id)
        await hass.async_block_till_done()
        number = next(
            e.entity_id
            for e in er.async_entries_for_config_entry(
                er.async_get(hass), config_entry.entry_id
            )
            if e.unique_id.endswith("power_limit")
        )
        cloud.get_node_settings.reset_mock()
        cloud.get_power_limit.reset_mock()

        await hass.services.async_call(
            "number", "set_value", {ATTR_ENTITY_ID: number, "value": 2000}, True
        )
        await _settle(hass)

    set_limit.assert_awaited_once()
    assert hass.states.get(number).state == "2000"
    cloud.get_node_settings.assert_not_awaited()
    cloud.get_power_limit.assert_not_awaited()


async def test_fallback_refresh_failure_is_logged(
    hass: HomeAssistant, ducaheat: Ducaheat, caplog: pytest.LogCaptureFixture
) -> None:
    """A failing fallback refresh is logged instead of crashing."""
    entry = await _setup(hass)
    _set_ws(hass, entry, healthy=False)
    coordinator = entry.runtime_data.coordinator

    with patch.object(
        coordinator, "async_refresh_heater", AsyncMock(side_effect=RuntimeError("x"))
    ):
        await _write(hass, entry, "lock")

    assert "Refresh fallback failed" in caplog.text
