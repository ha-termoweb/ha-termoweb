"""Debounced mode/setpoint write queue of the climate entity."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from conftest import (
    FakeCoordinator,
    _install_stubs,
    build_coordinator_device_state,
    build_entry_runtime,
)

_install_stubs()

from custom_components.termoweb.entities import climate as climate_module
from custom_components.termoweb.inventory import Inventory, build_node_inventory
from homeassistant.core import HomeAssistant

DEV_ID = "dev-write-queue"
ADDR = "1"


def _make_heater(
    backend: Any,
) -> tuple[HomeAssistant, climate_module.HeaterClimateEntity]:
    """Return hass and a manual-mode heater entity writing through ``backend``."""

    hass = HomeAssistant()
    nodes = {"nodes": [{"type": "htr", "addr": ADDR, "name": "Heater"}]}
    inventory = Inventory(DEV_ID, build_node_inventory(nodes))
    record = FakeCoordinator._normalise_device_record(
        build_coordinator_device_state(
            nodes=nodes,
            settings={"htr": {ADDR: {"mode": "manual", "stemp": "18.0", "units": "C"}}},
        )
    )
    coordinator = FakeCoordinator(
        hass,
        client=AsyncMock(),
        dev_id=DEV_ID,
        dev=record,
        nodes=None,
        inventory=inventory,
        data={DEV_ID: record},
    )
    build_entry_runtime(
        hass=hass,
        entry_id="entry-write-queue",
        dev_id=DEV_ID,
        coordinator=coordinator,
        backend=backend,
        inventory=inventory,
    )
    heater = climate_module.HeaterClimateEntity(
        coordinator, "entry-write-queue", DEV_ID, ADDR, "Heater"
    )
    heater.hass = hass
    heater._schedule_refresh_fallback = lambda: None
    return hass, heater


async def _wait_for(predicate: Any, timeout: float = 1.0) -> None:
    """Poll ``predicate`` until it is truthy or fail after ``timeout``."""

    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


@pytest.mark.asyncio
async def test_setpoint_queued_during_inflight_write_is_flushed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A set_temperature made while the POST is in flight is written afterwards."""

    monkeypatch.setattr(climate_module, "_WRITE_DEBOUNCE", 0.01)
    release = asyncio.Event()
    sent: list[float] = []

    async def _set_node_settings(*_args: Any, **kwargs: Any) -> None:
        sent.append(kwargs["stemp"])
        if len(sent) == 1:
            await release.wait()

    backend = AsyncMock()
    backend.set_node_settings = AsyncMock(side_effect=_set_node_settings)
    _hass, heater = _make_heater(backend)

    await heater.async_set_temperature(temperature=20.0)
    await _wait_for(lambda: len(sent) == 1)

    # Second change arrives while the first POST is still awaiting the backend.
    await heater.async_set_temperature(temperature=22.0)
    release.set()

    await _wait_for(lambda: len(sent) == 2)
    assert sent == [20.0, 22.0]
    await _wait_for(lambda: heater._write_task is None or heater._write_task.done())
    assert heater._pending_stemp is None
    assert heater._pending_mode is None
    assert backend.set_node_settings.await_count == 2


@pytest.mark.asyncio
async def test_write_task_uses_hass_background_task_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Debounced writes run as a named hass background task."""

    monkeypatch.setattr(climate_module, "_WRITE_DEBOUNCE", 0.01)
    backend = AsyncMock()
    hass, heater = _make_heater(backend)
    names: list[str] = []
    original = hass.async_create_background_task

    def _track(target: Any, name: str, eager_start: bool = True) -> Any:
        names.append(name)
        return original(target, name, eager_start)

    hass.async_create_background_task = _track

    await heater.async_set_temperature(temperature=21.0)
    assert names == [f"termoweb-write-{DEV_ID}-{ADDR}"]
    assert heater._write_task is not None
    await heater._write_task
    backend.set_node_settings.assert_awaited_once()


@pytest.mark.asyncio
async def test_removing_entity_during_debounce_sends_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing the entity during the debounce cancels the pending write."""

    monkeypatch.setattr(climate_module, "_WRITE_DEBOUNCE", 0.05)
    backend = AsyncMock()
    _hass, heater = _make_heater(backend)

    await heater.async_set_temperature(temperature=21.0)
    task = heater._write_task
    assert task is not None and not task.done()

    await heater.async_will_remove_from_hass()
    await asyncio.sleep(0.1)

    assert task.cancelled()
    assert heater._write_task is None
    backend.set_node_settings.assert_not_awaited()
