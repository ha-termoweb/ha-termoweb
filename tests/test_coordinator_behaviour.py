"""State/energy coordinator behaviours not covered by tests/test_coordinator.py."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from homeassistant.core import HomeAssistant
import pytest

from custom_components.termoweb.backend.base import normalise_sample_records
from tests.fakes.cloud import DEV_ID
from tests.fakes.coordinator import (
    Clock,
    energy,
    energy_coordinator,
    node_state,
    rest_client,
    state_coordinator,
)


async def test_poll_stores_thermostat_settings(hass: HomeAssistant) -> None:
    """Thermostats are polled with the heaters and land in the state store."""
    client = rest_client(
        get_node_settings=AsyncMock(return_value={"mode": "auto", "stemp": "19.5"})
    )
    coord = state_coordinator(hass, client, {"thm": ["2"]})

    await coord.async_refresh()

    client.get_node_settings.assert_awaited_once_with(DEV_ID, ("thm", "2"))
    assert node_state(coord, "thm", "2") == {"mode": "auto", "stemp": "19.5"}


async def test_pending_write_that_expires_during_the_read_is_merged(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pending write whose TTL runs out while the read is in flight stops deferring."""
    clock = Clock(100.0).install(monkeypatch)

    async def _slow_read(*_args: Any) -> dict[str, str]:
        clock.now = 106.0  # the 5 s TTL expires while the request is in flight
        return {"mode": "eco"}

    client = rest_client(get_node_settings=AsyncMock(side_effect=_slow_read))
    coord = state_coordinator(hass, client, {"htr": ["1"]})
    coord.register_pending_setting("htr", "1", mode="auto", stemp=None, ttl=5)

    await coord.async_refresh_heater(("htr", "1"))

    assert node_state(coord, "htr", "1") == {"mode": "eco"}


async def test_skipped_energy_poll_does_not_renotify_state_listeners(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """While WS samples hold the lease, the hourly poll republishes nothing new."""
    Clock(10_000.0).install(monkeypatch)
    state = state_coordinator(hass, rest_client(), {"htr": ["1"]})
    coord = energy_coordinator(hass, rest_client(), {"htr": ["1"]}, state=state)
    coord.handle_ws_samples(
        DEV_ID, {"htr": {"1": {"t": 9_000.0, "counter": 1_000}}}, lease_seconds=300
    )
    listener = MagicMock()
    state.async_add_listener(listener)

    await coord.async_refresh()

    coord.client.get_node_samples.assert_not_awaited()
    listener.assert_not_called()
    assert energy(coord, "htr", "1") == pytest.approx(1.0)


async def test_power_monitor_counters_scale_alike_in_poll_and_history(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live poll and the history import read pmo counters in the same unit."""
    Clock(10_000.0).install(monkeypatch)
    rows = [{"t": 9_000.0, "counter": 7_200_000}]
    coord = energy_coordinator(
        hass, rest_client(get_node_samples=AsyncMock(return_value=rows)), {"pmo": ["1"]}
    )

    await coord.async_refresh()
    imported = normalise_sample_records("pmo", rows)

    assert energy(coord, "pmo", "1") == pytest.approx(2.0)
    assert imported[0]["energy_wh"] == pytest.approx(2_000.0)
