"""Hourly REST energy polling: WS-aware, single pipeline, 429 backoff."""

from __future__ import annotations

import asyncio
from datetime import timedelta
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from conftest import _install_stubs, build_device_metadata_payload

_install_stubs()

from custom_components.termoweb import coordinator as coord_module
from custom_components.termoweb.backend.rest_client import BackendRateLimitError
from custom_components.termoweb.inventory import Inventory, build_node_inventory
from homeassistant.core import HomeAssistant

EnergyStateCoordinator = coord_module.EnergyStateCoordinator


class Clock:
    """Shared fake for ``time.time`` and ``time.monotonic``."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _inventory(*addrs: str) -> Inventory:
    nodes = {"nodes": [{"type": "htr", "addr": addr} for addr in addrs]}
    return Inventory("dev", build_node_inventory(nodes))


def _energy_coordinator(
    monkeypatch: pytest.MonkeyPatch, clock: Clock, *addrs: str
) -> tuple[EnergyStateCoordinator, AsyncMock, list[Any]]:
    """Return a coordinator with a sample-counting client and captured HH:05 jobs."""
    monkeypatch.setattr(coord_module.time, "time", clock)
    monkeypatch.setattr(coord_module, "time_mod", clock)

    async def _samples(_dev: str, _node: Any, _start: float, end: float) -> list:
        return [{"t": end, "counter": end}]

    client = types.SimpleNamespace(get_node_samples=AsyncMock(side_effect=_samples))
    jobs: list[Any] = []

    def _track(hass: Any, action: Any, **kwargs: Any) -> MagicMock:
        assert kwargs == {"minute": 5, "second": 0}
        jobs.append(action)
        return MagicMock()

    monkeypatch.setattr(coord_module, "async_track_time_change", _track)
    coord = EnergyStateCoordinator(HomeAssistant(), client, "dev", _inventory(*addrs))
    return coord, client.get_node_samples, jobs


async def _simulate_hour(
    coord: EnergyStateCoordinator,
    clock: Clock,
    hourly_job: Any,
    ws_every_min: int | None,
) -> None:
    """Run 60 minutes emulating HA scheduling, WS samples and the HH:05 job."""
    next_refresh: float | None = None

    def _reschedule() -> None:
        nonlocal next_refresh
        interval = coord.update_interval
        next_refresh = None if interval is None else clock.now + interval.seconds

    _reschedule()
    for minute in range(1, 61):
        clock.now = minute * 60.0
        if ws_every_min and minute % ws_every_min == 0:
            coord.handle_ws_samples(
                "dev",
                {"htr": {"A": {"t": clock.now, "counter": clock.now}}},
                lease_seconds=120,
            )
            _reschedule()  # HA reschedules on async_set_updated_data
        if next_refresh is not None and clock.now >= next_refresh:
            await coord.async_refresh()
            _reschedule()
        if minute == 5:
            await hourly_job(None)


@pytest.mark.parametrize("ws_every_min", [2, 4, 10, 30])
def test_ws_samples_never_increase_rest_sample_polls(
    monkeypatch: pytest.MonkeyPatch, ws_every_min: int
) -> None:
    """WS samples every N minutes never cause more REST fetches than no WS."""

    async def _count(ws: int | None) -> int:
        clock = Clock()
        coord, fetch, jobs = _energy_coordinator(monkeypatch, clock, "A", "B")
        await coord.async_config_entry_first_refresh()
        coord.async_start_hourly_poll()
        fetch.reset_mock()
        await _simulate_hour(coord, clock, jobs[0], ws)
        assert coord.update_interval is None
        return fetch.await_count

    baseline = asyncio.run(_count(None))
    with_ws = asyncio.run(_count(ws_every_min))

    assert baseline == 2  # one fetch per node at HH:05
    assert with_ws <= baseline
    if ws_every_min <= 2:
        assert with_ws == 0  # WS samples fresh at HH:05: no REST at all


def test_hourly_poll_runs_once_and_not_while_ws_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The HH:05 job fetches each node once, and nothing while WS is fresh."""
    clock = Clock()
    coord, fetch, jobs = _energy_coordinator(monkeypatch, clock, "A", "B")
    unsub = coord.async_start_hourly_poll()
    assert callable(unsub)
    assert len(jobs) == 1

    async def _run() -> None:
        clock.now = 3_900.0
        await jobs[0](None)
        assert fetch.await_count == 2
        assert {c.args[1] for c in fetch.await_args_list} == {
            ("htr", "A"),
            ("htr", "B"),
        }
        # Window ends now and spans the previous hour.
        _, _, start, end = fetch.await_args_list[0].args
        assert end - start == 3600

        clock.now = 7_400.0
        coord.handle_ws_samples(
            "dev", {"htr": {"A": {"t": 7_400.0, "counter": 9e6}}}, lease_seconds=120
        )
        clock.now = 7_500.0
        await jobs[0](None)
        assert fetch.await_count == 2  # WS fresh: skipped

    asyncio.run(_run())


def test_rate_limit_stops_node_loop_and_backs_off(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A 429 on node 1 skips node 2 and the following hourly polls."""
    clock = Clock()
    coord, fetch, jobs = _energy_coordinator(monkeypatch, clock, "A", "B")
    coord.async_start_hourly_poll()
    fetch.side_effect = BackendRateLimitError("429")

    async def _tick() -> int:
        before = fetch.await_count
        clock.now += 3600.0
        await jobs[0](None)
        return fetch.await_count - before

    async def _run() -> None:
        assert await _tick() == 1  # node A only; node B never called
        assert fetch.await_args.args[1] == ("htr", "A")
        assert await _tick() == 0  # backoff: skip 1 hourly poll
        assert await _tick() == 1  # 429 again -> skip 2
        assert await _tick() == 0
        assert await _tick() == 0
        fetch.side_effect = None
        fetch.return_value = [{"t": clock.now, "counter": 1000.0}]
        assert await _tick() == 2  # recovered: both nodes, backoff reset
        assert coord._rate_limit_backoff == 0  # noqa: SLF001
        fetch.side_effect = BackendRateLimitError("429")
        assert await _tick() == 1
        assert await _tick() == 0  # back to a 1-poll skip
        assert await _tick() == 1

    asyncio.run(_run())
    assert any("rate limited" in r.getMessage() for r in caplog.records)


def test_rate_limit_backoff_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Consecutive 429s double the skipped polls up to the cap."""
    clock = Clock()
    coord, fetch, _ = _energy_coordinator(monkeypatch, clock, "A")
    fetch.side_effect = BackendRateLimitError("429")

    async def _run() -> None:
        for _ in range(6):
            await coord.async_refresh()
            coord._skip_polls = 0  # noqa: SLF001 - force the next poll

    asyncio.run(_run())
    assert coord._rate_limit_backoff == coord_module.ENERGY_RATE_LIMIT_MAX_SKIP  # noqa: SLF001


def test_hourly_poll_swallows_update_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing hourly poll is logged, not raised into the time tracker."""
    clock = Clock()
    coord, fetch, jobs = _energy_coordinator(monkeypatch, clock, "A")
    coord.async_start_hourly_poll()
    monkeypatch.setattr(
        coord, "_poll_recent_samples", AsyncMock(side_effect=TimeoutError)
    )

    asyncio.run(jobs[0](None))

    assert coord.last_update_success is False


def _state_coordinator(client: Any) -> coord_module.StateCoordinator:
    inventory = _inventory("A")
    return coord_module.StateCoordinator(
        HomeAssistant(),
        client,
        60,
        "dev",
        build_device_metadata_payload("dev"),
        inventory=inventory,
    )


def test_resume_polling_keeps_rate_limit_backoff() -> None:
    """Resuming after WS suspension honours a pending rate-limit backoff."""
    coord = _state_coordinator(types.SimpleNamespace())
    coord.update_interval = None
    coord._backoff = 480  # noqa: SLF001

    coord.resume_polling(60)
    assert coord.update_interval == timedelta(seconds=480)

    coord._backoff = 0  # noqa: SLF001
    coord.resume_polling(60)
    assert coord.update_interval == timedelta(seconds=60)


def test_successful_poll_does_not_unsuspend_polling() -> None:
    """Clearing a backoff must not re-enable polling that WS suspended."""
    client = types.SimpleNamespace(
        get_node_settings=AsyncMock(return_value={"mode": "auto"})
    )
    coord = _state_coordinator(client)
    coord._backoff = 120  # noqa: SLF001
    coord.update_interval = None

    asyncio.run(coord.async_refresh())

    assert coord._backoff == 0  # noqa: SLF001
    assert coord.update_interval is None
