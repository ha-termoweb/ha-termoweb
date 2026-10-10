"""Tests for energy polling, sample throttling and history import."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from aiohttp import ClientError
from freezegun.api import FrozenDateTimeFactory
from homeassistant.components.recorder import Recorder, get_instance
from homeassistant.components.recorder.db_schema import StatisticsShortTerm
from homeassistant.components.recorder.models import StatisticMeanType
from homeassistant.components.recorder.statistics import (
    async_import_statistics,
    get_metadata,
    statistics_during_period,
)
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.update_coordinator import UpdateFailed
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import EnergyConverter
import pytest
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_fire_time_changed,
)
from pytest_homeassistant_custom_component.components.recorder.common import (
    async_wait_recording_done,
)
from pytest_homeassistant_custom_component.typing import RecorderInstanceGenerator
import voluptuous as vol

from custom_components.termoweb.backend.rest_client import (
    BackendAuthError,
    BackendRateLimitError,
)
from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.coordinator import (
    ENERGY_RATE_LIMIT_MAX_SKIP,
    EnergyStateCoordinator,
)
from custom_components.termoweb.domain.energy import EnergyNodeMetrics, EnergySnapshot
from custom_components.termoweb.domain.ids import NodeId, NodeType
from custom_components.termoweb.domain.state import DomainStateStore
from custom_components.termoweb.domain.view import DomainStateView
from custom_components.termoweb.identifiers import build_heater_energy_unique_id
from custom_components.termoweb.throttle import (
    MonotonicRateLimiter,
    default_samples_rate_limit_state,
    reset_samples_rate_limit_state,
)
from tests.fakes.cloud import DEV_ID, FakeCloud
from tests.fakes.coordinator import (
    Clock,
    energy,
    energy_coordinator,
    power,
    rest_client,
    state_coordinator,
)


def _samples(*points: tuple[float, float]) -> list[dict[str, float]]:
    """Return REST sample rows for ``(t, counter)`` points."""
    return [{"t": t, "counter": counter} for t, counter in points]


def _ws(addr: str, t: float, counter: float, node_type: str = "htr") -> dict:
    """Return a websocket ``samples`` update for one node."""
    return {node_type: {addr: {"t": t, "counter": counter}}}


# --- construction ------------------------------------------------------------


async def test_energy_coordinator_requires_an_inventory(hass: HomeAssistant) -> None:
    """The inventory is mandatory: without it energy cannot be tracked."""
    with pytest.raises(TypeError):
        EnergyStateCoordinator(hass, rest_client(), DEV_ID, None)  # type: ignore[arg-type]


async def test_energy_coordinator_is_never_interval_polled(
    hass: HomeAssistant,
) -> None:
    """REST energy is fetched only by the HH:05 job, never on an interval."""
    coord = energy_coordinator(hass, rest_client(), {"htr": ["1"]})

    assert coord.update_interval is None
    assert coord.data == EnergySnapshot(
        dev_id=DEV_ID, metrics={}, updated_at=coord.data.updated_at, ws_deadline=None
    )


# --- REST samples ------------------------------------------------------------


async def test_rest_samples_give_energy_then_derived_power(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Counters are Wh for heaters; power comes from two consecutive samples."""
    clock = Clock(1000.0).install(monkeypatch)
    client = rest_client(
        get_node_samples=AsyncMock(
            side_effect=[_samples((1000, 1.0)), _samples((1900, 1.5))]
        )
    )
    coord = energy_coordinator(hass, client, {"htr": ["1"]})

    await coord.async_refresh()
    assert energy(coord, "htr", "1") == pytest.approx(0.001)
    assert power(coord, "htr", "1") is None
    _dev, node, start, end = client.get_node_samples.await_args.args
    assert node == ("htr", "1")
    assert (start, end) == (-2600.0, 1000.0)  # the past hour

    clock.now = 1900.0
    await coord.async_refresh()
    assert energy(coord, "htr", "1") == pytest.approx(0.0015)
    assert power(coord, "htr", "1") == pytest.approx(2.0, rel=1e-3)


async def test_rest_energy_reaches_the_state_view(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Energy snapshots are written into the state coordinator's domain store."""
    Clock(1000.0).install(monkeypatch)
    client = rest_client(
        get_node_samples=AsyncMock(return_value=_samples((1000, 1000)))
    )
    state = state_coordinator(hass, client, {"htr": ["1"]})
    coord = energy_coordinator(hass, client, {"htr": ["1"]}, state=state)

    await coord.async_refresh()

    metric = state.domain_view.get_energy_metric("htr", "1")
    assert metric is not None
    assert metric.energy_kwh == pytest.approx(1.0)
    assert metric.source == "rest"


async def test_rest_counter_reset_restarts_energy_without_power(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A newer, lower counter is a meter reset: no negative power is derived."""
    clock = Clock(1000.0).install(monkeypatch)
    client = rest_client(
        get_node_samples=AsyncMock(
            side_effect=[_samples((1000, 5.0)), _samples((1900, 1.0))]
        )
    )
    coord = energy_coordinator(hass, client, {"htr": ["1"]})

    await coord.async_refresh()
    clock.now = 1900.0
    await coord.async_refresh()

    assert energy(coord, "htr", "1") == pytest.approx(0.001)
    assert power(coord, "htr", "1") is None


async def test_empty_rest_poll_keeps_previous_energy_and_power(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A poll without new samples republishes the cached values."""
    Clock(2000.0).install(monkeypatch)
    client = rest_client(
        get_node_samples=AsyncMock(
            side_effect=[_samples((1000, 1000)), _samples((1600, 2000)), []]
        )
    )
    coord = energy_coordinator(hass, client, {"htr": ["1"]})

    await coord.async_refresh()
    await coord.async_refresh()
    assert energy(coord, "htr", "1") == pytest.approx(2.0)
    assert power(coord, "htr", "1") == pytest.approx(6000.0)

    await coord.async_refresh()
    assert energy(coord, "htr", "1") == pytest.approx(2.0)
    assert power(coord, "htr", "1") == pytest.approx(6000.0)


@pytest.mark.parametrize(
    "rows",
    [
        [{"t": 1000, "counter": None}],
        [{"t": " ", "counter": "garbage"}],
        [{"counter": 1000}],
    ],
)
async def test_unusable_rest_samples_record_nothing(
    hass: HomeAssistant, rows: list[dict]
) -> None:
    """Rows without a timestamp or counter are skipped, not stored as zero."""
    client = rest_client(get_node_samples=AsyncMock(return_value=rows))
    coord = energy_coordinator(hass, client, {"htr": ["1"]})

    await coord.async_refresh()

    assert coord.last_update_success is True
    assert coord.data.metrics_for_type("htr") == {}


async def test_rest_counter_fallbacks(hass: HomeAssistant) -> None:
    """Rows without ``counter`` use ``counter_max``, then ``counter_min``."""
    client = rest_client(
        get_node_samples=AsyncMock(
            side_effect=[
                [{"t": 1000, "counter_max": 3000, "counter_min": 1000}],
                [{"t": 1000, "counter_min": 2000}],
            ]
        )
    )
    coord = energy_coordinator(hass, client, {"htr": ["1", "2"]})

    await coord.async_refresh()

    assert energy(coord, "htr", "1") == pytest.approx(3.0)
    assert energy(coord, "htr", "2") == pytest.approx(2.0)


@pytest.mark.parametrize("error", [ClientError("fail"), BackendAuthError("denied")])
async def test_per_node_fetch_errors_skip_that_node(
    hass: HomeAssistant, error: Exception
) -> None:
    """A failed node fetch records nothing for it and does not fail the poll."""
    client = rest_client(get_node_samples=AsyncMock(side_effect=error))
    coord = energy_coordinator(hass, client, {"htr": ["1"], "acm": ["2"]})

    await coord.async_refresh()

    assert coord.last_update_success is True
    assert client.get_node_samples.await_count == 2
    assert coord.data.metrics_for_type("htr") == {}


async def test_rest_timeout_fails_the_poll(hass: HomeAssistant) -> None:
    """A timeout fails the whole energy poll."""
    client = rest_client(get_node_samples=AsyncMock(side_effect=TimeoutError))
    coord = energy_coordinator(hass, client, {"htr": ["1"]})

    with pytest.raises(UpdateFailed, match="API timeout"):
        await coord.async_refresh()


async def test_rate_limit_keeps_nodes_fetched_before_it(
    hass: HomeAssistant,
) -> None:
    """A 429 stops the node loop; nodes already fetched keep their energy."""

    async def _samples_or_429(_dev: str, node: tuple[str, str], *_: Any) -> list:
        if node[0] == "htr":
            return _samples((0, 500))
        raise BackendRateLimitError("429")

    client = rest_client(get_node_samples=AsyncMock(side_effect=_samples_or_429))
    coord = energy_coordinator(hass, client, {"htr": ["1"], "acm": ["2"]})

    await coord.async_refresh()

    assert energy(coord, "htr", "1") == pytest.approx(0.5)
    assert energy(coord, "acm", "2") is None


# --- websocket samples -------------------------------------------------------


async def _ws_coordinator(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> EnergyStateCoordinator:
    """Return a coordinator advanced by two WS samples (10.0 -> 10.5 kWh)."""
    Clock(20_000.0).install(monkeypatch)
    coord = energy_coordinator(hass, rest_client(), {"htr": ["1"]})
    coord.handle_ws_samples(DEV_ID, _ws("1", 9_400.0, 10_000))
    coord.handle_ws_samples(DEV_ID, _ws("1", 10_000.0, 10_500))
    assert energy(coord, "htr", "1") == pytest.approx(10.5)
    assert power(coord, "htr", "1") == pytest.approx(3_000.0)
    return coord


async def test_rest_samples_older_than_ws_do_not_rewind_energy(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hourly REST samples older than the latest WS sample must not rewind energy."""
    coord = await _ws_coordinator(hass, monkeypatch)
    coord.client.get_node_samples.return_value = _samples(
        (3_600.0, 9_000.0), (7_200.0, 9_800.0)
    )

    await coord.async_refresh()
    assert energy(coord, "htr", "1") == pytest.approx(10.5)
    assert power(coord, "htr", "1") == pytest.approx(3_000.0)

    # A later WS sample still advances from the WS point, not history.
    coord.handle_ws_samples(DEV_ID, _ws("1", 10_600.0, 10_600))
    assert energy(coord, "htr", "1") == pytest.approx(10.6)
    assert power(coord, "htr", "1") == pytest.approx(600.0)


@pytest.mark.parametrize(
    ("t", "counter"),
    [(9_940.0, 10_400), (10_000.0, 10_450)],
    ids=["older", "same_timestamp"],
)
async def test_stale_ws_samples_are_ignored(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch, t: float, counter: float
) -> None:
    """A WS sample not newer than the latest one never changes the counter."""
    coord = await _ws_coordinator(hass, monkeypatch)

    coord.handle_ws_samples(DEV_ID, _ws("1", t, counter))

    assert energy(coord, "htr", "1") == pytest.approx(10.5)
    assert power(coord, "htr", "1") == pytest.approx(3_000.0)


async def test_newer_lower_ws_counter_is_a_meter_reset(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A newer WS sample with a lower counter restarts energy and drops power."""
    coord = await _ws_coordinator(hass, monkeypatch)

    coord.handle_ws_samples(DEV_ID, _ws("1", 10_600.0, 200))
    assert energy(coord, "htr", "1") == pytest.approx(0.2)
    assert power(coord, "htr", "1") is None

    # A REST sample newer than the reset point keeps advancing from it.
    coord.client.get_node_samples.return_value = _samples((11_200.0, 300))
    await coord.async_refresh()
    assert energy(coord, "htr", "1") == pytest.approx(0.3)
    assert power(coord, "htr", "1") == pytest.approx(600.0)


async def test_ws_and_rest_derive_the_same_energy_and_power(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same two samples give the same result via REST or WS."""
    Clock(4000.0).install(monkeypatch)
    poll_client = rest_client(
        get_node_samples=AsyncMock(
            side_effect=[_samples((1000.0, 1200.0)), _samples((1600.0, 2400.0))]
        )
    )
    poll = energy_coordinator(hass, poll_client, {"htr": ["1"]})
    await poll.async_refresh()
    await poll.async_refresh()

    ws_client = rest_client(
        get_node_samples=AsyncMock(return_value=_samples((1000.0, 1200.0)))
    )
    ws = energy_coordinator(hass, ws_client, {"htr": ["1"]})
    await ws.async_refresh()
    ws.handle_ws_samples(
        DEV_ID, {"htr": {"1": {"samples": [{"t": 1600.0, "counter": 2400.0}]}}}
    )

    assert energy(ws, "htr", "1") == pytest.approx(energy(poll, "htr", "1"))
    assert power(ws, "htr", "1") == pytest.approx(power(poll, "htr", "1"))
    assert power(ws, "htr", "1") == pytest.approx(7200.0)


async def test_power_monitor_counters_are_watt_seconds(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Power monitor counters convert from Ws to kWh before deriving power."""
    Clock(5000.0).install(monkeypatch)
    coord = energy_coordinator(hass, rest_client(), {"pmo": ["1"]})

    coord.handle_ws_samples(DEV_ID, _ws("1", 1000.0, 3_600_000.0, "pmo"))
    coord.handle_ws_samples(DEV_ID, _ws("1", 4600.0, 7_200_000.0, "pmo"))

    assert energy(coord, "pmo", "1") == pytest.approx(2.0)
    assert power(coord, "pmo", "1") == pytest.approx(1000.0)


@pytest.mark.parametrize(
    ("payload", "kwh"),
    [
        ({"t": 100.0, "counter": 500}, 0.5),
        ({"samples": {"t": 100.0, "counter": 500}}, 0.5),
        ({"samples": [{"t": 100.0, "counter": 500}]}, 0.5),
        ({"t": 100.0, "counter": {"value": 500}}, 0.5),
        ({"t": 100.0, "counter": {"counter": 500}}, 0.5),
        ({"t": 100.0, "counter": {"min": 200, "max": 300}}, 0.3),
        ({"t": 100.0, "counter_max": 300}, 0.3),
        ({"t": 100.0, "counter_min": 200}, 0.2),
        ({"t": 100.0, "value": 700}, 0.7),
        (
            [
                {"t": 100.0, "counter": 500},
                {"t": 200.0, "counter": 600},
                {"t": 150.0, "counter": 550},
            ],
            0.6,
        ),
    ],
)
async def test_ws_sample_payload_shapes(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch, payload: Any, kwh: float
) -> None:
    """Every WS sample shape the backends send yields the latest counter."""
    Clock(1000.0).install(monkeypatch)
    coord = energy_coordinator(hass, rest_client(), {"htr": ["1"]})

    coord.handle_ws_samples(DEV_ID, {"htr": {" 1 ": payload}})

    assert energy(coord, "htr", "1") == pytest.approx(kwh)
    assert coord.data.metrics_for_type("htr")["1"].source == "ws"


@pytest.mark.parametrize(
    ("dev_id", "updates"),
    [
        ("fedcba9876543210", _ws("1", 100, 1000)),
        (DEV_ID, _ws("1", 100, 1000, "thm")),  # untracked type
        (DEV_ID, _ws("9", 100, 1000)),  # untracked address
        (DEV_ID, {"": {"1": {"t": 100, "counter": 1000}}}),
        (DEV_ID, {"htr": {"1": {"t": 100.0}}}),  # no counter
        (DEV_ID, {"htr": {"1": {"counter": 100}}}),  # no timestamp
        (DEV_ID, {"htr": {"1": [{"samples": []}, {"samples": []}]}}),
        (DEV_ID, {"htr": {"1": 42}}),
        (DEV_ID, {"htr": {"1": "string"}}),
    ],
)
async def test_unusable_ws_samples_record_nothing(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    dev_id: str,
    updates: dict,
) -> None:
    """Foreign, untracked and unparseable WS samples store no energy."""
    Clock(1000.0).install(monkeypatch)
    coord = energy_coordinator(hass, rest_client(), {"htr": ["1"]})
    initial = coord.data

    coord.handle_ws_samples(dev_id, updates, lease_seconds=60)

    assert coord.data.metrics_for_type("htr") == {}
    if dev_id != DEV_ID:
        assert coord.data is initial  # nothing published for another gateway


# --- WS lease and REST polling -----------------------------------------------


async def test_ws_lease_suspends_rest_polling_until_it_expires(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fresh WS samples skip REST polls until lease + margin, never add polls."""
    clock = Clock(0.0).install(monkeypatch)
    client = rest_client(get_node_samples=AsyncMock(return_value=_samples((0.0, 1000))))
    coord = energy_coordinator(hass, client, {"htr": ["1"]})
    await coord.async_refresh()
    client.get_node_samples.reset_mock()

    clock.now = 3600.0
    coord.handle_ws_samples(
        DEV_ID,
        {"htr": {"1": {"samples": [{"t": 3600.0, "counter": 2000.0}]}}},
        lease_seconds=300.0,
    )
    assert energy(coord, "htr", "1") == pytest.approx(2.0)
    assert power(coord, "htr", "1") == pytest.approx(1000.0)
    assert coord.update_interval is None  # WS never schedules extra REST polls

    # Deadline = 3600 + 300 lease + max(60, 300 / 4) margin = 3975.
    clock.now = 3974.0
    await coord.async_refresh()
    client.get_node_samples.assert_not_awaited()
    assert energy(coord, "htr", "1") == pytest.approx(2.0)

    clock.now = 3975.0
    await coord.async_refresh()
    client.get_node_samples.assert_awaited_once()
    assert coord.update_interval is None


@pytest.mark.parametrize(("lease", "deadline"), [(10_000.0, 10_600.0), (0.0, None)])
async def test_ws_lease_margin_is_bounded(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    lease: float,
    deadline: float | None,
) -> None:
    """The margin after a lease is capped at 10 minutes; no lease, no suspension."""
    Clock(0.0).install(monkeypatch)
    coord = energy_coordinator(hass, rest_client(), {"htr": ["1"]})

    coord.handle_ws_samples(DEV_ID, _ws("1", 1.0, 1000), lease_seconds=lease)

    assert coord.data.ws_deadline == deadline


# --- HH:05 hourly poll ---------------------------------------------------------


def _counting_client() -> AsyncMock:
    """Return a client answering every sample fetch with one fresh row."""

    async def _rows(_dev: str, _node: Any, _start: float, end: float) -> list:
        return [{"t": end, "counter": end}]

    return rest_client(get_node_samples=AsyncMock(side_effect=_rows))


class HourlyTicker:
    """Fire successive HH:05:00 time changes, one hour apart."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Start at the next HH:05:00 after now."""
        self.hass = hass
        now = dt_util.utcnow()
        self.target = now.replace(minute=5, second=0, microsecond=0)
        if self.target <= now:
            self.target += timedelta(hours=1)

    async def __call__(self) -> None:
        """Fire the next HH:05 and wait for the poll to finish."""
        async_fire_time_changed(self.hass, self.target)
        await self.hass.async_block_till_done()
        self.target += timedelta(hours=1)


async def test_hourly_poll_fetches_each_node_once_and_not_while_ws_fresh(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The HH:05 job fetches the past hour per node, skipping it while WS is fresh."""
    clock = Clock(3_900.0).install(monkeypatch)
    client = _counting_client()
    coord = energy_coordinator(hass, client, {"htr": ["1", "2"]})
    unsub = coord.async_start_hourly_poll()
    tick = HourlyTicker(hass)

    await tick()
    fetch = client.get_node_samples
    assert {c.args[1] for c in fetch.await_args_list} == {("htr", "1"), ("htr", "2")}
    _, _, start, end = fetch.await_args_list[0].args
    assert end - start == 3600

    clock.now = 7_400.0
    coord.handle_ws_samples(DEV_ID, _ws("1", 7_400.0, 9e6), lease_seconds=120)
    clock.now = 7_500.0
    await tick()
    assert fetch.await_count == 2  # WS fresh: skipped

    unsub()
    await tick()
    assert fetch.await_count == 2  # unsubscribed


@pytest.mark.parametrize("ws_every_min", [2, 4, 10, 30])
async def test_ws_samples_never_increase_rest_sample_polls(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch, ws_every_min: int
) -> None:
    """WS samples every N minutes never cause more REST fetches than no WS."""

    async def _fetches_in_one_hour(ws_every: int | None) -> int:
        clock = Clock(0.0).install(monkeypatch)
        client = _counting_client()
        coord = energy_coordinator(hass, client, {"htr": ["1", "2"]})
        await coord.async_refresh()
        unsub = coord.async_start_hourly_poll()
        tick = HourlyTicker(hass)
        client.get_node_samples.reset_mock()
        for minute in range(1, 61):
            clock.now = minute * 60.0
            if ws_every and minute % ws_every == 0:
                coord.handle_ws_samples(
                    DEV_ID, _ws("1", clock.now, clock.now), lease_seconds=120
                )
            if minute == 5:
                await tick()
        unsub()
        assert coord.update_interval is None
        return client.get_node_samples.await_count

    baseline = await _fetches_in_one_hour(None)
    with_ws = await _fetches_in_one_hour(ws_every_min)

    assert baseline == 2  # one fetch per node at HH:05
    assert with_ws <= baseline
    if ws_every_min <= 2:
        assert with_ws == 0  # WS samples fresh at HH:05: no REST at all


async def test_rate_limit_skips_following_hourly_polls_with_capped_backoff(
    hass: HomeAssistant, caplog: pytest.LogCaptureFixture
) -> None:
    """Consecutive 429s skip 1, 2, 4... hourly polls; success resets the backoff."""
    client = _counting_client()
    fetch = client.get_node_samples
    fetch.side_effect = BackendRateLimitError("429")
    coord = energy_coordinator(hass, client, {"htr": ["1", "2"]})
    unsub = coord.async_start_hourly_poll()
    tick = HourlyTicker(hass)

    async def _tick() -> int:
        before = fetch.await_count
        await tick()
        return fetch.await_count - before

    assert await _tick() == 1  # node 1 only; node 2 never called
    assert fetch.await_args.args[1] == ("htr", "1")
    assert "rate limited" in caplog.text
    assert await _tick() == 0  # skip 1
    assert await _tick() == 1  # 429 again -> skip 2
    assert await _tick() == 0
    assert await _tick() == 0
    fetch.side_effect = None
    fetch.return_value = [{"t": 1.0, "counter": 1000.0}]
    assert await _tick() == 2  # recovered: both nodes
    fetch.side_effect = BackendRateLimitError("429")
    assert await _tick() == 1
    assert await _tick() == 0  # back to a 1-poll skip
    assert await _tick() == 1

    unsub()


async def test_rate_limit_skip_count_is_capped(hass: HomeAssistant) -> None:
    """Back-to-back 429s skip 1, 2, 4, 8 hourly polls, then stay at the cap."""
    client = rest_client(
        get_node_samples=AsyncMock(side_effect=BackendRateLimitError("429"))
    )
    coord = energy_coordinator(hass, client, {"htr": ["1"]})
    unsub = coord.async_start_hourly_poll()
    tick = HourlyTicker(hass)

    skipped_runs: list[int] = []
    skipped = 0
    await tick()  # first 429
    while len(skipped_runs) < 5:
        before = client.get_node_samples.await_count
        await tick()
        if client.get_node_samples.await_count == before:
            skipped += 1
        else:
            skipped_runs.append(skipped)
            skipped = 0
    unsub()

    cap = ENERGY_RATE_LIMIT_MAX_SKIP
    assert skipped_runs == [1, 2, 4, cap, cap]


async def test_failed_hourly_poll_is_logged_not_raised(
    hass: HomeAssistant, caplog: pytest.LogCaptureFixture
) -> None:
    """A failing hourly poll marks the update failed and logs; nothing propagates."""
    client = rest_client(get_node_samples=AsyncMock(side_effect=TimeoutError))
    coord = energy_coordinator(hass, client, {"htr": ["1"]})
    unsub = coord.async_start_hourly_poll()
    tick = HourlyTicker(hass)

    with caplog.at_level(logging.DEBUG):
        await tick()

    assert coord.last_update_success is False
    assert "Hourly energy poll failed" in caplog.text
    unsub()


async def test_listeners_follow_ws_energy_updates(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Entities listening to the energy coordinator see each new WS sample."""
    Clock(1000.0).install(monkeypatch)
    coord = energy_coordinator(hass, rest_client(), {"htr": ["1"]})
    listener = MagicMock()
    unsub = coord.async_add_listener(listener)

    coord.handle_ws_samples(DEV_ID, _ws("1", 100.0, 1000))
    coord.handle_ws_samples(DEV_ID, _ws("1", 100.0, 1000))  # duplicate: no change

    assert listener.call_count == 1
    unsub()


# --- domain view -------------------------------------------------------------


def test_energy_view_only_exposes_inventory_nodes() -> None:
    """Energy metrics for nodes outside the store inventory are not readable."""
    allowed = NodeId(NodeType.HEATER, "1")
    blocked = NodeId(NodeType.HEATER, "2")
    store = DomainStateStore([allowed])
    metric = EnergyNodeMetrics(energy_kwh=1.25, power_w=250.0, source="rest", ts=1.0)
    snapshot = EnergySnapshot(
        dev_id=DEV_ID,
        metrics={
            allowed: metric,
            blocked: EnergyNodeMetrics(
                energy_kwh=9.0, power_w=900.0, source="ws", ts=2.0
            ),
        },
        updated_at=500.0,
        ws_deadline=None,
    )

    assert store.set_energy_snapshot(snapshot) is True

    view = DomainStateView(DEV_ID, store)
    assert view.get_energy_metric(NodeType.HEATER, "1") == metric
    assert view.get_energy_metric(NodeType.HEATER, "2") is None
    assert view.get_energy_metrics_for_type(NodeType.HEATER) == {"1": metric}


NOW = datetime(2026, 1, 10, 12, 30, tzinfo=UTC)
NOW_END = int(datetime(2026, 1, 10, 12, 0, tzinfo=UTC).timestamp())
HOUR = 3600
DAY = 24 * HOUR
SERVICE = "import_energy_history"
# Hourly consumption in Wh repeats every five hours, counted from an aligned origin.
PATTERN_WH = (100, 150, 200, 250, 300)
ORIGIN = (NOW_END - 4000 * DAY) // (5 * HOUR) * (5 * HOUR)


@pytest.fixture(autouse=True)
def mock_recorder_before_hass(
    async_setup_recorder_instance: RecorderInstanceGenerator,
) -> None:
    """Make the recorder database ready before hass starts."""


@pytest.fixture(autouse=True)
def limiter_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record the shared limiter's waits instead of sleeping in real time."""
    sleeps: list[float] = []

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)

    limiter = default_samples_rate_limit_state()
    limiter.reset()
    monkeypatch.setattr(limiter, "monotonic", lambda: 1000.0)
    monkeypatch.setattr(limiter, "sleep", _sleep)
    yield sleeps
    limiter.reset()


def counter_wh(ts: int) -> int:
    """Return the cumulative counter (Wh) at hour-aligned ``ts``."""
    hours = (ts - ORIGIN) // HOUR
    return hours // 5 * sum(PATTERN_WH) + sum(PATTERN_WH[: hours % 5])


def kwh_between(start: int, end: int) -> float:
    """Return consumption in kWh between two hour-aligned timestamps."""
    return (counter_wh(end) - counter_wh(start)) / 1000


def hourly_samples() -> Callable[..., Any]:
    """Return a get_node_samples fake serving on-the-hour counter samples."""

    async def _samples(
        dev_id: str, node: tuple[str, str], start: float, end: float
    ) -> list[dict[str, Any]]:
        lo = int(start) - int(start) % HOUR
        return [
            {"t": ts, "counter": str(counter_wh(ts))}
            for ts in range(lo, int(end) + 1, HOUR)
            if ts >= start
        ]

    return _samples


async def _setup(hass: HomeAssistant, entry: MockConfigEntry) -> str:
    """Set up the entry and return the heater energy sensor's entity id."""
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    entity_id = er.async_get(hass).async_get_entity_id(
        "sensor", DOMAIN, build_heater_energy_unique_id(DEV_ID, "htr", "1")
    )
    assert entity_id is not None
    return entity_id


async def _import(hass: HomeAssistant, **data: Any) -> None:
    """Call the service and wait for the recorder to commit."""
    await hass.services.async_call(DOMAIN, SERVICE, data, blocking=True)
    await async_wait_recording_done(hass)


async def _sums(
    hass: HomeAssistant, entity_id: str, period: str = "hour"
) -> dict[int, float]:
    """Return ``{start_ts: sum}`` for every statistic of ``entity_id``."""
    rows = await get_instance(hass).async_add_executor_job(
        statistics_during_period,
        hass,
        datetime(2000, 1, 1, tzinfo=UTC),
        None,
        {entity_id},
        period,
        None,
        {"sum"},
    )
    return {int(row["start"]): row["sum"] for row in rows.get(entity_id, [])}


def _seed(
    hass: HomeAssistant, entity_id: str, rows: dict[int, float], table: Any = None
) -> None:
    """Write pre-existing statistics the way the recorder's sensor platform does."""
    metadata = {
        "has_sum": True,
        "mean_type": StatisticMeanType.NONE,
        "name": None,
        "source": "recorder",
        "statistic_id": entity_id,
        "unit_class": EnergyConverter.UNIT_CLASS,
        "unit_of_measurement": "kWh",
    }
    stats = [
        {"start": datetime.fromtimestamp(ts, UTC), "sum": value}
        for ts, value in rows.items()
    ]
    if table is None:
        async_import_statistics(hass, metadata, stats)
    else:
        get_instance(hass).async_import_statistics(metadata, stats, table)


def _progress(hass_storage: dict[str, Any], entry: MockConfigEntry) -> dict:
    """Return the stored per-node progress for ``entry``."""
    return hass_storage[f"{DOMAIN}.energy_import.{entry.entry_id}"]["data"]["nodes"]


def _assert_continuous(sums: dict[int, float], start: int, end: int, base: float):
    """Assert hourly sums from ``start`` to ``end`` follow the samples from ``base``."""
    for hour in range(start, end, HOUR):
        assert sums[hour] == pytest.approx(base + kwh_between(start, hour + HOUR))


async def test_import_writes_hourly_statistics(
    recorder_mock: Recorder,
    hass: HomeAssistant,
    freezer: FrozenDateTimeFactory,
    cloud: FakeCloud,
    config_entry: MockConfigEntry,
    hass_storage: dict[str, Any],
    limiter_sleeps: list[float],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A service call writes one statistic per hour with recorder metadata."""
    freezer.move_to(NOW)
    entity_id = await _setup(hass, config_entry)
    cloud.get_node_samples.reset_mock()
    cloud.get_node_samples.side_effect = hourly_samples()

    await _import(hass, max_history_retrieval=2)

    start = NOW_END - 2 * DAY
    sums = await _sums(hass, entity_id)
    assert sorted(sums) == list(range(start, NOW_END, HOUR))
    _assert_continuous(sums, start, NOW_END, 0.0)
    windows = [call.args[2:] for call in cloud.get_node_samples.call_args_list]
    assert windows == [(start, start + DAY), (start + DAY, NOW_END)]
    # Every fetch passed the shared limiter; the second one waited 0.5 s (2 q/s).
    assert limiter_sleeps == [pytest.approx(0.5)]

    meta = await get_instance(hass).async_add_executor_job(
        lambda: get_metadata(hass, statistic_ids={entity_id})
    )
    _, metadata = meta[entity_id]
    assert metadata["mean_type"] is StatisticMeanType.NONE
    assert metadata["has_sum"] is True
    assert metadata["source"] == "recorder"
    assert metadata["unit_of_measurement"] == "kWh"
    assert metadata["unit_class"] == EnergyConverter.UNIT_CLASS
    # HA >= 2026.11 rejects imports whose metadata omits unit_class.
    assert "doesn't specify unit_class" not in caplog.text
    assert _progress(hass_storage, config_entry) == {"htr:1": {"imported_from": start}}
    summary = config_entry.runtime_data.last_energy_import_summary
    assert summary["nodes"][0]["written"] == 48
    assert summary["nodes"][0]["requests"] == 2


async def test_consumption_is_booked_at_the_hour_it_happened(
    recorder_mock: Recorder,
    hass: HomeAssistant,
    freezer: FrozenDateTimeFactory,
    cloud: FakeCloud,
    config_entry: MockConfigEntry,
) -> None:
    """Counter delta between 22:00 and 23:00 lands in the statistic starting 22:00."""
    freezer.move_to(NOW)
    entity_id = await _setup(hass, config_entry)
    t22 = NOW_END - 14 * HOUR
    samples = [
        {"t": t22, "counter": "1000"},
        {"t": t22 + HOUR, "counter": "1500"},
        {"t": t22 + HOUR, "counter": "1500"},  # duplicate timestamp
        {"t": t22 + 2 * HOUR, "counter": "bad"},  # invalid sample
        {"t": t22 + 3 * HOUR, "counter": "2500"},
        {"t": t22 + 4 * HOUR, "counter": "100"},  # counter reset
        {"t": t22 + 5 * HOUR, "counter": "400"},
        {"t": t22 + 6 * HOUR, "counter": "350"},  # jitter below the reset threshold
        {"t": t22 - 2 * DAY, "counter": "0"},  # outside the requested chunk
    ]
    cloud.get_node_samples.side_effect = None
    cloud.get_node_samples.return_value = samples

    await _import(hass, max_history_retrieval=1)

    sums = await _sums(hass, entity_id)
    assert min(sums) == t22  # no statistics before the first sample
    assert sums[t22] == pytest.approx(0.5)
    assert sums[t22 + HOUR] == pytest.approx(1.5)  # 23:00 -> 01:00 gap
    assert sums[t22 + 2 * HOUR] == pytest.approx(1.5)
    assert sums[t22 + 3 * HOUR] == pytest.approx(1.5)  # reset: no negative step
    assert sums[t22 + 4 * HOUR] == pytest.approx(1.8)
    assert sums[NOW_END - HOUR] == pytest.approx(1.8)  # carried to the window end
    summary = config_entry.runtime_data.last_energy_import_summary
    assert summary["nodes"][0]["resets"] == 1


async def test_import_continues_from_previous_statistic(
    recorder_mock: Recorder,
    hass: HomeAssistant,
    freezer: FrozenDateTimeFactory,
    cloud: FakeCloud,
    config_entry: MockConfigEntry,
) -> None:
    """The imported series starts from the sum of the statistic before the window."""
    freezer.move_to(NOW)
    entity_id = await _setup(hass, config_entry)
    start = NOW_END - DAY
    _seed(hass, entity_id, {start - 5 * HOUR: 90.0, start - 3 * HOUR: 100.0})
    await async_wait_recording_done(hass)
    cloud.get_node_samples.side_effect = hourly_samples()

    await _import(hass, max_history_retrieval=1)

    sums = await _sums(hass, entity_id)
    assert sums[start - 3 * HOUR] == 100.0
    _assert_continuous(sums, start, NOW_END, 100.0)


async def test_overlap_is_overwritten_and_later_sums_follow(
    recorder_mock: Recorder,
    hass: HomeAssistant,
    freezer: FrozenDateTimeFactory,
    cloud: FakeCloud,
    config_entry: MockConfigEntry,
) -> None:
    """Existing rows in the window are replaced; later rows shift with no seam."""
    freezer.move_to(NOW)
    entity_id = await _setup(hass, config_entry)
    start = NOW_END - DAY
    # Recorder-compiled rows from 6 h before the window end, plus the current
    # hour (hourly and short-term), all on an unrelated base of 5000.
    existing = {
        ts: 5000.0 + (ts - NOW_END) / HOUR
        for ts in range(NOW_END - 6 * HOUR, NOW_END + HOUR, HOUR)
    }
    _seed(hass, entity_id, existing)
    _seed(hass, entity_id, {NOW_END + 5 * 60: 5000.5}, StatisticsShortTerm)
    await async_wait_recording_done(hass)
    cloud.get_node_samples.side_effect = hourly_samples()

    await _import(hass, max_history_retrieval=1)

    sums = await _sums(hass, entity_id)
    _assert_continuous(sums, start, NOW_END, 0.0)
    imported_end = kwh_between(start, NOW_END)
    # The row at NOW_END kept its own 1 kWh hour on top of the imported total.
    assert sums[NOW_END] == pytest.approx(imported_end + 1.0)
    short = await _sums(hass, entity_id, "5minute")
    assert short[NOW_END + 5 * 60] == pytest.approx(imported_end + 1.5)


async def test_larger_history_imports_the_older_range(
    recorder_mock: Recorder,
    hass: HomeAssistant,
    freezer: FrozenDateTimeFactory,
    cloud: FakeCloud,
    config_entry: MockConfigEntry,
    hass_storage: dict[str, Any],
) -> None:
    """A later call with more days fetches only the older range and rewrites forward."""
    freezer.move_to(NOW)
    entity_id = await _setup(hass, config_entry)
    cloud.get_node_samples.side_effect = hourly_samples()
    await _import(hass, max_history_retrieval=1)

    cloud.get_node_samples.reset_mock()
    await _import(hass)  # default 1 week, already covered back to 1 day
    await _import(hass, max_history_retrieval=1)
    assert cloud.get_node_samples.call_count == 6  # only the 6 missing days

    start = NOW_END - 7 * DAY
    windows = [call.args[2:] for call in cloud.get_node_samples.call_args_list]
    assert windows[0] == (start, start + DAY)
    assert windows[-1] == (NOW_END - 2 * DAY, NOW_END - DAY)
    sums = await _sums(hass, entity_id)
    assert sorted(sums) == list(range(start, NOW_END, HOUR))
    _assert_continuous(sums, start, NOW_END, 0.0)
    assert _progress(hass_storage, config_entry) == {"htr:1": {"imported_from": start}}

    cloud.get_node_samples.reset_mock()
    await _import(hass, max_history_retrieval=2, reset_progress=True)
    assert cloud.get_node_samples.call_count == 2
    sums = await _sums(hass, entity_id)
    # Re-imported in place: rows before the window keep their sums and the two
    # re-imported days continue from them.
    _assert_continuous(sums, start, NOW_END, 0.0)


@pytest.mark.parametrize(
    "error",
    [
        BackendAuthError("auth"),
        BackendRateLimitError("429"),
        ClientError("down"),
        TimeoutError(),
    ],
)
async def test_fetch_error_keeps_progress_and_resume_fills_the_gap(
    recorder_mock: Recorder,
    hass: HomeAssistant,
    freezer: FrozenDateTimeFactory,
    cloud: FakeCloud,
    config_entry: MockConfigEntry,
    hass_storage: dict[str, Any],
    error: Exception,
) -> None:
    """A failed fetch stops the import at the last written day; rerun resumes there."""
    freezer.move_to(NOW)
    entity_id = await _setup(hass, config_entry)
    start = NOW_END - 3 * DAY
    good = hourly_samples()
    calls = 0

    async def _flaky(*args: Any) -> list[dict[str, Any]]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise error
        return await good(*args)

    cloud.get_node_samples.side_effect = _flaky
    with pytest.raises(HomeAssistantError, match="progress is saved"):
        await _import(hass, max_history_retrieval=3)
    await async_wait_recording_done(hass)

    window = _progress(hass_storage, config_entry)["htr:1"]["window"]
    assert window["next"] == start + DAY
    assert max(await _sums(hass, entity_id)) == start + DAY - HOUR

    cloud.get_node_samples.reset_mock()
    await _import(hass, max_history_retrieval=3)

    windows = [call.args[2:] for call in cloud.get_node_samples.call_args_list]
    assert windows[0] == (start + DAY, start + 2 * DAY)
    sums = await _sums(hass, entity_id)
    assert sorted(sums) == list(range(start, NOW_END, HOUR))
    _assert_continuous(sums, start, NOW_END, 0.0)


async def test_second_call_while_running_is_rejected(
    recorder_mock: Recorder,
    hass: HomeAssistant,
    freezer: FrozenDateTimeFactory,
    cloud: FakeCloud,
    config_entry: MockConfigEntry,
) -> None:
    """A second call during an import raises instead of fetching the same days."""
    freezer.move_to(NOW)
    await _setup(hass, config_entry)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def _blocked(*args: Any) -> list[dict[str, Any]]:
        entered.set()
        await release.wait()
        return []

    cloud.get_node_samples.reset_mock()
    cloud.get_node_samples.side_effect = _blocked
    first = hass.async_create_task(_import(hass, max_history_retrieval=1))
    await entered.wait()

    with pytest.raises(ServiceValidationError, match="already running"):
        await hass.services.async_call(DOMAIN, SERVICE, {}, blocking=True)

    release.set()
    await first
    assert cloud.get_node_samples.call_count == 1


@pytest.mark.parametrize("value", [0, -1, 3651, "many"])
async def test_invalid_max_history_is_rejected(
    recorder_mock: Recorder,
    hass: HomeAssistant,
    freezer: FrozenDateTimeFactory,
    cloud: FakeCloud,
    config_entry: MockConfigEntry,
    value: Any,
) -> None:
    """Out-of-range or non-numeric day counts fail validation; nothing is fetched."""
    freezer.move_to(NOW)
    await _setup(hass, config_entry)
    cloud.get_node_samples.reset_mock()

    with pytest.raises(vol.Invalid):
        await hass.services.async_call(
            DOMAIN, SERVICE, {"max_history_retrieval": value}, blocking=True
        )
    cloud.get_node_samples.assert_not_called()


async def test_service_without_loaded_entry_raises(
    recorder_mock: Recorder,
    hass: HomeAssistant,
    freezer: FrozenDateTimeFactory,
    cloud: FakeCloud,
    config_entry: MockConfigEntry,
    hass_storage: dict[str, Any],
) -> None:
    """After unload the service reports that no entry can import; removal drops storage."""
    freezer.move_to(NOW)
    await _setup(hass, config_entry)
    cloud.get_node_samples.side_effect = hourly_samples()
    await _import(hass, max_history_retrieval="1")
    key = f"{DOMAIN}.energy_import.{config_entry.entry_id}"
    assert key in hass_storage

    assert await hass.config_entries.async_unload(config_entry.entry_id)
    with pytest.raises(ServiceValidationError, match="No loaded TermoWeb entry"):
        await hass.services.async_call(DOMAIN, SERVICE, {}, blocking=True)

    await hass.config_entries.async_remove(config_entry.entry_id)
    await hass.async_block_till_done()
    assert key not in hass_storage


async def test_node_without_energy_sensor_is_skipped(
    recorder_mock: Recorder,
    hass: HomeAssistant,
    freezer: FrozenDateTimeFactory,
    cloud: FakeCloud,
    config_entry: MockConfigEntry,
) -> None:
    """A node whose energy sensor is gone is not fetched."""
    freezer.move_to(NOW)
    entity_id = await _setup(hass, config_entry)
    er.async_get(hass).async_remove(entity_id)
    cloud.get_node_samples.reset_mock()

    await _import(hass, max_history_retrieval=1)

    cloud.get_node_samples.assert_not_called()


@pytest.mark.parametrize(
    ("imported", "expected_calls"),
    [(True, 0), (False, 7)],
)
async def test_legacy_option_progress_moves_to_storage(
    recorder_mock: Recorder,
    hass: HomeAssistant,
    freezer: FrozenDateTimeFactory,
    cloud: FakeCloud,
    config_entry: MockConfigEntry,
    hass_storage: dict[str, Any],
    imported: bool,
    expected_calls: int,
) -> None:
    """Legacy options are removed; only a completed legacy import is trusted."""
    freezer.move_to(NOW)
    legacy_from = NOW_END - 10 * DAY + 600
    entry = MockConfigEntry(
        domain=DOMAIN,
        unique_id=config_entry.unique_id,
        data=dict(config_entry.data),
        options={
            "energy_history_progress": {"htr:1": legacy_from, "1": legacy_from},
            "energy_history_imported": imported,
            "max_history_retrieved": 30,
            "keep": "me",
        },
    )
    await _setup(hass, entry)
    cloud.get_node_samples.reset_mock()
    cloud.get_node_samples.side_effect = hourly_samples()

    await _import(hass)

    assert dict(entry.options) == {"keep": "me"}
    assert cloud.get_node_samples.call_count == expected_calls
    expected_from = legacy_from - 600 if imported else NOW_END - 7 * DAY
    assert _progress(hass_storage, entry) == {"htr:1": {"imported_from": expected_from}}


def test_default_samples_rate_limit_state_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shared rate limiter is reused, throttles calls and can be reset."""

    current = 0.1

    def fake_monotonic() -> float:
        return current

    sleep_calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        nonlocal current
        sleep_calls.append(delay)
        current += delay

    limiter = default_samples_rate_limit_state()
    assert isinstance(limiter, MonotonicRateLimiter)
    assert isinstance(limiter.lock, asyncio.Lock)
    assert default_samples_rate_limit_state() is limiter

    monkeypatch.setattr(limiter, "monotonic", fake_monotonic)
    monkeypatch.setattr(limiter, "sleep", fake_sleep)
    reset_samples_rate_limit_state()

    asyncio.run(limiter.async_throttle())
    assert sleep_calls == [pytest.approx(0.4)]  # 2 queries per second

    current = 0.7
    asyncio.run(limiter.async_throttle())
    assert sleep_calls == [pytest.approx(0.4), pytest.approx(0.3)]

    reset_samples_rate_limit_state()
    asyncio.run(limiter.async_throttle())
    assert sleep_calls == [pytest.approx(0.4), pytest.approx(0.3)]

    reset_samples_rate_limit_state()


def test_async_throttle_invokes_on_wait_callback() -> None:
    """Rate limiter should report the computed delay to on_wait callback."""

    monotonic_values = iter([0.2, 1.2])

    def fake_monotonic() -> float:
        return next(monotonic_values)

    sleep_calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    on_wait_calls: list[float] = []

    def on_wait(delay: float) -> None:
        on_wait_calls.append(delay)

    limiter = MonotonicRateLimiter(
        lock=asyncio.Lock(),
        monotonic=fake_monotonic,
        sleep=fake_sleep,
        min_interval=1.0,
    )

    result = asyncio.run(limiter.async_throttle(on_wait=on_wait))

    assert result == pytest.approx(0.8)
    assert on_wait_calls == [pytest.approx(0.8)]
    assert sleep_calls == [pytest.approx(0.8)]
