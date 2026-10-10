"""EnergyStateCoordinator: REST/WS energy samples, derived power, hourly poll."""

from __future__ import annotations

from datetime import timedelta
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from aiohttp import ClientError
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import UpdateFailed
from homeassistant.util import dt as dt_util
import pytest
from pytest_homeassistant_custom_component.common import async_fire_time_changed

from custom_components.termoweb.backend.rest_client import (
    BackendAuthError,
    BackendRateLimitError,
)
from custom_components.termoweb.coordinator import (
    ENERGY_RATE_LIMIT_MAX_SKIP,
    EnergyStateCoordinator,
)
from custom_components.termoweb.domain.energy import EnergyNodeMetrics, EnergySnapshot
from custom_components.termoweb.domain.ids import NodeId, NodeType
from custom_components.termoweb.domain.state import DomainStateStore
from custom_components.termoweb.domain.view import DomainStateView
from tests_ha.fakes.coordinator import (
    DEV_ID,
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
