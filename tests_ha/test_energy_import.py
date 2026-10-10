"""Energy history import against the real recorder (statistics tables)."""

# Pytest fixtures arrive as positional arguments.
# ruff: noqa: PLR0917

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

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
from homeassistant.util.unit_conversion import EnergyConverter
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry
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
from custom_components.termoweb.identifiers import build_heater_energy_unique_id
from custom_components.termoweb.throttle import default_samples_rate_limit_state

from .conftest import DEV_ID, FakeCloud

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
    summary = hass.data[DOMAIN][config_entry.entry_id].last_energy_import_summary
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
    summary = hass.data[DOMAIN][config_entry.entry_id].last_energy_import_summary
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
