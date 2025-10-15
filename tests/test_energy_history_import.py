from __future__ import annotations

import sys
from importlib.machinery import ModuleSpec
from datetime import UTC, datetime, timedelta
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from custom_components.termoweb import energy
from custom_components.termoweb.energy import OPTION_ENERGY_HISTORY_PROGRESS
from tests.test_energy_recorder_imports import _install_fake_homeassistant


class _RecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, str], int, int]] = []

    async def get_node_samples(
        self,
        dev_id: str,
        node: tuple[str, str],
        start: int,
        stop: int,
    ) -> list[dict[str, Any]]:
        self.calls.append((dev_id, node, start, stop))
        return []


class _ImmediateRateLimiter:
    async def async_throttle(self, on_wait=None):  # type: ignore[override]
        if callable(on_wait):
            on_wait(0.0)
        return 0.0


class _FixedDatetime(datetime):
    _NOW = datetime(2024, 1, 2, 15, 45, 33, tzinfo=UTC)

    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        base = cls._NOW
        if tz is None:
            tz = base.tzinfo
        return cls(
            base.year,
            base.month,
            base.day,
            base.hour,
            base.minute,
            base.second,
            base.microsecond,
            tzinfo=tz,
        )


class _StubConfigEntry:
    def __init__(self, entry_id: str) -> None:
        self.entry_id = entry_id
        self.data: dict[str, Any] = {}
        self.options: dict[str, Any] = {}


class _StubConfigEntries:
    def __init__(self) -> None:
        self._entries: dict[str, _StubConfigEntry] = {}
        self.updated_entries: list[
            tuple[_StubConfigEntry, dict[str, Any] | None, dict[str, Any] | None]
        ] = []

    def add(self, entry: _StubConfigEntry) -> None:
        self._entries[entry.entry_id] = entry

    def async_update_entry(
        self,
        entry: _StubConfigEntry,
        *,
        data: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        if data is not None:
            entry.data = data
        if options is not None:
            entry.options = options
        self.updated_entries.append((entry, data, options))


@pytest.fixture
def stub_hass() -> SimpleNamespace:
    """Return a minimal Home Assistant stub for energy history tests."""

    config_entries = _StubConfigEntries()
    return SimpleNamespace(data={}, config_entries=config_entries)


@pytest.mark.asyncio
async def test_import_energy_history_fetches_until_current_minute(
    monkeypatch: pytest.MonkeyPatch,
    stub_hass,
    inventory_from_map,
) -> None:
    """Importer should request history through the current minute."""

    entry = _StubConfigEntry("entry")
    stub_hass.config_entries.add(entry)

    inventory = inventory_from_map({"htr": ["A"]}, dev_id="dev-1")
    client = _RecordingClient()

    stub_hass.data.setdefault(energy.DOMAIN, {})[entry.entry_id] = {
        "client": client,
        "dev_id": "dev-1",
        "inventory": inventory,
    }

    monkeypatch.setattr(energy, "datetime", _FixedDatetime, raising=False)
    monkeypatch.setattr(energy.er, "async_get", lambda hass: None, raising=False)

    await energy.async_import_energy_history(
        stub_hass,
        entry,
        rate_limit=_ImmediateRateLimiter(),
        max_days=1,
    )

    assert client.calls
    _, node, start, stop = client.calls[0]
    assert node == ("htr", "A")

    expected_stop = int(
        _FixedDatetime._NOW.replace(second=0, microsecond=0).timestamp()
    )
    assert stop == expected_stop
    assert start == expected_stop - 24 * 3600

    assert stub_hass.config_entries.updated_entries
    last_update = stub_hass.config_entries.updated_entries[-1]
    _, _, options = last_update
    assert isinstance(options, dict)
    assert OPTION_ENERGY_HISTORY_PROGRESS in options


def test_store_statistics_writes_external_series(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_store_statistics must always write to the external statistic id."""

    hass = object()
    metadata = {
        "source": "recorder",
        "statistic_id": "sensor.test_energy",
        "name": "Test Energy",
        "unit_of_measurement": "kWh",
        "has_sum": True,
        "has_mean": False,
    }
    stats = [{"start": datetime(2024, 1, 1, tzinfo=UTC), "sum": 5.0}]
    captured: dict[str, Any] = {}

    recorder_mod = ModuleType("homeassistant.components.recorder")
    statistics_mod = ModuleType("homeassistant.components.recorder.statistics")
    statistics_mod.__spec__ = ModuleSpec(  # type: ignore[attr-defined]
        "homeassistant.components.recorder.statistics", loader=None, is_package=False
    )

    def _capture_external_stats(hass_arg, metadata_arg, stats_arg) -> None:
        captured["hass"] = hass_arg
        captured["metadata"] = metadata_arg
        captured["stats"] = stats_arg

    statistics_mod.async_add_external_statistics = _capture_external_stats
    recorder_mod.statistics = statistics_mod  # type: ignore[attr-defined]

    _install_fake_homeassistant(monkeypatch, recorder_mod)
    monkeypatch.setitem(
        sys.modules,
        "homeassistant.components.recorder.statistics",
        statistics_mod,
    )

    energy._store_statistics(hass, metadata, stats)

    assert captured["hass"] is hass
    assert captured["metadata"]["statistic_id"] == "sensor:test_energy"
    assert captured["metadata"]["source"] == "sensor"
    assert captured["stats"] == stats
    assert metadata["statistic_id"] == "sensor.test_energy"


@pytest.mark.asyncio
async def test_import_energy_history_uses_union_statistics_for_offset(
    monkeypatch: pytest.MonkeyPatch,
    stub_hass,
    inventory_from_map,
) -> None:
    """Importer should compute offsets and clear overlaps across both statistic ids."""

    entry = _StubConfigEntry("entry")
    stub_hass.config_entries.add(entry)

    base_ts = 1_700_000_000
    samples = [
        {"t": base_ts, "counter": 1000},
        {"t": base_ts + 3600, "counter": 2000},
        {"t": base_ts + 7200, "counter": 3500},
        {"t": base_ts + 10_800, "counter": 4500},
    ]

    class _SampleClient:
        def __init__(self, payload: list[dict[str, int]]) -> None:
            self.payload = payload
            self.calls: list[tuple[str, tuple[str, str], int, int]] = []

        async def get_node_samples(
            self,
            dev_id: str,
            node: tuple[str, str],
            start: int,
            stop: int,
        ) -> list[dict[str, int]]:
            self.calls.append((dev_id, node, start, stop))
            return self.payload

    client = _SampleClient(samples)

    inventory = inventory_from_map({"htr": ["A"]}, dev_id="dev-1")
    stub_hass.data.setdefault(energy.DOMAIN, {})[entry.entry_id] = {
        "client": client,
        "dev_id": "dev-1",
        "inventory": inventory,
    }

    entity_id = "sensor.test_energy"

    class _Registry:
        def async_get_entity_id(
            self, domain: str, platform: str, unique_id: str
        ) -> str | None:
            if domain == "sensor" and platform == energy.DOMAIN:
                return entity_id
            return None

        def async_get(self, requested: str) -> SimpleNamespace | None:
            if requested == entity_id:
                return SimpleNamespace(original_name="Test Energy")
            return None

    registry = _Registry()

    monkeypatch.setattr(energy.er, "async_get", lambda hass: registry, raising=False)

    import_start_dt = datetime.fromtimestamp(base_ts, UTC).replace(
        minute=0, second=0, microsecond=0
    )
    import_end_dt = datetime.fromtimestamp(samples[-1]["t"], UTC).replace(
        minute=0, second=0, microsecond=0
    )

    dot_rows = [
        {
            "start": import_start_dt - timedelta(hours=2),
            "sum": 3.0,
            "state": 3.5,
        }
    ]
    external_rows = [
        {
            "start": import_start_dt - timedelta(hours=1),
            "sum": 4.0,
            "state": 4.5,
        },
        {
            "start": import_start_dt,
            "sum": 4.7,
            "state": 4.7,
        },
    ]

    requested_stat_sets: list[set[str]] = []

    async def _fake_stats_period(
        hass,
        start_time,
        end_time,
        statistic_ids,
    ) -> dict[str, list[dict[str, Any]]]:
        ids = set(statistic_ids)
        requested_stat_sets.append(ids)
        assert ids == {entity_id, "sensor:test_energy"}
        return {entity_id: dot_rows, "sensor:test_energy": external_rows}

    clear_calls: list[tuple[str, datetime, datetime]] = []

    async def _fake_clear_statistics(
        hass,
        statistic_id: str,
        *,
        start_time: datetime,
        end_time: datetime,
    ) -> str:
        clear_calls.append((statistic_id, start_time, end_time))
        return "clear"

    stored_stats: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []

    def _capture_store(
        hass,
        metadata: dict[str, Any],
        stats: list[dict[str, Any]],
    ) -> None:
        stored_stats.append((metadata, stats))

    monkeypatch.setattr(
        energy,
        "_statistics_during_period_compat",
        _fake_stats_period,
        raising=False,
    )
    monkeypatch.setattr(
        energy,
        "_clear_statistics_compat",
        _fake_clear_statistics,
        raising=False,
    )
    monkeypatch.setattr(energy, "_store_statistics", _capture_store, raising=False)

    await energy.async_import_energy_history(
        stub_hass,
        entry,
        rate_limit=_ImmediateRateLimiter(),
        max_days=1,
    )

    assert requested_stat_sets
    assert requested_stat_sets[0] == {entity_id, "sensor:test_energy"}

    assert len(clear_calls) == 2
    assert {call[0] for call in clear_calls} == {entity_id, "sensor:test_energy"}
    for _, start_time, end_time in clear_calls:
        assert start_time == import_start_dt
        assert end_time == import_end_dt + timedelta(hours=1)

    assert stored_stats
    stored_metadata, stored_entries = stored_stats[0]
    assert stored_metadata["statistic_id"] == entity_id
    assert len(stored_entries) == 3
    assert all(entry.keys() == {"start", "sum"} for entry in stored_entries)
    assert stored_entries[0]["sum"] == pytest.approx(5.0), stored_entries
    assert stored_entries[-1]["sum"] == pytest.approx(7.5)
