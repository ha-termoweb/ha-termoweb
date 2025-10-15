"""Tests for writing energy history into entity-backed statistics."""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from custom_components.termoweb import energy
from custom_components.termoweb.energy import SUMMARY_KEY_LAST_RUN


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


class _ImmediateRateLimiter:
    async def async_throttle(self, on_wait=None):  # type: ignore[override]
        if callable(on_wait):
            on_wait(0.0)
        return 0.0


@pytest.fixture
def stub_hass() -> SimpleNamespace:
    """Return a minimal Home Assistant stub."""

    config_entries = _StubConfigEntries()
    return SimpleNamespace(data={}, config_entries=config_entries)


@pytest.mark.asyncio
async def test_import_clears_both_ids_and_populates_dot_series(
    monkeypatch: pytest.MonkeyPatch,
    stub_hass: SimpleNamespace,
    inventory_from_map,
) -> None:
    """Importer should clear dot/colon ids then write only to the dot id."""

    entry = _StubConfigEntry("entry")
    stub_hass.config_entries.add(entry)

    inventory = inventory_from_map({"htr": ["A"]}, dev_id="dev-dot")
    client = ModuleType("client")

    async def _get_node_samples(dev_id, node, start, stop):
        return [
            {"t": 1_700_000_000, "counter": 1_000},
            {"t": 1_700_003_600, "counter": 2_000},
            {"t": 1_700_007_200, "counter": 3_500},
        ]

    client.get_node_samples = _get_node_samples  # type: ignore[attr-defined]

    stub_hass.data.setdefault(energy.DOMAIN, {})[entry.entry_id] = {
        "client": client,
        "dev_id": "dev-dot",
        "inventory": inventory,
    }

    entity_id = "sensor.dev_dot_energy"
    colon_id = "sensor:dev_dot_energy"

    class _Registry:
        def async_get_entity_id(
            self, domain: str, platform: str, unique_id: str
        ) -> str | None:
            if domain == "sensor" and platform == energy.DOMAIN:
                return entity_id
            return None

        def async_get(self, requested: str) -> SimpleNamespace | None:
            if requested == entity_id:
                return SimpleNamespace(original_name="Device Energy")
            return None

    registry = _Registry()
    monkeypatch.setattr(energy.er, "async_get", lambda hass: registry, raising=False)

    import_start = datetime(2023, 11, 14, 1, tzinfo=UTC)
    import_end = datetime(2023, 11, 14, 3, tzinfo=UTC)

    existing_rows: dict[str, list[dict[str, Any]]] = {
        entity_id: [
            {"start": import_start, "sum": 4.0},
            {"start": import_start + timedelta(hours=1), "sum": 4.5},
        ],
        colon_id: [
            {"start": import_start, "sum": 5.0},
        ],
    }

    async def _fake_stats_period(hass, start, end, stat_ids):
        return {stat_id: list(existing_rows.get(stat_id, [])) for stat_id in stat_ids}

    cleared_ids: list[str] = []

    async def _fake_clear_statistics(
        hass,
        statistic_id: str,
        *,
        start_time: datetime,
        end_time: datetime,
    ) -> str:
        cleared_ids.append(statistic_id)
        existing_rows[statistic_id] = []
        return "clear"

    written_rows: list[list[dict[str, Any]]] = []

    async def _capture_store(
        hass,
        metadata: dict[str, Any],
        stats: list[dict[str, Any]],
    ) -> None:
        written_rows.append(list(stats))
        existing = {row["start"]: dict(row) for row in existing_rows.get(entity_id, [])}
        for row in stats:
            existing[row["start"]] = dict(row)
        existing_rows[entity_id] = sorted(
            existing.values(), key=lambda row: row["start"]
        )

    async def _noop_guard(*args, **kwargs) -> None:
        return None

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
    monkeypatch.setattr(energy, "_enforce_monotonic_sum", _noop_guard, raising=False)

    await energy.async_import_energy_history(
        stub_hass,
        entry,
        rate_limit=_ImmediateRateLimiter(),
        max_days=1,
    )

    assert colon_id in cleared_ids and entity_id in cleared_ids
    dot_rows = existing_rows.get(entity_id, [])
    colon_rows = existing_rows.get(colon_id, [])
    assert not colon_rows
    assert written_rows
    assert len(dot_rows) == 2
    assert all("sum" in row for row in dot_rows)
    stored = written_rows[0]
    assert len(stored) == 2
    assert all("state" not in row for row in stored)
    assert stored[0]["sum"] < stored[1]["sum"]


@pytest.mark.asyncio
async def test_import_uses_colon_history_for_sum_offset(
    monkeypatch: pytest.MonkeyPatch,
    stub_hass: SimpleNamespace,
    inventory_from_map,
) -> None:
    """Importer should adopt the newest pre-window colon sum as offset."""

    entry = _StubConfigEntry("entry")
    stub_hass.config_entries.add(entry)

    inventory = inventory_from_map({"htr": ["B"]}, dev_id="dev-colon")
    client = ModuleType("client")

    base_ts = 1_700_500_000
    samples = [
        {"t": base_ts, "counter": 2_000},
        {"t": base_ts + 3_600, "counter": 2_500},
    ]

    async def _get_node_samples(dev_id, node, start, stop):
        return list(samples)

    client.get_node_samples = _get_node_samples  # type: ignore[attr-defined]

    stub_hass.data.setdefault(energy.DOMAIN, {})[entry.entry_id] = {
        "client": client,
        "dev_id": "dev-colon",
        "inventory": inventory,
    }

    entity_id = "sensor.dev_colon_energy"
    colon_id = "sensor:dev_colon_energy"

    class _Registry:
        def async_get_entity_id(self, domain, platform, unique_id):
            if domain == "sensor" and platform == energy.DOMAIN:
                return entity_id
            return None

        def async_get(self, requested):
            if requested == entity_id:
                return SimpleNamespace(original_name="Colon Offset")
            return None

    registry = _Registry()
    monkeypatch.setattr(energy.er, "async_get", lambda hass: registry, raising=False)

    import_start = datetime.fromtimestamp(base_ts, UTC).replace(
        minute=0, second=0, microsecond=0
    )
    colon_pre_rows = [
        {"start": import_start - timedelta(hours=1), "sum": 12.0, "state": 12.5}
    ]

    async def _fake_stats_period(hass, start, end, stat_ids):
        result: dict[str, list[dict[str, Any]]] = {}
        for stat_id in stat_ids:
            if stat_id == colon_id:
                result[stat_id] = list(colon_pre_rows)
            else:
                result[stat_id] = []
        return result

    async def _fake_clear_statistics(**kwargs):
        return "clear"

    captured: list[dict[str, Any]] = []

    async def _capture_store(hass, metadata, stats):
        captured.extend(stats)

    async def _noop_guard(*args, **kwargs) -> None:
        return None

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
    monkeypatch.setattr(energy, "_enforce_monotonic_sum", _noop_guard, raising=False)

    await energy.async_import_energy_history(
        stub_hass,
        entry,
        rate_limit=_ImmediateRateLimiter(),
        max_days=1,
    )

    assert captured
    first = captured[0]
    assert first["sum"] == pytest.approx(12.5)
    assert first["start"] == import_start + timedelta(hours=1)


@pytest.mark.asyncio
async def test_import_guard_clamps_descending_live_hour(
    monkeypatch: pytest.MonkeyPatch,
    stub_hass: SimpleNamespace,
    inventory_from_map,
) -> None:
    """Monotonic guard should rewrite the live hour when sum decreases."""

    entry = _StubConfigEntry("entry")
    stub_hass.config_entries.add(entry)

    inventory = inventory_from_map({"htr": ["C"]}, dev_id="dev-guard")
    client = ModuleType("client")

    base_ts = 1_700_800_000
    samples = [
        {"t": base_ts, "counter": 500},
        {"t": base_ts + 3_600, "counter": 1_500},
        {"t": base_ts + 7_200, "counter": 2_000},
    ]

    async def _get_node_samples(dev_id, node, start, stop):
        return list(samples)

    client.get_node_samples = _get_node_samples  # type: ignore[attr-defined]

    stub_hass.data.setdefault(energy.DOMAIN, {})[entry.entry_id] = {
        "client": client,
        "dev_id": "dev-guard",
        "inventory": inventory,
    }

    entity_id = "sensor.dev_guard_energy"

    class _Registry:
        def async_get_entity_id(self, domain, platform, unique_id):
            if domain == "sensor" and platform == energy.DOMAIN:
                return entity_id
            return None

        def async_get(self, requested):
            if requested == entity_id:
                return SimpleNamespace(original_name="Guard Energy")
            return None

    registry = _Registry()
    monkeypatch.setattr(energy.er, "async_get", lambda hass: registry, raising=False)

    import_start_dt = datetime.fromtimestamp(base_ts, UTC).replace(
        minute=0, second=0, microsecond=0
    )
    import_end_dt = datetime.fromtimestamp(samples[-1]["t"], UTC).replace(
        minute=0, second=0, microsecond=0
    )

    seam_rows = [
        {"start": import_start_dt - timedelta(hours=1), "sum": 2.0},
        {"start": import_end_dt + timedelta(hours=1), "sum": 3.4},
        {"start": import_end_dt + timedelta(hours=2), "sum": 3.0},
    ]

    async def _fake_stats_period(hass, start, end, stat_ids):
        ids = set(stat_ids)
        if ids == {entity_id}:
            return {entity_id: seam_rows}
        return {}

    async def _fake_clear_statistics(**kwargs):
        return "clear"

    stored_calls: list[list[dict[str, Any]]] = []

    async def _capture_store(hass, metadata, stats):
        stored_calls.append(list(stats))

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

    statistics_mod = ModuleType("homeassistant.components.recorder.statistics")

    async def _noop_import_statistics(hass, metadata, stats):
        return None

    statistics_mod.async_import_statistics = _noop_import_statistics
    recorder_mod = ModuleType("homeassistant.components.recorder")
    recorder_mod.statistics = statistics_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules,
        "homeassistant.components.recorder.statistics",
        statistics_mod,
    )
    monkeypatch.setitem(
        sys.modules,
        "homeassistant.components.recorder",
        recorder_mod,
    )

    real_store = energy._store_statistics

    async def _store_wrapper(hass, metadata, stats):
        await _capture_store(hass, metadata, stats)
        await real_store(hass, metadata, stats)

    monkeypatch.setattr(energy, "_store_statistics", _store_wrapper, raising=False)

    await energy.async_import_energy_history(
        stub_hass,
        entry,
        rate_limit=_ImmediateRateLimiter(),
        max_days=1,
    )

    assert len(stored_calls) >= 2
    seam_update = stored_calls[-1]
    assert seam_update == [
        {"start": import_end_dt + timedelta(hours=2), "sum": seam_rows[1]["sum"]}
    ]

    summary = stub_hass.data[energy.DOMAIN][entry.entry_id][SUMMARY_KEY_LAST_RUN]
    node_summary = summary["nodes"][0]
    assert node_summary["written"] >= 2
