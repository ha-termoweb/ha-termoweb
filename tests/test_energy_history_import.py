from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from conftest import build_entry_runtime
from custom_components.termoweb import energy
from custom_components.termoweb.energy import OPTION_ENERGY_HISTORY_PROGRESS


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

    build_entry_runtime(
        hass=stub_hass,
        entry_id=entry.entry_id,
        dev_id="dev-1",
        inventory=inventory,
        client=client,
        config_entry=entry,
    )

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


@pytest.mark.asyncio
async def test_import_energy_history_rejects_unsupported_node_types(
    stub_hass,
    inventory_from_map,
) -> None:
    """Importer should reject node types missing from the inventory."""

    entry = _StubConfigEntry("entry-node-types")
    stub_hass.config_entries.add(entry)

    inventory = inventory_from_map({"htr": ["A"]}, dev_id="dev-node-type")
    client = _RecordingClient()

    build_entry_runtime(
        hass=stub_hass,
        entry_id=entry.entry_id,
        dev_id="dev-node-type",
        inventory=inventory,
        client=client,
        config_entry=entry,
    )

    with pytest.raises(ValueError) as err:
        await energy.async_import_energy_history(
            stub_hass,
            entry,
            rate_limit=_ImmediateRateLimiter(),
            node_types=("unknown",),
        )

    assert "Unsupported node_types" in str(err.value)
    assert client.calls == []


@pytest.mark.asyncio
async def test_import_energy_history_filters_unknown_addresses(
    monkeypatch: pytest.MonkeyPatch,
    stub_hass,
    inventory_from_map,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Importer should ignore unknown addresses while logging a warning."""

    entry = _StubConfigEntry("entry-addresses")
    stub_hass.config_entries.add(entry)

    inventory = inventory_from_map({"htr": ["A"]}, dev_id="dev-address")
    client = _RecordingClient()

    build_entry_runtime(
        hass=stub_hass,
        entry_id=entry.entry_id,
        dev_id="dev-address",
        inventory=inventory,
        client=client,
        config_entry=entry,
    )

    monkeypatch.setattr(energy, "datetime", _FixedDatetime, raising=False)
    monkeypatch.setattr(energy.er, "async_get", lambda hass: None, raising=False)

    async def _fake_stats_period(*args, **kwargs) -> dict[str, list[dict[str, Any]]]:
        return {}

    async def _fake_clear_stats(*args, **kwargs) -> str:
        return "clear"

    async def _capture_store(*args, **kwargs) -> None:
        return None

    async def _noop_enforce(*args, **kwargs) -> None:
        return None

    monkeypatch.setattr(
        energy,
        "_statistics_during_period",
        _fake_stats_period,
        raising=False,
    )
    monkeypatch.setattr(
        energy,
        "_clear_statistics",
        _fake_clear_stats,
        raising=False,
    )
    monkeypatch.setattr(energy, "_store_statistics", _capture_store, raising=False)
    monkeypatch.setattr(energy, "_enforce_monotonic_sum", _noop_enforce, raising=False)

    with caplog.at_level(logging.WARNING):
        await energy.async_import_energy_history(
            stub_hass,
            entry,
            rate_limit=_ImmediateRateLimiter(),
            max_days=1,
            addresses=("A", "Z"),
        )

    assert client.calls
    assert client.calls[0][1] == ("htr", "A")
    assert "unknown addresses" in caplog.text


@pytest.mark.asyncio
async def test_store_statistics_imports_entity_series(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_store_statistics must import rows into the entity statistic id."""

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

    async def _capture_import_stats(hass_arg, metadata_arg, stats_arg) -> None:
        captured["hass"] = hass_arg
        captured["metadata"] = metadata_arg
        captured["stats"] = stats_arg

    monkeypatch.setattr(
        energy.recorder_stats,
        "async_import_statistics",
        _capture_import_stats,
        raising=False,
    )

    await energy._store_statistics(hass, metadata, stats)

    assert captured["hass"] is hass
    assert captured["metadata"]["statistic_id"] == "sensor.test_energy"
    assert captured["metadata"]["source"] == "recorder"
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
    build_entry_runtime(
        hass=stub_hass,
        entry_id=entry.entry_id,
        dev_id="dev-1",
        inventory=inventory,
        client=client,
        config_entry=entry,
    )

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

    external_id = "sensor:test_energy"
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

    remaining_rows: dict[str, list[dict[str, Any]]] = {
        entity_id: [dict(row) for row in dot_rows],
        external_id: [dict(row) for row in external_rows],
    }

    requested_stat_sets: list[set[str]] = []

    async def _fake_stats_period(
        hass,
        start_time,
        end_time,
        statistic_ids,
    ) -> dict[str, list[dict[str, Any]]]:
        ids = set(statistic_ids)
        requested_stat_sets.append(ids)
        response: dict[str, list[dict[str, Any]]] = {}
        for stat_id in ids:
            rows = [
                dict(row)
                for row in remaining_rows.get(stat_id, [])
                if start_time <= row["start"] < end_time
            ]
            response[stat_id] = rows
        return response

    clear_calls: list[tuple[str, datetime, datetime]] = []

    async def _fake_clear_statistics(
        hass,
        statistic_id: str,
        *,
        start_time: datetime,
        end_time: datetime,
    ) -> str:
        clear_calls.append((statistic_id, start_time, end_time))
        existing = remaining_rows.get(statistic_id, [])
        if existing:
            remaining_rows[statistic_id] = [
                row for row in existing if not (start_time <= row["start"] < end_time)
            ]
        return "clear"

    stored_stats: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []

    async def _capture_store(
        hass,
        metadata: dict[str, Any],
        stats: list[dict[str, Any]],
    ) -> None:
        stored_stats.append((metadata, stats))
        stat_id = metadata["statistic_id"]
        existing = {row["start"]: dict(row) for row in remaining_rows.get(stat_id, [])}
        for row in stats:
            existing[row["start"]] = dict(row)
        remaining_rows[stat_id] = sorted(
            existing.values(), key=lambda row: row["start"]
        )

    monkeypatch.setattr(energy, "_statistics_during_period", _fake_stats_period)
    monkeypatch.setattr(energy, "_clear_statistics", _fake_clear_statistics)
    monkeypatch.setattr(energy, "_store_statistics", _capture_store, raising=False)

    await energy.async_import_energy_history(
        stub_hass,
        entry,
        rate_limit=_ImmediateRateLimiter(),
        max_days=1,
    )

    assert requested_stat_sets
    assert requested_stat_sets[0] == {entity_id, external_id}
    assert requested_stat_sets[-1] == {entity_id}

    assert len(clear_calls) == 2
    assert {call[0] for call in clear_calls} == {entity_id, external_id}
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

    remaining_dot = [
        row
        for row in remaining_rows.get(entity_id, [])
        if import_start_dt <= row["start"] < import_end_dt + timedelta(hours=1)
    ]
    assert len(remaining_dot) == 3
    assert remaining_dot[0]["sum"] == pytest.approx(5.0)
    assert remaining_dot[-1]["sum"] == pytest.approx(7.5)

    remaining_external = [
        row
        for row in remaining_rows.get(external_id, [])
        if import_start_dt <= row["start"] < import_end_dt + timedelta(hours=1)
    ]
    assert not remaining_external


@pytest.mark.asyncio
async def test_import_skips_duplicate_sample_timestamps(
    monkeypatch: pytest.MonkeyPatch,
    stub_hass,
    inventory_from_map,
) -> None:
    """Importer should ignore duplicate raw samples sharing the same timestamp."""

    entry = _StubConfigEntry("entry")
    stub_hass.config_entries.add(entry)

    base_ts = 1_700_100_000
    samples = [
        {"t": base_ts, "counter": 1_000},
        {"t": base_ts, "counter": 1_200},
        {"t": base_ts + 3600, "counter": 2_200},
    ]

    class _SampleClient:
        def __init__(self, payload: list[dict[str, int]]) -> None:
            self.payload = payload

        async def get_node_samples(self, dev_id, node, start, stop):
            return self.payload

    client = _SampleClient(samples)

    inventory = inventory_from_map({"htr": ["A"]}, dev_id="dev-dup")
    build_entry_runtime(
        hass=stub_hass,
        entry_id=entry.entry_id,
        dev_id="dev-dup",
        inventory=inventory,
        client=client,
        config_entry=entry,
    )

    entity_id = "sensor.dup_energy"
    external_id = "sensor:dup_energy"

    class _Registry:
        def async_get_entity_id(self, domain, platform, unique_id):
            if domain == "sensor" and platform == energy.DOMAIN:
                return entity_id
            return None

        def async_get(self, requested):
            if requested == entity_id:
                return SimpleNamespace(original_name="Dup Energy")
            return None

    registry = _Registry()
    monkeypatch.setattr(energy.er, "async_get", lambda hass: registry, raising=False)

    async def _fake_stats_period(hass, start_time, end_time, statistic_ids):
        ids = set(statistic_ids)
        if ids == {entity_id, external_id}:
            return {entity_id: [], external_id: []}
        if ids == {external_id}:
            return {external_id: []}
        return {}

    async def _fake_clear_statistics(
        hass,
        statistic_id: str,
        *,
        start_time: datetime,
        end_time: datetime,
    ) -> str:
        return "clear"

    stored_payloads: list[list[dict[str, Any]]] = []

    async def _capture_store(
        hass, metadata: dict[str, Any], stats: list[dict[str, Any]]
    ) -> None:
        stored_payloads.append(stats)

    async def _noop_enforce(*args, **kwargs):
        return None

    monkeypatch.setattr(
        energy,
        "_statistics_during_period",
        _fake_stats_period,
        raising=False,
    )
    monkeypatch.setattr(
        energy,
        "_clear_statistics",
        _fake_clear_statistics,
        raising=False,
    )
    monkeypatch.setattr(energy, "_store_statistics", _capture_store, raising=False)
    monkeypatch.setattr(energy, "_enforce_monotonic_sum", _noop_enforce, raising=False)

    await energy.async_import_energy_history(
        stub_hass,
        entry,
        rate_limit=_ImmediateRateLimiter(),
        max_days=1,
    )

    assert stored_payloads, "No statistics were recorded"
    stats = stored_payloads[0]
    assert len(stats) == 1
    expected_start = datetime.fromtimestamp(base_ts + 3600, UTC).replace(
        minute=0, second=0, microsecond=0
    )
    assert stats[0]["start"] == expected_start
    assert stats[0]["sum"] == pytest.approx(1.2)


@pytest.mark.asyncio
async def test_import_coalesces_samples_within_hour(
    monkeypatch: pytest.MonkeyPatch,
    stub_hass,
    inventory_from_map,
) -> None:
    """Importer should merge multiple samples that fall within one hour."""

    entry = _StubConfigEntry("entry")
    stub_hass.config_entries.add(entry)

    base_ts = 1_700_300_000
    samples = [
        {"t": base_ts, "counter": 1_000},
        {"t": base_ts + 60, "counter": 1_300},
        {"t": base_ts + 120, "counter": 1_600},
        {"t": base_ts + 3_600, "counter": 2_600},
    ]

    class _SampleClient:
        def __init__(self, payload: list[dict[str, int]]) -> None:
            self.payload = payload

        async def get_node_samples(self, dev_id, node, start, stop):
            return self.payload

    client = _SampleClient(samples)

    inventory = inventory_from_map({"htr": ["A"]}, dev_id="dev-merge")
    build_entry_runtime(
        hass=stub_hass,
        entry_id=entry.entry_id,
        dev_id="dev-merge",
        inventory=inventory,
        client=client,
        config_entry=entry,
    )

    entity_id = "sensor.merge_energy"
    external_id = "sensor:merge_energy"

    class _Registry:
        def async_get_entity_id(self, domain, platform, unique_id):
            if domain == "sensor" and platform == energy.DOMAIN:
                return entity_id
            return None

        def async_get(self, requested):
            if requested == entity_id:
                return SimpleNamespace(original_name="Merge Energy")
            return None

    registry = _Registry()
    monkeypatch.setattr(energy.er, "async_get", lambda hass: registry, raising=False)

    async def _fake_stats_period(hass, start_time, end_time, statistic_ids):
        ids = set(statistic_ids)
        if ids == {entity_id, external_id}:
            return {entity_id: [], external_id: []}
        if ids == {external_id}:
            return {external_id: []}
        return {}

    async def _fake_clear_statistics(
        hass,
        statistic_id: str,
        *,
        start_time: datetime,
        end_time: datetime,
    ) -> str:
        return "clear"

    stored_payloads: list[list[dict[str, Any]]] = []

    async def _capture_store(
        hass, metadata: dict[str, Any], stats: list[dict[str, Any]]
    ) -> None:
        stored_payloads.append(stats)

    async def _noop_enforce(*args, **kwargs):
        return None

    monkeypatch.setattr(
        energy,
        "_statistics_during_period",
        _fake_stats_period,
        raising=False,
    )
    monkeypatch.setattr(
        energy,
        "_clear_statistics",
        _fake_clear_statistics,
        raising=False,
    )
    monkeypatch.setattr(energy, "_store_statistics", _capture_store, raising=False)
    monkeypatch.setattr(energy, "_enforce_monotonic_sum", _noop_enforce, raising=False)

    await energy.async_import_energy_history(
        stub_hass,
        entry,
        rate_limit=_ImmediateRateLimiter(),
        max_days=1,
    )

    assert stored_payloads, "No statistics were recorded"
    stats = stored_payloads[0]
    assert len(stats) == 2

    first_start = datetime.fromtimestamp(base_ts, UTC).replace(
        minute=0, second=0, microsecond=0
    )
    second_start = datetime.fromtimestamp(base_ts + 3_600, UTC).replace(
        minute=0, second=0, microsecond=0
    )

    assert stats[0]["start"] == first_start
    assert stats[1]["start"] == second_start
    assert stats[0]["sum"] == pytest.approx(0.6)
    assert stats[1]["sum"] == pytest.approx(1.6)


@pytest.mark.asyncio
async def test_import_skips_negative_deltas_after_reset(
    monkeypatch: pytest.MonkeyPatch,
    stub_hass,
    inventory_from_map,
) -> None:
    """Importer should drop negative deltas caused by counter resets."""

    entry = _StubConfigEntry("entry")
    stub_hass.config_entries.add(entry)

    base_ts = 1_700_200_000
    samples = [
        {"t": base_ts, "counter": 5_000},
        {"t": base_ts + 3600, "counter": 200},
        {"t": base_ts + 7200, "counter": 700},
    ]

    class _SampleClient:
        def __init__(self, payload: list[dict[str, int]]) -> None:
            self.payload = payload

        async def get_node_samples(self, dev_id, node, start, stop):
            return self.payload

    client = _SampleClient(samples)

    inventory = inventory_from_map({"htr": ["B"]}, dev_id="dev-reset")
    runtime = build_entry_runtime(
        hass=stub_hass,
        entry_id=entry.entry_id,
        dev_id="dev-reset",
        inventory=inventory,
        client=client,
        config_entry=entry,
    )

    entity_id = "sensor.reset_energy"
    external_id = "sensor:reset_energy"

    class _Registry:
        def async_get_entity_id(self, domain, platform, unique_id):
            if domain == "sensor" and platform == energy.DOMAIN:
                return entity_id
            return None

        def async_get(self, requested):
            if requested == entity_id:
                return SimpleNamespace(original_name="Reset Energy")
            return None

    registry = _Registry()
    monkeypatch.setattr(energy.er, "async_get", lambda hass: registry, raising=False)

    async def _fake_stats_period(hass, start_time, end_time, statistic_ids):
        ids = set(statistic_ids)
        if ids == {entity_id, external_id}:
            return {entity_id: [], external_id: []}
        if ids == {external_id}:
            return {external_id: []}
        return {}

    async def _fake_clear_statistics(
        hass,
        statistic_id: str,
        *,
        start_time: datetime,
        end_time: datetime,
    ) -> str:
        return "clear"

    stored_payloads: list[list[dict[str, Any]]] = []

    async def _capture_store(
        hass, metadata: dict[str, Any], stats: list[dict[str, Any]]
    ) -> None:
        stored_payloads.append(stats)

    async def _noop_enforce(*args, **kwargs):
        return None

    monkeypatch.setattr(
        energy,
        "_statistics_during_period",
        _fake_stats_period,
        raising=False,
    )
    monkeypatch.setattr(
        energy,
        "_clear_statistics",
        _fake_clear_statistics,
        raising=False,
    )
    monkeypatch.setattr(energy, "_store_statistics", _capture_store, raising=False)
    monkeypatch.setattr(energy, "_enforce_monotonic_sum", _noop_enforce, raising=False)

    await energy.async_import_energy_history(
        stub_hass,
        entry,
        rate_limit=_ImmediateRateLimiter(),
        max_days=1,
    )

    assert stored_payloads, "No statistics were recorded"
    stats = stored_payloads[0]
    assert len(stats) == 1
    expected_start = datetime.fromtimestamp(base_ts + 7200, UTC).replace(
        minute=0, second=0, microsecond=0
    )
    assert stats[0]["start"] == expected_start
    assert stats[0]["sum"] == pytest.approx(0.5)

    summary = runtime.last_energy_import_summary
    node_summary = summary["nodes"][0]
    assert node_summary["resets"] == 1


@pytest.mark.asyncio
async def test_enforce_monotonic_sum_clamps_descending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Monotonic guard should clamp descending sums within the window."""

    entity_id = "sensor.foo_energy"
    import_start = datetime(2024, 1, 1, 12, tzinfo=UTC)
    import_end = import_start + timedelta(hours=1)

    seam_rows = [
        {"start": import_start - timedelta(hours=1), "sum": 8.0},
        {"start": import_end, "sum": 7.5},
    ]

    async def _fake_collect_statistics(hass, stat_id, start, end):
        assert stat_id == entity_id
        assert start == import_start - timedelta(hours=1)
        assert end == import_end + timedelta(hours=6)
        return list(seam_rows)

    monkeypatch.setattr(
        energy,
        "_collect_statistics",
        _fake_collect_statistics,
        raising=False,
    )
    monkeypatch.setattr(energy.er, "async_get", lambda hass: None, raising=False)

    rewrites: list[dict[str, Any]] = []

    async def _capture_store(
        hass, metadata: dict[str, Any], rows: list[dict[str, Any]]
    ) -> None:
        rewrites.extend(rows)
        assert metadata["statistic_id"] == entity_id

    monkeypatch.setattr(energy, "_store_statistics", _capture_store, raising=False)

    await energy._enforce_monotonic_sum(
        SimpleNamespace(),
        entity_id,
        import_start,
        import_end,
    )

    assert rewrites == [{"start": import_end, "sum": 8.0}]


@pytest.mark.asyncio
async def test_import_enforces_monotonic_sum_at_seam(
    monkeypatch: pytest.MonkeyPatch,
    stub_hass,
    inventory_from_map,
) -> None:
    """Importer must clamp recorder seam hours using the monotonic guard."""

    entry = _StubConfigEntry("entry")
    stub_hass.config_entries.add(entry)

    base_ts = 1_700_100_000
    samples = [
        {"t": base_ts, "counter": 1_000},
        {"t": base_ts + 3_600, "counter": 2_000},
        {"t": base_ts + 7_200, "counter": 3_400},
    ]

    class _SampleClient:
        def __init__(self, payload: list[dict[str, int]]) -> None:
            self.payload = payload

        async def get_node_samples(
            self,
            dev_id: str,
            node: tuple[str, str],
            start: int,
            stop: int,
        ) -> list[dict[str, int]]:
            return self.payload

    client = _SampleClient(samples)

    inventory = inventory_from_map({"htr": ["A"]}, dev_id="dev-1")
    build_entry_runtime(
        hass=stub_hass,
        entry_id=entry.entry_id,
        dev_id="dev-1",
        inventory=inventory,
        client=client,
        config_entry=entry,
    )

    entity_id = "sensor.test_energy"
    external_id = "sensor:test_energy"

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

    seam_rows = [
        {"start": import_start_dt - timedelta(hours=1), "sum": 4.0},
        {"start": import_end_dt, "sum": 6.5},
        {"start": import_end_dt + timedelta(hours=1), "sum": 6.0},
    ]

    async def _fake_stats_period(
        hass,
        start_time,
        end_time,
        statistic_ids,
    ) -> dict[str, list[dict[str, Any]]]:
        ids = set(statistic_ids)
        if ids == {entity_id, external_id}:
            return {}
        if ids == {entity_id}:
            return {entity_id: seam_rows}
        return {}

    monkeypatch.setattr(
        energy,
        "_statistics_during_period",
        _fake_stats_period,
        raising=False,
    )

    async def _fake_clear_statistics(
        hass,
        statistic_id: str,
        *,
        start_time: datetime,
        end_time: datetime,
    ) -> str:
        return "clear"

    monkeypatch.setattr(
        energy,
        "_clear_statistics",
        _fake_clear_statistics,
        raising=False,
    )

    stored_stats: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []

    async def _capture_store(
        hass,
        metadata: dict[str, Any],
        stats: list[dict[str, Any]],
    ) -> None:
        stored_stats.append((metadata, list(stats)))

    monkeypatch.setattr(energy, "_store_statistics", _capture_store, raising=False)

    await energy.async_import_energy_history(
        stub_hass,
        entry,
        rate_limit=_ImmediateRateLimiter(),
        max_days=1,
    )

    assert stored_stats, "Import should write statistics"
    assert len(stored_stats) >= 2
    metadata, initial_rows = stored_stats[0]
    assert metadata["statistic_id"] == entity_id
    assert len(initial_rows) >= 2
    _, seam_rewrite = stored_stats[-1]
    assert seam_rewrite == [{"start": import_end_dt + timedelta(hours=1), "sum": 6.5}]


# ---------------------------------------------------------------------------
# _store_statistics edge cases (lines 113, 117)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_statistics_empty_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    """_store_statistics should return early for empty stats list (line 113)."""

    calls: list[Any] = []

    async def _capture(*args: Any) -> None:
        calls.append(args)

    monkeypatch.setattr(
        energy.recorder_stats, "async_import_statistics", _capture, raising=False
    )

    await energy._store_statistics(object(), {"statistic_id": "sensor.test"}, [])
    assert calls == []


@pytest.mark.asyncio
async def test_store_statistics_missing_stat_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """_store_statistics should raise for missing/invalid statistic_id (line 117)."""

    with pytest.raises(ValueError, match="statistic_id"):
        await energy._store_statistics(object(), {}, [{"start": "x", "sum": 1.0}])

    with pytest.raises(ValueError, match="statistic_id"):
        await energy._store_statistics(
            object(), {"statistic_id": "nope"}, [{"start": "x", "sum": 1.0}]
        )


# ---------------------------------------------------------------------------
# _collect_statistics: empty period (line 158)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collect_statistics_empty_period(monkeypatch: pytest.MonkeyPatch) -> None:
    """_collect_statistics should return empty list for empty period (line 158)."""

    async def _empty_period(*args: Any, **kwargs: Any) -> dict:
        return {}

    monkeypatch.setattr(energy, "_statistics_during_period", _empty_period)

    result = await energy._collect_statistics(
        object(),
        "sensor.test",
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 2, tzinfo=UTC),
    )
    assert result == []


# ---------------------------------------------------------------------------
# _enforce_monotonic_sum edge cases (lines 175, 182, 197-198, 202)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enforce_monotonic_sum_no_dot_in_entity_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_enforce_monotonic_sum should return early if no dot in entity_id (line 175)."""

    await energy._enforce_monotonic_sum(
        object(), "nope", datetime.now(UTC), datetime.now(UTC)
    )


@pytest.mark.asyncio
async def test_enforce_monotonic_sum_no_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """_enforce_monotonic_sum should return early with empty rows (line 182)."""

    async def _empty_collect(*args: Any, **kwargs: Any) -> list:
        return []

    monkeypatch.setattr(energy, "_collect_statistics", _empty_collect)

    await energy._enforce_monotonic_sum(
        object(), "sensor.test", datetime.now(UTC), datetime.now(UTC)
    )


@pytest.mark.asyncio
async def test_enforce_monotonic_sum_with_non_monotonic_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_enforce_monotonic_sum should rewrite non-monotonic sums (lines 197-198, 202)."""

    now = datetime(2024, 1, 1, tzinfo=UTC)
    rows = [
        {"start": now, "sum": None, "state": None},  # sum=None, skipped by first pass
        {"start": now + timedelta(hours=1), "sum": 5.0, "state": 5.0},
        {"start": now + timedelta(hours=2), "sum": 3.0, "state": 3.0},  # decrease!
        {"start": now + timedelta(hours=3), "sum": "invalid", "state": None},  # bad
    ]

    async def _fake_collect(*args: Any, **kwargs: Any) -> list:
        return rows

    stored: list[tuple[dict, list]] = []

    async def _fake_store(hass: Any, metadata: dict, stats: list) -> None:
        stored.append((metadata, stats))

    class _FakeRegistry:
        def async_get(self, entity_id: str) -> Any:
            return SimpleNamespace(original_name="Test")

    monkeypatch.setattr(energy, "_collect_statistics", _fake_collect)
    monkeypatch.setattr(energy, "_store_statistics", _fake_store)
    monkeypatch.setattr(
        energy.er, "async_get", lambda hass: _FakeRegistry(), raising=False
    )

    await energy._enforce_monotonic_sum(object(), "sensor.test", now, now + timedelta(hours=3))

    assert len(stored) == 1
    _, rewrites = stored[0]
    # The decrease at hour 2 and the invalid at hour 3 should both be rewritten to 5.0
    assert len(rewrites) == 2
    assert all(r["sum"] == 5.0 for r in rewrites)


# ---------------------------------------------------------------------------
# _statistics_during_period (line 78)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_statistics_during_period_calls_recorder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_statistics_during_period should call recorder with correct params (line 78)."""

    captured: dict[str, Any] = {}

    async def _fake_get_stats(
        hass: Any,
        start_time: Any,
        end_time: Any,
        statistic_ids: Any,
        period: str = "hour",
        types: set[str] | None = None,
    ) -> dict:
        captured["start"] = start_time
        captured["end"] = end_time
        captured["ids"] = statistic_ids
        captured["period"] = period
        captured["types"] = types
        return {}

    monkeypatch.setattr(
        energy.recorder_stats,
        "async_get_statistics_during_period",
        _fake_get_stats,
        raising=False,
    )

    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 2, tzinfo=UTC)
    result = await energy._statistics_during_period(
        object(), start, end, {"sensor.test"}
    )
    assert result == {}
    assert captured["period"] == "hour"
    assert captured["types"] == {"state", "sum"}


# ---------------------------------------------------------------------------
# async_import_energy_history: various guard branches
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_import_energy_history_rejects_non_inventory_nodes(
    stub_hass,
    inventory_from_map,
) -> None:
    """Importer should raise TypeError for non-Inventory nodes argument (line 266)."""

    entry = _StubConfigEntry("entry-type-guard")
    stub_hass.config_entries.add(entry)

    inventory = inventory_from_map({"htr": ["A"]}, dev_id="dev-type-guard")
    build_entry_runtime(
        hass=stub_hass,
        entry_id=entry.entry_id,
        dev_id="dev-type-guard",
        inventory=inventory,
        client=_RecordingClient(),
        config_entry=entry,
    )

    with pytest.raises(TypeError, match="Inventory instance"):
        await energy.async_import_energy_history(
            stub_hass,
            entry,
            nodes="not-an-inventory",  # type: ignore[arg-type]
            rate_limit=_ImmediateRateLimiter(),
        )


@pytest.mark.asyncio
async def test_import_energy_history_no_inventory_in_runtime(
    monkeypatch: pytest.MonkeyPatch,
    stub_hass,
    inventory_from_map,
) -> None:
    """Importer should log error when inventory is missing from runtime (lines 275-279)."""

    entry = _StubConfigEntry("entry-no-inv")
    stub_hass.config_entries.add(entry)

    runtime = build_entry_runtime(
        hass=stub_hass,
        entry_id=entry.entry_id,
        dev_id="dev-no-inv",
        allow_missing_inventory=True,
        client=_RecordingClient(),
        config_entry=entry,
    )
    runtime.inventory = None

    monkeypatch.setattr(energy, "datetime", _FixedDatetime, raising=False)

    await energy.async_import_energy_history(
        stub_hass,
        entry,
        rate_limit=_ImmediateRateLimiter(),
    )


@pytest.mark.asyncio
async def test_import_energy_history_already_imported(
    stub_hass,
    inventory_from_map,
) -> None:
    """Importer should return early when already imported (line 441-442)."""

    entry = _StubConfigEntry("entry-done")
    entry.options[energy.OPTION_ENERGY_HISTORY_IMPORTED] = True
    stub_hass.config_entries.add(entry)

    inventory = inventory_from_map({"htr": ["A"]}, dev_id="dev-done")
    client = _RecordingClient()
    build_entry_runtime(
        hass=stub_hass,
        entry_id=entry.entry_id,
        dev_id="dev-done",
        inventory=inventory,
        client=client,
        config_entry=entry,
    )

    await energy.async_import_energy_history(
        stub_hass,
        entry,
        rate_limit=_ImmediateRateLimiter(),
    )
    assert client.calls == []


@pytest.mark.asyncio
async def test_import_energy_history_day_chunk_validation(
    monkeypatch: pytest.MonkeyPatch,
    stub_hass,
    inventory_from_map,
) -> None:
    """Importer should handle invalid/negative/large day_chunk_hours (lines 377-394)."""

    entry = _StubConfigEntry("entry-chunk")
    stub_hass.config_entries.add(entry)

    inventory = inventory_from_map({"htr": ["A"]}, dev_id="dev-chunk")
    client = _RecordingClient()
    build_entry_runtime(
        hass=stub_hass,
        entry_id=entry.entry_id,
        dev_id="dev-chunk",
        inventory=inventory,
        client=client,
        config_entry=entry,
    )

    monkeypatch.setattr(energy, "datetime", _FixedDatetime, raising=False)
    monkeypatch.setattr(energy.er, "async_get", lambda hass: None, raising=False)

    # Test negative day_chunk_hours (should default to 24)
    await energy.async_import_energy_history(
        stub_hass,
        entry,
        rate_limit=_ImmediateRateLimiter(),
        max_days=1,
        day_chunk_hours=-5,
    )
    assert client.calls  # Should still work with default

    # Reset
    client.calls.clear()
    entry.options.pop(energy.OPTION_ENERGY_HISTORY_IMPORTED, None)

    # Test > 24 (should be capped to 24)
    await energy.async_import_energy_history(
        stub_hass,
        entry,
        rate_limit=_ImmediateRateLimiter(),
        max_days=1,
        day_chunk_hours=48,
    )


@pytest.mark.asyncio
async def test_import_energy_history_no_entity_id(
    monkeypatch: pytest.MonkeyPatch,
    stub_hass,
    inventory_from_map,
) -> None:
    """Importer should continue when no entity_id is found (lines 578-610)."""

    entry = _StubConfigEntry("entry-no-entity")
    stub_hass.config_entries.add(entry)

    base_ts = 1_700_000_000

    class _SampleClient:
        async def get_node_samples(self, *args: Any) -> list[dict[str, int]]:
            return [{"t": base_ts, "counter": 1000}]

    inventory = inventory_from_map({"htr": ["A"]}, dev_id="dev-no-entity")
    build_entry_runtime(
        hass=stub_hass,
        entry_id=entry.entry_id,
        dev_id="dev-no-entity",
        inventory=inventory,
        client=_SampleClient(),
        config_entry=entry,
    )

    monkeypatch.setattr(energy, "datetime", _FixedDatetime, raising=False)
    # Return None for all entity lookups
    monkeypatch.setattr(energy.er, "async_get", lambda hass: None, raising=False)

    await energy.async_import_energy_history(
        stub_hass,
        entry,
        rate_limit=_ImmediateRateLimiter(),
        max_days=1,
    )


@pytest.mark.asyncio
async def test_import_energy_history_no_valid_timestamps(
    monkeypatch: pytest.MonkeyPatch,
    stub_hass,
    inventory_from_map,
) -> None:
    """Importer should handle samples with no valid timestamps (lines 585-610)."""

    entry = _StubConfigEntry("entry-bad-ts")
    stub_hass.config_entries.add(entry)

    class _SampleClient:
        async def get_node_samples(self, *args: Any) -> list[dict[str, Any]]:
            return [{"t": "invalid", "counter": 1000}]

    inventory = inventory_from_map({"htr": ["A"]}, dev_id="dev-bad-ts")
    build_entry_runtime(
        hass=stub_hass,
        entry_id=entry.entry_id,
        dev_id="dev-bad-ts",
        inventory=inventory,
        client=_SampleClient(),
        config_entry=entry,
    )

    monkeypatch.setattr(energy, "datetime", _FixedDatetime, raising=False)

    class _FakeRegistry:
        def async_get_entity_id(self, *args: Any) -> str:
            return "sensor.test"

        def async_get(self, *args: Any) -> Any:
            return SimpleNamespace(original_name="Test")

    monkeypatch.setattr(
        energy.er, "async_get", lambda hass: _FakeRegistry(), raising=False
    )

    await energy.async_import_energy_history(
        stub_hass,
        entry,
        rate_limit=_ImmediateRateLimiter(),
        max_days=1,
    )


@pytest.mark.asyncio
async def test_import_energy_history_empty_stats_after_processing(
    monkeypatch: pytest.MonkeyPatch,
    stub_hass,
    inventory_from_map,
) -> None:
    """Importer should handle the case when stats list is empty after processing (lines 819-845)."""

    entry = _StubConfigEntry("entry-empty-stats")
    stub_hass.config_entries.add(entry)

    base_ts = 1_700_000_000
    # Only one sample = no delta to compute
    samples = [{"t": base_ts, "counter": 1000}]

    class _SampleClient:
        async def get_node_samples(self, *args: Any) -> list[dict[str, int]]:
            return samples

    inventory = inventory_from_map({"htr": ["A"]}, dev_id="dev-empty-stats")
    build_entry_runtime(
        hass=stub_hass,
        entry_id=entry.entry_id,
        dev_id="dev-empty-stats",
        inventory=inventory,
        client=_SampleClient(),
        config_entry=entry,
    )

    monkeypatch.setattr(energy, "datetime", _FixedDatetime, raising=False)

    class _FakeRegistry:
        def async_get_entity_id(self, *args: Any) -> str:
            return "sensor.test_energy"

        def async_get(self, *args: Any) -> Any:
            return SimpleNamespace(original_name="Test Energy")

    monkeypatch.setattr(
        energy.er, "async_get", lambda hass: _FakeRegistry(), raising=False
    )

    async def _empty_period(*args: Any, **kwargs: Any) -> dict:
        return {}

    async def _fake_clear(*args: Any, **kwargs: Any) -> str:
        return "delete"

    monkeypatch.setattr(energy, "_statistics_during_period", _empty_period)
    monkeypatch.setattr(energy, "_clear_statistics", _fake_clear)

    stored: list[Any] = []

    async def _capture_store(hass: Any, metadata: dict, stats: list) -> None:
        stored.append(stats)

    monkeypatch.setattr(energy, "_store_statistics", _capture_store, raising=False)

    await energy.async_import_energy_history(
        stub_hass,
        entry,
        rate_limit=_ImmediateRateLimiter(),
        max_days=1,
    )

    # With only one sample, previous_kwh is set but no delta computed
    # Stats should be empty and no statistics written
    assert stored == []


@pytest.mark.asyncio
async def test_import_energy_history_no_runtime(
    monkeypatch: pytest.MonkeyPatch,
    stub_hass,
) -> None:
    """Importer should handle missing runtime gracefully (lines 259-261)."""

    entry = _StubConfigEntry("entry-no-runtime")
    stub_hass.config_entries.add(entry)

    await energy.async_import_energy_history(
        stub_hass,
        entry,
        rate_limit=_ImmediateRateLimiter(),
    )
