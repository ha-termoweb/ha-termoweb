"""Tests for enforcing monotonic energy statistics sums."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from custom_components.termoweb import energy


@pytest.mark.asyncio
async def test_enforce_monotonic_sum_rewrites_decreasing_hour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clamp a decreasing external hour to the prior sum."""

    hass = object()
    import_start = datetime(2024, 1, 1, 12, tzinfo=UTC)
    import_end = import_start + timedelta(hours=1)

    async def _fake_collect(
        hass_arg: Any,
        statistic_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> list[dict[str, Any]]:
        assert hass_arg is hass
        assert statistic_id == "sensor.sample_energy"
        assert start_time == import_start - timedelta(hours=1)
        assert end_time == import_end + timedelta(hours=6)
        return [
            {"start": import_start, "sum": 10.0},
            {"start": import_end, "sum": 9.5},
        ]

    rewrites: list[dict[str, Any]] = []

    async def _fake_store(
        hass_arg: Any, metadata: dict[str, Any], rows: list[dict[str, Any]]
    ) -> None:
        assert hass_arg is hass
        assert metadata["statistic_id"] == "sensor.sample_energy"
        rewrites.extend(rows)

    monkeypatch.setattr(energy, "_collect_statistics", _fake_collect, raising=False)
    monkeypatch.setattr(energy.er, "async_get", lambda hass: None, raising=False)
    monkeypatch.setattr(energy, "_store_statistics", _fake_store, raising=False)

    await energy._enforce_monotonic_sum(
        hass,
        "sensor.sample_energy",
        import_start,
        import_end,
    )

    assert rewrites == [{"start": import_end, "sum": 10.0}]


@pytest.mark.asyncio
async def test_enforce_monotonic_sum_clamps_import_to_live_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adjust the first live hour when it dips below the imported sum."""

    hass = object()
    import_start = datetime(2024, 2, 1, 0, tzinfo=UTC)
    import_end = import_start + timedelta(hours=2)
    seam_hour = import_end + timedelta(hours=1)

    rows = [
        {"start": import_start - timedelta(hours=1), "sum": 4.5},
        {"start": import_end, "sum": 6.2},
        {"start": seam_hour, "sum": 6.0},
        {"start": seam_hour + timedelta(hours=1), "sum": 7.1},
    ]

    async def _fake_collect(
        hass_arg: Any,
        statistic_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> list[dict[str, Any]]:
        assert hass_arg is hass
        assert statistic_id == "sensor.seam_energy"
        assert start_time == import_start - timedelta(hours=1)
        assert end_time == import_end + timedelta(hours=6)
        return rows

    rewrites: list[dict[str, Any]] = []

    async def _fake_store(
        hass_arg: Any, metadata: dict[str, Any], rows_arg: list[dict[str, Any]]
    ) -> None:
        assert hass_arg is hass
        assert metadata["statistic_id"] == "sensor.seam_energy"
        rewrites.extend(rows_arg)

    monkeypatch.setattr(energy, "_collect_statistics", _fake_collect, raising=False)
    monkeypatch.setattr(energy.er, "async_get", lambda hass: None, raising=False)
    monkeypatch.setattr(energy, "_store_statistics", _fake_store, raising=False)

    await energy._enforce_monotonic_sum(
        hass,
        "sensor.seam_energy",
        import_start,
        import_end,
    )

    assert rewrites == [{"start": seam_hour, "sum": 6.2}]
