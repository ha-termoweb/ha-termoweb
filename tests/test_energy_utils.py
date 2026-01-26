"""Tests for energy utilities."""

from datetime import UTC, datetime
import logging

from custom_components.termoweb import energy
from custom_components.termoweb.energy import _iso_date
import pytest


def test_iso_date_for_recent_timestamp() -> None:
    """_iso_date should convert timestamp to ISO date string."""

    assert _iso_date(1_700_000_000) == "2023-11-14"


def test_iso_date_for_unix_epoch() -> None:
    """_iso_date should convert zero to the Unix epoch date."""

    assert _iso_date(0) == "1970-01-01"


@pytest.mark.asyncio
async def test_clear_statistics_prefers_delete_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_clear_statistics should call the recorder delete helper when available."""

    called: dict[str, object] = {}

    async def async_delete_statistics(
        _hass: object,
        statistic_ids: list[str],
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> None:
        """Capture delete calls for verification."""

        called["statistic_ids"] = statistic_ids
        called["start_time"] = start_time
        called["end_time"] = end_time

    monkeypatch.setattr(
        energy.recorder_stats,
        "async_delete_statistics",
        async_delete_statistics,
        raising=False,
    )

    start_time = datetime(2024, 2, 1, tzinfo=UTC)
    end_time = datetime(2024, 2, 2, tzinfo=UTC)

    result = await energy._clear_statistics(
        object(),
        "sensor.energy",
        start_time=start_time,
        end_time=end_time,
    )

    assert result == "delete"
    assert called["statistic_ids"] == ["sensor.energy"]
    assert called["start_time"] == start_time
    assert called["end_time"] == end_time


@pytest.mark.asyncio
async def test_clear_statistics_uses_clear_helper_without_keywords(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_clear_statistics should retry without keywords if needed."""

    calls: list[list[str]] = []

    async def async_clear_statistics(
        _hass: object,
        statistic_ids: list[str],
    ) -> None:
        """Capture clear calls for verification."""

        calls.append(statistic_ids)

    monkeypatch.delattr(energy.recorder_stats, "async_delete_statistics", raising=False)
    monkeypatch.setattr(
        energy.recorder_stats,
        "async_clear_statistics",
        async_clear_statistics,
        raising=False,
    )

    result = await energy._clear_statistics(
        object(),
        "sensor.energy",
        start_time=datetime(2024, 2, 1, tzinfo=UTC),
        end_time=datetime(2024, 2, 2, tzinfo=UTC),
    )

    assert result == "delete"
    assert calls == [["sensor.energy"]]


@pytest.mark.asyncio
async def test_clear_statistics_logs_when_helpers_missing(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_clear_statistics should log when recorder helpers are missing."""

    monkeypatch.delattr(energy.recorder_stats, "async_delete_statistics", raising=False)
    monkeypatch.delattr(energy.recorder_stats, "async_clear_statistics", raising=False)

    with caplog.at_level(logging.ERROR):
        result = await energy._clear_statistics(
            object(),
            "sensor.energy",
        )

    assert result == "delete"
    assert "recorder statistics delete helper unavailable" in caplog.text
