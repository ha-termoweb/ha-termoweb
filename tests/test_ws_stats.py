"""Unit tests for the :class:`WSStats` dataclass."""

from __future__ import annotations

from custom_components.termoweb.backend.ws_client import WSStats


def test_ws_stats_defaults() -> None:
    """WSStats should expose explicit zero/empty defaults."""

    stats = WSStats()

    assert stats.frames_total == 0
    assert stats.events_total == 0
    assert stats.last_event_ts == 0.0
    assert stats.last_paths is None


def test_ws_stats_mutation_usage() -> None:
    """Mimic frame/event accounting updates to document expected usage."""

    stats = WSStats()

    stats.frames_total += 3
    stats.events_total += 2
    stats.last_event_ts = 123.456
    stats.last_paths = ["/devices", "/devices/node"]

    assert stats.frames_total == 3
    assert stats.events_total == 2
    assert stats.last_event_ts == 123.456
    assert stats.last_paths == ["/devices", "/devices/node"]
