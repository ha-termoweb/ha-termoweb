"""Behavioural tests for :class:`WsHealthTracker`."""

from __future__ import annotations

import time

import pytest

from custom_components.termoweb.backend.ws_health import WsHealthTracker


def test_ws_health_tracker_payload_flow() -> None:
    """Exercise the happy-path lifecycle for payload freshness tracking."""

    tracker = WsHealthTracker("dev01")
    base = 1_000.0

    assert tracker.payload_stale is True
    assert tracker.set_payload_window(30.0) is False
    assert tracker.payload_stale_after == 30.0

    changed = tracker.mark_payload(timestamp=base)
    assert changed is True
    assert tracker.last_payload_at == base
    assert tracker.last_heartbeat_at == base
    assert tracker.payload_stale is False

    tracker.update_status("healthy", healthy_since=base - 120, timestamp=base - 60)
    assert tracker.healthy_minutes(now=base + 10.0) == 2

    assert tracker.mark_heartbeat(timestamp=base + 10.0) is False
    assert tracker.last_heartbeat_at == base + 10.0
    assert tracker.payload_stale is False

    assert tracker.refresh_payload_state(now=base + 40.0) is True
    assert tracker.payload_stale is True

    assert tracker.stale_deadline() == pytest.approx(base + 30.0)

    snapshot = tracker.snapshot(now=base + 40.0)
    assert snapshot == {
        "status": "healthy",
        "healthy_since": base - 120,
        "healthy_minutes": 2,
        "last_status_at": base - 60,
        "last_heartbeat_at": base + 10.0,
        "last_payload_at": base,
        "payload_stale": True,
        "payload_stale_after": 30.0,
    }


def test_ws_health_tracker_rejects_invalid_stale_after() -> None:
    """Ensure invalid staleness windows are ignored across setter paths."""

    tracker = WsHealthTracker("dev01")

    assert tracker.set_payload_window(None) is False
    assert tracker.payload_stale_after is None

    for invalid in (-5, 0, "bad-input"):
        assert tracker.set_payload_window(invalid) is False
        assert tracker.payload_stale_after is None

    base = time.time()
    assert tracker.mark_payload(timestamp=base, stale_after="noop") is True
    assert tracker.payload_stale_after is None

    assert tracker.set_payload_window(15.0) is False
    assert tracker.payload_stale_after == 15.0

    assert tracker.set_payload_window("still-bad") is False
    assert tracker.payload_stale_after == 15.0

    assert tracker.mark_payload(timestamp=base + 1.0, stale_after=-3) is False
    assert tracker.payload_stale_after == 15.0
