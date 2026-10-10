"""Tests for the websocket payload-freshness and health tracker."""

from __future__ import annotations

import time

import pytest

from custom_components.termoweb.backend.ws_health import WsHealthTracker


def test_mark_payload_and_heartbeat_updates() -> None:
    """Test payload updates refresh heartbeat timestamps and staleness."""

    tracker = WsHealthTracker("dev")
    tracker.set_payload_window(120)
    changed = tracker.mark_payload(timestamp=1_000.0, stale_after=120)
    assert changed is True
    assert tracker.last_payload_at == 1_000.0
    assert tracker.last_heartbeat_at == 1_000.0
    assert tracker.payload_stale is False

    changed = tracker.mark_heartbeat(timestamp=1_050.0)
    assert changed is False
    assert tracker.last_heartbeat_at == 1_050.0
    assert tracker.last_payload_at == 1_000.0
    assert tracker.payload_stale is False


def test_staleness_detection_and_refresh() -> None:
    """Test payload staleness detection transitions across the threshold."""

    tracker = WsHealthTracker("dev")
    tracker.mark_payload(timestamp=1_000.0, stale_after=30)
    assert tracker.payload_stale is False
    assert tracker.is_payload_stale(now=1_029.0) is False
    assert tracker.is_payload_stale(now=1_030.0) is True

    changed = tracker.refresh_payload_state(now=1_030.0)
    assert changed is True
    assert tracker.payload_stale is True

    changed = tracker.mark_payload(timestamp=1_040.0)
    assert changed is True
    assert tracker.payload_stale is False
    assert tracker.stale_deadline() == pytest.approx(1_070.0)


def test_stale_deadline_requires_positive_threshold() -> None:
    """Stale deadline should be None without a positive payload window."""

    tracker = WsHealthTracker("dev")
    tracker.mark_payload(timestamp=1_500.0)
    assert tracker.last_payload_at == 1_500.0
    # ``payload_stale_after`` is ``None`` until explicitly configured.
    assert tracker.payload_stale_after is None
    assert tracker.stale_deadline() is None


def test_update_status_resets_health_state() -> None:
    """Test status transitions update healthy timestamps and reset state."""

    tracker = WsHealthTracker("dev")
    status_changed, health_changed = tracker.update_status(
        "healthy", healthy_since=1_000.0, timestamp=1_000.0
    )
    assert status_changed is True
    assert health_changed is True
    assert tracker.healthy_since == 1_000.0
    assert tracker.healthy_minutes(now=1_120.0) == 2

    snapshot = tracker.snapshot(now=1_120.0)
    assert snapshot["status"] == "healthy"
    assert snapshot["healthy_minutes"] == 2

    status_changed, health_changed = tracker.update_status(
        "degraded", timestamp=1_130.0, reset_health=True
    )
    assert status_changed is True
    assert health_changed is True
    assert tracker.healthy_since is None
    assert tracker.healthy_minutes(now=1_200.0) == 0

    status_changed, health_changed = tracker.update_status(
        "degraded", timestamp=1_150.0
    )
    assert status_changed is False
    assert health_changed is False


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

    # Confirm reapplying the same positive value leaves the tracker unchanged.
    assert tracker.set_payload_window(15.0) is False
    assert tracker.payload_stale_after == 15.0

    assert tracker.set_payload_window("still-bad") is False
    assert tracker.payload_stale_after == 15.0

    assert tracker.mark_payload(timestamp=base + 1.0, stale_after=-3) is False
    assert tracker.payload_stale_after == 15.0


def test_ws_health_tracker_future_healthy_since() -> None:
    """Healthy minutes should not go negative when the transition is in the future."""

    tracker = WsHealthTracker("dev01")
    base = time.time()

    tracker.update_status("healthy", healthy_since=base + 60.0, timestamp=base)

    assert tracker.healthy_minutes(now=base) == 0
