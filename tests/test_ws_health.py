"""Unit tests for the websocket health tracker helper."""

from __future__ import annotations

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
