"""Tests for websocket status payload tracking helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from custom_components.termoweb.backend.ws_client import _WSStatusMixin
from custom_components.termoweb.const import DOMAIN


class DummyHass:
    """Provide a minimal Home Assistant stub with a data bucket."""

    def __init__(self) -> None:
        self.data: dict[str, Any] = {}


@dataclass
class StubTracker:
    """Stub websocket tracker exposing payload and heartbeat hooks."""

    mark_payload_result: bool = False
    mark_heartbeat_result: bool = False
    refresh_result: bool = False
    next_payload_stale: bool = False
    status: str = "stub"
    payload_stale: bool = False
    last_payload_at: float | None = None
    last_heartbeat_at: float | None = None
    payload_stale_after: float | None = None
    mark_payload_calls: list[dict[str, Any]] = field(default_factory=list)
    mark_heartbeat_calls: list[dict[str, Any]] = field(default_factory=list)
    refresh_calls: list[dict[str, Any]] = field(default_factory=list)

    def mark_payload(
        self,
        *,
        timestamp: float | None = None,
        stale_after: float | None = None,
    ) -> bool:
        """Record payload arguments for verification."""

        self.mark_payload_calls.append(
            {"timestamp": timestamp, "stale_after": stale_after}
        )
        self.last_payload_at = timestamp
        if timestamp is not None:
            self.last_heartbeat_at = timestamp
        if stale_after is not None:
            self.payload_stale_after = stale_after
        self.payload_stale = self.next_payload_stale
        return self.mark_payload_result

    def mark_heartbeat(self, *, timestamp: float | None = None) -> bool:
        """Track heartbeat invocations for assertions."""

        self.mark_heartbeat_calls.append({"timestamp": timestamp})
        self.last_heartbeat_at = timestamp
        self.payload_stale = self.next_payload_stale
        return self.mark_heartbeat_result

    def refresh_payload_state(self, *, now: float | None = None) -> bool:
        """Capture refresh requests and update staleness."""

        self.refresh_calls.append({"now": now})
        self.payload_stale = self.next_payload_stale
        return self.refresh_result


class TrackingStatusClient(_WSStatusMixin):
    """Expose websocket status helpers with injectable tracker."""

    def __init__(self, hass: DummyHass, tracker: StubTracker) -> None:
        self.hass = hass
        self.entry_id = "entry"
        self.dev_id = "device"
        self._tracker = tracker
        self.notifications: list[dict[str, Any]] = []

    def _ws_health_tracker(self) -> StubTracker:
        """Return the stub tracker supplied by the test."""

        return self._tracker

    def _notify_ws_status(
        self,
        tracker: StubTracker,
        *,
        reason: str,
        health_changed: bool = False,
        payload_changed: bool = False,
    ) -> None:
        """Store dispatcher payloads for inspection."""

        payload: dict[str, Any] = {
            "dev_id": self.dev_id,
            "status": tracker.status,
            "reason": reason,
            "payload_stale": tracker.payload_stale,
        }
        if health_changed:
            payload["health_changed"] = True
        if payload_changed:
            payload["payload_changed"] = True
        self.notifications.append(payload)


class LegacyStatusClient(_WSStatusMixin):
    """Populate legacy websocket fields for tracker bootstrap tests."""

    def __init__(self, hass: DummyHass) -> None:
        self.hass = hass
        self.entry_id = "entry"
        self.dev_id = "device"
        self._status = "legacy"
        self._healthy_since = 111.1
        self._last_payload_at = 222.2
        self._last_heartbeat_at = 333.3


def test_ws_health_tracker_bootstraps_legacy_state() -> None:
    """Ensure tracker initialization consumes legacy mixin attributes."""

    hass = DummyHass()
    client = LegacyStatusClient(hass)

    assert hass.data == {}

    tracker = client._ws_health_tracker()

    assert tracker.status == "legacy"
    assert tracker.healthy_since == 111.1
    assert tracker.last_payload_at == 222.2
    assert tracker.last_heartbeat_at == 333.3

    assert DOMAIN in hass.data
    assert client.entry_id in hass.data[DOMAIN]
    entry_bucket = hass.data[DOMAIN][client.entry_id]
    assert "ws_trackers" in entry_bucket
    assert entry_bucket["ws_trackers"][client.dev_id] is tracker

    assert client._ws_health_tracker() is tracker


def test_mark_ws_payload_dispatches_staleness_changes() -> None:
    """Ensure payload timestamps call the tracker and expose staleness flags."""

    hass = DummyHass()
    tracker = StubTracker(
        mark_payload_result=True, next_payload_stale=True, status="ok"
    )
    client = TrackingStatusClient(hass, tracker)

    client._mark_ws_payload(timestamp=111.1, stale_after=15.0, reason="payload")

    assert tracker.mark_payload_calls == [{"timestamp": 111.1, "stale_after": 15.0}]
    state = client._ws_state_bucket()
    assert state["last_payload_at"] == 111.1
    assert state["last_heartbeat_at"] == 111.1
    assert state["payload_stale"] is True
    assert client.notifications == [
        {
            "dev_id": "device",
            "status": "ok",
            "reason": "payload",
            "payload_stale": True,
            "payload_changed": True,
        }
    ]


def test_mark_ws_heartbeat_includes_payload_staleness() -> None:
    """Verify heartbeat tracking forwards timestamps and staleness metadata."""

    hass = DummyHass()
    tracker = StubTracker(mark_heartbeat_result=True, next_payload_stale=False)
    client = TrackingStatusClient(hass, tracker)

    client._mark_ws_heartbeat(timestamp=222.2, reason="beat")

    assert tracker.mark_heartbeat_calls == [{"timestamp": 222.2}]
    state = client._ws_state_bucket()
    assert state["last_heartbeat_at"] == 222.2
    assert state["payload_stale"] is False
    assert client.notifications == [
        {
            "dev_id": "device",
            "status": "stub",
            "reason": "beat",
            "payload_stale": False,
            "payload_changed": True,
        }
    ]


def test_refresh_ws_payload_state_surfaces_tracker_changes() -> None:
    """Confirm refresh requests propagate staleness transitions."""

    hass = DummyHass()
    tracker = StubTracker(refresh_result=True, next_payload_stale=True)
    client = TrackingStatusClient(hass, tracker)

    client._refresh_ws_payload_state(now=333.3, reason="refresh")

    assert tracker.refresh_calls == [{"now": 333.3}]
    state = client._ws_state_bucket()
    assert state["payload_stale"] is True
    assert client.notifications == [
        {
            "dev_id": "device",
            "status": "stub",
            "reason": "refresh",
            "payload_stale": True,
            "payload_changed": True,
        }
    ]
