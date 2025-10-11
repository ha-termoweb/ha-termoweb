"""Websocket health tracking primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any


@dataclass
class WsHealthTracker:
    """Track websocket health, payload freshness, and heartbeat timestamps."""

    dev_id: str
    status: str = "stopped"
    healthy_since: float | None = None
    last_status_at: float | None = None
    last_heartbeat_at: float | None = None
    last_payload_at: float | None = None
    payload_stale_after: float | None = None
    _payload_stale: bool = field(default=True, init=False, repr=False)

    def update_status(
        self,
        status: str,
        *,
        healthy_since: float | None = None,
        timestamp: float | None = None,
        reset_health: bool = False,
    ) -> tuple[bool, bool]:
        """Update the tracked status and return (status_changed, health_changed)."""

        now = timestamp or time.time()
        status_changed = False
        health_changed = False

        if status != self.status:
            self.status = status
            self.last_status_at = now
            status_changed = True

        if status == "healthy":
            candidate = healthy_since if healthy_since is not None else now
            if self.healthy_since != candidate:
                self.healthy_since = candidate
                health_changed = True
        elif reset_health:
            if self.healthy_since is not None:
                self.healthy_since = None
                health_changed = True

        return status_changed, health_changed

    def set_payload_window(self, stale_after: float | None) -> bool:
        """Set the payload freshness window and return True if staleness changed."""

        if stale_after is None:
            return False
        try:
            candidate = float(stale_after)
        except (TypeError, ValueError):
            return False
        if candidate <= 0:
            return False
        if self.payload_stale_after == candidate:
            return False
        self.payload_stale_after = candidate
        return self.refresh_payload_state()

    def mark_payload(
        self,
        *,
        timestamp: float | None = None,
        stale_after: float | None = None,
    ) -> bool:
        """Record an application payload and return True if staleness changed."""

        now = timestamp or time.time()
        staleness_changed = False
        if stale_after is not None:
            staleness_changed = self.set_payload_window(stale_after)
        self.last_payload_at = now
        if self.last_heartbeat_at is None or now >= self.last_heartbeat_at:
            self.last_heartbeat_at = now
        changed = self.refresh_payload_state(now=now)
        return changed or staleness_changed

    def mark_heartbeat(self, *, timestamp: float | None = None) -> bool:
        """Record a transport heartbeat and return True if staleness changed."""

        now = timestamp or time.time()
        if self.last_heartbeat_at is None or now >= self.last_heartbeat_at:
            self.last_heartbeat_at = now
        return self.refresh_payload_state(now=now)

    def refresh_payload_state(self, *, now: float | None = None) -> bool:
        """Recalculate payload staleness and return True if it changed."""

        is_stale = self.is_payload_stale(now=now)
        if is_stale == self._payload_stale:
            return False
        self._payload_stale = is_stale
        return True

    def is_payload_stale(self, *, now: float | None = None) -> bool:
        """Return True when the last payload timestamp exceeds ``payload_stale_after``."""

        if self.last_payload_at is None:
            return True
        threshold = self.payload_stale_after
        if threshold is None or threshold <= 0:
            return False
        current = now or time.time()
        if current <= self.last_payload_at:
            return False
        return (current - self.last_payload_at) >= threshold

    @property
    def payload_stale(self) -> bool:
        """Return the cached payload staleness state."""

        return self._payload_stale

    def healthy_minutes(self, *, now: float | None = None) -> int:
        """Return the number of minutes spent in the healthy state."""

        if self.healthy_since is None:
            return 0
        current = now or time.time()
        if current <= self.healthy_since:
            return 0
        return int((current - self.healthy_since) / 60)

    def stale_deadline(self) -> float | None:
        """Return the monotonic timestamp when the payload becomes stale."""

        if self.last_payload_at is None:
            return None
        threshold = self.payload_stale_after
        if threshold is None or threshold <= 0:
            return None
        return self.last_payload_at + threshold

    def snapshot(self, *, now: float | None = None) -> dict[str, Any]:
        """Return a serializable snapshot of the tracker state."""

        current = now or time.time()
        return {
            "status": self.status,
            "healthy_since": self.healthy_since,
            "healthy_minutes": self.healthy_minutes(now=current),
            "last_status_at": self.last_status_at,
            "last_heartbeat_at": self.last_heartbeat_at,
            "last_payload_at": self.last_payload_at,
            "payload_stale": self.payload_stale,
            "payload_stale_after": self.payload_stale_after,
        }


__all__ = ["WsHealthTracker"]
