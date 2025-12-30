"""Tests for resetting the Ducaheat websocket payload window."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from custom_components.termoweb.backend import ducaheat_ws


class DummyHass:
    """Provide a minimal Home Assistant stub with a data bucket."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.data: dict[str, Any] = {}


class StubTracker:
    """Record payload window updates for verification."""

    def __init__(self) -> None:
        self.calls: list[float] = []
        self.payload_stale_after: float | None = None
        self.payload_stale: bool = False
        self.status = "stub"

    def set_payload_window(self, window: float | None) -> bool:
        """Capture payload window updates and report a state change."""

        if window is None:
            raise AssertionError("window must not be None")
        self.calls.append(window)
        self.payload_stale_after = window
        return True


def _make_client(
    monkeypatch: pytest.MonkeyPatch, tracker: StubTracker
) -> ducaheat_ws.DucaheatWSClient:
    """Create a websocket client with a patched health tracker."""

    loop = asyncio.new_event_loop()
    hass = DummyHass(loop)

    monkeypatch.setattr(
        ducaheat_ws.DucaheatWSClient,
        "_ws_health_tracker",
        lambda self: tracker,
    )
    monkeypatch.setattr(
        ducaheat_ws.DucaheatWSClient,
        "_notify_ws_status",
        lambda *_, **__: None,
    )

    client = ducaheat_ws.DucaheatWSClient(
        hass,
        entry_id="entry",
        dev_id="device",
        api_client=SimpleNamespace(_session=object()),
        coordinator=SimpleNamespace(),
        session=SimpleNamespace(),
    )

    return client


def test_reset_payload_window_restores_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_reset_payload_window should restore defaults and update state buckets."""

    tracker = StubTracker()
    client = _make_client(monkeypatch, tracker)

    try:
        tracker.calls.clear()
        tracker.payload_stale_after = 480.0
        tracker.payload_stale = True

        client._payload_stale_after = 480.0
        client._payload_window_hint = 60.0
        client._payload_window_source = "hint"

        state = client._ws_state_bucket()
        state.update(
            {
                "payload_stale_after": 480.0,
                "payload_window_hint": 60.0,
                "payload_window_source": "hint",
                "payload_stale": False,
            }
        )

        client._notify_ws_status = MagicMock()

        client._reset_payload_window(source="manual")

        default_window = client._default_payload_window

        assert client._payload_stale_after == default_window
        assert client._payload_window_hint is None
        assert client._payload_window_source == "manual"

        assert tracker.calls == [default_window]
        assert tracker.payload_stale_after == default_window

        state_after = client._ws_state_bucket()
        assert state_after["payload_stale_after"] == default_window
        assert state_after["payload_window_hint"] is None
        assert state_after["payload_window_source"] == "manual"
        assert state_after["payload_stale"] is tracker.payload_stale
    finally:
        client.hass.loop.close()
