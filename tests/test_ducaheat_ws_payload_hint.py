"""Tests for payload window hints in the Ducaheat websocket client.

Covers no-op rejection of invalid candidates, clamping of extreme values,
and tracker initialisation with maximum cadence hints.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest

from conftest import DummyREST, DummyCoordinator, build_entry_runtime
from custom_components.termoweb.backend import ducaheat_ws
from custom_components.termoweb.backend.ducaheat_ws import (
    DucaheatWSClient,
    _PAYLOAD_WINDOW_MAX,
    _PAYLOAD_WINDOW_MIN,
    _PAYLOAD_WINDOW_MARGIN_FLOOR,
    _PAYLOAD_WINDOW_MARGIN_RATIO,
)
from custom_components.termoweb.const import DOMAIN

from homeassistant.core import HomeAssistant
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Tracker stub used by the clamp test
# ---------------------------------------------------------------------------


class TrackerStub:
    """Record payload window updates from the websocket client."""

    def __init__(self) -> None:
        self.calls: list[float] = []
        self.payload_stale_after: float | None = None
        self.payload_stale = False

    def set_payload_window(self, value: float | None) -> bool:
        if value is None:
            return False
        candidate = float(value)
        self.calls.append(candidate)
        self.payload_stale_after = candidate
        return False


# ---------------------------------------------------------------------------
# Client factories
# ---------------------------------------------------------------------------


def _make_simple_client() -> DucaheatWSClient:
    """Instantiate a websocket client with stub dependencies (no monkeypatch)."""

    hass = HomeAssistant()
    hass.data.setdefault(DOMAIN, {})["entry"] = {}
    client = DucaheatWSClient(
        hass,
        entry_id="entry",
        dev_id="device",
        api_client=DummyREST(),
        coordinator=DummyCoordinator(),
        session=SimpleNamespace(),  # type: ignore[arg-type]
    )
    return client


def _make_tracked_client(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[DucaheatWSClient, list[float]]:
    """Return a websocket client with deterministic tracker behaviour."""

    calls: list[float] = []

    original = ducaheat_ws.WsHealthTracker.set_payload_window

    def _spy(self: ducaheat_ws.WsHealthTracker, stale_after: float | None) -> bool:
        if isinstance(stale_after, (int, float)):
            candidate = float(stale_after)
            if not any(math.isclose(candidate, existing) for existing in calls):
                calls.append(candidate)
        else:
            calls.append(math.nan)
        return original(self, stale_after)

    monkeypatch.setattr(ducaheat_ws.WsHealthTracker, "set_payload_window", _spy)

    hass = ducaheat_ws.HomeAssistant()
    build_entry_runtime(
        hass=hass,
        entry_id="entry",
        dev_id="device",
        coordinator=DummyCoordinator(),
    )
    session = SimpleNamespace()
    client = ducaheat_ws.DucaheatWSClient(
        hass,
        entry_id="entry",
        dev_id="device",
        api_client=DummyREST(),
        coordinator=DummyCoordinator(),
        session=session,  # type: ignore[arg-type]
    )

    return client, calls


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_payload_window_hint_rejects_invalid_candidates() -> None:
    """Ensure invalid cadence hints do not update payload windows."""

    client = _make_simple_client()

    default_window = client._payload_stale_after
    default_hint = client._payload_window_hint

    tracker_stub = SimpleNamespace(set_payload_window=MagicMock())
    tracker_factory = MagicMock(return_value=tracker_stub)
    client._ws_health_tracker = tracker_factory  # type: ignore[assignment]

    client._apply_payload_window_hint(
        source="cadence",
        lease_seconds=None,
        candidates=[None, "NaN", -5],
    )

    assert client._payload_stale_after == default_window
    assert client._payload_window_hint == default_hint
    tracker_stub.set_payload_window.assert_not_called()
    tracker_factory.assert_not_called()


def test_payload_window_hint_clamps_extreme_values() -> None:
    """Extremely small and large hints should clamp to the configured bounds."""

    client = _make_simple_client()
    tracker = TrackerStub()
    client._ws_health_tracker = lambda: tracker  # type: ignore[assignment]

    client._apply_payload_window_hint(source="test", lease_seconds=1)

    assert client._payload_stale_after == pytest.approx(_PAYLOAD_WINDOW_MIN)
    assert client._payload_window_hint == pytest.approx(1.0)
    assert tracker.calls[0] == pytest.approx(_PAYLOAD_WINDOW_MIN)

    client._apply_payload_window_hint(source="test", lease_seconds=10_000)

    assert client._payload_stale_after == pytest.approx(_PAYLOAD_WINDOW_MAX)
    assert client._payload_window_hint == pytest.approx(10_000.0)
    assert tracker.calls[1] == pytest.approx(_PAYLOAD_WINDOW_MAX)
    assert tracker.calls == pytest.approx([_PAYLOAD_WINDOW_MIN, _PAYLOAD_WINDOW_MAX])


def test_ws_health_tracker_applies_max_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tracker should clamp cadence hints to the permitted maximum window."""

    client, calls = _make_tracked_client(monkeypatch)

    tracker = client._ws_health_tracker()

    assert tracker.payload_stale_after == pytest.approx(client._payload_stale_after)

    hint = 120.0
    expected_margin = max(
        hint * _PAYLOAD_WINDOW_MARGIN_RATIO,
        _PAYLOAD_WINDOW_MARGIN_FLOOR,
    )
    expected_window = min(
        _PAYLOAD_WINDOW_MAX,
        max(_PAYLOAD_WINDOW_MIN, hint + expected_margin),
    )

    assert client._payload_window_hint == pytest.approx(hint)
    assert client._payload_stale_after == pytest.approx(expected_window)

    window_calls = [call for call in calls if math.isclose(call, expected_window)]
    assert window_calls == [expected_window]
