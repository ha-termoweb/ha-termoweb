"""Ensure websocket tracker initialisation applies the maximum cadence hint."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest

from conftest import build_entry_runtime
from custom_components.termoweb.backend import ducaheat_ws


class DummyREST:
    """Provide a REST client stub with a reusable session."""

    def __init__(self) -> None:
        self._session = SimpleNamespace()

    async def authed_headers(self) -> dict[str, str]:  # pragma: no cover - helper
        """Return canned headers to satisfy the websocket client."""

        return {}


class DummyCoordinator:
    """Expose the minimal coordinator interface required by the client."""

    def handle_ws_samples(
        self, *_: Any, **__: Any
    ) -> None:  # pragma: no cover - helper
        """Ignore sample updates; tests patch the tracker directly."""

        return None


def _make_client(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ducaheat_ws.DucaheatWSClient, list[float]]:
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


def test_ws_health_tracker_applies_max_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tracker should clamp cadence hints to the permitted maximum window."""

    client, calls = _make_client(monkeypatch)

    tracker = client._ws_health_tracker()

    assert tracker.payload_stale_after == pytest.approx(client._payload_stale_after)

    hint = 120.0
    expected_margin = max(
        hint * ducaheat_ws._PAYLOAD_WINDOW_MARGIN_RATIO,
        ducaheat_ws._PAYLOAD_WINDOW_MARGIN_FLOOR,
    )
    expected_window = min(
        ducaheat_ws._PAYLOAD_WINDOW_MAX,
        max(ducaheat_ws._PAYLOAD_WINDOW_MIN, hint + expected_margin),
    )

    assert client._payload_window_hint == pytest.approx(hint)
    assert client._payload_stale_after == pytest.approx(expected_window)

    window_calls = [call for call in calls if math.isclose(call, expected_window)]
    assert window_calls == [expected_window]
