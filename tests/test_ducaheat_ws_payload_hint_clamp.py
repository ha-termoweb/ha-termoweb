"""Tests for payload window hint clamping in the Ducaheat websocket client."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, MutableMapping

import pytest

from homeassistant.core import HomeAssistant

from custom_components.termoweb.backend.ducaheat_ws import (
    DucaheatWSClient,
    _PAYLOAD_WINDOW_MAX,
    _PAYLOAD_WINDOW_MIN,
    DOMAIN,
)


class DummyREST:
    """Provide the minimal REST contract required by the websocket client."""

    def __init__(self) -> None:
        """Initialise the dummy REST client with a session placeholder."""

        self._session = SimpleNamespace()

    async def authed_headers(self) -> MutableMapping[str, str]:
        """Return headers resembling an authenticated REST request."""

        return {"Authorization": "Bearer token"}


class DummyCoordinator:
    """Expose coordinator attributes used by the websocket client."""

    def __init__(self) -> None:
        """Initialise the dummy coordinator with minimal state."""

        self.data: MutableMapping[str, Any] = {"device": {}}


class TrackerStub:
    """Record payload window updates from the websocket client."""

    def __init__(self) -> None:
        """Initialise the tracker stub."""

        self.calls: list[float] = []
        self.payload_stale_after: float | None = None
        self.payload_stale = False

    def set_payload_window(self, value: float | None) -> bool:
        """Record the requested payload window and update state."""

        if value is None:
            return False
        candidate = float(value)
        self.calls.append(candidate)
        self.payload_stale_after = candidate
        return False


def _make_client() -> DucaheatWSClient:
    """Instantiate a websocket client with stub dependencies."""

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


def test_payload_window_hint_clamps_extreme_values() -> None:
    """Extremely small and large hints should clamp to the configured bounds."""

    client = _make_client()
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
