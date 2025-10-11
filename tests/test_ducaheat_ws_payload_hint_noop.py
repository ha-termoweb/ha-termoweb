"""Tests for no-op cadence hints in the Ducaheat websocket client."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping
from unittest.mock import MagicMock

from homeassistant.core import HomeAssistant

from custom_components.termoweb.backend.ducaheat_ws import DucaheatWSClient, DOMAIN


class DummyREST:
    """Provide the minimal REST client contract for the websocket client."""

    def __init__(self) -> None:
        """Initialise the dummy REST client."""

        self._session = SimpleNamespace()

    async def authed_headers(self) -> Mapping[str, str]:
        """Return headers containing an access token."""

        return {"Authorization": "Bearer token"}


class DummyCoordinator:
    """Expose coordinator storage accessed by the websocket client."""

    def __init__(self) -> None:
        """Initialise the dummy coordinator."""

        self.data: dict[str, Any] = {"device": {}}


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


def test_payload_window_hint_rejects_invalid_candidates() -> None:
    """Ensure invalid cadence hints do not update payload windows."""

    client = _make_client()

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
