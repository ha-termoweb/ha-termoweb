"""Unit tests for websocket status notification payloads."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from custom_components.termoweb.backend.ws_client import _WSStatusMixin
from custom_components.termoweb.backend.ws_health import WsHealthTracker
from custom_components.termoweb.const import signal_ws_status


class StubStatusClient(_WSStatusMixin):
    """Provide a minimal client exposing the websocket status helper."""

    def __init__(self) -> None:
        self.hass = SimpleNamespace()
        self.entry_id = "entry"
        self.dev_id = "device"
        self.dispatch_calls: list[tuple[Any, str, dict[str, Any]]] = []
        self._dispatcher_mock = self._record_dispatch

    def _record_dispatch(self, hass: Any, signal: str, payload: dict[str, Any]) -> None:
        """Capture dispatcher calls for later assertions."""

        self.dispatch_calls.append((hass, signal, payload))


@pytest.mark.parametrize(
    ("health_changed", "payload_changed", "expected_flags"),
    [
        (False, False, set()),
        (True, False, {"health_changed"}),
        (False, True, {"payload_changed"}),
        (True, True, {"health_changed", "payload_changed"}),
    ],
)
def test_notify_ws_status_includes_expected_payload_keys(
    health_changed: bool, payload_changed: bool, expected_flags: set[str]
) -> None:
    """Verify the dispatcher payload includes required metadata and optional flags."""

    client = StubStatusClient()
    tracker = WsHealthTracker(client.dev_id)

    client._notify_ws_status(
        tracker,
        reason="unit-test",
        health_changed=health_changed,
        payload_changed=payload_changed,
    )

    assert len(client.dispatch_calls) == 1
    hass, signal, payload = client.dispatch_calls[0]
    assert hass is client.hass
    assert signal == signal_ws_status(client.entry_id)

    base_keys = {"dev_id", "status", "reason", "payload_stale"}
    assert set(payload) == base_keys | expected_flags

    assert payload["dev_id"] == client.dev_id
    assert payload["status"] == tracker.status
    assert payload["reason"] == "unit-test"
    assert payload["payload_stale"] is tracker.payload_stale

    for flag in expected_flags:
        assert payload[flag] is True

    for flag in {"health_changed", "payload_changed"} - expected_flags:
        assert flag not in payload
