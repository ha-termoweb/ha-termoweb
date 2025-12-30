"""Tests covering TermoWeb websocket event validation failures."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from custom_components.termoweb.backend.termoweb_ws import TermoWebWSClient


def _make_client() -> TermoWebWSClient:
    """Return a minimally configured TermoWeb websocket client."""

    client = object.__new__(TermoWebWSClient)
    client._dispatch_nodes = MagicMock(name="_dispatch_nodes")
    client._translate_path_update = MagicMock(name="_translate_path_update")
    client._coordinator = SimpleNamespace(data={})
    client.dev_id = "device-id"
    client.entry_id = "entry-id"
    client.hass = SimpleNamespace()
    client._stats = SimpleNamespace(last_event_ts=None)
    client._payload_idle_window = 240.0
    client._mark_event = MagicMock(name="_mark_event")
    client._mark_ws_payload = MagicMock(name="_mark_ws_payload")
    client._inventory = None
    return client


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(None, id="non-dict"),
        pytest.param({"name": "data"}, id="missing-args"),
        pytest.param({"name": "data", "args": []}, id="empty-args"),
        pytest.param({"name": "data", "args": [[]]}, id="empty-batch"),
        pytest.param(
            {"name": "data", "args": [[{"body": {"foo": "bar"}}]]},
            id="missing-path",
        ),
    ],
)
def test_handle_event_invalid_payloads(payload: Any) -> None:
    """Ensure invalid websocket payloads never reach node dispatch."""

    client = _make_client()

    client._handle_event(payload)  # type: ignore[arg-type]

    client._dispatch_nodes.assert_not_called()
    client._translate_path_update.assert_not_called()
