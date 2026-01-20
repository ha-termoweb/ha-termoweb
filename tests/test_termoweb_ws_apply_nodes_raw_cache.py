"""Tests for websocket node payload forwarding without caching."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from tests.test_termoweb_ws_protocol import _make_client


@pytest.mark.usefixtures("monkeypatch")
def test_apply_nodes_payload_does_not_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runtime updates should be forwarded without duplicating node payloads."""

    client, _sio, _dispatcher = _make_client(monkeypatch)
    client._forward_sample_updates = MagicMock()
    client._mark_event = MagicMock()
    payload: dict[str, Any] = {
        "nodes": {
            "htr": {
                "status": {
                    "1": {"power": 5, "temp": 21},
                    "2": {"power": 3},
                },
                "samples": {"1": {"power": 12}},
            }
        }
    }

    assert not hasattr(client, "_nodes_raw")

    client._apply_nodes_payload(payload, merge=True, event="update")

    client._forward_sample_updates.assert_called_once()
    client._mark_event.assert_called_once()
    assert not hasattr(client, "_nodes_raw")
