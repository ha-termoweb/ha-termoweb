"""Tests for caching raw websocket node payloads."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from tests.test_termoweb_ws_protocol import _make_client


@pytest.mark.usefixtures("monkeypatch")
def test_apply_nodes_payload_caches_deep_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caching should store a deep copy that is isolated from mutations."""

    client, _sio, _dispatcher = _make_client(monkeypatch)
    client._dispatch_nodes = MagicMock(return_value={})
    client._forward_sample_updates = MagicMock()
    client._mark_event = MagicMock()
    client._nodes_raw = {}

    payload: dict[str, Any] = {
        "nodes": {
            "htr": {
                "status": {
                    "1": {"power": 5, "temp": 21},
                    "2": {"power": 3},
                },
                "settings": {"1": {"mode": "eco"}},
            }
        }
    }

    original_status = payload["nodes"]["htr"]["status"]["1"]
    client._apply_nodes_payload(payload, merge=True, event="update")

    cached_status = client._nodes_raw["htr"]["status"]["1"]
    assert cached_status == {"power": 5, "temp": 21}
    assert cached_status is not original_status

    payload["nodes"]["htr"]["status"]["1"]["temp"] = 42
    payload["nodes"]["htr"]["status"]["2"]["power"] = 9
    payload["nodes"]["htr"]["settings"]["1"]["mode"] = "comfort"

    assert client._nodes_raw["htr"]["status"]["1"]["temp"] == 21
    assert client._nodes_raw["htr"]["status"]["2"]["power"] == 3
    assert client._nodes_raw["htr"]["settings"]["1"]["mode"] == "eco"

    # Subsequent updates should merge into the cached copy without sharing references.
    incremental = {
        "nodes": {
            "htr": {
                "status": {"1": {"temp": 20}},
                "settings": {"1": {"mode": "comfort"}},
            }
        }
    }
    client._apply_nodes_payload(incremental, merge=True, event="update")

    assert client._nodes_raw["htr"]["status"]["1"] == {"power": 5, "temp": 20}
    assert client._nodes_raw["htr"]["settings"]["1"] == {"mode": "comfort"}
    assert client._nodes_raw["htr"]["status"]["1"] is not incremental["nodes"]["htr"]["status"]["1"]
    assert (
        client._nodes_raw["htr"]["settings"]["1"]
        is not incremental["nodes"]["htr"]["settings"]["1"]
    )

    # Ensure the cached copy survives mutations of the incremental payload.
    incremental["nodes"]["htr"]["status"]["1"]["temp"] = -5
    incremental["nodes"]["htr"]["settings"]["1"]["mode"] = "away"
    assert client._nodes_raw["htr"]["status"]["1"]["temp"] == 20
    assert client._nodes_raw["htr"]["settings"]["1"]["mode"] == "comfort"
