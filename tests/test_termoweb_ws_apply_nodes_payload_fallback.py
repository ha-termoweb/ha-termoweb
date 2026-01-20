"""Tests for websocket node payload fallback behaviour."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from tests.test_termoweb_ws_protocol import _make_client


def test_apply_nodes_payload_uses_raw_payload_on_normaliser_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normaliser failures should not prevent dispatching the original payload."""

    client, _sio, _dispatcher = _make_client(monkeypatch)
    payload: dict[str, Any] = {
        "nodes": {"htr": {"samples": {"1": {"power": 10}}}},
    }

    def raise_runtime_error(_nodes: Any) -> None:
        raise RuntimeError("normaliser failure")

    client._client.normalise_ws_nodes = raise_runtime_error  # type: ignore[attr-defined]

    client._forward_sample_updates = MagicMock()

    client._apply_nodes_payload(payload, merge=True, event="update")

    client._forward_sample_updates.assert_called_once_with(
        {"htr": {"samples": {"1": {"power": 10}}, "lease_seconds": None}}
    )
