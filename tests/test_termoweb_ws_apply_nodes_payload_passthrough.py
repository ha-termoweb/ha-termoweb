"""Tests for passthrough behaviour in `_apply_nodes_payload`."""

from __future__ import annotations

import logging
from typing import Any, Mapping
from unittest.mock import MagicMock

import pytest

from tests.test_termoweb_ws_protocol import _make_client


def test_apply_nodes_payload_passthrough(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Translated path updates should dispatch nodes without debug warnings."""

    client, _sio, _dispatcher = _make_client(monkeypatch)

    translated_nodes: dict[str, Any] = {
        "htr": {"status": {"1": {"temp": 21}}},
    }
    translator = MagicMock(return_value=translated_nodes)
    client._translate_path_update = translator  # type: ignore[assignment]

    dispatched: list[Mapping[str, Any]] = []

    def record_dispatch(payload: Mapping[str, Any]) -> dict[str, list[str]]:
        dispatched.append(payload)
        return {}

    client._dispatch_nodes = record_dispatch  # type: ignore[assignment]

    with caplog.at_level(logging.DEBUG):
        client._apply_nodes_payload({}, merge=True, event="update")

    translator.assert_called_once_with({})
    assert dispatched == [translated_nodes]
    assert "WS: update without nodes" not in caplog.messages
