"""Tests for passthrough behaviour in `_apply_nodes_payload`."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from tests.test_termoweb_ws_protocol import _make_client


def test_apply_nodes_payload_passthrough(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Translated path updates should skip debug warnings."""

    client, _sio, _dispatcher = _make_client(monkeypatch)

    translated_nodes: dict[str, Any] = {
        "htr": {"status": {"1": {"temp": 21}}},
    }
    translator = MagicMock(return_value=translated_nodes)
    client._translate_path_update = translator  # type: ignore[assignment]

    client._forward_sample_updates = MagicMock()

    with caplog.at_level(logging.DEBUG):
        client._apply_nodes_payload({}, merge=True, event="update")

    translator.assert_called_once_with({})
    assert "WS: update without nodes" not in caplog.messages
    client._forward_sample_updates.assert_not_called()
