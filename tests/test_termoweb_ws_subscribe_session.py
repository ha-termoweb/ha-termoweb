"""Tests for the legacy session metadata subscription helper."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from custom_components.termoweb.backend import termoweb_ws as module
from tests.test_termoweb_ws_protocol import _make_client


@pytest.mark.asyncio
async def test_subscribe_session_metadata_issues_single_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_subscribe_session_metadata should emit a single subscribe frame."""

    client, _, _ = _make_client(monkeypatch)
    send = AsyncMock()
    client._send_text = send

    await module.TermoWebWSClient._subscribe_session_metadata(client)

    expected = "5::{namespace}:{payload}".format(
        namespace=client._namespace,
        payload='{"name":"subscribe","args":["/mgr/session"]}',
    )
    send.assert_awaited_once_with(expected)
    assert send.await_count == 1
