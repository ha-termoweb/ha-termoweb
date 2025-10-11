"""Tests for TermoWeb websocket snapshot requests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from custom_components.termoweb.backend.termoweb_ws import TermoWebWSClient
from homeassistant.core import HomeAssistant


@pytest.mark.asyncio
async def test_send_snapshot_request_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the snapshot request sends the expected Socket.IO payload."""

    hass = HomeAssistant()
    hass.loop = SimpleNamespace(
        call_soon_threadsafe=lambda callback, *args: callback(*args),
        is_running=lambda: False,
    )
    hass.data = {}

    monkeypatch.setattr(TermoWebWSClient, "_install_write_hook", lambda self: None)

    api_client = SimpleNamespace(_session=SimpleNamespace(closed=False))
    coordinator = SimpleNamespace(update_nodes=AsyncMock(), data={})

    client = TermoWebWSClient(
        hass,
        entry_id="entry",
        dev_id="device",
        api_client=api_client,
        coordinator=coordinator,
        session=SimpleNamespace(closed=False),
    )

    send_text = AsyncMock()
    monkeypatch.setattr(client, "_send_text", send_text)

    await client._send_snapshot_request()

    send_text.assert_awaited_once_with(
        '5::/api/v2/socket_io:{"name":"dev_data","args":[]}'
    )
