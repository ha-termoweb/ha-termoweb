"""Tests for Ducaheat websocket handshake URL construction."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from urllib.parse import parse_qsl, urlsplit

import pytest

from custom_components.termoweb.backend.ducaheat_ws import DucaheatWSClient


@pytest.mark.asyncio
async def test_build_handshake_url_preserves_query_items() -> None:
    """Ducaheat handshake URLs should use the correct host/path and query."""

    hass = SimpleNamespace(loop=asyncio.get_running_loop())
    session = object()
    api_client = SimpleNamespace(_session=session)

    client = DucaheatWSClient(
        hass,
        entry_id="entry-123",
        dev_id="device-456",
        api_client=api_client,
        coordinator=object(),
        session=session,
    )

    params = {
        "token": "fixed-token",
        "dev_id": "device-456",
        "EIO": "3",
        "transport": "polling",
        "t": "PABCDEFG",
    }

    url = client._build_handshake_url(params)

    parsed = urlsplit(url)
    assert parsed.scheme == "https"
    assert parsed.netloc == "api-tevolve.termoweb.net"
    assert parsed.path == "/socket.io/"
    assert dict(parse_qsl(parsed.query)) == params
