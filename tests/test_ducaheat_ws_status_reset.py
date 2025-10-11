"""Unit tests for websocket status health resets."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from custom_components.termoweb.backend.ducaheat_ws import DucaheatWSClient


def test_status_should_reset_health() -> None:
    """Ducaheat websockets reset health unless status is healthy."""

    loop = asyncio.new_event_loop()
    try:
        hass = SimpleNamespace(loop=loop, data={})
        session = SimpleNamespace()
        api_client = SimpleNamespace(_session=session)
        coordinator = SimpleNamespace(loop=loop, data={})

        client = DucaheatWSClient(
            hass,
            entry_id="entry",
            dev_id="device",
            api_client=api_client,  # type: ignore[arg-type]
            coordinator=coordinator,
            session=session,  # type: ignore[arg-type]
        )

        assert client._status_should_reset_health("connected") is True
        assert client._status_should_reset_health("healthy") is False
    finally:
        loop.close()
