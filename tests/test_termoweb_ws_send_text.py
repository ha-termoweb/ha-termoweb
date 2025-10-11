"""Tests for TermoWebWSClient._send_text behavior."""

import asyncio

import pytest

from custom_components.termoweb.backend.termoweb_ws import TermoWebWSClient


class StubSession:
    """Minimal stub for aiohttp.ClientSession."""


class StubAPIClient:
    """API client stub with attachable attributes."""


class StubHass:
    """Home Assistant stub exposing the event loop."""

    def __init__(self) -> None:
        """Store the asyncio loop for the websocket client."""

        self.loop = asyncio.get_event_loop()


class StubWebSocket:
    """Stub websocket recording sent text frames."""

    def __init__(self) -> None:
        """Prepare the frame storage list."""

        self.sent_frames: list[str] = []

    async def send_str(self, data: str) -> None:
        """Record the websocket frame payload."""

        self.sent_frames.append(data)


@pytest.mark.asyncio
async def test_send_text_without_connection_raises_runtime_error() -> None:
    """Raise RuntimeError when attempting to send without websocket."""

    hass = StubHass()
    client = TermoWebWSClient(
        hass,
        entry_id="entry",
        dev_id="dev",
        api_client=StubAPIClient(),
        coordinator=object(),
        session=StubSession(),
    )

    with pytest.raises(RuntimeError, match="websocket not connected"):
        await client._send_text("payload")


@pytest.mark.asyncio
async def test_send_text_with_stub_websocket_sends_payload() -> None:
    """Send payload via stub websocket without raising."""

    hass = StubHass()
    client = TermoWebWSClient(
        hass,
        entry_id="entry",
        dev_id="dev",
        api_client=StubAPIClient(),
        coordinator=object(),
        session=StubSession(),
    )
    stub_ws = StubWebSocket()
    client._ws = stub_ws

    await client._send_text("payload")

    assert stub_ws.sent_frames == ["payload"]
