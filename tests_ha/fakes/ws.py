"""Doubles for websocket client tests: REST client, session and sockets."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping
from types import SimpleNamespace
from typing import Any, Self
from unittest.mock import AsyncMock

import aiohttp
from homeassistant.core import HomeAssistant

from custom_components.termoweb.coordinator import StateCoordinator
from custom_components.termoweb.inventory import Inventory, build_node_inventory
from custom_components.termoweb.runtime import EntryRuntime
from tests_ha.fakes.runtime import build_entry_runtime

ENTRY_ID = "entry"
DEV_ID = "device"


class DummyREST:
    """The parts of ``RESTClient`` a websocket client uses."""

    def __init__(
        self,
        *,
        api_base: str | None = "https://api.termoweb",
        authed_headers: dict[str, str] | None = None,
        session: Any = None,
    ) -> None:
        """Record the configured token headers and session."""
        self._session = session if session is not None else SimpleNamespace()
        self._headers = authed_headers or {"Authorization": "Bearer token"}
        self._ensure_token = AsyncMock()
        self._access_token: str | None = "token"
        self.api_base = api_base
        self.user_agent = "agent"
        self.requested_with = "requested"

    async def authed_headers(self) -> dict[str, str]:
        """Return the bearer headers."""
        return self._headers

    def normalise_ws_nodes(self, nodes: dict[str, Any]) -> dict[str, Any]:
        """Return the nodes unchanged (codec pass-through)."""
        return nodes

    async def refresh_token(self) -> None:
        """Drop the cached token and fetch a new one."""
        self._access_token = None
        await self._ensure_token()


def make_inventory(
    nodes: Iterable[Mapping[str, Any]] = ({"type": "htr", "addr": "1"},),
    dev_id: str = DEV_ID,
) -> Inventory:
    """Return an inventory with ``nodes``."""
    return Inventory(dev_id, build_node_inventory({"nodes": list(nodes)}))


def make_runtime(
    hass: HomeAssistant,
    inventory: Inventory | None = None,
    *,
    energy_coordinator: Any | None = None,
) -> EntryRuntime:
    """Attach a runtime with a real ``StateCoordinator`` to a config entry."""
    inventory = inventory or make_inventory()
    coordinator = StateCoordinator(
        hass, DummyREST(), 30, DEV_ID, {"name": "Home"}, inventory
    )
    return build_entry_runtime(
        hass=hass,
        entry_id=ENTRY_ID,
        dev_id=DEV_ID,
        inventory=inventory,
        coordinator=coordinator,
        energy_coordinator=energy_coordinator,
    )


# Ducaheat Engine.IO transport doubles

OPEN_PACKET = '0{"sid":"abc","pingInterval":25000,"pingTimeout":60000}'


def polling_body(*packets: str) -> bytes:
    """Return an Engine.IO v3 binary polling payload carrying ``packets``."""
    body = b""
    for packet in packets:
        data = packet.encode()
        digits = [int(d) for d in str(len(data))]
        body += bytes([0, *digits, 0xFF]) + data
    return body


class StubResponse:
    """``aiohttp`` response context manager with a fixed status and body."""

    def __init__(self, *, status: int = 200, body: bytes = b"") -> None:
        """Store the canned status and body."""
        self.status = status
        self._body = body

    async def read(self) -> bytes:
        """Return the body."""
        return self._body

    async def __aenter__(self) -> Self:
        """Enter the response context."""
        return self

    async def __aexit__(self, *_: object) -> None:
        """Leave the response context."""


class StubWebSocket:
    """Websocket used during the handshake: queued text frames, recorded sends."""

    def __init__(self, frames: Iterable[str] = ("3probe",)) -> None:
        """Queue the frames the server will send."""
        self.sent: list[str] = []
        self.closed = False
        self._receive: asyncio.Queue[str] = asyncio.Queue()
        for frame in frames:
            self._receive.put_nowait(frame)

    async def send_str(self, payload: str) -> None:
        """Record an outgoing frame."""
        self.sent.append(payload)

    async def receive_str(self) -> str:
        """Return the next queued frame (waits forever when empty)."""
        return await self._receive.get()

    async def close(self, *, code: int, message: bytes) -> None:
        """Record the close handshake."""
        self.closed = True
        self.close_args = (code, message)


class QueueWebSocket:
    """Websocket used by the read loop: replays messages, then closes."""

    def __init__(self, messages: Iterable[Any]) -> None:
        """Queue ``aiohttp`` messages (or text frames) for ``receive``."""
        self.closed = False
        self.sent: list[str] = []
        self._messages = [
            SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=m)
            if isinstance(m, str)
            else m
            for m in messages
        ]

    async def receive(self) -> Any:
        """Return the next message, then a CLOSE."""
        if self._messages:
            return self._messages.pop(0)
        return SimpleNamespace(type=aiohttp.WSMsgType.CLOSE, data=None)

    async def send_str(self, payload: str) -> None:
        """Record an outgoing frame."""
        self.sent.append(payload)

    def exception(self) -> str:
        """Return the transport error of an ERROR message."""
        return "boom"


class StubSession:
    """``aiohttp`` session answering the Engine.IO polling handshake."""

    def __init__(self, ws: StubWebSocket) -> None:
        """Serve ``ws`` on upgrade; every response is a 200 by default."""
        self.calls: list[tuple[str, str]] = []
        self.ws = ws
        self.open_body = polling_body(OPEN_PACKET)
        self.status = {"open": 200, "post": 200, "drain": 200}

    def get(self, url: str, *, headers: dict[str, str]) -> StubResponse:
        """Answer the open GET (no sid) and the drain GET (with sid)."""
        self.calls.append(("GET", url))
        if "sid=" not in url:
            return StubResponse(status=self.status["open"], body=self.open_body)
        return StubResponse(status=self.status["drain"], body=b"6:40[]")

    def post(self, url: str, *, headers: dict[str, str], data: bytes) -> StubResponse:
        """Accept the namespace POST."""
        self.calls.append(("POST", url))
        return StubResponse(status=self.status["post"])

    async def ws_connect(self, url: str, **_: Any) -> StubWebSocket:
        """Return the websocket."""
        self.calls.append(("WS", url))
        return self.ws
