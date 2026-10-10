"""Virtual clock and fake aiohttp transport for websocket protocol scenarios.

The websocket clients are driven through their real entry points
(``start``/``stop``) against a scripted aiohttp session. Time is virtual: the
clock replaces ``time`` and ``asyncio.sleep``/``timeout``/``wait_for`` inside
the websocket modules only, so reconnect backoff, heartbeats, idle windows and
handshake timeouts run instantly and deterministically while Home Assistant
keeps its real event loop and clock.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
import contextlib
import heapq
import itertools
import json
import time
from types import ModuleType, SimpleNamespace
from typing import Any

import aiohttp
import pytest

from custom_components.termoweb.backend import ws_client, ws_health

_real_sleep = asyncio.sleep


class VirtualClock:
    """Deterministic clock whose sleeps and timeouts fire on ``advance``."""

    def __init__(self) -> None:
        """Start at the real wall time so timestamps look plausible."""
        self.now = time.time()
        self._timers: list[tuple[float, int, Callable[[], None]]] = []
        self._seq = itertools.count()
        self.sleeps: list[float] = []

    def time(self) -> float:
        """Return the virtual wall clock."""
        return self.now

    def monotonic(self) -> float:
        """Return the virtual monotonic clock."""
        return self.now

    def _call_at(self, when: float, callback: Callable[[], None]) -> None:
        """Run ``callback`` once virtual time reaches ``when``."""
        heapq.heappush(self._timers, (when, next(self._seq), callback))

    async def sleep(self, delay: float, result: Any = None) -> Any:
        """Sleep in virtual time; zero-delay sleeps just yield."""
        if delay <= 0:
            return await _real_sleep(0, result)
        self.sleeps.append(delay)
        fut = asyncio.get_running_loop().create_future()

        def _wake() -> None:
            if not fut.done():
                fut.set_result(result)

        self._call_at(self.now + delay, _wake)
        return await fut

    def timeout(self, delay: float | None) -> asyncio.Timeout:
        """Return a real ``asyncio.Timeout`` that expires in virtual time."""
        cm = asyncio.timeout(None)
        if delay is None:
            return cm

        def _expire() -> None:
            with contextlib.suppress(RuntimeError):  # already left the block
                cm.reschedule(asyncio.get_running_loop().time())

        self._call_at(self.now + delay, _expire)
        return cm

    async def wait_for(self, aw: Any, timeout: float | None) -> Any:
        """Await ``aw`` with a virtual-time timeout."""
        if timeout is None:
            return await aw
        async with self.timeout(timeout):
            return await aw

    async def advance(self, seconds: float) -> None:
        """Move time forward, firing due timers in order and letting tasks run."""
        target = self.now + seconds
        await settle()
        while self._timers and self._timers[0][0] <= target:
            when, _, callback = heapq.heappop(self._timers)
            self.now = max(self.now, when)
            callback()
            await settle()
        self.now = target
        await settle()


class _AsyncioProxy(ModuleType):
    """``asyncio`` stand-in routing sleeps and timeouts to the virtual clock."""

    def __init__(self, clock: VirtualClock) -> None:
        """Bind the virtual sleep, timeout and wait_for."""
        super().__init__("asyncio")
        self.sleep = clock.sleep
        self.timeout = clock.timeout
        self.wait_for = clock.wait_for

    def __getattr__(self, name: str) -> Any:
        """Delegate everything else to the real asyncio."""
        return getattr(asyncio, name)


def install_clock(
    monkeypatch: pytest.MonkeyPatch, *modules: ModuleType
) -> VirtualClock:
    """Give ``modules`` (plus the shared ws helpers) a virtual clock."""
    clock = VirtualClock()
    fake_time = SimpleNamespace(time=clock.time, monotonic=clock.monotonic)
    proxy = _AsyncioProxy(clock)
    for module in (ws_client, ws_health, *modules):
        monkeypatch.setattr(module, "time", fake_time)
        if hasattr(module, "asyncio"):
            monkeypatch.setattr(module, "asyncio", proxy)
    return clock


async def settle(rounds: int = 30) -> None:
    """Let ready tasks run until the fake transport has nothing left to do."""
    for _ in range(rounds):
        await _real_sleep(0)


async def until(predicate: Callable[[], bool], what: str = "condition") -> None:
    """Yield to the loop until ``predicate`` holds (bounded in real time)."""
    try:
        async with asyncio.timeout(5):
            while not predicate():
                await _real_sleep(0.001)
    except TimeoutError:
        raise AssertionError(f"timed out waiting for {what}") from None


def _msg(kind: aiohttp.WSMsgType, data: Any = None) -> SimpleNamespace:
    """Build an aiohttp-like websocket message."""
    return SimpleNamespace(type=kind, data=data, extra=None)


class FakeWS:
    """Scriptable aiohttp websocket mirroring ``ClientWebSocketResponse``."""

    def __init__(self, url: str) -> None:
        """Create an open socket with an empty inbound queue."""
        self.url = url
        self.sent: list[str] = []
        self.closed = False
        self.close_code: int | None = None
        self.close_calls: list[tuple[int | None, bytes | None]] = []
        self._inbox: asyncio.Queue[SimpleNamespace] = asyncio.Queue()

    def feed(self, *frames: str) -> None:
        """Queue server TEXT frames."""
        for frame in frames:
            self._inbox.put_nowait(_msg(aiohttp.WSMsgType.TEXT, frame))

    def feed_binary(self, data: bytes) -> None:
        """Queue a server BINARY frame."""
        self._inbox.put_nowait(_msg(aiohttp.WSMsgType.BINARY, data))

    def server_close(self, code: int = 1000) -> None:
        """Simulate the server closing the connection."""
        self.closed = True
        self.close_code = code
        self._inbox.put_nowait(_msg(aiohttp.WSMsgType.CLOSE, code))

    def drop(self) -> None:
        """Simulate the transport dying without a close handshake."""
        self.closed = True
        self.close_code = 1006
        self._inbox.put_nowait(_msg(aiohttp.WSMsgType.CLOSED))

    async def send_str(self, data: str) -> None:
        """Record a client frame; fail like aiohttp once closed."""
        if self.closed:
            raise aiohttp.ClientConnectionResetError(
                "Cannot write to closing transport"
            )
        self.sent.append(data)

    async def close(
        self, *, code: int | None = None, message: bytes | None = None
    ) -> bool:
        """Close from the client side and wake any reader."""
        self.close_calls.append((code, message))
        if not self.closed:
            self.closed = True
            self.close_code = code
            self._inbox.put_nowait(_msg(aiohttp.WSMsgType.CLOSED))
        return True

    def exception(self) -> BaseException | None:
        """Return no transport error."""
        return None

    async def receive(self) -> SimpleNamespace:
        """Return the next inbound message."""
        return await self._inbox.get()

    async def receive_str(self) -> str:
        """Return the next TEXT frame like aiohttp does."""
        msg = await self.receive()
        if msg.type != aiohttp.WSMsgType.TEXT:
            raise TypeError(f"Received message {msg.type} is not str")
        return msg.data

    def __aiter__(self) -> FakeWS:
        """Iterate inbound messages like aiohttp."""
        return self

    async def __anext__(self) -> SimpleNamespace:
        """Stop iterating on close, like aiohttp."""
        msg = await self.receive()
        if msg.type in (
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSING,
            aiohttp.WSMsgType.CLOSED,
        ):
            raise StopAsyncIteration
        return msg


class FakeResponse:
    """Async context manager standing in for an aiohttp response."""

    def __init__(
        self,
        *,
        status: int = 200,
        body: bytes | str = b"",
        error: BaseException | None = None,
        hang: bool = False,
    ) -> None:
        """Script the response status, body, failure or a never-ending read."""
        self.status = status
        self._body = body.encode() if isinstance(body, str) else body
        self._error = error
        self._hang = hang

    async def __aenter__(self) -> FakeResponse:  # noqa: PYI034
        """Enter the response, or raise the scripted transport error."""
        if self._error is not None:
            raise self._error
        return self

    async def __aexit__(self, *_exc: object) -> None:
        """Release nothing."""

    async def _payload(self) -> bytes:
        if self._hang:
            await asyncio.get_running_loop().create_future()
        return self._body

    async def text(self) -> str:
        """Return the body as text."""
        return (await self._payload()).decode()

    async def read(self) -> bytes:
        """Return the raw body."""
        return await self._payload()


class FakeSession:
    """aiohttp.ClientSession stand-in serving scripted HTTP and websockets.

    ``responses`` maps ``"GET"``/``"POST"`` to callables building the next
    response from the request URL; every websocket opened is kept in
    ``sockets`` and pre-loaded with the frames returned by ``ws_frames``.
    """

    closed = False

    def __init__(
        self,
        *,
        get: Callable[[str], FakeResponse],
        post: Callable[[str], FakeResponse] | None = None,
        ws_frames: Callable[[int], Iterable[str]] | None = None,
        ws_factory: Callable[[str], FakeWS] | None = None,
    ) -> None:
        """Store the request handlers."""
        self._get = get
        self._post = post
        self._ws_frames = ws_frames
        self._ws_factory = ws_factory or FakeWS
        self.requests: list[tuple[str, str]] = []
        self.sockets: list[FakeWS] = []

    def get(self, url: str, **_kwargs: Any) -> FakeResponse:
        """Serve a GET."""
        self.requests.append(("GET", url))
        return self._get(url)

    def post(self, url: str, **_kwargs: Any) -> FakeResponse:
        """Serve a POST."""
        self.requests.append(("POST", url))
        assert self._post is not None
        return self._post(url)

    async def ws_connect(self, url: str, **_kwargs: Any) -> FakeWS:
        """Open a scripted websocket."""
        ws = self._ws_factory(url)
        if self._ws_frames is not None:
            ws.feed(*self._ws_frames(len(self.sockets)))
        self.sockets.append(ws)
        return ws


def eio_polling_body(*packets: str) -> bytes:
    """Encode Engine.IO v3 binary polling packets."""
    out = b""
    for pkt in packets:
        data = pkt.encode()
        out += b"\x00" + bytes(int(d) for d in str(len(data))) + b"\xff" + data
    return out


def sio09_event(namespace: str, name: str, payload: Any) -> str:
    """Return a Socket.IO 0.9 event frame."""
    body = json.dumps({"name": name, "args": [payload]}, separators=(",", ":"))
    return f"5::{namespace}:{body}"


def sio_event(namespace: str, name: str, *args: Any) -> str:
    """Return a Socket.IO v2 (Engine.IO v3) event frame."""
    return f"42{namespace}," + json.dumps([name, *args], separators=(",", ":"))


def leaked_tasks(prefix: str) -> list[asyncio.Task[Any]]:
    """Return unfinished tasks whose coroutine belongs to ``prefix``."""
    return [
        task
        for task in asyncio.all_tasks()
        if not task.done()
        and getattr(task.get_coro(), "__qualname__", "").startswith(prefix)
    ]
