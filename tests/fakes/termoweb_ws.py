"""Transport doubles for the websocket clients.

``WSFakeSession`` extends the REST session double with the two calls the
websocket clients make: ``get`` (Socket.IO / Engine.IO polling handshakes) and
``ws_connect``. ``FakeWebSocket`` is a scriptable ``ClientWebSocketResponse``.
``SleepController`` replaces ``asyncio.sleep`` inside the websocket modules so
reconnect backoffs pass instantly while periodic loops park until released.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
import copy
from types import ModuleType
from typing import Any, Self

import aiohttp
import pytest

from tests.fakes.rest import FakeSession, MockResponse

_END = object()


def token_response(token: str = "tok") -> MockResponse:
    """Return a successful OAuth token response."""
    return MockResponse(
        200,
        {"access_token": token, "expires_in": 3600},
        headers={"Content-Type": "application/json"},
    )


class FakeWebSocket:
    """Scriptable ``aiohttp.ClientWebSocketResponse``.

    Frames are queued with :meth:`feed`; :meth:`end` (or :meth:`close`) ends
    the stream. Both ``async for`` and ``receive()`` are supported.
    """

    close_code: int | None = None
    close_reason: str | None = None

    def __init__(self, url: str = "", frames: tuple[str, ...] = ()) -> None:
        """Queue the initial ``frames`` as TEXT messages."""
        self.url = url
        self.sent: list[str] = []
        self.closed = False
        self.close_calls: list[dict[str, Any]] = []
        self.error: BaseException | None = None
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        for frame in frames:
            self.feed(frame)

    def feed(self, data: str | bytes) -> None:
        """Queue a server TEXT (str) or BINARY (bytes) frame."""
        msg_type = (
            aiohttp.WSMsgType.BINARY
            if isinstance(data, bytes)
            else aiohttp.WSMsgType.TEXT
        )
        self.feed_message(msg_type, data)

    def feed_message(self, msg_type: aiohttp.WSMsgType, data: Any = None) -> None:
        """Queue a raw message of ``msg_type``."""
        self._queue.put_nowait(aiohttp.WSMessage(msg_type, data, None))

    def end(self) -> None:
        """End the stream once the queued frames are consumed."""
        self._queue.put_nowait(_END)

    async def send_str(self, data: str) -> None:
        """Record a client frame."""
        if self.closed:
            raise aiohttp.ClientConnectionResetError("send on closed websocket")
        self.sent.append(data)

    async def close(self, **kwargs: Any) -> bool:
        """Close the socket and end the stream."""
        self.close_calls.append(kwargs)
        if not self.closed:
            self.closed = True
            self.end()
        return True

    def exception(self) -> BaseException | None:
        """Return the transport error reported with an ERROR message."""
        return self.error

    async def _next(self) -> Any:
        """Return the next queued item, keeping the end marker sticky."""
        item = await self._queue.get()
        if item is _END:
            self._queue.put_nowait(_END)
        return item

    async def receive(self, timeout: float | None = None) -> aiohttp.WSMessage:
        """Return the next message; CLOSED once the stream ended."""
        item = await self._next()
        if item is _END:
            return aiohttp.WSMessage(aiohttp.WSMsgType.CLOSED, None, None)
        return item

    async def receive_str(self, timeout: float | None = None) -> str:
        """Return the next TEXT frame, like ``aiohttp`` (else raise TypeError)."""
        msg = await self.receive()
        if msg.type is not aiohttp.WSMsgType.TEXT:
            raise TypeError(f"Received message {msg.type} is not str")
        return msg.data

    def __aiter__(self) -> FakeWebSocket:
        """Iterate over the queued messages."""
        return self

    async def __anext__(self) -> aiohttp.WSMessage:
        """Return the next message or stop at the end of the stream."""
        item = await self._next()
        if item is _END:
            raise StopAsyncIteration
        return item


class HandshakeResponse:
    """Response double for ``session.get`` handshakes (``text`` and ``read``)."""

    def __init__(self, status: int = 200, body: str | bytes = "") -> None:
        """Store the status and the body."""
        self.status = status
        self._body = body

    async def __aenter__(self) -> Self:
        """Enter the request context."""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Leave the request context."""

    async def text(self) -> str:
        """Return the body as text."""
        body = self._body
        return body.decode() if isinstance(body, bytes) else body

    async def read(self) -> bytes:
        """Return the body as bytes."""
        body = self._body
        return body if isinstance(body, bytes) else body.encode()


class WSFakeSession(FakeSession):
    """REST session double that also serves websocket handshakes and upgrades.

    ``get`` pops scripted responses (or exceptions) from :attr:`handshakes`;
    when none are queued it calls :attr:`default_get` (default: a healthy
    Socket.IO 0.9 handshake with a fresh sid). ``post`` to a ``/socket.io/``
    URL is an Engine.IO polling POST and answers 200; other POSTs are REST.
    ``ws_connect`` returns a new :class:`FakeWebSocket` (queued with
    :attr:`greeting`) unless :attr:`connect` replaces it.
    """

    def __init__(self) -> None:
        """Start with no scripted handshakes or sockets."""
        super().__init__()
        self.handshakes: deque[Any] = deque()
        self.get_calls: list[tuple[str, dict[str, Any]]] = []
        self.connect_calls: list[tuple[str, dict[str, Any]]] = []
        self.sockets: list[FakeWebSocket] = []
        self.connect: Callable[..., Awaitable[Any]] | None = None
        self.default_get: Callable[[str], HandshakeResponse] = self._socketio_handshake
        self.polling_posts: list[tuple[str, dict[str, Any]]] = []
        self.greeting: tuple[str, ...] = ()

    def _socketio_handshake(self, _url: str) -> HandshakeResponse:
        """Answer a Socket.IO 0.9 handshake with a fresh sid."""
        return HandshakeResponse(200, f"sid{len(self.get_calls)}:60:60:websocket")

    def push_request(self, *responses: Any) -> None:
        """Queue responses for ``request`` ahead of those already queued."""
        self._request_queue[0:0] = responses

    def get(self, url: str, **kwargs: Any) -> HandshakeResponse:
        """Record a handshake GET and return the next scripted response."""
        self.get_calls.append((url, copy.deepcopy(kwargs)))
        if self.handshakes:
            result = self.handshakes.popleft()
            if isinstance(result, BaseException):
                raise result
            return result
        return self.default_get(url)

    def post(self, url: str, *, data: Any = None, **kwargs: Any) -> Any:
        """Answer Engine.IO polling POSTs; delegate REST POSTs."""
        if "/socket.io/" in url:
            self.polling_posts.append((url, {"data": data, **kwargs}))
            return HandshakeResponse(200, b"ok")
        return super().post(url, data=data, **kwargs)

    async def ws_connect(self, url: str, **kwargs: Any) -> FakeWebSocket:
        """Record the upgrade and return a new scriptable socket."""
        self.connect_calls.append((url, kwargs))
        if self.connect is not None:
            return await self.connect(url, **kwargs)
        socket = FakeWebSocket(url, self.greeting)
        self.sockets.append(socket)
        return socket


class _AsyncioProxy(ModuleType):
    """``asyncio`` stand-in whose ``sleep`` is the controller's."""

    def __init__(self, sleep: Callable[..., Awaitable[Any]]) -> None:
        """Bind the replacement sleep."""
        super().__init__("asyncio")
        self.sleep = sleep

    def __getattr__(self, name: str) -> Any:
        """Delegate everything else to the real module."""
        return getattr(asyncio, name)


class SleepController:
    """Fake ``asyncio.sleep`` for the websocket modules.

    Sleeps issued by the ``runner`` task (reconnect backoff) are recorded in
    :attr:`backoffs` and return at once; connection rate-limiter waits (from
    the module passed as ``limiter_module``) go to :attr:`throttled`. All
    other sleeps (heartbeat, keep-alive and idle-monitor loops) park until
    :meth:`release` resolves them or their task is cancelled.
    """

    def __init__(self) -> None:
        """Start with nothing recorded."""
        self.runner: asyncio.Task[Any] | None = None
        self.backoffs: list[float] = []
        self.throttled: list[float] = []
        self.parked: list[tuple[float, asyncio.Future[None]]] = []
        self.on_backoff: Callable[[float], None] | None = None

    def install(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *modules: ModuleType,
        limiter_module: ModuleType | None = None,
    ) -> None:
        """Replace ``asyncio`` in ``modules`` (and the limiter's) with proxies.

        Install before building the client: the rate limiter binds its sleep
        when it is created.
        """
        proxy = _AsyncioProxy(self.sleep)
        for module in modules:
            monkeypatch.setattr(module, "asyncio", proxy)
        if limiter_module is not None:
            monkeypatch.setattr(
                limiter_module, "asyncio", _AsyncioProxy(self._throttle)
            )

    async def _throttle(self, delay: float, result: Any = None) -> Any:
        """Record a rate-limiter wait and return at once."""
        if delay > 0:
            self.throttled.append(delay)
        await asyncio.sleep(0)
        return result

    async def sleep(self, delay: float, result: Any = None) -> Any:
        """Record ``delay``; return now for the runner, park otherwise."""
        if delay <= 0:
            await asyncio.sleep(0)
            return result
        if self.runner is not None and asyncio.current_task() is self.runner:
            self.backoffs.append(delay)
            if self.on_backoff is not None:
                self.on_backoff(delay)
            await asyncio.sleep(0)
            return result
        future = asyncio.get_running_loop().create_future()
        self.parked.append((delay, future))
        await future
        return result

    def parked_delays(self) -> list[float]:
        """Return the delays of the sleeps still parked."""
        return [delay for delay, future in self.parked if not future.done()]

    def release(self, delay: float) -> int:
        """Wake every parked sleep of ``delay`` seconds; return how many."""
        woken = 0
        for parked_delay, future in self.parked:
            if parked_delay == delay and not future.done():
                future.set_result(None)
                woken += 1
        self.parked = [item for item in self.parked if not item[1].done()]
        return woken


async def until(predicate: Callable[[], bool], *, timeout: float = 5) -> None:
    """Yield to the loop until ``predicate`` holds (bounded)."""
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0)
