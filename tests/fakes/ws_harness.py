"""The websocket test harness: clocks, a fake aiohttp transport and REST doubles.

The websocket clients are driven through their real entry points against a
scripted aiohttp session (:class:`WSFakeSession`) that hands out scriptable
sockets (:class:`FakeWS`). Two ways to control time:

* :class:`VirtualClock` (via :func:`install_clock`) replaces ``time`` and
  ``asyncio.sleep``/``timeout``/``wait_for`` inside the websocket modules, so
  backoffs, heartbeats, idle windows and handshake timeouts run instantly and
  deterministically on ``advance`` while Home Assistant keeps its real loop.
* :class:`SleepController` returns reconnect backoffs at once and parks every
  other sleep until the test releases it; :class:`OffsetClock` shifts the
  module's wall clock.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable, Iterable, Mapping
import contextlib
import copy
import heapq
import itertools
import json
import time
from types import ModuleType, SimpleNamespace
from typing import Any, Self
from unittest.mock import AsyncMock

import aiohttp
from homeassistant.core import HomeAssistant
import pytest

from custom_components.termoweb.backend import ws_client, ws_health
from custom_components.termoweb.coordinator import StateCoordinator
from custom_components.termoweb.inventory import Inventory, build_node_inventory
from custom_components.termoweb.runtime import EntryRuntime
from tests.fakes.rest import FakeSession, MockResponse
from tests.fakes.runtime import build_entry_runtime

_real_sleep = asyncio.sleep
_END = object()

ENTRY_ID = "entry"
DEV_ID = "device"


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------


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
    """``asyncio`` stand-in routing the given functions to a fake clock."""

    def __init__(self, **overrides: Callable[..., Any]) -> None:
        """Bind the replacement ``sleep`` (and optionally ``timeout``/``wait_for``)."""
        super().__init__("asyncio")
        for name, func in overrides.items():
            setattr(self, name, func)

    def __getattr__(self, name: str) -> Any:
        """Delegate everything else to the real asyncio."""
        return getattr(asyncio, name)


def install_clock(
    monkeypatch: pytest.MonkeyPatch, *modules: ModuleType
) -> VirtualClock:
    """Give ``modules`` (plus the shared ws helpers) a virtual clock."""
    clock = VirtualClock()
    fake_time = SimpleNamespace(time=clock.time, monotonic=clock.monotonic)
    proxy = _AsyncioProxy(
        sleep=clock.sleep, timeout=clock.timeout, wait_for=clock.wait_for
    )
    for module in (ws_client, ws_health, *modules):
        monkeypatch.setattr(module, "time", fake_time)
        if hasattr(module, "asyncio"):
            monkeypatch.setattr(module, "asyncio", proxy)
    return clock


class OffsetClock:
    """Wall clock for a websocket module: real time plus a test offset."""

    def __init__(self) -> None:
        """Start without an offset."""
        self.offset = 0.0

    def time(self) -> float:
        """Return the shifted wall-clock time."""
        return time.time() + self.offset


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
        proxy = _AsyncioProxy(sleep=self.sleep)
        for module in modules:
            monkeypatch.setattr(module, "asyncio", proxy)
        if limiter_module is not None:
            monkeypatch.setattr(
                limiter_module, "asyncio", _AsyncioProxy(sleep=self._throttle)
            )

    async def _throttle(self, delay: float, result: Any = None) -> Any:
        """Record a rate-limiter wait and return at once."""
        if delay > 0:
            self.throttled.append(delay)
        await _real_sleep(0)
        return result

    async def sleep(self, delay: float, result: Any = None) -> Any:
        """Record ``delay``; return now for the runner, park otherwise."""
        if delay <= 0:
            await _real_sleep(0)
            return result
        if self.runner is not None and asyncio.current_task() is self.runner:
            self.backoffs.append(delay)
            if self.on_backoff is not None:
                self.on_backoff(delay)
            await _real_sleep(0)
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


async def settle(rounds: int = 30) -> None:
    """Let ready tasks run until the fake transport has nothing left to do."""
    for _ in range(rounds):
        await _real_sleep(0)


async def until(
    predicate: Callable[[], bool], what: str = "condition", *, timeout: float = 5
) -> None:
    """Yield to the loop until ``predicate`` holds (bounded in real time)."""
    try:
        async with asyncio.timeout(timeout):
            while not predicate():
                await _real_sleep(0.001)
    except TimeoutError:
        raise AssertionError(f"timed out waiting for {what}") from None


def leaked_tasks(prefix: str) -> list[asyncio.Task[Any]]:
    """Return unfinished tasks whose coroutine belongs to ``prefix``."""
    return [
        task
        for task in asyncio.all_tasks()
        if not task.done()
        and getattr(task.get_coro(), "__qualname__", "").startswith(prefix)
    ]


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


class FakeWS:
    """Scriptable ``aiohttp.ClientWebSocketResponse``.

    Server frames are queued with :meth:`feed` (or :meth:`feed_message`).
    :meth:`end`, :meth:`drop` and a client :meth:`close` end the stream for
    good: ``receive`` then returns CLOSED and ``async for`` stops.
    """

    def __init__(self, url: str = "", frames: Iterable[str] = ()) -> None:
        """Create an open socket with ``frames`` queued as TEXT messages."""
        self.url = url
        self.sent: list[str] = []
        self.closed = False
        self.close_code: int | None = None
        self.close_calls: list[tuple[int | None, bytes | None]] = []
        self.error: BaseException | None = None
        self._inbox: asyncio.Queue[Any] = asyncio.Queue()
        self.feed(*frames)

    def feed(self, *frames: str | bytes) -> None:
        """Queue server TEXT (str) or BINARY (bytes) frames."""
        for frame in frames:
            kind = (
                aiohttp.WSMsgType.BINARY
                if isinstance(frame, bytes)
                else aiohttp.WSMsgType.TEXT
            )
            self.feed_message(kind, frame)

    def feed_message(self, kind: aiohttp.WSMsgType, data: Any = None) -> None:
        """Queue a raw message of ``kind``."""
        self._inbox.put_nowait(aiohttp.WSMessage(kind, data, None))

    def end(self) -> None:
        """End the stream once the queued frames are consumed."""
        self._inbox.put_nowait(_END)

    def server_close(self, code: int = 1000) -> None:
        """Simulate the server closing the connection."""
        self.closed = True
        self.close_code = code
        self.feed_message(aiohttp.WSMsgType.CLOSE, code)

    def drop(self) -> None:
        """Simulate the transport dying without a close handshake."""
        self.closed = True
        self.close_code = 1006
        self.end()

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
            self.end()
        return True

    def exception(self) -> BaseException | None:
        """Return the transport error reported with an ERROR message."""
        return self.error

    async def receive(self, timeout: float | None = None) -> aiohttp.WSMessage:
        """Return the next message; CLOSED once the stream ended."""
        item = await self._inbox.get()
        if item is _END:
            self._inbox.put_nowait(_END)  # stay ended
            return aiohttp.WSMessage(aiohttp.WSMsgType.CLOSED, None, None)
        return item

    async def receive_str(self, timeout: float | None = None) -> str:
        """Return the next TEXT frame like aiohttp does (else raise TypeError)."""
        msg = await self.receive()
        if msg.type is not aiohttp.WSMsgType.TEXT:
            raise TypeError(f"Received message {msg.type} is not str")
        return msg.data

    def __aiter__(self) -> FakeWS:
        """Iterate inbound messages like aiohttp."""
        return self

    async def __anext__(self) -> aiohttp.WSMessage:
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
    """Async context manager standing in for an aiohttp handshake response."""

    def __init__(
        self,
        status: int = 200,
        body: bytes | str = b"",
        *,
        error: BaseException | None = None,
        hang: bool = False,
    ) -> None:
        """Script the response status, body, failure or a never-ending read."""
        self.status = status
        self._body = body.encode() if isinstance(body, str) else body
        self._error = error
        self._hang = hang

    async def __aenter__(self) -> Self:
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


class WSFakeSession(FakeSession):
    """REST session double that also serves websocket handshakes and upgrades.

    ``get`` pops scripted responses (or exceptions) from :attr:`handshakes`,
    else calls :attr:`default_get` (default: a healthy Socket.IO 0.9
    handshake with a fresh sid). A POST to a ``/socket.io/`` URL is an
    Engine.IO polling POST answered by :attr:`polling_post` (default: 200);
    other POSTs are REST. ``ws_connect`` builds a socket with
    :attr:`ws_factory`, queues ``ws_frames(index)`` (or :attr:`greeting`) on it
    and keeps it in :attr:`sockets`, unless :attr:`connect` replaces it.
    :attr:`calls` logs every handshake GET, polling POST and upgrade in order.
    """

    def __init__(
        self,
        *,
        get: Callable[[str], FakeResponse] | None = None,
        post: Callable[[str], FakeResponse] | None = None,
        ws_frames: Callable[[int], Iterable[str]] | None = None,
        ws_factory: Callable[[str], FakeWS] | None = None,
    ) -> None:
        """Store the request handlers; nothing is scripted by default."""
        super().__init__()
        self.handshakes: deque[Any] = deque()
        self.default_get = get or self._socketio_handshake
        self.polling_post = post or (lambda _url: FakeResponse(200, b"ok"))
        self.ws_frames = ws_frames
        self.ws_factory = ws_factory or FakeWS
        self.greeting: tuple[str, ...] = ()
        self.connect: Callable[..., Awaitable[Any]] | None = None
        self.calls: list[tuple[str, str]] = []
        self.get_calls: list[tuple[str, dict[str, Any]]] = []
        self.connect_calls: list[tuple[str, dict[str, Any]]] = []
        self.sockets: list[FakeWS] = []

    def _socketio_handshake(self, _url: str) -> FakeResponse:
        """Answer a Socket.IO 0.9 handshake with a fresh sid."""
        return FakeResponse(200, f"sid{len(self.get_calls)}:60:60:websocket")

    def push_request(self, *responses: Any) -> None:
        """Queue responses for ``request`` ahead of those already queued."""
        self._request_queue[0:0] = responses

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        """Record a handshake GET and return the next scripted response."""
        self.calls.append(("GET", url))
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
            self.calls.append(("POST", url))
            return self.polling_post(url)
        return super().post(url, data=data, **kwargs)

    async def ws_connect(self, url: str, **kwargs: Any) -> FakeWS:
        """Record the upgrade and return a scripted socket."""
        self.calls.append(("WS", url))
        self.connect_calls.append((url, kwargs))
        if self.connect is not None:
            return await self.connect(url, **kwargs)
        ws = self.ws_factory(url)
        frames = self.ws_frames(len(self.sockets)) if self.ws_frames else self.greeting
        ws.feed(*frames)
        self.sockets.append(ws)
        return ws


# ---------------------------------------------------------------------------
# Wire encoders
# ---------------------------------------------------------------------------


def polling_body(*packets: str) -> bytes:
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


# ---------------------------------------------------------------------------
# REST and runtime doubles
# ---------------------------------------------------------------------------


def token_response(token: str = "tok") -> MockResponse:
    """Return a successful OAuth token response."""
    return MockResponse(
        200,
        {"access_token": token, "expires_in": 3600},
        headers={"Content-Type": "application/json"},
    )


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
