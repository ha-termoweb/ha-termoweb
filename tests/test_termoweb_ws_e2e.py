"""End-to-end TermoWeb Socket.IO 0.9 session against a fake aiohttp transport.

The production ``TermoWebWSClient`` is built through the real backend factory
(``create_backend`` -> ``TermoWebBackend.create_ws_client``) while the
``socketio`` package is replaced by a poisoned module that records any use.
The session is then driven through handshake, namespace join, snapshot,
update push, heartbeat, server disconnect, reconnect and stop.  This proves the
production client never relies on python-socketio or on the socketio-only
members of its former base class.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import importlib
import json
import sys
import types
from types import SimpleNamespace
from typing import Any

import aiohttp
from conftest import DummyREST, build_device_metadata_payload, build_entry_runtime
import pytest

from homeassistant.core import HomeAssistant

DEV_ID = "0123456789abcdef"  # synthetic gateway id
ENTRY_ID = "entry-e2e"
NAMESPACE = "/api/v2/socket_io"

# Members that only existed to drive ``socketio.AsyncClient``.
SOCKETIO_ONLY_MEMBERS = (
    "_sio",
    "_connect_once",
    "_wait_for_events",
    "_build_engineio_target",
    "_handle_connection_lost",
    "_wrap_background_task",
    "_on_connect",
    "_on_disconnect",
    "_on_namespace_connect",
    "_on_namespace_disconnect",
    "_on_dev_handshake",
    "_on_dev_data",
    "_on_update",
    "_on_reconnect",
    "_on_reconnect_failed",
    "_on_connect_error",
    "_on_error",
    "_register_debug_catch_all",
    "_subscribe_heater_samples",
)


class _PoisonedModule(types.ModuleType):
    """Module stand-in that records every non-dunder attribute access."""

    def __init__(self, name: str, touched: list[str]) -> None:
        super().__init__(name)
        self._touched = touched

    def __getattr__(self, attr: str) -> Any:
        if attr.startswith("__"):
            raise AttributeError(attr)
        self._touched.append(f"{self.__name__}.{attr}")
        raise AssertionError(f"{self.__name__}.{attr} used by production code")


class FakeWS:
    """Scriptable aiohttp websocket: frames are fed by the test."""

    close_code = None
    close_reason = None

    def __init__(self, url: str, msg_type: Any) -> None:
        self.url = url
        self.sent: list[str] = []
        self.closed = False
        self._msg_type = msg_type
        self._queue: asyncio.Queue[Any] = asyncio.Queue()

    def feed(self, text: str) -> None:
        """Queue a server TEXT frame."""
        self._queue.put_nowait(SimpleNamespace(type=self._msg_type.TEXT, data=text))

    async def send_str(self, data: str) -> None:
        """Record a client frame."""
        if self.closed:
            raise RuntimeError("send on closed websocket")
        self.sent.append(data)

    async def close(self, **_kwargs: Any) -> bool:
        """Close the socket and end iteration."""
        self.closed = True
        self._queue.put_nowait(None)
        return True

    def exception(self) -> None:
        """Return no transport error."""
        return None

    def __aiter__(self) -> FakeWS:
        return self

    async def __anext__(self) -> Any:
        msg = await self._queue.get()
        if msg is None:
            raise StopAsyncIteration
        return msg


class _HandshakeResponse:
    """Successful Socket.IO 0.9 handshake response."""

    status = 200

    def __init__(self, sid: str) -> None:
        self._body = f"{sid}:60:60:websocket,xhr-polling"

    async def text(self) -> str:
        return self._body

    async def __aenter__(self) -> _HandshakeResponse:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None


class FakeSession:
    """aiohttp.ClientSession stand-in serving handshakes and websockets."""

    closed = False

    def __init__(self, msg_type: Any) -> None:
        self._msg_type = msg_type
        self.handshakes: list[str] = []
        self.sockets: list[FakeWS] = []

    def get(self, url: str, **_kwargs: Any) -> _HandshakeResponse:
        self.handshakes.append(url)
        return _HandshakeResponse(f"sid{len(self.handshakes)}")

    async def ws_connect(self, url: str, **_kwargs: Any) -> FakeWS:
        ws = FakeWS(url, self._msg_type)
        self.sockets.append(ws)
        return ws


def _event(name: str, payload: Any) -> str:
    """Return a Socket.IO 0.9 event frame for the TermoWeb namespace."""
    body = json.dumps({"name": name, "args": [payload]}, separators=(",", ":"))
    return f"5::{NAMESPACE}:{body}"


async def _until(predicate: Callable[[], bool], real_sleep: Any) -> None:
    """Yield to the loop until ``predicate`` holds (bounded)."""
    async with asyncio.timeout(5):
        while not predicate():
            await real_sleep(0.001)


@pytest.mark.asyncio
async def test_termoweb_socketio09_session_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production client completes a full session without python-socketio."""

    # --- Poison ``socketio`` and import the backend modules afresh --------
    socketio_touched: list[str] = []
    monkeypatch.setitem(
        sys.modules, "socketio", _PoisonedModule("socketio", socketio_touched)
    )
    backend_pkg = importlib.import_module("custom_components.termoweb.backend")
    for name in ("termoweb", "termoweb_ws"):
        full = f"custom_components.termoweb.backend.{name}"
        if full in sys.modules:
            monkeypatch.delitem(sys.modules, full)
        if hasattr(backend_pkg, name):
            monkeypatch.setattr(backend_pkg, name, getattr(backend_pkg, name))

    from custom_components.termoweb.backend import (  # noqa: PLC0415
        create_backend,
        ws_client as ws_client_module,
    )
    from custom_components.termoweb.const import signal_ws_status  # noqa: PLC0415
    from custom_components.termoweb.coordinator import (  # noqa: PLC0415
        StateCoordinator,
    )
    from custom_components.termoweb.domain import state_to_dict  # noqa: PLC0415
    from custom_components.termoweb.inventory import (  # noqa: PLC0415
        Inventory,
        build_node_inventory,
    )

    # --- Sleep control: block long-period loops, record the rest ---------
    real_sleep = asyncio.sleep
    sleeps: list[float] = []

    async def _sleep(delay: float, *args: Any, **kwargs: Any) -> Any:
        if delay >= 20:  # heartbeat (27 s), RTC keep-alive (30 s), idle (60 s)
            await asyncio.get_running_loop().create_future()
        if delay > 0:
            sleeps.append(delay)
        return await real_sleep(0, *args, **kwargs)

    monkeypatch.setattr(asyncio, "sleep", _sleep)

    # --- Build the production object graph ------------------------------
    hass = HomeAssistant()
    hass.loop = asyncio.get_running_loop()
    inventory = Inventory(
        DEV_ID,
        build_node_inventory({"nodes": [{"type": "htr", "addr": "1", "name": "H"}]}),
    )
    coordinator = StateCoordinator(
        hass,
        client=SimpleNamespace(),
        base_interval=30,
        dev_id=DEV_ID,
        device=build_device_metadata_payload(DEV_ID),
        inventory=inventory,
        entry_id=ENTRY_ID,
    )
    build_entry_runtime(
        hass=hass,
        entry_id=ENTRY_ID,
        dev_id=DEV_ID,
        inventory=inventory,
        coordinator=coordinator,
    )

    session = FakeSession(aiohttp.WSMsgType)
    rest = DummyREST()
    rest._session = session
    rest.get_rtc_time = lambda _dev_id: real_sleep(0)

    backend = create_backend(brand="termoweb", client=rest)
    client = backend.create_ws_client(
        hass, ENTRY_ID, DEV_ID, coordinator, inventory=inventory
    )
    ws_module = sys.modules["custom_components.termoweb.backend.termoweb_ws"]
    monkeypatch.setattr(ws_module.random, "uniform", lambda _a, _b: 1.0)

    assert type(client) is ws_module.TermoWebWSClient
    assert "_sio" not in vars(client)
    assert not any(
        cls.__module__.split(".")[0] in {"socketio", "engineio"}
        for cls in type(client).__mro__
    )

    # --- Guard every socketio-only member on the live instance -----------
    members_touched: list[str] = []

    def _guard(name: str) -> property:
        def _fget(_self: Any) -> Any:
            members_touched.append(name)
            raise AssertionError(f"socketio-only member {name} accessed")

        return property(_fget)

    guarded = type(
        "GuardedTermoWebWSClient",
        (type(client),),
        {name: _guard(name) for name in SOCKETIO_ONLY_MEMBERS},
    )
    client.__class__ = guarded

    statuses: list[str] = []

    def _dispatch(_hass: Any, signal: str, payload: dict[str, Any]) -> None:
        if signal == signal_ws_status(ENTRY_ID):
            statuses.append(payload["status"])

    monkeypatch.setattr(ws_client_module, "async_dispatcher_send", _dispatch)

    def _heater() -> dict[str, Any] | None:
        state = coordinator.domain_view.get_heater_state("htr", "1")
        return state_to_dict(state) if state is not None else None

    # --- Session 1: handshake -> join -> snapshot -> update -> heartbeat --
    task = client.start()
    await _until(lambda: len(session.sockets) == 1, real_sleep)
    ws1 = session.sockets[0]
    await _until(lambda: len(ws1.sent) >= 4, real_sleep)

    assert session.handshakes[0].startswith("https://api.termoweb/socket.io/1/?")
    assert f"dev_id={DEV_ID}" in session.handshakes[0]
    assert ws1.url.startswith("https://api.termoweb/socket.io/1/websocket/sid1?")
    assert ws1.sent[:4] == [
        f"1::{NAMESPACE}",
        f'5::{NAMESPACE}:{{"name":"dev_data","args":[]}}',
        f'5::{NAMESPACE}:{{"name":"subscribe","args":["/mgr/session"]}}',
        f'5::{NAMESPACE}:{{"name":"subscribe","args":["/htr/1/samples"]}}',
    ]
    await _until(lambda: "connected" in statuses, real_sleep)
    assert _heater() is None

    ws1.feed("1::")
    ws1.feed(f"1::{NAMESPACE}")
    ws1.feed(
        _event(
            "dev_data",
            {"nodes": {"htr": {"settings": {"1": {"mode": "auto", "stemp": "20.0"}}}}},
        )
    )
    await _until(lambda: _heater() is not None, real_sleep)
    assert _heater()["mode"] == "auto"
    assert _heater()["stemp"] == "20.0"
    await _until(lambda: "healthy" in statuses, real_sleep)

    ws1.feed(_event("update", {"path": "/htr/1/settings", "body": {"stemp": "22.5"}}))
    await _until(lambda: _heater()["stemp"] == "22.5", real_sleep)
    assert _heater()["mode"] == "auto"

    sent_before = len(ws1.sent)
    ws1.feed("2::")
    await _until(lambda: len(ws1.sent) > sent_before, real_sleep)
    assert ws1.sent[sent_before:] == ["2::"]
    assert client._ws_health_tracker().last_heartbeat_at is not None

    # --- Server disconnect -> backoff -> reconnect ------------------------
    ws1.feed("0::")
    await _until(lambda: len(session.sockets) == 2, real_sleep)
    ws2 = session.sockets[1]
    await _until(lambda: len(ws2.sent) >= 4, real_sleep)
    assert ws1.closed
    assert len(session.handshakes) == 2
    assert ws2.url.startswith("https://api.termoweb/socket.io/1/websocket/sid2?")
    assert ws2.sent[0] == f"1::{NAMESPACE}"
    assert 5.0 in sleeps  # first backoff step after a session with payloads
    await _until(lambda: statuses.count("connected") >= 2, real_sleep)

    # --- Clean shutdown ----------------------------------------------------
    await client.stop()

    assert task.done()
    if not task.cancelled():
        assert task.exception() is None
    assert ws2.closed
    assert client._task is None
    assert client._ws is None
    assert client._hb_task is None
    assert client._rtc_keepalive_task is None
    assert client._idle_monitor_task is None
    pending = [
        t
        for t in asyncio.all_tasks()
        if t is not asyncio.current_task() and not t.done()
    ]
    assert pending == []

    collapsed = [s for i, s in enumerate(statuses) if i == 0 or statuses[i - 1] != s]
    assert collapsed[0] == "starting"
    assert collapsed[-1] == "stopped"
    order = ["starting", "connected", "healthy", "disconnected", "connected"]
    it = iter(collapsed)
    assert all(step in it for step in order), collapsed

    assert members_touched == []
    assert socketio_touched == []
