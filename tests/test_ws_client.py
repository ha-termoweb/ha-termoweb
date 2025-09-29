from __future__ import annotations

import asyncio
import codecs
import copy
import json
import logging
import types
from typing import Any, Iterable
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from conftest import _install_stubs

_install_stubs()

import custom_components.termoweb.nodes as nodes
import custom_components.termoweb.utils as utils
import custom_components.termoweb.ws_client as ws_core


def _load_ws_client(
    *,
    get_responses: Iterable[Any] | None = None,
    ws_connect_results: Iterable[Any] | None = None,
):
    _install_stubs()
    module = ws_core
    testing = module.aiohttp.testing
    defaults = getattr(testing, "_defaults", None)
    if defaults is not None:
        defaults.get_responses = list(get_responses or [])
        defaults.ws_connect_results = list(ws_connect_results or [])
    return module


def test_handshake_error_attributes_and_start_reuses_task() -> None:
    module = _load_ws_client()

    err = module.HandshakeError(418, "https://example", "short body")
    assert err.status == 418
    assert err.url == "https://example"
    assert err.body_snippet == "short body"

    created_tasks: list[str | None] = []

    class DummyTask:
        def __init__(self, coro: Any) -> None:
            self._coro = coro
            self.cancelled = False

        def done(self) -> bool:
            return False

        def cancel(self) -> None:
            self.cancelled = True
            try:
                self._coro.close()
            except RuntimeError:
                pass

    def fake_create_task(coro: Any, *, name: str | None = None) -> DummyTask:
        created_tasks.append(name)
        # Avoid "coroutine was never awaited" warnings.
        try:
            coro.close()
        except RuntimeError:
            pass
        return DummyTask(coro)

    hass = types.SimpleNamespace(
        loop=types.SimpleNamespace(create_task=fake_create_task)
    )
    coordinator = types.SimpleNamespace()
    api = types.SimpleNamespace(_session=None)

    client = module.WebSocket09Client(
        hass,
        entry_id="entry",
        dev_id="dev",
        api_client=api,
        coordinator=coordinator,
    )

    task1 = client.start()
    task2 = client.start()

    assert task1 is task2
    assert client._task is task1
    assert created_tasks == [f"{module.DOMAIN}-ws-dev"]


def test_stop_handles_exceptions_and_updates_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        module = _load_ws_client()

        hass = types.SimpleNamespace(
            loop=asyncio.get_event_loop(),
            data={module.DOMAIN: {"entry": {}}},
        )

        def fake_prepare(entry_data: Any, *, default_name_simple: Any) -> tuple[Any, Any, Any, Any]:
            return ([], {}, {"htr": [], "acm": []}, lambda *_args: "")

        monkeypatch.setattr(module, "prepare_heater_platform_data", fake_prepare)

        coordinator = types.SimpleNamespace()
        api = types.SimpleNamespace(_session=None)
        client = module.WebSocket09Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=api,
            coordinator=coordinator,
        )

        update_calls: list[str] = []
        client._update_status = MagicMock(
            side_effect=lambda status: update_calls.append(status)
        )

        class FakeHBTask:
            def __init__(self) -> None:
                self.cancelled = False

            def cancel(self) -> None:
                self.cancelled = True

            def done(self) -> bool:
                return False

            def __await__(self):
                async def _inner() -> None:
                    if self.cancelled:
                        raise asyncio.CancelledError()

                return _inner().__await__()

        class ExplodingWS:
            def __init__(self) -> None:
                self.calls = 0

            async def close(self, *args: Any, **kwargs: Any) -> None:
                self.calls += 1
                raise RuntimeError("close failed")

        client._hb_task = FakeHBTask()
        client._idle_restart_task = FakeHBTask()
        ws = ExplodingWS()
        client._ws = ws
        client._task = asyncio.create_task(asyncio.sleep(0.1))

        await client.stop()

        assert update_calls[-1] == "stopped"
        assert client._hb_task is None
        assert client._idle_restart_task is None
        assert client._ws is None
        assert client._task is None
        assert ws.calls == 1

    asyncio.run(_run())


def test_apply_nodes_payload_logs_without_changes(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Incremental updates without address changes should emit a debug log."""

    module = _load_ws_client()
    caplog.set_level(logging.DEBUG)

    hass = types.SimpleNamespace(
        loop=types.SimpleNamespace(create_task=lambda coro, name=None: None)
    )
    client = module.WebSocket09Client(
        hass,
        entry_id="entry",
        dev_id="dev",
        api_client=types.SimpleNamespace(_session=None),
        coordinator=types.SimpleNamespace(),
    )

    monkeypatch.setattr(
        module.WebSocket09Client,
        "_collect_update_addresses",
        staticmethod(lambda _nodes: []),
    )
    client._merge_nodes = MagicMock()
    client._build_nodes_snapshot = MagicMock(return_value={"nodes": {}})
    client._dispatch_nodes = MagicMock()
    client._mark_event = MagicMock()
    client._nodes_raw = {"htr": {}}

    payload = {"nodes": {"htr": {"settings": {}}}}
    client._apply_nodes_payload(payload, merge=True, event="update")

    assert any(
        "update event without address changes" in record.getMessage()
        for record in caplog.records
    )
    client._merge_nodes.assert_called_once()


def test_collect_update_addresses_ignores_non_mappings() -> None:
    """Collector should ignore non-dict sections while gathering addresses."""

    module = _load_ws_client()
    payload = {
        "htr": {
            "settings": {"1": {"mode": "auto"}},
            "energy": [],
            "other": None,
        },
        "bad": "invalid",
    }

    result = module.WebSocket09Client._collect_update_addresses(payload)

    assert result == [("htr", "1")]


def test_ws_url_uses_client_api_base() -> None:
    module = _load_ws_client()

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        hass = types.SimpleNamespace(
            loop=loop,
            data={module.DOMAIN: {"entry": {"ws_state": {}}}},
        )
        coordinator = types.SimpleNamespace()

        class TokenClient:
            api_base = "https://api-tevolve.example.com/base/"

            async def _authed_headers(self) -> dict[str, str]:
                return {"Authorization": "Bearer scoped-token"}

        client = module.WebSocket09Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=TokenClient(),
            coordinator=coordinator,
        )

        url = await client.ws_url()
        assert url == (
            "https://api-tevolve.example.com/base/api/v2/socket_io?token=scoped-token"
        )

    asyncio.run(_run())


def test_engineio_handshake_decodes_non_utf8_bytes() -> None:
    """Handshake bodies with invalid UTF-8 bytes should still parse."""

    module = _load_ws_client()
    raw = (
        b"96:0\xff{"
        b'"sid":"abc123","pingInterval":31000,"pingTimeout":62000}'
    )

    decoded = module.WebSocketClient._decode_engineio_handshake(
        raw, "windows-1252"
    )
    handshake = module.WebSocketClient._parse_engineio_handshake(decoded)

    assert handshake.sid == "abc123"
    assert handshake.ping_interval == pytest.approx(31.0)
    assert handshake.ping_timeout == pytest.approx(62.0)


def test_engineio_handshake_decode_fallback_ignore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback decoding should ignore undecodable bytes when codecs are missing."""

    module = _load_ws_client()
    original_decode = codecs.decode

    def fake_decode(data: bytes, encoding: str = "utf-8", errors: str = "strict") -> str:
        normalized = encoding.replace("_", "-").lower()
        if normalized in {"iso-8859-1", "iso8859-1", "latin-1", "latin1"}:
            raise LookupError
        return original_decode(data, encoding, errors)

    monkeypatch.setattr(codecs, "decode", fake_decode)

    data = b"\xff\xfeinvalid"
    decoded = module.WebSocketClient._decode_engineio_handshake(data, None)

    assert decoded == "invalid"


def test_runner_dispatches_engineio() -> None:
    module = _load_ws_client()

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        hass = types.SimpleNamespace(
            loop=loop, data={module.DOMAIN: {"entry": {"ws_state": {}}}}
        )

        class FakeClient:
            api_base = "https://api-tevolve.termoweb.net/"

            async def _authed_headers(self) -> dict[str, str]:
                return {"Authorization": "Bearer tok"}

        client = module.WebSocket09Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=FakeClient(),
            coordinator=types.SimpleNamespace(),
            protocol="engineio2",
        )

        calls: list[str] = []

        async def fake_engineio(self: Any) -> None:
            calls.append("engineio")

        async def fake_socketio(self: Any) -> None:
            calls.append("socketio")

        client._run_engineio_v2 = types.MethodType(fake_engineio, client)
        client._run_socketio_09 = types.MethodType(fake_socketio, client)
        client._protocol_hint = "engineio2"

        await client._runner()

        assert calls == ["engineio"]

    asyncio.run(_run())


def test_run_engineio_v2_handles_flow_and_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_ws_client()

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        hass = types.SimpleNamespace(
            loop=loop, data={module.DOMAIN: {"entry": {"ws_state": {}}}}
        )
        coordinator = types.SimpleNamespace()
        session = module.aiohttp.testing.FakeClientSession()

        class FakeClient:
            api_base = "https://api-tevolve.termoweb.net/"

            async def _authed_headers(self) -> dict[str, str]:
                return {"Authorization": "Bearer token"}

        client = module.WebSocket09Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=FakeClient(),
            coordinator=coordinator,
            session=session,
            protocol="engineio2",
        )

        statuses: list[str] = []
        client._update_status = MagicMock(side_effect=lambda status: statuses.append(status))

        handshake_effects: list[Any] = [
            module.HandshakeError(503, "https://api", "fail"),
            module.EngineIOHandshake(sid="sid-a", ping_interval=10.0, ping_timeout=20.0),
            module.EngineIOHandshake(sid="sid-b", ping_interval=12.0, ping_timeout=24.0),
        ]

        async def fake_handshake(self: Any) -> module.EngineIOHandshake:
            effect = handshake_effects.pop(0)
            if isinstance(effect, Exception):
                raise effect
            return effect

        async def fake_connect(self: Any, sid: str) -> None:
            self._engineio_ws = module.aiohttp.testing.FakeWebSocket()
            await self._engineio_send("40")

        read_effects: list[Any] = [RuntimeError("boom"), asyncio.CancelledError()]

        async def fake_read_loop(self: Any) -> None:
            effect = read_effects.pop(0)
            if isinstance(effect, asyncio.CancelledError):
                raise effect
            if isinstance(effect, Exception):
                raise effect
            self._closing = True
            return None

        async def fake_ping_loop(self: Any) -> None:
            await asyncio.sleep(0)

        send_calls: list[str] = []

        async def fake_send(self: Any, data: str) -> None:
            send_calls.append(data)
            if self._engineio_ws:
                await self._engineio_ws.send_str(data)

        client._engineio_handshake = types.MethodType(fake_handshake, client)
        client._engineio_connect = types.MethodType(fake_connect, client)
        client._engineio_read_loop = types.MethodType(fake_read_loop, client)
        client._engineio_ping_loop = types.MethodType(fake_ping_loop, client)
        client._engineio_send = types.MethodType(fake_send, client)

        sleep_delays: list[float] = []

        async def fake_sleep(delay: float) -> None:
            sleep_delays.append(delay)

        monkeypatch.setattr(module.asyncio, "sleep", fake_sleep)
        monkeypatch.setattr(ws_core.asyncio, "sleep", fake_sleep)
        monkeypatch.setattr(module.random, "uniform", lambda a, b: 1.0)

        await client._run_engineio_v2()

        assert statuses[0] == "connecting"
        assert "connected" in statuses
        assert statuses[-1] == "disconnected"
        assert client._engineio_sid == "sid-b"
        assert client._ping_task is None
        assert client._engineio_ws is None
        assert sleep_delays
        assert send_calls

    asyncio.run(_run())


def test_run_engineio_v2_waits_without_session() -> None:
    module = _load_ws_client()

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        hass = types.SimpleNamespace(
            loop=loop, data={module.DOMAIN: {"entry": {"ws_state": {}}}}
        )
        coordinator = types.SimpleNamespace()

        class FakeClient:
            api_base = "https://api-tevolve.termoweb.net/"

            async def _authed_headers(self) -> dict[str, str]:
                return {"Authorization": "Bearer token"}

        client = module.WebSocket09Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=FakeClient(),
            coordinator=coordinator,
            session=None,
            protocol="engineio2",
        )

        client._stop_event.set()
        await client._run_engineio_v2()

    asyncio.run(_run())


def test_engineio_handshake_parsing_and_errors() -> None:
    module = _load_ws_client()

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        hass = types.SimpleNamespace(
            loop=loop, data={module.DOMAIN: {"entry": {"ws_state": {}}}}
        )
        coordinator = types.SimpleNamespace()

        class FakeClient:
            api_base = "https://api-tevolve.termoweb.net/"

            async def _authed_headers(self) -> dict[str, str]:
                return {"Authorization": "Bearer token"}

        session = module.aiohttp.testing.FakeClientSession(
            get_responses=[
                (
                    200,
                    '0{"sid":"abc","pingInterval":5000,"pingTimeout":"15000"}',
                )
            ]
        )

        client = module.WebSocket09Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=FakeClient(),
            coordinator=coordinator,
            session=session,
            protocol="engineio2",
        )

        handshake = await client._engineio_handshake()
        assert handshake.sid == "abc"
        assert handshake.ping_interval == 5.0
        assert handshake.ping_timeout == 15.0

        client._session = module.aiohttp.testing.FakeClientSession(
            get_responses=[(500, "oops")]
        )
        with pytest.raises(module.HandshakeError) as err:
            await client._engineio_handshake()
        assert err.value.body_snippet == "oops"

        class RaisingContext:
            async def __aenter__(self) -> None:
                raise module.aiohttp.ClientError("boom")

            async def __aexit__(self, exc_type, exc, tb) -> bool:
                return False

        class RaisingSession:
            def get(self, *args: Any, **kwargs: Any) -> RaisingContext:
                return RaisingContext()

        client._session = RaisingSession()
        with pytest.raises(module.HandshakeError) as err2:
            await client._engineio_handshake()
        assert err2.value.status == -1

        parsed = module.WebSocketClient._parse_engineio_handshake(
            '0{"sid":"sid-x","pingInterval":"bad","pingTimeout":null}'
        )
        assert parsed.ping_interval == 25.0
        assert parsed.ping_timeout == 60.0

        with pytest.raises(RuntimeError):
            module.WebSocketClient._parse_engineio_handshake("no-json")
        with pytest.raises(RuntimeError):
            module.WebSocketClient._parse_engineio_handshake("0{}")
        with pytest.raises(RuntimeError):
            module.WebSocketClient._parse_engineio_handshake(
                '0{"sid":"abc","pingInterval":bad}'
            )

    asyncio.run(_run())


def test_engineio_connect_and_send() -> None:
    module = _load_ws_client()

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        hass = types.SimpleNamespace(
            loop=loop, data={module.DOMAIN: {"entry": {"ws_state": {}}}}
        )
        coordinator = types.SimpleNamespace()

        class FakeClient:
            api_base = "https://api-tevolve.termoweb.net/api/"

            async def _authed_headers(self) -> dict[str, str]:
                return {"Authorization": "Bearer abc"}

        session = module.aiohttp.testing.FakeClientSession(
            ws_connect_results=[module.aiohttp.testing.FakeWebSocket()]
        )

        client = module.WebSocket09Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=FakeClient(),
            coordinator=coordinator,
            session=session,
            protocol="engineio2",
        )

        await client._engineio_connect("sid-123")
        assert client._engineio_ws is not None
        assert session.ws_connect_calls
        call = session.ws_connect_calls[0]
        assert "sid=sid-123" in call["url"]
        assert "token=abc" in call["url"]
        assert call["kwargs"]["protocols"] == ("websocket",)
        assert client._engineio_ws.sent[0] == "40"

        await client._engineio_send("2")
        assert client._engineio_ws.sent[-1] == "2"

        client._engineio_ws = None
        await client._engineio_send("ignored")

    asyncio.run(_run())


def test_engineio_ping_loop_handles_cancel_and_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_ws_client()

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        hass = types.SimpleNamespace(
            loop=loop, data={module.DOMAIN: {"entry": {"ws_state": {}}}}
        )
        coordinator = types.SimpleNamespace()

        class FakeClient:
            api_base = "https://api-tevolve.termoweb.net/"

            async def _authed_headers(self) -> dict[str, str]:
                return {"Authorization": "Bearer token"}

        client = module.WebSocket09Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=FakeClient(),
            coordinator=coordinator,
            protocol="engineio2",
        )

        client._engineio_ping_interval = 1.0

        send_calls: list[str] = []
        send_event = asyncio.Event()

        async def fake_send(data: str) -> None:
            send_calls.append(data)
            if len(send_calls) >= 3:
                send_event.set()

        client._engineio_send = AsyncMock(side_effect=fake_send)

        orig_sleep = asyncio.sleep
        sleep_calls: list[float] = []

        async def fast_sleep(delay: float) -> None:
            sleep_calls.append(delay)
            await orig_sleep(0)

        monkeypatch.setattr(module.asyncio, "sleep", fast_sleep)
        monkeypatch.setattr(ws_core.asyncio, "sleep", fast_sleep)

        task = asyncio.create_task(client._engineio_ping_loop())
        await asyncio.wait_for(send_event.wait(), timeout=0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert send_calls[:3] == ["2", "2", "2"]
        assert sleep_calls[:3] == [pytest.approx(1.0)] * 3

        client._engineio_send = AsyncMock(side_effect=RuntimeError("fail"))
        client._engineio_ping_interval = 0.0
        await client._engineio_ping_loop()

    asyncio.run(_run())


def test_subscribe_samples_uses_coordinator_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_ws_client()

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        hass = types.SimpleNamespace(loop=loop, data={})
        coordinator_nodes = {"nodes": [{"type": "htr", "addr": "1"}]}
        coordinator = types.SimpleNamespace(_nodes=coordinator_nodes)

        client = module.WebSocket09Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=types.SimpleNamespace(_session=None),
            coordinator=coordinator,
        )

        hass.data[module.DOMAIN][client.entry_id] = None

        prepare_calls: list[Any] = []

        def fake_prepare(entry_data: Any, *, default_name_simple: Any) -> tuple[Any, Any, Any, Any]:
            prepare_calls.append(entry_data)
            assert entry_data.get("nodes") is coordinator_nodes
            inventory = [types.SimpleNamespace(type="htr", addr="1")]
            return (inventory, {}, {"htr": ["1"], "acm": []}, lambda *_args: "")

        monkeypatch.setattr(module, "prepare_heater_platform_data", fake_prepare)

        client._send_text = AsyncMock(return_value=None)

        await client._subscribe_htr_samples()

        assert prepare_calls
        assert client._send_text.await_count >= 1

    asyncio.run(_run())


def test_engineio_read_loop_processes_messages() -> None:
    module = _load_ws_client()

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        hass = types.SimpleNamespace(
            loop=loop, data={module.DOMAIN: {"entry": {"ws_state": {}}}}
        )
        coordinator = types.SimpleNamespace()

        client = module.WebSocket09Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=types.SimpleNamespace(_session=None),
            coordinator=coordinator,
            protocol="engineio2",
        )

        client._engineio_send = AsyncMock(return_value=None)
        client._mark_event = MagicMock()
        client._record_heartbeat = MagicMock()
        client._on_frame = MagicMock()

        ws = module.aiohttp.testing.FakeWebSocket(
            messages=[
                {"type": 99, "data": None},
                {
                    "type": module.aiohttp.WSMsgType.TEXT,
                    "data": "3",
                },
                {
                    "type": module.aiohttp.WSMsgType.TEXT,
                    "data": "2",
                },
                {
                    "type": module.aiohttp.WSMsgType.BINARY,
                    "data": b'42{"event":"update","data":{}}',
                },
                {
                    "type": module.aiohttp.WSMsgType.TEXT,
                    "data": "41",
                    "extra": "bye",
                },
            ]
        )

        client._engineio_ws = ws

        with pytest.raises(RuntimeError, match="disconnect"):
            await client._engineio_read_loop()

        assert client._engineio_send.call_args_list[-1].args[0] == "3"
        assert client._record_heartbeat.called
        client._mark_event.assert_not_called()
        assert client._on_frame.called

    asyncio.run(_run())


def test_engineio_read_loop_handles_closed_and_error_states() -> None:
    module = _load_ws_client()

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        hass = types.SimpleNamespace(
            loop=loop, data={module.DOMAIN: {"entry": {"ws_state": {}}}}
        )
        coordinator = types.SimpleNamespace()

        client = module.WebSocket09Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=types.SimpleNamespace(_session=None),
            coordinator=coordinator,
            protocol="engineio2",
        )

        ws_closed = module.aiohttp.testing.FakeWebSocket(
            messages=[
                {
                    "type": module.aiohttp.WSMsgType.CLOSED,
                    "data": None,
                    "extra": "done",
                }
            ]
        )
        closed_exc = RuntimeError("closed")
        ws_closed.set_exception(closed_exc)
        client._engineio_ws = ws_closed
        with pytest.raises(RuntimeError) as err:
            await client._engineio_read_loop()
        assert err.value is closed_exc

        ws_closed2 = module.aiohttp.testing.FakeWebSocket(
            messages=[
                {
                    "type": module.aiohttp.WSMsgType.CLOSED,
                    "data": None,
                    "extra": "bye",
                }
            ]
        )
        client._engineio_ws = ws_closed2
        with pytest.raises(RuntimeError, match="engine.io websocket closed"):
            await client._engineio_read_loop()

        ws_error = module.aiohttp.testing.FakeWebSocket(
            messages=[{"type": module.aiohttp.WSMsgType.ERROR, "data": None}]
        )
        ws_error.set_exception(None)
        client._engineio_ws = ws_error
        with pytest.raises(RuntimeError, match="websocket error"):
            await client._engineio_read_loop()

    asyncio.run(_run())


def test_engineio_read_loop_waits_without_socket() -> None:
    module = _load_ws_client()

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        hass = types.SimpleNamespace(
            loop=loop, data={module.DOMAIN: {"entry": {"ws_state": {}}}}
        )
        coordinator = types.SimpleNamespace()

        client = module.WebSocket09Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=types.SimpleNamespace(_session=None),
            coordinator=coordinator,
            protocol="engineio2",
        )

        client._engineio_ws = None
        client._stop_event.set()
        await client._engineio_read_loop()

    asyncio.run(_run())


def test_engineio_read_loop_raises_error_exception() -> None:
    module = _load_ws_client()

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        hass = types.SimpleNamespace(
            loop=loop, data={module.DOMAIN: {"entry": {"ws_state": {}}}}
        )
        coordinator = types.SimpleNamespace()

        client = module.WebSocket09Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=types.SimpleNamespace(_session=None),
            coordinator=coordinator,
            protocol="engineio2",
        )

        ws = module.aiohttp.testing.FakeWebSocket(
            messages=[{"type": module.aiohttp.WSMsgType.ERROR, "data": None}]
        )
        ws.set_exception(RuntimeError("boom"))
        client._engineio_ws = ws

        with pytest.raises(RuntimeError, match="boom"):
            await client._engineio_read_loop()

    asyncio.run(_run())


@pytest.mark.parametrize(
    "context",
    ["websocket", "engine.io websocket"],
)
def test_ws_payload_stream_raises_close_for_both_protocols(context: str) -> None:
    module = _load_ws_client()

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        hass = types.SimpleNamespace(
            loop=loop, data={module.DOMAIN: {"entry": {"ws_state": {}}}}
        )
        coordinator = types.SimpleNamespace()
        client = module.WebSocket09Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=types.SimpleNamespace(_session=None),
            coordinator=coordinator,
        )

        class DummyWS:
            def __init__(self) -> None:
                self.close_code = 4000

            async def receive(self) -> Any:
                return types.SimpleNamespace(
                    type=module.aiohttp.WSMsgType.CLOSE,
                    data=None,
                    extra="gone",
                )

            def exception(self) -> None:
                return None

        ws = DummyWS()
        with pytest.raises(
            RuntimeError, match=f"{context} closed: code=4000 reason=gone"
        ):
            await anext(client._ws_payload_stream(ws, context=context))

    asyncio.run(_run())


def test_handle_payload_helpers_update_nodes() -> None:
    module = _load_ws_client()

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        record: dict[str, Any] = {"ws_state": {}, "nodes": {}, "node_inventory": []}
        hass = types.SimpleNamespace(loop=loop, data={module.DOMAIN: {"entry": record}})

        class FakeEnergyCoordinator:
            def __init__(self) -> None:
                self.updates: list[dict[str, list[str]]] = []

            def update_addresses(self, addrs: dict[str, list[str]]) -> None:
                self.updates.append({k: list(v) for k, v in addrs.items()})

        record["energy_coordinator"] = FakeEnergyCoordinator()

        class RecordingCoordinator:
            def __init__(self) -> None:
                self.calls: list[tuple[Any, Any]] = []
                self.data: dict[str, Any] = {}

            def update_nodes(self, nodes: Any, inventory: Any) -> None:
                self.calls.append((nodes, inventory))

        coordinator = RecordingCoordinator()

        client = module.WebSocket09Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=types.SimpleNamespace(_session=None),
            coordinator=coordinator,
        )

        assert client._extract_nodes("invalid") is None
        client._handle_handshake("bad-payload")
        client._handle_handshake({"devs": [{"id": "dev"}]})
        assert client._handshake_payload is not None

        client._handle_dev_data({})
        client._handle_update({})

        initial_payload = {"nodes": {"htr": {"status": {"01": {"temp": 20}}}}}
        client._handle_dev_data(initial_payload)

        assert client._nodes_raw == initial_payload["nodes"]
        assert client._nodes_raw is not initial_payload["nodes"]
        assert client._nodes["nodes"]["htr"]["status"]["01"]["temp"] == 20

        incremental = {"nodes": {"htr": {"status": {"02": {"temp": 22}}}}}
        client._handle_update(incremental)

        assert client._nodes_raw["htr"]["status"]["02"]["temp"] == 22
        assert client._nodes["nodes"]["htr"]["status"] == {
            "01": {"temp": 20},
            "02": {"temp": 22},
        }

        await asyncio.sleep(0)

        assert coordinator.calls
        assert "nodes" in record
        assert "node_inventory" in record

    asyncio.run(_run())


def test_stop_cleans_engineio_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        module = _load_ws_client()

        hass = types.SimpleNamespace(
            loop=asyncio.get_event_loop(), data={module.DOMAIN: {"entry": {}}}
        )

        def fake_prepare(entry_data: Any, *, default_name_simple: Any) -> tuple[Any, Any, Any, Any]:
            return ([], {}, {"htr": [], "acm": []}, lambda *_args: "")

        monkeypatch.setattr(module, "prepare_heater_platform_data", fake_prepare)

        coordinator = types.SimpleNamespace()
        api = types.SimpleNamespace(_session=None)
        client = module.WebSocket09Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=api,
            coordinator=coordinator,
        )

        updates: list[str] = []
        client._update_status = MagicMock(side_effect=lambda status: updates.append(status))

        class FakeTask:
            def __init__(self) -> None:
                self.cancelled = False

            def cancel(self) -> None:
                self.cancelled = True

            def done(self) -> bool:
                return False

            def __await__(self):  # type: ignore[override]
                async def _inner() -> None:
                    return None

                return _inner().__await__()

        class DummyWS:
            def __init__(self) -> None:
                self.closed = 0

            async def close(self, *args: Any, **kwargs: Any) -> None:
                self.closed += 1

        ping_task = FakeTask()
        client._ping_task = ping_task
        engine_ws = DummyWS()
        client._engineio_ws = engine_ws
        client._task = asyncio.create_task(asyncio.sleep(0))

        await asyncio.sleep(0)
        await client.stop()

        assert ping_task.cancelled
        assert client._ping_task is None
        assert engine_ws.closed == 1
        assert client._engineio_ws is None
        assert client._task is None
        assert updates[-1] == "stopped"

    asyncio.run(_run())


def test_runner_backoff_after_handshake_errors(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    module = _load_ws_client()
    caplog.set_level(logging.INFO, logger=module.__name__)

    async def _run() -> None:
        hass = types.SimpleNamespace(
            loop=asyncio.get_event_loop(),
            data={module.DOMAIN: {"entry": {}}},
        )
        coordinator = types.SimpleNamespace()
        api = types.SimpleNamespace(
            _session=types.SimpleNamespace(),
            _authed_headers=AsyncMock(
                return_value={"Authorization": "Bearer cached"}
            ),
            _ensure_token=AsyncMock(),
        )

        client = module.WebSocket09Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=api,
            coordinator=coordinator,
            handshake_fail_threshold=3,
        )

        statuses: list[str] = []
        fail_counts: list[int] = []
        fail_starts: list[float] = []
        last_record: tuple[int, float] | None = None

        def record_status(status: str) -> None:
            nonlocal last_record
            statuses.append(status)
            if status == "disconnected":
                record = (client._hs_fail_count, client._hs_fail_start)
                if record != last_record:
                    fail_counts.append(record[0])
                    fail_starts.append(record[1])
                    last_record = record

        client._update_status = MagicMock(side_effect=record_status)

        time_values = iter([100.0, 160.0])

        def fake_time() -> float:
            try:
                return next(time_values)
            except StopIteration:
                return 160.0

        monkeypatch.setattr(module.time, "time", fake_time)
        monkeypatch.setattr(ws_core.time, "time", fake_time)

        sleep_calls: list[float] = []

        async def fake_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        monkeypatch.setattr(module.asyncio, "sleep", fake_sleep)
        monkeypatch.setattr(ws_core.asyncio, "sleep", fake_sleep)

        jitter_args: list[tuple[float, float]] = []

        def fake_uniform(a: float, b: float) -> float:
            jitter_args.append((a, b))
            return 1.1

        monkeypatch.setattr(module.random, "uniform", fake_uniform)
        monkeypatch.setattr(ws_core.random, "uniform", fake_uniform)

        attempt = 0

        async def failing_handshake() -> Any:
            nonlocal attempt
            attempt += 1
            if attempt <= 3:
                raise module.HandshakeError(500, f"url-{attempt}", f"body-{attempt}")
            raise asyncio.CancelledError()

        client._handshake = failing_handshake  # type: ignore[assignment]

        await client._runner()

        assert attempt == 4
        assert statuses == [
            "starting",
            "disconnected",
            "disconnected",
            "disconnected",
            "disconnected",
            "stopped",
        ]
        assert fail_counts == [1, 2, 0]
        assert fail_starts == [100.0, 100.0, 0.0]
        assert sleep_calls == [
            pytest.approx(5 * 1.1),
            pytest.approx(10 * 1.1),
            pytest.approx(30 * 1.1),
        ]
        assert jitter_args == [(0.8, 1.2)] * 3
        assert client._backoff_idx == 3
        assert client._hs_fail_count == 0
        assert client._hs_fail_start == 0.0

        warning_messages = [
            rec.message for rec in caplog.records if rec.levelno == logging.WARNING
        ]
        assert any("handshake failed 3 times" in msg for msg in warning_messages)

    asyncio.run(_run())


def test_runner_handles_handshake_events_and_disconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        module = _load_ws_client()
        dispatcher = MagicMock()
        monkeypatch.setattr(module, "async_dispatcher_send", dispatcher)
        aiohttp = module.aiohttp

        orig_sleep = asyncio.sleep

        async def _text_frame(data: str) -> dict[str, Any]:
            await orig_sleep(0)
            return {"type": aiohttp.WSMsgType.TEXT, "data": data}

        event_payload = {
            "name": "data",
            "args": [
                [
                    {
                        "path": "/mgr/nodes",
                        "body": {
                            "nodes": [
                                {"addr": "01", "type": "htr"},
                                {"addr": "02", "type": "HTR"},
                            ]
                        },
                    },
                    {"path": "/htr/01/settings", "body": {"temp": 21}},
                    {"path": "/htr/02/settings", "body": {"temp": 22}},
                    {"path": "/htr/01/advanced_setup", "body": {"adv": True}},
                    {
                        "path": "/htr/02/samples",
                        "body": [{"ts": 1, "val": 5}],
                    },
                    {"path": "/misc", "body": {"foo": "bar"}},
                ]
            ],
        }
        event_str = f"5::{module.WS_NAMESPACE}:{json.dumps(event_payload, separators=(',', ':'))}"

        ws = aiohttp.testing.FakeWebSocket(
            messages=[
                _text_frame("2::"),
                _text_frame(event_str),
                _text_frame(f"5::{module.WS_NAMESPACE}:not-json"),
                _text_frame("0::"),
            ]
        )

        session = aiohttp.ClientSession(
            get_responses=[
                {"status": 401, "body": "unauthorized"},
                {"status": 200, "body": "abc123:25:60:websocket"},
            ],
            ws_connect_results=[ws],
        )

        loop = asyncio.get_event_loop()
        hass = types.SimpleNamespace(
            loop=loop,
            data={module.DOMAIN: {"entry": {}}},
        )

        class DummyCoordinator:
            def __init__(self) -> None:
                self.data = {
                    "dev": {
                        "dev_id": "dev",
                        "name": "Device dev",
                        "raw": {},
                        "connected": True,
                        "nodes": None,
                        "htr": {"addrs": [], "settings": {}},
                    }
                }

            def _addrs(self) -> list[str]:
                return ["01", "02"]

        coordinator = DummyCoordinator()

        api = types.SimpleNamespace(
            _session=session,
            _authed_headers=AsyncMock(
                side_effect=[
                    {"Authorization": "Bearer expired"},
                    {"Authorization": "Bearer refreshed"},
                    {"Authorization": "Bearer refreshed"},
                ]
            ),
            _ensure_token=AsyncMock(),
            _access_token="expired",
        )

        client = module.WebSocket09Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=api,
            coordinator=coordinator,
            session=session,
        )

        client._backoff_idx = 3

        now = 1_000.0

        def fake_time() -> float:
            nonlocal now
            now += 1.0
            return now

        monkeypatch.setattr(module.time, "time", fake_time)

        sleep_calls: list[float] = []

        async def fast_sleep(delay: float) -> None:
            sleep_calls.append(delay)
            await orig_sleep(0)

        monkeypatch.setattr(module.asyncio, "sleep", fast_sleep)
        monkeypatch.setattr(ws_core.asyncio, "sleep", fast_sleep)

        orig_read_loop = module.WebSocket09Client._read_loop

        async def read_wrapper(self: Any) -> None:
            try:
                await orig_read_loop(self)
            except RuntimeError:
                self._closing = True
                raise

        client._read_loop = types.MethodType(read_wrapper, client)

        await client._runner()

        assert api._ensure_token.await_count == 1
        assert api._authed_headers.await_count == 3
        assert len(session.get_calls) == 2
        assert "token=expired" in session.get_calls[0]["url"]
        assert "token=refreshed" in session.get_calls[1]["url"]
        assert client._backoff_idx == 0
        assert api._access_token is None
        assert client._hb_send_interval == pytest.approx(11.25)
        assert sleep_calls and sleep_calls[0] == pytest.approx(11.25)
        assert client._connected_since is not None

        assert len(session.ws_connect_calls) == 1
        ws_call = session.ws_connect_calls[0]
        assert "token=refreshed" in ws_call["url"]
        assert ws_call["kwargs"]["timeout"] == 15

        assert ws.sent[0] == f"1::{module.WS_NAMESPACE}"
        assert ws.sent[1] == '5::/api/v2/socket_io:{"name":"dev_data","args":[]}'
        subscribe_msgs = [msg for msg in ws.sent if "\"subscribe\"" in msg]
        assert sorted(subscribe_msgs) == sorted(
            [
                '5::/api/v2/socket_io:{"name":"subscribe","args":["/htr/01/samples"]}',
                '5::/api/v2/socket_io:{"name":"subscribe","args":["/htr/02/samples"]}',
            ]
        )

        assert client._stats.frames_total == 4
        assert client._stats.events_total == 1

        dev_state = coordinator.data["dev"]
        assert dev_state["nodes"] == {
            "nodes": [
                {"addr": "01", "type": "htr"},
                {"addr": "02", "type": "HTR"},
            ]
        }
        assert dev_state["htr"]["addrs"] == ["01", "02"]
        assert dev_state["htr"]["settings"]["01"] == {"temp": 21}
        assert dev_state["htr"]["settings"]["02"] == {"temp": 22}
        assert dev_state["htr"]["advanced"]["01"] == {"adv": True}
        assert dev_state["raw"]["misc"] == {"foo": "bar"}

        status_signal = module.signal_ws_status("entry")
        data_signal = module.signal_ws_data("entry")
        status_payloads = [
            call.args[2]
            for call in dispatcher.call_args_list
            if call.args[1] == status_signal
        ]
        assert [p["status"] for p in status_payloads] == [
            "starting",
            "connected",
            "disconnected",
            "stopped",
        ]
        assert all(p["dev_id"] == "dev" for p in status_payloads)

        data_payloads = [
            call.args[2]
            for call in dispatcher.call_args_list
            if call.args[1] == data_signal
        ]
        assert len(data_payloads) == 5
        aggregate_payloads = [
            payload
            for payload in data_payloads
            if "kind" not in payload
            and "nodes" in payload
            and "nodes_by_type" in payload
        ]
        assert len(aggregate_payloads) == 1
        assert aggregate_payloads[0]["nodes"] == {
            "nodes": [
                {"addr": "01", "type": "htr"},
                {"addr": "02", "type": "HTR"},
            ]
        }
        assert aggregate_payloads[0]["nodes_by_type"] == {
            "htr": {"addrs": ["01", "02"]},
        }

        per_node_payloads = [payload for payload in data_payloads if "kind" in payload]
        assert len(per_node_payloads) == 4
        ts = client._stats.last_event_ts
        assert {
            (p["kind"], p["addr"])
            for p in per_node_payloads
        } == {
            ("nodes", None),
            ("htr_settings", "01"),
            ("htr_settings", "02"),
            ("htr_samples", "02"),
        }
        for payload in per_node_payloads:
            assert payload["dev_id"] == "dev"
            assert payload["ts"] == ts

        ws_state = hass.data[module.DOMAIN]["entry"]["ws_state"]["dev"]
        assert ws_state["status"] == "stopped"
        assert ws_state["frames_total"] == 4
        assert ws_state["events_total"] == 1
        assert ws_state["last_event_at"] == ts
        assert ws_state["healthy_since"] is None

    asyncio.run(_run())


def test_runner_cleans_up_close_errors() -> None:
    async def _run() -> None:
        module = _load_ws_client()
        loop = asyncio.get_event_loop()
        hass = types.SimpleNamespace(
            loop=loop,
            data={module.DOMAIN: {"entry": {"ws_state": {}}}},
        )
        coordinator = types.SimpleNamespace(data={})
        session = module.aiohttp.testing.FakeClientSession()
        client = module.WebSocket09Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=types.SimpleNamespace(_session=session),
            coordinator=coordinator,
            session=session,
        )

        class FakeTask:
            def __init__(self) -> None:
                self.cancelled = False

            def cancel(self) -> None:
                self.cancelled = True

        fake_task = FakeTask()
        client._hb_task = fake_task  # type: ignore[assignment]

        class FailingWS:
            async def close(self) -> None:
                raise RuntimeError("close boom")

        client._ws = FailingWS()

        async def failing_handshake() -> tuple[str, int]:
            client._closing = True
            raise RuntimeError("boom")

        client._handshake = failing_handshake  # type: ignore[assignment]

        await client._runner()

        assert fake_task.cancelled is True
        assert client._ws is None

    asyncio.run(_run())


def test_read_loop_bubbles_exception_on_close():
    async def _run() -> None:
        module = _load_ws_client()
        Client = module.WebSocket09Client
        hass = types.SimpleNamespace(
            loop=asyncio.get_event_loop(),
            data={module.DOMAIN: {}},
        )
        api = types.SimpleNamespace(_session=None)
        coordinator = types.SimpleNamespace()
        client = Client(hass, entry_id="e", dev_id="d", api_client=api, coordinator=coordinator)

        aiohttp = module.aiohttp

        class DummyWS:
            def __init__(self):
                self.close_code = 1006

            async def receive(self):
                return types.SimpleNamespace(type=aiohttp.WSMsgType.CLOSED, data=None, extra="bye")

            def exception(self):
                return RuntimeError("boom")

        client._ws = DummyWS()
        with pytest.raises(RuntimeError, match="boom"):
            await client._read_loop()

    asyncio.run(_run())


def test_read_loop_handles_error_frames_and_health(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        module = _load_ws_client()
        Client = module.WebSocket09Client
        hass = types.SimpleNamespace(
            loop=asyncio.get_event_loop(),
            data={module.DOMAIN: {}},
        )
        api = types.SimpleNamespace(_session=None)
        coordinator = types.SimpleNamespace()
        client = Client(hass, entry_id="e", dev_id="d", api_client=api, coordinator=coordinator)

        aiohttp = module.aiohttp
        namespace = module.WS_NAMESPACE

        messages = [
            (
                types.SimpleNamespace(
                    type=aiohttp.WSMsgType.TEXT, data="2::", extra=None
                ),
                None,
            ),
            (
                types.SimpleNamespace(type=999, data="ignored", extra=None),
                None,
            ),
            (
                types.SimpleNamespace(
                    type=aiohttp.WSMsgType.TEXT,
                    data=f"5::{namespace}:not-json",
                    extra=None,
                ),
                None,
            ),
            (
                types.SimpleNamespace(
                    type=aiohttp.WSMsgType.ERROR, data=None, extra=None
                ),
                ValueError("socket exploded"),
            ),
        ]

        class DummyWS:
            def __init__(self, entries: list[tuple[Any, Any]]) -> None:
                self._entries = entries
                self.close_code = None
                self.recorded: list[Any] = []
                self._current_exc: BaseException | None = None

            async def receive(self) -> Any:
                msg, exc = self._entries.pop(0)
                self._current_exc = exc
                self.recorded.append(msg.type)
                return msg

            def exception(self) -> BaseException | None:
                return self._current_exc

        ws = DummyWS(messages)
        client._ws = ws
        client._connected_since = 600.0
        client._healthy_since = None

        updates: list[str] = []
        client._update_status = MagicMock(
            side_effect=lambda status: updates.append(status)
        )

        monkeypatch.setattr(module.time, "time", lambda: 1000.0)
        monkeypatch.setattr(ws_core.time, "time", lambda: 1000.0)

        with pytest.raises(ValueError, match="socket exploded"):
            await client._read_loop()

        assert ws.recorded == [
            aiohttp.WSMsgType.TEXT,
            999,
            aiohttp.WSMsgType.TEXT,
            aiohttp.WSMsgType.ERROR,
        ]
        assert client._stats.frames_total == 2
        assert client._stats.last_event_ts == 0.0
        assert client._healthy_since is None
        assert updates == []

    asyncio.run(_run())


def test_connect_ws_uses_secure_endpoint() -> None:
    async def _run() -> None:
        module = _load_ws_client()
        aiohttp = module.aiohttp
        ws_obj = types.SimpleNamespace()
        session = aiohttp.ClientSession(ws_connect_results=[ws_obj])

        loop = asyncio.get_event_loop()
        hass = types.SimpleNamespace(loop=loop)
        coordinator = types.SimpleNamespace()
        api = types.SimpleNamespace(
            _session=session,
            _authed_headers=AsyncMock(
                return_value={"Authorization": "Bearer active-token"}
            ),
            _ensure_token=AsyncMock(),
        )

        client = module.WebSocket09Client(
            hass,
            entry_id="entry",
            dev_id="dev-42",
            api_client=api,
            coordinator=coordinator,
            session=session,
        )

        await client._connect_ws("SID123")

        assert client._ws is ws_obj
        assert len(session.ws_connect_calls) == 1
        ws_call = session.ws_connect_calls[0]
        expected_url = (
            f"{module.API_BASE.replace('https://', 'wss://')}/socket.io/1/websocket/"
            f"SID123?token=active-token&dev_id=dev-42"
        )
        assert ws_call["url"] == expected_url
        assert ws_call["kwargs"]["heartbeat"] is None
        assert ws_call["kwargs"]["timeout"] == 15

    asyncio.run(_run())


def test_handshake_refresh_failure_raises_handshake_error() -> None:
    async def _run() -> None:
        module = _load_ws_client()
        aiohttp = module.aiohttp
        session = aiohttp.ClientSession(
            get_responses=[
                {"status": 401, "body": "unauthorized"},
                {"status": 503, "body": "server fail"},
            ]
        )

        hass = types.SimpleNamespace(loop=asyncio.get_event_loop())
        coordinator = types.SimpleNamespace()
        api = types.SimpleNamespace(
            _session=session,
            _authed_headers=AsyncMock(
                side_effect=[
                    {"Authorization": "Bearer stale"},
                    {"Authorization": "Bearer fresh"},
                ]
            ),
            _ensure_token=AsyncMock(),
            _access_token="stale",
        )

        client = module.WebSocket09Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=api,
            coordinator=coordinator,
            session=session,
        )

        with pytest.raises(module.HandshakeError) as ctx:
            await client._handshake()

        err = ctx.value
        assert err.status == 503
        assert err.body_snippet == "server fail"
        assert "token=fresh" in err.url
        assert api._ensure_token.await_count == 1
        assert api._authed_headers.await_count == 2
        assert len(session.get_calls) == 2
        assert "token=stale" in session.get_calls[0]["url"]
        assert "token=fresh" in session.get_calls[1]["url"]

    asyncio.run(_run())


def test_handshake_wraps_client_error() -> None:
    async def _run() -> None:
        module = _load_ws_client()
        aiohttp = module.aiohttp

        def raise_client_error(*, url: str, timeout: Any | None = None) -> None:
            raise aiohttp.ClientError("boom")

        session = aiohttp.ClientSession(get_responses=[raise_client_error])

        hass = types.SimpleNamespace(loop=asyncio.get_event_loop())
        coordinator = types.SimpleNamespace()
        api = types.SimpleNamespace(
            _session=session,
            _authed_headers=AsyncMock(
                return_value={"Authorization": "Bearer cached"}
            ),
            _ensure_token=AsyncMock(),
        )

        client = module.WebSocket09Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=api,
            coordinator=coordinator,
            session=session,
        )

        with pytest.raises(module.HandshakeError) as ctx:
            await client._handshake()

        err = ctx.value
        assert err.status == -1
        assert "token=cached" in err.url
        assert err.body_snippet == "boom"
        assert isinstance(err.__cause__, aiohttp.ClientError)
        assert api._authed_headers.await_count == 1
        assert api._ensure_token.await_count == 0

    asyncio.run(_run())


@pytest.mark.parametrize(
    "response, expected_status, expected_body",
    [
        ({"status": 403, "body": "denied"}, 403, "denied"),
        ({"status": 503, "body": "oops"}, 503, "oops"),
    ],
)
def test_handshake_status_error_raises_handshake_error(
    response: dict[str, object], expected_status: int, expected_body: str
) -> None:
    async def _run() -> None:
        module = _load_ws_client()
        session = module.aiohttp.testing.FakeClientSession(get_responses=[response])

        hass = types.SimpleNamespace(loop=asyncio.get_event_loop())
        coordinator = types.SimpleNamespace()
        api = types.SimpleNamespace(
            _session=session,
            _authed_headers=AsyncMock(return_value={"Authorization": "Bearer cached"}),
            _ensure_token=AsyncMock(),
        )

        client = module.WebSocket09Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=api,
            coordinator=coordinator,
            session=session,
        )

        with pytest.raises(module.HandshakeError) as ctx:
            await client._handshake()

        err = ctx.value
        assert err.status == expected_status
        assert err.body_snippet == expected_body
        assert "token=cached" in err.url
        assert api._ensure_token.await_count == 0
        assert api._authed_headers.await_count == 1

    asyncio.run(_run())


def test_handle_event_updates_state_and_dispatch(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    module = _load_ws_client()
    module.async_dispatcher_send = MagicMock()
    loop = asyncio.new_event_loop()
    caplog.set_level(logging.DEBUG, logger=module.__name__)
    class RecordingCoordinator:
        def __init__(self) -> None:
            self.data = {
                "dev": {
                    "dev_id": "dev",
                    "name": "Device dev",
                    "raw": {"existing": True},
                    "connected": True,
                    "nodes": None,
                    "nodes_by_type": {"htr": {"addrs": [], "settings": {}, "advanced": {}, "samples": {}}},
                    "htr": {"addrs": [], "settings": {}, "advanced": {}, "samples": {}},
                }
            }

        def update_nodes(self, nodes: dict[str, Any], inventory: list[Any]) -> None:
            node_updates.append((nodes, inventory))

    node_updates: list[tuple[dict[str, Any], list[Any]]] = []
    energy_updates: list[Any] = []

    class FakeEnergyCoordinator:
        def update_addresses(self, addrs: Iterable[str] | dict[str, Iterable[str]]) -> None:
            if isinstance(addrs, dict):
                energy_updates.append({k: list(v) for k, v in addrs.items()})
            else:
                energy_updates.append(list(addrs))

    dispatch_payloads: list[dict[str, Any]] = []

    orig_dispatch = module.WebSocket09Client._dispatch_nodes

    def record_dispatch(self: Any, payload: dict[str, Any]) -> dict[str, list[str]]:
        dispatch_payloads.append(copy.deepcopy(payload))
        return orig_dispatch(self, payload)

    monkeypatch.setattr(module.WebSocket09Client, "_dispatch_nodes", record_dispatch)

    hass = types.SimpleNamespace(
        loop=loop,
        data={
            module.DOMAIN: {
                "entry": {
                    "ws_state": {},
                    "energy_coordinator": FakeEnergyCoordinator(),
                }
            }
        },
    )
    coordinator = RecordingCoordinator()
    api = types.SimpleNamespace(_session=types.SimpleNamespace())
    client = module.WebSocket09Client(
        hass,
        entry_id="entry",
        dev_id="dev",
        api_client=api,
        coordinator=coordinator,
    )
    monkeypatch.setattr(module.time, "time", lambda: 1000.0)
    monkeypatch.setattr(ws_core.time, "time", lambda: 1000.0)

    event = {
        "name": "data",
        "args": [
            [
                {
                    "path": "/mgr/nodes",
                    "body": {
                        "nodes": [
                            {"addr": "01", "type": "htr"},
                            {"addr": "02", "type": "HTR"},
                            {"addr": "03", "type": "acm"},
                        ]
                    },
                },
                {"path": "/htr/01/settings", "body": {"temp": 21}},
                {"path": "/htr/01/advanced_setup", "body": {"adv": True}},
                {
                    "path": "/htr/02/samples",
                    "body": [{"ts": 1, "val": 2}],
                },
                {"path": "/acm/03/settings", "body": {"mode": "eco"}},
                {"path": "/misc", "body": {"foo": "bar"}},
            ]
        ],
    }

    client._handle_event(event)
    loop.run_until_complete(asyncio.sleep(0))

    assert dispatch_payloads == [event["args"][0][0]["body"]]

    dev_data = coordinator.data["dev"]
    assert dev_data["nodes"] == {
        "nodes": [
            {"addr": "01", "type": "htr"},
            {"addr": "02", "type": "HTR"},
            {"addr": "03", "type": "acm"},
        ]
    }
    assert dev_data["htr"]["addrs"] == ["01", "02"]
    assert dev_data["htr"]["settings"]["01"] == {"temp": 21}
    assert dev_data["htr"]["advanced"]["01"] == {"adv": True}
    assert dev_data["nodes_by_type"]["acm"]["addrs"] == ["03"]
    assert dev_data["nodes_by_type"]["acm"]["settings"]["03"] == {"mode": "eco"}
    assert dev_data["raw"]["misc"] == {"foo": "bar"}
    assert client._stats.events_total == 1

    aggregate_payloads = [
        call.args[2]
        for call in module.async_dispatcher_send.call_args_list
        if call.args[1] == module.signal_ws_data("entry")
        and "nodes" in call.args[2]
        and "nodes_by_type" in call.args[2]
        and "kind" not in call.args[2]
    ]
    assert len(aggregate_payloads) == 1
    assert aggregate_payloads[0]["nodes"] == {
        "nodes": [
            {"addr": "01", "type": "htr"},
            {"addr": "02", "type": "HTR"},
            {"addr": "03", "type": "acm"},
        ]
    }
    assert aggregate_payloads[0]["nodes_by_type"] == {
        "acm": {"addrs": ["03"]},
        "htr": {"addrs": ["01", "02"]},
    }

    module.async_dispatcher_send.assert_has_calls(
        [
            call(
                hass,
                module.signal_ws_data("entry"),
                {
                    "dev_id": "dev",
                    "ts": 1000.0,
                    "addr": None,
                    "kind": "nodes",
                    "node_type": None,
                },
            ),
            call(
                hass,
                module.signal_ws_data("entry"),
                {
                    "dev_id": "dev",
                    "ts": 1000.0,
                    "addr": "01",
                    "kind": "htr_settings",
                    "node_type": "htr",
                },
            ),
            call(
                hass,
                module.signal_ws_data("entry"),
                {
                    "dev_id": "dev",
                    "ts": 1000.0,
                    "addr": "02",
                    "kind": "htr_samples",
                    "node_type": "htr",
                },
            ),
            call(
                hass,
                module.signal_ws_data("entry"),
                {
                    "dev_id": "dev",
                    "ts": 1000.0,
                    "addr": "03",
                    "kind": "acm_settings",
                    "node_type": "acm",
                },
            ),
        ],
        any_order=True,
    )
    assert module.async_dispatcher_send.call_count == 5

    assert len(node_updates) == 1
    assert node_updates[0][0] == {
        "nodes": [
            {"addr": "01", "type": "htr"},
            {"addr": "02", "type": "HTR"},
            {"addr": "03", "type": "acm"},
        ]
    }
    assert [node.addr for node in node_updates[0][1]] == ["01", "02", "03"]
    assert energy_updates == [{"acm": ["03"], "htr": ["01", "02"]}]
    assert "unknown node types" not in caplog.text

    ws_state = hass.data[module.DOMAIN]["entry"]["ws_state"]["dev"]
    assert ws_state["last_event_at"] == 1000.0
    assert ws_state["events_total"] == 1
    assert ws_state["frames_total"] == 0
    loop.close()


def test_handle_event_logs_unknown_types(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    module = _load_ws_client()
    module.async_dispatcher_send = MagicMock()
    caplog.set_level(logging.DEBUG, logger=module.__name__)

    loop = asyncio.new_event_loop()

    class RecordingCoordinator:
        def __init__(self) -> None:
            self.data = {
                "dev": {
                    "dev_id": "dev",
                    "name": "Device dev",
                    "raw": {},
                    "connected": True,
                    "nodes": None,
                    "nodes_by_type": {},
                    "htr": {"addrs": [], "settings": {}},
                }
            }

        def update_nodes(self, nodes: dict[str, Any], inventory: list[Any]) -> None:
            node_updates.append((nodes, inventory))

    node_updates: list[tuple[dict[str, Any], list[Any]]] = []
    energy_updates: list[Any] = []

    class FakeEnergyCoordinator:
        def update_addresses(self, addrs: Iterable[str] | dict[str, Iterable[str]]) -> None:
            if isinstance(addrs, dict):
                energy_updates.append({k: list(v) for k, v in addrs.items()})
            else:
                energy_updates.append(list(addrs))

    hass = types.SimpleNamespace(
        loop=loop,
        data={
            module.DOMAIN: {
                "entry": {
                    "ws_state": {},
                    "energy_coordinator": FakeEnergyCoordinator(),
                }
            }
        },
    )
    coordinator = RecordingCoordinator()
    api = types.SimpleNamespace(_session=types.SimpleNamespace())
    client = module.WebSocket09Client(
        hass,
        entry_id="entry",
        dev_id="dev",
        api_client=api,
        coordinator=coordinator,
    )

    event = {
        "name": "data",
        "args": [
            [
                {
                    "path": "/mgr/nodes",
                    "body": {
                        "nodes": [
                            {"addr": "01", "type": "htr"},
                            {"addr": "09", "type": "gizmo"},
                        ]
                    },
                }
            ]
        ],
    }

    client._handle_event(event)

    assert any("unknown node types in inventory: gizmo" in rec.message for rec in caplog.records)
    assert energy_updates == [{"htr": ["01"]}]
    assert node_updates
    loop.close()


def test_subscribe_htr_samples_sends_expected_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_ws_client()
    Client = module.WebSocket09Client
    original = Client._ensure_type_bucket
    call_record: list[str] = []

    def wrapper(self: Any, dev_map: Any, nodes_by_type: Any, node_type: str) -> Any:
        call_record.append(node_type)
        return original(self, dev_map, nodes_by_type, node_type)

    monkeypatch.setattr(Client, "_ensure_type_bucket", wrapper)

    async def _run() -> None:
        energy_updates: list[Any] = []
        helper_calls: list[tuple[Any, Any, dict[str, list[str]]]] = []
        prepare_calls: list[Any] = []

        class FakeEnergyCoordinator:
            def update_addresses(self, addrs: Any) -> None:
                if isinstance(addrs, dict):
                    energy_updates.append({k: list(v) for k, v in addrs.items()})
                else:
                    energy_updates.append(list(addrs))

        hass = types.SimpleNamespace(
            loop=asyncio.get_event_loop(),
            data={module.DOMAIN: {"entry": {"energy_coordinator": FakeEnergyCoordinator()}}},
        )

        inventory_nodes = [
            types.SimpleNamespace(addr="01", type="htr"),
            types.SimpleNamespace(addr="02", type="htr"),
            types.SimpleNamespace(addr="A1", type="acm"),
        ]

        def fake_prepare(entry_data: Any, *, default_name_simple: Any) -> tuple[Any, Any, Any, Any]:
            prepare_calls.append(entry_data)
            assert entry_data is hass.data[module.DOMAIN]["entry"]
            return (
                list(inventory_nodes),
                {},
                {"htr": ["01", "02"], "acm": ["A1"]},
                lambda *_args: "",
            )

        monkeypatch.setattr(module, "prepare_heater_platform_data", fake_prepare)

        class RecordingCoordinator:
            def __init__(self) -> None:
                htr_bucket: dict[str, Any] = {
                    "addrs": ["01"],
                    "settings": {},
                    "advanced": {},
                    "samples": {},
                }
                self.data = {
                    "dev": {
                        "nodes_by_type": {"htr": htr_bucket},
                        "htr": htr_bucket,
                    }
                }
                self._node_inventory = module.build_node_inventory(
                    {
                        "nodes": [
                            {"addr": "01", "type": "htr"},
                            {"addr": "02", "type": "htr"},
                            {"addr": "A1", "type": "acm"},
                        ]
                    }
                )

        coordinator = RecordingCoordinator()
        api = types.SimpleNamespace(_session=None)
        client = Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=api,
            coordinator=coordinator,
        )

        original_helper = Client._apply_heater_addresses

        def helper(self: Any, addr_map: Any, *, inventory: Any = None) -> Any:
            normalized = original_helper(self, addr_map, inventory=inventory)
            helper_calls.append((addr_map, inventory, normalized))
            return normalized

        monkeypatch.setattr(Client, "_apply_heater_addresses", helper)

        class DummyWS:
            def __init__(self) -> None:
                self.sent: list[str] = []

            async def send_str(self, data: str) -> None:
                self.sent.append(data)

        ws = DummyWS()
        client._ws = ws

        await client._subscribe_htr_samples()

        assert helper_calls
        normalized_map = helper_calls[0][2]
        other_types = sorted(nt for nt in normalized_map if nt != "htr")
        expected_payloads = []
        for node_type in ["htr", *other_types]:
            for addr in normalized_map.get(node_type, []):
                expected_payloads.append(
                    f'5::/api/v2/socket_io:{{"name":"subscribe","args":["/{node_type}/{addr}/samples"]}}'
                )

        assert ws.sent == expected_payloads
        dev_map = coordinator.data["dev"]
        assert dev_map["nodes_by_type"]["acm"]["addrs"] == ["A1"]
        assert dev_map["htr"] is dev_map["nodes_by_type"]["htr"]
        assert energy_updates == [{"htr": ["01", "02"], "acm": ["A1"]}]
        assert "htr" in call_record
        assert "acm" in call_record
        assert helper_calls[0][0] == {"htr": ["01", "02"], "acm": ["A1"]}
        inventory = helper_calls[0][1]
        assert inventory is not None
        assert sorted((getattr(node, "type", None), getattr(node, "addr", None)) for node in inventory) == [
            ("acm", "A1"),
            ("htr", "01"),
            ("htr", "02"),
        ]
        assert prepare_calls == [hass.data[module.DOMAIN]["entry"]]

    asyncio.run(_run())


def test_subscribe_htr_samples_returns_when_no_addresses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        module = _load_ws_client()
        Client = module.WebSocket09Client

        hass = types.SimpleNamespace(
            loop=asyncio.get_event_loop(), data={module.DOMAIN: {"entry": {}}}
        )

        def fake_prepare(entry_data: Any, *, default_name_simple: Any) -> tuple[Any, Any, Any, Any]:
            return ([], {}, {"htr": [], "acm": []}, lambda *_args: "")

        monkeypatch.setattr(module, "prepare_heater_platform_data", fake_prepare)

        coordinator = types.SimpleNamespace(_node_inventory=[], _nodes={}, data={})
        api = types.SimpleNamespace(_session=None)
        client = Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=api,
            coordinator=coordinator,
        )

        class DummyWS:
            def __init__(self) -> None:
                self.sent: list[str] = []

            async def send_str(self, data: str) -> None:
                self.sent.append(data)

        ws = DummyWS()
        client._ws = ws

        await client._subscribe_htr_samples()

        assert ws.sent == []

    asyncio.run(_run())


def test_subscribe_htr_samples_uses_cached_and_raw_inventory(monkeypatch):
    async def _run() -> None:
        module = _load_ws_client()
        Client = module.WebSocket09Client

        class TruthyEmpty(list):
            def __bool__(self) -> bool:  # pragma: no cover - behaviour hook
                return True

        prepare_calls: list[Any] = []

        def fake_prepare(entry_data: Any, *, default_name_simple: Any) -> tuple[Any, Any, Any, Any]:
            prepare_calls.append(entry_data)
            cached_inventory = entry_data.get("node_inventory")
            if isinstance(cached_inventory, list) and cached_inventory:
                inventory = list(cached_inventory)
            else:
                nodes_payload = entry_data.get("nodes")
                inventory = module.build_node_inventory(nodes_payload)
            heater_addrs = [
                getattr(node, "addr", None)
                for node in inventory
                if getattr(node, "type", "").lower() == "htr"
            ]
            return (
                inventory,
                {},
                {"htr": heater_addrs, "acm": TruthyEmpty()},
                lambda *_args: "",
            )

        monkeypatch.setattr(module, "prepare_heater_platform_data", fake_prepare)

        hass = types.SimpleNamespace(loop=asyncio.get_event_loop())
        cached_inventory = module.build_node_inventory({"nodes": [{"type": "htr", "addr": "01"}]})
        hass.data = {
            module.DOMAIN: {
                "entry": {
                    "node_inventory": cached_inventory,
                    "nodes": {"nodes": [{"type": "htr", "addr": "02"}]},
                    "energy_coordinator": types.SimpleNamespace(update_addresses=MagicMock()),
                }
            }
        }

        coordinator = types.SimpleNamespace(
            _node_inventory=[], _nodes={"nodes": []}, data={"dev": {}}, hass=hass
        )
        api = types.SimpleNamespace(_session=None)
        client = Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=api,
            coordinator=coordinator,
        )

        class DummyWS:
            def __init__(self) -> None:
                self.sent: list[str] = []

            async def send_str(self, data: str) -> None:
                self.sent.append(data)

        ws = DummyWS()
        client._ws = ws

        # First call should use cached inventory
        await client._subscribe_htr_samples()
        assert ws.sent == [
            '5::/api/v2/socket_io:{"name":"subscribe","args":["/htr/01/samples"]}'
        ]
        record = hass.data[module.DOMAIN]["entry"]
        assert [node.addr for node in record["node_inventory"]] == ["01"]

        # Second call should rebuild from raw nodes when cache is absent
        record["node_inventory"] = []
        coordinator._node_inventory = []
        ws.sent.clear()

        await client._subscribe_htr_samples()
        assert ws.sent == [
            '5::/api/v2/socket_io:{"name":"subscribe","args":["/htr/02/samples"]}'
        ]
        assert [node.addr for node in record["node_inventory"]] == ["02"]
        assert prepare_calls[-1] is hass.data[module.DOMAIN]["entry"]

    asyncio.run(_run())


def test_subscribe_htr_samples_handles_empty_non_htr(monkeypatch):
    async def _run() -> None:
        module = _load_ws_client()
        Client = module.WebSocket09Client

        class Stage1Sequence:
            def __iter__(self):
                return iter(())

            def __bool__(self) -> bool:
                return True

        def fake_prepare(entry_data: Any, *, default_name_simple: Any) -> tuple[Any, Any, Any, Any]:
            inventory = module.build_node_inventory({"nodes": [{"type": "htr", "addr": "01"}]})
            return (
                inventory,
                {},
                {"htr": [node.addr for node in inventory], "acm": Stage1Sequence()},
                lambda *_args: "",
            )

        monkeypatch.setattr(module, "prepare_heater_platform_data", fake_prepare)

        hass = types.SimpleNamespace(
            loop=asyncio.get_event_loop(),
            data={module.DOMAIN: {"entry": {"energy_coordinator": types.SimpleNamespace(update_addresses=MagicMock())}}},
        )
        coordinator = types.SimpleNamespace(
            _node_inventory=module.build_node_inventory({"nodes": [{"type": "htr", "addr": "01"}]}),
            _nodes={},
            data={"dev": {}},
        )
        api = types.SimpleNamespace(_session=None)
        client = Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=api,
            coordinator=coordinator,
        )

        class DummyWS:
            def __init__(self) -> None:
                self.sent: list[str] = []

            async def send_str(self, data: str) -> None:
                self.sent.append(data)

        ws = DummyWS()
        client._ws = ws

        await client._subscribe_htr_samples()

        # Only the heater address should be subscribed; accumulator skipped via continue
        assert ws.sent == [
            '5::/api/v2/socket_io:{"name":"subscribe","args":["/htr/01/samples"]}'
        ]

    asyncio.run(_run())


def test_dispatch_nodes_uses_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_ws_client()
    Client = module.WebSocket09Client

    helper_calls: list[tuple[Any, Any]] = []

    original_helper = Client._apply_heater_addresses

    def helper(self: Any, addr_map: Any, *, inventory: Any = None) -> Any:
        helper_calls.append((addr_map, inventory))
        return original_helper(self, addr_map, inventory=inventory)

    monkeypatch.setattr(Client, "_apply_heater_addresses", helper)

    dispatcher_calls: list[tuple[str, dict[str, Any]]] = []

    def fake_dispatcher(hass: Any, signal: str, payload: dict[str, Any]) -> None:
        dispatcher_calls.append((signal, payload))

    monkeypatch.setattr(module, "async_dispatcher_send", fake_dispatcher)
    monkeypatch.setattr(ws_core, "async_dispatcher_send", fake_dispatcher)

    class FakeEnergyCoordinator:
        def __init__(self) -> None:
            self.calls: list[Any] = []

        def update_addresses(self, addrs: Any) -> None:
            self.calls.append(addrs)

    energy = FakeEnergyCoordinator()

    class DummyLoop:
        def call_soon_threadsafe(self, callback: Any) -> None:
            callback()

    hass = types.SimpleNamespace(
        loop=DummyLoop(),
        data={module.DOMAIN: {"entry": {"ws_state": {}, "energy_coordinator": energy}}},
    )

    class RecordingCoordinator:
        def __init__(self) -> None:
            self.nodes_updates: list[Any] = []
            self.data = {
                "dev": {
                    "nodes_by_type": {},
                    "htr": {
                        "addrs": [],
                        "settings": {},
                        "advanced": {},
                        "samples": {},
                    },
                }
            }

        def update_nodes(self, nodes: dict[str, Any], inventory: list[Any]) -> None:
            self.nodes_updates.append((nodes, inventory))

    coordinator = RecordingCoordinator()
    api = types.SimpleNamespace(_session=None)
    client = Client(
        hass,
        entry_id="entry",
        dev_id="dev",
        api_client=api,
        coordinator=coordinator,
    )

    payload = {
        "nodes": [
            {"addr": "01", "type": "htr"},
            {"addr": "02", "type": "htr"},
            {"addr": "A1", "type": "acm"},
        ]
    }

    result = client._dispatch_nodes(payload)

    assert helper_calls
    addr_map_arg = helper_calls[0][0]
    assert {
        key: sorted(value) for key, value in addr_map_arg.items()
    } == {"acm": ["A1"], "htr": ["01", "02"]}
    inventory_arg = helper_calls[0][1]
    assert inventory_arg is None
    assert coordinator.nodes_updates
    raw_nodes_arg, inventory = coordinator.nodes_updates[0]
    assert raw_nodes_arg == payload
    assert energy.calls == [{"htr": ["01", "02"], "acm": ["A1"]}]
    record = hass.data[module.DOMAIN]["entry"]
    assert "node_inventory" in record and record["node_inventory"] is inventory
    assert sorted(
        (getattr(node, "type", None), getattr(node, "addr", None)) for node in inventory
    ) == [("acm", "A1"), ("htr", "01"), ("htr", "02")]
    assert coordinator.nodes_updates[0][1] is record["node_inventory"]
    assert {
        key: sorted(value) for key, value in result.items()
    } == {"acm": ["A1"], "htr": ["01", "02"]}
    assert dispatcher_calls


def test_dispatch_nodes_reuses_cached_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_ws_client()
    Client = module.WebSocket09Client

    helper_calls: list[tuple[Any, Any]] = []

    original_helper = Client._apply_heater_addresses

    def helper(self: Any, addr_map: Any, *, inventory: Any = None) -> Any:
        helper_calls.append((addr_map, inventory))
        return original_helper(self, addr_map, inventory=inventory)

    monkeypatch.setattr(Client, "_apply_heater_addresses", helper)

    class FakeEnergyCoordinator:
        def __init__(self) -> None:
            self.calls: list[Any] = []

        def update_addresses(self, addrs: Any) -> None:
            self.calls.append(addrs)

    energy = FakeEnergyCoordinator()

    class DummyLoop:
        def call_soon_threadsafe(self, callback: Any) -> None:
            callback()

    cached_payload = {"nodes": [{"addr": "01", "type": "htr"}]}
    cached_inventory = module.build_node_inventory(cached_payload)

    hass = types.SimpleNamespace(
        loop=DummyLoop(),
        data={
            module.DOMAIN: {
                "entry": {
                    "ws_state": {},
                    "energy_coordinator": energy,
                    "node_inventory": list(cached_inventory),
                    "nodes": {},
                }
            }
        },
    )

    class RecordingCoordinator:
        def __init__(self) -> None:
            self.nodes_updates: list[Any] = []
            self.data: dict[str, Any] = {}

        def update_nodes(self, nodes: dict[str, Any], inventory: list[Any]) -> None:
            self.nodes_updates.append((nodes, inventory))

    coordinator = RecordingCoordinator()
    api = types.SimpleNamespace(_session=None)
    client = Client(
        hass,
        entry_id="entry",
        dev_id="dev",
        api_client=api,
        coordinator=coordinator,
    )

    def explode_build(raw_nodes: Any) -> list[Any]:  # pragma: no cover - defensive
        raise AssertionError("build_node_inventory should not be called")

    monkeypatch.setattr(nodes, "build_node_inventory", explode_build)

    payload = {"nodes": [{"addr": "01", "type": "htr"}]}

    result = client._dispatch_nodes(payload)

    assert result == {"htr": ["01"]}
    assert helper_calls
    assert helper_calls[0][1] is None
    assert coordinator.nodes_updates
    raw_nodes_arg, inventory = coordinator.nodes_updates[0]
    assert raw_nodes_arg == payload
    record = hass.data[module.DOMAIN]["entry"]
    assert record["nodes"] == payload
    assert record["node_inventory"] is inventory
    assert inventory and inventory[0] is cached_inventory[0]
    assert energy.calls == [{"htr": ["01"]}]


def test_mark_event_promotes_to_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_ws_client()
    module.async_dispatcher_send = MagicMock()
    loop = asyncio.new_event_loop()
    hass = types.SimpleNamespace(
        loop=loop,
        data={module.DOMAIN: {"entry": {"ws_state": {}}}},
    )
    coordinator = types.SimpleNamespace()
    api = types.SimpleNamespace(_session=None)
    client = module.WebSocket09Client(
        hass,
        entry_id="entry",
        dev_id="dev",
        api_client=api,
        coordinator=coordinator,
    )
    client._connected_since = 500.0
    client._healthy_since = None
    client._stats.frames_total = 7
    client._stats.events_total = 3
    monkeypatch.setattr(module.time, "time", lambda: 805.0)
    monkeypatch.setattr(ws_core.time, "time", lambda: 805.0)

    client._mark_event(paths=None)

    ws_state = hass.data[module.DOMAIN]["entry"]["ws_state"]["dev"]
    assert ws_state["status"] == "healthy"
    assert ws_state["healthy_since"] == 805.0
    assert ws_state["last_event_at"] == 805.0
    assert ws_state["frames_total"] == 7
    assert ws_state["events_total"] == 3
    assert client._healthy_since == 805.0
    module.async_dispatcher_send.assert_called_with(
        hass,
        module.signal_ws_status("entry"),
        {"dev_id": "dev", "status": "healthy"},
    )
    loop.close()


def test_status_and_event_share_state_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_ws_client()
    module.async_dispatcher_send = MagicMock()
    loop = asyncio.new_event_loop()
    hass = types.SimpleNamespace(loop=loop, data={})
    coordinator = types.SimpleNamespace()
    api = types.SimpleNamespace(_session=None)
    client = module.WebSocket09Client(
        hass,
        entry_id="entry",
        dev_id="dev",
        api_client=api,
        coordinator=coordinator,
    )

    state = hass.data[module.DOMAIN]["entry"]["ws_state"]["dev"]
    assert client._ws_state_bucket() is state

    monkeypatch.setattr(module.time, "time", lambda: 1000.0)
    client._update_status("connecting")
    assert hass.data[module.DOMAIN]["entry"]["ws_state"]["dev"] is state
    assert state["status"] == "connecting"

    client._stats.frames_total = 4
    client._stats.events_total = 2
    monkeypatch.setattr(module.time, "time", lambda: 1500.0)
    client._mark_event(paths=None, count_event=True)

    assert hass.data[module.DOMAIN]["entry"]["ws_state"]["dev"] is state
    assert state["status"] == "connecting"
    assert state["frames_total"] == 4
    assert state["events_total"] == 3
    assert state["last_event_at"] == 1500.0

    loop.close()


def test_heartbeat_loop_sends_until_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        module = _load_ws_client()
        dispatcher = MagicMock()
        monkeypatch.setattr(module, "async_dispatcher_send", dispatcher)
        loop = asyncio.get_event_loop()
        hass = types.SimpleNamespace(loop=loop, data={module.DOMAIN: {"entry": {}}})
        coordinator = types.SimpleNamespace()
        api = types.SimpleNamespace(_session=None)
        client = module.WebSocket09Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=api,
            coordinator=coordinator,
        )
        client._hb_send_interval = 1.5

        send_calls: list[str] = []
        send_event = asyncio.Event()

        async def fake_send(data: str) -> None:
            send_calls.append(data)
            if len(send_calls) >= 3:
                send_event.set()

        client._send_text = AsyncMock(side_effect=fake_send)

        orig_sleep = asyncio.sleep
        sleep_calls: list[float] = []

        async def fast_sleep(delay: float) -> None:
            sleep_calls.append(delay)
            await orig_sleep(0)

        monkeypatch.setattr(module.asyncio, "sleep", fast_sleep)
        monkeypatch.setattr(ws_core.asyncio, "sleep", fast_sleep)

        now = 2_000.0

        def fake_time() -> float:
            nonlocal now
            now += 1.0
            return now

        monkeypatch.setattr(module.time, "time", fake_time)

        client._update_status("connected")
        status_signal = module.signal_ws_status("entry")

        task = asyncio.create_task(client._heartbeat_loop())
        await asyncio.wait_for(send_event.wait(), timeout=0.2)
        task.cancel()
        await task

        client._update_status("disconnected")

        assert send_calls[:3] == ["2::", "2::", "2::"]
        assert sleep_calls[:3] == [pytest.approx(1.5)] * 3

        status_payloads = [
            call.args[2]
            for call in dispatcher.call_args_list
            if call.args[1] == status_signal
        ]
        assert status_payloads[-2:] == [
            {"dev_id": "dev", "status": "connected"},
            {"dev_id": "dev", "status": "disconnected"},
        ]

        ws_state = hass.data[module.DOMAIN]["entry"].setdefault("ws_state", {})["dev"]
        assert ws_state["status"] == "disconnected"
        assert ws_state["frames_total"] == client._stats.frames_total
        assert ws_state["events_total"] == client._stats.events_total
        assert ws_state["last_event_at"] is None

    asyncio.run(_run())


def test_is_running_property() -> None:
    async def _run() -> None:
        module = _load_ws_client()
        hass = types.SimpleNamespace(
            loop=asyncio.get_event_loop(),
            data={module.DOMAIN: {"entry": {"ws_state": {}}}},
        )
        coordinator = types.SimpleNamespace()
        api = types.SimpleNamespace(_session=None)
        client = module.WebSocket09Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=api,
            coordinator=coordinator,
        )
        assert client.is_running() is False
        task = client.start()
        assert client.is_running() is True
        await client.stop()
        assert client.is_running() is False
        await asyncio.sleep(0)
        if not task.done():
            task.cancel()

    asyncio.run(_run())


def test_handshake_success_resets_backoff() -> None:
    async def _run() -> None:
        module = _load_ws_client(get_responses=[(200, "abc:15:0:websocket")])
        session = module.aiohttp.testing.FakeClientSession()
        hass = types.SimpleNamespace(
            loop=asyncio.get_event_loop(),
            data={module.DOMAIN: {"entry": {"ws_state": {}}}},
        )
        coordinator = types.SimpleNamespace()
        async def authed_headers() -> dict[str, str]:
            return {"Authorization": "Bearer tok"}

        api = types.SimpleNamespace(
            _session=session, _authed_headers=authed_headers
        )
        client = module.WebSocket09Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=api,
            coordinator=coordinator,
            session=session,
        )
        client._backoff_idx = 3
        sid, hb = await client._handshake()
        assert (sid, hb) == ("abc", 15)
        assert client._backoff_idx == 0

    asyncio.run(_run())


def test_heartbeat_loop_handles_send_errors() -> None:
    async def _run() -> None:
        module = _load_ws_client()
        hass = types.SimpleNamespace(
            loop=asyncio.get_event_loop(),
            data={module.DOMAIN: {"entry": {"ws_state": {}}}},
        )
        coordinator = types.SimpleNamespace()
        api = types.SimpleNamespace(_session=None)
        client = module.WebSocket09Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=api,
            coordinator=coordinator,
        )
        client._hb_send_interval = 0

        async def fail_send(_data: str) -> None:
            raise RuntimeError("boom")

        client._send_text = fail_send  # type: ignore[assignment]
        task = asyncio.create_task(client._heartbeat_loop())
        result = await asyncio.wait_for(task, timeout=0.1)
        assert result is None

    asyncio.run(_run())


def test_socketio_heartbeat_triggers_idle_restart(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    module = _load_ws_client()
    caplog.set_level(logging.WARNING, logger=module.__name__)

    async def _run() -> None:
        loop = asyncio.get_event_loop()
        hass = types.SimpleNamespace(
            loop=loop, data={module.DOMAIN: {"entry": {"ws_state": {}}}}
        )
        coordinator = types.SimpleNamespace()

        client = module.WebSocket09Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=types.SimpleNamespace(_session=None),
            coordinator=coordinator,
            protocol="socketio09",
        )

        class FakeWS:
            def __init__(self) -> None:
                self.calls = 0
                self.closed = False

            async def close(self, *args: Any, **kwargs: Any) -> None:
                self.calls += 1
                self.closed = True

        ws = FakeWS()
        client._ws = ws
        client._payload_idle_window = 5.0
        client._last_event_at = 0.0
        client._healthy_since = None

        monkeypatch.setattr(module.time, "time", lambda: 20.0)

        client._record_heartbeat(source="socketio09")
        await asyncio.sleep(0)

        assert ws.calls == 1
        assert client._healthy_since is None
        assert client._stats.events_total == 0
        assert client._idle_restart_task is None
        assert any("no payloads" in record.getMessage() for record in caplog.records)

    asyncio.run(_run())


def test_engineio_heartbeat_triggers_idle_restart(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    module = _load_ws_client()
    caplog.set_level(logging.WARNING, logger=module.__name__)

    async def _run() -> None:
        loop = asyncio.get_event_loop()
        hass = types.SimpleNamespace(
            loop=loop, data={module.DOMAIN: {"entry": {"ws_state": {}}}}
        )
        coordinator = types.SimpleNamespace()

        client = module.WebSocket09Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=types.SimpleNamespace(_session=None),
            coordinator=coordinator,
            protocol="engineio2",
        )

        class FakeWS:
            def __init__(self) -> None:
                self.calls = 0
                self.closed = False

            async def close(self, *args: Any, **kwargs: Any) -> None:
                self.calls += 1
                self.closed = True

        ws = FakeWS()
        client._engineio_ws = ws
        client._payload_idle_window = 5.0
        client._last_event_at = 0.0
        client._healthy_since = None

        monkeypatch.setattr(module.time, "time", lambda: 20.0)

        client._record_heartbeat(source="engineio2")
        await asyncio.sleep(0)

        assert ws.calls == 1
        assert client._healthy_since is None
        assert client._stats.events_total == 0
        assert client._idle_restart_task is None
        assert any(
            "no payloads" in record.getMessage() and "engineio2" in record.getMessage()
            for record in caplog.records
        )

    asyncio.run(_run())


def test_record_heartbeat_respects_idle_window(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_ws_client()
    hass = types.SimpleNamespace(loop=types.SimpleNamespace(create_task=lambda *_: None))
    coordinator = types.SimpleNamespace()
    client = module.WebSocket09Client(
        hass,
        entry_id="entry",
        dev_id="dev",
        api_client=types.SimpleNamespace(_session=None),
        coordinator=coordinator,
    )
    client._payload_idle_window = 30.0
    client._last_event_at = 15.0
    client._schedule_idle_restart = MagicMock()
    monkeypatch.setattr(module.time, "time", lambda: 40.0)

    client._record_heartbeat(source="socketio09")

    client._schedule_idle_restart.assert_not_called()


def test_schedule_idle_restart_skips_when_pending() -> None:
    module = _load_ws_client()
    hass = types.SimpleNamespace(loop=types.SimpleNamespace(create_task=lambda *_: None))
    coordinator = types.SimpleNamespace()
    client = module.WebSocket09Client(
        hass,
        entry_id="entry",
        dev_id="dev",
        api_client=types.SimpleNamespace(_session=None),
        coordinator=coordinator,
    )
    client._idle_restart_pending = True
    client._schedule_idle_restart(idle_for=10.0, source="socketio09")

    assert client._idle_restart_pending is True
    assert client._idle_restart_task is None


def test_schedule_idle_restart_without_loop(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    module = _load_ws_client()
    caplog.set_level(logging.WARNING, logger=module.__name__)

    async def _run() -> None:
        loop = asyncio.get_event_loop()
        hass = types.SimpleNamespace(
            loop=loop, data={module.DOMAIN: {"entry": {"ws_state": {}}}}
        )
        coordinator = types.SimpleNamespace()
        client = module.WebSocket09Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=types.SimpleNamespace(_session=None),
            coordinator=coordinator,
        )

        class FakeWS:
            def __init__(self) -> None:
                self.calls = 0
                self.closed = False

            async def close(self, *args: Any, **kwargs: Any) -> None:
                self.calls += 1
                self.closed = True

        ws = FakeWS()
        client._ws = ws
        client.hass.loop = None

        client._schedule_idle_restart(idle_for=12.0, source="socketio09")
        await asyncio.sleep(0)

        assert ws.calls == 1
        assert client._idle_restart_pending is False
        assert client._idle_restart_task is None
        assert any("socketio09" in record.getMessage() for record in caplog.records)

    asyncio.run(_run())


def test_cancel_idle_restart_cancels_task() -> None:
    module = _load_ws_client()

    async def _run() -> None:
        loop = asyncio.get_event_loop()
        hass = types.SimpleNamespace(
            loop=loop, data={module.DOMAIN: {"entry": {"ws_state": {}}}}
        )
        coordinator = types.SimpleNamespace()
        client = module.WebSocket09Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=types.SimpleNamespace(_session=None),
            coordinator=coordinator,
        )

        async def pending() -> None:
            await asyncio.sleep(0.5)

        task = asyncio.create_task(pending())
        client._idle_restart_task = task
        client._idle_restart_pending = True

        client._cancel_idle_restart()
        await asyncio.sleep(0)

        assert task.cancelled()
        assert client._idle_restart_task is None
        assert client._idle_restart_pending is False

    asyncio.run(_run())


def test_read_loop_returns_when_ws_missing() -> None:
    async def _run() -> None:
        module = _load_ws_client()
        hass = types.SimpleNamespace(
            loop=asyncio.get_event_loop(),
            data={module.DOMAIN: {"entry": {"ws_state": {}}}},
        )
        coordinator = types.SimpleNamespace()
        api = types.SimpleNamespace(_session=None)
        client = module.WebSocket09Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=api,
            coordinator=coordinator,
        )
        client._ws = None
        assert await client._read_loop() is None

    asyncio.run(_run())


@pytest.mark.parametrize(
    ("ws_attr", "context", "method", "terminal", "error_match"),
    [
        ("_ws", "websocket", "_read_loop", "0::", "server disconnect"),
        (
            "_engineio_ws",
            "engine.io websocket",
            "_engineio_read_loop",
            "41",
            "engine.io server disconnect",
        ),
    ],
)
def test_read_loops_delegate_to_shared_stream(
    ws_attr: str, context: str, method: str, terminal: str, error_match: str
) -> None:
    module = _load_ws_client()

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        hass = types.SimpleNamespace(
            loop=loop, data={module.DOMAIN: {"entry": {"ws_state": {}}}}
        )
        coordinator = types.SimpleNamespace()
        client = module.WebSocket09Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=types.SimpleNamespace(_session=None),
            coordinator=coordinator,
        )

        ws = module.aiohttp.testing.FakeWebSocket()
        ws.queue_message({
            "type": module.aiohttp.WSMsgType.TEXT,
            "data": terminal,
        })
        setattr(client, ws_attr, ws)

        seen: list[str] = []
        contexts: list[str] = []
        orig_stream = client._ws_payload_stream

        async def fake_stream(self: Any, ws_obj: Any, *, context: str) -> Any:
            assert ws_obj is ws
            contexts.append(context)
            async for payload in orig_stream(ws_obj, context=context):
                seen.append(payload)
                yield payload

        client._ws_payload_stream = types.MethodType(fake_stream, client)

        with pytest.raises(RuntimeError, match=error_match):
            await getattr(client, method)()

        assert client._stats.frames_total == 1
        assert seen == [terminal]
        assert contexts == [context]

    asyncio.run(_run())


def test_read_loop_handles_close_and_error_messages() -> None:
    async def _run() -> None:
        module = _load_ws_client()
        session = module.aiohttp.testing.FakeClientSession()
        hass = types.SimpleNamespace(
            loop=asyncio.get_event_loop(),
            data={module.DOMAIN: {"entry": {"ws_state": {}}}},
        )
        coordinator = types.SimpleNamespace()
        api = types.SimpleNamespace(_session=session)
        client = module.WebSocket09Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=api,
            coordinator=coordinator,
            session=session,
        )

        ws = module.aiohttp.testing.FakeWebSocket()
        ws.queue_message({
            "type": module.aiohttp.WSMsgType.TEXT,
            "data": f"1::{module.WS_NAMESPACE}",
        })
        ws.queue_message({
            "type": module.aiohttp.WSMsgType.CLOSED,
            "extra": "bye",
        })
        ws.close_code = 1000
        client._ws = ws
        with pytest.raises(RuntimeError, match="websocket closed"):
            await client._read_loop()

        ws2 = module.aiohttp.testing.FakeWebSocket()
        ws2.queue_message({"type": module.aiohttp.WSMsgType.ERROR})
        client._ws = ws2
        with pytest.raises(RuntimeError, match="websocket error"):
            await client._read_loop()

    asyncio.run(_run())


def test_handle_event_seeds_device_state() -> None:
    module = _load_ws_client()
    loop = types.SimpleNamespace(create_task=lambda *_args, **_kwargs: None)
    hass = types.SimpleNamespace(loop=loop, data={})
    coordinator = types.SimpleNamespace(data=None)
    api = types.SimpleNamespace(_session=None)
    client = module.WebSocket09Client(
        hass,
        entry_id="entry",
        dev_id="dev",
        api_client=api,
        coordinator=coordinator,
    )
    client._stats.frames_total = 0
    event = {
        "name": "data",
        "args": [
            [
                {"path": "/mgr/nodes", "body": {"nodes": [{"addr": "A", "type": "htr"}]}},
                {"path": "/htr/A/settings", "body": {"mode": "auto"}},
                {"path": "/htr/A/advanced_setup", "body": {"foo": "bar"}},
                {"path": "/htr/A/samples", "body": []},
                {"path": "/status", "body": {"ok": True}},
            ]
        ],
    }
    client._handle_event(event)
    assert "dev" in coordinator.data
    dev_state = coordinator.data["dev"]
    assert dev_state["nodes"] == {"nodes": [{"addr": "A", "type": "htr"}]}
    assert dev_state["htr"]["settings"]["A"]["mode"] == "auto"


def test_handle_event_invalid_inputs_are_ignored() -> None:
    module = _load_ws_client()
    loop = asyncio.new_event_loop()
    hass = types.SimpleNamespace(
        loop=loop,
        data={module.DOMAIN: {"entry": {"ws_state": {}}}},
    )
    coordinator = types.SimpleNamespace(data={})
    session = module.aiohttp.testing.FakeClientSession()
    api = types.SimpleNamespace(_session=session)
    client = module.WebSocket09Client(
        hass,
        entry_id="entry",
        dev_id="dev",
        api_client=api,
        coordinator=coordinator,
        session=session,
    )

    client._handle_event("not-a-dict")
    client._handle_event({"name": "other", "args": []})
    client._handle_event({"name": "data", "args": "bad"})
    client._handle_event({"name": "data", "args": ["bad"]})
    client._handle_event({"name": "data", "args": [["not-dict"]]})
    client._handle_event({"name": "data", "args": [[{"path": None, "body": {}}]]})


def test_handle_event_adds_missing_htr_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_ws_client()
    Client = module.WebSocket09Client
    original = Client._ensure_type_bucket
    call_record: list[str] = []

    def wrapper(self: Any, dev_map: Any, nodes_by_type: Any, node_type: str) -> Any:
        call_record.append(node_type)
        return original(self, dev_map, nodes_by_type, node_type)

    monkeypatch.setattr(Client, "_ensure_type_bucket", wrapper)

    loop = asyncio.new_event_loop()
    hass = types.SimpleNamespace(
        loop=loop,
        data={module.DOMAIN: {"entry": {"energy_coordinator": types.SimpleNamespace(update_addresses=MagicMock())}}},
    )
    class FakeDevMap(dict):
        def __contains__(self, key: Any) -> bool:
            if key == "htr":
                return False
            return super().__contains__(key)

    coordinator = types.SimpleNamespace(data={"dev": FakeDevMap({"nodes_by_type": {}})})
    session = module.aiohttp.testing.FakeClientSession()
    api = types.SimpleNamespace(_session=session)
    client = Client(
        hass,
        entry_id="entry",
        dev_id="dev",
        api_client=api,
        coordinator=coordinator,
        session=session,
    )

    event = {
        "name": "data",
        "args": [
            [
                {"path": "", "body": {}},
                {
                    "path": "/api/v2/devs/dev/mgr/nodes",
                    "body": {"nodes": [{"type": "htr", "addr": "1"}]},
                },
            ]
        ],
    }

    client._handle_event(event)
    dev_state = coordinator.data["dev"]
    assert dev_state["htr"]["addrs"] == ["1"]
    assert "htr" in call_record

    loop.close()


def test_parse_handshake_body_defaults() -> None:
    module = _load_ws_client()
    session = module.aiohttp.testing.FakeClientSession()
    loop = types.SimpleNamespace(create_task=lambda *_args, **_kwargs: None)
    hass = types.SimpleNamespace(
        loop=loop,
        data={module.DOMAIN: {"entry": {"ws_state": {}}}},
    )
    coordinator = types.SimpleNamespace()
    api = types.SimpleNamespace(_session=session)
    client = module.WebSocket09Client(
        hass,
        entry_id="entry",
        dev_id="dev",
        api_client=api,
        coordinator=coordinator,
        session=session,
    )
    sid, hb = client._parse_handshake_body("sid:not-a-number")
    assert sid == "sid"
    assert hb == 60


def test_parse_handshake_body_invalid() -> None:
    module = _load_ws_client()
    session = module.aiohttp.testing.FakeClientSession()
    loop = types.SimpleNamespace(create_task=lambda *_args, **_kwargs: None)
    hass = types.SimpleNamespace(
        loop=loop,
        data={module.DOMAIN: {"entry": {"ws_state": {}}}},
    )
    coordinator = types.SimpleNamespace()
    api = types.SimpleNamespace(_session=session)
    client = module.WebSocket09Client(
        hass,
        entry_id="entry",
        dev_id="dev",
        api_client=api,
        coordinator=coordinator,
        session=session,
    )

    with pytest.raises(RuntimeError):
        client._parse_handshake_body("invalid")


def test_send_text_no_ws() -> None:
    module = _load_ws_client()
    loop = types.SimpleNamespace(create_task=lambda *_args, **_kwargs: None)
    hass = types.SimpleNamespace(loop=loop, data={})
    coordinator = types.SimpleNamespace()
    api = types.SimpleNamespace(_session=None)
    client = module.WebSocket09Client(
        hass,
        entry_id="entry",
        dev_id="dev",
        api_client=api,
        coordinator=coordinator,
    )
    asyncio.run(client._send_text("data"))


def test_force_refresh_token_handles_missing_attribute() -> None:
    async def _run() -> None:
        module = _load_ws_client()
        hass = types.SimpleNamespace(loop=asyncio.get_event_loop(), data={})
        coordinator = types.SimpleNamespace()
        session = module.aiohttp.testing.FakeClientSession()
        client = module.WebSocket09Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=types.SimpleNamespace(_session=session),
            coordinator=coordinator,
            session=session,
        )

        class TokenClient:
            def __init__(self) -> None:
                self.calls = 0

            async def _ensure_token(self) -> None:
                self.calls += 1

            def __setattr__(self, name: str, value: Any) -> None:
                if name == "_access_token":
                    raise RuntimeError("fail")
                object.__setattr__(self, name, value)

        token_client = TokenClient()
        client._client = token_client  # type: ignore[assignment]
        await client._force_refresh_token()
        assert token_client.calls == 1

    asyncio.run(_run())


def test_api_base_fallback_to_default() -> None:
    module = _load_ws_client()
    loop = types.SimpleNamespace(create_task=lambda *_args, **_kwargs: None)
    hass = types.SimpleNamespace(loop=loop, data={})
    coordinator = types.SimpleNamespace()
    client = module.WebSocket09Client(
        hass,
        entry_id="entry",
        dev_id="dev",
        api_client=types.SimpleNamespace(_session=None),
        coordinator=coordinator,
    )
    client._client = types.SimpleNamespace(api_base="")  # type: ignore[assignment]
    assert client._api_base() == module.API_BASE


def test_api_base_strips_trailing_slash() -> None:
    module = _load_ws_client()
    loop = types.SimpleNamespace(create_task=lambda *_args, **_kwargs: None)
    hass = types.SimpleNamespace(loop=loop, data={})
    coordinator = types.SimpleNamespace()
    client = module.WebSocket09Client(
        hass,
        entry_id="entry",
        dev_id="dev",
        api_client=types.SimpleNamespace(_session=None),
        coordinator=coordinator,
    )
    client._client = types.SimpleNamespace(api_base="https://api.example.com/path/")  # type: ignore[assignment]
    assert client._api_base() == "https://api.example.com/path"


def test_runner_cleanup_handles_ws_close_errors() -> None:
    async def _run() -> None:
        module = _load_ws_client()
        session = module.aiohttp.testing.FakeClientSession()
        loop = asyncio.get_event_loop()
        hass = types.SimpleNamespace(
            loop=loop,
            data={module.DOMAIN: {"entry": {"ws_state": {}}}},
        )
        coordinator = types.SimpleNamespace()
        api = types.SimpleNamespace(
            _session=session,
            _authed_headers=AsyncMock(
                return_value={"Authorization": "Bearer cached"}
            ),
            _ensure_token=AsyncMock(),
        )

        client = module.WebSocket09Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=api,
            coordinator=coordinator,
            session=session,
        )

        class BoomWS:
            def __init__(self) -> None:
                self.close_calls = 0

            async def close(self) -> None:
                self.close_calls += 1
                raise RuntimeError("close fail")

            def exception(self) -> BaseException | None:
                return None

        boom_ws = BoomWS()

        async def fake_connect(_sid: str) -> None:
            client._ws = boom_ws

        client._connect_ws = fake_connect  # type: ignore[assignment]
        client._join_namespace = AsyncMock()
        client._send_snapshot_request = AsyncMock()
        client._subscribe_htr_samples = AsyncMock()
        client._heartbeat_loop = AsyncMock()

        async def fake_read_loop() -> None:
            client._closing = True
            raise RuntimeError("read fail")

        client._read_loop = fake_read_loop  # type: ignore[assignment]
        client._handshake = AsyncMock(return_value=("sid", 15))

        await client._runner()

        assert boom_ws.close_calls == 1

    asyncio.run(_run())


def test_handle_event_basic_validation() -> None:
    module = _load_ws_client()
    session = module.aiohttp.testing.FakeClientSession()
    loop = types.SimpleNamespace(create_task=lambda *_args, **_kwargs: None)
    coordinator = types.SimpleNamespace(data={}, _addrs=lambda: [])
    hass = types.SimpleNamespace(
        loop=loop,
        data={module.DOMAIN: {"entry": {"ws_state": {}}}},
    )
    api = types.SimpleNamespace(_session=session)
    client = module.WebSocket09Client(
        hass,
        entry_id="entry",
        dev_id="dev",
        api_client=api,
        coordinator=coordinator,
        session=session,
    )

    client._handle_event(None)
    client._handle_event({"name": "noop", "args": []})
    client._handle_event({"name": "data", "args": "bad"})
    client._handle_event({"name": "data", "args": ["bad"]})
    client._handle_event({"name": "data", "args": [[123]]})
    client._handle_event({"name": "data", "args": [[{"path": None, "body": {}}]]})

    assert coordinator.data == {}


def test_parse_handshake_body_requires_two_parts() -> None:
    module = _load_ws_client()
    session = module.aiohttp.testing.FakeClientSession()
    loop = types.SimpleNamespace(create_task=lambda *_args, **_kwargs: None)
    hass = types.SimpleNamespace(
        loop=loop,
        data={module.DOMAIN: {"entry": {"ws_state": {}}}},
    )
    coordinator = types.SimpleNamespace()
    api = types.SimpleNamespace(_session=session)
    client = module.WebSocket09Client(
        hass,
        entry_id="entry",
        dev_id="dev",
        api_client=api,
        coordinator=coordinator,
        session=session,
    )

    with pytest.raises(RuntimeError):
        client._parse_handshake_body("single")


def test_mark_event_unique_paths() -> None:
    module = _load_ws_client()
    loop = types.SimpleNamespace(create_task=lambda *_args, **_kwargs: None)
    hass = types.SimpleNamespace(
        loop=loop,
        data={module.DOMAIN: {"entry": {"ws_state": {}}}},
    )
    coordinator = types.SimpleNamespace()
    client = module.WebSocket09Client(
        hass,
        entry_id="entry",
        dev_id="dev",
        api_client=types.SimpleNamespace(_session=None),
        coordinator=coordinator,
    )
    old_level = module._LOGGER.level
    module._LOGGER.setLevel(logging.DEBUG)
    try:
        client._mark_event(paths=["/a", "/a", "/b", "/c", "/d", "/e"])
    finally:
        module._LOGGER.setLevel(old_level)
    assert client._stats.last_paths == ["/a", "/b", "/c", "/d", "/e"]


def test_detect_protocol_uses_hint_and_base() -> None:
    module = _load_ws_client()

    loop = types.SimpleNamespace(create_task=lambda *_args, **_kwargs: None)
    hass = types.SimpleNamespace(loop=loop, data={module.DOMAIN: {"entry": {}}})
    coordinator = types.SimpleNamespace()
    client = module.WebSocketClient(
        hass,
        entry_id="entry",
        dev_id="dev",
        api_client=types.SimpleNamespace(api_base="https://api-tevolve.example", _session=None),
        coordinator=coordinator,
    )

    assert client._detect_protocol() == "engineio2"

    client._protocol_hint = "socketio09"
    assert client._detect_protocol() == "socketio09"


def test_engineio_ws_client_flow(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    module = _load_ws_client()

    dispatcher_calls: list[tuple[str, dict[str, object]]] = []

    def fake_dispatcher(hass, signal: str, payload: dict[str, object]) -> None:
        dispatcher_calls.append((signal, payload))

    monkeypatch.setattr(module, "async_dispatcher_send", fake_dispatcher)
    monkeypatch.setattr(ws_core, "async_dispatcher_send", fake_dispatcher)

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        caplog.set_level(logging.DEBUG, logger=module.__name__)

        node_updates: list[tuple[dict[str, Any], list[Any]]] = []
        energy_updates: list[Any] = []

        class RecordingCoordinator:
            def update_nodes(self, nodes: dict[str, Any], inventory: list[Any]) -> None:
                node_updates.append((copy.deepcopy(nodes), list(inventory)))

        class FakeEnergyCoordinator:
            def update_addresses(self, addrs: Iterable[str] | dict[str, Iterable[str]]) -> None:
                if isinstance(addrs, dict):
                    energy_updates.append({k: list(v) for k, v in addrs.items()})
                else:
                    energy_updates.append(list(addrs))

        hass = types.SimpleNamespace(
            loop=loop,
            data={
                module.DOMAIN: {
                    "entry": {
                        "ws_state": {},
                        "energy_coordinator": FakeEnergyCoordinator(),
                    }
                }
            },
        )
        coordinator = RecordingCoordinator()

        class FakeClient:
            api_base = "https://api-tevolve.termoweb.net/"

            async def _authed_headers(self) -> dict[str, str]:
                return {"Authorization": "Bearer tok"}

        client = module.WebSocketClient(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=FakeClient(),
            coordinator=coordinator,
            protocol="engineio2",
        )

        assert client._detect_protocol() == "engineio2"
        client._protocol = "engineio2"
        client._update_status("starting")
        client._update_status("connecting")
        client._update_status("connected")

        handshake_payload = {
            "devs": [{"id": "dev"}],
            "permissions": {"dev": ["read"]},
        }
        client._on_frame(
            json.dumps({"event": "dev_handshake", "data": handshake_payload})
        )
        await asyncio.sleep(0)
        handshake_payload["devs"][0]["id"] = "mutated"
        assert client._handshake_payload is not None
        assert client._handshake_payload["devs"][0]["id"] == "dev"

        initial_update = {
            "nodes": {
                "htr": {"status": {"01": {"temp": 20}}},
                "acm": {"status": {"A1": {"mode": "eco"}}},
            }
        }
        client._on_frame(json.dumps({"event": "update", "data": initial_update}))
        await asyncio.sleep(0)

        dev_data_payload = {
            "nodes": {
                "htr": {"status": {"01": {"temp": 20}}},
                "acm": {"status": {"A1": {"mode": "eco"}}},
                "raw": {"meta": {"foo": "bar"}},
            }
        }
        client._on_frame(json.dumps({"event": "dev_data", "data": dev_data_payload}))
        await asyncio.sleep(0)

        incremental_update = {
            "nodes": {
                "htr": {"status": {"02": {"temp": 21}}},
                "acm": {"status": {"A2": {"mode": "boost"}}},
                "raw": {"meta": {"foo": "bar", "extra": True}},
                "metrics": 3,
            }
        }
        client._on_frame(json.dumps({"event": "update", "data": incremental_update}))
        await asyncio.sleep(0)

        client._on_frame(json.dumps({"event": "update", "data": None}))
        client._on_frame(json.dumps({"event": "unknown", "data": {}}))
        client._on_frame(json.dumps("literal"))
        client._on_frame("not-json")
        await asyncio.sleep(0)

        await client.stop()
        await asyncio.sleep(0)

        status_signal = module.signal_ws_status("entry")
        data_signal = module.signal_ws_data("entry")

        status_updates = [
            payload["status"]
            for signal, payload in dispatcher_calls
            if signal == status_signal
        ]
        assert status_updates[0] == "starting"
        assert "healthy" in status_updates
        assert status_updates[-1] == "stopped"

        data_payloads = [
            payload for signal, payload in dispatcher_calls if signal == data_signal
        ]
        assert len(data_payloads) == 3
        assert data_payloads[0]["nodes"] == initial_update["nodes"]
        assert data_payloads[1]["nodes"] == dev_data_payload["nodes"]
        assert data_payloads[2]["nodes"]["htr"]["status"] == {
            "01": {"temp": 20},
            "02": {"temp": 21},
        }
        assert data_payloads[2]["nodes"]["acm"]["status"] == {
            "A1": {"mode": "eco"},
            "A2": {"mode": "boost"},
        }
        assert data_payloads[2]["nodes"]["raw"] == {
            "meta": {"foo": "bar", "extra": True},
        }
        assert len(node_updates) == len(energy_updates) == 3
        assert node_updates[0][0] == initial_update["nodes"]
        assert node_updates[1][0] == dev_data_payload["nodes"]
        assert node_updates[2][0]["htr"]["status"]["02"]["temp"] == 21
        assert node_updates[2][0]["acm"]["status"]["A2"]["mode"] == "boost"
        assert all(set(update.keys()) == {"htr"} and not update["htr"] for update in energy_updates)
        assert client._nodes["nodes_by_type"]["htr"]["status"]["02"]["temp"] == 21
        assert client._nodes["nodes_by_type"]["acm"]["status"]["A2"]["mode"] == "boost"
        assert client._healthy_since is not None
        assert client._status == "stopped"
        assert "unknown node types" not in caplog.text

    asyncio.run(_run())


def test_engineio_logs_unknown_types(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    module = _load_ws_client()

    dispatcher_calls: list[tuple[str, dict[str, object]]] = []

    def fake_dispatcher(hass, signal: str, payload: dict[str, object]) -> None:
        dispatcher_calls.append((signal, payload))

    monkeypatch.setattr(module, "async_dispatcher_send", fake_dispatcher)
    monkeypatch.setattr(ws_core, "async_dispatcher_send", fake_dispatcher)

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        caplog.set_level(logging.DEBUG, logger=module.__name__)

        node_updates: list[tuple[dict[str, Any], list[Any]]] = []
        energy_updates: list[Any] = []

        class RecordingCoordinator:
            def update_nodes(self, nodes: dict[str, Any], inventory: list[Any]) -> None:
                node_updates.append((copy.deepcopy(nodes), list(inventory)))

        class FakeEnergyCoordinator:
            def update_addresses(self, addrs: Iterable[str] | dict[str, Iterable[str]]) -> None:
                if isinstance(addrs, dict):
                    energy_updates.append({k: list(v) for k, v in addrs.items()})
                else:
                    energy_updates.append(list(addrs))

        hass = types.SimpleNamespace(
            loop=loop,
            data={
                module.DOMAIN: {
                    "entry": {
                        "ws_state": {},
                        "energy_coordinator": FakeEnergyCoordinator(),
                    }
                }
            },
        )
        coordinator = RecordingCoordinator()

        class FakeClient:
            api_base = "https://api-tevolve.termoweb.net/"

            async def _authed_headers(self) -> dict[str, str]:
                return {"Authorization": "Bearer tok"}

        client = module.WebSocketClient(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=FakeClient(),
            coordinator=coordinator,
            protocol="engineio2",
        )

        client._protocol = "engineio2"
        payload = {
            "nodes": {
                "nodes": [
                    {"addr": "X", "type": "gizmo"},
                    {"addr": "01", "type": "htr"},
                ]
            }
        }
        client._handle_dev_data(payload)
        await asyncio.sleep(0)

        assert any(
            "unknown node types in inventory: gizmo" in rec.message
            for rec in caplog.records
        )
        assert energy_updates == [{"htr": ["01"]}]
        assert node_updates

    asyncio.run(_run())


def test_engineio_stop_handles_cancelled_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_ws_client()

    dispatcher_calls: list[tuple[str, dict[str, object]]] = []

    def fake_dispatcher(hass, signal: str, payload: dict[str, object]) -> None:
        dispatcher_calls.append((signal, payload))

    monkeypatch.setattr(module, "async_dispatcher_send", fake_dispatcher)
    monkeypatch.setattr(ws_core, "async_dispatcher_send", fake_dispatcher)

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        hass = types.SimpleNamespace(loop=loop, data={})

        class FakeClient:
            api_base = "https://api-tevolve.termoweb.net/"

            async def _authed_headers(self) -> dict[str, str]:
                return {"Authorization": "Bearer token"}

        client = module.WebSocketClient(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=FakeClient(),
            coordinator=types.SimpleNamespace(),
            protocol="engineio2",
        )

        async def hanging_runner(self: module.WebSocketClient) -> None:
            self._protocol = "engineio2"
            self._update_status("connecting")
            try:
                await asyncio.Future()
            finally:
                self._update_status("stopped")

        client._runner = types.MethodType(hanging_runner, client)

        task = client.start()
        await asyncio.sleep(0)
        assert not task.done()

        task.cancel()
        await asyncio.sleep(0)

        await client.stop()
        await asyncio.sleep(0)

        assert client._task is None
        status_signal = module.signal_ws_status("entry")
        status_updates = [
            payload["status"]
            for signal, payload in dispatcher_calls
            if signal == status_signal
        ]
        assert status_updates[-1] == "stopped"

    asyncio.run(_run())


def test_engineio_ws_url_requires_token() -> None:
    module = _load_ws_client()

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        hass = types.SimpleNamespace(
            loop=loop,
            data={module.DOMAIN: {"entry": {"ws_state": {}}}},
        )
        coordinator = types.SimpleNamespace()

        class NoTokenClient:
            api_base = "https://api-tevolve.termoweb.net/"

            async def _authed_headers(self) -> dict[str, str]:
                return {}

        client = module.WebSocketClient(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=NoTokenClient(),
            coordinator=coordinator,
            protocol="engineio2",
        )

        with pytest.raises(RuntimeError):
            await client.ws_url()

    asyncio.run(_run())
