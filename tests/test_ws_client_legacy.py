from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, Iterable
from unittest.mock import AsyncMock, MagicMock, call

import pytest

WS_CLIENT_PATH = (
    Path(__file__).resolve().parents[1]
    / "custom_components"
    / "termoweb"
    / "ws_client_legacy.py"
)


def _load_ws_client(
    *,
    get_responses: Iterable[Any] | None = None,
    ws_connect_results: Iterable[Any] | None = None,
):
    package = "custom_components.termoweb"
    sys.modules.setdefault("custom_components", types.ModuleType("custom_components"))
    termoweb_pkg = types.ModuleType(package)
    termoweb_pkg.__path__ = [str(WS_CLIENT_PATH.parent)]
    sys.modules[package] = termoweb_pkg

    sys.modules.pop(f"{package}.ws_client_legacy", None)

    ha = types.ModuleType("homeassistant")
    ha_core = types.ModuleType("homeassistant.core")
    ha_core.HomeAssistant = object  # type: ignore[attr-defined]
    ha_helpers = types.ModuleType("homeassistant.helpers")
    ha_dispatcher = types.ModuleType("homeassistant.helpers.dispatcher")

    def _send(*args, **kwargs):
        return None

    ha_dispatcher.async_dispatcher_send = _send
    sys.modules["homeassistant"] = ha
    sys.modules["homeassistant.core"] = ha_core
    sys.modules["homeassistant.helpers"] = ha_helpers
    sys.modules["homeassistant.helpers.dispatcher"] = ha_dispatcher

    aiohttp_stub = types.ModuleType("aiohttp")

    class WSMsgType:
        TEXT = 1
        BINARY = 2
        CLOSED = 3
        CLOSE = 4
        ERROR = 5

    aiohttp_stub.WSMsgType = WSMsgType

    class FakeClientTimeout:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    aiohttp_stub.ClientTimeout = FakeClientTimeout

    class ClientError(Exception):
        pass

    aiohttp_stub.ClientError = ClientError
    aiohttp_stub.WSCloseCode = types.SimpleNamespace(GOING_AWAY=1001)

    default_get_script = list(get_responses or [])
    default_ws_script = list(ws_connect_results or [])

    class FakeHTTPResponse:
        def __init__(
            self,
            status: int,
            body: Any,
            *,
            headers: dict[str, Any] | None = None,
        ) -> None:
            self.status = status
            self._body = body
            self.headers = headers or {}

        async def text(self) -> str:
            body = self._body
            if asyncio.iscoroutine(body):
                body = await body
            if callable(body):
                body = body()
            if isinstance(body, bytes):
                return body.decode("utf-8", "ignore")
            return str(body or "")

    class FakeGetContext:
        def __init__(self, response: FakeHTTPResponse) -> None:
            self._response = response

        async def __aenter__(self) -> FakeHTTPResponse:
            return self._response

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            return False

    def _coerce_response(entry: Any) -> FakeHTTPResponse:
        if isinstance(entry, FakeHTTPResponse):
            return entry
        if isinstance(entry, tuple):
            status = int(entry[0])
            body = entry[1] if len(entry) > 1 else ""
            headers = entry[2] if len(entry) > 2 else None
            return FakeHTTPResponse(status, body, headers=headers)
        if isinstance(entry, dict):
            return FakeHTTPResponse(
                int(entry.get("status", 200)),
                entry.get("body", ""),
                headers=entry.get("headers"),
            )
        return FakeHTTPResponse(200, entry)

    class FakeWebSocket:
        def __init__(self, messages: Iterable[Any] | None = None) -> None:
            self._messages = list(messages or [])
            self.sent: list[str] = []
            self.close_code: int | None = None
            self._exception: BaseException | None = None

        def queue_message(self, message: Any) -> None:
            self._messages.append(message)

        async def receive(self) -> Any:
            if not self._messages:
                return types.SimpleNamespace(
                    type=aiohttp_stub.WSMsgType.CLOSED, data=None, extra=None
                )
            msg = self._messages.pop(0)
            if callable(msg):
                msg = msg()
            if asyncio.iscoroutine(msg):
                msg = await msg
            if isinstance(msg, dict):
                return types.SimpleNamespace(
                    type=msg.get("type", aiohttp_stub.WSMsgType.TEXT),
                    data=msg.get("data"),
                    extra=msg.get("extra"),
                )
            return msg

        async def send_str(self, data: str) -> None:
            self.sent.append(data)

        async def close(self, code: int | None = None, message: bytes | None = None) -> None:
            self.close_code = code

        def exception(self) -> BaseException | None:
            return self._exception

        def set_exception(self, exc: BaseException | None) -> None:
            self._exception = exc

    class FakeClientSession:
        def __init__(
            self,
            *,
            get_responses: Iterable[Any] | None = None,
            ws_connect_results: Iterable[Any] | None = None,
        ) -> None:
            self._get_script = list(get_responses or default_get_script)
            self._ws_script = list(ws_connect_results or default_ws_script)
            self.get_calls: list[dict[str, Any]] = []
            self.ws_connect_calls: list[dict[str, Any]] = []

        def queue_get(self, response: Any) -> None:
            self._get_script.append(response)

        def queue_ws(self, result: Any) -> None:
            self._ws_script.append(result)

        def get(self, url: str, *, timeout: Any | None = None) -> FakeGetContext:
            if not self._get_script:
                raise AssertionError("No scripted GET response available")
            entry = self._get_script.pop(0)
            if callable(entry):
                entry = entry(url=url, timeout=timeout)
            response = _coerce_response(entry)
            self.get_calls.append({"url": url, "timeout": timeout})
            return FakeGetContext(response)

        async def ws_connect(self, url: str, **kwargs: Any) -> Any:
            if not self._ws_script:
                raise AssertionError("No scripted ws_connect result available")
            entry = self._ws_script.pop(0)
            if callable(entry):
                entry = entry(url=url, **kwargs)
            if asyncio.iscoroutine(entry):
                entry = await entry
            if isinstance(entry, dict) and "messages" in entry:
                entry = FakeWebSocket(entry["messages"])
            self.ws_connect_calls.append({"url": url, "kwargs": kwargs})
            return entry

    aiohttp_stub.ClientSession = FakeClientSession
    aiohttp_stub.ClientWebSocketResponse = FakeWebSocket
    aiohttp_stub.testing = types.SimpleNamespace(
        FakeClientSession=FakeClientSession,
        FakeWebSocket=FakeWebSocket,
        FakeHTTPResponse=FakeHTTPResponse,
    )
    sys.modules["aiohttp"] = aiohttp_stub

    spec = importlib.util.spec_from_file_location(
        f"{package}.ws_client_legacy", WS_CLIENT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[f"{package}.ws_client_legacy"] = module
    spec.loader.exec_module(module)
    return module


def test_runner_retries_handshake_and_resets_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        module = _load_ws_client()
        aiohttp = sys.modules["aiohttp"]

        session = aiohttp.ClientSession(
            get_responses=[
                {"status": 500, "body": "fail"},
                {"status": 200, "body": "abc123:25:60:websocket"},
            ]
        )

        loop = asyncio.get_event_loop()
        hass = types.SimpleNamespace(loop=loop, data={module.DOMAIN: {"entry": {}}})
        coordinator = types.SimpleNamespace()
        api = types.SimpleNamespace(
            _session=session,
            _authed_headers=AsyncMock(
                side_effect=[
                    {"Authorization": "Bearer old"},
                    {"Authorization": "Bearer new"},
                ]
            ),
            _ensure_token=AsyncMock(),
        )

        client = module.TermoWebWSLegacyClient(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=api,
            coordinator=coordinator,
            session=session,
        )

        orig_update = client._update_status
        statuses: list[str] = []

        def capture_update(self, status: str) -> None:
            statuses.append(status)
            orig_update(status)

        client._update_status = types.MethodType(capture_update, client)

        sleeps: list[tuple[float, int]] = []

        async def fake_sleep(delay: float) -> None:
            sleeps.append((delay, client._backoff_idx))

        monkeypatch.setattr(module.asyncio, "sleep", fake_sleep)
        monkeypatch.setattr(module.random, "uniform", lambda a, b: 1.0)

        connect_backoff: list[int] = []

        async def fake_connect(sid: str) -> None:
            connect_backoff.append(client._backoff_idx)

        client._connect_ws = AsyncMock(side_effect=fake_connect)
        client._join_namespace = AsyncMock(return_value=None)
        client._send_snapshot_request = AsyncMock(return_value=None)
        client._subscribe_htr_samples = AsyncMock(return_value=None)
        client._heartbeat_loop = AsyncMock(return_value=None)
        client._read_loop = AsyncMock(side_effect=asyncio.CancelledError())

        await client._runner()

        assert len(sleeps) == 1
        assert sleeps[0] == (5.0, 1)
        assert client._connect_ws.await_args == call("abc123")
        assert api._authed_headers.await_count == 2
        assert connect_backoff == [0]
        assert client._backoff_idx == 0
        assert statuses.count("disconnected") >= 1

    asyncio.run(_run())


def test_read_loop_bubbles_exception_on_close():
    async def _run() -> None:
        module = _load_ws_client()
        Client = module.TermoWebWSLegacyClient
        hass = types.SimpleNamespace(loop=asyncio.get_event_loop())
        api = types.SimpleNamespace(_session=None)
        coordinator = types.SimpleNamespace()
        client = Client(hass, entry_id="e", dev_id="d", api_client=api, coordinator=coordinator)

        aiohttp = sys.modules["aiohttp"]

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


def test_connect_ws_uses_secure_endpoint() -> None:
    async def _run() -> None:
        module = _load_ws_client()
        aiohttp = sys.modules["aiohttp"]
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

        client = module.TermoWebWSLegacyClient(
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


def test_handshake_refreshes_token_after_401():
    async def _run() -> None:
        module = _load_ws_client()
        Client = module.TermoWebWSLegacyClient
        aiohttp = sys.modules["aiohttp"]
        session = aiohttp.ClientSession(
            get_responses=[
                {"status": 401, "body": "unauthorized"},
                {"status": 200, "body": "abc123:25:60:websocket"},
            ]
        )
        hass = types.SimpleNamespace(loop=asyncio.get_event_loop())
        coordinator = types.SimpleNamespace()
        api = types.SimpleNamespace(
            _session=session,
            _authed_headers=AsyncMock(
                side_effect=
                [
                    {"Authorization": "Bearer old"},
                    {"Authorization": "Bearer new"},
                ]
            ),
            _ensure_token=AsyncMock(),
        )
        client = Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=api,
            coordinator=coordinator,
            session=session,
        )
        client._force_refresh_token = AsyncMock(return_value=None)
        client._backoff_idx = 3

        sid, hb = await client._handshake()

        assert sid == "abc123"
        assert hb == 25
        assert client._force_refresh_token.await_count == 1
        assert api._authed_headers.await_count == 2
        assert len(session.get_calls) == 2
        assert client._backoff_idx == 0

    asyncio.run(_run())


def test_read_loop_handles_frames_and_disconnect() -> None:
    async def _run() -> None:
        module = _load_ws_client()
        Client = module.TermoWebWSLegacyClient
        hass = types.SimpleNamespace(loop=asyncio.get_event_loop())
        api = types.SimpleNamespace(_session=None)
        coordinator = types.SimpleNamespace()
        client = Client(hass, entry_id="e", dev_id="d", api_client=api, coordinator=coordinator)

        aiohttp = sys.modules["aiohttp"]
        ws = aiohttp.testing.FakeWebSocket(
            messages=[
                {"type": aiohttp.WSMsgType.TEXT, "data": "2::"},
                {"type": aiohttp.WSMsgType.TEXT, "data": "5::/api/v2/socket_io"},
                {
                    "type": aiohttp.WSMsgType.TEXT,
                    "data": f"5::{module.WS_NAMESPACE}:not-json",
                },
                {"type": aiohttp.WSMsgType.TEXT, "data": "0::"},
            ]
        )
        client._ws = ws
        mark = MagicMock()
        handle = MagicMock()
        client._mark_event = mark
        client._handle_event = handle

        with pytest.raises(RuntimeError, match="server disconnect"):
            await client._read_loop()

        mark.assert_any_call(paths=None)
        assert handle.call_count == 0
        assert client._stats.frames_total == 4

    asyncio.run(_run())


def test_handle_event_updates_state_and_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_ws_client()
    module.async_dispatcher_send = MagicMock()
    loop = asyncio.new_event_loop()
    hass = types.SimpleNamespace(
        loop=loop,
        data={module.DOMAIN: {"entry": {"ws_state": {}}}},
    )
    coordinator = types.SimpleNamespace(
        data={
            "dev": {
                "dev_id": "dev",
                "name": "Device dev",
                "raw": {"existing": True},
                "connected": True,
                "nodes": None,
                "htr": {"addrs": [], "settings": {}},
            }
        }
    )
    api = types.SimpleNamespace(_session=types.SimpleNamespace())
    client = module.TermoWebWSLegacyClient(
        hass,
        entry_id="entry",
        dev_id="dev",
        api_client=api,
        coordinator=coordinator,
    )
    monkeypatch.setattr(module.time, "time", lambda: 1000.0)

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
                        ]
                    },
                },
                {"path": "/htr/01/settings", "body": {"temp": 21}},
                {"path": "/htr/01/advanced_setup", "body": {"adv": True}},
                {
                    "path": "/htr/02/samples",
                    "body": [{"ts": 1, "val": 2}],
                },
                {"path": "/misc", "body": {"foo": "bar"}},
            ]
        ],
    }

    client._handle_event(event)

    dev_data = coordinator.data["dev"]
    assert dev_data["nodes"] == {
        "nodes": [{"addr": "01", "type": "htr"}, {"addr": "02", "type": "HTR"}]
    }
    assert dev_data["htr"]["addrs"] == ["01", "02"]
    assert dev_data["htr"]["settings"]["01"] == {"temp": 21}
    assert dev_data["htr"]["advanced"]["01"] == {"adv": True}
    assert dev_data["raw"]["misc"] == {"foo": "bar"}
    assert client._stats.events_total == 1
    module.async_dispatcher_send.assert_has_calls(
        [
            call(
                hass,
                module.signal_ws_data("entry"),
                {"dev_id": "dev", "ts": 1000.0, "addr": None, "kind": "nodes"},
            ),
            call(
                hass,
                module.signal_ws_data("entry"),
                {"dev_id": "dev", "ts": 1000.0, "addr": "01", "kind": "htr_settings"},
            ),
            call(
                hass,
                module.signal_ws_data("entry"),
                {"dev_id": "dev", "ts": 1000.0, "addr": "02", "kind": "htr_samples"},
            ),
        ]
    )
    assert module.async_dispatcher_send.call_count == 3
    loop.close()


def test_subscribe_htr_samples_sends_expected_payloads():
    async def _run() -> None:
        module = _load_ws_client()
        Client = module.TermoWebWSLegacyClient
        hass = types.SimpleNamespace(loop=asyncio.get_event_loop())
        coordinator = types.SimpleNamespace(_addrs=lambda: ["01", "02"])
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

        assert ws.sent == [
            '5::/api/v2/socket_io:{"name":"subscribe","args":["/htr/01/samples"]}',
            '5::/api/v2/socket_io:{"name":"subscribe","args":["/htr/02/samples"]}',
        ]

    asyncio.run(_run())


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
    client = module.TermoWebWSLegacyClient(
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


def test_heartbeat_loop_sends_until_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        module = _load_ws_client()
        Client = module.TermoWebWSLegacyClient
        loop = asyncio.get_event_loop()
        hass = types.SimpleNamespace(loop=loop, data={module.DOMAIN: {"entry": {}}})
        coordinator = types.SimpleNamespace()
        api = types.SimpleNamespace(_session=None)
        client = Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=api,
            coordinator=coordinator,
        )
        client._hb_send_interval = 1.0

        send_calls: list[str] = []
        send_event = asyncio.Event()

        async def fake_send(data: str) -> None:
            send_calls.append(data)
            if len(send_calls) >= 2:
                send_event.set()

        client._send_text = AsyncMock(side_effect=fake_send)

        orig_sleep = module.asyncio.sleep
        sleep_calls: list[float] = []

        async def fake_sleep(delay: float) -> None:
            sleep_calls.append(delay)
            await orig_sleep(0)

        monkeypatch.setattr(module.asyncio, "sleep", fake_sleep)

        task = asyncio.create_task(client._heartbeat_loop())
        await asyncio.wait_for(send_event.wait(), timeout=0.1)
        task.cancel()
        await task

        assert len(send_calls) >= 2
        assert all(payload == "2::" for payload in send_calls[:2])
        assert sleep_calls[:2] == [1.0, 1.0]

    asyncio.run(_run())
