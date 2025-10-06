"""Additional Ducaheat websocket client coverage tests."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from custom_components.termoweb.backend import ducaheat_ws


class DummyREST:
    """Provide the minimal interface required by the websocket client."""

    def __init__(self) -> None:
        self._session = SimpleNamespace()
        self._headers = {"Authorization": "Bearer rest-token"}

    async def authed_headers(self) -> dict[str, str]:
        """Return cached REST headers with an access token."""

        return self._headers


class StubResponse:
    """Async context manager returning a predetermined payload."""

    def __init__(self, *, status: int = 200, body: bytes = b"") -> None:
        self.status = status
        self._body = body
        self._read = AsyncMock(return_value=body)

    async def read(self) -> bytes:
        """Return the response body."""

        return await self._read()

    async def __aenter__(self) -> "StubResponse":
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None


class StubWebSocket:
    """Simple websocket stub capturing sent frames."""

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.closed = False
        self._receive = asyncio.Queue[str]()
        self._receive.put_nowait("3probe")

    async def send_str(self, payload: str) -> None:
        """Record outgoing websocket frames."""

        self.sent.append(payload)

    async def receive_str(self) -> str:
        """Return the next queued incoming frame."""

        return await self._receive.get()

    async def close(self, *, code: int, message: bytes) -> None:
        """Record that the websocket has been closed."""

        self.closed = True


class StubSession:
    """Sequence driven aiohttp session stub for handshake operations."""

    def __init__(self, ws: StubWebSocket) -> None:
        self.calls: list[tuple[str, str]] = []
        self._ws = ws
        self._open_body = ducaheat_ws._encode_polling_packet(
            '0{"sid":"abc","pingInterval":25000,"pingTimeout":60000}'
        )

    def get(self, url: str, *, headers: dict[str, str]) -> StubResponse:
        """Return responses for the open and drain polling calls."""

        self.calls.append(("GET", url))
        if "sid=" not in url:
            return StubResponse(body=self._open_body)
        return StubResponse(body=b"6:40[]")

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        data: bytes,
    ) -> StubResponse:
        """Record POST invocations during handshake."""

        self.calls.append(("POST", url))
        return StubResponse(body=b"")

    async def ws_connect(self, url: str, **_: Any) -> StubWebSocket:
        """Return the preconfigured websocket stub."""

        self.calls.append(("WS", url))
        return self._ws


def _make_client(monkeypatch: pytest.MonkeyPatch) -> ducaheat_ws.DucaheatWSClient:
    """Create a websocket client with deterministic helpers."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    ws = StubWebSocket()
    session = StubSession(ws)
    hass = SimpleNamespace(loop=loop, data={ducaheat_ws.DOMAIN: {"entry": {}}})
    client = ducaheat_ws.DucaheatWSClient(
        hass,
        entry_id="entry",
        dev_id="device",
        api_client=DummyREST(),
        coordinator=SimpleNamespace(update_nodes=AsyncMock()),
        session=session,  # type: ignore[arg-type]
    )
    monkeypatch.setattr(client, "_get_token", AsyncMock(return_value="token"))
    monkeypatch.setattr(ducaheat_ws, "_rand_t", lambda: "P123456")
    monkeypatch.setattr(
        ducaheat_ws,
        "collect_heater_sample_addresses",
        lambda *_args, **_kwargs: ([], {"htr": ["1"]}, {}),
    )
    monkeypatch.setattr(
        ducaheat_ws,
        "normalize_heater_addresses",
        lambda mapping: (mapping, {}),
    )
    monkeypatch.setattr(
        ducaheat_ws,
        "heater_sample_subscription_targets",
        lambda mapping: [(kind, addr) for kind, addrs in mapping.items() for addr in addrs],
    )
    return client


@pytest.mark.asyncio
async def test_connect_once_performs_full_handshake(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exhaust the polling and websocket upgrade handshake."""

    statuses: list[str] = []
    monkeypatch.setattr(
        ducaheat_ws._WSCommon,
        "_update_status",
        lambda self, status: statuses.append(status),
    )
    monkeypatch.setattr(
        ducaheat_ws,
        "_decode_polling_packets",
        lambda body: ['0{"sid":"abc","pingInterval":25000,"pingTimeout":60000}'],
    )
    client = _make_client(monkeypatch)
    await client._connect_once()

    assert client._ws is not None
    assert statuses[-1] == "connected"
    assert "42/api/v2/socket_io,[\"dev_data\"]" in client._ws.sent
    assert client._ws.sent.count("3") == 0  # handshake should not issue pong during setup


def test_rand_t_token_format() -> None:
    """Random polling tokens should be eight alphanumeric characters with a P prefix."""

    token = ducaheat_ws._rand_t()

    assert token.startswith("P")
    assert len(token) == 8
    assert token[1:].isalnum()


def test_decode_polling_packets_additional_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise defensive Engine.IO polling decoder branches."""

    # Short buffers should trigger the early break condition.
    assert ducaheat_ws._decode_polling_packets(b"\x00\x00\x00") == []

    # Invalid digit bytes should abort parsing without raising.
    assert ducaheat_ws._decode_polling_packets(b"\x00\x0A") == []

    # Missing digit markers should also result in an empty decode.
    assert ducaheat_ws._decode_polling_packets(b"\x00\xFF\x00\x00") == []

    # Length overruns should be handled gracefully.
    assert ducaheat_ws._decode_polling_packets(b"\x00\x02\xFF\x00") == []

    class FailingPayload(bytes):
        def decode(self, *_: Any, **__: Any) -> str:  # pragma: no cover - exercised via test
            raise ValueError

    class ByteLike:
        def __init__(self, data: bytes) -> None:
            self._data = data

        def __len__(self) -> int:
            return len(self._data)

        def __getitem__(self, item: Any) -> Any:
            result = self._data[item]
            if isinstance(item, slice) and item.start and item.start >= 3:
                return FailingPayload(result)
            return result

    # Emulate a decode failure to reach the exception branch.
    assert ducaheat_ws._decode_polling_packets(ByteLike(b"\x00\x01\xFF\x00")) == []

    # Trigger gzip decompression fallback by raising from gzip.decompress.
    monkeypatch.setattr(ducaheat_ws, "gzip", SimpleNamespace(decompress=MagicMock(side_effect=OSError)))
    assert ducaheat_ws._decode_polling_packets(b"\x1f\x8bbad") == []


def test_brand_headers_apply_defaults() -> None:
    """Brand headers should retain required defaults when optional values are missing."""

    headers = ducaheat_ws._brand_headers("", "")

    assert headers["User-Agent"] == ducaheat_ws.USER_AGENT
    assert headers["X-Requested-With"] == ""
    assert headers["Accept-Language"] == ducaheat_ws.ACCEPT_LANGUAGE


def test_client_requires_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing without an aiohttp session should fail."""

    hass = SimpleNamespace(loop=asyncio.new_event_loop(), data={})
    with pytest.raises(RuntimeError):
        ducaheat_ws.DucaheatWSClient(
            hass,
            entry_id="entry",
            dev_id="device",
            api_client=SimpleNamespace(_session=None),  # type: ignore[arg-type]
            coordinator=SimpleNamespace(),
        )


@pytest.mark.asyncio
async def test_start_and_runner_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    """_runner should cycle through connect, read, and disconnect before stopping."""

    client = _make_client(monkeypatch)
    statuses: list[str] = []
    monkeypatch.setattr(client, "_disconnect", AsyncMock())
    monkeypatch.setattr(client, "_update_status", lambda status: statuses.append(status))

    async def _connect_once() -> None:
        statuses.append("connect_once")

    async def _read_loop() -> None:
        statuses.append("read_loop")
        raise asyncio.CancelledError

    monkeypatch.setattr(client, "_connect_once", AsyncMock(side_effect=_connect_once))
    monkeypatch.setattr(client, "_read_loop_ws", AsyncMock(side_effect=_read_loop))

    await client._runner()

    assert statuses[0] == "starting"
    assert "connect_once" in statuses
    assert "read_loop" in statuses
    assert statuses[-1] == "stopped"


@pytest.mark.asyncio
async def test_start_method_reuses_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """start should cache the created asyncio task."""

    client = _make_client(monkeypatch)
    monkeypatch.setattr(client, "_runner", AsyncMock(return_value=None))

    task_one = client.start()
    task_two = client.start()

    assert task_one is task_two
    await asyncio.wait_for(task_one, 0.1)


def test_is_running_reflects_task_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """is_running should track whether the background task is active."""

    client = _make_client(monkeypatch)
    monkeypatch.setattr(client, "_runner", AsyncMock(return_value=None))

    assert client.is_running() is False
    task = client.start()
    assert client.is_running() is True
    loop = client._loop
    loop.run_until_complete(asyncio.sleep(0))
    assert client.is_running() is False


@pytest.mark.asyncio
async def test_connect_once_open_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Different handshake failures should surface as HandshakeError instances."""

    async def run(
        expected_status: int,
        *,
        open_status: int = 200,
        post_status: int = 200,
        drain_status: int = 200,
        decode_return: list[str] | None = None,
        open_body: bytes | None = None,
    ) -> None:
        with monkeypatch.context() as patch:
            client = _make_client(patch)
            ws = client._session._ws  # type: ignore[attr-defined]
            call_count = 0

            if decode_return is not None:
                patch.setattr(ducaheat_ws, "_decode_polling_packets", lambda _body: decode_return)

            def fake_get(url: str, *, headers: dict[str, str]) -> StubResponse:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return StubResponse(status=open_status, body=open_body or client._session._open_body)  # type: ignore[attr-defined]
                return StubResponse(status=drain_status)

            def fake_post(
                url: str, *, headers: dict[str, str], data: bytes
            ) -> StubResponse:
                return StubResponse(status=post_status)

            patch.setattr(client._session, "get", fake_get)
            patch.setattr(client._session, "post", fake_post)
            patch.setattr(client._session, "ws_connect", AsyncMock(return_value=ws))

            with pytest.raises(ducaheat_ws.HandshakeError) as err:
                await client._connect_once()
            assert err.value.status == expected_status

    await run(403, open_status=403)
    await run(590, decode_return=[])
    await run(592, decode_return=['0{"pingInterval":1,"pingTimeout":1}'])
    await run(500, decode_return=['0{"sid":"abc"}'], post_status=500)
    await run(504, decode_return=['0{"sid":"abc"}'], drain_status=504)


@pytest.mark.asyncio
async def test_connect_once_probe_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unexpected probe acknowledgements should not abort the handshake."""

    statuses: list[str] = []
    monkeypatch.setattr(
        ducaheat_ws._WSCommon,
        "_update_status",
        lambda self, status: statuses.append(status),
    )
    client = _make_client(monkeypatch)
    monkeypatch.setattr(
        ducaheat_ws,
        "_decode_polling_packets",
        lambda _body: ['0{"sid":"abc"}'],
    )
    assert client._session._ws is not None  # type: ignore[attr-defined]
    client._session._ws._receive = asyncio.Queue()  # type: ignore[attr-defined]
    client._session._ws._receive.put_nowait("weird")  # type: ignore[attr-defined]

    await client._connect_once()

    assert statuses[-1] == "connected"
    assert any(frame == "5" for frame in client._ws.sent)  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_read_loop_additional_flows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise ping, namespace, and event handling branches."""

    client = _make_client(monkeypatch)
    send_history: list[str] = []

    class RichWS:
        def __init__(self) -> None:
            self.closed = False

        async def send_str(self, payload: str) -> None:
            send_history.append(payload)

        def __aiter__(self) -> Any:
            async def _iterate() -> AsyncIterator[Any]:
                yield SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="1ignore")
                yield SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="42")
                yield SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="42/api,[\"ping\"]")
                yield SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="442/invalid")
                yield SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="442invalid")
                yield SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="442[]")
                yield SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="442[\"message\",\"ping\"]")
                yield SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="442[\"other\",{}]")
                yield SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="440")
                yield SimpleNamespace(
                    type=aiohttp.WSMsgType.TEXT,
                    data="442[\"dev_data\",{\"nodes\":{\"htr\":{\"samples\":{\"1\":{}}}}}]",
                )
                yield SimpleNamespace(
                    type=aiohttp.WSMsgType.TEXT,
                    data="442[\"update\",{\"body\":{\"temp\":1},\"path\":\"/path\"}]",
                )
            return _iterate()

    monkeypatch.setattr(client, "_dispatch_nodes", MagicMock())
    emit_mock = AsyncMock()
    monkeypatch.setattr(client, "_emit_sio", emit_mock)
    updates: list[str] = []
    monkeypatch.setattr(client, "_update_status", lambda status: updates.append(status))
    client._ws = RichWS()  # type: ignore[assignment]

    await client._read_loop_ws()

    assert send_history.count("3/api") == 1
    assert any(call.args == ("message", "pong") for call in emit_mock.await_args_list)
    assert "healthy" in updates


@pytest.mark.asyncio
async def test_read_loop_handles_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Error frames should raise runtime exceptions."""

    client = _make_client(monkeypatch)

    class ErrorWS:
        def __init__(self) -> None:
            self.closed = False

        def __aiter__(self) -> Any:
            async def _iterate() -> AsyncIterator[Any]:
                yield SimpleNamespace(type=aiohttp.WSMsgType.ERROR, data=None)
            return _iterate()

        def exception(self) -> str:
            return "boom"

    client._ws = ErrorWS()  # type: ignore[assignment]

    with pytest.raises(RuntimeError):
        await client._read_loop_ws()


@pytest.mark.asyncio
async def test_read_loop_handles_close(monkeypatch: pytest.MonkeyPatch) -> None:
    """Close frames should surface as runtime errors."""

    client = _make_client(monkeypatch)

    class CloseWS:
        def __init__(self) -> None:
            self.closed = False

        def __aiter__(self) -> Any:
            async def _iterate() -> AsyncIterator[Any]:
                yield SimpleNamespace(type=aiohttp.WSMsgType.CLOSE)
            return _iterate()

    client._ws = CloseWS()  # type: ignore[assignment]

    with pytest.raises(RuntimeError):
        await client._read_loop_ws()


@pytest.mark.asyncio
async def test_read_loop_processes_update_event(monkeypatch: pytest.MonkeyPatch) -> None:
    """Update events should invoke the brief logger and mark the connection healthy."""

    client = _make_client(monkeypatch)
    log_calls: list[Any] = []
    monkeypatch.setattr(client, "_log_update_brief", lambda body: log_calls.append(body))
    statuses: list[str] = []
    monkeypatch.setattr(client, "_update_status", lambda status: statuses.append(status))

    class UpdateWS:
        def __init__(self) -> None:
            self.closed = False

        def __aiter__(self) -> Any:
            async def _iterate() -> AsyncIterator[Any]:
                yield SimpleNamespace(
                    type=aiohttp.WSMsgType.TEXT,
                    data="442[\"update\",{\"body\":{\"temp\":1},\"path\":\"/path\"}]",
                )
            return _iterate()

    client._ws = UpdateWS()  # type: ignore[assignment]

    await client._read_loop_ws()

    assert log_calls and log_calls[0]["body"]["temp"] == 1
    assert statuses and statuses[-1] == "healthy"


def test_log_nodes_summary_branches(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Ensure node summary logging honours the configured log level."""

    client = _make_client(monkeypatch)
    nodes = {"htr": {"samples": {"1": {}}, "settings": {"2": {}}, "status": 3}}

    monkeypatch.setattr(ducaheat_ws._LOGGER, "isEnabledFor", lambda level: False)
    client._log_nodes_summary(nodes)

    monkeypatch.setattr(ducaheat_ws._LOGGER, "isEnabledFor", lambda level: True)
    with caplog.at_level(logging.INFO):
        client._log_nodes_summary(nodes)

    assert "htr=2" in caplog.text


def test_log_update_brief_variants(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """_log_update_brief should handle mapping and non-mapping payloads."""

    client = _make_client(monkeypatch)
    monkeypatch.setattr(ducaheat_ws._LOGGER, "isEnabledFor", lambda level: True)

    with caplog.at_level(logging.DEBUG):
        client._log_update_brief({"path": "/x", "body": {"one": 1, "two": 2, "three": 3}})
        client._log_update_brief("not a mapping")

    assert "path=/x" in caplog.text


@pytest.mark.asyncio
async def test_emit_sio_requires_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Attempting to emit without a websocket should raise a runtime error."""

    client = _make_client(monkeypatch)
    client._ws = None

    with pytest.raises(RuntimeError):
        await client._emit_sio("evt")


@pytest.mark.asyncio
async def test_get_token_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """The access token should be extracted from the Authorization header."""

    client = _make_client(monkeypatch)
    client._get_token = ducaheat_ws.DucaheatWSClient._get_token.__get__(  # type: ignore[method-assign]
        client, ducaheat_ws.DucaheatWSClient
    )
    token = await client._get_token()

    assert token == "rest-token"


@pytest.mark.asyncio
async def test_read_loop_returns_when_websocket_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The websocket reader should no-op when no connection is bound."""

    client = _make_client(monkeypatch)
    client._ws = None

    await client._read_loop_ws()



@pytest.mark.asyncio
async def test_read_loop_handles_engineio_and_socketio_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Feed assorted Engine.IO frames through the websocket reader."""

    client = _make_client(monkeypatch)
    send_history: list[str] = []
    client._ws = SimpleNamespace(closed=False)

    class FakeWS:
        def __init__(self) -> None:
            self.closed = False

        async def send_str(self, payload: str) -> None:
            send_history.append(payload)

        def __aiter__(self) -> Any:
            async def _iterator() -> Any:
                yield SimpleNamespace(
                    type=aiohttp.WSMsgType.TEXT,
                    data="2",
                )
                yield SimpleNamespace(
                    type=aiohttp.WSMsgType.TEXT,
                    data="42[\"message\",\"ping\"]",
                )
                yield SimpleNamespace(
                    type=aiohttp.WSMsgType.TEXT,
                    data="40",
                )
                yield SimpleNamespace(
                    type=aiohttp.WSMsgType.TEXT,
                    data="42{\"evt\":\"ignored\"}",
                )
                yield SimpleNamespace(
                    type=aiohttp.WSMsgType.TEXT,
                    data="42[\"dev_data\",{\"nodes\":{\"htr\":{\"settings\":{\"1\":{}}}}}]",
                )
                yield SimpleNamespace(
                    type=aiohttp.WSMsgType.TEXT,
                    data="42[\"update\",{\"body\":{\"path\":{}}}]",
                )
            return _iterator()

    fake_ws = FakeWS()
    client._ws = fake_ws  # type: ignore[assignment]

    monkeypatch.setattr(client, "_dispatch_nodes", MagicMock())
    monkeypatch.setattr(client, "_emit_sio", AsyncMock())
    monkeypatch.setattr(client, "_update_status", lambda status: None)
    await client._read_loop_ws()

    assert any(frame == "3" for frame in send_history)


@pytest.mark.asyncio
async def test_disconnect_closes_websocket(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure _disconnect invokes the aiohttp close sequence."""

    client = _make_client(monkeypatch)
    close_called: dict[str, Any] = {}

    class ClosingWS:
        closed = False

        async def close(self, *, code: int, message: bytes) -> None:
            close_called["code"] = code
            close_called["message"] = message

    client._ws = ClosingWS()  # type: ignore[assignment]
    await client._disconnect("reason")

    assert close_called["code"] == aiohttp.WSCloseCode.GOING_AWAY
    assert close_called["message"] == b"reason"
    assert client._ws is None


@pytest.mark.asyncio
async def test_get_token_requires_authorization(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing Authorization header should raise a runtime error."""

    client = _make_client(monkeypatch)
    client._get_token = ducaheat_ws.DucaheatWSClient._get_token.__get__(  # type: ignore[method-assign]
        client, ducaheat_ws.DucaheatWSClient
    )
    monkeypatch.setattr(client._client, "authed_headers", AsyncMock(return_value={}))

    with pytest.raises(RuntimeError):
        await client._get_token()
