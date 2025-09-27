from __future__ import annotations

import asyncio
import json
import logging
import types
from typing import Any, Iterable
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from conftest import _install_stubs

_install_stubs()

import custom_components.termoweb.ws_client as ws_core

ws_client_legacy = ws_core


def _load_ws_client(
    *,
    get_responses: Iterable[Any] | None = None,
    ws_connect_results: Iterable[Any] | None = None,
):
    _install_stubs()
    module = ws_client_legacy
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


def test_stop_handles_exceptions_and_updates_status() -> None:
    async def _run() -> None:
        module = _load_ws_client()

        hass = types.SimpleNamespace(
            loop=asyncio.get_event_loop(),
            data={module.DOMAIN: {"entry": {}}},
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
        ws = ExplodingWS()
        client._ws = ws
        client._task = asyncio.create_task(asyncio.sleep(0.1))

        await client.stop()

        assert update_calls[-1] == "stopped"
        assert client._hb_task is None
        assert client._ws is None
        assert client._task is None
        assert ws.calls == 1

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
        assert len(data_payloads) == 4
        ts = client._stats.last_event_ts
        assert {
            (p["kind"], p["addr"])
            for p in data_payloads
        } == {
            ("nodes", None),
            ("htr_settings", "01"),
            ("htr_settings", "02"),
            ("htr_samples", "02"),
        }
        for payload in data_payloads:
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
        assert client._stats.last_event_ts == 1000.0
        assert client._healthy_since == 1000.0
        assert updates == ["healthy"]

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


def test_handshake_status_error_raises_handshake_error() -> None:
    async def _run() -> None:
        module = _load_ws_client()
        aiohttp = module.aiohttp
        session = aiohttp.ClientSession(get_responses=[{"status": 403, "body": "denied"}])

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
        assert err.status == 403
        assert err.body_snippet == "denied"
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
    assert module.async_dispatcher_send.call_count == 4

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
    assert energy_updates == [{"gizmo": ["09"], "htr": ["01"]}]
    assert node_updates
    loop.close()


def test_subscribe_htr_samples_sends_expected_payloads():
    async def _run() -> None:
        module = _load_ws_client()
        Client = module.WebSocket09Client
        energy_updates: list[Any] = []

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
            '5::/api/v2/socket_io:{"name":"subscribe","args":["/acm/A1/samples"]}',
        ]

        dev_map = coordinator.data["dev"]
        assert dev_map["nodes_by_type"]["acm"]["addrs"] == ["A1"]
        assert dev_map["htr"] is dev_map["nodes_by_type"]["htr"]
        assert energy_updates == [{"htr": ["01", "02"], "acm": ["A1"]}]

    asyncio.run(_run())


def test_subscribe_htr_samples_returns_when_no_addresses():
    async def _run() -> None:
        module = _load_ws_client()
        Client = module.WebSocket09Client

        hass = types.SimpleNamespace(
            loop=asyncio.get_event_loop(), data={module.DOMAIN: {"entry": {}}}
        )
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

        def fake_addresses(inventory: list[Any], known_types: Any = None) -> tuple[dict[str, Any], set[str]]:
            mapping: dict[str, Any] = {"htr": [node.addr for node in inventory if getattr(node, "type", "") == "htr"]}
            mapping["acm"] = TruthyEmpty()
            return mapping, set()

        monkeypatch.setattr(module, "addresses_by_node_type", fake_addresses)

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

        class Stage1List(list):
            def __bool__(self) -> bool:
                return True

        class Stage2List(list):
            def __bool__(self) -> bool:
                return False

        orig_list = list

        class StageAwareList(list):
            def __new__(cls, iterable=()):  # type: ignore[override]
                if isinstance(iterable, Stage1Sequence):
                    return Stage1List(orig_list(iterable))
                if isinstance(iterable, Stage1List):
                    return Stage2List(orig_list(iterable))
                return list.__new__(cls, orig_list(iterable))

        monkeypatch.setitem(module.__dict__, "list", StageAwareList)

        def fake_addresses(inventory: list[Any], known_types: Any = None) -> tuple[dict[str, Any], set[str]]:
            mapping: dict[str, Any] = {"htr": [node.addr for node in inventory if getattr(node, "type", "") == "htr"]}
            mapping["acm"] = Stage1Sequence()
            return mapping, set()

        monkeypatch.setattr(module, "addresses_by_node_type", fake_addresses)

        hass = types.SimpleNamespace(
            loop=asyncio.get_event_loop(),
            data={module.DOMAIN: {"entry": {"energy_coordinator": types.SimpleNamespace(update_addresses=MagicMock())}}},
        )
        coordinator = types.SimpleNamespace(
            _node_inventory=module.list(
                module.build_node_inventory({"nodes": [{"type": "htr", "addr": "01"}]})
            ),
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


def test_handle_event_adds_missing_htr_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_ws_client()
    Client = module.WebSocket09Client

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


def test_handshake_status_error_raises_handshake_error() -> None:
    async def _run() -> None:
        module = _load_ws_client(get_responses=[{"status": 503, "body": "oops"}])
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

        with pytest.raises(module.HandshakeError) as err:
            await client._handshake()

        assert err.value.status == 503

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
