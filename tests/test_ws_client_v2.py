import asyncio
import types
from typing import Any
from unittest.mock import AsyncMock

import pytest

from conftest import _install_stubs

_install_stubs()

import custom_components.termoweb.ws_client_v2 as ws_client_v2  # noqa: E402


def _configure_defaults(*, get_responses: list[Any] | None = None, ws_results: list[Any] | None = None) -> None:
    testing = ws_client_v2.aiohttp.testing
    defaults = getattr(testing, "_defaults", None)
    if defaults is not None:
        defaults.get_responses = list(get_responses or [])
        defaults.ws_connect_results = list(ws_results or [])


def test_runner_performs_handshake_and_dispatches_events(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ws_client_v2
    _configure_defaults(
        get_responses=[
            {
                "status": 200,
                "body": '97:0{"sid":"abc","pingInterval":20000,"pingTimeout":50000}',
            }
        ],
        ws_results=[module.aiohttp.testing.FakeWebSocket()],
    )

    times = iter([1000.0, 1001.0, 1002.0, 1003.0])

    def fake_time() -> float:
        try:
            return next(times)
        except StopIteration:
            return 1003.0

    monkeypatch.setattr(module.time, "time", fake_time)

    async def _run() -> tuple[Any, Any, list[tuple[str, dict[str, Any]]], list[str]]:
        hass = types.SimpleNamespace(
            loop=asyncio.get_running_loop(),
            data={module.DOMAIN: {"entry": {"ws_state": {}}}},
        )
        coordinator = types.SimpleNamespace(data={})
        session = module.aiohttp.testing.FakeClientSession()
        original_get = session.get

        def patched_get(url: str, *, timeout=None, headers=None):
            return original_get(url, timeout=timeout)

        session.get = patched_get  # type: ignore[assignment]
        api = types.SimpleNamespace(
            _session=session,
            api_base="https://api.example.com",
            _authed_headers=AsyncMock(return_value={"Authorization": "Bearer tok"}),
            _ensure_token=AsyncMock(),
        )

        dispatch_calls: list[tuple[str, dict[str, Any]]] = []
        data_signal = module.signal_ws_data("entry")

        def fake_dispatch(hass_obj: Any, signal: str, payload: dict[str, Any]) -> None:
            dispatch_calls.append((signal, payload))

        monkeypatch.setattr(module, "async_dispatcher_send", fake_dispatch)

        client = module.TermoWebWSV2Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=api,
            coordinator=coordinator,
        )

        sid, ping_interval, ping_timeout = await client._handshake()
        assert sid == "abc"
        assert ping_interval == 20.0
        assert ping_timeout == 50.0

        await client._connect_ws(sid)
        await client._join_namespace()
        await client._request_snapshot()

        client._handle_event(
            "dev_handshake",
            [
                {
                    "devices": [
                        {
                            "dev_id": "dev",
                            "updates": [
                                {
                                    "path": "/mgr/nodes",
                                    "body": {"nodes": [{"addr": "A", "type": "htr"}]},
                                }
                            ],
                        }
                    ]
                }
            ],
        )

        ws_instance = getattr(client, "_ws", None)
        sent_frames = getattr(ws_instance, "sent", []) if ws_instance else []

        return session, coordinator, dispatch_calls, sent_frames

    session, coordinator, dispatch_calls, sent_frames = asyncio.run(_run())

    assert session.get_calls[0]["url"].startswith("https://api.example.com/api/v2/socket_io/")
    assert session.ws_connect_calls[0]["url"] == (
        "wss://api.example.com/api/v2/socket_io/?EIO=3&transport=websocket&sid=abc"
    )
    assert any(frame.startswith(f"40{module.WS_NAMESPACE}") for frame in sent_frames)
    assert any(
        frame.startswith(f'42{module.WS_NAMESPACE},["dev_data",{{"dev_id":"dev"}}]')
        for frame in sent_frames
    )

    assert "dev" in coordinator.data
    dev_state = coordinator.data["dev"]
    assert dev_state["nodes"]["nodes"][0]["addr"] == "A"

    data_signal = module.signal_ws_data("entry")
    data_events = [payload for signal, payload in dispatch_calls if signal == data_signal]
    assert data_events
    assert data_events[0]["kind"] == "nodes"


def test_handle_event_merges_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ws_client_v2
    monkeypatch.setattr(module.time, "time", lambda: 2000.0)

    hass = types.SimpleNamespace(
        loop=types.SimpleNamespace(create_task=lambda coro, name=None: coro),
        data={module.DOMAIN: {"entry": {"ws_state": {}}}},
    )
    coordinator = types.SimpleNamespace(
        data={"dev": {"dev_id": "dev", "htr": {"addrs": [], "settings": {}}}}
    )
    api = types.SimpleNamespace(
        _session=module.aiohttp.testing.FakeClientSession(),
        api_base="https://api.example.com",
        _authed_headers=AsyncMock(return_value={"Authorization": "Bearer tok"}),
        _ensure_token=AsyncMock(),
    )

    dispatch_calls: list[tuple[str, dict[str, Any]]] = []

    def fake_dispatch(hass_obj: Any, signal: str, payload: dict[str, Any]) -> None:
        dispatch_calls.append((signal, payload))

    monkeypatch.setattr(module, "async_dispatcher_send", fake_dispatch)

    client = module.TermoWebWSV2Client(
        hass,
        entry_id="entry",
        dev_id="dev",
        api_client=api,
        coordinator=coordinator,
    )

    updates = [
        {"path": "/mgr/nodes", "body": {"nodes": [{"addr": "A", "type": "htr"}]}},
        {"path": "/htr/A/settings", "body": {"mode": "auto"}},
        {"path": "/htr/A/advanced_setup", "body": {"boost": 15}},
        {"path": "/htr/A/samples", "body": [1, 2]},
        {"path": "/misc/info", "body": {"v": 1}},
    ]

    client._handle_event("dev_data", [{"dev_id": "dev", "updates": updates}])

    dev_state = coordinator.data["dev"]
    assert dev_state["nodes"]["nodes"][0]["addr"] == "A"
    assert dev_state["htr"]["settings"]["A"]["mode"] == "auto"
    assert dev_state["htr"]["advanced"]["A"]["boost"] == 15
    assert dev_state["raw"]["misc_info"] == {"v": 1}

    signals = [s for s, _ in dispatch_calls]
    data_signal = module.signal_ws_data("entry")
    assert signals.count(data_signal) == 3


def test_runner_retries_handshake(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ws_client_v2
    _configure_defaults(
        get_responses=[
            {"status": 500, "body": "fail"},
            {"status": 200, "body": '97:0{"sid":"abc","pingInterval":25000,"pingTimeout":60000}'},
        ],
        ws_results=[module.aiohttp.testing.FakeWebSocket()],
    )

    async def _run() -> tuple[int, Any]:
        monkeypatch.setattr(module.asyncio, "sleep", AsyncMock())
        monkeypatch.setattr(module.random, "uniform", lambda a, b: 1.0)

        hass = types.SimpleNamespace(
            loop=asyncio.get_running_loop(),
            data={module.DOMAIN: {"entry": {"ws_state": {}}}},
        )
        coordinator = types.SimpleNamespace(data={})
        session = module.aiohttp.testing.FakeClientSession()
        original_get = session.get

        def patched_get(url: str, *, timeout=None, headers=None):
            return original_get(url, timeout=timeout)

        session.get = patched_get  # type: ignore[assignment]
        api = types.SimpleNamespace(
            _session=session,
            api_base="https://api.example.com",
            _authed_headers=AsyncMock(return_value={"Authorization": "Bearer tok"}),
            _ensure_token=AsyncMock(),
        )

        client = module.TermoWebWSV2Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=api,
            coordinator=coordinator,
        )

        handshake_calls = 0
        orig_handshake = client._handshake

        async def wrapped_handshake() -> tuple[str, float, float]:
            nonlocal handshake_calls
            handshake_calls += 1
            return await orig_handshake()

        client._handshake = wrapped_handshake  # type: ignore[assignment]
        client._read_loop = AsyncMock(side_effect=asyncio.CancelledError())  # type: ignore[assignment]

        await client._runner()

        return handshake_calls, session

    handshake_calls, session = asyncio.run(_run())

    assert handshake_calls == 2
    assert len(session.get_calls) == 2


def test_stop_cancels_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ws_client_v2

    async def _run() -> list[str]:
        hass = types.SimpleNamespace(
            loop=asyncio.get_running_loop(),
            data={module.DOMAIN: {"entry": {"ws_state": {}}}},
        )
        coordinator = types.SimpleNamespace(data={})
        api = types.SimpleNamespace(
            _session=module.aiohttp.testing.FakeClientSession(),
            api_base="https://api.example.com",
            _authed_headers=AsyncMock(return_value={"Authorization": "Bearer tok"}),
            _ensure_token=AsyncMock(),
        )

        client = module.TermoWebWSV2Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=api,
            coordinator=coordinator,
        )

        update_calls: list[str] = []
        monkeypatch.setattr(
            module,
            "async_dispatcher_send",
            lambda *args, **kwargs: update_calls.append(args[1]),
        )

        class FakeHBTask:
            def __init__(self) -> None:
                self.cancelled = False

            def cancel(self) -> None:
                self.cancelled = True

            def __await__(self):
                async def _inner() -> None:
                    if self.cancelled:
                        raise asyncio.CancelledError()

                return _inner().__await__()

        class ClosingWS:
            def __init__(self) -> None:
                self.calls = 0

            async def close(self, *args: Any, **kwargs: Any) -> None:
                self.calls += 1
                raise RuntimeError("fail")

        client._hb_task = FakeHBTask()
        client._ws = ClosingWS()
        client._task = asyncio.create_task(asyncio.sleep(0.1))

        await client.stop()

        assert client._hb_task is None
        assert client._ws is None
        assert client._task is None
        return update_calls

    update_calls = asyncio.run(_run())
    assert update_calls[-1] == module.signal_ws_status("entry")


def test_start_reuses_existing_task(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ws_client_v2

    async def _run() -> tuple[int, bool]:
        hass = types.SimpleNamespace(
            data={module.DOMAIN: {"entry": {"ws_state": {}}}},
        )
        loop = asyncio.get_running_loop()
        created: list[asyncio.Task] = []

        def create_task(coro: Any, name: str | None = None) -> asyncio.Task:
            task = loop.create_task(coro)
            created.append(task)
            return task

        hass.loop = types.SimpleNamespace(create_task=create_task)

        api = types.SimpleNamespace(
            _session=module.aiohttp.testing.FakeClientSession(),
            api_base="https://api.example.com",
            _authed_headers=AsyncMock(return_value={"Authorization": "Bearer tok"}),
            _ensure_token=AsyncMock(),
        )
        coordinator = types.SimpleNamespace(data={})

        client = module.TermoWebWSV2Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=api,
            coordinator=coordinator,
        )
        stop_event = asyncio.Event()

        async def runner() -> None:
            await stop_event.wait()

        client._runner = runner  # type: ignore[assignment]

        task = client.start()
        await asyncio.sleep(0)
        same = client.start()
        stop_event.set()
        await task
        return len(created), same is task, client.is_running()

    count, same_task, running = asyncio.run(_run())
    assert count == 1
    assert same_task
    assert not running


def test_runner_handles_handshake_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ws_client_v2

    async def _run() -> tuple[list[str], int, int]:
        hass = types.SimpleNamespace(
            loop=asyncio.get_running_loop(),
            data={module.DOMAIN: {"entry": {"ws_state": {}}}},
        )
        api = types.SimpleNamespace(
            _session=module.aiohttp.testing.FakeClientSession(),
            api_base="https://api.example.com",
            _authed_headers=AsyncMock(return_value={"Authorization": "Bearer tok"}),
            _ensure_token=AsyncMock(),
        )
        coordinator = types.SimpleNamespace(data={})

        client = module.TermoWebWSV2Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=api,
            coordinator=coordinator,
            handshake_fail_threshold=1,
        )

        statuses: list[str] = []
        monkeypatch.setattr(client, "_update_status", lambda status: statuses.append(status))

        client._handshake = AsyncMock(
            side_effect=[
                module.HandshakeError(500, "http://example", "body"),
                asyncio.CancelledError(),
            ]
        )
        sleep = AsyncMock()
        monkeypatch.setattr(module.asyncio, "sleep", sleep)
        monkeypatch.setattr(module.random, "uniform", lambda *_: 1.0)

        await client._runner()

        return statuses, client._hs_fail_count, sleep.await_count

    statuses, fail_count, sleep_calls = asyncio.run(_run())
    assert statuses[0] == "starting"
    assert statuses[-1] == "stopped"
    assert statuses.count("disconnected") == 2
    assert fail_count == 0
    assert sleep_calls == 1


def test_runner_cleans_up_after_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ws_client_v2

    async def _run() -> tuple[list[str], bool, bool]:
        loop = asyncio.get_running_loop()
        created_tasks: list[asyncio.Task] = []

        def fake_create_task(coro: Any, name: str | None = None) -> asyncio.Task:
            task = loop.create_task(coro)
            created_tasks.append(task)
            return task

        hass = types.SimpleNamespace(
            loop=types.SimpleNamespace(create_task=fake_create_task),
            data={module.DOMAIN: {"entry": {"ws_state": {}}}},
        )
        api = types.SimpleNamespace(
            _session=module.aiohttp.testing.FakeClientSession(),
            api_base="https://api.example.com",
            _authed_headers=AsyncMock(return_value={"Authorization": "Bearer tok"}),
            _ensure_token=AsyncMock(),
        )
        coordinator = types.SimpleNamespace(data={})

        client = module.TermoWebWSV2Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=api,
            coordinator=coordinator,
        )

        statuses: list[str] = []
        monkeypatch.setattr(client, "_update_status", lambda status: statuses.append(status))

        client._handshake = AsyncMock(return_value=("sid", 25.0, 60.0))
        client._join_namespace = AsyncMock()  # type: ignore[assignment]
        client._request_snapshot = AsyncMock()  # type: ignore[assignment]
        client._read_loop = AsyncMock(side_effect=RuntimeError("boom"))  # type: ignore[assignment]

        close_called: list[bool] = []

        async def fake_connect(sid: str) -> None:
            ws = module.aiohttp.testing.FakeWebSocket()

            async def close() -> None:
                close_called.append(True)

            ws.close = close  # type: ignore[assignment]
            client._ws = ws

        client._connect_ws = fake_connect  # type: ignore[assignment]
        client._heartbeat_loop = AsyncMock()  # type: ignore[assignment]
        sleep = AsyncMock(side_effect=lambda _: setattr(client, "_closing", True))
        monkeypatch.setattr(module.asyncio, "sleep", sleep)
        monkeypatch.setattr(module.random, "uniform", lambda *_: 1.0)

        await client._runner()

        return statuses, close_called != [], len(created_tasks)

    statuses, ws_closed, hb_tasks = asyncio.run(_run())
    assert "connected" in statuses
    assert statuses[-1] == "stopped"
    assert ws_closed
    assert hb_tasks > 0


def test_handshake_refreshes_token_after_401(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ws_client_v2
    _configure_defaults(
        get_responses=[
            {"status": 401, "body": "unauthorized"},
            {
                "status": 200,
                "body": '97:0{"sid":"abc","pingInterval":20000,"pingTimeout":40000}',
            },
        ]
    )

    async def _run() -> tuple[str, float, float, int]:
        hass = types.SimpleNamespace(
            loop=asyncio.get_running_loop(),
            data={module.DOMAIN: {"entry": {"ws_state": {}}}},
        )
        session = module.aiohttp.testing.FakeClientSession()
        original_get = session.get

        def patched_get(url: str, *, timeout=None, headers=None):
            return original_get(url, timeout=timeout)

        session.get = patched_get  # type: ignore[assignment]
        api = types.SimpleNamespace(
            _session=session,
            api_base="https://api.example.com",
            _authed_headers=AsyncMock(return_value={"Authorization": "Bearer tok"}),
            _ensure_token=AsyncMock(),
        )
        coordinator = types.SimpleNamespace(data={})

        client = module.TermoWebWSV2Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=api,
            coordinator=coordinator,
        )
        refresher = AsyncMock()
        client._force_refresh_token = refresher  # type: ignore[assignment]

        sid, ping_interval, ping_timeout = await client._handshake()
        return sid, ping_interval, ping_timeout, refresher.await_count

    sid, ping_interval, ping_timeout, refresh_calls = asyncio.run(_run())
    assert sid == "abc"
    assert ping_interval == 20.0
    assert ping_timeout == 40.0
    assert refresh_calls == 1


def test_handshake_raises_on_transport_error() -> None:
    module = ws_client_v2

    async def _run() -> None:
        hass = types.SimpleNamespace(
            loop=asyncio.get_running_loop(),
            data={module.DOMAIN: {"entry": {"ws_state": {}}}},
        )
        session = module.aiohttp.testing.FakeClientSession()

        def raising_get(*args: Any, **kwargs: Any):
            raise TimeoutError("boom")

        session.get = raising_get  # type: ignore[assignment]
        api = types.SimpleNamespace(
            _session=session,
            api_base="https://api.example.com",
            _authed_headers=AsyncMock(return_value={"Authorization": "Bearer tok"}),
            _ensure_token=AsyncMock(),
        )
        coordinator = types.SimpleNamespace(data={})
        client = module.TermoWebWSV2Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=api,
            coordinator=coordinator,
        )

        with pytest.raises(module.HandshakeError) as exc:
            await client._handshake()
        assert exc.value.status == -1

    asyncio.run(_run())


def test_connect_ws_transforms_scheme(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ws_client_v2

    async def _run() -> tuple[str, str]:
        hass = types.SimpleNamespace(
            loop=asyncio.get_running_loop(),
            data={module.DOMAIN: {"entry": {"ws_state": {}}}},
        )
        session = module.aiohttp.testing.FakeClientSession(
            ws_connect_results=[module.aiohttp.testing.FakeWebSocket()]
        )
        api = types.SimpleNamespace(
            _session=session,
            api_base="http://api.example.com",
            _authed_headers=AsyncMock(return_value={"Authorization": "Bearer tok"}),
            _ensure_token=AsyncMock(),
        )
        coordinator = types.SimpleNamespace(data={})
        client = module.TermoWebWSV2Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=api,
            coordinator=coordinator,
        )

        client._get_token = AsyncMock(return_value="tok")  # type: ignore[assignment]
        await client._connect_ws("abc")
        first_url = session.ws_connect_calls[0]["url"]

        api.api_base = "https://secure.example.com"
        session.queue_ws(module.aiohttp.testing.FakeWebSocket())
        await client._connect_ws("xyz")
        second_url = session.ws_connect_calls[1]["url"]
        return first_url, second_url

    first_url, second_url = asyncio.run(_run())
    assert first_url.startswith("ws://")
    assert second_url.startswith("wss://")


def test_heartbeat_loop_sends_and_handles_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ws_client_v2

    async def _run() -> tuple[int, int]:
        hass = types.SimpleNamespace(
            loop=types.SimpleNamespace(create_task=lambda coro, name=None: coro),
            data={module.DOMAIN: {"entry": {"ws_state": {}}}},
        )
        api = types.SimpleNamespace(
            _session=module.aiohttp.testing.FakeClientSession(),
            api_base="https://api.example.com",
            _authed_headers=AsyncMock(return_value={"Authorization": "Bearer tok"}),
            _ensure_token=AsyncMock(),
        )
        coordinator = types.SimpleNamespace(data={})
        client = module.TermoWebWSV2Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=api,
            coordinator=coordinator,
        )
        send = AsyncMock(side_effect=[None, module.aiohttp.ClientError()])
        client._send_text = send  # type: ignore[assignment]
        client._hb_send_interval = 0

        task = asyncio.create_task(client._heartbeat_loop())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        await task
        return send.await_count, task.done()

    send_calls, done = asyncio.run(_run())
    assert send_calls == 2
    assert done


def test_read_loop_processes_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ws_client_v2

    async def _run() -> tuple[int, list[str], float]:
        hass = types.SimpleNamespace(
            loop=types.SimpleNamespace(create_task=lambda coro, name=None: coro),
            data={module.DOMAIN: {"entry": {"ws_state": {}}}},
        )
        api = types.SimpleNamespace(
            _session=module.aiohttp.testing.FakeClientSession(),
            api_base="https://api.example.com",
            _authed_headers=AsyncMock(return_value={"Authorization": "Bearer tok"}),
            _ensure_token=AsyncMock(),
        )
        coordinator = types.SimpleNamespace(data={})
        client = module.TermoWebWSV2Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=api,
            coordinator=coordinator,
        )

        event_payload = '["dev_data",{"dev_id":"dev"}]'
        messages = [
            {"type": module.aiohttp.WSMsgType.TEXT, "data": "3"},
            {"type": module.aiohttp.WSMsgType.TEXT, "data": "2"},
            {
                "type": module.aiohttp.WSMsgType.TEXT,
                "data": '0{"pingInterval":5000,"pingTimeout":15000}',
            },
            {
                "type": module.aiohttp.WSMsgType.BINARY,
                "data": f"42{module.WS_NAMESPACE},[\"ignored\"]".encode(),
            },
            {
                "type": module.aiohttp.WSMsgType.TEXT,
                "data": f"42{module.WS_NAMESPACE},{event_payload}",
            },
            {
                "type": module.aiohttp.WSMsgType.TEXT,
                "data": f"40{module.WS_NAMESPACE}",
            },
            {"type": module.aiohttp.WSMsgType.TEXT, "data": "41"},
        ]
        ws = module.aiohttp.testing.FakeWebSocket(messages)
        client._ws = ws

        send = AsyncMock()
        client._send_text = send  # type: ignore[assignment]

        handled: list[str] = []

        def record_event(name: str, args: list[Any]) -> None:
            handled.append(name)

        client._handle_event = record_event  # type: ignore[assignment]

        with pytest.raises(RuntimeError):
            await client._read_loop()

        return send.await_count, handled, client._hb_send_interval

    send_calls, handled_events, hb_interval = asyncio.run(_run())
    assert send_calls == 1  # ack for ping
    assert handled_events == ["ignored", "dev_data"]
    assert hb_interval == 5.0


def test_read_loop_errors_and_close(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ws_client_v2

    async def _run() -> tuple[str, str]:
        hass = types.SimpleNamespace(
            loop=types.SimpleNamespace(create_task=lambda coro, name=None: coro),
            data={module.DOMAIN: {"entry": {"ws_state": {}}}},
        )
        api = types.SimpleNamespace(
            _session=module.aiohttp.testing.FakeClientSession(),
            api_base="https://api.example.com",
            _authed_headers=AsyncMock(return_value={"Authorization": "Bearer tok"}),
            _ensure_token=AsyncMock(),
        )
        coordinator = types.SimpleNamespace(data={})
        client = module.TermoWebWSV2Client(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=api,
            coordinator=coordinator,
        )

        close_ws = module.aiohttp.testing.FakeWebSocket(
            [{"type": module.aiohttp.WSMsgType.CLOSE, "extra": "bye"}]
        )
        close_ws.set_exception(RuntimeError("ws fail"))
        client._ws = close_ws
        with pytest.raises(RuntimeError) as exc1:
            await client._read_loop()
        msg1 = str(exc1.value)

        error_ws = module.aiohttp.testing.FakeWebSocket(
            [{"type": module.aiohttp.WSMsgType.ERROR, "extra": None}]
        )
        client._ws = error_ws
        with pytest.raises(RuntimeError) as exc2:
            await client._read_loop()
        msg2 = str(exc2.value)

        return msg1, msg2

    message_close, message_error = asyncio.run(_run())
    assert "ws fail" in message_close
    assert "websocket error" in message_error


def test_handle_open_frame_updates_intervals() -> None:
    module = ws_client_v2
    hass = types.SimpleNamespace(
        loop=types.SimpleNamespace(create_task=lambda coro, name=None: coro),
        data={module.DOMAIN: {"entry": {"ws_state": {}}}},
    )
    api = types.SimpleNamespace(
        _session=module.aiohttp.testing.FakeClientSession(),
        api_base="https://api.example.com",
        _authed_headers=AsyncMock(return_value={"Authorization": "Bearer tok"}),
        _ensure_token=AsyncMock(),
    )
    coordinator = types.SimpleNamespace(data={})
    client = module.TermoWebWSV2Client(
        hass,
        entry_id="entry",
        dev_id="dev",
        api_client=api,
        coordinator=coordinator,
    )

    client._handle_open_frame("0not json")
    client._handle_open_frame('0{"pingInterval":1000,"pingTimeout":4000}')
    assert 0 < client._hb_send_interval <= 30
    assert client._ping_timeout == 4.0


def test_parse_event_frame_validates_namespace() -> None:
    module = ws_client_v2
    hass = types.SimpleNamespace(
        loop=types.SimpleNamespace(create_task=lambda coro, name=None: coro),
        data={module.DOMAIN: {"entry": {"ws_state": {}}}},
    )
    api = types.SimpleNamespace(
        _session=module.aiohttp.testing.FakeClientSession(),
        api_base="https://api.example.com",
        _authed_headers=AsyncMock(return_value={"Authorization": "Bearer tok"}),
        _ensure_token=AsyncMock(),
    )
    coordinator = types.SimpleNamespace(data={})
    client = module.TermoWebWSV2Client(
        hass,
        entry_id="entry",
        dev_id="dev",
        api_client=api,
        coordinator=coordinator,
    )

    assert client._parse_event_frame("42/other,[\"x\"]") is None
    assert client._parse_event_frame("42/") is None
    assert client._parse_event_frame("42{not json}") is None
    assert client._parse_event_frame("42[]") is None
    assert client._parse_event_frame("42[123]") is None
    assert client._parse_event_frame("42[\"evt\",1,2]") == ("evt", [1, 2])


def test_extract_updates_handles_handshake_and_defaults() -> None:
    module = ws_client_v2
    hass = types.SimpleNamespace(
        loop=types.SimpleNamespace(create_task=lambda coro, name=None: coro),
        data={module.DOMAIN: {"entry": {"ws_state": {}}}},
    )
    api = types.SimpleNamespace(
        _session=module.aiohttp.testing.FakeClientSession(),
        api_base="https://api.example.com",
        _authed_headers=AsyncMock(return_value={"Authorization": "Bearer tok"}),
        _ensure_token=AsyncMock(),
    )
    coordinator = types.SimpleNamespace(data={})
    client = module.TermoWebWSV2Client(
        hass,
        entry_id="entry",
        dev_id="fallback",
        api_client=api,
        coordinator=coordinator,
    )

    result = client._extract_updates(
        "dev_handshake",
        [
            {
                "devices": [
                    {"id": 123, "updates": [{"path": "/mgr/nodes", "body": {}}]},
                    {"serial_id": "fallback", "htr": {"settings": {"1": {"mode": "auto"}}}},
                ]
            }
        ],
    )
    assert len(result) == 2
    assert result[0][0] == "123"
    assert result[1][0] == "fallback"

    generic = client._extract_updates(
        "dev_data",
        [
            {"updates": [{"path": "/misc", "body": {"v": 1}}]},
            "ignore",
            None,
        ],
    )
    assert generic[0][0] == "fallback"


def test_apply_updates_initializes_device_and_deduplicates() -> None:
    module = ws_client_v2
    hass = types.SimpleNamespace(
        loop=types.SimpleNamespace(create_task=lambda coro, name=None: coro),
        data={module.DOMAIN: {"entry": {"ws_state": {}}}},
    )
    api = types.SimpleNamespace(
        _session=module.aiohttp.testing.FakeClientSession(),
        api_base="https://api.example.com",
        _authed_headers=AsyncMock(return_value={"Authorization": "Bearer tok"}),
        _ensure_token=AsyncMock(),
    )
    coordinator = types.SimpleNamespace(data={})
    client = module.TermoWebWSV2Client(
        hass,
        entry_id="entry",
        dev_id="dev",
        api_client=api,
        coordinator=coordinator,
    )

    paths, updated_nodes, setting_addrs, sample_addrs = client._apply_updates(
        "dev",
        [
            {"path": "/mgr/nodes", "body": {"nodes": [{"addr": "1", "type": "htr"}]}},
            {"path": "/htr/1/settings", "body": {"mode": "auto"}},
            {"path": "/htr/1/settings", "body": {"mode": "manual"}},
            {"path": "/htr/1/samples", "body": [1]},
            {"path": "/misc/info", "body": {"v": 1}},
        ],
    )

    assert "/mgr/nodes" in paths
    assert updated_nodes
    assert setting_addrs == ["1"]
    assert sample_addrs == ["1"]
    assert coordinator.data["dev"]["raw"]["misc_info"] == {"v": 1}


def test_parse_handshake_validates_payload() -> None:
    module = ws_client_v2
    hass = types.SimpleNamespace(
        loop=types.SimpleNamespace(create_task=lambda coro, name=None: coro),
        data={module.DOMAIN: {"entry": {"ws_state": {}}}},
    )
    api = types.SimpleNamespace(
        _session=module.aiohttp.testing.FakeClientSession(),
        api_base="https://api.example.com",
        _authed_headers=AsyncMock(return_value={"Authorization": "Bearer tok"}),
        _ensure_token=AsyncMock(),
    )
    coordinator = types.SimpleNamespace(data={})
    client = module.TermoWebWSV2Client(
        hass,
        entry_id="entry",
        dev_id="dev",
        api_client=api,
        coordinator=coordinator,
    )

    with pytest.raises(TypeError):
        client._parse_handshake("[]")
    with pytest.raises(TypeError):
        client._parse_handshake('{"pingInterval":1000}')

    sid, interval, timeout = client._parse_handshake(
        '97:0{"sid":"abc","pingInterval":5000,"pingTimeout":15000}'
    )
    assert sid == "abc"
    assert interval == 5.0
    assert timeout == 15.0


def test_send_text_checks_ws() -> None:
    module = ws_client_v2
    hass = types.SimpleNamespace(
        loop=types.SimpleNamespace(create_task=lambda coro, name=None: coro),
        data={module.DOMAIN: {"entry": {"ws_state": {}}}},
    )
    api = types.SimpleNamespace(
        _session=module.aiohttp.testing.FakeClientSession(),
        api_base="https://api.example.com",
        _authed_headers=AsyncMock(return_value={"Authorization": "Bearer tok"}),
        _ensure_token=AsyncMock(),
    )
    coordinator = types.SimpleNamespace(data={})
    client = module.TermoWebWSV2Client(
        hass,
        entry_id="entry",
        dev_id="dev",
        api_client=api,
        coordinator=coordinator,
    )

    asyncio.run(client._send_text("hello"))
    ws = module.aiohttp.testing.FakeWebSocket()
    client._ws = ws
    asyncio.run(client._send_text("world"))
    assert ws.sent == ["world"]


def test_token_helpers_and_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ws_client_v2
    hass = types.SimpleNamespace(
        loop=types.SimpleNamespace(create_task=lambda coro, name=None: coro),
        data={module.DOMAIN: {"entry": {"ws_state": {}}}},
    )
    api = types.SimpleNamespace(
        _session=module.aiohttp.testing.FakeClientSession(),
        api_base="https://api.example.com/",
        _authed_headers=AsyncMock(return_value={"Authorization": "Bearer tok"}),
        _ensure_token=AsyncMock(),
        _access_token="old",
    )
    coordinator = types.SimpleNamespace(data={})
    client = module.TermoWebWSV2Client(
        hass,
        entry_id="entry",
        dev_id="dev",
        api_client=api,
        coordinator=coordinator,
    )

    token = asyncio.run(client._get_token())
    assert token == "tok"
    asyncio.run(client._force_refresh_token())
    assert api._ensure_token.await_count == 1
    assert client._api_base() == "https://api.example.com"

    api.api_base = None
    assert client._api_base().startswith("https://api-tevolve")


def test_update_status_records_state(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ws_client_v2
    hass = types.SimpleNamespace(
        loop=types.SimpleNamespace(create_task=lambda coro, name=None: coro),
        data={module.DOMAIN: {"entry": {"ws_state": {}}}},
    )
    api = types.SimpleNamespace(
        _session=module.aiohttp.testing.FakeClientSession(),
        api_base="https://api.example.com",
        _authed_headers=AsyncMock(return_value={"Authorization": "Bearer tok"}),
        _ensure_token=AsyncMock(),
    )
    coordinator = types.SimpleNamespace(data={})
    client = module.TermoWebWSV2Client(
        hass,
        entry_id="entry",
        dev_id="dev",
        api_client=api,
        coordinator=coordinator,
    )
    client._stats.frames_total = 2
    client._stats.events_total = 1
    client._stats.last_event_ts = 1234.0
    statuses: list[dict[str, Any]] = []

    def record_dispatch(hass_obj: Any, signal: str, payload: dict[str, Any]) -> None:
        statuses.append(payload)

    monkeypatch.setattr(module, "async_dispatcher_send", record_dispatch)

    client._update_status("connected")

    state = hass.data[module.DOMAIN]["entry"]["ws_state"]["dev"]
    assert state["status"] == "connected"
    assert statuses[0]["status"] == "connected"


def test_mark_event_tracks_health(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ws_client_v2
    hass = types.SimpleNamespace(
        loop=types.SimpleNamespace(create_task=lambda coro, name=None: coro),
        data={module.DOMAIN: {"entry": {"ws_state": {}}}},
    )
    api = types.SimpleNamespace(
        _session=module.aiohttp.testing.FakeClientSession(),
        api_base="https://api.example.com",
        _authed_headers=AsyncMock(return_value={"Authorization": "Bearer tok"}),
        _ensure_token=AsyncMock(),
    )
    coordinator = types.SimpleNamespace(data={})
    client = module.TermoWebWSV2Client(
        hass,
        entry_id="entry",
        dev_id="dev",
        api_client=api,
        coordinator=coordinator,
    )

    client._connected_since = 10
    client._healthy_since = None
    client._stats.frames_total = 5
    client._stats.events_total = 1

    statuses: list[str] = []

    def fake_dispatch(*args: Any, **kwargs: Any) -> None:
        pass

    def record_status(status: str) -> None:
        statuses.append(status)

    monkeypatch.setattr(module, "async_dispatcher_send", fake_dispatch)
    monkeypatch.setattr(client, "_update_status", record_status)
    monkeypatch.setattr(module._LOGGER, "isEnabledFor", lambda level: True)
    monkeypatch.setattr(module.time, "time", lambda: 360.0)

    client._mark_event(paths=["/a", "/a", "/b", "/c", "/d", "/e"])  # 6 entries -> dedupe
    assert statuses[-1] == "healthy"
    assert client._stats.last_paths == ["/a", "/b", "/c", "/d", "/e"][:5]
