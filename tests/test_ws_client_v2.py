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
