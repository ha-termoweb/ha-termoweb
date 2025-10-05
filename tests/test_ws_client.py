import asyncio
from copy import deepcopy
import logging
import logging
import time
from contextlib import suppress
from types import SimpleNamespace
from typing import Any, Mapping, cast
from unittest.mock import AsyncMock, MagicMock, call

import pytest

import custom_components.termoweb.ws_client as module
from custom_components.termoweb.installation import InstallationSnapshot


class DummyREST:
    """Minimal REST client stub for websocket tests."""

    def __init__(self, base: str = "https://api.example.com/api/v2") -> None:
        self.api_base = base
        self._session = SimpleNamespace()
        self._headers = {"Authorization": "Bearer token"}
        self._ensure_token = AsyncMock()
        self._set_node_settings = AsyncMock(return_value={"status": "ok"})
        self._get_rtc_time = AsyncMock(return_value={"status": "ok"})

    async def _authed_headers(self) -> dict[str, str]:
        return self._headers

    async def set_node_settings(self, *args: Any, **kwargs: Any) -> Any:
        return await self._set_node_settings(*args, **kwargs)

    async def get_rtc_time(self, dev_id: str) -> Any:
        return await self._get_rtc_time(dev_id)


@pytest.fixture(autouse=True)
def patch_async_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ``socketio.AsyncClient`` with a controllable stub."""

    class StubAsyncClient:
        def __init__(self, **_: Any) -> None:
            self.connected = False
            self.events: dict[tuple[str, str | None], Any] = {}
            self.last_emit: tuple[str, Any | None, str | None] | None = None
            self.eio = SimpleNamespace(
                start_background_task=lambda target, *a, **kw: None,
                http=None,
            )

        def on(self, event: str, *, handler: Any, namespace: str | None = None) -> None:
            self.events[(event, namespace)] = handler

        async def connect(self, *args: Any, **kwargs: Any) -> None:
            self.connected = True
            self.connect_args = (args, kwargs)

        async def disconnect(self) -> None:
            self.connected = False

        async def emit(
            self,
            event: str,
            data: Any | None = None,
            *,
            namespace: str | None = None,
        ) -> None:
            self.last_emit = (event, data, namespace)

    monkeypatch.setattr(module.socketio, "AsyncClient", StubAsyncClient)


def _make_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    hass_loop: Any | None = None,
    rest: DummyREST | None = None,
) -> module.WebSocketClient:
    """Helper to instantiate a websocket client with test doubles."""

    if hass_loop is None:

        class _DummyTask:
            def __init__(self, coro: Any) -> None:
                self._coro = coro
                self._cancelled = False
                self._completed = False
                cr_code = getattr(coro, "cr_code", None)
                if cr_code and cr_code.co_name == "_execute_mock_call":
                    try:
                        coro.send(None)
                    except StopIteration:
                        self._completed = True
                    except RuntimeError:
                        # Raised if the coroutine was already awaited/closed.
                        self._completed = True

            def _close(self) -> None:
                closer = getattr(self._coro, "close", None)
                if callable(closer):
                    try:
                        closer()
                    except RuntimeError:
                        # Closing an already-finished coroutine raises RuntimeError.
                        pass

            def cancel(self) -> None:
                self._cancelled = True
                self._close()

            def done(self) -> bool:
                return self._cancelled or self._completed

            def __await__(self):  # type: ignore[no-untyped-def]
                if self._cancelled or self._completed:

                    async def _noop() -> None:
                        return None

                    return _noop().__await__()
                return self._coro.__await__()

            def __del__(self) -> None:
                if not (self._cancelled or self._completed):
                    self._close()

        def _create_task(coro: Any, **kwargs: Any) -> _DummyTask:
            return _DummyTask(coro)

        hass_loop = SimpleNamespace(
            create_task=_create_task,
            call_soon_threadsafe=lambda cb, *args: cb(*args),
        )
    hass = SimpleNamespace(loop=hass_loop, data={})
    coordinator = SimpleNamespace(data={}, update_nodes=MagicMock())
    rest_client = rest or DummyREST()
    # Avoid dispatch callbacks firing during tests by capturing them.
    dispatcher_mock = MagicMock()
    monkeypatch.setattr(module, "async_dispatcher_send", dispatcher_mock)
    client = module.WebSocketClient(
        hass,
        entry_id="entry",
        dev_id="device",
        api_client=rest_client,
        coordinator=coordinator,
    )
    client._dispatcher_mock = dispatcher_mock  # type: ignore[attr-defined]
    return client


def _make_ducaheat_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    hass_loop: Any | None = None,
    namespace: str = "/",
    rest: DummyREST | None = None,
) -> module.DucaheatWSClient:
    """Return a Ducaheat websocket client configured for tests."""

    if hass_loop is None:

        def _create_task(coro: Any, **_: Any) -> Any:
            closer = getattr(coro, "close", None)
            if callable(closer):
                closer()
            return SimpleNamespace(done=lambda: True)

        hass_loop = SimpleNamespace(
            create_task=_create_task,
            call_soon_threadsafe=lambda cb, *args: cb(*args),
        )
    hass = SimpleNamespace(loop=hass_loop, data={module.DOMAIN: {"entry": {}}})
    coordinator = SimpleNamespace(
        data={}, update_nodes=MagicMock(), async_request_refresh=AsyncMock()
    )
    rest_client = rest or DummyREST()
    dispatcher_mock = MagicMock()
    monkeypatch.setattr(module, "async_dispatcher_send", dispatcher_mock)
    client = module.DucaheatWSClient(
        hass,
        entry_id="entry",
        dev_id="device",
        api_client=rest_client,
        coordinator=coordinator,
        session=SimpleNamespace(),
        namespace=namespace,
    )
    client._dispatcher_mock = dispatcher_mock  # type: ignore[attr-defined]
    return client


def test_websocket_client_default_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the base client defaults to the legacy namespace."""

    client = _make_client(monkeypatch)
    assert client._namespace == module.WS_NAMESPACE


def test_ducaheat_client_default_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the Ducaheat client uses the API v2 namespace by default."""

    client = _make_ducaheat_client(monkeypatch)
    assert client._namespace == "/"
    assert ("dev_data", "/") in client._sio.events
    assert ("disconnect", "/") in client._sio.events


@pytest.mark.asyncio
async def test_ducaheat_connect_uses_brand_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure the Ducaheat client sets required brand headers."""

    client = _make_ducaheat_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    client._stop_event = asyncio.Event()
    connect_mock = AsyncMock()
    client._sio.connect = connect_mock

    await client._connect_once()

    headers = connect_mock.await_args.kwargs["headers"]
    assert headers["Origin"] == "https://localhost"
    assert headers["X-Requested-With"] == "net.termoweb.ducaheat.app"


def test_translate_path_update_parses_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    """Translate Ducaheat style path payloads into node updates."""

    client = _make_ducaheat_client(monkeypatch)

    assert client._translate_path_update(None) is None
    assert client._translate_path_update({"nodes": {}}) is None

    sample_payload = {
        "path": "/devs/dev/htr/001/samples",
        "body": {"energy": 1},
    }
    translated = client._translate_path_update(sample_payload)
    assert translated == {"htr": {"samples": {"001": {"energy": 1}}}}

    nested_payload = {
        "path": "/htr/001/setup/limits/max",
        "body": {"value": 10},
    }
    translated_nested = client._translate_path_update(nested_payload)
    assert translated_nested == {
        "htr": {
            "settings": {
                "001": {"setup": {"limits": {"max": {"value": 10}}}}
            }
        }
    }

    status_payload = {
        "path": "/acm/2/status",
        "body": {"state": "ok"},
    }
    translated_status = client._translate_path_update(status_payload)
    assert translated_status == {"acm": {"status": {"2": {"state": "ok"}}}}


def test_translate_path_update_edge_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    """Handle malformed path payloads in the translator."""

    client = _make_ducaheat_client(monkeypatch)

    assert client._translate_path_update({"path": "", "body": {}}) is None
    assert client._translate_path_update({"path": "/foo", "body": {}}) is None
    assert (
        client._translate_path_update({"path": "/devs/dev/ /001/status", "body": {}})
        is None
    )
    assert (
        client._translate_path_update({"path": "/devs/dev/unknown/001", "body": {}})
        is None
    )
    assert (
        client._translate_path_update({"path": "/devs/dev/htr/001", "body": {}})
        is None
    )
    assert (
        client._translate_path_update({"path": "/devs/dev/htr/ /status", "body": {}})
        is None
    )

    assert client._resolve_update_section(None) == (None, None)
    assert client._resolve_update_section("advanced_setup") == (
        "advanced",
        "advanced_setup",
    )
    assert client._resolve_update_section("setup") == ("settings", "setup")


def test_handshake_error_records_context() -> None:
    """Ensure the custom handshake exception preserves response metadata."""

    err = module.HandshakeError(401, "https://example", "unauthorized")
    assert err.status == 401
    assert err.url == "https://example"
    assert err.body_snippet == "unauthorized"
    assert "handshake failed" in str(err)


@pytest.mark.asyncio
async def test_error_handlers_log_payloads(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify socket error callbacks emit raw payloads at debug level."""

    client = _make_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    caplog.set_level(logging.DEBUG)

    connect_error = client._sio.events[("connect_error", None)]
    error_handler = client._sio.events[("error", None)]
    reconnect_failed = client._sio.events[("reconnect_failed", None)]
    ns_disconnect = client._sio.events[("disconnect", module.WS_NAMESPACE)]

    await connect_error({"detail": "bad token"})
    assert "connect_error payload: {'detail': 'bad token'}" in caplog.text

    caplog.clear()
    await error_handler("boom")
    assert "error event payload: boom" in caplog.text

    caplog.clear()
    await reconnect_failed("server down")
    assert "reconnect_failed details: server down" in caplog.text

    caplog.clear()
    await ns_disconnect("transport closed")
    assert (
        f"namespace disconnect ({module.WS_NAMESPACE}): transport closed" in caplog.text
    )


def test_ws_state_bucket_initialises_missing_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify hass.data is created when absent."""

    hass_loop = SimpleNamespace(
        create_task=lambda coro, **_: SimpleNamespace(done=lambda: True),
        call_soon_threadsafe=lambda cb, *args: cb(*args),
    )
    hass = SimpleNamespace(loop=hass_loop)
    coordinator = SimpleNamespace(data={}, update_nodes=MagicMock())
    dispatcher = MagicMock()
    monkeypatch.setattr(module, "async_dispatcher_send", dispatcher)
    client = module.WebSocketClient(
        hass,
        entry_id="entry",
        dev_id="device",
        api_client=DummyREST(),
        coordinator=coordinator,
    )
    bucket = client._ws_state_bucket()
    assert module.DOMAIN in hass.data  # type: ignore[attr-defined]
    assert hass.data[module.DOMAIN]["entry"]["ws_state"]["device"] is bucket  # type: ignore[index]


def _make_legacy_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    hass_loop: Any | None = None,
) -> module.TermoWebWSClient:
    """Return a TermoWeb legacy websocket client with patched dependencies."""

    if hass_loop is None:

        def _create_task(coro: Any, **_: Any) -> Any:
            closer = getattr(coro, "close", None)
            if callable(closer):
                closer()
            return SimpleNamespace(done=lambda: True)

        hass_loop = SimpleNamespace(
            create_task=_create_task,
            call_soon_threadsafe=lambda cb, *args: cb(*args),
        )
    hass = SimpleNamespace(loop=hass_loop, data={module.DOMAIN: {"entry": {}}})
    coordinator = SimpleNamespace(data={}, update_nodes=MagicMock())
    rest_client = DummyREST()
    dispatcher_mock = MagicMock()
    monkeypatch.setattr(module, "async_dispatcher_send", dispatcher_mock)
    client = module.TermoWebWSClient(
        hass,
        entry_id="entry",
        dev_id="device",
        api_client=rest_client,
        coordinator=coordinator,
        session=SimpleNamespace(),
    )
    client._dispatcher_mock = dispatcher_mock  # type: ignore[attr-defined]
    return client


@pytest.mark.asyncio
async def test_legacy_write_restart_after_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Schedule an immediate restart when a write follows long inactivity."""

    client = _make_legacy_client(monkeypatch)
    rest_client = client._client
    now = module.time.time()
    client._payload_idle_window = 600.0
    client._last_payload_at = now - 700
    client._stats.last_event_ts = now - 700
    client._idle_restart_pending = False
    client._idle_restart_task = None

    await rest_client.set_node_settings("device", {"addr": 1})

    assert rest_client._set_node_settings.await_count == 1
    assert client._idle_restart_pending is True
    assert client._idle_restart_task is not None


@pytest.mark.asyncio
async def test_legacy_write_recent_payload_skips_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid restarting when payloads have been received recently."""

    client = _make_legacy_client(monkeypatch)
    rest_client = client._client
    now = module.time.time()
    client._payload_idle_window = 600.0
    client._last_payload_at = now - 120
    client._stats.last_event_ts = now - 120
    client._idle_restart_pending = False
    client._idle_restart_task = None

    await rest_client.set_node_settings("device", {"addr": 2})

    assert rest_client._set_node_settings.await_count == 1
    assert client._idle_restart_pending is False
    assert client._idle_restart_task is None


@pytest.mark.asyncio
async def test_legacy_write_other_device_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ignore writes for other devices when monitoring restart triggers."""

    client = _make_legacy_client(monkeypatch)
    rest_client = client._client
    now = module.time.time()
    client._stats.last_event_ts = now - 700
    client._idle_restart_pending = False
    client._idle_restart_task = None

    await rest_client.set_node_settings("other", {"addr": 3})

    assert rest_client._set_node_settings.await_count == 1
    assert client._idle_restart_pending is False
    assert client._idle_restart_task is None


@pytest.mark.asyncio
async def test_legacy_rtc_keepalive_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Poll the REST API on a fixed cadence while connected."""

    client = _make_legacy_client(monkeypatch)
    rest_client = client._client

    calls: list[str] = []

    async def fake_get(dev_id: str) -> dict[str, Any]:
        calls.append(dev_id)
        return {"ok": True}

    rest_client.get_rtc_time = AsyncMock(side_effect=fake_get)

    intervals: list[float] = []

    async def fake_sleep(delay: float) -> None:
        intervals.append(delay)
        if len(intervals) >= 2:
            client._closing = True

    monkeypatch.setattr(module.asyncio, "sleep", fake_sleep)

    client._closing = False
    client._rtc_keepalive_interval = 0.05

    await client._rtc_keepalive_loop()

    assert calls and all(dev_id == "device" for dev_id in calls)
    assert len(calls) >= 2
    assert intervals[0] == pytest.approx(0.05)


def test_http_wrapping_handles_missing_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure http assignments tolerate attribute errors and reuse existing sessions."""

    class AltAsyncClient:
        def __init__(self, **_: Any) -> None:
            object.__setattr__(self, "_http_raise", True)
            object.__setattr__(self, "connected", False)
            object.__setattr__(self, "events", {})
            object.__setattr__(
                self, "last_emit", cast(tuple[str, Any | None, str | None] | None, None)
            )
            object.__setattr__(
                self,
                "eio",
                SimpleNamespace(
                    start_background_task=lambda target, *a, **kw: None,
                    http=SimpleNamespace(closed=True),
                ),
            )

        def __setattr__(self, name: str, value: Any) -> None:
            if name == "http" and getattr(self, "_http_raise", False):
                object.__setattr__(self, "_http_raise", False)
                raise AttributeError("http not writable")
            object.__setattr__(self, name, value)

        def on(self, event: str, *, handler: Any, namespace: str | None = None) -> None:
            self.events[(event, namespace)] = handler

        async def connect(self, *args: Any, **kwargs: Any) -> None:
            self.connected = True
            self.connect_args = (args, kwargs)

        async def disconnect(self) -> None:
            self.connected = False

        async def emit(
            self,
            event: str,
            data: Any | None = None,
            *,
            namespace: str | None = None,
        ) -> None:
            self.last_emit = (event, data, namespace)

    monkeypatch.setattr(module.socketio, "AsyncClient", AltAsyncClient)
    client = _make_client(monkeypatch)
    assert getattr(client._sio, "http").closed is True
    assert getattr(client._sio.eio, "http").closed is True


@pytest.mark.asyncio
async def test_ws_url_and_engineio_target(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_ducaheat_client(monkeypatch, hass_loop=asyncio.get_event_loop())

    ws_url = await client.ws_url()
    assert ws_url == "https://api.example.com/socket.io?token=token&dev_id=device"

    base, path = await client._build_engineio_target()
    assert base == "https://api.example.com/socket.io?token=token&dev_id=device"
    assert path == "socket.io"


@pytest.mark.asyncio
async def test_ws_url_appends_missing_api_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure websocket targets append the API version path when absent."""

    rest = DummyREST(base="https://api.example.com")
    client = _make_ducaheat_client(
        monkeypatch,
        hass_loop=asyncio.get_event_loop(),
        namespace="/",
        rest=rest,
    )

    ws_url = await client.ws_url()
    assert ws_url == "https://api.example.com/socket.io?token=token&dev_id=device"

    base, path = await client._build_engineio_target()
    assert base == "https://api.example.com/socket.io?token=token&dev_id=device"
    assert path == "socket.io"


@pytest.mark.asyncio
async def test_wrap_background_task_runs_coroutine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    captured: list[str] = []

    async def runner(value: str) -> None:
        captured.append(value)

    task = client._wrap_background_task(runner, "ok")
    await task
    assert captured == ["ok"]


def test_wrap_background_task_with_sync_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = asyncio.new_event_loop()
    client = _make_client(monkeypatch, hass_loop=loop)
    task = client._wrap_background_task(lambda: "value")
    result = loop.run_until_complete(task)
    assert asyncio.iscoroutine(result)
    result.close()
    loop.close()


def test_is_running_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch)
    client._task = SimpleNamespace(done=lambda: False)
    assert client.is_running()
    client._task = SimpleNamespace(done=lambda: True)
    assert not client.is_running()


def test_legacy_handle_event_dispatches_and_logs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Ensure the legacy client publishes updates and logs affected nodes."""

    client = _make_legacy_client(monkeypatch)
    dispatcher: MagicMock = client._dispatcher_mock  # type: ignore[attr-defined]
    monkeypatch.setattr(
        module,
        "ensure_node_inventory",
        MagicMock(return_value=[SimpleNamespace(type="htr", addr="1")]),
    )
    monkeypatch.setattr(
        module,
        "addresses_by_node_type",
        MagicMock(return_value=({"htr": ["1"]}, set())),
    )
    monkeypatch.setattr(
        module,
        "normalize_heater_addresses",
        MagicMock(return_value=({"htr": ["1"]}, {})),
    )
    caplog.set_level(logging.DEBUG)

    snapshot_event = {
        "name": "data",
        "args": [
            [
                {
                    "path": "/mgr/nodes",
                    "body": {"htr": {"settings": {"1": {"mode": "eco"}}}},
                }
            ]
        ],
    }
    client._handle_event(snapshot_event)
    assert client._coordinator.update_nodes.called
    assert "nodes" in client._nodes  # type: ignore[attr-defined]
    payloads = [call.args[2] for call in dispatcher.mock_calls if len(call.args) >= 3]
    assert any(
        isinstance(payload, dict) and payload.get("kind") == "nodes"
        for payload in payloads
    )

    dispatcher.reset_mock()
    update_event = {
        "name": "data",
        "args": [[{"path": "/htr/1/settings", "body": {"mode": "comfort"}}]],
    }
    client._handle_event(update_event)
    payloads = [call.args[2] for call in dispatcher.mock_calls if len(call.args) >= 3]
    assert any(
        isinstance(payload, dict)
        and payload.get("kind") == "htr_settings"
        and payload.get("addr") == "1"
        for payload in payloads
    )
    assert "legacy update for htr/1" in caplog.text


def test_legacy_mark_event_tracks_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Update the payload timestamp when legacy batches arrive."""

    client = _make_legacy_client(monkeypatch)
    monkeypatch.setattr(module.time, "time", lambda: 1000.0)

    client._last_payload_at = None
    client._mark_event(paths=None, count_event=True)

    assert client._last_payload_at == pytest.approx(1000.0)
    assert (
        client._ws_state_bucket()["last_payload_at"]
        == pytest.approx(1000.0)
    )


def test_legacy_heartbeat_does_not_cancel_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Record heartbeat activity without clearing idle restart state."""

    client = _make_legacy_client(monkeypatch)
    client._idle_restart_pending = True
    client._last_payload_at = 900.0
    monkeypatch.setattr(module.time, "time", lambda: 1200.0)

    client._record_heartbeat(source="socketio09")

    assert client._idle_restart_pending is True
    assert client._last_payload_at == pytest.approx(900.0)
    assert client._last_heartbeat_at == pytest.approx(1200.0)


@pytest.mark.asyncio
async def test_maybe_restart_after_write_ignores_heartbeat_activity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Schedule a restart when writes follow heartbeat-only activity."""

    client = _make_legacy_client(monkeypatch)
    client._payload_idle_window = 600.0
    client._last_payload_at = 1000.0

    timestamps = [1500.0, 1900.0]

    def fake_time() -> float:
        value = timestamps.pop(0) if timestamps else 1900.0
        return value

    monkeypatch.setattr(module.time, "time", fake_time)

    client._record_heartbeat(source="socketio09")

    scheduled: list[tuple[float, str]] = []

    def fake_schedule(*, idle_for: float, source: str) -> None:
        scheduled.append((idle_for, source))

    monkeypatch.setattr(client, "_schedule_idle_restart", fake_schedule)

    await client.maybe_restart_after_write()

    assert scheduled == [(900.0, "write notification")]


def test_idle_restart_state_reflected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expose idle restart state changes through the websocket bucket."""

    captured: dict[str, Any] = {}

    def _create_task(coro: Any, **_: Any) -> Any:
        captured["coro"] = coro
        return SimpleNamespace(cancel=lambda: None, done=lambda: False)

    hass_loop = SimpleNamespace(
        create_task=_create_task,
        call_soon_threadsafe=lambda cb, *args: cb(*args),
    )
    client = _make_legacy_client(monkeypatch, hass_loop=hass_loop)

    state = client._ws_state_bucket()
    assert state["idle_restart_pending"] is False

    client._schedule_idle_restart(idle_for=30.0, source="test case")
    assert state["idle_restart_pending"] is True

    client._cancel_idle_restart()
    assert state["idle_restart_pending"] is False

    coro = captured.get("coro")
    if coro is not None:
        coro.close()


@pytest.mark.asyncio
async def test_legacy_idle_monitor_schedules_restart_after_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Schedule a websocket restart when payloads stop despite heartbeats."""

    client = _make_legacy_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    client._ws = SimpleNamespace(closed=False)
    client._disconnected.clear()
    client._payload_idle_window = 10
    client._last_payload_at = 900.0

    monkeypatch.setattr(module.time, "time", lambda: 920.0)
    monkeypatch.setattr(module.asyncio, "sleep", AsyncMock(return_value=None))

    triggered: list[tuple[float, str]] = []

    def fake_schedule(*, idle_for: float, source: str) -> None:
        triggered.append((idle_for, source))

    client._schedule_idle_restart = fake_schedule  # type: ignore[assignment]

    client._record_heartbeat(source="socketio09")
    await client._idle_monitor()

    assert triggered
    idle_for, source = triggered[0]
    assert source == "idle monitor payload timeout"
    assert idle_for == pytest.approx(20.0)


@pytest.mark.asyncio
async def test_legacy_idle_monitor_reset_by_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fresh payloads reset the idle timer and avoid restarts."""

    client = _make_legacy_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    client._ws = SimpleNamespace(closed=False)
    client._disconnected.clear()
    client._payload_idle_window = 30

    times: list[float] = [900.0, 905.0]

    def fake_time() -> float:
        return times.pop(0) if times else 905.0

    monkeypatch.setattr(module.time, "time", fake_time)

    async def fake_sleep(_: float) -> None:
        client._closing = True

    monkeypatch.setattr(module.asyncio, "sleep", fake_sleep)

    triggered: list[tuple[float, str]] = []

    def fake_schedule(*, idle_for: float, source: str) -> None:
        triggered.append((idle_for, source))

    client._schedule_idle_restart = fake_schedule  # type: ignore[assignment]

    client._mark_event(paths=None, count_event=True)
    await client._idle_monitor()

    assert triggered == []


@pytest.mark.asyncio
async def test_legacy_idle_monitor_retries_failed_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry failed subscription refreshes before scheduling restarts."""

    client = _make_legacy_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    client._ws = SimpleNamespace(closed=False)
    client._disconnected.clear()
    client._payload_idle_window = 100
    client._last_payload_at = 1000.0
    client._subscription_refresh_failed = True

    refresh = AsyncMock(side_effect=RuntimeError("refresh boom"))
    monkeypatch.setattr(client, "_refresh_subscription", refresh)
    monkeypatch.setattr(module.asyncio, "sleep", AsyncMock(return_value=None))
    monkeypatch.setattr(module.time, "time", lambda: 1010.0)

    triggered: list[tuple[float, str]] = []

    def fake_schedule(*, idle_for: float, source: str) -> None:
        triggered.append((idle_for, source))

    client._schedule_idle_restart = fake_schedule  # type: ignore[assignment]

    await client._idle_monitor()

    assert refresh.await_count == 1
    assert triggered == [(10.0, "idle monitor retry failed")]


def test_apply_nodes_payload_handles_missing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    client = _make_client(monkeypatch)
    caplog.set_level(logging.DEBUG)
    client._apply_nodes_payload({}, merge=False, event="dev_data")
    assert "without nodes" in caplog.text


def test_apply_nodes_payload_handles_normaliser_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    client = _make_client(monkeypatch)

    def broken_normaliser(_: Any) -> Any:
        raise RuntimeError

    client._client.normalise_ws_nodes = broken_normaliser  # type: ignore[attr-defined]
    caplog.set_level(logging.DEBUG)
    payload = {"nodes": {"htr": {"settings": {"1": {}}}}}
    client._apply_nodes_payload(payload, merge=False, event="dev_data")
    assert client._nodes_raw["htr"]["settings"]["1"] == {}


def test_apply_nodes_payload_debug_logging(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    client = _make_client(monkeypatch)
    caplog.set_level(logging.DEBUG)
    payload = {"nodes": {"htr": {"settings": {"1": {}}}}}
    client._apply_nodes_payload(payload, merge=False, event="dev_data")
    assert "snapshot contains" in caplog.text

    client._collect_update_addresses = MagicMock(return_value=[])
    client._apply_nodes_payload({"nodes": {}}, merge=True, event="update")
    assert "without address changes" in caplog.text


def test_apply_nodes_payload_logs_changed_addresses(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    client = _make_client(monkeypatch)
    client._nodes_raw = {"htr": {"settings": {"1": {"temp": 20}}}}
    caplog.set_level(logging.DEBUG)
    payload = {"nodes": {"htr": {"settings": {"1": {"temp": 22}}}}}
    client._apply_nodes_payload(payload, merge=True, event="update")
    assert "update event for" in caplog.text


def test_apply_nodes_payload_skips_invalid_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ignore invalid node types and sample addresses when forwarding updates."""

    client = _make_client(monkeypatch)
    forward = MagicMock()
    client._forward_sample_updates = forward  # type: ignore[assignment]

    payload = {
        "nodes": {
            5: {"samples": {"2": {"temp": 19}}},
            "htr": {"samples": {"": {"temp": 20}, "1": {"temp": 21}}},
        }
    }

    client._apply_nodes_payload(payload, merge=False, event="dev_data")

    forward.assert_called_once()
    updates = forward.call_args[0][0]
    assert "htr" in updates
    assert updates["htr"] == {"1": {"temp": 21}}


@pytest.mark.asyncio
async def test_ducaheat_update_logging_is_condensed(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Ensure Ducaheat overrides emit condensed address summaries."""

    client = _make_ducaheat_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    client._coordinator.data = {"device": {"nodes_by_type": {}}}
    caplog.clear()
    caplog.set_level(logging.DEBUG)

    payload = {"nodes": {"htr": {"settings": {"1": {"temp": 23}}}}}
    await client._on_update(payload)

    ducaheat_logs = [
        message for message in caplog.messages if "(ducaheat): update" in message
    ]
    assert ducaheat_logs and "htr/1" in ducaheat_logs[0]
    assert "temp" not in ducaheat_logs[0]


@pytest.mark.asyncio
async def test_ducaheat_translates_path_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Translate Ducaheat ``{"path": ..., "body": ...}`` updates into nodes."""

    client = _make_ducaheat_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    client._nodes_raw = {"htr": {"status": {"1": {"mode": "eco"}}}}
    client._nodes = client._build_nodes_snapshot(client._nodes_raw)
    client._coordinator.data = {
        "device": {
            "nodes": deepcopy(client._nodes_raw),
            "nodes_by_type": {"htr": {"addrs": ["1"]}},
        }
    }
    client._coordinator.update_nodes.reset_mock()
    client._dispatcher_mock.reset_mock()

    payload = {
        "path": "/api/v2/devs/device/htr/1/status",
        "body": {"mode": "comfort"},
    }

    await client._on_update(payload)
    await asyncio.sleep(0)

    assert client._nodes_raw["htr"]["status"]["1"]["mode"] == "comfort"

    assert client._coordinator.update_nodes.call_count == 1
    raw_nodes = client._coordinator.update_nodes.call_args[0][0]
    assert raw_nodes["htr"]["status"]["1"]["mode"] == "comfort"

    dispatched_payloads = [
        call.args[2]
        for call in client._dispatcher_mock.mock_calls
        if len(call.args) >= 3
    ]
    assert any(
        isinstance(payload, Mapping)
        and payload.get("nodes", {})
        .get("htr", {})
        .get("status", {})
        .get("1", {})
        .get("mode")
        == "comfort"
        for payload in dispatched_payloads
    )


@pytest.mark.asyncio
async def test_ducaheat_dev_data_node_list_translated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Translate list-based dev_data snapshots into node dictionaries."""

    client = _make_ducaheat_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    client._coordinator.update_nodes.reset_mock()
    client._dispatcher_mock.reset_mock()

    payload = {
        "nodes": [
            {
                "type": "HTR",
                "addr": 2,
                "status": {"mode": "eco"},
                "setup": {"name": "Heater"},
                "prog": {"weekday": []},
                "prog_temps": {"comfort": 21},
                "advanced_setup": {"boost": False},
            },
            {
                "type": "pmo",
                "addr": "A1",
                "status": {"power": 123},
                "settings": {"tariff": "base"},
                "name": "Monitor",
            },
        ]
    }

    await client._on_dev_data(payload)
    await asyncio.sleep(0)

    assert client._nodes_raw["htr"]["status"]["2"]["mode"] == "eco"
    assert client._nodes_raw["htr"]["settings"]["2"]["setup"]["name"] == "Heater"
    assert client._nodes_raw["htr"]["settings"]["2"]["prog_temps"]["comfort"] == 21
    assert (
        client._nodes_raw["htr"]["advanced"]["2"]["advanced_setup"]["boost"] is False
    )
    assert client._nodes_raw["pmo"]["status"]["A1"]["power"] == 123
    assert client._nodes_raw["pmo"]["settings"]["A1"]["name"] == "Monitor"

    assert client._coordinator.update_nodes.call_count == 1
    raw_nodes = client._coordinator.update_nodes.call_args[0][0]
    assert raw_nodes["htr"]["settings"]["2"]["prog"]["weekday"] == []

    dispatched_payloads = [
        call.args[2]
        for call in client._dispatcher_mock.mock_calls
        if len(call.args) >= 3
    ]
    assert dispatched_payloads
    assert any(
        isinstance(data, Mapping)
        and data.get("nodes", {})
        .get("htr", {})
        .get("status", {})
        .get("2", {})
        .get("mode")
        == "eco"
        for data in dispatched_payloads
    )


def test_translate_nodes_list_skips_invalid_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure non-mapping and incomplete node entries are ignored."""

    client = _make_ducaheat_client(monkeypatch)
    result = client._translate_nodes_list(
        [
            None,
            {"type": "htr", "addr": " ", "status": {"mode": "eco"}},
        ]
    )

    assert result == {}


def test_translate_nodes_list_handles_invalid_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drop entries with unsupported key formats or sections."""

    client = _make_ducaheat_client(monkeypatch)
    result = client._translate_nodes_list(
        [
            {"type": "htr", "addr": 1, 5: {"mode": "eco"}},
            {"type": "htr", "addr": "1", "": {"mode": "eco"}},
        ]
    )

    assert result == {}


def test_ducaheat_summarise_addresses_handles_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure the address summary gracefully handles missing nodes."""

    client = _make_ducaheat_client(monkeypatch)
    assert client._summarise_addresses({}) == "no node addresses"
    assert (
        client._summarise_addresses({"nodes": {"htr": {"settings": {}}}})
        == "no node addresses"
    )


@pytest.mark.asyncio
async def test_ducaheat_connection_lost_triggers_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trigger a coordinator refresh once per interval during fallback."""

    client = _make_ducaheat_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    client._fallback_min_interval = 5.0
    coordinator = client._coordinator
    refresh = coordinator.async_request_refresh
    current_time = 1_000_000.0

    def fake_time() -> float:
        return current_time

    monkeypatch.setattr(module.time, "time", fake_time)

    await client._handle_connection_lost(RuntimeError("ws error"))
    assert refresh.await_count == 0
    assert client._restart_count == 1

    current_time += client._fallback_min_interval
    await client._handle_connection_lost(RuntimeError("ws error"))
    assert refresh.await_count == 1
    state = client._ws_state_bucket()
    assert state["status"] == "fallback"
    assert state["last_fallback_at"] == current_time
    assert state["last_fallback_error"].startswith("RuntimeError")

    await client._handle_connection_lost(RuntimeError("ws error"))
    assert refresh.await_count == 1

    current_time += client._fallback_min_interval
    await client._handle_connection_lost(RuntimeError("ws error"))
    assert refresh.await_count == 2

    client._mark_event(paths=None, count_event=True)
    assert client._restart_count == 0

    current_time += client._fallback_min_interval
    await client._handle_connection_lost(RuntimeError("ws error"))
    assert refresh.await_count == 2

    current_time += client._fallback_min_interval
    await client._handle_connection_lost(RuntimeError("ws error"))
    assert refresh.await_count == 3
    assert client._ws_state_bucket()["status"] == "fallback"


@pytest.mark.asyncio
async def test_ducaheat_fallback_handles_missing_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Handle coordinators that do not expose a refresh coroutine."""

    hass_loop = SimpleNamespace(
        create_task=lambda coro, **_: SimpleNamespace(done=lambda: True),
        call_soon_threadsafe=MagicMock(),
    )
    client = _make_ducaheat_client(monkeypatch, hass_loop=hass_loop)
    client._restart_count = 1
    client._fallback_last_refresh = 0.0
    client._fallback_min_interval = 0.0
    monkeypatch.setattr(module.time, "time", lambda: 1000.0)
    client._coordinator.async_request_refresh = None

    await client._handle_connection_lost(None)

    state = client._ws_state_bucket()
    assert state["status"] == "fallback"
    assert hass_loop.call_soon_threadsafe.call_count == 0


@pytest.mark.asyncio
async def test_ducaheat_fallback_type_error_schedules_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Schedule refresh callbacks when synchronous refresh raises TypeError."""

    call_soon = MagicMock()

    hass_loop = SimpleNamespace(
        create_task=lambda coro, **_: SimpleNamespace(done=lambda: True),
        call_soon_threadsafe=call_soon,
    )
    client = _make_ducaheat_client(monkeypatch, hass_loop=hass_loop)
    client._restart_count = 1
    client._fallback_last_refresh = 0.0
    client._fallback_min_interval = 0.0
    monkeypatch.setattr(module.time, "time", lambda: 1000.0)

    def raise_type_error() -> None:
        raise TypeError("bad signature")

    client._coordinator.async_request_refresh = raise_type_error

    await client._handle_connection_lost(None)

    assert call_soon.call_count == 1


@pytest.mark.asyncio
async def test_ducaheat_fallback_skips_non_coroutine_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return early when the refresh callback is synchronous."""

    hass_loop = SimpleNamespace(
        create_task=lambda coro, **_: SimpleNamespace(done=lambda: True),
        call_soon_threadsafe=MagicMock(),
    )
    client = _make_ducaheat_client(monkeypatch, hass_loop=hass_loop)
    client._restart_count = 1
    client._fallback_last_refresh = 0.0
    client._fallback_min_interval = 0.0
    monkeypatch.setattr(module.time, "time", lambda: 1000.0)
    refresh_mock = MagicMock(return_value=None)
    client._coordinator.async_request_refresh = refresh_mock

    await client._handle_connection_lost(None)

    assert refresh_mock.call_count == 1
    assert hass_loop.call_soon_threadsafe.call_count == 0


@pytest.mark.asyncio
async def test_ducaheat_fallback_logs_refresh_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch exceptions raised by the refresh coroutine."""

    hass_loop = SimpleNamespace(
        create_task=lambda coro, **_: SimpleNamespace(done=lambda: True),
        call_soon_threadsafe=MagicMock(),
    )
    client = _make_ducaheat_client(monkeypatch, hass_loop=hass_loop)
    client._restart_count = 1
    client._fallback_last_refresh = 0.0
    client._fallback_min_interval = 0.0
    monkeypatch.setattr(module.time, "time", lambda: 1000.0)
    refresh_mock = AsyncMock(side_effect=RuntimeError("refresh failed"))
    client._coordinator.async_request_refresh = refresh_mock

    await client._handle_connection_lost(None)

    assert refresh_mock.await_count == 1


@pytest.mark.asyncio
async def test_ducaheat_fallback_skips_recent_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip triggering fallback when payloads arrived moments ago."""

    client = _make_ducaheat_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    client._restart_count = 1
    client._fallback_last_refresh = 0.0
    client._fallback_min_interval = 10.0
    current_time = 1_000_000.0
    client._stats.last_event_ts = current_time - 1
    monkeypatch.setattr(module.time, "time", lambda: current_time)

    await client._handle_connection_lost(None)

    assert client._fallback_last_refresh == 0.0
    assert client._coordinator.async_request_refresh.await_count == 0


@pytest.mark.asyncio
async def test_runner_handles_error_and_reconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    statuses: list[str] = []
    monkeypatch.setattr(client, "_update_status", statuses.append)

    call_log: list[str] = []

    async def fake_connect_once() -> None:
        call_log.append("connect")
        if len(call_log) == 1:
            raise RuntimeError("boom")
        client._closing = True

    async def fake_wait_for_events() -> None:
        call_log.append("wait")
        client._closing = True

    async def fake_disconnect(*, reason: str) -> None:
        call_log.append(f"disconnect:{reason}")

    monkeypatch.setattr(client, "_connect_once", fake_connect_once)
    monkeypatch.setattr(client, "_wait_for_events", fake_wait_for_events)
    monkeypatch.setattr(client, "_disconnect", fake_disconnect)
    monkeypatch.setattr(module.asyncio, "sleep", AsyncMock())

    await client._runner()

    assert call_log == [
        "connect",
        "disconnect:loop cleanup",
        "connect",
        "wait",
        "disconnect:loop cleanup",
    ]
    assert statuses[0] == "starting"
    assert statuses[-1] == "stopped"


@pytest.mark.asyncio
async def test_runner_invokes_hook_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the runner awaits the connection lost hook on errors."""

    client = _make_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    captured: list[Exception | None] = []

    async def failing_connect() -> None:
        raise RuntimeError("boom")

    async def fake_disconnect(*args: Any, **kwargs: Any) -> None:
        return None

    async def hook(err: Exception | None) -> None:
        captured.append(err)
        client._closing = True

    monkeypatch.setattr(client, "_connect_once", failing_connect)
    monkeypatch.setattr(client, "_wait_for_events", AsyncMock())
    monkeypatch.setattr(client, "_disconnect", fake_disconnect)
    monkeypatch.setattr(client, "_handle_connection_lost", hook)

    await client._runner()

    assert len(captured) == 1
    assert isinstance(captured[0], RuntimeError)


@pytest.mark.asyncio
async def test_runner_invokes_hook_on_clean_disconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invoke the connection lost hook when the loop exits cleanly."""

    client = _make_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    captured: list[Exception | None] = []

    async def fake_disconnect(*args: Any, **kwargs: Any) -> None:
        return None

    async def hook(err: Exception | None) -> None:
        captured.append(err)
        client._closing = True

    monkeypatch.setattr(client, "_connect_once", AsyncMock())
    monkeypatch.setattr(client, "_wait_for_events", AsyncMock())
    monkeypatch.setattr(client, "_disconnect", fake_disconnect)
    monkeypatch.setattr(client, "_handle_connection_lost", hook)

    await client._runner()

    assert captured == [None]


@pytest.mark.asyncio
async def test_runner_propagates_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch, hass_loop=asyncio.get_event_loop())

    async def cancel_connect() -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(client, "_connect_once", cancel_connect)
    monkeypatch.setattr(client, "_disconnect", AsyncMock())

    with pytest.raises(asyncio.CancelledError):
        await client._runner()


@pytest.mark.asyncio
async def test_idle_monitor_triggers_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    client._sio.connected = True
    client._payload_idle_window = 10
    client._last_event_at = time.time() - 5
    triggered: list[tuple[float, str]] = []

    def fake_schedule(*, idle_for: float, source: str) -> None:
        triggered.append((idle_for, source))

    client._schedule_idle_restart = fake_schedule  # type: ignore[assignment]
    monkeypatch.setattr(
        client,
        "_refresh_subscription",
        AsyncMock(side_effect=RuntimeError("refresh boom")),
    )
    monkeypatch.setattr(module.asyncio, "sleep", AsyncMock(return_value=None))
    monkeypatch.setattr(client, "_disconnect", AsyncMock())

    await client._idle_monitor()
    assert triggered and triggered[0][1] == "idle monitor refresh failed"
    await asyncio.sleep(0)
    client._cancel_idle_restart()
    assert client._idle_restart_pending is False


@pytest.mark.asyncio
async def test_idle_monitor_exits_when_disconnected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    client._sio.connected = False
    client._disconnected.set()
    monkeypatch.setattr(module.asyncio, "sleep", AsyncMock(return_value=None))
    await client._idle_monitor()


@pytest.mark.asyncio
async def test_idle_monitor_handles_transient_disconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    client._sio.connected = False
    client._disconnected.clear()
    sleep_calls = 0

    async def fake_sleep(_: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls > 1:
            client._closing = True

    monkeypatch.setattr(module.asyncio, "sleep", fake_sleep)
    await client._idle_monitor()
    assert sleep_calls >= 2


@pytest.mark.asyncio
async def test_idle_monitor_skips_when_no_last_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    client._sio.connected = True
    client._last_event_at = None
    client._stats.last_event_ts = 0

    async def fake_sleep(_: float) -> None:
        client._closing = True

    monkeypatch.setattr(module.asyncio, "sleep", fake_sleep)
    await client._idle_monitor()


@pytest.mark.asyncio
async def test_idle_monitor_retries_failed_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry idle refreshes when the previous keep-alive failed."""

    client = _make_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    client._sio.connected = True
    client._payload_idle_window = 10
    client._last_event_at = time.time() - 5
    client._subscription_refresh_failed = True

    triggered: list[str] = []

    def fake_schedule(*, idle_for: float, source: str) -> None:
        triggered.append(source)

    client._schedule_idle_restart = fake_schedule  # type: ignore[assignment]

    async def failing_refresh(*, reason: str) -> None:
        raise RuntimeError("refresh boom")

    monkeypatch.setattr(client, "_refresh_subscription", failing_refresh)
    monkeypatch.setattr(module.asyncio, "sleep", AsyncMock(return_value=None))

    await client._idle_monitor()

    assert triggered == ["idle monitor retry failed"]


@pytest.mark.asyncio
async def test_idle_monitor_refresh_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refresh idle websocket session without scheduling a restart."""

    client = _make_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    client._sio.connected = True
    client._payload_idle_window = 1
    client._last_event_at = time.time() - 5

    calls: list[str] = []

    async def refresh(*, reason: str) -> None:
        calls.append(reason)
        client._closing = True

    monkeypatch.setattr(client, "_refresh_subscription", refresh)
    monkeypatch.setattr(module.asyncio, "sleep", AsyncMock(return_value=None))

    await client._idle_monitor()

    assert calls == ["idle monitor"]


def test_start_and_stop_cancel_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    loop = asyncio.new_event_loop()
    hass_loop = SimpleNamespace(create_task=loop.create_task)
    client = _make_client(monkeypatch, hass_loop=hass_loop)

    async def dummy_runner(self: module.WebSocketClient) -> None:
        await asyncio.sleep(0)

    monkeypatch.setattr(module.WebSocketClient, "_runner", dummy_runner)

    task = client.start()
    assert client.start() is task

    client._idle_restart_task = loop.create_task(asyncio.sleep(0))
    client._idle_monitor_task = loop.create_task(asyncio.sleep(0))
    client._sio.connected = True
    client._sio.disconnect = AsyncMock()

    loop.run_until_complete(client.stop())
    assert client._sio.disconnect.await_count == 1

    loop.run_until_complete(asyncio.sleep(0))
    loop.close()


@pytest.mark.asyncio
async def test_refresh_subscription_requires_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail keep-alive refresh when the websocket is disconnected."""

    client = _make_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    client._sio.connected = False

    with pytest.raises(RuntimeError, match="websocket not connected"):
        await client._refresh_subscription(reason="unit-test")


@pytest.mark.asyncio
async def test_refresh_subscription_updates_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful keep-alive refresh updates bookkeeping fields."""

    client = _make_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    client._sio.connected = True

    emit_calls: list[tuple[str, Any | None, str | None]] = []

    async def fake_emit(
        event: str,
        data: Any | None = None,
        *,
        namespace: str | None = None,
    ) -> None:
        emit_calls.append((event, data, namespace))

    client._sio.emit = fake_emit  # type: ignore[assignment]
    subscribe_mock = AsyncMock()
    monkeypatch.setattr(client, "_subscribe_heater_samples", subscribe_mock)
    monkeypatch.setattr(module.time, "time", lambda: 1000.0)
    monkeypatch.setattr(module._LOGGER, "isEnabledFor", lambda level: True)

    info_calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        module._LOGGER, "info", lambda *args, **kwargs: info_calls.append(args)
    )

    await client._refresh_subscription(reason="unit")

    assert emit_calls[0] == ("dev_data", None, module.WS_NAMESPACE)
    assert subscribe_mock.await_count == 1
    assert client._subscription_refresh_failed is False
    assert client._subscription_refresh_last_success == pytest.approx(1000.0)
    assert any("unit" in " ".join(str(part) for part in call) for call in info_calls)


def test_forward_sample_updates_handles_missing_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gracefully handle missing coordinator records for sample forwarding."""

    client = _make_client(monkeypatch)

    client.hass.data = {}
    client._forward_sample_updates({"htr": {"1": {}}})

    client.hass.data = {module.DOMAIN: {"entry": {"energy_coordinator": object()}}}
    client._forward_sample_updates({"htr": {"1": {}}})

    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    class Handler:
        def handle_ws_samples(self, *args: Any, **kwargs: Any) -> None:
            calls.append((args, kwargs))

    client.hass.data = {module.DOMAIN: {"entry": {"energy_coordinator": Handler()}}}
    client._forward_sample_updates({"htr": {"1": {"samples": []}}})

    assert calls
    args, kwargs = calls[0]
    assert args[0] == "device"
    assert kwargs == {}


def test_dispatch_nodes_records_unknown_types(monkeypatch: pytest.MonkeyPatch) -> None:
    """Include unknown node types in the dispatched payload copy."""

    client = _make_client(monkeypatch)
    record = {
        "nodes": {"nodes": [{"type": "foo", "addr": "1"}, {"type": "htr", "addr": "A"}]}
    }
    client.hass.data[module.DOMAIN] = {"entry": record}
    client._dispatcher_mock.reset_mock()

    snapshot = {
        "nodes": {
            "htr": {
                "settings": {"A": {}},
                "addrs": ["A"],
                "advanced": {},
                "samples": {},
            }
        },
        "nodes_by_type": {
            "htr": {
                "addrs": ["A"],
                "settings": {"A": {}},
                "advanced": {},
                "samples": {},
            }
        },
    }

    addr_map = client._dispatch_nodes(snapshot)

    payload = client._dispatcher_mock.call_args[0][2]
    assert payload["unknown_types"] == ["foo"]
    assert addr_map["htr"] == ["A"]


@pytest.mark.asyncio
async def test_connect_once_invokes_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    client._stop_event = asyncio.Event()
    monkeypatch.setattr(
        client,
        "_build_engineio_target",
        AsyncMock(return_value=("https://socket", "socket.io")),
    )
    connect_mock = AsyncMock()
    client._sio.connect = connect_mock
    await client._connect_once()
    connect_mock.assert_awaited()


@pytest.mark.asyncio
async def test_connect_once_respects_stop_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    client._stop_event = asyncio.Event()
    client._stop_event.set()
    client._sio.connect = AsyncMock()
    await client._connect_once()
    client._sio.connect.assert_not_called()


@pytest.mark.asyncio
async def test_wait_for_events_handles_disconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    client._stop_event = asyncio.Event()
    client._disconnected = asyncio.Event()
    client._stop_event.set()
    await client._wait_for_events()


@pytest.mark.asyncio
async def test_disconnect_logs_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    client._sio.connected = True
    client._sio.disconnect = AsyncMock(side_effect=RuntimeError("boom"))
    await client._disconnect(reason="test")
    assert client._disconnected.is_set()


@pytest.mark.asyncio
async def test_get_token_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    client._client._authed_headers = AsyncMock(return_value={})  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError):
        await client._get_token()


def test_api_base_default(monkeypatch: pytest.MonkeyPatch) -> None:
    rest = DummyREST(base="")
    client = _make_client(monkeypatch, rest=rest)
    assert client._api_base() == module.API_BASE


@pytest.mark.asyncio
async def test_force_refresh_token_resets_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    rest = client._client
    rest._access_token = "abc"  # type: ignore[attr-defined]
    await client._force_refresh_token()
    assert rest._ensure_token.await_count == 1  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_build_engineio_target_handles_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rest = DummyREST(base="http://")
    client = _make_client(monkeypatch, rest=rest, hass_loop=asyncio.get_event_loop())
    with pytest.raises(RuntimeError):
        await client._build_engineio_target()


@pytest.mark.asyncio
async def test_on_connect_and_disconnect_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch)
    client._loop = asyncio.get_event_loop()
    client._sio.emit = AsyncMock()
    client._idle_monitor_task = asyncio.create_task(asyncio.sleep(0))
    await asyncio.sleep(0)
    await client._on_connect()
    client._sio.emit.assert_not_awaited()
    await client._on_disconnect()


@pytest.mark.asyncio
async def test_on_namespace_connect_requests_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure the websocket client requests the initial device snapshot."""

    client = _make_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    client._sio.emit = AsyncMock()
    await client._on_namespace_connect()
    expected = [call("dev_data", namespace=client._namespace)]
    if client._namespace != "/":
        expected.insert(0, call("join", namespace=client._namespace))
    assert client._sio.emit.await_args_list == expected
    await client._on_disconnect()


@pytest.mark.asyncio
async def test_on_connect_debug_catch_all(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Verify that catch-all logging is registered only in DEBUG mode."""

    client = _make_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    client._sio.emit = AsyncMock()
    caplog.set_level(logging.DEBUG, logger=module.__name__)

    await client._on_connect()

    catch_all = client._sio.events.get(("*", client._namespace))
    assert catch_all is not None

    await catch_all("dev_handshake", {"ok": True})
    assert "catch-all" in caplog.text

    caplog.clear()
    caplog.set_level(logging.INFO, logger=module.__name__)
    await catch_all("dev_data", {"another": True})
    assert "catch-all" not in caplog.text

    client._register_debug_catch_all()

    await client._on_disconnect()


@pytest.mark.asyncio
async def test_on_namespace_connect_emits_join_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    client._sio.emit = AsyncMock(side_effect=RuntimeError("boom"))
    await client._on_connect()
    await client._on_namespace_connect()
    assert client._idle_monitor_task is not None
    client._idle_monitor_task.cancel()
    with suppress(asyncio.CancelledError):
        await client._idle_monitor_task


@pytest.mark.asyncio
async def test_on_reconnect_logs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    client = _make_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    caplog.set_level(logging.DEBUG)
    await client._on_reconnect()
    assert "reconnect" in caplog.text


def test_collect_update_addresses_extracts(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch)
    nodes = {
        "htr": {"settings": {"1": {"temp": 20}, "2": None}},
        "aux": {"samples": {"3": 10}},
    }
    addresses = client._collect_update_addresses(nodes)
    assert addresses == [("aux", "3"), ("htr", "1")]


def test_collect_update_addresses_skips_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client(monkeypatch)
    nodes = {"htr": [1, 2], 10: {"settings": {"1": {}}}}
    assert client._collect_update_addresses(nodes) == []


def test_collect_update_addresses_skips_non_mapping_sections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client(monkeypatch)
    nodes = {"htr": {"settings": [1, 2]}}
    assert client._collect_update_addresses(nodes) == []


def test_extract_nodes_failure_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch)
    assert client._extract_nodes([]) is None
    assert client._extract_nodes({}) is None


def test_apply_heater_addresses_updates_energy(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch)
    energy_coordinator = SimpleNamespace(updated=None)

    def update_addresses(mapping: dict[str, list[str]]) -> None:
        energy_coordinator.updated = mapping

    energy_coordinator.update_addresses = update_addresses  # type: ignore[attr-defined]
    client.hass.data[module.DOMAIN]["entry"] = {
        "energy_coordinator": energy_coordinator
    }
    initial_data = {"device": {"nodes_by_type": {}}}
    client._coordinator.data = initial_data
    result = client._apply_heater_addresses({"htr": [1, 2]})
    assert result["htr"] == ["1", "2"]
    assert energy_coordinator.updated == {"htr": ["1", "2"]}
    assert client._coordinator.data is not initial_data


def test_apply_nodes_payload_forwards_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sample updates should be forwarded to the energy coordinator."""

    client = _make_client(monkeypatch)
    calls: list[tuple[str, Mapping[str, Mapping[str, Any]]]] = []

    def handle_ws_samples(
        dev_id: str,
        updates: Mapping[str, Mapping[str, Any]],
    ) -> None:
        calls.append((dev_id, updates))

    energy_coordinator = SimpleNamespace(handle_ws_samples=handle_ws_samples)
    client.hass.data[module.DOMAIN]["entry"]["energy_coordinator"] = energy_coordinator

    payload = {
        "nodes": {
            "htr": {
                "samples": {
                    " 1 ": [{"t": 10.0, "counter": 1000}],
                }
            }
        }
    }

    client._apply_nodes_payload(payload, merge=False, event="dev_data")

    assert calls
    dev_id, updates = calls[0]
    assert dev_id == "device"
    assert "htr" in updates
    assert updates["htr"] == {"1": [{"t": 10.0, "counter": 1000}]}


def test_dispatch_nodes_handles_raw_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch)
    updates: list[dict[str, list[str]]] = []

    def record_update(mapping: dict[str, list[str]]) -> None:
        updates.append(mapping)

    client.hass.data[module.DOMAIN]["entry"] = {
        "energy_coordinator": SimpleNamespace(update_addresses=record_update)
    }
    client._coordinator.update_nodes = MagicMock()
    payload = {"htr": {"settings": {"1": {"temp": 20}}}}
    result = client._dispatch_nodes(payload)
    assert isinstance(result, dict)
    client._coordinator.update_nodes.assert_called()
    record = client.hass.data[module.DOMAIN]["entry"]
    assert "nodes" in record
    assert isinstance(client._coordinator.data, dict)
    assert updates and isinstance(updates[0], dict)


def test_dispatch_nodes_updates_snapshot_record(monkeypatch: pytest.MonkeyPatch) -> None:
    """Snapshot records should be updated when new nodes payloads arrive."""

    client = _make_client(monkeypatch)
    base_nodes = {"nodes": [{"type": "htr", "addr": "1", "name": "Heater"}]}
    snapshot = InstallationSnapshot(dev_id="device", raw_nodes=base_nodes)
    energy_coordinator = SimpleNamespace(update_addresses=MagicMock())
    record = {"snapshot": snapshot, "energy_coordinator": energy_coordinator}
    client.hass.data = {module.DOMAIN: {client.entry_id: record}}
    client._coordinator.update_nodes = MagicMock()
    client._coordinator.data = {client.dev_id: {}}

    payload = {
        "nodes": base_nodes,
        "nodes_by_type": {"htr": {"addrs": ["1"], "settings": {"1": {}}, "advanced": {}, "samples": {}}},
    }

    addr_map = client._dispatch_nodes(payload)

    assert addr_map["htr"] == ["1"]
    assert "node_inventory" in record
    assert energy_coordinator.update_addresses.call_count == 1
    client._coordinator.update_nodes.assert_called_once()


def test_heater_sample_subscription_targets_with_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Derive heater sample subscriptions from snapshot-backed records."""

    client = _make_client(monkeypatch)
    nodes_payload = {"nodes": [{"type": "htr", "addr": "1"}, {"type": "acm", "addr": "2"}]}
    snapshot = InstallationSnapshot(dev_id="device", raw_nodes=nodes_payload)
    energy_coordinator = SimpleNamespace(update_addresses=MagicMock())
    record = {"snapshot": snapshot, "energy_coordinator": energy_coordinator}
    client.hass.data = {module.DOMAIN: {client.entry_id: record}}
    client._coordinator.data = {client.dev_id: {}}

    inventory = list(snapshot.inventory)

    def fake_collect(record_data, *, coordinator=None):
        return inventory, {"htr": ["1"], "acm": ["2"]}, {"htr": "htr"}

    monkeypatch.setattr(module, "collect_heater_sample_addresses", fake_collect)

    targets = client._heater_sample_subscription_targets()

    assert targets
    assert record["node_inventory"]
    energy_coordinator.update_addresses.assert_called_once()
    coordinator_data = client._coordinator.data[client.dev_id]
    assert "nodes_by_type" in coordinator_data
    assert coordinator_data["nodes_by_type"]["htr"]["addrs"]


def test_schedule_and_cancel_idle_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    loop = asyncio.new_event_loop()
    client = _make_client(monkeypatch, hass_loop=loop)
    client._disconnect = AsyncMock()
    client._schedule_idle_restart(idle_for=10, source="test")
    assert client._idle_restart_task is not None
    assert client._idle_restart_pending is True
    loop.run_until_complete(asyncio.sleep(0))
    assert client._disconnect.await_count == 1
    client._cancel_idle_restart()
    loop.close()


def test_handle_handshake_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch)
    client._handle_handshake("not a dict")


def test_apply_heater_addresses_inventory_and_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client(monkeypatch)
    client.hass.data[module.DOMAIN]["entry"] = {}
    client._coordinator.data = {"device": {"nodes_by_type": {}}}
    result = client._apply_heater_addresses({}, inventory=["inv"])
    assert client.hass.data[module.DOMAIN]["entry"]["node_inventory"] == ["inv"]
    assert result == {"htr": []}


def test_apply_heater_addresses_skips_empty_non_heater(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client(monkeypatch)
    client.hass.data[module.DOMAIN]["entry"] = {}
    client._coordinator.data = {"device": {"nodes_by_type": {}}}
    result = client._apply_heater_addresses({"acm": []})
    assert result.get("acm") == []
    dev_map = client._coordinator.data.get(client.dev_id, {})  # type: ignore[assignment]
    assert "acm" not in dev_map.get("nodes_by_type", {})


def test_merge_nodes_handles_non_dict() -> None:
    target = {"a": {"b": 1}}
    module.WebSocketClient._merge_nodes(target, {"a": 5, "c": 6})
    assert target == {"a": 5, "c": 6}


def test_merge_nodes_replaces_non_mapping_target() -> None:
    target = {"a": 1}
    module.WebSocketClient._merge_nodes(target, {"a": {"b": 2}})
    assert target["a"] == {"b": 2}


def test_mark_event_with_paths(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    client = _make_client(monkeypatch)
    caplog.set_level(logging.DEBUG)
    client._mark_event(paths=["/a", "/a", "/b"], count_event=False)
    assert client._stats.events_total == 1
    assert client._stats.last_paths == ["/a", "/b"]


def test_mark_event_limits_last_paths(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    client = _make_client(monkeypatch)
    caplog.set_level(logging.DEBUG)
    paths = [f"/{i}" for i in range(10)]
    client._mark_event(paths=paths, count_event=False)
    assert len(client._stats.last_paths or []) == 5


def test_mark_event_counts_without_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch)
    client._mark_event(paths=None, count_event=True)
    assert client._stats.events_total == 1


def test_schedule_idle_restart_early_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch)
    client._closing = True
    client._schedule_idle_restart(idle_for=10, source="test")
    assert client._idle_restart_task is None


def test_schedule_idle_restart_when_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch)
    client._idle_restart_pending = True
    client._schedule_idle_restart(idle_for=10, source="test")
    assert client._idle_restart_task is None


def test_cancel_idle_restart_with_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    loop = asyncio.new_event_loop()
    client = _make_client(monkeypatch, hass_loop=loop)
    task = loop.create_task(asyncio.sleep(0.1))
    client._idle_restart_task = task
    client._idle_restart_pending = True
    client._cancel_idle_restart()
    assert client._idle_restart_task is None
    assert client._idle_restart_pending is False
    loop.run_until_complete(asyncio.sleep(0))
    assert task.cancelled()
    loop.close()


@pytest.mark.asyncio
async def test_ducaheat_client_extended_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = module.DucaheatWSClient(
        SimpleNamespace(loop=asyncio.get_event_loop(), data={}),
        entry_id="entry",
        dev_id="device",
        api_client=DummyREST(),
        coordinator=SimpleNamespace(data={}, update_nodes=MagicMock()),
    )
    client._sio.emit = AsyncMock()
    await client._on_connect()
    client._sio.emit.assert_not_awaited()
    await client._on_disconnect()
    await client._on_dev_handshake({})
    await client._on_dev_data({"nodes": {}})
    await client._on_update({"nodes": {}})


@pytest.mark.asyncio
async def test_ducaheat_connect_matches_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the Ducaheat client connects with the minimal reference contract."""

    client = _make_ducaheat_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    client._stop_event = asyncio.Event()
    monkeypatch.setattr(client, "_get_token", AsyncMock(return_value="abc"))
    monkeypatch.setattr(
        client, "_api_base", lambda: "https://api-tevolve.termoweb.net/api/v2"
    )
    connect_mock = AsyncMock()
    client._sio.connect = connect_mock

    await client._connect_once()

    assert connect_mock.await_count == 1
    args, kwargs = connect_mock.await_args
    assert args == (
        "https://api-tevolve.termoweb.net/socket.io?token=abc&dev_id=device",
    )
    assert kwargs["headers"] == {
        "Origin": "https://localhost",
        "User-Agent": module.USER_AGENT,
        "Accept-Language": module.ACCEPT_LANGUAGE,
        "X-Requested-With": "net.termoweb.ducaheat.app",
    }
    assert kwargs["transports"] == ["polling", "websocket"]
    assert kwargs["namespaces"] == ["/"]
    assert kwargs["socketio_path"] == "socket.io"
    assert kwargs["wait"] is True
    assert kwargs["wait_timeout"] == 15


@pytest.mark.asyncio
async def test_ducaheat_build_engineio_target(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure Ducaheat Engine.IO targets use the root namespace path."""

    client = _make_ducaheat_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    monkeypatch.setattr(client, "_get_token", AsyncMock(return_value="TOKEN"))
    monkeypatch.setattr(
        client, "_api_base", lambda: "https://api-tevolve.termoweb.net/api/v2"
    )

    url, path = await client._build_engineio_target()

    assert url == "https://api-tevolve.termoweb.net/socket.io?token=TOKEN&dev_id=device"
    assert path == "socket.io"


@pytest.mark.asyncio
async def test_ducaheat_connect_honours_stop_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure the Ducaheat client aborts when stop was requested."""

    client = _make_ducaheat_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    client._stop_event = asyncio.Event()
    client._stop_event.set()
    token_mock = AsyncMock()
    monkeypatch.setattr(client, "_get_token", token_mock)
    client._sio.connect = AsyncMock()

    await client._connect_once()

    token_mock.assert_not_awaited()
    client._sio.connect.assert_not_called()


@pytest.mark.asyncio
async def test_ducaheat_connect_logs_request_details(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Ensure connect attempts emit detailed debug logging."""

    client = _make_ducaheat_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    client._stop_event = asyncio.Event()
    caplog.set_level(logging.DEBUG)
    connect_mock = AsyncMock()
    client._sio.connect = connect_mock

    await client._connect_once()

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "request headers" in messages
    assert "connect target base" in messages
    assert "to***en" in messages
    connect_mock.assert_awaited_once()


def test_ducaheat_connect_response_logging(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify the response logger captures engine.io metadata."""

    client = _make_ducaheat_client(monkeypatch)
    caplog.set_level(logging.DEBUG)
    client._connect_response_logged = False
    client._sio.connection_url = "https://api.example.com/socket.io?token=abc"
    client._sio.connection_headers = {"Authorization": "Bearer secret-token"}
    client._sio.connection_transports = ["websocket"]
    client._sio.connection_namespaces = ["/"]
    response = SimpleNamespace(
        status=101,
        headers={"Upgrade": "websocket", "Set-Cookie": "SESSION=abcdef"},
    )
    client._sio.eio = SimpleNamespace(
        sid="sid123",
        transport="websocket",
        ws=SimpleNamespace(response=response),
    )

    client._log_connect_response()

    text = caplog.text
    assert "sid123" in text
    assert "websocket response headers" in text
    assert "Bearer secret-token" not in text
    assert "Bearer secr" in text
    assert "SESSION=abcdef" not in text


def test_ducaheat_helper_sanitisation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise helper sanitisation branches for coverage."""

    client = _make_ducaheat_client(monkeypatch)
    assert client._redact_value("") == ""
    assert client._redact_value("abcd") == "***"
    assert client._redact_value("abcdef") == "ab***ef"
    assert client._redact_value("abcdefghijkl") == "abcd...ijkl"

    headers = client._sanitise_headers(
        {
            "Authorization": "Bearer secret",
            "Cookie": "SESSIONID",
            "X-Bytes": b"value",
        }
    )
    assert "secret" not in headers["Authorization"]
    assert headers["Cookie"] != "SESSIONID"
    assert headers["X-Bytes"] == "value"

    params = client._sanitise_params({"token": "secret", "dev_id": "device"})
    assert params["token"] != "secret"
    assert params["dev_id"] == "de...ce"

    headers_no_token = client._sanitise_headers({"Authorization": "Bearer"})
    assert headers_no_token["Authorization"].startswith("Be")


def test_ducaheat_connect_response_no_engineio(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Ensure connect response logging tolerates missing engine.io context."""

    client = _make_ducaheat_client(monkeypatch)
    caplog.set_level(logging.DEBUG)
    client._connect_response_logged = False
    client._sio.connection_url = "https://example/ws"
    client._sio.eio = None

    client._log_connect_response()

    assert "connected URL" in caplog.text


def test_ducaheat_connect_response_type_error_headers(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Cover the fallback branch when response headers do not coerce to dict."""

    client = _make_ducaheat_client(monkeypatch)
    caplog.set_level(logging.DEBUG)
    client._connect_response_logged = False
    bad_headers = SimpleNamespace(items=lambda: {"Upgrade": "websocket"})
    response = SimpleNamespace(status=101, headers=bad_headers)
    client._sio.eio = SimpleNamespace(
        sid=None,
        transport=None,
        ws=SimpleNamespace(response=response),
    )

    client._log_connect_response()

    assert "websocket response headers" in caplog.text


def test_ducaheat_connect_response_skips_repeat(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Ensure repeated calls do not re-log response details."""

    client = _make_ducaheat_client(monkeypatch)
    caplog.set_level(logging.DEBUG)
    client._connect_response_logged = False
    client._sio.connection_url = "https://example/ws"
    client._sio.eio = SimpleNamespace(ws=SimpleNamespace(response=None))

    client._log_connect_response()
    first_count = len(caplog.records)
    client._log_connect_response()
    assert len(caplog.records) == first_count


@pytest.mark.asyncio
async def test_ducaheat_on_namespace_connect_emits_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate connect handler joins and requests data on the API namespace."""

    client = _make_ducaheat_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    emit_mock = AsyncMock()
    client._sio.emit = emit_mock

    await client._on_namespace_connect()

    assert emit_mock.await_args_list == [call("dev_data", namespace="/")]


@pytest.mark.asyncio
async def test_dev_data_triggers_sample_subscriptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    entry = client.hass.data[module.DOMAIN]["entry"]

    def fake_collect(record: Any, *, coordinator: Any = None) -> Any:
        assert record is entry
        assert coordinator is client._coordinator
        return (["inventory"], {"htr": ["1", "2"], "acm": ["7"]}, {})

    monkeypatch.setattr(module, "collect_heater_sample_addresses", fake_collect)
    emit_mock = AsyncMock()
    client._sio.emit = emit_mock

    await client._on_dev_data({"nodes": {}})

    expected = [
        call("subscribe", "/htr/1/samples", namespace=module.WS_NAMESPACE),
        call("subscribe", "/htr/2/samples", namespace=module.WS_NAMESPACE),
        call("subscribe", "/acm/7/samples", namespace=module.WS_NAMESPACE),
    ]
    assert emit_mock.await_args_list == expected


@pytest.mark.asyncio
async def test_reconnect_resubscribes_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    entry = client.hass.data[module.DOMAIN]["entry"]

    def fake_collect(record: Any, *, coordinator: Any = None) -> Any:
        assert record is entry
        assert coordinator is client._coordinator
        return (["inventory"], {"htr": ["9"]}, {})

    monkeypatch.setattr(module, "collect_heater_sample_addresses", fake_collect)
    emit_mock = AsyncMock()
    client._sio.emit = emit_mock

    await client._on_reconnect()

    assert emit_mock.await_args_list == [
        call("subscribe", "/htr/9/samples", namespace=module.WS_NAMESPACE)
    ]


@pytest.mark.asyncio
async def test_sample_subscription_uses_coordinator_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    entry = client.hass.data[module.DOMAIN]["entry"]
    client._coordinator._addrs = lambda: ["4"]  # type: ignore[attr-defined]

    def fake_collect(record: Any, *, coordinator: Any | None = None) -> Any:
        assert record is entry
        assert coordinator is client._coordinator
        fallback = coordinator._addrs() if coordinator else []  # type: ignore[attr-defined]
        assert fallback == ["4"]
        return (
            [SimpleNamespace(type="acm", addr="2")],
            {"htr": list(fallback), "acm": ["2"]},
            {},
        )

    monkeypatch.setattr(module, "collect_heater_sample_addresses", fake_collect)
    emit_mock = AsyncMock()
    client._sio.emit = emit_mock

    await client._on_dev_data({"nodes": {}})

    assert emit_mock.await_args_list == [
        call("subscribe", "/htr/4/samples", namespace=module.WS_NAMESPACE),
        call("subscribe", "/acm/2/samples", namespace=module.WS_NAMESPACE),
    ]


def test_heater_sample_subscription_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the helper returns ordered subscription targets."""

    client = _make_client(monkeypatch)
    entry = client.hass.data[module.DOMAIN]["entry"]

    def fake_collect(
        record: Mapping[str, Any], *, coordinator: Any | None = None
    ) -> Any:
        assert record is entry
        assert coordinator is client._coordinator
        return (
            [SimpleNamespace(type="htr", addr="ignored")],
            {"acm": ["2"], "htr": ["1", "3"]},
            {},
        )

    monkeypatch.setattr(module, "collect_heater_sample_addresses", fake_collect)

    targets = client._heater_sample_subscription_targets()

    assert targets == [("htr", "1"), ("htr", "3"), ("acm", "2")]


@pytest.mark.asyncio
async def test_sample_subscription_logs_helper_errors(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    client = _make_client(monkeypatch, hass_loop=asyncio.get_event_loop())

    def raising_collect(*_: Any, **__: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(module, "collect_heater_sample_addresses", raising_collect)
    emit_mock = AsyncMock()
    client._sio.emit = emit_mock

    caplog.set_level(logging.DEBUG)
    await client._on_dev_data({"nodes": {}})

    assert "sample subscription setup failed" in caplog.text
    emit_mock.assert_not_called()


@pytest.mark.asyncio
async def test_sample_subscription_handles_fallback_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client(monkeypatch, hass_loop=asyncio.get_event_loop())

    def fake_collect(record: Any, *, coordinator: Any | None = None) -> Any:
        assert coordinator is client._coordinator
        try:
            coordinator._addrs()  # type: ignore[attr-defined]
        except TypeError:
            pass
        return (
            [SimpleNamespace(type="acm", addr="1")],
            {"acm": ["1"]},
            {},
        )

    monkeypatch.setattr(module, "collect_heater_sample_addresses", fake_collect)

    def bad_addrs() -> None:
        raise TypeError

    client._coordinator._addrs = bad_addrs  # type: ignore[attr-defined]
    emit_mock = AsyncMock()
    client._sio.emit = emit_mock

    await client._on_dev_data({"nodes": {}})

    assert emit_mock.await_args_list == [
        call("subscribe", "/acm/1/samples", namespace=module.WS_NAMESPACE)
    ]


@pytest.mark.asyncio
async def test_legacy_sample_subscription_uses_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_legacy_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    entry = client.hass.data[module.DOMAIN]["entry"]

    def fake_collect(record: Any, *, coordinator: Any = None) -> Any:
        assert record is entry
        assert coordinator is client._coordinator
        return (["inventory"], {"htr": ["5"], "acm": ["8"]}, {})

    monkeypatch.setattr(module, "collect_heater_sample_addresses", fake_collect)

    send_mock = AsyncMock()
    client._send_text = send_mock  # type: ignore[assignment]

    await client._subscribe_htr_samples()

    expected_payloads = [
        f'5::{module.WS_NAMESPACE}:{{"name":"subscribe","args":["/htr/5/samples"]}}',
        f'5::{module.WS_NAMESPACE}:{{"name":"subscribe","args":["/acm/8/samples"]}}',
    ]
    assert [call.args[0] for call in send_mock.await_args_list] == expected_payloads


@pytest.mark.asyncio
async def test_legacy_session_subscription_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure the legacy session subscription uses the expected frame."""

    client = _make_legacy_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    send_mock = AsyncMock()
    client._send_text = send_mock  # type: ignore[assignment]

    await client._subscribe_session_metadata()

    send_mock.assert_awaited_once_with(
        f'5::{module.WS_NAMESPACE}:{{"name":"subscribe","args":["/mgr/session"]}}'
    )


@pytest.mark.asyncio
async def test_ducaheat_client_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    client = module.DucaheatWSClient(
        SimpleNamespace(loop=asyncio.get_event_loop(), data={}),
        entry_id="entry",
        dev_id="device",
        api_client=DummyREST(),
        coordinator=SimpleNamespace(data={}, update_nodes=MagicMock()),
    )
    await client._on_dev_handshake({})
    await client._on_dev_data({"nodes": {}})
    await client._on_update({"nodes": {}})


@pytest.mark.asyncio
async def test_ducaheat_debug_logging(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    client = module.DucaheatWSClient(
        SimpleNamespace(loop=asyncio.get_event_loop(), data={}),
        entry_id="entry",
        dev_id="device",
        api_client=DummyREST(),
        coordinator=SimpleNamespace(data={}, update_nodes=MagicMock()),
    )
    await client._on_connect()
    await client._on_disconnect()
    await client._on_dev_handshake({})
    await client._on_dev_data({"nodes": {}})
    await client._on_update({"nodes": {}})
    assert caplog.text.count("ducaheat") >= 4


@pytest.mark.asyncio
async def test_ducaheat_handles_root_namespace_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure updates arriving on '/' still trigger handlers."""

    client = _make_ducaheat_client(
        monkeypatch,
        hass_loop=asyncio.get_event_loop(),
        namespace="/",
    )
    monkeypatch.setattr(
        module,
        "collect_heater_sample_addresses",
        lambda *a, **k: ([], {}, {}),
    )
    handler = client._sio.events[("dev_data", "/")]

    await handler({"nodes": {}})

    assert client._stats.frames_total == 1
