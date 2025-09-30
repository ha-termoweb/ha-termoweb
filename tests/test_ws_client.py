import asyncio
import logging
import time
from contextlib import suppress
from types import SimpleNamespace
from typing import Any, Mapping, cast
from unittest.mock import AsyncMock, MagicMock, call

import pytest

import custom_components.termoweb.ws_client as module


class DummyREST:
    """Minimal REST client stub for websocket tests."""

    def __init__(self, base: str = "https://api.example.com/api/v2") -> None:
        self.api_base = base
        self._session = SimpleNamespace()
        self._headers = {"Authorization": "Bearer token"}
        self._ensure_token = AsyncMock()

    async def _authed_headers(self) -> dict[str, str]:
        return self._headers


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

        def on(
            self, event: str, *, handler: Any, namespace: str | None = None
        ) -> None:
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
    coordinator = SimpleNamespace(data={}, update_nodes=MagicMock())
    rest_client = DummyREST()
    dispatcher_mock = MagicMock()
    monkeypatch.setattr(module, "async_dispatcher_send", dispatcher_mock)
    client = module.DucaheatWSClient(
        hass,
        entry_id="entry",
        dev_id="device",
        api_client=rest_client,
        coordinator=coordinator,
        session=SimpleNamespace(),
    )
    client._dispatcher_mock = dispatcher_mock  # type: ignore[attr-defined]
    return client


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
    assert f"namespace disconnect ({module.WS_NAMESPACE}): transport closed" in caplog.text


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


def test_http_wrapping_handles_missing_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
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

        def on(
            self, event: str, *, handler: Any, namespace: str | None = None
        ) -> None:
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
    client = _make_client(monkeypatch, hass_loop=asyncio.get_event_loop())

    ws_url = await client.ws_url()
    assert (
        ws_url
        == "https://api.example.com/api/v2/socket_io?token=token&dev_id=device"
    )

    base, path = await client._build_engineio_target()
    assert (
        base
        == "https://api.example.com/api/v2/socket_io?token=token&dev_id=device"
    )
    assert path == "api/v2/socket_io"


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


def test_wrap_background_task_with_sync_function(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setattr(client, "_restart_subscription_refresh", MagicMock())
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


def test_legacy_event_configures_subscription(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the legacy client extracts TTL metadata from data payloads."""

    client = _make_legacy_client(monkeypatch)
    refresh_loop = AsyncMock()
    monkeypatch.setattr(client, "_subscription_refresh_loop", refresh_loop)
    created: list[dict[str, Any]] = []

    def capture_task(coro: Any, **kwargs: Any) -> Any:
        created.append({"coro": coro, "kwargs": kwargs})
        closer = getattr(coro, "close", None)
        if callable(closer):
            closer()
        return SimpleNamespace(done=lambda: True)

    client._loop.create_task = capture_task  # type: ignore[assignment]
    monkeypatch.setattr(module.time, "time", lambda: 100.0)

    event = {
        "name": "data",
        "args": [[{"path": "/mgr/session", "body": {"lease": {"ttl": 250}}}]],
    }
    client._handle_event(event)

    assert client._subscription_ttl == pytest.approx(250.0)
    assert client._payload_idle_window == pytest.approx(375.0)
    assert client._subscription_refresh_due == pytest.approx(350.0)
    assert client._legacy_subscription_configured is True
    assert client._subscription_refresh_task is not None
    assert refresh_loop.call_count == 1
    assert created


def test_legacy_event_updates_ttl_after_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure default TTL is used until a payload provides an explicit value."""

    client = _make_legacy_client(monkeypatch)
    refresh_loop = AsyncMock()
    monkeypatch.setattr(client, "_subscription_refresh_loop", refresh_loop)
    tasks: list[Any] = []

    def capture_task(coro: Any, **kwargs: Any) -> Any:
        tasks.append((coro, kwargs))
        closer = getattr(coro, "close", None)
        if callable(closer):
            closer()
        return SimpleNamespace(done=lambda: True)

    client._loop.create_task = capture_task  # type: ignore[assignment]

    monkeypatch.setattr(module.time, "time", lambda: 200.0)
    client._handle_event({"name": "data", "args": [[{"path": "/mgr/session", "body": {}}]]})
    assert client._subscription_ttl == pytest.approx(module._DEFAULT_SUBSCRIPTION_TTL)
    assert client._legacy_subscription_configured is False
    assert len(tasks) == 1

    monkeypatch.setattr(module.time, "time", lambda: 400.0)
    client._handle_event(
        {
            "name": "data",
            "args": [[{"path": "/mgr/session", "body": {"lease": {"timeout": "150"}}}]],
        }
    )
    assert client._subscription_ttl == pytest.approx(150.0)
    assert client._legacy_subscription_configured is True
    assert len(tasks) == 2
    assert refresh_loop.call_count == 2


@pytest.mark.asyncio
async def test_legacy_refresh_subscription_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check the legacy lease renewal updates bookkeeping on success."""

    client = _make_legacy_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    client._ws = SimpleNamespace(closed=False)
    client._subscription_ttl = 120.0
    client._send_snapshot_request = AsyncMock()
    client._subscribe_htr_samples = AsyncMock()
    monkeypatch.setattr(module.time, "time", lambda: 1000.0)

    await client._refresh_subscription(reason="periodic test")

    assert client._subscription_refresh_failed is False
    assert client._subscription_refresh_last_success == pytest.approx(1000.0)
    assert client._subscription_refresh_due == pytest.approx(1120.0)


@pytest.mark.asyncio
async def test_legacy_refresh_subscription_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure legacy lease renewal failures schedule a restart."""

    client = _make_legacy_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    client._ws = SimpleNamespace(closed=False)
    client._subscription_ttl = 180.0
    client._payload_idle_window = 540.0
    client._send_snapshot_request = AsyncMock(side_effect=RuntimeError("boom"))
    client._subscribe_htr_samples = AsyncMock()
    monkeypatch.setattr(module.time, "time", lambda: 500.0)

    with pytest.raises(RuntimeError):
        await client._refresh_subscription(reason="failure test")

    assert client._subscription_refresh_failed is True
    assert client._idle_restart_pending is True
    assert client._subscription_refresh_last_attempt == pytest.approx(500.0)


@pytest.mark.asyncio
async def test_ws_url_adds_suffix_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    rest = DummyREST(base="https://api.otherhost.com")
    client = _make_client(monkeypatch, rest=rest, hass_loop=asyncio.get_event_loop())
    ws_url = await client.ws_url()
    assert ws_url.startswith("https://api.otherhost.com/api/v2/socket_io")
    base, path = await client._build_engineio_target()
    assert path == "api/v2/socket_io"
    assert base.startswith(
        "https://api.otherhost.com/api/v2/socket_io?token=token&dev_id=device"
    )


@pytest.mark.asyncio
async def test_apply_nodes_payload_updates_state(monkeypatch: pytest.MonkeyPatch) -> None:
    hass_loop = asyncio.get_event_loop()
    client = _make_client(monkeypatch, hass_loop=hass_loop)
    client._coordinator.data = {"device": {"nodes_by_type": {}}}

    nodes_initial = {"nodes": {"htr": {"settings": {"1": {"temp": 20}}}}}
    client._handle_dev_data(nodes_initial)
    assert client._nodes["nodes"]["htr"]["settings"]["1"]["temp"] == 20
    assert client._stats.events_total == 1

    nodes_update = {"nodes": {"htr": {"settings": {"1": {"temp": 21}}}}}
    client._handle_update(nodes_update)
    assert client._nodes["nodes"]["htr"]["settings"]["1"]["temp"] == 21
    assert client._stats.events_total == 2

    hass_state = client.hass.data[module.DOMAIN]["entry"]
    assert hass_state["nodes"]["htr"]["settings"]["1"]["temp"] == 21
    client._coordinator.update_nodes.assert_called()
    assert client._dispatcher_mock.called


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
async def test_runner_handles_error_and_reconnect(monkeypatch: pytest.MonkeyPatch) -> None:
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
    client._payload_idle_window = 1
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
async def test_idle_monitor_exits_when_disconnected(monkeypatch: pytest.MonkeyPatch) -> None:
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
async def test_idle_monitor_skips_when_no_last_event(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    client._sio.connected = True
    client._last_event_at = None
    client._stats.last_event_ts = 0

    async def fake_sleep(_: float) -> None:
        client._closing = True

    monkeypatch.setattr(module.asyncio, "sleep", fake_sleep)
    await client._idle_monitor()


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
async def test_connect_once_invokes_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    client._stop_event = asyncio.Event()
    monkeypatch.setattr(
        client,
        "_build_engineio_target",
        AsyncMock(return_value=("https://socket", "api/v2/socket_io")),
    )
    connect_mock = AsyncMock()
    client._sio.connect = connect_mock
    await client._connect_once()
    connect_mock.assert_awaited()


@pytest.mark.asyncio
async def test_connect_once_respects_stop_event(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    client._stop_event = asyncio.Event()
    client._stop_event.set()
    client._sio.connect = AsyncMock()
    await client._connect_once()
    client._sio.connect.assert_not_called()


@pytest.mark.asyncio
async def test_wait_for_events_handles_disconnect(monkeypatch: pytest.MonkeyPatch) -> None:
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
async def test_force_refresh_token_resets_access(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    rest = client._client
    rest._access_token = "abc"  # type: ignore[attr-defined]
    await client._force_refresh_token()
    assert rest._ensure_token.await_count == 1  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_build_engineio_target_handles_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
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
    await client._on_disconnect()


@pytest.mark.asyncio
async def test_on_connect_requests_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the websocket client requests the initial device snapshot."""

    client = _make_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    client._sio.emit = AsyncMock()
    await client._on_connect()
    assert client._sio.emit.await_args_list == [
        call("join", namespace=module.WS_NAMESPACE),
        call("dev_data", namespace=module.WS_NAMESPACE),
    ]
    await client._on_disconnect()


@pytest.mark.asyncio
async def test_on_connect_emits_join_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    client._sio.emit = AsyncMock(side_effect=RuntimeError("boom"))
    await client._on_connect()
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


def test_collect_update_addresses_skips_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
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
    client.hass.data[module.DOMAIN]["entry"] = {"energy_coordinator": energy_coordinator}
    initial_data = {"device": {"nodes_by_type": {}}}
    client._coordinator.data = initial_data
    result = client._apply_heater_addresses({"htr": [1, 2]})
    assert result["htr"] == ["1", "2"]
    assert energy_coordinator.updated == {"htr": ["1", "2"]}
    assert client._coordinator.data is not initial_data


def test_apply_nodes_payload_forwards_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sample updates should be forwarded to the energy coordinator."""

    client = _make_client(monkeypatch)
    calls: list[tuple[str, Mapping[str, Mapping[str, Any]], float | None]] = []

    def handle_ws_samples(
        dev_id: str,
        updates: Mapping[str, Mapping[str, Any]],
        *,
        lease_seconds: float | None = None,
    ) -> None:
        calls.append((dev_id, updates, lease_seconds))

    energy_coordinator = SimpleNamespace(handle_ws_samples=handle_ws_samples)
    client.hass.data[module.DOMAIN]["entry"]["energy_coordinator"] = energy_coordinator
    client._subscription_ttl = 123.0

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
    dev_id, updates, lease = calls[0]
    assert dev_id == "device"
    assert lease == pytest.approx(123.0)
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


def test_handle_handshake_extracts_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch)
    refresh_loop = AsyncMock()
    monkeypatch.setattr(client, "_subscription_refresh_loop", refresh_loop)
    monkeypatch.setattr(module.time, "time", lambda: 1000.0)

    payload = {"lease": {"timeout": "450"}}
    client._handle_handshake(payload)

    assert client._subscription_ttl == pytest.approx(450.0)
    assert client._payload_idle_window == pytest.approx(675.0)
    assert client._subscription_refresh_due == pytest.approx(1450.0)
    assert client._subscription_refresh_task is not None
    refresh_loop.assert_called_once()


def test_handle_handshake_uses_default_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch)
    refresh_loop = AsyncMock()
    monkeypatch.setattr(client, "_subscription_refresh_loop", refresh_loop)
    monkeypatch.setattr(module.time, "time", lambda: 2000.0)

    client._handle_handshake({"message": "no ttl here"})

    assert client._subscription_ttl == pytest.approx(module._DEFAULT_SUBSCRIPTION_TTL)
    assert client._subscription_refresh_due == pytest.approx(
        2000.0 + module._DEFAULT_SUBSCRIPTION_TTL
    )
    assert client._payload_idle_window >= module._DEFAULT_SUBSCRIPTION_TTL
    refresh_loop.assert_called_once()


@pytest.mark.asyncio
async def test_subscription_refresh_loop_schedules_and_refreshes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client(monkeypatch, hass_loop=asyncio.get_event_loop())
    client._sio.connected = True
    client._subscription_ttl = 200.0
    client._closing = False

    current = 0.0

    async def fake_sleep(delay: float) -> None:
        nonlocal current
        current += delay

    def fake_time() -> float:
        return current

    sleep_calls: list[float] = []

    async def tracking_sleep(delay: float) -> None:
        sleep_calls.append(delay)
        await fake_sleep(delay)

    monkeypatch.setattr(module.asyncio, "sleep", tracking_sleep)
    monkeypatch.setattr(module.time, "time", fake_time)
    monkeypatch.setattr(module.random, "uniform", lambda *_: 0.9)

    client._sio.emit = AsyncMock()
    client._subscribe_heater_samples = AsyncMock()
    refresh_reasons: list[str] = []
    original_refresh = client._refresh_subscription

    async def capture_refresh(*, reason: str) -> None:
        refresh_reasons.append(reason)
        await original_refresh(reason=reason)
        client._closing = True

    monkeypatch.setattr(client, "_refresh_subscription", capture_refresh)

    await client._subscription_refresh_loop()

    assert refresh_reasons == ["periodic renewal"]
    assert sleep_calls and sleep_calls[0] == pytest.approx(144.0, rel=0.01)
    assert client._subscription_refresh_last_attempt == pytest.approx(current)
    assert client._subscription_refresh_last_success == pytest.approx(current)
    assert client._subscription_refresh_failed is False
    client._closing = False


def test_apply_heater_addresses_inventory_and_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch)
    client.hass.data[module.DOMAIN]["entry"] = {}
    client._coordinator.data = {"device": {"nodes_by_type": {}}}
    result = client._apply_heater_addresses({}, inventory=["inv"])
    assert client.hass.data[module.DOMAIN]["entry"]["node_inventory"] == ["inv"]
    assert result == {"htr": []}


def test_apply_heater_addresses_skips_empty_non_heater(monkeypatch: pytest.MonkeyPatch) -> None:
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
async def test_ducaheat_client_extended_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    client = module.DucaheatWSClient(
        SimpleNamespace(loop=asyncio.get_event_loop(), data={}),
        entry_id="entry",
        dev_id="device",
        api_client=DummyREST(),
        coordinator=SimpleNamespace(data={}, update_nodes=MagicMock()),
    )
    client._sio.emit = AsyncMock()
    await client._on_connect()
    await client._on_disconnect()
    await client._on_dev_handshake({})
    await client._on_dev_data({"nodes": {}})
    await client._on_update({"nodes": {}})


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


