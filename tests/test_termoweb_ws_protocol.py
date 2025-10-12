"""Extended tests for TermoWeb websocket protocol flows."""

from __future__ import annotations

import asyncio
import logging
import threading
from types import MappingProxyType, SimpleNamespace
from typing import Any, Mapping
from urllib.parse import parse_qsl, urlsplit
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.termoweb.backend import termoweb_ws as module
from custom_components.termoweb.backend import ws_client as ws_client_module
from custom_components.termoweb.backend.sanitize import (
    mask_identifier,
    redact_token_fragment,
)
from custom_components.termoweb.inventory import Inventory, build_node_inventory
from custom_components.termoweb.backend.ws_client import NodeDispatchContext
from homeassistant.core import HomeAssistant


def translate_update(payload: Any) -> Any:
    """Translate websocket path updates using the default namespace resolver."""

    return ws_client_module.translate_path_update(
        payload,
        resolve_section=module.WebSocketClient._resolve_update_section,
    )


INVALID_TRANSLATION_PAYLOADS: list[Any] = [
    "invalid",
    {"nodes": {}},
    {"path": 123, "body": {}},
    {"path": "/", "body": {}},
    {"path": "/api/devs/device", "body": {}},
    {"path": "/api/htr", "body": {}},
    {"path": "/htr", "body": {}},
    {"path": "/api/devs/device/htr/", "body": {}},
    {"path": "/api/devs/device/htr//settings", "body": {}},
    {"path": "/api/devs/device/ /settings", "body": {}},
    {"path": "/api/devs/device/htr/ /settings", "body": {}},
]


class DummyREST:
    """Provide just enough of the REST client interface for websocket tests."""

    def __init__(
        self,
        *,
        requested_with: str | None = "requested",
        api_base: str | None = "https://api.termoweb",
        authed_headers: dict[str, str] | None = None,
    ) -> None:
        self._session = SimpleNamespace(closed=True)
        self._ensure_token = AsyncMock()
        headers = authed_headers or {"Authorization": "Bearer token"}
        self.authed_headers = AsyncMock(return_value=headers)
        self.api_base = api_base
        self.user_agent = "agent"
        self.requested_with = requested_with


class StubAsyncClient:
    """Socket.IO client stub recording method invocations."""

    def __init__(
        self,
        allow_http_error: bool = False,
        *,
        existing_eio_http: Any | None = None,
        **_: Any,
    ) -> None:
        object.__setattr__(self, "events", {})
        object.__setattr__(self, "_connected", False)
        object.__setattr__(self, "connect_calls", [])
        object.__setattr__(self, "disconnect_calls", 0)
        object.__setattr__(self, "emit_calls", [])
        object.__setattr__(self, "_allow_http_error", allow_http_error)
        object.__setattr__(self, "_http_attempts", 0)
        object.__setattr__(
            self,
            "eio",
            SimpleNamespace(start_background_task=None, http=existing_eio_http),
        )
        object.__setattr__(self, "http", None)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "http" and getattr(self, "_allow_http_error", False):
            attempts = getattr(self, "_http_attempts")
            if attempts == 0:
                object.__setattr__(self, "_http_attempts", attempts + 1)
                object.__setattr__(self, "_allow_http_error", False)
                raise AttributeError("http is managed dynamically")
        object.__setattr__(self, name, value)

    def on(self, event: str, *, handler: Any, namespace: str | None = None) -> None:
        self.events[(event, namespace)] = handler

    async def connect(self, *args: Any, **kwargs: Any) -> None:
        self.connect_calls.append((args, kwargs))
        object.__setattr__(self, "_connected", True)

    async def disconnect(self) -> None:
        object.__setattr__(self, "_connected", False)
        object.__setattr__(self, "disconnect_calls", self.disconnect_calls + 1)

    async def emit(self, event: str, data: Any | None = None, *, namespace: str | None = None) -> None:
        self.emit_calls.append((event, data, namespace))

    @property
    def connected(self) -> bool:
        return getattr(self, "_connected")


def _make_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    hass_loop: Any | None = None,
    allow_http_error: bool = False,
    requested_with: str | None = "requested",
    rest_headers: dict[str, str] | None = None,
    session: Any | None = None,
    existing_eio_http: Any | None = None,
    api_base: str | None = "https://api.termoweb",
) -> tuple[module.WebSocketClient, StubAsyncClient, MagicMock]:
    """Instantiate a ``WebSocketClient`` with a controllable AsyncClient stub."""

    holder: dict[str, StubAsyncClient] = {}

    def factory(**kwargs: Any) -> StubAsyncClient:
        stub = StubAsyncClient(
            allow_http_error=allow_http_error,
            existing_eio_http=existing_eio_http,
            **kwargs,
        )
        holder["client"] = stub
        return stub

    monkeypatch.setattr(module.socketio, "AsyncClient", factory)
    dispatcher = MagicMock()
    monkeypatch.setattr(module, "async_dispatcher_send", dispatcher)

    if hass_loop is None:
        hass_loop = SimpleNamespace(
            create_task=lambda coro, **_: SimpleNamespace(done=lambda: False),
            call_soon_threadsafe=lambda cb, *args: cb(*args),
            is_running=lambda: False,
        )

    hass = HomeAssistant()
    hass.loop = hass_loop
    hass.loop_thread_id = threading.get_ident()
    hass.data.setdefault(module.DOMAIN, {})["entry"] = {}
    coordinator = SimpleNamespace(update_nodes=MagicMock(), data={})
    client = module.WebSocketClient(
        hass,
        entry_id="entry",
        dev_id="device",
        api_client=DummyREST(
            requested_with=requested_with,
            api_base=api_base,
            authed_headers=rest_headers,
        ),
        coordinator=coordinator,
        session=session or SimpleNamespace(closed=True),
    )
    return client, holder["client"], dispatcher


def test_handshake_error_exposes_fields() -> None:
    """The TermoWeb handshake error should record status, URL and body."""

    error = module.HandshakeError(
        503,
        "https://example/ws",
        "body",
        response_snippet="body",
    )
    assert str(error) == "handshake failed: status=503, detail=body"
    assert error.status == 503
    assert error.url == "https://example/ws"
    assert error.detail == "body"
    assert error.response_snippet == "body"


def test_init_handles_socketio_http_attribute(monkeypatch: pytest.MonkeyPatch) -> None:
    """Initialisation should recover when AsyncClient rejects ``http`` assignment."""

    client, sio, _ = _make_client(monkeypatch, allow_http_error=True)
    assert isinstance(client, module.WebSocketClient)
    assert sio._http_attempts == 1
    assert sio.http is not None
    assert sio.eio.http is not None


def test_init_populates_defaults_and_preserves_existing_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Initialisation should fill default metadata and respect existing Engine.IO state."""

    existing_http = SimpleNamespace(closed=False, preserved=True)
    session = SimpleNamespace()

    client, sio, _ = _make_client(
        monkeypatch,
        requested_with=None,
        session=session,
        existing_eio_http=existing_http,
    )

    assert client._requested_with == module.get_brand_requested_with(module.BRAND_TERMOWEB)
    assert hasattr(client._sio.http, "closed")
    assert sio.eio.http is existing_http


@pytest.mark.asyncio
async def test_connect_once_invokes_socketio_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    """_connect_once should reset backoff and call the AsyncClient."""

    client, sio, _ = _make_client(monkeypatch)
    monkeypatch.setattr(client, "_build_engineio_target", AsyncMock(return_value=("wss://ws", "socket.io")))
    client._stop_event.clear()

    await client._connect_once()

    assert sio.connect_calls
    assert client._backoff_idx == 0


@pytest.mark.asyncio
async def test_connect_once_aborts_when_stopping(monkeypatch: pytest.MonkeyPatch) -> None:
    """_connect_once should exit early when stop is requested."""

    client, sio, _ = _make_client(monkeypatch)
    client._stop_event.set()
    await client._connect_once()
    assert not sio.connect_calls


@pytest.mark.asyncio
async def test_ws_url_returns_target(monkeypatch: pytest.MonkeyPatch) -> None:
    """ws_url should proxy to _build_engineio_target."""

    client, _sio, _ = _make_client(monkeypatch)
    monkeypatch.setattr(
        client,
        "_build_engineio_target",
        AsyncMock(return_value=("wss://example/ws", "socket.io")),
    )

    assert await client.ws_url() == "wss://example/ws"


@pytest.mark.asyncio
async def test_debug_probe_handles_logging(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """debug_probe should respect logging configuration and handle emit failures."""

    client, sio, _ = _make_client(monkeypatch)

    monkeypatch.setattr(module._LOGGER, "isEnabledFor", lambda level: False)
    await client.debug_probe()

    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(module._LOGGER, "isEnabledFor", lambda level: True)
    sio.emit = AsyncMock(return_value=None)
    await client.debug_probe()
    assert "debug probe dev_data emitted" in caplog.text

    sio.emit = AsyncMock(side_effect=RuntimeError("boom"))
    await client.debug_probe()


@pytest.mark.asyncio
async def test_wait_for_events_cancels_pending_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    """_wait_for_events should cancel whichever wait completes second."""

    loop = asyncio.get_running_loop()
    hass_loop = SimpleNamespace(
        create_task=lambda coro, **kwargs: loop.create_task(coro, **kwargs),
        call_soon_threadsafe=lambda cb, *args: loop.call_soon(cb, *args),
    )
    client, _sio, _ = _make_client(monkeypatch, hass_loop=hass_loop)
    client._stop_event = asyncio.Event()
    client._disconnected = asyncio.Event()
    client._stop_event.set()
    client._loop = loop

    await client._wait_for_events()


@pytest.mark.asyncio
async def test_runner_handles_errors_and_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """The connection runner should retry until ``_closing`` is set."""

    client, _sio, dispatcher = _make_client(monkeypatch)
    call_order: list[str] = []

    async def connect_once() -> None:
        call_order.append("connect")

    async def wait_for_events() -> None:
        call_order.append("wait")
        raise RuntimeError("boom")

    async def disconnect(**_: Any) -> None:
        call_order.append("disconnect")

    async def lost(error: Exception | None) -> None:
        call_order.append(f"lost:{type(error).__name__ if error else 'none'}")
        client._closing = True

    client._connect_once = AsyncMock(side_effect=connect_once)  # type: ignore[attr-defined]
    client._wait_for_events = AsyncMock(side_effect=wait_for_events)  # type: ignore[attr-defined]
    client._disconnect = AsyncMock(side_effect=disconnect)  # type: ignore[attr-defined]
    client._handle_connection_lost = AsyncMock(side_effect=lost)  # type: ignore[attr-defined]

    async def fake_sleep(delay: float) -> None:
        call_order.append(f"sleep:{delay}")
        client._closing = True

    monkeypatch.setattr(module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(module.random, "uniform", lambda a, b: 1.0)

    await client._runner()

    assert call_order == ["connect", "wait", "disconnect", "lost:RuntimeError"]


@pytest.mark.asyncio
async def test_runner_propagates_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    """The runner should re-raise cancellation requests."""

    client, _sio, _ = _make_client(monkeypatch)
    client._connect_once = AsyncMock(side_effect=asyncio.CancelledError())  # type: ignore[attr-defined]

    with pytest.raises(asyncio.CancelledError):
        await client._runner()


@pytest.mark.asyncio
async def test_runner_performs_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """The runner should wait using the backoff sequence when retries are needed."""

    client, _sio, _ = _make_client(monkeypatch)

    async def connect_once() -> None:
        raise RuntimeError("boom")

    client._connect_once = AsyncMock(side_effect=connect_once)  # type: ignore[attr-defined]

    async def fake_sleep(delay: float) -> None:
        client._closing = True

    monkeypatch.setattr(module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(module.random, "uniform", lambda a, b: 1.0)

    await client._runner()


@pytest.mark.asyncio
async def test_handle_connection_lost_updates_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Losing the connection should persist restart metadata."""

    client, _sio, _ = _make_client(monkeypatch)
    state = client._ws_state_bucket()
    assert state.get("restart_count") is None

    await client._handle_connection_lost(RuntimeError("boom"))

    assert state["restart_count"] == 1
    assert "RuntimeError" in state["last_disconnect_error"]


def test_mark_event_tracks_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """_mark_event should record recent event paths and update status."""

    client, _sio, dispatcher = _make_client(monkeypatch)
    monkeypatch.setattr(module._LOGGER, "isEnabledFor", lambda level: True)
    client._mark_event(paths=["/a", "/b", "/c", "/d", "/e", "/f"], count_event=False)
    assert client._stats.last_paths == ["/a", "/b", "/c", "/d", "/e"]


@pytest.mark.asyncio
async def test_disconnect_logs_exceptions(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """_disconnect should swallow errors from the socket client."""

    client, sio, _ = _make_client(monkeypatch)
    object.__setattr__(sio, "_connected", True)
    sio.disconnect = AsyncMock(side_effect=RuntimeError("boom"))
    caplog.set_level(logging.DEBUG)
    await client._disconnect(reason="tests")
    assert "disconnect due to tests failed" in caplog.text


@pytest.mark.asyncio
async def test_disconnect_calls_socketio(monkeypatch: pytest.MonkeyPatch) -> None:
    """_disconnect should call the AsyncClient when connected."""

    client, sio, _ = _make_client(monkeypatch)
    await sio.connect()
    await client._disconnect(reason="tests")
    assert sio.disconnect_calls == 1
    assert client._disconnected.is_set()


@pytest.mark.asyncio
async def test_get_token_requires_authorization(monkeypatch: pytest.MonkeyPatch) -> None:
    """_get_token should raise when the Authorization header is missing."""

    client, _sio, _ = _make_client(monkeypatch, rest_headers={"Authorization": ""})
    with pytest.raises(RuntimeError):
        await client._get_token()


@pytest.mark.asyncio
async def test_force_refresh_token_resets_access(monkeypatch: pytest.MonkeyPatch) -> None:
    """_force_refresh_token should clear cached credentials and ensure tokens."""

    client, _sio, _ = _make_client(monkeypatch)
    client._client._access_token = "token"  # type: ignore[attr-defined]
    await client._force_refresh_token()
    client._client._ensure_token.assert_awaited()  # type: ignore[attr-defined]


def test_api_base_defaults_to_constant(monkeypatch: pytest.MonkeyPatch) -> None:
    """_api_base should fall back to the default when the client lacks one."""

    client, _sio, _ = _make_client(monkeypatch, api_base=None)
    assert client._api_base() == module.API_BASE


def test_ws_state_bucket_initialises_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    """_ws_state_bucket should create storage on hass when missing."""

    client, _sio, _ = _make_client(monkeypatch)
    client.hass = SimpleNamespace(loop=None)
    client._ws_state = None
    bucket = client._ws_state_bucket()
    assert isinstance(bucket, dict)


@pytest.mark.asyncio
async def test_build_engineio_target_uses_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Engine.IO URL builder should include the token and device id."""

    client, _sio, _ = _make_client(monkeypatch)
    monkeypatch.setattr(client, "_get_token", AsyncMock(return_value="tok"))
    url, path = await client._build_engineio_target()
    assert url.startswith("https://api.termoweb")
    assert "token=tok" in url and "dev_id=device" in url
    assert path == "socket.io"


@pytest.mark.asyncio
async def test_on_connect_schedules_idle_monitor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Connecting should reset metrics and schedule the idle monitor."""

    loop = asyncio.get_running_loop()
    created: list[asyncio.Task] = []

    def create_task(coro: Any, **_: Any) -> asyncio.Task:
        if isinstance(coro, asyncio.Task):
            task = coro
        else:
            task = loop.create_task(coro)
        created.append(task)
        return task

    hass_loop = SimpleNamespace(
        create_task=create_task,
        call_soon_threadsafe=lambda cb, *args: loop.call_soon(cb, *args),
    )
    client, _sio, dispatcher = _make_client(monkeypatch, hass_loop=hass_loop)
    await client._on_connect()
    assert created


@pytest.mark.asyncio
async def test_namespace_connect_emits_join(monkeypatch: pytest.MonkeyPatch) -> None:
    """Joining the namespace should emit join and dev_data events."""

    client, sio, _ = _make_client(monkeypatch)
    await client._on_namespace_connect()
    assert ("join", None, module.WS_NAMESPACE) in sio.emit_calls
    assert ("dev_data", None, module.WS_NAMESPACE) in sio.emit_calls


@pytest.mark.asyncio
async def test_namespace_connect_handles_failure(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Namespace join failures should be logged at DEBUG level."""

    client, sio, _ = _make_client(monkeypatch)
    sio.emit = AsyncMock(side_effect=RuntimeError("boom"))
    caplog.set_level(logging.DEBUG)
    await client._on_namespace_connect()
    assert "namespace join failed" in caplog.text


def test_register_debug_catch_all_installs_handler(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Debug catch-all registration should wrap the AsyncClient when DEBUG is enabled."""

    client, sio, _ = _make_client(monkeypatch)
    caplog.set_level("DEBUG", logger=module._LOGGER.name)
    client._register_debug_catch_all()
    assert ("*", module.WS_NAMESPACE) in sio.events


@pytest.mark.asyncio
async def test_register_debug_catch_all_reuses_handler(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Calling register twice should reuse the existing handler."""

    client, sio, _ = _make_client(monkeypatch)
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(module._LOGGER, "isEnabledFor", lambda level: True)
    client._register_debug_catch_all()
    handler = sio.events[("*", module.WS_NAMESPACE)]
    client._register_debug_catch_all()
    await handler("event", 1, key="value")
    monkeypatch.setattr(module._LOGGER, "isEnabledFor", lambda level: False)
    await handler("ignored")


@pytest.mark.asyncio
async def test_on_disconnect_cancels_monitor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disconnecting should cancel the idle monitor task and set the flag."""

    client, _sio, _ = _make_client(monkeypatch)
    client._idle_monitor_task = asyncio.create_task(asyncio.sleep(0))
    await asyncio.sleep(0)
    await client._on_disconnect()
    assert client._idle_monitor_task is None
    assert client._disconnected.is_set()


@pytest.mark.asyncio
async def test_misc_event_logging(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Event handlers should log diagnostic messages when DEBUG is enabled."""

    client, _sio, _ = _make_client(monkeypatch)
    monkeypatch.setattr(client, "_subscribe_heater_samples", AsyncMock())
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(module._LOGGER, "isEnabledFor", lambda level: True)

    await client._on_reconnect()
    await client._on_connect_error({"error": "boom"})
    await client._on_error({"error": "boom"})
    await client._on_reconnect_failed({"attempts": 3})
    await client._on_namespace_disconnect("bye")

    assert "reconnect event" in caplog.text


@pytest.mark.asyncio
async def test_refresh_subscription_emits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refreshing the subscription should emit dev_data and resubscribe samples."""

    client, sio, _ = _make_client(monkeypatch)
    await sio.connect()
    monkeypatch.setattr(client, "_subscribe_heater_samples", AsyncMock())
    await client._refresh_subscription(reason="timer")
    assert ("dev_data", None, module.WS_NAMESPACE) in sio.emit_calls
    client._subscribe_heater_samples.assert_awaited()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_refresh_subscription_requires_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refreshing while disconnected should raise an error."""

    client, _sio, _ = _make_client(monkeypatch)
    with pytest.raises(RuntimeError):
        await client._refresh_subscription(reason="disconnected")


@pytest.mark.asyncio
async def test_idle_monitor_refreshes_and_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    """The idle monitor should attempt to refresh and exit when closing."""

    client, sio, _ = _make_client(monkeypatch)
    await sio.connect()
    client._payload_idle_window = 1.0
    client._stats.last_event_ts = 100.0
    client._subscription_refresh_failed = False
    client._closing = False
    refresh_calls: list[str] = []

    async def fake_refresh(**_: Any) -> None:
        refresh_calls.append("refresh")
        client._closing = True

    real_sleep = asyncio.sleep

    async def fake_sleep(_: float) -> None:
        await real_sleep(0)

    monkeypatch.setattr(client, "_refresh_subscription", AsyncMock(side_effect=fake_refresh))
    monkeypatch.setattr(module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(module.time, "time", lambda: 200.0)

    await client._idle_monitor()

    assert refresh_calls == ["refresh"]


@pytest.mark.asyncio
async def test_idle_monitor_breaks_when_disconnected(monkeypatch: pytest.MonkeyPatch) -> None:
    """The idle monitor should exit when disconnected from the socket."""

    client, sio, _ = _make_client(monkeypatch)
    client._closing = False
    client._disconnected.set()
    object.__setattr__(sio, "_connected", False)

    async def fake_sleep(_: float) -> None:
        client._closing = True

    monkeypatch.setattr(module.asyncio, "sleep", fake_sleep)
    await client._idle_monitor()


@pytest.mark.asyncio
async def test_idle_monitor_skips_without_last_event(monkeypatch: pytest.MonkeyPatch) -> None:
    """Idle monitor should continue when no last event timestamp is available."""

    client, sio, _ = _make_client(monkeypatch)
    await sio.connect()
    client._closing = False
    client._disconnected.clear()
    client._last_event_at = None
    client._stats.last_event_ts = 0.0

    async def fake_sleep(_: float) -> None:
        client._closing = True

    monkeypatch.setattr(module.asyncio, "sleep", fake_sleep)
    await client._idle_monitor()


@pytest.mark.asyncio
async def test_idle_monitor_waits_for_disconnect(monkeypatch: pytest.MonkeyPatch) -> None:
    """Idle monitor should continue when disconnected flag is unset."""

    client, sio, _ = _make_client(monkeypatch)
    client._closing = False
    client._disconnected.clear()
    object.__setattr__(sio, "_connected", False)

    async def fake_sleep(_: float) -> None:
        client._closing = True

    monkeypatch.setattr(module.asyncio, "sleep", fake_sleep)
    await client._idle_monitor()


@pytest.mark.asyncio
async def test_idle_monitor_schedules_restart_on_refresh_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refresh failures should schedule idle restarts."""

    client, sio, _ = _make_client(monkeypatch)
    await sio.connect()
    client._last_event_at = 1.0
    client._payload_idle_window = 1.0
    monkeypatch.setattr(module.time, "time", lambda: 5.0)
    client._closing = False
    client._schedule_idle_restart = MagicMock()
    client._refresh_subscription = AsyncMock(side_effect=RuntimeError("boom"))

    async def fake_sleep(_: float) -> None:
        client._closing = True

    monkeypatch.setattr(module.asyncio, "sleep", fake_sleep)
    await client._idle_monitor()
    client._schedule_idle_restart.assert_called_once()


@pytest.mark.asyncio
async def test_idle_monitor_retries_failed_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the previous refresh failed the monitor should retry quickly."""

    client, sio, _ = _make_client(monkeypatch)
    await sio.connect()
    client._last_event_at = 5.0
    client._payload_idle_window = 100.0
    client._subscription_refresh_failed = True
    monkeypatch.setattr(module.time, "time", lambda: 10.0)
    client._closing = False
    client._schedule_idle_restart = MagicMock()
    client._refresh_subscription = AsyncMock(side_effect=RuntimeError("boom"))

    async def fake_sleep(_: float) -> None:
        client._closing = True

    monkeypatch.setattr(module.asyncio, "sleep", fake_sleep)
    await client._idle_monitor()
    client._schedule_idle_restart.assert_called_once()


def test_translate_path_update_and_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Path based updates should map onto node sections."""

    client, _sio, _ = _make_client(monkeypatch)
    payload = {
        "path": "/api/devs/device/htr/1/settings/temp",
        "body": {"value": 20},
    }
    translated = translate_update(payload)
    assert translated == {"htr": {"settings": {"1": {"temp": {"value": 20}}}}}
    assert client._translate_path_update(payload) == translated
    setup_payload = {
        "path": "/api/devs/device/htr/1/setup/program",
        "body": {"foo": 1},
    }
    translated_setup = translate_update(setup_payload)
    assert translated_setup == {"htr": {"settings": {"1": {"setup": {"program": {"foo": 1}}}}}}
    assert client._translate_path_update(setup_payload) == translated_setup
    assert module.WebSocketClient._resolve_update_section("advanced_setup") == ("advanced", "advanced_setup")
    assert module.WebSocketClient._resolve_update_section("prog") == ("settings", "prog")
    assert module.WebSocketClient._resolve_update_section(None) == (None, None)


def test_translate_path_update_invalid_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid payloads should return None from the path translator."""

    client, _sio, _ = _make_client(monkeypatch)
    for payload in INVALID_TRANSLATION_PAYLOADS:
        assert translate_update(payload) is None
        assert client._translate_path_update(payload) is None


def test_translate_path_update_rejects_unknown_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Path translation should ignore unknown node types and addresses."""

    client, _sio, _ = _make_client(monkeypatch)
    for payload in (
        {"path": "/api/devs/device/ /1/settings", "body": {"v": 1}},
        {"path": "/api/devs/device/htr/ /settings", "body": {"v": 1}},
    ):
        assert translate_update(payload) is None
        assert client._translate_path_update(payload) is None


def test_handle_handshake_logging(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Handshake handling should log keys and ignore invalid payloads."""

    client, _sio, _ = _make_client(monkeypatch)
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(module._LOGGER, "isEnabledFor", lambda level: True)
    client._handle_handshake({"alpha": 1, "beta": 2})
    assert client._handshake_payload == {"alpha": 1, "beta": 2}
    client._handle_handshake("invalid")


def test_forward_sample_updates_invokes_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sample update forwarding should call the energy coordinator hook."""

    client, _sio, _ = _make_client(monkeypatch)
    handler = MagicMock()
    client.hass.data[module.DOMAIN]["entry"]["energy_coordinator"] = SimpleNamespace(
        handle_ws_samples=handler
    )
    client._forward_sample_updates(
        {"htr": {"samples": {"1": {"power": 10}}, "lease_seconds": 30}}
    )
    handler.assert_called_once()
    assert handler.call_args.kwargs.get("lease_seconds") == 30


def test_forward_sample_updates_handles_missing_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sample update forwarding should safely no-op when no handler exists."""

    client, _sio, _ = _make_client(monkeypatch)
    client.hass.data = {}
    client._forward_sample_updates({"htr": {"samples": {"1": {}}}})
    client.hass.data = {module.DOMAIN: {"entry": {}}}
    client._forward_sample_updates({"htr": {"samples": {"1": {}}}})


def test_apply_nodes_payload_debug_branches(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Applying node payloads should log diagnostic information and filter invalid data."""

    client, _sio, _ = _make_client(monkeypatch)
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(module._LOGGER, "isEnabledFor", lambda level: True)
    client._dispatch_nodes = MagicMock(return_value={})
    client._forward_sample_updates = MagicMock()
    client._mark_event = MagicMock()
    client._collect_update_addresses = MagicMock(
        side_effect=[[("htr", "1")], [], [], []]
    )
    client._client.normalise_ws_nodes = lambda nodes: nodes

    client._apply_nodes_payload({}, merge=False, event="dev_data")

    nodes_payload = {
        "nodes": {
            1: {"samples": {"1": {"power": 5}}},
            "htr": {"samples": {"bad": {"power": 3}, "1": {"power": 10}}},
            "acm": {"samples": []},
        }
    }
    client._apply_nodes_payload(nodes_payload, merge=True, event="update")

    client._apply_nodes_payload(
        {"nodes": {"htr": {"samples": {"1": {"power": 7}}}}},
        merge=True,
        event="update",
    )

    client._apply_nodes_payload(
        {"nodes": {"htr": {"samples": {"1": {"power": 8}}}}},
        merge=False,
        event="dev_data",
    )

    client._apply_nodes_payload(
        {"nodes": {"htr": {"samples": {"": {"power": 9}}}}},
        merge=True,
        event="update",
    )

    assert client._forward_sample_updates.called


def test_handle_dev_data_and_update(monkeypatch: pytest.MonkeyPatch) -> None:
    """Direct handlers should call into the payload merger."""

    client, _sio, _ = _make_client(monkeypatch)
    client._apply_nodes_payload = MagicMock()  # type: ignore[attr-defined]
    client._handle_dev_data({"nodes": {"htr": {}}})
    client._apply_nodes_payload.assert_called_with(
        {"nodes": {"htr": {}}}, merge=False, event="dev_data"
    )
    client._apply_nodes_payload.reset_mock()
    client._handle_update({"path": "value"})
    client._apply_nodes_payload.assert_called_with(
        {"path": "value"}, merge=True, event="update"
    )


def test_extract_and_translate_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    """List based node payloads should be converted into the mapping schema."""

    client, _sio, _ = _make_client(monkeypatch)
    payload = {"nodes": [{"type": "htr", "addr": "1", "settings": {"temp": 20}}]}
    nodes = client._extract_nodes(payload)
    assert nodes and "htr" in nodes
    assert payload["nodes"]["htr"]


def test_translate_nodes_list_invalid_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    """translate_nodes_list should skip invalid entries gracefully."""

    client, _sio, _ = _make_client(monkeypatch)
    nodes = client._translate_nodes_list(
        [
            "invalid",
            {"type": None, "addr": "1"},
            {"type": "htr", "addr": None},
            {"type": "htr", "addr": "", "settings": {}},
            {"type": "htr", "addr": "1", 1: {}},
            {"type": "htr", "addr": "2", "": {}},
            {"type": "htr", "addr": "3", "advanced_setup": {"k": 1}},
            {"type": "htr", "addr": "3", "advanced_setup": {"j": 2}},
        ]
    )
    assert "htr" in nodes and "advanced" in nodes["htr"]


def test_translate_nodes_list_merges_nested_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Multiple entries for the same node should merge nested payloads."""

    client, _sio, _ = _make_client(monkeypatch)
    merged = client._translate_nodes_list(
        [
            {
                "type": "htr",
                "addr": "1",
                "advanced_setup": {"first": 1},
            },
            {
                "type": "htr",
                "addr": "1",
                "advanced_setup": {"second": 2},
            },
            {
                "type": "htr",
                "addr": "1",
                "status": {"mode": "auto"},
            },
        ]
    )
    advanced = merged["htr"]["advanced"]["1"]
    assert advanced == {"advanced_setup": {"second": 2}}
    assert merged["htr"]["status"]["1"] == {"mode": "auto"}


def test_apply_nodes_payload_merges_and_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Applying node payloads should normalize and dispatch updates."""

    client, _sio, dispatcher = _make_client(monkeypatch)
    client._collect_update_addresses = MagicMock(return_value=[("htr", "1")])  # type: ignore[attr-defined]
    client._dispatch_nodes = MagicMock(return_value={"htr": ["1"]})  # type: ignore[attr-defined]
    client._forward_sample_updates = MagicMock()  # type: ignore[attr-defined]
    client._mark_event = MagicMock()  # type: ignore[attr-defined]

    snapshot_payload = {"nodes": {"htr": {"settings": {"1": {"temp": 20}}}}}
    client._apply_nodes_payload(snapshot_payload, merge=False, event="dev_data")
    client._dispatch_nodes.assert_called_with(snapshot_payload["nodes"])

    update_payload = {"path": "/api/devs/device/htr/1/samples", "body": {"power": 5}}
    client._apply_nodes_payload(update_payload, merge=True, event="update")
    client._forward_sample_updates.assert_called()
    client._mark_event.assert_called()


def test_collect_update_addresses_handles_invalid_entries() -> None:
    """collect_update_addresses should filter invalid keys."""

    nodes = {
        "htr": {"settings": {"1": {}, 2: None}, "samples": {"1": {"power": 10}}, "extra": []},
        3: {},
    }
    addresses = module.WebSocketClient._collect_update_addresses(nodes)
    assert addresses == [("htr", "1")]


def test_collect_update_addresses_skips_non_mapping_sections() -> None:
    """Non-mapping sections should be ignored when collecting addresses."""

    nodes = {
        "htr": {"settings": [], "samples": {"1": {"power": 5}}, "advanced": "nope"},
        "acm": "invalid",
    }
    addresses = module.WebSocketClient._collect_update_addresses(nodes)
    assert addresses == [("htr", "1")]


def test_dispatch_nodes_with_inventory(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """dispatch_nodes should support pre-existing inventory records."""

    client, _sio, dispatcher = _make_client(monkeypatch)
    record = client.hass.data[module.DOMAIN]["entry"]
    record["inventory"] = Inventory(client.dev_id, {"nodes": {}}, ())
    energy = SimpleNamespace(update_addresses=MagicMock(), handle_ws_samples=MagicMock())
    record["energy_coordinator"] = energy
    client._coordinator.update_nodes = MagicMock()
    client._coordinator.data = {
        "device": {"addresses_by_type": {}, "settings": {}}
    }
    nodes_payload = {"nodes": [{"type": "htr", "addr": "1"}]}
    client._inventory = Inventory(
        client.dev_id,
        nodes_payload,
        build_node_inventory(nodes_payload),
    )
    caplog.set_level(logging.DEBUG)

    result = client._dispatch_nodes({"nodes": {"htr": {"settings": {"1": {}}}}})
    assert isinstance(result, dict)
    client._coordinator.update_nodes.assert_not_called()
    dispatcher.assert_called_once()
    _, _, payload = dispatcher.call_args[0]
    assert "nodes" not in payload
    assert payload["inventory"] is client._inventory
    assert payload["inventory_addresses"] == {"htr": ["1"]}
    assert client._inventory.addresses_by_type["htr"] == ["1"]


def test_dispatch_nodes_handles_unknown_types(monkeypatch: pytest.MonkeyPatch) -> None:
    """dispatch_nodes should ignore unsupported types when inventory filters them."""

    client, _sio, dispatcher = _make_client(monkeypatch)
    record = client.hass.data[module.DOMAIN]["entry"]
    record["energy_coordinator"] = SimpleNamespace(update_addresses=MagicMock())
    client._coordinator.update_nodes = MagicMock()

    client._inventory = Inventory(
        client.dev_id,
        {"nodes": [{"type": "foo", "addr": "9"}]},
        build_node_inventory([{"type": "foo", "addr": "9"}]),
    )
    client._dispatch_nodes({"nodes": {}})
    client._coordinator.update_nodes.assert_not_called()
    dispatcher.assert_called_once()
    _, _, payload = dispatcher.call_args[0]
    assert "nodes" not in payload
    assert payload["inventory_addresses"] == {"foo": ["9"]}
    assert "unknown_types" not in payload


def test_dispatch_nodes_uses_inventory_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inventory payload should backfill missing node payload data."""

    client, _sio, dispatcher = _make_client(monkeypatch)
    record = client.hass.data[module.DOMAIN]["entry"]
    record["energy_coordinator"] = SimpleNamespace(update_addresses=MagicMock())
    client._coordinator.update_nodes = MagicMock()
    node_inventory = build_node_inventory([{"type": "htr", "addr": "2"}])

    class TrackingInventory(Inventory):
        def __init__(self) -> None:
            super().__init__(
                client.dev_id,
                {"nodes": {"htr": {"settings": {"2": {"temp": 21}}}}},
                node_inventory,
            )
            object.__setattr__(self, "payload_calls", 0)

        @property
        def payload(self) -> Any:
            current = getattr(self, "payload_calls", 0)
            object.__setattr__(self, "payload_calls", current + 1)
            return Inventory.payload.fget(self)

    inventory = TrackingInventory()
    client._inventory = inventory

    def fake_prepare(*args: Any, **kwargs: Any) -> NodeDispatchContext:
        return NodeDispatchContext(
            payload=None,
            inventory=inventory,
            record=record,
        )

    monkeypatch.setattr(module, "_prepare_nodes_dispatch", fake_prepare)

    client._dispatch_nodes({"nodes": None})

    client._coordinator.update_nodes.assert_not_called()
    dispatcher.assert_called_once()


def test_handle_event_includes_inventory_addresses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy websocket events should include inventory metadata in payloads."""

    hass = HomeAssistant()
    hass.loop = SimpleNamespace(
        call_soon_threadsafe=lambda cb, *args: cb(*args),
        is_running=lambda: False,
    )
    hass.data.setdefault(module.DOMAIN, {})["entry"] = {}
    coordinator = SimpleNamespace(data={}, update_nodes=MagicMock())

    dispatcher = MagicMock()
    monkeypatch.setattr(module, "async_dispatcher_send", dispatcher)
    monkeypatch.setattr(module.TermoWebWSClient, "_install_write_hook", lambda self: None)
    monkeypatch.setattr(
        module.TermoWebWSClient,
        "_dispatch_nodes",
        lambda self, payload: {"htr": ["2"]},
    )

    monkeypatch.setattr(
        module.TermoWebWSClient,
        "_update_legacy_section",
        lambda self, **_: True,
    )
    monkeypatch.setattr(
        module.TermoWebWSClient,
        "_legacy_section_for_path",
        staticmethod(lambda path: "settings"),
    )

    client = module.TermoWebWSClient(
        hass,
        entry_id="entry",
        dev_id="device",
        api_client=DummyREST(),
        coordinator=coordinator,
        session=SimpleNamespace(closed=False),
    )

    event_payload = {
        "name": "data",
        "args": [
            [
                {
                    "path": "/devs/device/mgr/nodes",
                    "body": {"htr": {"settings": {"2": {"mode": "auto"}}}},
                },
                {
                    "path": "/devs/device/htr/2/settings",
                    "body": {"mode": "auto"},
                },
            ]
        ],
    }

    client._handle_event(event_payload)

    payloads = [call.args[2] for call in dispatcher.call_args_list]
    assert payloads
    nodes_payload = next(
        (payload for payload in payloads if payload.get("kind") == "nodes"),
        None,
    )
    assert nodes_payload is not None
    assert nodes_payload["inventory_addresses"] == {"htr": ["2"]}

    settings_payload = next(
        (
            payload
            for payload in payloads
            if payload.get("kind") == "htr_settings"
        ),
        None,
    )
    assert settings_payload is not None
    assert settings_payload["inventory_addresses"] == {"htr": ["2"]}

    dev_state = client._coordinator.data.get("device")
    assert isinstance(dev_state, Mapping)
    assert "nodes" not in dev_state
    assert "addresses_by_type" not in dev_state


def test_update_legacy_settings_updates_settings_map(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings updates should refresh both legacy and normalized caches."""

    client, _sio, _ = _make_client(monkeypatch)
    dev_map: dict[str, Any] = {"settings": {"htr": {"1": {"mode": "manual"}}}}

    updated = module.TermoWebWSClient._update_legacy_section(
        client,
        node_type="htr",
        addr=" 01 ",
        section="settings",
        body={"mode": "auto"},
        dev_map=dev_map,
    )

    assert updated is True
    settings_map = dev_map["settings"]["htr"]
    assert settings_map["1"]["mode"] == "manual"
    assert settings_map["01"]["mode"] == "auto"

    settings_map["01"]["pending"] = {"mode": "auto"}

    updated_again = module.TermoWebWSClient._update_legacy_section(
        client,
        node_type="htr",
        addr=" 01 ",
        section="settings",
        body={"mode": "eco"},
        dev_map=dev_map,
    )

    assert updated_again is True
    assert settings_map["01"]["mode"] == "eco"
    assert settings_map["01"]["pending"] == {"mode": "auto"}
def test_apply_heater_addresses_normalises_from_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Applying heater addresses should mirror inventory-normalised addresses."""

    client, _sio, _ = _make_client(monkeypatch)
    client._coordinator.data = {"device": {"settings": {}}}
    raw_nodes = {"nodes": [{"type": "htr", "addr": "1"}, {"type": "acm", "addr": "2"}]}
    inventory = Inventory(
        client.dev_id,
        raw_nodes,
        build_node_inventory(raw_nodes),
    )
    normalized = client._apply_heater_addresses(
        {"htr": ["1"], "acm": ["2"]}, inventory=inventory
    )
    heater_map, heater_aliases = inventory.heater_sample_address_map
    power_map, power_aliases = inventory.power_monitor_sample_address_map
    assert normalized["htr"] == heater_map["htr"]
    assert normalized["acm"] == heater_map["acm"]
    assert client._coordinator.data == {"device": {"settings": {}}}
    record = client.hass.data[module.DOMAIN]["entry"]
    assert "sample_aliases" not in record


def test_apply_heater_addresses_includes_power_monitors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Power monitor addresses should flow to the energy coordinator."""

    client, _sio, _ = _make_client(monkeypatch)
    energy_coordinator = SimpleNamespace(update_addresses=MagicMock())
    record = client.hass.data[module.DOMAIN]["entry"]
    record["energy_coordinator"] = energy_coordinator

    nodes_payload = {
        "nodes": [
            {"type": "htr", "addr": "1"},
            {"type": "pmo", "addr": "9"},
        ]
    }
    inventory = Inventory(
        client.dev_id,
        nodes_payload,
        build_node_inventory(nodes_payload),
    )

    normalized = client._apply_heater_addresses({"htr": ["1"]}, inventory=inventory)

    assert normalized["pmo"] == ["9"]
    energy_coordinator.update_addresses.assert_called_once_with(inventory)
    assert record.get("inventory") is inventory
    assert "sample_aliases" not in record

def test_apply_heater_addresses_requires_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inventory must be present when applying heater addresses."""

    client, _sio, _ = _make_client(monkeypatch)
    normalized = client._apply_heater_addresses({"acm": []}, inventory=None)
    assert normalized == {}
    assert client._coordinator.data == {}


def test_apply_heater_addresses_updates_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Applying heater addresses should refresh stored inventory when present."""

    client, _sio, _ = _make_client(monkeypatch)
    record = client.hass.data[module.DOMAIN]["entry"]
    record["energy_coordinator"] = SimpleNamespace(update_addresses=MagicMock())
    raw_nodes = {"nodes": [{"type": "htr", "addr": "1"}]}
    inventory = Inventory(
        client.dev_id,
        raw_nodes,
        build_node_inventory(raw_nodes),
    )
    client._apply_heater_addresses(
        {"htr": ["1"]},
        inventory=inventory,
    )
    assert record.get("inventory") is inventory


def test_heater_sample_subscription_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Subscription helper should forward inventory subscription targets."""

    client, _sio, _ = _make_client(monkeypatch)
    record = client.hass.data[module.DOMAIN]["entry"]
    raw_nodes = {
        "nodes": [
            {"type": "acm", "addr": "2"},
            {"type": "htr", "addr": "1"},
        ]
    }
    node_inventory = build_node_inventory(raw_nodes)
    inventory = Inventory(client.dev_id, raw_nodes, node_inventory)
    record["inventory"] = inventory

    def fake_resolve(
        record_map: Mapping[str, Any] | None,
        *,
        dev_id: str | None = None,
        nodes_payload: Any | None = None,
    ) -> Any:
        assert record_map is record
        assert dev_id == client.dev_id
        assert nodes_payload is None
        return SimpleNamespace(
            inventory=inventory,
            source="inventory",
            raw_count=len(node_inventory),
            filtered_count=len(node_inventory),
        )

    monkeypatch.setattr(module, "resolve_record_inventory", fake_resolve)

    client._inventory = inventory

    targets = client._heater_sample_subscription_targets()
    assert targets == inventory.heater_sample_targets


def test_heater_sample_subscription_targets_ignore_fallback_addrs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coordinator-provided addresses are ignored when inventory is empty."""

    client, _sio, _ = _make_client(monkeypatch)
    record = client.hass.data[module.DOMAIN]["entry"]
    client._coordinator._addrs = lambda: [" 3 ", "3", "4"]
    inventory = Inventory(client.dev_id, {}, [])

    monkeypatch.setattr(
        module,
        "resolve_record_inventory",
        lambda *_, **__: SimpleNamespace(
            inventory=inventory,
            source="inventory",
            raw_count=0,
            filtered_count=0,
        ),
    )

    targets = client._heater_sample_subscription_targets()

    assert targets == []


def test_heater_sample_subscription_targets_logs_missing_inventory(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Missing shared inventory should be logged without rebuilding."""

    client, _sio, _ = _make_client(monkeypatch)
    record = client.hass.data[module.DOMAIN]["entry"]
    record.pop("inventory", None)

    monkeypatch.setattr(
        module,
        "resolve_record_inventory",
        lambda *_, **__: SimpleNamespace(
            inventory=None,
            source="fallback",
            raw_count=0,
            filtered_count=0,
        ),
    )

    client._inventory = None
    client._coordinator._addrs = lambda: ["11"]

    with caplog.at_level(logging.ERROR):
        targets = client._heater_sample_subscription_targets()

    assert any(
        "Unable to resolve shared inventory" in record.message for record in caplog.records
    )
    assert targets == []


def test_heater_sample_targets_use_record_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Record-scoped inventory containers should be reused."""

    client, _sio, _ = _make_client(monkeypatch)
    record = client.hass.data[module.DOMAIN]["entry"]
    raw_nodes = {"nodes": [{"type": "htr", "addr": "12"}]}
    inventory = Inventory(
        client.dev_id,
        raw_nodes,
        build_node_inventory(raw_nodes),
    )
    record["inventory"] = inventory
    client._inventory = None

    def fake_resolve(
        record_map: Mapping[str, Any] | None,
        *,
        dev_id: str | None = None,
        nodes_payload: Any | None = None,
    ) -> Any:
        assert record_map is record
        assert dev_id == client.dev_id
        assert nodes_payload is None
        return SimpleNamespace(
            inventory=inventory,
            source="inventory",
            raw_count=len(inventory.nodes),
            filtered_count=len(inventory.nodes),
        )

    monkeypatch.setattr(module, "resolve_record_inventory", fake_resolve)

    targets = client._heater_sample_subscription_targets()

    assert targets == inventory.heater_sample_targets
    assert client._inventory is inventory


def test_heater_sample_targets_build_from_record_raw_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw node payloads should seed inventory rebuilds when cached inventory is missing."""

    client, _sio, _ = _make_client(monkeypatch)
    record = client.hass.data[module.DOMAIN]["entry"]
    raw_nodes = {"nodes": [{"type": "htr", "addr": "13"}]}
    record.clear()
    record["nodes"] = raw_nodes
    record.pop("inventory", None)

    client._inventory = None

    targets = client._heater_sample_subscription_targets()

    assert isinstance(client._inventory, Inventory)
    assert client._inventory.payload == raw_nodes
    assert any(node.addr == "13" for node in client._inventory.nodes)
    assert targets == client._inventory.heater_sample_targets


def test_apply_heater_addresses_filters_non_heaters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-heater node types should be ignored when applying addresses."""

    client, _sio, _ = _make_client(monkeypatch)
    record = client.hass.data[module.DOMAIN]["entry"]

    nodes_payload = {"nodes": [{"type": "htr", "addr": "6"}]}
    inventory_container = Inventory(
        "device",
        nodes_payload,
        build_node_inventory(nodes_payload),
    )

    normalized = client._apply_heater_addresses(
        {"foo": ["5"], "htr": ["6"]},
        inventory=inventory_container,
    )

    assert "foo" not in normalized
    assert normalized["htr"] == ["6"]


def test_apply_heater_addresses_logs_invalid_inventory(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unexpected inventory inputs should be ignored with a debug message."""

    client, _sio, _ = _make_client(monkeypatch)
    record = client.hass.data[module.DOMAIN]["entry"]

    raw_nodes = {"nodes": [{"type": "htr", "addr": "4"}]}
    inventory = Inventory(
        client.dev_id,
        raw_nodes,
        build_node_inventory(raw_nodes),
    )
    client._inventory = inventory

    with caplog.at_level("DEBUG"):
        normalized = client._apply_heater_addresses(
            {"htr": ["4"]},
            inventory=[SimpleNamespace(type="htr", addr="4")],
        )

    assert normalized["htr"] == ["4"]
    assert "ignoring unexpected inventory container" in caplog.text






@pytest.mark.asyncio
async def test_subscribe_heater_samples_emits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Subscribing to heater samples should emit for each address."""

    client, sio, _ = _make_client(monkeypatch)
    monkeypatch.setattr(
        client,
        "_heater_sample_subscription_targets",
        lambda: [("htr", "1"), ("aux", "2")],
    )
    await client._subscribe_heater_samples()
    assert ("subscribe", "/htr/1/samples", module.WS_NAMESPACE) in sio.emit_calls
    assert ("subscribe", "/aux/2/samples", module.WS_NAMESPACE) in sio.emit_calls


@pytest.mark.asyncio
async def test_subscribe_heater_samples_logs_errors(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Subscribing should log when emit fails."""

    client, sio, _ = _make_client(monkeypatch)
    monkeypatch.setattr(client, "_heater_sample_subscription_targets", lambda: [("htr", "1")])
    sio.emit = AsyncMock(side_effect=RuntimeError("boom"))
    caplog.set_level(logging.DEBUG)
    await client._subscribe_heater_samples()
    assert "sample subscription setup failed" in caplog.text


@pytest.mark.asyncio
async def test_schedule_idle_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scheduling an idle restart should create a task and reset flags afterwards."""

    loop = asyncio.get_running_loop()
    hass_loop = SimpleNamespace(
        create_task=lambda coro, **kwargs: loop.create_task(coro, **kwargs),
        call_soon_threadsafe=lambda cb, *args: loop.call_soon(cb, *args),
    )
    client, _sio, _ = _make_client(monkeypatch, hass_loop=hass_loop)
    client._closing = False
    client._schedule_idle_restart(idle_for=10, source="test")
    assert client._idle_restart_pending is True
    task = client._idle_restart_task
    assert task is not None
    await asyncio.sleep(0)
    await task
    assert client._idle_restart_pending is False


def test_schedule_idle_restart_ignored_when_closing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scheduling should be skipped when already closing."""

    client, _sio, _ = _make_client(monkeypatch)
    client._closing = True
    client._schedule_idle_restart(idle_for=10, source="closing")
    assert client._idle_restart_task is None


@pytest.mark.asyncio
async def test_cancel_idle_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancelling an idle restart should cancel the task."""

    loop = asyncio.get_running_loop()
    hass_loop = SimpleNamespace(
        create_task=lambda coro, **kwargs: loop.create_task(coro, **kwargs),
        call_soon_threadsafe=lambda cb, *args: loop.call_soon(cb, *args),
    )
    client, _sio, _ = _make_client(monkeypatch, hass_loop=hass_loop)
    client._closing = False
    client._schedule_idle_restart(idle_for=10, source="test")
    task = client._idle_restart_task
    assert task is not None
    client._cancel_idle_restart()
    assert client._idle_restart_task is None


def test_header_sanitizers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Header and URL sanitisation helpers should redact sensitive values."""

    client, _sio, _ = _make_client(monkeypatch)

    headers = client._brand_headers(origin="https://app")
    assert headers["X-Requested-With"] == "requested"
    client._requested_with = ""
    headers = client._brand_headers(origin="https://app")
    assert headers["Origin"] == "https://app"
    assert headers["User-Agent"] == "agent"
    assert headers["Accept-Language"] == module.ACCEPT_LANGUAGE

    assert redact_token_fragment("   ") == ""
    assert redact_token_fragment("") == ""
    assert redact_token_fragment("abc") == "***"
    assert redact_token_fragment("abcdefgh") == "ab***gh"
    assert redact_token_fragment("abcdefghijklmnop") == "abcd...mnop"

    assert mask_identifier("abcd") == "***"
    assert mask_identifier("abcdefgh") == "ab...gh"
    assert mask_identifier("abcdefghijklmnop") == "abcdef...mnop"

    sanitised = client._sanitise_headers(
        {
            "Authorization": "Bearer secret-token",
            "Cookie": "session",
            "X-Test": b"value",
        }
    )
    assert "..." in sanitised["Authorization"]
    assert "***" in sanitised["Cookie"]
    assert sanitised["X-Test"] == "value"

    sanitised = client._sanitise_headers({"Authorization": "token"})
    assert "***" in sanitised["Authorization"]

    params = client._sanitise_params(
        {"token": "abc12345", "dev_id": "dev123", "sid": "session", "q": "ok"}
    )
    assert params["token"] == "{token}"
    assert params["dev_id"] == "{dev_id}"
    assert params["sid"] == "{sid}"

    sanitised_url = client._sanitise_url(
        "https://host/socket?token=abc&dev_id=12345&sid=session"
    )
    sanitised_query = dict(parse_qsl(urlsplit(sanitised_url).query))
    assert sanitised_query["token"] == "{token}"
    assert sanitised_query["dev_id"] == "{dev_id}"
    assert sanitised_query["sid"] == "{sid}"
    sanitised_ws_url = client._sanitise_url(
        "https://host/socket.io/1/websocket/abc123?transport=websocket&sid=session"
    )
    parsed_ws_url = urlsplit(sanitised_ws_url)
    ws_query = dict(parse_qsl(parsed_ws_url.query))
    assert parsed_ws_url.path.endswith("/socket.io/1/websocket/{sid}")
    assert ws_query["sid"] == "{sid}"
    assert client._sanitise_url("not a url") == "not a url"
    assert client._sanitise_url("http://[::1") == "http://[::1"


def test_redaction_helpers_handle_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Token and identifier masking should treat whitespace as empty."""

    client, _sio, _ = _make_client(monkeypatch)
    assert redact_token_fragment("   ") == ""
    assert mask_identifier("   ") == ""
    params = client._sanitise_params(
        {"token": "   ", "dev_id": "   ", "sid": "   "}
    )
    assert params["token"] == "{token}"
    assert params["dev_id"] == "{dev_id}"
    assert params["sid"] == "{sid}"


@pytest.mark.asyncio
async def test_wrap_background_task_handles_sync_callable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-coroutine targets should be wrapped in an async task."""

    loop = asyncio.get_running_loop()
    hass_loop = SimpleNamespace(
        create_task=lambda coro, **kwargs: loop.create_task(coro, **kwargs),
        call_soon_threadsafe=lambda cb, *args: loop.call_soon(cb, *args),
    )
    client, _sio, _ = _make_client(monkeypatch, hass_loop=hass_loop)

    result: list[int] = []

    def add_value(value: int) -> int:
        result.append(value)
        return value * 2

    task = client._wrap_background_task(add_value, 3)
    await task

    assert result == [3]
    assert asyncio.iscoroutine(task.result())


@pytest.mark.asyncio
async def test_start_returns_existing_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """start should return the original task when invoked multiple times."""

    loop = asyncio.get_running_loop()
    hass_loop = SimpleNamespace(
        create_task=lambda coro, **kwargs: loop.create_task(coro, **kwargs),
        call_soon_threadsafe=lambda cb, *args: loop.call_soon(cb, *args),
    )
    client, _sio, _ = _make_client(monkeypatch, hass_loop=hass_loop)

    ready = asyncio.Event()

    async def fake_runner() -> None:
        await ready.wait()

    monkeypatch.setattr(client, "_runner", fake_runner)

    task1 = client.start()
    task2 = client.start()
    assert task1 is task2

    ready.set()
    await asyncio.wait_for(task1, timeout=0.1)


@pytest.mark.asyncio
async def test_start_and_stop_manage_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    """start should spawn tasks and stop should cancel them cleanly."""

    loop = asyncio.get_running_loop()
    hass_loop = SimpleNamespace(
        create_task=lambda coro, **kwargs: loop.create_task(coro, **kwargs),
        call_soon_threadsafe=lambda cb, *args: loop.call_soon(cb, *args),
    )
    client, _sio, _ = _make_client(monkeypatch, hass_loop=hass_loop)

    runner_gate = asyncio.Event()

    async def runner() -> None:
        try:
            await runner_gate.wait()
        except asyncio.CancelledError:
            raise

    monkeypatch.setattr(client, "_runner", runner)
    monkeypatch.setattr(client, "_disconnect", AsyncMock())
    client._idle_restart_task = loop.create_task(asyncio.sleep(0))
    client._idle_monitor_task = loop.create_task(asyncio.sleep(0))

    task = client.start()
    assert client.is_running() is True
    runner_gate.set()
    await asyncio.sleep(0)
    await client.stop()
    assert client.is_running() is False
    assert task.cancelled() or task.done()


def test_handle_connection_lost_records_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Connection loss metadata should be stored in the state bucket."""

    client, _sio, _ = _make_client(monkeypatch)
    bucket = client._ws_state_bucket()
    loop = asyncio.new_event_loop()
    loop.run_until_complete(client._handle_connection_lost(RuntimeError("boom")))
    loop.close()
    assert bucket["restart_count"] == 1
    assert "RuntimeError" in bucket["last_disconnect_error"]


@pytest.mark.asyncio
async def test_build_engineio_target_handles_invalid_base(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid API bases should raise runtime errors."""

    client, _sio, _ = _make_client(monkeypatch)
    url, path = await client._build_engineio_target()
    assert path == "socket.io"
    assert "token" in url and "dev_id" in url

    monkeypatch.setattr(client, "_api_base", lambda: "http://")
    with pytest.raises(RuntimeError):
        await client._build_engineio_target()


def test_register_debug_catch_all_requires_debug(monkeypatch: pytest.MonkeyPatch) -> None:
    """Debug catch-all should register only when debugging is enabled."""

    client, sio, _ = _make_client(monkeypatch)
    client._debug_catch_all_registered = False
    monkeypatch.setattr(module._LOGGER, "isEnabledFor", lambda level: True)
    client._register_debug_catch_all()
    assert client._debug_catch_all_registered is True
    assert ("*", client._namespace) in sio.events


@pytest.mark.asyncio
async def test_event_handlers_update_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    """dev_handshake, dev_data, and update events should update counters."""

    client, sio, _ = _make_client(monkeypatch)
    monkeypatch.setattr(client, "_handle_dev_data", MagicMock())
    monkeypatch.setattr(client, "_handle_update", MagicMock())
    monkeypatch.setattr(client, "_subscribe_heater_samples", AsyncMock())
    monkeypatch.setattr(module._LOGGER, "isEnabledFor", lambda level: True)

    await client._on_dev_handshake({"hello": "world"})
    assert client._stats.frames_total == 1

    await client._on_dev_data({"nodes": {}})
    await client._on_update({})
    assert client._stats.frames_total == 3
    client._handle_dev_data.assert_called_once()
    client._handle_update.assert_called_once()


@pytest.mark.asyncio
async def test_refresh_subscription_behaviour(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refreshing subscriptions should emit dev_data and resubscribe."""

    client, sio, _ = _make_client(monkeypatch)
    sio._connected = True
    emit = AsyncMock()
    monkeypatch.setattr(client._sio, "emit", emit)
    monkeypatch.setattr(client, "_subscribe_heater_samples", AsyncMock())
    monkeypatch.setattr(module._LOGGER, "isEnabledFor", lambda level: True)

    await client._refresh_subscription(reason="manual")
    emit.assert_called_with("dev_data", namespace=client._namespace)
    assert client._subscription_refresh_failed is False

    sio._connected = False
    with pytest.raises(RuntimeError):
        await client._refresh_subscription(reason="not connected")


def test_apply_nodes_payload_translation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Node payload application should merge data and notify listeners."""

    client, _sio, dispatcher = _make_client(monkeypatch)
    original_dispatch = client._dispatch_nodes
    client._dispatch_nodes = MagicMock(side_effect=original_dispatch)
    client._handshake_payload = {"nodes": {}}
    client._handle_handshake({"nodes": {"htr": {"status": {"1": {"temp": 20}}}}})
    client._apply_nodes_payload(
        {"nodes": {"htr": {"status": {"1": {"temp": 25}}}}}, merge=True, event="update"
    )
    client._dispatch_nodes.assert_called_with({"htr": {"status": {"1": {"temp": 25}}}})
    dispatcher.assert_called()


def test_translate_nodes_list_handles_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    """List-based node payloads should be normalised into mappings."""

    client, _sio, _ = _make_client(monkeypatch)
    nodes = [
        {"type": "htr", "addr": "1", "status": {"temp": 21}},
        {"type": "htr", "addr": "1", "advanced_setup": {"mode": "eco"}},
        {"type": "bad", "addr": None},
    ]
    translated = client._translate_nodes_list(nodes)
    assert translated["htr"]["status"]["1"]["temp"] == 21
    assert translated["htr"]["advanced"]["1"]["advanced_setup"]["mode"] == "eco"


def test_forward_sample_updates_invokes_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forwarding sample updates should notify the energy coordinator handler."""

    client, _sio, _ = _make_client(monkeypatch)
    handler_called: dict[str, Any] = {}
    energy_handler = SimpleNamespace(
        handle_ws_samples=lambda dev_id, payload, **kwargs: handler_called.update({
            "dev_id": dev_id,
            "payload": payload,
            "lease": kwargs.get("lease_seconds"),
        })
    )
    client.hass.data[module.DOMAIN]["entry"]["energy_coordinator"] = energy_handler
    client._forward_sample_updates({"htr": {"samples": {"1": {"temp": 20}}}})
    assert handler_called["dev_id"] == "device"
    assert handler_called["payload"]["htr"]["1"]["temp"] == 20


def test_extract_nodes_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    """_extract_nodes should handle dicts, lists, and invalid payloads."""

    client, _sio, _ = _make_client(monkeypatch)
    assert client._extract_nodes({"nodes": {"htr": {}}}) == {"htr": {}}
    converted = client._extract_nodes({"nodes": [{"type": "htr", "addr": "1", "status": {}}]})
    assert "htr" in converted
    assert client._extract_nodes("not a dict") is None


def test_resolve_update_section_variants() -> None:
    """Update section resolver should map known segments consistently."""

    assert module.WebSocketClient._resolve_update_section(None) == (None, None)
    assert module.WebSocketClient._resolve_update_section("status") == ("status", None)
    assert module.WebSocketClient._resolve_update_section("advanced_setup") == (
        "advanced",
        "advanced_setup",
    )
    assert module.WebSocketClient._resolve_update_section("setup") == ("settings", "setup")
    assert module.WebSocketClient._resolve_update_section("unknown") == ("settings", "unknown")

