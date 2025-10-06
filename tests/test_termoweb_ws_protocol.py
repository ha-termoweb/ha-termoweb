"""Extended tests for TermoWeb websocket protocol flows."""

from __future__ import annotations

import asyncio
import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.termoweb.backend import termoweb_ws as module


class DummyREST:
    """Provide just enough of the REST client interface for websocket tests."""

    def __init__(self) -> None:
        self._session = SimpleNamespace(closed=True)
        self._ensure_token = AsyncMock()
        self._authed_headers = AsyncMock(return_value={"Authorization": "Bearer token"})
        self.api_base = "https://api.termoweb"
        self.user_agent = "agent"
        self.requested_with = "requested"


class StubAsyncClient:
    """Socket.IO client stub recording method invocations."""

    def __init__(self, allow_http_error: bool = False, **_: Any) -> None:
        object.__setattr__(self, "events", {})
        object.__setattr__(self, "_connected", False)
        object.__setattr__(self, "connect_calls", [])
        object.__setattr__(self, "disconnect_calls", 0)
        object.__setattr__(self, "emit_calls", [])
        object.__setattr__(self, "_allow_http_error", allow_http_error)
        object.__setattr__(self, "_http_attempts", 0)
        object.__setattr__(self, "eio", SimpleNamespace(start_background_task=None, http=None))
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
) -> tuple[module.WebSocketClient, StubAsyncClient, MagicMock]:
    """Instantiate a ``WebSocketClient`` with a controllable AsyncClient stub."""

    holder: dict[str, StubAsyncClient] = {}

    def factory(**kwargs: Any) -> StubAsyncClient:
        stub = StubAsyncClient(allow_http_error=allow_http_error, **kwargs)
        holder["client"] = stub
        return stub

    monkeypatch.setattr(module.socketio, "AsyncClient", factory)
    dispatcher = MagicMock()
    monkeypatch.setattr(module, "async_dispatcher_send", dispatcher)

    if hass_loop is None:
        hass_loop = SimpleNamespace(
            create_task=lambda coro, **_: SimpleNamespace(done=lambda: False),
            call_soon_threadsafe=lambda cb, *args: cb(*args),
        )

    hass = SimpleNamespace(loop=hass_loop, data={module.DOMAIN: {"entry": {}}})
    coordinator = SimpleNamespace(update_nodes=MagicMock(), data={})
    client = module.WebSocketClient(
        hass,
        entry_id="entry",
        dev_id="device",
        api_client=DummyREST(),
        coordinator=coordinator,
        session=SimpleNamespace(closed=True),
    )
    return client, holder["client"], dispatcher


def test_handshake_error_exposes_fields() -> None:
    """The TermoWeb handshake error should record status, URL and body."""

    error = module.HandshakeError(503, "https://example/ws", "body")
    assert str(error) == "handshake failed (status=503)"
    assert error.status == 503
    assert error.url == "https://example/ws"
    assert error.body_snippet == "body"


def test_init_handles_socketio_http_attribute(monkeypatch: pytest.MonkeyPatch) -> None:
    """Initialisation should recover when AsyncClient rejects ``http`` assignment."""

    client, sio, _ = _make_client(monkeypatch, allow_http_error=True)
    assert isinstance(client, module.WebSocketClient)
    assert sio._http_attempts == 1
    assert sio.http is not None
    assert sio.eio.http is not None


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

    async def disconnect(**_: Any) -> None:
        call_order.append("disconnect")

    async def lost(error: Exception | None) -> None:
        call_order.append(f"lost:{type(error).__name__ if error else 'none'}")
        client._closing = True

    client._connect_once = AsyncMock(side_effect=connect_once)  # type: ignore[attr-defined]
    client._wait_for_events = AsyncMock(side_effect=wait_for_events)  # type: ignore[attr-defined]
    client._disconnect = AsyncMock(side_effect=disconnect)  # type: ignore[attr-defined]
    client._handle_connection_lost = AsyncMock(side_effect=lost)  # type: ignore[attr-defined]

    await client._runner()

    assert call_order == ["connect", "wait", "disconnect", "lost:none"]
    dispatcher.assert_called()


@pytest.mark.asyncio
async def test_handle_connection_lost_updates_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Losing the connection should persist restart metadata."""

    client, _sio, _ = _make_client(monkeypatch)
    state = client._ws_state_bucket()
    assert state.get("restart_count") is None

    await client._handle_connection_lost(RuntimeError("boom"))

    assert state["restart_count"] == 1
    assert "RuntimeError" in state["last_disconnect_error"]


@pytest.mark.asyncio
async def test_disconnect_calls_socketio(monkeypatch: pytest.MonkeyPatch) -> None:
    """_disconnect should call the AsyncClient when connected."""

    client, sio, _ = _make_client(monkeypatch)
    await sio.connect()
    await client._disconnect(reason="tests")
    assert sio.disconnect_calls == 1
    assert client._disconnected.is_set()


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
        task = loop.create_task(asyncio.sleep(0))
        created.append(task)
        return task

    hass_loop = SimpleNamespace(
        create_task=create_task,
        call_soon_threadsafe=lambda cb, *args: loop.call_soon(cb, *args),
    )
    client, _sio, dispatcher = _make_client(monkeypatch, hass_loop=hass_loop)
    await client._on_connect()
    assert created
    dispatcher.assert_called()


@pytest.mark.asyncio
async def test_namespace_connect_emits_join(monkeypatch: pytest.MonkeyPatch) -> None:
    """Joining the namespace should emit join and dev_data events."""

    client, sio, _ = _make_client(monkeypatch)
    await client._on_namespace_connect()
    assert ("join", None, module.WS_NAMESPACE) in sio.emit_calls
    assert ("dev_data", None, module.WS_NAMESPACE) in sio.emit_calls


def test_register_debug_catch_all_installs_handler(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Debug catch-all registration should wrap the AsyncClient when DEBUG is enabled."""

    client, sio, _ = _make_client(monkeypatch)
    caplog.set_level("DEBUG", logger=module._LOGGER.name)
    client._register_debug_catch_all()
    assert ("*", module.WS_NAMESPACE) in sio.events


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
async def test_refresh_subscription_emits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refreshing the subscription should emit dev_data and resubscribe samples."""

    client, sio, _ = _make_client(monkeypatch)
    await sio.connect()
    monkeypatch.setattr(client, "_subscribe_heater_samples", AsyncMock())
    await client._refresh_subscription(reason="timer")
    assert ("dev_data", None, module.WS_NAMESPACE) in sio.emit_calls
    client._subscribe_heater_samples.assert_awaited()  # type: ignore[attr-defined]


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


def test_translate_path_update_and_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Path based updates should map onto node sections."""

    client, _sio, _ = _make_client(monkeypatch)
    payload = {
        "path": "/api/devs/device/htr/1/settings/temp",
        "body": {"value": 20},
    }
    translated = client._translate_path_update(payload)
    assert translated == {"htr": {"settings": {"1": {"temp": {"value": 20}}}}}
    assert module.WebSocketClient._resolve_update_section("advanced_setup") == ("advanced", "advanced_setup")
    assert module.WebSocketClient._resolve_update_section("prog") == ("settings", "prog")
    assert module.WebSocketClient._resolve_update_section(None) == (None, None)


def test_forward_sample_updates_invokes_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sample update forwarding should call the energy coordinator hook."""

    client, _sio, _ = _make_client(monkeypatch)
    handler = MagicMock()
    client.hass.data[module.DOMAIN]["entry"]["energy_coordinator"] = SimpleNamespace(
        handle_ws_samples=handler
    )
    client._forward_sample_updates({"htr": {"1": {"power": 10}}})
    handler.assert_called_once()


def test_extract_and_translate_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    """List based node payloads should be converted into the mapping schema."""

    client, _sio, _ = _make_client(monkeypatch)
    payload = {"nodes": [{"type": "htr", "addr": "1", "settings": {"temp": 20}}]}
    nodes = client._extract_nodes(payload)
    assert nodes and "htr" in nodes
    assert payload["nodes"]["htr"]


def test_apply_nodes_payload_merges_and_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Applying node payloads should build snapshots and dispatch updates."""

    client, _sio, dispatcher = _make_client(monkeypatch)
    client._collect_update_addresses = MagicMock(return_value=[("htr", "1")])  # type: ignore[attr-defined]
    client._dispatch_nodes = MagicMock(return_value={"htr": ["1"]})  # type: ignore[attr-defined]
    client._forward_sample_updates = MagicMock()  # type: ignore[attr-defined]
    client._mark_event = MagicMock()  # type: ignore[attr-defined]

    snapshot_payload = {"nodes": {"htr": {"settings": {"1": {"temp": 20}}}}}
    client._apply_nodes_payload(snapshot_payload, merge=False, event="dev_data")
    assert client._nodes["nodes"]["htr"]

    update_payload = {"path": "/api/devs/device/htr/1/samples", "body": {"power": 5}}
    client._apply_nodes_payload(update_payload, merge=True, event="update")
    client._forward_sample_updates.assert_called()
    client._mark_event.assert_called()


def test_ensure_type_bucket_and_build_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Helper methods should populate node buckets and snapshot structures."""

    client, _sio, _ = _make_client(monkeypatch)
    dev_map: dict[str, Any] = {}
    nodes_by_type: dict[str, Any] = {}
    bucket = client._ensure_type_bucket(dev_map, nodes_by_type, "htr")
    assert "settings" in bucket and dev_map["htr"] is bucket
    snapshot = module.WebSocketClient._build_nodes_snapshot({"htr": {"settings": {"1": {}}}})
    assert "nodes" in snapshot and "nodes_by_type" in snapshot


def test_apply_heater_addresses_updates_coordinator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Applying heater addresses should update the coordinator data map."""

    client, _sio, _ = _make_client(monkeypatch)
    client._coordinator.data = {"device": {}}
    normalized = client._apply_heater_addresses({"htr": ["1"]}, inventory=[("htr", "1")])
    assert normalized["htr"] == ["1"]
    assert client._coordinator.data["device"]["nodes_by_type"]


def test_heater_sample_subscription_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Subscription helper should normalise addresses before returning targets."""

    client, _sio, _ = _make_client(monkeypatch)
    monkeypatch.setattr(
        module,
        "collect_heater_sample_addresses",
        lambda record, coordinator=None: ([("htr", "1")], {"htr": ["1"]}, {}),
    )
    targets = client._heater_sample_subscription_targets()
    assert targets == [("htr", "1")]


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


def test_header_sanitizers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Header and URL sanitisation helpers should redact sensitive values."""

    client, _sio, _ = _make_client(monkeypatch)
    client._requested_with = ""

    headers = client._brand_headers(origin="https://app")
    assert headers["Origin"] == "https://app"
    assert headers["User-Agent"] == "agent"
    assert headers["Accept-Language"] == module.ACCEPT_LANGUAGE

    assert client._redact_value("") == ""
    assert client._redact_value("abc") == "***"
    assert client._redact_value("abcdefgh") == "ab***gh"
    assert client._redact_value("abcdefghijklmnop") == "abcd...mnop"

    assert client._mask_identifier("abcd") == "***"
    assert client._mask_identifier("abcdefgh") == "ab...gh"
    assert client._mask_identifier("abcdefghijklmnop") == "abcdef...mnop"

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

    params = client._sanitise_params({"token": "abc12345", "dev_id": "dev123", "q": "ok"})
    assert params["token"].startswith("ab") and params["token"].endswith("45")
    assert params["dev_id"].startswith("de") and params["dev_id"].endswith("23")

    sanitised_url = client._sanitise_url("https://host/socket?token=abc&dev_id=12345")
    assert "abc" not in sanitised_url
    assert "12345" not in sanitised_url
    assert client._sanitise_url("not a url") == "not a url"


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
    client._nodes = {}
    client._nodes_raw = {}
    client._handshake_payload = {"nodes": {}}
    client._handle_handshake({"nodes": {"htr": {"status": {"1": {"temp": 20}}}}})
    client._apply_nodes_payload(
        {"nodes": {"htr": {"status": {"1": {"temp": 25}}}}}, merge=True, event="update"
    )
    assert client._nodes["htr"]["status"]["1"]["temp"] == 25
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
        handle_ws_samples=lambda dev_id, payload: handler_called.update({
            "dev_id": dev_id,
            "payload": payload,
        })
    )
    client.hass.data[module.DOMAIN]["entry"]["energy_coordinator"] = energy_handler
    client._forward_sample_updates({"htr": {"samples": {"1": {"temp": 20}}}})
    assert handler_called["dev_id"] == "device"
    assert handler_called["payload"]["htr"]["samples"]["1"]["temp"] == 20


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

