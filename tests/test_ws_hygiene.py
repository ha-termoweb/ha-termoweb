"""Regression tests for WS status hygiene, diagnostics persistence and teardown."""

from __future__ import annotations

import asyncio
import logging
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiohttp
from conftest import DummyREST, build_entry_runtime, listen_ws_status
import pytest
from test_ducaheat_ws_protocol import QueueWebSocket, StubWebSocket, _make_client

from custom_components.termoweb.backend import ducaheat_ws, termoweb_ws
from custom_components.termoweb.inventory import Inventory, build_node_inventory


def _text(data: str) -> SimpleNamespace:
    """Return a websocket TEXT message."""

    return SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=data)


def _termoweb_client(**kwargs: Any) -> termoweb_ws.TermoWebWSClient:
    """Build a production TermoWeb client on a stub runtime."""

    loop = SimpleNamespace(create_task=lambda coro, **_: coro.close())
    hass = SimpleNamespace(loop=loop, data={})
    inventory = Inventory(
        "device", build_node_inventory([{"type": "htr", "addr": "1"}])
    )
    build_entry_runtime(
        hass=hass, entry_id="entry", dev_id="device", inventory=inventory
    )
    api = kwargs.pop("api_client", None) or DummyREST()
    return termoweb_ws.TermoWebWSClient(
        hass,
        entry_id="entry",
        dev_id="device",
        api_client=api,
        coordinator=kwargs.pop("coordinator", SimpleNamespace()),
        session=SimpleNamespace(closed=False),
        inventory=inventory,
    )


# ---------------------------------------------------------------------------
# Status notifications only on change (B4/B5, review M4)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ducaheat_pongs_while_stale_do_not_flap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stale payload + periodic pongs: one healthy->connected change, no flapping."""

    client = _make_client(monkeypatch)
    listener = listen_ws_status(monkeypatch, client)
    tracker = client._ws_health_tracker()
    now = time.time()
    client._mark_ws_payload(timestamp=now - 1000, stale_after=240)
    client._update_status("healthy")
    tracker.refresh_payload_state(now=now)
    assert tracker.payload_stale is True
    listener.reset_mock()
    seen: list[str] = []
    real_update = client._update_status

    def _record(status: str) -> None:
        seen.append(status)
        real_update(status)

    monkeypatch.setattr(client, "_update_status", _record)
    client._ws = QueueWebSocket([_text("3") for _ in range(6)])

    with pytest.raises(RuntimeError, match="websocket closed"):
        await client._read_loop_ws()

    assert seen == ["connected"]
    assert tracker.status == "connected"
    statuses = [call.args[0]["status"] for call in listener.call_args_list]
    assert statuses == ["connected"]


@pytest.mark.asyncio
async def test_ducaheat_fresh_frames_do_not_dispatch_per_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """While healthy and fresh, pongs and updates must not re-dispatch status."""

    client = _make_client(monkeypatch)
    client._mark_ws_payload(timestamp=time.time(), stale_after=240)
    client._update_status("healthy")
    listener = listen_ws_status(monkeypatch, client)
    update = '42/api/v2/socket_io,["update",{"path":"/htr/1/status","body":{"stemp":"20.0"}}]'
    client._ws = QueueWebSocket([_text("3"), _text(update), _text("3"), _text(update)])

    with pytest.raises(RuntimeError, match="websocket closed"):
        await client._read_loop_ws()

    assert client._ws_health_tracker().status == "healthy"
    status_calls = [
        c for c in listener.call_args_list if c.args[0]["reason"] == "status"
    ]
    assert status_calls == []


def test_update_status_notifies_only_on_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeating the same status must neither dispatch nor recompute polling."""

    client = _termoweb_client()
    listener = listen_ws_status(monkeypatch, client)
    sync = MagicMock()
    monkeypatch.setattr(client, "_sync_gateway_connection_state", sync)

    for _ in range(3):
        client._update_status("connected")
    client._update_status("healthy")
    client._update_status("healthy")

    assert [c.args[0]["status"] for c in listener.call_args_list] == [
        "connected",
        "healthy",
    ]
    assert sync.call_count == 2


# ---------------------------------------------------------------------------
# Diagnostics counters survive reconnects (B6, review M11)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ducaheat_counters_survive_disconnect_cleared_on_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reconnect keeps cumulative counters and the tracker; stop clears them."""

    client = _make_client(monkeypatch)
    runtime = client.hass.data["termoweb"]["entry"]
    client._increment_state_counter("subscribe_attempts_total")
    client._increment_state_counter("subscribe_fail_total")
    client._increment_state_counter("recovery_attempts_total")
    client._record_parse_error(now=time.time(), reason="json")
    tracker = client._ws_health_tracker()

    await client._disconnect("loop")
    await client._disconnect("loop")

    state = runtime.ws_state["device"]
    assert state["subscribe_attempts_total"] == 1
    assert state["subscribe_fail_total"] == 1
    assert state["recovery_attempts_total"] == 1
    assert state["parse_errors_total"] == 1
    assert runtime.ws_trackers["device"] is tracker
    assert client._ws_health_tracker() is tracker

    await client.stop()

    assert "device" not in runtime.ws_state
    assert "device" not in runtime.ws_trackers


def test_ducaheat_payload_window_is_documented_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reading the tracker must not silently shrink the 240 s default window."""

    client = _make_client(monkeypatch)
    tracker = client._ws_health_tracker()
    _ = client._ws_health

    assert client._payload_stale_after == ducaheat_ws._PAYLOAD_WINDOW_DEFAULT
    assert tracker.payload_stale_after == ducaheat_ws._PAYLOAD_WINDOW_DEFAULT


# ---------------------------------------------------------------------------
# Parse-error disconnect is single and serialized (B9, review M8)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ducaheat_parse_errors_cause_exactly_one_disconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three bad frames end the read loop; the runner disconnects exactly once."""

    client = _make_client(monkeypatch)
    client._loop = asyncio.get_running_loop()
    client._status = "connected"
    disconnects: list[str] = []
    real_disconnect = client._disconnect

    async def _counting(reason: str) -> None:
        disconnects.append(reason)
        await real_disconnect(reason)

    monkeypatch.setattr(client, "_disconnect", _counting)
    frames = [_text("42{bad"), _text("42{bad"), _text("42[]"), _text("3")]
    ws = QueueWebSocket(frames)
    monkeypatch.setattr(client, "_connect_once", AsyncMock())
    client._ws = ws

    real_sleep = asyncio.sleep

    async def _sleep(_delay: float) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(ducaheat_ws.asyncio, "sleep", _sleep)
    client._connect_limiter._sleep = AsyncMock()

    with pytest.raises(asyncio.CancelledError):
        await client._runner()
    await real_sleep(0)

    assert disconnects == ["loop"]
    assert ws._index == 3  # the frame after the third error was never read


@pytest.mark.asyncio
async def test_ducaheat_disconnects_are_serialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent disconnect calls run one after the other, never interleaved."""

    client = _make_client(monkeypatch)
    active: list[int] = []
    overlaps: list[int] = []
    real_stop = client._stop_idle_monitor

    async def _slow_stop() -> None:
        active.append(1)
        if len(active) > 1:
            overlaps.append(len(active))
        await asyncio.sleep(0)
        await real_stop()
        active.pop()

    monkeypatch.setattr(client, "_stop_idle_monitor", _slow_stop)

    await asyncio.gather(client._disconnect("a"), client._disconnect("b"))

    assert overlaps == []


# ---------------------------------------------------------------------------
# Handler failures are logged at ERROR and do not kill the loop (review M12)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ducaheat_delta_handler_error_logged_and_loop_continues(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A failing ``handle_ws_deltas`` is an ERROR; later frames still process."""

    client = _make_client(monkeypatch)
    client._coordinator.handle_ws_deltas = MagicMock(side_effect=ValueError("boom"))
    update = '42/api/v2/socket_io,["update",{"path":"/htr/1/status","body":{"stemp":"20.0"}}]'
    client._ws = QueueWebSocket([_text(update), _text(update)])

    with (
        caplog.at_level(logging.DEBUG, logger=ducaheat_ws.__name__),
        pytest.raises(RuntimeError, match="websocket closed"),
    ):
        await client._read_loop_ws()

    assert client._coordinator.handle_ws_deltas.call_count == 2
    errors = [
        r for r in caplog.records if "failed to apply websocket deltas" in r.message
    ]
    assert errors and all(r.levelno == logging.ERROR for r in errors)


def test_termoweb_delta_handler_error_logged_at_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """TermoWeb surfaces state-store failures at ERROR instead of DEBUG."""

    coordinator = SimpleNamespace(handle_ws_deltas=MagicMock(side_effect=KeyError("x")))
    client = _termoweb_client(coordinator=coordinator)

    with caplog.at_level(logging.DEBUG, logger=termoweb_ws.__name__):
        client._handle_update({"nodes": {"htr": {"settings": {"1": {"stemp": "20"}}}}})

    errors = [
        r for r in caplog.records if "failed to apply websocket deltas" in r.message
    ]
    assert [r.levelno for r in errors] == [logging.ERROR]


def test_sample_handler_error_logged_above_debug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Energy sample handler failures must be visible, not hidden at DEBUG."""

    client = _termoweb_client()
    runtime = client.hass.data["termoweb"]["entry"]
    runtime.energy_coordinator = SimpleNamespace(
        handle_ws_samples=MagicMock(side_effect=RuntimeError("bad"))
    )

    with caplog.at_level(logging.DEBUG):
        client._forward_sample_updates({"htr": {"samples": {"1": {"counter": 1}}}})

    errors = [
        r for r in caplog.records if "forwarding heater samples failed" in r.message
    ]
    assert [r.levelno for r in errors] == [logging.ERROR]


# ---------------------------------------------------------------------------
# Cancellation propagates (B11)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_termoweb_runner_propagates_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling the TermoWeb runner leaves the task cancelled, not completed."""

    client = _termoweb_client()
    started = asyncio.Event()

    async def _handshake() -> tuple[str, int]:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr(client, "_handshake", _handshake)
    task = asyncio.create_task(client._runner())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert task.cancelled()
    assert client._ws_health_tracker().status == "stopped"


@pytest.mark.asyncio
async def test_ducaheat_runner_propagates_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling the Ducaheat runner leaves the task cancelled, not completed."""

    client = _make_client(monkeypatch)
    started = asyncio.Event()

    async def _connect_once() -> None:
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(client, "_connect_once", _connect_once)
    task = asyncio.create_task(client._runner())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert task.cancelled()
    assert client._ws_health_tracker().status == "stopped"


# ---------------------------------------------------------------------------
# ws_connect timeouts (B8)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_termoweb_ws_connect_uses_ws_timeout_and_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TermoWeb passes ClientWSTimeout and bounds the upgrade with a timeout."""

    client = _termoweb_client(
        api_client=DummyREST(authed_headers={"Authorization": "Bearer t"})
    )
    calls: list[dict[str, Any]] = []

    async def _hang(url: str, **kwargs: Any) -> Any:
        calls.append(kwargs)
        await asyncio.Event().wait()

    client._session.ws_connect = _hang
    monkeypatch.setattr(termoweb_ws, "_WS_CONNECT_TIMEOUT", 0.01)

    with pytest.raises(TimeoutError):
        await client._connect_ws("sid")

    assert isinstance(calls[0]["timeout"], aiohttp.ClientWSTimeout)
    assert calls[0]["timeout"].ws_close == termoweb_ws._WS_CLOSE_TIMEOUT


@pytest.mark.asyncio
async def test_ducaheat_ws_connect_uses_ws_timeout_and_bound(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Ducaheat passes ClientWSTimeout, bounds the upgrade, and keeps sid at DEBUG."""

    client = _make_client(monkeypatch, ws=StubWebSocket())
    calls: list[dict[str, Any]] = []

    async def _hang(url: str, **kwargs: Any) -> Any:
        calls.append(kwargs)
        await asyncio.Event().wait()

    monkeypatch.setattr(client._session, "ws_connect", _hang)
    monkeypatch.setattr(ducaheat_ws, "_WS_CONNECT_TIMEOUT", 0.01)
    monkeypatch.setattr(
        ducaheat_ws,
        "_decode_polling_packets",
        lambda _body: ['0{"sid":"abc","pingInterval":25000}'],
    )

    with (
        caplog.at_level(logging.INFO, logger=ducaheat_ws.__name__),
        pytest.raises(TimeoutError),
    ):
        await client._connect_once()

    assert isinstance(calls[0]["timeout"], aiohttp.ClientWSTimeout)
    assert not [r for r in caplog.records if "sid=" in r.getMessage()]


# ---------------------------------------------------------------------------
# Token parsing and payload immutability (F7, F8)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("header", ["Bearer", "Bearer ", "", None])
async def test_get_token_rejects_malformed_header(header: str | None) -> None:
    """A header without a token raises RuntimeError, never IndexError."""

    headers = {} if header is None else {"Authorization": header}
    api = DummyREST()
    api._headers = headers
    tw = _termoweb_client(api_client=api)
    hass = SimpleNamespace(loop=asyncio.get_running_loop(), data={})
    build_entry_runtime(hass=hass, entry_id="entry", dev_id="device")
    dh = ducaheat_ws.DucaheatWSClient(
        hass,
        entry_id="entry",
        dev_id="device",
        api_client=api,
        coordinator=SimpleNamespace(),
        session=SimpleNamespace(),
    )

    with pytest.raises(RuntimeError):
        await tw._get_token()
    with pytest.raises(RuntimeError):
        await dh._get_token()


@pytest.mark.asyncio
async def test_get_token_returns_bearer_value() -> None:
    """A well-formed header yields the bare token."""

    tw = _termoweb_client(
        api_client=DummyREST(authed_headers={"Authorization": "Bearer abc"})
    )

    assert await tw._get_token() == "abc"


def test_extract_nodes_does_not_mutate_payload() -> None:
    """List-shaped node payloads are translated without rewriting the frame."""

    client = _termoweb_client()
    nodes = [{"type": "htr", "addr": "1", "settings": {"stemp": "20"}}]
    payload = {"nodes": nodes}

    mapped = client._extract_nodes(payload)

    assert mapped == {"htr": {"settings": {"1": {"stemp": "20"}}}}
    assert payload["nodes"] is nodes
