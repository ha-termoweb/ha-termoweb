"""Tests for the Ducaheat websocket client (backend/ducaheat_ws.py)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
import gzip
import json
import logging
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qsl, urlsplit

import aiohttp
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
import pytest

from custom_components.termoweb.backend import create_backend, ducaheat_ws, ws_client
from custom_components.termoweb.backend.ducaheat_ws import DucaheatWSClient
from custom_components.termoweb.backend.rest_client import RESTClient
from custom_components.termoweb.backend.ws_client import HandshakeError
from custom_components.termoweb.const import (
    BRAND_DUCAHEAT,
    get_brand_api_base,
    signal_ws_status,
)
from custom_components.termoweb.coordinator import StateCoordinator
from custom_components.termoweb.domain import state_to_dict
from custom_components.termoweb.inventory import Inventory, build_node_inventory
from custom_components.termoweb.runtime import EntryRuntime
from tests.fakes.rest import LatchedResponse
from tests.fakes.runtime import build_entry_runtime
from tests.fakes.termoweb_ws import (
    FakeWebSocket,
    HandshakeResponse,
    SleepController,
    WSFakeSession,
    token_response,
    until,
)
from tests.fakes.ws import (
    DEV_ID,
    ENTRY_ID,
    DummyREST,
    QueueWebSocket,
    StubSession,
    StubWebSocket,
    make_inventory,
    make_runtime,
    polling_body,
)

NS = ducaheat_ws.WS_NAMESPACE


@pytest.fixture(autouse=True)
def _no_upgrade_drain(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the real 1 s post-upgrade drain wait."""
    monkeypatch.setattr(ducaheat_ws, "_UPGRADE_DRAIN_TIMEOUT", 0.0)


@pytest.fixture
def runtime(hass: HomeAssistant) -> EntryRuntime:
    """Return the entry runtime with one heater."""
    return make_runtime(hass)


@pytest.fixture
async def client(hass: HomeAssistant, runtime: EntryRuntime) -> DucaheatWSClient:
    """Return a Ducaheat client on a handshake-capable session; stop it after."""
    ws_client = DucaheatWSClient(
        hass,
        entry_id=ENTRY_ID,
        dev_id=DEV_ID,
        api_client=DummyREST(authed_headers={"Authorization": "Bearer rest-token"}),
        coordinator=runtime.coordinator,
        session=StubSession(StubWebSocket()),
        inventory=runtime.inventory,
    )
    yield ws_client
    await ws_client.stop()


def _statuses(monkeypatch: pytest.MonkeyPatch, client: DucaheatWSClient) -> list[str]:
    """Record every status the client reports (the real method still runs)."""
    statuses: list[str] = []
    real = client._update_status

    def _record(status: str) -> None:
        statuses.append(status)
        real(status)

    monkeypatch.setattr(client, "_update_status", _record)
    return statuses


async def _read(client: DucaheatWSClient, *messages: Any) -> QueueWebSocket:
    """Run the read loop over ``messages`` until the socket closes."""
    ws = QueueWebSocket(messages)
    client._ws = ws
    if client._status == "stopped":
        client._status = "connected"
    with pytest.raises(RuntimeError, match="websocket closed"):
        await client._read_loop_ws()
    return ws


def _event(name: str, payload: Any) -> str:
    """Return a namespaced Socket.IO event frame."""
    return f"42{NS}," + json.dumps([name, payload], separators=(",", ":"))


# ---------------------------------------------------------------------------
# Handshake
# ---------------------------------------------------------------------------


async def test_connect_once_performs_full_handshake(
    client: DucaheatWSClient, runtime: EntryRuntime
) -> None:
    """Polling open, namespace POST, drain and upgrade end in ``connected``."""
    await client._connect_once()

    session = client._session
    assert [method for method, _ in session.calls] == ["GET", "POST", "GET", "WS"]
    assert all("token=rest-token" in url for _, url in session.calls)
    ws = session.ws
    assert ws.sent == ["2probe", "5", f"40{NS}"]  # no pong, no dev_data yet
    assert client._pending_dev_data is True
    assert runtime.ws_state[DEV_ID]["status"] == "connected"
    assert client._ping_interval == pytest.approx(25.0)

    await client._disconnect("test")
    assert ws.closed
    assert ws.close_args == (aiohttp.WSCloseCode.GOING_AWAY, b"test")
    assert client._ws is None
    assert client._pending_dev_data is False
    assert client._keepalive_task is None


@pytest.mark.parametrize(
    ("frames", "sent"),
    [
        (["2", "3probe"], ["2probe", "3", "5", f"40{NS}"]),  # ping during probe
        (["weird", "3probe"], ["2probe", "5", f"40{NS}"]),  # unknown frame skipped
    ],
)
async def test_probe_tolerates_interleaved_frames(
    client: DucaheatWSClient, frames: list[str], sent: list[str]
) -> None:
    """Pings are answered and unknown frames ignored until the probe ack."""
    client._session.ws = StubWebSocket(frames)

    await client._connect_once()

    assert client._session.ws.sent == sent


async def test_probe_timeout_raises(
    client: DucaheatWSClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing probe ack fails the handshake."""
    client._session.ws = StubWebSocket(["2"])
    monkeypatch.setattr(ducaheat_ws, "_PROBE_ACK_TIMEOUT", 0.05)

    with pytest.raises(HandshakeError) as err:
        await client._connect_once()
    assert err.value.status == 408


@pytest.mark.parametrize(
    ("expected", "open_body", "status"),
    [
        (403, None, {"open": 403}),
        (590, b"", {}),  # no OPEN packet
        (592, b'0{"pingInterval":1}', {}),  # OPEN without sid
        (500, None, {"post": 500}),
        (504, None, {"drain": 504}),
    ],
)
async def test_handshake_failures_raise_with_status(
    client: DucaheatWSClient,
    expected: int,
    open_body: bytes | None,
    status: dict[str, int],
) -> None:
    """Every handshake step failure surfaces as ``HandshakeError`` with a status."""
    session = client._session
    if open_body is not None:
        session.open_body = polling_body(open_body.decode()) if open_body else b""
    session.status.update(status)

    with pytest.raises(HandshakeError) as err:
        await client._connect_once()
    assert err.value.status == expected


async def test_handshake_url_targets_the_ducaheat_host(
    client: DucaheatWSClient,
) -> None:
    """Handshake URLs use the Tevolve host, the Engine.IO path and the query."""
    params = {
        "token": "fixed-token",
        "dev_id": "device-456",
        "EIO": "3",
        "transport": "polling",
        "t": "PABCDEFG",
    }

    parsed = urlsplit(client._build_handshake_url(params))

    assert parsed.scheme == "https"
    assert parsed.netloc == "api-tevolve.termoweb.net"
    assert parsed.path == "/socket.io/"
    assert dict(parse_qsl(parsed.query)) == params


async def test_polling_headers_extend_brand_headers(client: DucaheatWSClient) -> None:
    """Polling headers add browser-like fields to the shared brand set."""
    headers = client._polling_headers()
    assert headers.items() >= client._brand_headers(origin="https://localhost").items()
    assert headers["Referer"] == "https://localhost/"
    assert headers["Connection"] == "keep-alive"


# ---------------------------------------------------------------------------
# Engine.IO polling framing
# ---------------------------------------------------------------------------


def test_rand_t_token_format() -> None:
    """Polling cache-busters are ``P`` plus seven alphanumerics."""
    token = ducaheat_ws._rand_t()
    assert token.startswith("P")
    assert len(token) == 8
    assert token[1:].isalnum()


def test_polling_packets_round_trip_plain_and_gzip() -> None:
    """Encoded packets carry an ASCII length; binary framing decodes, gzip too."""
    assert ducaheat_ws._encode_polling_packet("40/message") == b"10:40/message"

    body = bytes([0, 1, 0, 0xFF]) + b"40/message"
    assert ducaheat_ws._decode_polling_packets(body) == ["40/message"]
    assert ducaheat_ws._decode_polling_packets(gzip.compress(body)) == ["40/message"]


@pytest.mark.parametrize(
    "body",
    [
        b"\x00\x00\x00",  # truncated
        b"\x00\x0a",  # invalid length digit
        b"\x00\xff\x00\x00",  # no length digits
        b"\x00\x02\xff\x00",  # length overruns the body
        b"\x1f\x8bbad",  # corrupt gzip
    ],
)
def test_decode_polling_packets_rejects_malformed_bodies(body: bytes) -> None:
    """Malformed polling bodies decode to nothing instead of raising."""
    assert ducaheat_ws._decode_polling_packets(body) == []


# ---------------------------------------------------------------------------
# Sending
# ---------------------------------------------------------------------------


async def test_emit_sio_frames_namespaced_events(client: DucaheatWSClient) -> None:
    """Events are sent as namespaced Socket.IO ``42`` frames."""
    client._ws = StubWebSocket()

    await client._emit_sio("subscribe", "/htr/1/status")
    await client._emit_sio("sample", {"x": 1})

    assert client._ws.sent == [
        f'42{NS},["subscribe","/htr/1/status"]',
        f'42{NS},["sample",{{"x":1}}]',
    ]


async def test_sends_are_serialised(client: DucaheatWSClient) -> None:
    """Concurrent emits never interleave on the socket."""

    class ConcurrencyWebSocket(StubWebSocket):
        sending = False

        async def send_str(self, payload: str) -> None:
            assert not self.sending, "send_str called concurrently"
            self.sending = True
            await asyncio.sleep(0)
            self.sent.append(payload)
            self.sending = False

    client._ws = ConcurrencyWebSocket()
    await asyncio.gather(
        client._emit_sio("message", "one"), client._emit_sio("message", "two")
    )
    assert len(client._ws.sent) == 2


async def test_send_requires_the_current_open_socket(
    client: DucaheatWSClient,
) -> None:
    """Sending without a socket, or on a stale one, raises."""
    client._ws = None
    with pytest.raises(RuntimeError):
        await client._emit_sio("evt")
    client._ws = StubWebSocket()
    with pytest.raises(RuntimeError):
        await client._send_str("payload", context="stale", ws=StubWebSocket())


# ---------------------------------------------------------------------------
# Read loop
# ---------------------------------------------------------------------------


async def test_engineio_pong_is_a_heartbeat_not_health(
    client: DucaheatWSClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pong refreshes the heartbeat but does not mark the socket healthy."""
    statuses = _statuses(monkeypatch, client)

    await _read(client, "3", "43")

    assert "healthy" not in statuses
    assert client._ws_health.last_heartbeat_at is not None


async def test_dev_data_marks_healthy_and_seeds_the_store(
    client: DucaheatWSClient, runtime: EntryRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dev_data snapshot marks the socket healthy and replaces node state."""
    monkeypatch.setattr(ducaheat_ws.time, "time", lambda: 1_234.0)
    client._ws = QueueWebSocket([])  # subscriptions are emitted on this socket

    ws = await _read(
        client,
        _event("dev_data", {"nodes": {"htr": {"settings": {"1": {"mode": "auto"}}}}}),
    )

    state = runtime.ws_state[DEV_ID]
    assert state["status"] == "healthy"
    assert state["healthy_since"] == 1_234.0
    assert state["last_event_at"] == 1_234.0
    view = runtime.coordinator.domain_view
    assert view.get_heater_state("htr", "1").mode == "auto"
    assert f'42{NS},["subscribe","/htr/1/samples"]' in ws.sent
    assert f'42{NS},["subscribe","/htr/1/status"]' in ws.sent


@pytest.mark.parametrize(
    "dev_data",
    [
        json.dumps({"nodes": {"htr": {"status": {"1": {"power": 5}}}}}),  # string
        {"nodes": [{"type": "htr", "addr": "1", "status": {"power": 7}}]},  # list
    ],
)
async def test_dev_data_variants_trigger_subscription(
    client: DucaheatWSClient, monkeypatch: pytest.MonkeyPatch, dev_data: Any
) -> None:
    """String-wrapped and list-shaped snapshots are decoded and subscribed."""
    subscribe = AsyncMock(return_value=2)
    monkeypatch.setattr(client, "_subscribe_feeds", subscribe)

    await _read(client, _event("dev_data", dev_data))

    subscribe.assert_awaited_once()
    assert "now" in subscribe.await_args.kwargs


async def test_extract_dev_data_payload_walks_wrappers(
    client: DucaheatWSClient,
) -> None:
    """dev_data payloads are found inside data/body wrappers and nested lists."""
    nodes = {"nodes": {"htr": {}}}

    assert client._extract_dev_data_payload([{"data": nodes}]) is nodes
    assert client._extract_dev_data_payload([[nodes]]) is nodes
    assert client._extract_dev_data_payload([({"body": nodes},)]) is nodes
    listed = client._extract_dev_data_payload(
        [[{"nodes": [{"type": "htr", "addr": "1", "status": {"p": 1}}]}]]
    )
    assert listed["nodes"]["htr"]["status"]["1"]["p"] == 1
    assert client._extract_dev_data_payload(["not json"]) is None


@pytest.mark.parametrize(
    ("path", "body", "check"),
    [
        ("settings", {"mode": "eco"}, lambda state: state.mode == "eco"),
        ("status", {"stemp": "19.5"}, lambda state: state.stemp == "19.5"),
    ],
)
async def test_update_event_applies_a_delta(
    client: DucaheatWSClient,
    runtime: EntryRuntime,
    path: str,
    body: dict[str, Any],
    check: Any,
) -> None:
    """Path updates merge into the store and mark the socket healthy."""
    await _read(
        client,
        _event("update", {"body": body, "path": f"/api/v2/devs/device/htr/1/{path}"}),
    )

    assert check(runtime.coordinator.domain_view.get_heater_state("htr", "1"))
    assert runtime.ws_state[DEV_ID]["status"] == "healthy"


async def test_sample_update_is_forwarded_to_energy(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sample path updates reach the energy coordinator, not the state store."""
    handler = MagicMock()
    runtime = make_runtime(
        hass, energy_coordinator=SimpleNamespace(handle_ws_samples=handler)
    )
    client = DucaheatWSClient(
        hass,
        entry_id=ENTRY_ID,
        dev_id=DEV_ID,
        api_client=DummyREST(),
        coordinator=runtime.coordinator,
        session=SimpleNamespace(),
        inventory=runtime.inventory,
    )

    await _read(
        client,
        _event("update", {"body": {"power": 10}, "path": "/devs/device/htr/1/samples"}),
    )

    handler.assert_called_once_with(
        DEV_ID, {"htr": {"1": {"power": 10}}}, lease_seconds=None
    )
    await client.stop()


async def test_mixed_frames_answer_pings_and_messages(
    client: DucaheatWSClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pings get pongs, ``message: ping`` gets ``pong``, junk frames are skipped."""
    emit = AsyncMock()
    monkeypatch.setattr(client, "_emit_sio", emit)
    statuses = _statuses(monkeypatch, client)

    ws = await _read(
        client,
        "1ignore",
        "42",
        '42/api,["ping"]',
        "442/invalid",
        "442invalid",
        '442["message","ping"]',
        '442["other",{}]',
        "440",
        '442["dev_data",{"nodes":{"htr":{"samples":{"1":{}}}}}]',
        '442["update",{"body":{"temp":1},"path":"/path"}]',
    )

    assert ws.sent.count("3/api") == 1
    assert any(call.args == ("message", "pong") for call in emit.await_args_list)
    assert "healthy" in statuses


async def test_engineio_ping_and_namespace_open_request_dev_data(
    client: DucaheatWSClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An Engine.IO ping is answered and the namespace ack asks for dev_data."""
    emit = AsyncMock()
    monkeypatch.setattr(client, "_emit_sio", emit)
    client._pending_dev_data = True

    ws = await _read(
        client,
        "2",
        '42["message","ping"]',
        "40",
        '42{"evt":"ignored"}',
        '42["dev_data",{"nodes":{"htr":{"settings":{"1":{}}}}}]',
        '42["update",{"body":{"path":{}}}]',
    )

    assert "3" in ws.sent
    assert any(call.args == ("dev_data",) for call in emit.await_args_list)


@pytest.mark.parametrize(
    "message",
    [
        SimpleNamespace(type=aiohttp.WSMsgType.ERROR, data=None),
        SimpleNamespace(type=aiohttp.WSMsgType.CLOSE, data=None),
    ],
)
async def test_read_loop_raises_on_error_or_close(
    client: DucaheatWSClient, message: Any
) -> None:
    """Transport errors and close frames end the read loop with an error."""
    client._ws = QueueWebSocket([message])
    client._status = "connected"
    with pytest.raises(RuntimeError):
        await client._read_loop_ws()


async def test_parse_error_does_not_block_update_stream(
    client: DucaheatWSClient, runtime: EntryRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed events are counted and later updates still flow."""
    client._pending_dev_data = False
    monkeypatch.setattr(client, "_maybe_subscribe", AsyncMock(return_value=0))
    monkeypatch.setattr(client, "_emit_sio", AsyncMock())

    await _read(
        client,
        '42["update"',
        '42["update",{"path":"/htr/1/status","body":{"status":{"on":true}}}]',
    )

    state = runtime.ws_state[DEV_ID]
    assert state["parse_errors_total"] == 1
    assert state["last_update_event_at"] is not None
    assert client._stats.events_total == 1


# ---------------------------------------------------------------------------
# Namespace acknowledgements and subscriptions
# ---------------------------------------------------------------------------


async def test_namespace_ack_requests_dev_data_and_replays_subscriptions(
    client: DucaheatWSClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The namespace ack asks for a snapshot and replays known subscriptions."""
    client._pending_dev_data = True
    client._subscription_paths = {"/htr/1/status", "/htr/1/samples"}
    emit = AsyncMock()
    monkeypatch.setattr(client, "_emit_sio", emit)
    statuses = _statuses(monkeypatch, client)

    await _read(client, f"40{NS}")

    assert client._pending_dev_data is False
    assert statuses == ["connected"]  # no payload yet: not healthy
    assert [call.args for call in emit.await_args_list] == [
        ("dev_data",),
        ("subscribe", "/htr/1/samples"),
        ("subscribe", "/htr/1/status"),
    ]


@pytest.mark.parametrize(
    ("prefix", "pending"),
    [(f"40{NS},", True), ("40/wrong,", False)],
)
async def test_namespace_frame_with_embedded_event(
    client: DucaheatWSClient,
    monkeypatch: pytest.MonkeyPatch,
    prefix: str,
    pending: bool,
) -> None:
    """Events glued to a namespace frame are processed, whatever the namespace."""
    client._pending_dev_data = pending
    monkeypatch.setattr(client, "_emit_sio", AsyncMock())
    monkeypatch.setattr(client, "_replay_subscription_paths", AsyncMock())
    subscribe = AsyncMock(return_value=2)
    monkeypatch.setattr(client, "_subscribe_feeds", subscribe)
    statuses = _statuses(monkeypatch, client)

    await _read(
        client,
        prefix + f'42{NS},["dev_data",{{"nodes":{{"htr":{{"status":{{"1":{{}}}}}}}}}}]',
    )

    subscribe.assert_awaited_once()
    assert statuses[-1] == "healthy"


async def test_namespace_ack_for_another_namespace_is_ignored(
    client: DucaheatWSClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare ack for another namespace changes nothing."""
    client._pending_dev_data = True
    emit = AsyncMock()
    monkeypatch.setattr(client, "_emit_sio", emit)
    statuses = _statuses(monkeypatch, client)

    await _read(client, "40/other")

    assert client._pending_dev_data is True
    emit.assert_not_awaited()
    assert statuses == []


@pytest.mark.parametrize(
    ("nodes", "paths"),
    [
        ([{"type": "htr", "addr": "7"}], ["/htr/7/samples", "/htr/7/status"]),
        ([], []),
    ],
)
async def test_subscribe_feeds_follows_the_inventory(
    client: DucaheatWSClient,
    monkeypatch: pytest.MonkeyPatch,
    nodes: list[dict[str, str]],
    paths: list[str],
) -> None:
    """Each inventory heater gets samples and status subscriptions."""
    client._inventory = make_inventory(nodes)
    emit = AsyncMock()
    monkeypatch.setattr(client, "_emit_sio", emit)

    assert await client._subscribe_feeds() == len(paths)

    assert [call.args for call in emit.await_args_list] == [
        ("subscribe", path) for path in paths
    ]
    assert client._subscription_paths == set(paths)


async def test_subscribe_telemetry_tracks_success_and_failure(
    client: DucaheatWSClient, runtime: EntryRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Subscription attempts count successes and failures in the ws state."""
    client._ws = StubWebSocket()
    emit = AsyncMock(side_effect=[RuntimeError("send failed"), None, None])
    monkeypatch.setattr(client, "_emit_sio", emit)

    assert await client._subscribe_feeds(now=10.0) == 0

    state = runtime.ws_state[DEV_ID]
    assert state["subscribe_attempts_total"] == 1
    assert state["subscribe_fail_total"] == 1
    assert state["subscribe_success_total"] == 0
    assert client._pending_subscribe is True

    assert await client._subscribe_feeds(now=200.0) == 2

    assert state["subscribe_attempts_total"] == 2
    assert state["subscribe_fail_total"] == 1
    assert state["subscribe_success_total"] == 1
    assert state["last_subscribe_success_at"] == 200.0
    assert client._pending_subscribe is False


async def test_failed_subscription_lengthens_backoff(
    client: DucaheatWSClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A subscription attempt that subscribes nothing backs off further."""
    client._ws = StubWebSocket()
    client._status = "connected"
    client._pending_subscribe = True
    client._last_subscribe_attempt_ts = 10.0
    subscribe = AsyncMock(return_value=0)
    monkeypatch.setattr(client, "_subscribe_feeds", subscribe)

    assert await client._maybe_subscribe(20.0) == 0

    assert client._subscribe_backoff_s > ducaheat_ws._SUBSCRIBE_BACKOFF_INITIAL
    assert subscribe.await_args.kwargs["now"] == 20.0


# ---------------------------------------------------------------------------
# Keepalive
# ---------------------------------------------------------------------------


async def test_keepalive_sends_pings_at_the_negotiated_cadence(
    client: DucaheatWSClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The keepalive pings a little before the server's ping interval."""
    real_sleep = asyncio.sleep
    sleeps: list[float] = []

    async def fast_sleep(delay: float) -> None:
        sleeps.append(delay)
        await real_sleep(0)

    client._session.open_body = polling_body(
        '0{"sid":"abc","pingInterval":200,"pingTimeout":600}'
    )
    monkeypatch.setattr(ducaheat_ws.asyncio, "sleep", fast_sleep)

    await client._connect_once()
    task = client._keepalive_task
    client._start_keepalive()  # already running: no second loop
    assert client._keepalive_task is task
    await real_sleep(0)
    await real_sleep(0)

    assert "2" in client._ws.sent
    assert sleeps[0] == pytest.approx(0.18, rel=0.05)

    await client._disconnect("test")
    assert client._keepalive_task is None
    client._start_keepalive()  # no socket: nothing starts
    assert client._keepalive_task is None


async def test_keepalive_stops_when_the_socket_is_swapped_or_fails(
    client: DucaheatWSClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replaced socket or a failing send ends the keepalive loop."""
    real_sleep = asyncio.sleep

    class FailingWS(StubWebSocket):
        fail = False

        async def send_str(self, payload: str) -> None:
            if self.fail:
                raise RuntimeError("boom")
            await super().send_str(payload)

    first, second = FailingWS(), FailingWS()
    client._ws = first
    client._ping_interval = 0.2
    sleeps = 0

    async def fast_sleep(delay: float) -> None:
        nonlocal sleeps
        sleeps += 1
        if sleeps == 2:
            client._ws = second
            second.fail = True
        await real_sleep(0)

    monkeypatch.setattr(ducaheat_ws.asyncio, "sleep", fast_sleep)

    client._keepalive_task = asyncio.create_task(client._keepalive_loop())
    await client._keepalive_task

    assert first.sent.count("2") == 1
    assert second.sent == []
    assert client._keepalive_task is None


# ---------------------------------------------------------------------------
# Lease refresh and idle recovery
# ---------------------------------------------------------------------------


def _healthy(client: DucaheatWSClient, *, last_payload_at: float) -> None:
    """Put the client in a healthy, subscribed state with a payload timestamp."""
    client._ws = StubWebSocket()
    client._status = "healthy"
    client._pending_dev_data = False
    tracker = client._ws_health
    tracker.last_payload_at = last_payload_at


async def test_soft_refresh_renews_leases_before_the_payload_window(
    client: DucaheatWSClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Close to the payload window the client re-requests dev_data and replays."""
    _healthy(client, last_payload_at=1_000.0)
    client._subscription_paths = {"/htr/1/status"}
    client._payload_window_hint = 10.0
    client._ws_health.payload_stale_after = 20.0
    emit, replay = AsyncMock(), AsyncMock()
    monkeypatch.setattr(client, "_emit_sio", emit)
    monkeypatch.setattr(client, "_replay_subscription_paths", replay)

    assert await client._maybe_refresh_leases(now=1_009.0) is True
    emit.assert_awaited_once_with("dev_data")
    replay.assert_awaited_once()


async def test_soft_refresh_waits_for_the_lease_threshold(
    client: DucaheatWSClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Too early in the window, nothing is sent."""
    _healthy(client, last_payload_at=500.0)
    client._payload_window_hint = 20.0
    emit = AsyncMock()
    monkeypatch.setattr(client, "_emit_sio", emit)

    assert await client._maybe_refresh_leases(now=510.0) is False
    emit.assert_not_awaited()


async def test_soft_refresh_reports_a_failed_emit(
    client: DucaheatWSClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing dev_data request is not counted as a refresh."""
    _healthy(client, last_payload_at=800.0)
    client._payload_window_hint = 12.0
    emit = AsyncMock(side_effect=RuntimeError("boom"))
    replay = AsyncMock()
    monkeypatch.setattr(client, "_emit_sio", emit)
    monkeypatch.setattr(client, "_replay_subscription_paths", replay)

    assert await client._maybe_refresh_leases(now=812.0) is False
    emit.assert_awaited_once_with("dev_data")
    replay.assert_not_awaited()


async def test_soft_refresh_ignores_an_unusable_window_hint(
    client: DucaheatWSClient,
) -> None:
    """A non-numeric window hint never triggers a refresh."""
    client._payload_window_hint = object()
    client._ws_health.last_payload_at = 100.0
    assert await client._maybe_refresh_leases(now=150.0) is False


async def test_idle_check_triggers_soft_recovery(
    client: DucaheatWSClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An idle socket re-requests a snapshot, replays and resubscribes."""
    client._ws = StubWebSocket()
    client._status = "healthy"
    client._pending_dev_data = False
    client._last_event_at = 0.0
    emit, replay, subscribe = AsyncMock(), AsyncMock(), AsyncMock()
    monkeypatch.setattr(client, "_emit_sio", emit)
    monkeypatch.setattr(client, "_replay_subscription_paths", replay)
    monkeypatch.setattr(client, "_maybe_subscribe", subscribe)

    assert await client._handle_idle_check(400.0) is False

    emit.assert_awaited_once_with("dev_data")
    replay.assert_awaited_once()
    subscribe.assert_awaited_once_with(400.0)
    assert client._idle_recovery_attempts == 1


async def test_repeated_idle_recovery_escalates_to_reconnect(
    client: DucaheatWSClient, runtime: EntryRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recovery attempts are counted; the third one reconnects."""
    _healthy(client, last_payload_at=0.0)
    monkeypatch.setattr(client, "_emit_sio", AsyncMock())
    monkeypatch.setattr(client, "_replay_subscription_paths", AsyncMock())
    monkeypatch.setattr(client, "_maybe_subscribe", AsyncMock())
    disconnect = AsyncMock(side_effect=lambda reason: setattr(client, "_ws", None))
    monkeypatch.setattr(client, "_disconnect", disconnect)

    for idx in range(3):
        now = 1_000.0 + idx * 50.0
        await client._recover_from_idle(
            now=now, idle_for=now - 100.0, payload_stale=True
        )
        if idx == 0:
            state = runtime.ws_state[DEV_ID]
            assert state["recovery_attempts_total"] == 1
            assert state["last_recovery_at"] == 1_000.0

    disconnect.assert_awaited_once()
    assert client._idle_recovery_attempts >= 3


async def test_received_frame_resets_idle_recovery(client: DucaheatWSClient) -> None:
    """Any frame ends an idle episode."""
    client._idle_timeout_flag = True
    client._idle_recovery_attempts = 2
    client._idle_recovery_window_start = 10.0

    client._record_frame(timestamp=12.0)

    assert client._idle_timeout_flag is False
    assert client._idle_recovery_attempts == 0
    assert client._idle_recovery_window_start == 0.0
    assert client._last_idle_recovery_at == 12.0


async def test_idle_monitor_task_lifecycle(
    client: DucaheatWSClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The idle monitor task is cancelled and cleared on stop."""
    started = asyncio.Event()

    async def _monitor() -> None:
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(client, "_idle_monitor", _monitor)
    client._ws = StubWebSocket()
    client._start_idle_monitor()
    await asyncio.wait_for(started.wait(), 1.0)
    task = client._idle_monitor_task

    await client._stop_idle_monitor()
    assert task.cancelled()
    assert client._idle_monitor_task is None


# ---------------------------------------------------------------------------
# Payload window (cadence hints)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value", [None, -1, 0, "-5", float("inf"), float("-inf"), float("nan"), object()]
)
async def test_normalise_cadence_value_rejects_invalid(
    client: DucaheatWSClient, value: Any
) -> None:
    """Non-positive, non-finite and non-numeric cadences are rejected."""
    assert client._normalise_cadence_value(value) is None


async def test_cadence_values_and_candidates(client: DucaheatWSClient) -> None:
    """Positive cadences are accepted and collected once per (cyclic) mapping."""
    assert client._normalise_cadence_value("30") == pytest.approx(30.0)
    assert client._normalise_cadence_value(12.5) == pytest.approx(12.5)

    shared: dict[str, object] = {"poll_seconds": "15"}
    shared["self"] = shared
    payload: dict[str, object] = {
        "lease_seconds": "5",
        "nested": {"cadence_seconds": "10"},
        "ignored": {"poll_seconds": "nan", "cadence_seconds": "-1"},
        "loop1": shared,
        "loop2": shared,
        "unrelated": {"cadence": "25"},
    }
    payload["nested"]["again"] = payload

    assert sorted(client._extract_cadence_candidates(payload)) == [5.0, 10.0, 15.0]


async def test_sample_lease_tunes_the_payload_window(
    client: DucaheatWSClient, runtime: EntryRuntime
) -> None:
    """A sample lease sets the window to lease plus margin and records it."""
    assert client._payload_stale_after == pytest.approx(240.0)

    result = client._collect_sample_updates(
        {"htr": {"samples": {"1": {"power": 5}}, "lease_seconds": 60}}
    )

    assert result["htr"]["lease_seconds"] == 60
    assert client._payload_window_hint == pytest.approx(60.0)
    assert client._payload_stale_after == pytest.approx(75.0)
    state = runtime.ws_state[DEV_ID]
    assert state["payload_stale_after"] == pytest.approx(75.0)
    assert state["payload_window_source"] == "sample_updates"
    assert client._ws_health.payload_stale_after == pytest.approx(75.0)


async def test_payload_window_hint_ignores_invalid_candidates(
    client: DucaheatWSClient,
) -> None:
    """Unusable cadence candidates leave the window untouched."""
    window, hint = client._payload_stale_after, client._payload_window_hint

    client._apply_payload_window_hint(
        source="cadence", lease_seconds=None, candidates=[None, "NaN", -5]
    )

    assert client._payload_stale_after == window
    assert client._payload_window_hint == hint


@pytest.mark.parametrize(
    ("lease", "window"),
    [(1, ducaheat_ws._PAYLOAD_WINDOW_MIN), (10_000, ducaheat_ws._PAYLOAD_WINDOW_MAX)],
)
async def test_payload_window_is_clamped(
    client: DucaheatWSClient, lease: int, window: float
) -> None:
    """Extreme leases clamp the window to the configured bounds."""
    client._apply_payload_window_hint(source="test", lease_seconds=lease)

    assert client._payload_stale_after == pytest.approx(window)
    assert client._payload_window_hint == pytest.approx(float(lease))
    assert client._ws_health.payload_stale_after == pytest.approx(window)


async def test_disconnect_restores_the_default_payload_window(
    client: DucaheatWSClient, runtime: EntryRuntime
) -> None:
    """Disconnecting forgets the hinted window."""
    client._apply_payload_window_hint(source="test", lease_seconds=30)
    assert client._payload_stale_after == pytest.approx(45.0)

    await client._disconnect("testing")

    assert client._payload_stale_after == pytest.approx(240.0)
    assert client._ws_health.payload_stale_after == pytest.approx(240.0)
    state = runtime.ws_state[DEV_ID]
    assert state["payload_window_hint"] is None
    assert state["payload_window_source"] == "disconnect"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def test_runner_throttles_before_dialing(
    client: DucaheatWSClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every attempt waits for a connection slot; failures report disconnected."""
    limiter = SimpleNamespace(wait_for_slot=AsyncMock())
    client._connect_limiter = limiter
    connect = AsyncMock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr(client, "_connect_once", connect)
    statuses = _statuses(monkeypatch, client)

    def _stop_at_first_backoff() -> float:
        raise asyncio.CancelledError

    monkeypatch.setattr(client, "_next_backoff", _stop_at_first_backoff)

    with pytest.raises(asyncio.CancelledError):
        await client._runner()

    limiter.wait_for_slot.assert_awaited_once()
    connect.assert_awaited_once()
    assert statuses == ["starting", "disconnected", "stopped"]


async def test_runner_resets_backoff_after_a_session_with_payloads(
    client: DucaheatWSClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A session that delivered payloads restarts the reconnect backoff."""
    client._backoff_idx = 3
    monkeypatch.setattr(client, "_connect_once", AsyncMock())

    async def _read_with_payload() -> None:
        client._mark_ws_payload(timestamp=ducaheat_ws.time.time() + 1)
        raise RuntimeError("websocket closed")

    def _stop_at_first_backoff() -> float:
        raise asyncio.CancelledError

    monkeypatch.setattr(client, "_read_loop_ws", _read_with_payload)
    monkeypatch.setattr(client, "_next_backoff", _stop_at_first_backoff)

    with pytest.raises(asyncio.CancelledError):
        await client._runner()

    assert client._backoff_idx == 0


async def test_idle_monitor_exits_when_the_check_says_so(
    client: DucaheatWSClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The monitor loop ends (and forgets its task) once a check returns True."""
    client._ws = StubWebSocket()
    check = AsyncMock(return_value=True)
    monkeypatch.setattr(client, "_handle_idle_check", check)
    monkeypatch.setattr(client, "_idle_monitor_interval", lambda: 0.0)

    await client._idle_monitor()

    check.assert_awaited_once()
    assert client._idle_monitor_task is None


LIFECYCLE_DEV_ID = "fedcba9876543210"  # synthetic gateway id
LIFECYCLE_ENTRY_ID = "entry-ducaheat-ws"
TOKEN = "tok-fedcba9876543210"
NODES = {"nodes": [{"type": "htr", "addr": "1", "name": "Living room"}]}
SID = "Sx1"


def polling(packet: str) -> bytes:
    """Encode one Engine.IO v3 binary polling packet."""
    data = packet.encode()
    return bytes([0]) + bytes(int(d) for d in str(len(data))) + b"\xff" + data


def update(stemp: str) -> str:
    """Return a Socket.IO status update frame for heater 1."""
    body = {"path": "/htr/1/status", "body": {"stemp": stemp}}
    return f"42{NS},{json.dumps(['update', body])}"


class Clock:
    """Wall clock for the websocket module: real time plus a test offset."""

    def __init__(self) -> None:
        """Start without an offset."""
        self.offset = 0.0

    def time(self) -> float:
        """Return the shifted wall-clock time."""
        return time.time() + self.offset


@dataclass
class Harness:
    """Everything a Ducaheat websocket test touches."""

    session: WSFakeSession
    coordinator: StateCoordinator
    runtime: EntryRuntime
    client: ducaheat_ws.DucaheatWSClient
    sleeps: SleepController
    clock: Clock
    statuses: list[dict[str, Any]] = field(default_factory=list)

    def start(self) -> asyncio.Task[None]:
        """Start the client and let the sleep fake recognise its runner."""
        task = self.client.start()
        self.sleeps.runner = task
        return task

    async def connected(self, count: int = 1) -> FakeWebSocket:
        """Wait until socket ``count`` opened the namespace and asked for data."""
        await until(lambda: len(self.session.sockets) >= count)
        ws = self.session.sockets[count - 1]
        await until(lambda: any('"dev_data"' in frame for frame in ws.sent))
        return ws

    def frames_seen(self) -> int:
        """Return the frame counter exposed in the diagnostics bucket."""
        return self.runtime.ws_state[LIFECYCLE_DEV_ID].get("frames_total", 0)

    def stemp(self) -> str | None:
        """Return heater 1's target temperature from the domain state."""
        state = self.coordinator.domain_view.get_heater_state("htr", "1")
        return state_to_dict(state).get("stemp") if state is not None else None

    def status_dispatches(self) -> list[str]:
        """Return the statuses dispatched for status transitions."""
        return [p["status"] for p in self.statuses if p["reason"] == "status"]


@pytest.fixture
async def dh(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> AsyncGenerator[Harness]:
    """Return a Ducaheat websocket harness; the client is stopped afterwards."""
    sleeps = SleepController()
    sleeps.install(monkeypatch, ducaheat_ws, limiter_module=ws_client)
    clock = Clock()
    monkeypatch.setattr(
        ducaheat_ws,
        "time",
        SimpleNamespace(time=clock.time, monotonic=time.monotonic),
    )
    # The server's post-upgrade noop ("6") is already queued, so the drain
    # discards it without waiting.
    monkeypatch.setattr(ducaheat_ws, "_UPGRADE_DRAIN_TIMEOUT", 0.0)

    session = WSFakeSession()
    session.queue_post(LatchedResponse(token_response(TOKEN)))
    session.default_get = lambda url: HandshakeResponse(
        200,
        b"ok"
        if "sid=" in url
        else polling(
            "0" + json.dumps({"sid": SID, "pingInterval": 25000, "pingTimeout": 60000})
        ),
    )
    session.greeting = ("3probe", "6", f"40{NS}")
    rest = RESTClient(
        session,
        "user@example.com",
        "secret",
        api_base=get_brand_api_base(BRAND_DUCAHEAT),
    )
    inventory = Inventory(LIFECYCLE_DEV_ID, build_node_inventory(NODES))
    coordinator = StateCoordinator(
        hass,
        rest,
        30,
        LIFECYCLE_DEV_ID,
        {"dev_id": LIFECYCLE_DEV_ID, "name": "Home"},
        inventory,
    )
    runtime = build_entry_runtime(
        hass=hass,
        entry_id=LIFECYCLE_ENTRY_ID,
        dev_id=LIFECYCLE_DEV_ID,
        inventory=inventory,
        coordinator=coordinator,
        client=rest,
        brand=BRAND_DUCAHEAT,
    )
    backend = create_backend(brand=BRAND_DUCAHEAT, client=rest)
    client = backend.create_ws_client(
        hass, LIFECYCLE_ENTRY_ID, LIFECYCLE_DEV_ID, coordinator, inventory=inventory
    )
    assert type(client) is ducaheat_ws.DucaheatWSClient
    harness = Harness(session, coordinator, runtime, client, sleeps, clock)

    @callback
    def _on_status(payload: dict[str, Any]) -> None:
        harness.statuses.append(payload)

    unsub = async_dispatcher_connect(
        hass, signal_ws_status(LIFECYCLE_ENTRY_ID), _on_status
    )
    yield harness
    await client.stop()
    unsub()


async def healthy(h: Harness) -> FakeWebSocket:
    """Start the client and wait for a session that delivered data."""
    h.start()
    ws = await h.connected()
    ws.feed(update("20.0"))
    await until(lambda: h.runtime.ws_trackers[LIFECYCLE_DEV_ID].status == "healthy")
    return ws


# ---------------------------------------------------------------------------
# Status notifications only on change (B4/B5)
# ---------------------------------------------------------------------------


async def test_pongs_while_stale_do_not_flap(dh: Harness) -> None:
    """Stale payloads plus periodic pongs: one healthy->connected, no flapping."""
    ws = await healthy(dh)
    assert dh.status_dispatches()[-1] == "healthy"
    before = dh.status_dispatches()

    dh.clock.offset = 1000.0  # far past the 240 s payload window
    frames = dh.frames_seen()
    for _ in range(6):
        ws.feed("3")
    await until(lambda: dh.frames_seen() >= frames + 6)

    assert dh.runtime.ws_trackers[LIFECYCLE_DEV_ID].status == "connected"
    assert dh.status_dispatches() == [*before, "connected"]


async def test_fresh_frames_do_not_redispatch_status(dh: Harness) -> None:
    """While healthy and fresh, pongs and updates do not re-dispatch status."""
    ws = await healthy(dh)
    before = dh.status_dispatches()

    for stemp in ("20.5", "21.0"):
        ws.feed("3")
        ws.feed(update(stemp))
    await until(lambda: dh.stemp() == "21.0")

    assert dh.runtime.ws_trackers[LIFECYCLE_DEV_ID].status == "healthy"
    assert dh.status_dispatches() == before


# ---------------------------------------------------------------------------
# Reconnect: backoff, diagnostics counters, parse errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("with_payload", "expected"),
    [(True, [5.0, 5.0, 5.0, 5.0]), (False, [5.0, 10.0, 30.0, 120.0])],
)
async def test_backoff_restarts_after_a_session_with_payloads(
    dh: Harness, with_payload: bool, expected: list[float]
) -> None:
    """Backoff restarts after a session that delivered data; else it escalates."""
    seen: list[str] = []
    dh.sleeps.on_backoff = lambda _delay: seen.append(
        dh.runtime.ws_trackers[LIFECYCLE_DEV_ID].status
    )
    dh.start()
    for count in range(1, 6):
        ws = await dh.connected(count)
        if with_payload:
            ws.feed(update("20.0"))
            await until(
                lambda: dh.runtime.ws_trackers[LIFECYCLE_DEV_ID].status == "healthy"
            )
        ws.end()

    await until(lambda: len(dh.sleeps.backoffs) >= 4)
    assert dh.sleeps.backoffs[:4] == expected
    # The gateway is reported down before the client waits to reconnect.
    assert set(seen) == {"disconnected"}


async def test_counters_survive_reconnect_and_are_cleared_on_stop(
    dh: Harness,
) -> None:
    """Diagnostics counters and the tracker persist across sessions until stop."""
    ws = await healthy(dh)
    tracker = dh.runtime.ws_trackers[LIFECYCLE_DEV_ID]
    assert tracker.payload_stale_after == ducaheat_ws._PAYLOAD_WINDOW_DEFAULT  # noqa: SLF001
    ws.feed("42{bad")
    await until(
        lambda: dh.runtime.ws_state[LIFECYCLE_DEV_ID]["parse_errors_total"] == 1
    )
    attempts = dh.runtime.ws_state[LIFECYCLE_DEV_ID]["subscribe_attempts_total"]
    assert attempts >= 1

    ws.end()
    await dh.connected(2)

    state = dh.runtime.ws_state[LIFECYCLE_DEV_ID]
    assert state["parse_errors_total"] == 1
    assert state["subscribe_attempts_total"] >= attempts
    assert dh.runtime.ws_trackers[LIFECYCLE_DEV_ID] is tracker

    await dh.client.stop()

    assert LIFECYCLE_DEV_ID not in dh.runtime.ws_state
    assert LIFECYCLE_DEV_ID not in dh.runtime.ws_trackers


async def test_repeated_parse_errors_reconnect_once(dh: Harness) -> None:
    """Three bad frames end the session with a single disconnect, then reconnect."""
    ws = await healthy(dh)

    for frame in ("42{bad", "42{bad", "42[]"):
        ws.feed(frame)

    await dh.connected(2)
    assert len(ws.close_calls) == 1
    assert dh.sleeps.backoffs == [5.0]


# ---------------------------------------------------------------------------
# Handler failures, cancellation, bounded upgrade
# ---------------------------------------------------------------------------


async def test_delta_handler_error_is_logged_and_reading_continues(
    dh: Harness, caplog: pytest.LogCaptureFixture
) -> None:
    """A failing state store update is an ERROR; later frames still apply."""
    ws = await healthy(dh)
    real_handle = dh.coordinator.handle_ws_deltas
    calls = 0

    def _flaky(*args: Any, **kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("boom")
        real_handle(*args, **kwargs)

    caplog.set_level(logging.DEBUG, logger=ducaheat_ws.__name__)
    with patch.object(dh.coordinator, "handle_ws_deltas", _flaky):
        ws.feed(update("17.0"))
        ws.feed(update("19.0"))
        await until(lambda: dh.stemp() == "19.0")

    errors = [
        r for r in caplog.records if "failed to apply websocket deltas" in r.message
    ]
    assert [r.levelno for r in errors] == [logging.ERROR]
    assert len(dh.session.sockets) == 1


async def test_cancelling_the_runner_propagates(dh: Harness) -> None:
    """Cancelling the client task leaves it cancelled and reports stopped."""
    started = asyncio.Event()

    class _HangingOpen(HandshakeResponse):
        async def read(self) -> bytes:
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    dh.session.handshakes.append(_HangingOpen())
    task = dh.start()
    await started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert task.cancelled()
    assert dh.runtime.ws_trackers[LIFECYCLE_DEV_ID].status == "stopped"


async def test_websocket_upgrade_is_bounded_and_sid_stays_out_of_info_logs(
    dh: Harness, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A hanging upgrade times out and is retried; the sid is never logged at INFO."""
    monkeypatch.setattr(ws_client, "_WS_CONNECT_TIMEOUT", 0.01)

    async def _hang(url: str, **kwargs: Any) -> FakeWebSocket:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    dh.session.connect = _hang
    caplog.set_level(logging.INFO, logger=ducaheat_ws.__name__)
    dh.start()

    await until(lambda: len(dh.session.connect_calls) >= 2)
    _, kwargs = dh.session.connect_calls[0]
    assert isinstance(kwargs["timeout"], aiohttp.ClientWSTimeout)
    assert dh.sleeps.backoffs[:1] == [5.0]
    assert SID not in caplog.text
    assert TOKEN not in caplog.text
