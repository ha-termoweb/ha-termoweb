"""Tests for the Ducaheat Engine.IO v3 / Socket.IO websocket client.

The fakes sit at the transport boundary only: an ``aiohttp`` session that
answers the polling handshake and a websocket that replays queued frames.
The client runs against a real Home Assistant, a real ``StateCoordinator``
and the entry runtime it would use in production.
"""

from __future__ import annotations

import asyncio
import gzip
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import parse_qsl, urlsplit

import aiohttp
from homeassistant.core import HomeAssistant
import pytest

from custom_components.termoweb.backend import ducaheat_ws
from custom_components.termoweb.backend.ducaheat_ws import DucaheatWSClient
from custom_components.termoweb.backend.ws_client import HandshakeError
from custom_components.termoweb.runtime import EntryRuntime
from tests_ha.fakes.ws import (
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
    client._inventory = None
    emit = AsyncMock()
    monkeypatch.setattr(client, "_emit_sio", emit)

    with pytest.raises(TypeError):
        await client._subscribe_feeds(now=10.0)

    state = runtime.ws_state[DEV_ID]
    assert state["subscribe_attempts_total"] == 1
    assert state["subscribe_fail_total"] == 1
    assert state["subscribe_success_total"] == 0
    assert client._pending_subscribe is False
    emit.assert_not_awaited()

    client._inventory = runtime.inventory
    client._pending_subscribe = True
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
