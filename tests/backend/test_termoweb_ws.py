"""TermoWeb Socket.IO 0.9 websocket client on real Home Assistant.

The production ``TermoWebWSClient`` is built by the real backend factory on a
real ``RESTClient``, ``StateCoordinator`` and ``Inventory``. The only fakes are
the aiohttp session/socket (``tests.fakes.ws_harness``) and the clock
inside the websocket module, so reconnect backoffs pass instantly.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
import json
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import aiohttp
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
import pytest

from custom_components.termoweb.backend import create_backend, termoweb_ws, ws_client
from custom_components.termoweb.backend.rest_client import RESTClient
from custom_components.termoweb.const import (
    API_BASE,
    BRAND_TERMOWEB,
    WS_NAMESPACE,
    get_brand_requested_with,
    get_brand_user_agent,
    signal_ws_status,
)
from custom_components.termoweb.coordinator import StateCoordinator
from custom_components.termoweb.domain import state_to_dict
from custom_components.termoweb.inventory import Inventory, build_node_inventory
from custom_components.termoweb.runtime import EntryRuntime
from tests.fakes.rest import LatchedResponse, MockResponse
from tests.fakes.runtime import build_entry_runtime
from tests.fakes.ws_harness import (
    FakeResponse,
    FakeWS,
    OffsetClock,
    SleepController,
    WSFakeSession,
    token_response,
    until,
)

DEV_ID = "0123456789abcdef"  # synthetic gateway id
ENTRY_ID = "entry-ws"
TOKEN = "tok-0123456789abcdef"
NS = WS_NAMESPACE
NODES = {
    "nodes": [
        {"type": "htr", "addr": "1", "name": "Living room"},
        {"type": "acm", "addr": "2", "name": "Hall"},
        {"type": "pmo", "addr": "3", "name": "Meter"},
        {"type": "thm", "addr": "4", "name": "Thermostat"},
    ]
}
HB_INTERVAL = 27.0  # 0.45 * the 60 s handshake heartbeat timeout
RTC_INTERVAL = 30.0
IDLE_CHECK = 60.0
PAYLOAD_WINDOW = 240.0


def event(name: str, payload: Any) -> str:
    """Return a Socket.IO 0.9 event frame for the TermoWeb namespace."""
    body = json.dumps({"name": name, "args": [payload]}, separators=(",", ":"))
    return f"5::{NS}:{body}"


def rtc_ok() -> MockResponse:
    """Return a successful RTC time response."""
    return MockResponse(
        200,
        {"y": 2026, "n": 10, "d": 10, "h": 12, "m": 0, "s": 0},
        headers={"Content-Type": "application/json"},
    )


@dataclass
class Harness:
    """Everything a TermoWeb websocket test touches."""

    hass: HomeAssistant
    session: WSFakeSession
    rest: RESTClient
    coordinator: StateCoordinator
    runtime: EntryRuntime
    client: termoweb_ws.TermoWebWSClient
    sleeps: SleepController
    clock: OffsetClock
    statuses: list[dict[str, Any]] = field(default_factory=list)

    def start(self) -> asyncio.Task[None]:
        """Start the client and let the sleep fake recognise its runner."""
        task = self.client.start()
        self.sleeps.runner = task
        return task

    async def connected(self, count: int = 1) -> FakeWS:
        """Wait until socket ``count`` finished its join and subscriptions."""
        await until(lambda: len(self.session.sockets) >= count)
        ws = self.session.sockets[count - 1]
        await until(lambda: len(ws.sent) >= 5)
        return ws

    def heater(self, addr: str = "1", node_type: str = "htr") -> dict[str, Any] | None:
        """Return the domain state of a heater as a dict."""
        state = self.coordinator.domain_view.get_heater_state(node_type, addr)
        return state_to_dict(state) if state is not None else None

    def status_changes(self) -> list[str]:
        """Return dispatched statuses with consecutive repeats collapsed."""
        seen = [payload["status"] for payload in self.statuses]
        return [s for i, s in enumerate(seen) if i == 0 or seen[i - 1] != s]


async def build_harness(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    *,
    api_base: str | None = None,
) -> Harness:
    """Build the production client graph on the fake transport."""
    sleeps = SleepController()
    sleeps.install(monkeypatch, termoweb_ws, limiter_module=ws_client)
    clock = OffsetClock()
    monkeypatch.setattr(termoweb_ws, "time", SimpleNamespace(time=clock.time))
    monkeypatch.setattr(
        termoweb_ws, "random", SimpleNamespace(uniform=lambda _a, _b: 1.0)
    )

    session = WSFakeSession()
    session.queue_post(LatchedResponse(token_response(TOKEN)))
    session.queue_request(LatchedResponse(rtc_ok()))
    kwargs = {} if api_base is None else {"api_base": api_base}
    rest = RESTClient(session, "user@example.com", "secret", **kwargs)
    inventory = Inventory(DEV_ID, build_node_inventory(NODES))
    coordinator = StateCoordinator(
        hass, rest, 30, DEV_ID, {"dev_id": DEV_ID, "name": "Home"}, inventory
    )
    runtime = build_entry_runtime(
        hass=hass,
        entry_id=ENTRY_ID,
        dev_id=DEV_ID,
        inventory=inventory,
        coordinator=coordinator,
        client=rest,
    )
    backend = create_backend(brand=BRAND_TERMOWEB, client=rest)
    client = backend.create_ws_client(
        hass, ENTRY_ID, DEV_ID, coordinator, inventory=inventory
    )
    assert type(client) is termoweb_ws.TermoWebWSClient
    harness = Harness(hass, session, rest, coordinator, runtime, client, sleeps, clock)

    @callback
    def _on_status(payload: dict[str, Any]) -> None:
        harness.statuses.append(payload)

    async_dispatcher_connect(hass, signal_ws_status(ENTRY_ID), _on_status)
    return harness


@pytest.fixture
async def tw(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> AsyncGenerator[Harness]:
    """Return a TermoWeb websocket harness; the client is stopped afterwards."""
    harness = await build_harness(hass, monkeypatch)
    yield harness
    await harness.client.stop()


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


async def test_session_end_to_end(
    tw: Harness, caplog: pytest.LogCaptureFixture
) -> None:
    """Handshake, join, snapshot, update, heartbeat, reconnect and clean stop."""
    caplog.set_level(logging.DEBUG)
    task = tw.start()
    ws1 = await tw.connected()

    url, kwargs = tw.session.get_calls[0]
    assert url.startswith(f"{API_BASE}/socket.io/1/?")
    assert f"token={TOKEN}" in url and f"dev_id={DEV_ID}" in url
    headers = kwargs["headers"]
    assert headers["User-Agent"] == get_brand_user_agent(BRAND_TERMOWEB)
    assert headers["X-Requested-With"] == get_brand_requested_with(BRAND_TERMOWEB)
    assert headers["Origin"] == "https://localhost"
    assert ws1.url.startswith(f"{API_BASE}/socket.io/1/websocket/sid1?")
    _, connect_kwargs = tw.session.connect_calls[0]
    assert isinstance(connect_kwargs["timeout"], aiohttp.ClientWSTimeout)
    assert connect_kwargs["timeout"].ws_close == ws_client._WS_CLOSE_TIMEOUT  # noqa: SLF001
    assert connect_kwargs["autoclose"] is False
    assert connect_kwargs["heartbeat"] is None
    # Join, snapshot, session metadata, then samples for heaters only (no pmo).
    assert ws1.sent == [
        f"1::{NS}",
        f'5::{NS}:{{"name":"dev_data","args":[]}}',
        f'5::{NS}:{{"name":"subscribe","args":["/mgr/session"]}}',
        f'5::{NS}:{{"name":"subscribe","args":["/htr/1/samples"]}}',
        f'5::{NS}:{{"name":"subscribe","args":["/acm/2/samples"]}}',
    ]
    await until(lambda: "connected" in tw.status_changes())
    assert tw.heater() is None

    ws1.feed("1::")
    ws1.feed(f"1::{NS}")
    ws1.feed(
        event(
            "dev_data",
            {"nodes": {"htr": {"settings": {"1": {"mode": "auto", "stemp": "20.0"}}}}},
        )
    )
    await until(lambda: tw.heater() is not None)
    assert tw.heater()["mode"] == "auto"
    assert tw.heater().get("stemp") == "20.0"
    await until(lambda: "healthy" in tw.status_changes())
    assert tw.coordinator.gateway_connected

    ws1.feed(event("update", {"path": "/htr/1/settings", "body": {"stemp": "22.5"}}))
    await until(lambda: tw.heater().get("stemp") == "22.5")
    assert tw.heater()["mode"] == "auto"

    # Server heartbeat: acknowledged and recorded.
    sent_before = len(ws1.sent)
    ws1.feed("2::")
    await until(lambda: len(ws1.sent) > sent_before)
    assert ws1.sent[sent_before:] == ["2::"]
    assert tw.runtime.ws_trackers[DEV_ID].last_heartbeat_at is not None

    # Server disconnect: back off 5 s (the session had payloads), reconnect.
    ws1.feed("0::")
    ws2 = await tw.connected(2)
    assert ws1.closed
    assert len(tw.session.get_calls) == 2
    assert ws2.url.startswith(f"{API_BASE}/socket.io/1/websocket/sid2?")
    assert tw.sleeps.backoffs == [5.0]
    await until(lambda: tw.status_changes().count("connected") >= 2)

    await tw.client.stop()

    assert task.done()
    assert ws2.closed
    assert tw.status_changes() == [
        "starting",
        "connected",
        "healthy",
        "disconnected",
        "connected",
        "disconnected",
        "stopped",
    ]
    # Teardown leaves no websocket state behind and no tasks running.
    assert DEV_ID not in tw.runtime.ws_state
    assert DEV_ID not in tw.runtime.ws_trackers
    assert tw.sleeps.parked_delays() == []
    # The token and the gateway id never reach the logs in clear text.
    assert TOKEN not in caplog.text
    assert f"dev_id={DEV_ID}" not in caplog.text


@pytest.mark.parametrize(
    ("api_base", "socket_base"),
    [
        ("https://example.com/api", "https://example.com/api"),
        ("https://example.com/api/", "https://example.com/api"),
        ("example.com", "https://example.com"),
        ("example.com/api", "https://example.com/api"),
    ],
)
async def test_websocket_url_follows_the_rest_api_base(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    api_base: str,
    socket_base: str,
) -> None:
    """The websocket upgrade is made against the REST client's API base."""
    harness = await build_harness(hass, monkeypatch, api_base=api_base)
    harness.start()
    ws = await harness.connected()
    await harness.client.stop()

    assert ws.url.startswith(f"{socket_base}/socket.io/1/websocket/sid1?")


async def test_quiet_session_does_not_redispatch_status(tw: Harness) -> None:
    """Updates and heartbeats on a healthy session dispatch no status changes."""
    tw.start()
    ws = await tw.connected()
    ws.feed(
        event("dev_data", {"nodes": {"htr": {"settings": {"1": {"mode": "auto"}}}}})
    )
    await until(lambda: "healthy" in tw.status_changes())
    status_dispatches = [p for p in tw.statuses if p["reason"] == "status"]

    for stemp in ("19.0", "19.5", "20.0"):
        ws.feed(event("update", {"path": "/htr/1/settings", "body": {"stemp": stemp}}))
        ws.feed("2::")
    await until(lambda: tw.heater().get("stemp") == "20.0")

    assert [p for p in tw.statuses if p["reason"] == "status"] == status_dispatches


# ---------------------------------------------------------------------------
# Reconnect, backoff and rate limiting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("with_payload", "expected"),
    [(True, [5.0, 5.0, 5.0, 5.0]), (False, [5.0, 10.0, 30.0, 120.0])],
)
async def test_backoff_restarts_after_a_session_with_payloads(
    tw: Harness, with_payload: bool, expected: list[float]
) -> None:
    """Backoff restarts after a session that delivered data; else it escalates."""
    seen: list[str] = []
    tw.sleeps.on_backoff = lambda _delay: seen.append(
        tw.runtime.ws_trackers[DEV_ID].status
    )
    tw.start()
    for count in range(1, 6):
        ws = await tw.connected(count)
        if with_payload:
            ws.feed(
                event("update", {"path": "/htr/1/settings", "body": {"stemp": "20"}})
            )
        ws.feed("0::")

    await until(lambda: len(tw.sleeps.backoffs) >= 4)
    assert tw.sleeps.backoffs[:4] == expected
    # The gateway is reported down before the client waits to reconnect.
    assert set(seen) == {"disconnected"}


async def test_handshake_rejections_refresh_the_token_and_back_off(
    tw: Harness, caplog: pytest.LogCaptureFixture
) -> None:
    """A 401 handshake forces a new token; retries are rate limited and spaced."""
    tw.session.handshakes.extend(FakeResponse(401, "unauthorized") for _ in range(6))
    caplog.set_level(logging.WARNING, logger=termoweb_ws.__name__)
    tw.start()

    await until(lambda: len(tw.session.get_calls) >= 7)
    await tw.connected()

    # One token for the first attempt, then a refresh per rejected handshake.
    assert len(tw.session.post_calls) == 1 + 6
    assert tw.sleeps.backoffs[:6] == [5.0, 10.0, 30.0, 120.0, 300.0, 300.0]
    # Every retry went through the connection rate limiter.
    assert len(tw.sleeps.throttled) == 6
    # Five failures in a row produce one warning, not one per attempt.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert [r.getMessage().split(" over ")[0] for r in warnings] == [
        "WS: handshake failed 5 times"
    ]
    assert TOKEN not in caplog.text


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(FakeResponse(503, "busy"), id="http-error"),
        pytest.param(FakeResponse(200, "garbage"), id="no-fields"),
        pytest.param(FakeResponse(200, "sid:abc:60:websocket"), id="bad-timeout"),
        pytest.param(aiohttp.ClientConnectionError("refused"), id="client-error"),
        pytest.param(TimeoutError(), id="timeout"),
    ],
)
async def test_failed_handshake_is_retried(tw: Harness, failure: Any) -> None:
    """Any handshake failure leads to a retry, without a new token."""
    tw.session.handshakes.append(failure)
    tw.start()

    await tw.connected()

    assert len(tw.session.get_calls) == 2
    assert len(tw.session.connect_calls) == 1
    assert tw.sleeps.backoffs == [5.0]
    assert len(tw.session.post_calls) == 1


@pytest.mark.parametrize(
    "headers",
    [
        {"Authorization": "Bearer"},
        {"Authorization": "Bearer "},
        {"Authorization": ""},
        {},
    ],
)
async def test_missing_token_never_reaches_the_handshake(
    tw: Harness, headers: dict[str, str]
) -> None:
    """A token-less Authorization header is a retryable error, not a crash."""
    with patch.object(tw.rest, "authed_headers", AsyncMock(return_value=headers)):
        tw.start()
        await until(lambda: len(tw.sleeps.backoffs) >= 2)

    assert tw.session.get_calls == []
    assert tw.sleeps.backoffs[:2] == [5.0, 10.0]


async def test_websocket_upgrade_is_bounded(
    tw: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hanging upgrade times out and is retried instead of blocking forever."""
    monkeypatch.setattr(ws_client, "_WS_CONNECT_TIMEOUT", 0.01)

    async def _hang(url: str, **kwargs: Any) -> FakeWS:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    tw.session.connect = _hang
    tw.start()

    await until(lambda: len(tw.session.connect_calls) >= 2)
    assert tw.sleeps.backoffs[:1] == [5.0]


async def test_cancelling_the_runner_propagates(tw: Harness) -> None:
    """Cancelling the client task leaves it cancelled and reports stopped."""
    started = asyncio.Event()

    class _HangingHandshake(FakeResponse):
        async def text(self) -> str:
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    tw.session.handshakes.append(_HangingHandshake())
    task = tw.start()
    await started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert task.cancelled()
    assert tw.runtime.ws_trackers[DEV_ID].status == "stopped"


@pytest.mark.parametrize(
    "ending",
    [
        pytest.param(lambda ws: ws.feed_message(aiohttp.WSMsgType.CLOSE), id="close"),
        pytest.param(lambda ws: ws.feed_message(aiohttp.WSMsgType.ERROR), id="error"),
        pytest.param(lambda ws: ws.end(), id="stream-end"),
        pytest.param(lambda ws: ws.feed("0::"), id="server-disconnect"),
    ],
)
async def test_transport_end_reconnects(tw: Harness, ending: Any) -> None:
    """Close, error and end-of-stream all end the session and reconnect."""
    tw.start()
    ws = await tw.connected()

    ending(ws)

    await tw.connected(2)
    assert ws.closed


# ---------------------------------------------------------------------------
# Frame routing into the domain state
# ---------------------------------------------------------------------------


async def test_frames_update_domain_state(tw: Harness) -> None:
    """Snapshots, path updates, data batches and power limits reach the store."""
    tw.start()
    ws = await tw.connected()

    # Snapshot with list-shaped nodes, as some gateways send it.
    ws.feed(
        event(
            "dev_data",
            {"nodes": [{"type": "htr", "addr": "1", "settings": {"mode": "auto"}}]},
        )
    )
    await until(lambda: tw.heater() is not None)
    assert tw.heater()["mode"] == "auto"

    # A binary frame carrying an update is decoded like text.
    ws.feed(
        event("update", {"path": "/htr/1/settings", "body": {"stemp": "21.0"}}).encode()
    )
    # A legacy data batch: gateway power limit and a path update.
    ws.feed(
        event(
            "data",
            [
                {"path": "/htr_system/power_limit", "body": {"power_limit": "2500"}},
                {"path": "/htr/1/settings", "body": {"mode": "manual"}},
            ],
        )
    )
    await until(lambda: tw.heater()["mode"] == "manual")
    assert tw.heater().get("stemp") == "21.0"
    assert tw.coordinator.domain_view.get_power_limit() == 2500

    # A power-limit path update outside a batch.
    ws.feed(
        event(
            "update",
            {
                "path": "/api/devs/x/htr_system/power_limit",
                "body": {"power_limit": "1800"},
            },
        )
    )
    await until(lambda: tw.coordinator.domain_view.get_power_limit() == 1800)
    assert len(tw.session.sockets) == 1


@pytest.mark.parametrize(
    "frame",
    [
        pytest.param(f"5::{NS}:{{not json", id="bad-json"),
        pytest.param(f'5::{NS}:"not-an-event"', id="not-a-dict"),
        pytest.param(
            event("data", [{"body": {"mode": "off"}}]), id="batch-without-path"
        ),
        pytest.param(event("data", ["junk"]), id="batch-junk-item"),
        pytest.param(event("update", {"path": 123, "body": {}}), id="bad-path"),
        pytest.param(
            event("update", {"nodes": {"htr": {"settings": {"9": {"mode": "off"}}}}}),
            id="unknown-node",
        ),
        pytest.param(event("dev_data", "junk"), id="bad-snapshot"),
        pytest.param(event("dev_handshake", "junk"), id="bad-handshake"),
        pytest.param(event("mystery", {}), id="unknown-event"),
        pytest.param(b"\xff\xfe", id="undecodable-binary"),
    ],
)
async def test_unusable_frames_are_ignored(tw: Harness, frame: str | bytes) -> None:
    """Junk frames change nothing and the session keeps reading."""
    tw.start()
    ws = await tw.connected()
    ws.feed(
        event("dev_data", {"nodes": {"htr": {"settings": {"1": {"mode": "auto"}}}}})
    )
    await until(lambda: tw.heater() is not None)
    before = tw.heater()

    ws.feed(frame)
    ws.feed(event("update", {"path": "/htr/1/settings", "body": {"stemp": "23.0"}}))
    await until(lambda: tw.heater().get("stemp") == "23.0")

    after = tw.heater()
    assert after.pop("stemp") == "23.0"
    before.pop("stemp", None)
    assert after == before
    assert tw.heater("9") is None
    assert len(tw.session.sockets) == 1


async def test_handshake_event_marks_the_session_healthy(tw: Harness) -> None:
    """The ``dev_handshake`` event counts as data and records its keys."""
    tw.start()
    ws = await tw.connected()

    ws.feed(event("dev_handshake", {"version": 1, "dev_id": "x"}))

    await until(lambda: "healthy" in tw.status_changes())
    assert tw.runtime.ws_state[DEV_ID]["handshake_keys"] == ("dev_id", "version")


async def test_samples_are_forwarded_to_the_energy_coordinator(tw: Harness) -> None:
    """Sample pushes go to the energy coordinator, also when normalising fails."""
    handler = tw.runtime.energy_coordinator.handle_ws_samples
    tw.start()
    ws = await tw.connected()

    ws.feed(event("update", {"path": "/htr/1/samples", "body": {"t": 1, "counter": 5}}))
    await until(lambda: handler.call_count == 1)
    assert handler.call_args.args[0] == DEV_ID
    assert handler.call_args.args[1] == {"htr": {"1": {"t": 1, "counter": 5}}}

    with patch.object(tw.rest, "normalise_ws_nodes", side_effect=RuntimeError("bad")):
        ws.feed(
            event(
                "update",
                {
                    "nodes": {
                        "acm": {"samples": {"2": {"counter": 7}}, "lease_seconds": 90}
                    }
                },
            )
        )
        await until(lambda: handler.call_count == 2)
    assert handler.call_args.args[1] == {"acm": {"2": {"counter": 7}}}
    assert handler.call_args.kwargs["lease_seconds"] == 90

    # Power monitor samples feed the energy sensors; thermostats have none.
    ws.feed(event("update", {"nodes": {"thm": {"samples": {"4": {"counter": 1}}}}}))
    ws.feed(event("update", {"nodes": {"pmo": {"samples": {"3": {"counter": 2}}}}}))
    await until(lambda: handler.call_count == 3)
    assert handler.call_args.args[1] == {"pmo": {"3": {"counter": 2}}}


async def test_handler_failures_are_errors_and_reading_continues(
    tw: Harness, caplog: pytest.LogCaptureFixture
) -> None:
    """Store and energy handler failures are logged at ERROR; later frames apply."""
    caplog.set_level(logging.DEBUG)
    tw.runtime.energy_coordinator.handle_ws_samples.side_effect = RuntimeError("bad")
    real_handle = tw.coordinator.handle_ws_deltas
    calls = 0

    def _flaky(*args: Any, **kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KeyError("boom")
        real_handle(*args, **kwargs)

    tw.start()
    ws = await tw.connected()
    with patch.object(tw.coordinator, "handle_ws_deltas", _flaky):
        ws.feed(event("update", {"path": "/htr/1/settings", "body": {"stemp": "17"}}))
        ws.feed(event("update", {"path": "/htr/1/samples", "body": {"counter": 1}}))
        ws.feed(event("update", {"path": "/htr/1/settings", "body": {"stemp": "19"}}))
        await until(
            lambda: tw.heater() is not None and tw.heater().get("stemp") == "19"
        )

    errors = {r.getMessage() for r in caplog.records if r.levelno == logging.ERROR}
    assert "WS: failed to apply websocket deltas" in errors
    assert any("forwarding heater samples failed" in msg for msg in errors)
    assert len(tw.session.sockets) == 1


# ---------------------------------------------------------------------------
# Keep-alive loops and idle restarts
# ---------------------------------------------------------------------------


async def test_client_heartbeat_and_rtc_keepalive(tw: Harness) -> None:
    """The client sends ``2::`` heartbeats and polls the RTC to keep the session."""
    tw.session.clear_calls()
    tw.start()
    ws = await tw.connected()
    await until(
        lambda: (
            {HB_INTERVAL, RTC_INTERVAL, IDLE_CHECK} <= set(tw.sleeps.parked_delays())
        )
    )
    rtc_path = f"/api/v2/devs/{DEV_ID}/mgr/rtc/time"

    def rtc_calls() -> int:
        return sum(rtc_path in url for _m, url, _kw in tw.session.request_calls)

    assert rtc_calls() == 1
    sent_before = len(ws.sent)

    assert tw.sleeps.release(HB_INTERVAL) == 1
    await until(lambda: len(ws.sent) > sent_before)
    assert ws.sent[sent_before:] == ["2::"]

    # A failing RTC poll does not end the keep-alive loop.
    tw.session.push_request(aiohttp.ClientConnectionError("down"))
    assert tw.sleeps.release(RTC_INTERVAL) == 1
    await until(lambda: RTC_INTERVAL in tw.sleeps.parked_delays())
    assert tw.sleeps.release(RTC_INTERVAL) == 1
    await until(lambda: rtc_calls() >= 3)
    assert len(tw.session.sockets) == 1


async def test_idle_session_is_restarted(tw: Harness) -> None:
    """No payload for longer than the window: the idle monitor reconnects."""
    tw.start()
    ws = await tw.connected()
    ws.feed(
        event("dev_data", {"nodes": {"htr": {"settings": {"1": {"mode": "auto"}}}}})
    )
    await until(lambda: "healthy" in tw.status_changes())
    await until(lambda: IDLE_CHECK in tw.sleeps.parked_delays())

    tw.clock.offset = PAYLOAD_WINDOW + 60
    tw.sleeps.release(IDLE_CHECK)

    await tw.connected(2)
    assert ws.close_calls[0] == (aiohttp.WSCloseCode.GOING_AWAY, b"idle restart")


@pytest.mark.parametrize(("idle_for", "restarts"), [(30.0, False), (300.0, True)])
async def test_write_after_a_quiet_period_restarts_the_session(
    tw: Harness, idle_for: float, restarts: bool
) -> None:
    """A REST write after a long silence restarts the websocket; a recent one not."""
    tw.start()
    ws = await tw.connected()
    ws.feed(
        event("dev_data", {"nodes": {"htr": {"settings": {"1": {"mode": "auto"}}}}})
    )
    await until(lambda: "healthy" in tw.status_changes())
    tw.session.push_request(
        MockResponse(201, {}, headers={"Content-Type": "application/json"})
    )

    tw.clock.offset = idle_for
    await tw.rest.set_node_settings(DEV_ID, ("htr", "1"), mode="manual")

    if restarts:
        await tw.connected(2)
    else:
        await asyncio.sleep(0)
        assert len(tw.session.sockets) == 1
        assert not ws.closed
