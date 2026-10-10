"""Ducaheat websocket session lifecycle on real Home Assistant.

Status hygiene, reconnect backoff, parse-error handling and teardown of the
production ``DucaheatWSClient``, driven end to end through its Engine.IO
polling handshake and websocket upgrade on the fake transport.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
import json
import logging
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import aiohttp
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
import pytest

from custom_components.termoweb.backend import create_backend, ducaheat_ws, ws_client
from custom_components.termoweb.backend.rest_client import RESTClient
from custom_components.termoweb.const import (
    BRAND_DUCAHEAT,
    WS_NAMESPACE,
    get_brand_api_base,
    signal_ws_status,
)
from custom_components.termoweb.coordinator import StateCoordinator
from custom_components.termoweb.domain import state_to_dict
from custom_components.termoweb.inventory import Inventory, build_node_inventory
from custom_components.termoweb.runtime import EntryRuntime
from tests_ha.fakes.rest import LatchedResponse
from tests_ha.fakes.runtime import build_entry_runtime
from tests_ha.fakes.termoweb_ws import (
    FakeWebSocket,
    HandshakeResponse,
    SleepController,
    WSFakeSession,
    token_response,
    until,
)

DEV_ID = "fedcba9876543210"  # synthetic gateway id
ENTRY_ID = "entry-ducaheat-ws"
TOKEN = "tok-fedcba9876543210"
NS = WS_NAMESPACE
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
        return self.runtime.ws_state[DEV_ID].get("frames_total", 0)

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
        brand=BRAND_DUCAHEAT,
    )
    backend = create_backend(brand=BRAND_DUCAHEAT, client=rest)
    client = backend.create_ws_client(
        hass, ENTRY_ID, DEV_ID, coordinator, inventory=inventory
    )
    assert type(client) is ducaheat_ws.DucaheatWSClient
    harness = Harness(session, coordinator, runtime, client, sleeps, clock)

    @callback
    def _on_status(payload: dict[str, Any]) -> None:
        harness.statuses.append(payload)

    unsub = async_dispatcher_connect(hass, signal_ws_status(ENTRY_ID), _on_status)
    yield harness
    await client.stop()
    unsub()


async def healthy(h: Harness) -> FakeWebSocket:
    """Start the client and wait for a session that delivered data."""
    h.start()
    ws = await h.connected()
    ws.feed(update("20.0"))
    await until(lambda: h.runtime.ws_trackers[DEV_ID].status == "healthy")
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

    assert dh.runtime.ws_trackers[DEV_ID].status == "connected"
    assert dh.status_dispatches() == [*before, "connected"]


async def test_fresh_frames_do_not_redispatch_status(dh: Harness) -> None:
    """While healthy and fresh, pongs and updates do not re-dispatch status."""
    ws = await healthy(dh)
    before = dh.status_dispatches()

    for stemp in ("20.5", "21.0"):
        ws.feed("3")
        ws.feed(update(stemp))
    await until(lambda: dh.stemp() == "21.0")

    assert dh.runtime.ws_trackers[DEV_ID].status == "healthy"
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
        dh.runtime.ws_trackers[DEV_ID].status
    )
    dh.start()
    for count in range(1, 6):
        ws = await dh.connected(count)
        if with_payload:
            ws.feed(update("20.0"))
            await until(lambda: dh.runtime.ws_trackers[DEV_ID].status == "healthy")
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
    tracker = dh.runtime.ws_trackers[DEV_ID]
    assert tracker.payload_stale_after == ducaheat_ws._PAYLOAD_WINDOW_DEFAULT  # noqa: SLF001
    ws.feed("42{bad")
    await until(lambda: dh.runtime.ws_state[DEV_ID]["parse_errors_total"] == 1)
    attempts = dh.runtime.ws_state[DEV_ID]["subscribe_attempts_total"]
    assert attempts >= 1

    ws.end()
    await dh.connected(2)

    state = dh.runtime.ws_state[DEV_ID]
    assert state["parse_errors_total"] == 1
    assert state["subscribe_attempts_total"] >= attempts
    assert dh.runtime.ws_trackers[DEV_ID] is tracker

    await dh.client.stop()

    assert DEV_ID not in dh.runtime.ws_state
    assert DEV_ID not in dh.runtime.ws_trackers


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
    assert dh.runtime.ws_trackers[DEV_ID].status == "stopped"


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
