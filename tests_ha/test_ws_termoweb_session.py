"""TermoWeb Socket.IO 0.9 sessions driven end to end on real Home Assistant.

The real ``TermoWebWSClient`` runs against a scripted aiohttp transport and a
virtual clock (see ``ws_harness``). The integration itself is set up for real,
so frames land in the real ``DomainStateStore`` and writes go through the real
climate entity.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import aiohttp
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.termoweb import (
    climate as climate_module,
    entity as entity_module,
)
from custom_components.termoweb.backend import termoweb_ws
from custom_components.termoweb.backend.rest_client import RESTClient
from custom_components.termoweb.backend.termoweb_ws import TermoWebWSClient
from custom_components.termoweb.const import WS_NAMESPACE, signal_ws_status
from custom_components.termoweb.domain import state_to_dict
from custom_components.termoweb.inventory import Inventory, build_node_inventory

from .conftest import DEV_ID, FakeCloud
from .ws_harness import (
    FakeResponse,
    FakeSession,
    FakeWS,
    VirtualClock,
    install_clock,
    leaked_tasks,
    settle,
    sio09_event,
    until,
)

TOKEN = "tkn-secret"  # synthetic
DEV_B = "fedcba9876543210"  # second synthetic gateway on the same account
NS = WS_NAMESPACE
JOIN = f"1::{NS}"
SNAPSHOT_REQUEST = f'5::{NS}:{{"name":"dev_data","args":[]}}'
HANDSHAKE_OK = "sid:60:60:websocket,xhr-polling"


def _event(name: str, payload: Any) -> str:
    return sio09_event(NS, name, payload)


def _ok_handshake(_url: str) -> FakeResponse:
    return FakeResponse(body=HANDSHAKE_OK)


class Env:
    """Integration set up for real, plus helpers to build websocket clients."""

    def __init__(
        self, hass: HomeAssistant, entry: MockConfigEntry, clock: VirtualClock
    ) -> None:
        """Record websocket status signals for the entry."""
        self.hass = hass
        self.entry = entry
        self.clock = clock
        self.statuses: dict[str, list[str]] = {}

        @callback
        def _on_status(payload: dict[str, Any]) -> None:
            self.statuses.setdefault(payload["dev_id"], []).append(payload["status"])

        entry.async_on_unload(
            async_dispatcher_connect(hass, signal_ws_status(entry.entry_id), _on_status)
        )

    @property
    def runtime(self) -> Any:
        """Return the entry runtime."""
        return self.entry.runtime_data

    def client(
        self,
        session: FakeSession,
        *,
        dev_id: str = DEV_ID,
        inventory: Inventory | None = None,
    ) -> TermoWebWSClient:
        """Build the production client exactly as the backend factory does."""
        return TermoWebWSClient(
            self.hass,
            entry_id=self.entry.entry_id,
            dev_id=dev_id,
            api_client=self.runtime.client,
            coordinator=self.runtime.coordinator,
            session=session,
            inventory=inventory or self.runtime.inventory,
        )

    def heater(self) -> dict[str, Any] | None:
        """Return the stored heater state."""
        state = self.runtime.coordinator.domain_view.get_heater_state("htr", "1")
        return state_to_dict(state) if state is not None else None

    def status(self, dev_id: str = DEV_ID) -> str | None:
        """Return the last websocket status published for ``dev_id``."""
        seen = self.statuses.get(dev_id)
        return seen[-1] if seen else None

    async def advance_until(
        self, predicate: Callable[[], bool], *, step: float = 1.0, limit: float = 3600
    ) -> None:
        """Advance virtual time in ``step`` increments until ``predicate`` holds."""
        elapsed = 0.0
        while not predicate():
            assert elapsed < limit, "virtual time limit reached"
            await self.clock.advance(step)
            elapsed += step


@pytest.fixture
def write_mock() -> Generator[AsyncMock]:
    """Accept heater writes at the REST boundary, without the write debounce."""
    with (
        patch.object(RESTClient, "set_node_settings", AsyncMock(return_value={})) as m,
        patch.object(climate_module, "_WRITE_DEBOUNCE", 0),
        patch.object(entity_module, "WS_ECHO_FALLBACK_REFRESH", 0),
    ):
        yield m


@pytest.fixture
async def env(
    hass: HomeAssistant,
    cloud: FakeCloud,
    config_entry: MockConfigEntry,
    write_mock: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> Env:
    """Set up the integration and give the websocket modules a virtual clock."""
    caplog.set_level(logging.DEBUG, logger="custom_components.termoweb")
    clock = install_clock(monkeypatch, termoweb_ws)
    monkeypatch.setattr(termoweb_ws, "random", SimpleNamespace(uniform=lambda *_: 1.0))
    monkeypatch.setattr(
        RESTClient,
        "authed_headers",
        AsyncMock(return_value={"Authorization": f"Bearer {TOKEN}"}),
    )
    config_entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(config_entry.entry_id)
    await hass.async_block_till_done()
    return Env(hass, config_entry, clock)


async def _stop(client: TermoWebWSClient) -> None:
    """Stop the client and prove nothing it started is left running."""
    await client.stop()
    await settle()
    assert client._task is None  # noqa: SLF001
    assert leaked_tasks("TermoWebWSClient") == []


async def test_handshake_failures_back_off_and_recover(
    env: Env, caplog: pytest.LogCaptureFixture
) -> None:
    """Every handshake failure mode retries with growing backoff, then connects."""
    script = iter(
        [
            FakeResponse(status=503, body="busy"),
            FakeResponse(error=aiohttp.ClientConnectionError("refused")),
            FakeResponse(hang=True),  # no answer within the 15 s handshake timeout
            FakeResponse(body="garbage"),
            FakeResponse(body="sid:soon:60:websocket"),
            FakeResponse(body=HANDSHAKE_OK),  # socket comes back already closed
            FakeResponse(body=HANDSHAKE_OK),
        ]
    )
    sockets = iter([True, False])

    def _ws(url: str) -> FakeWS:
        ws = FakeWS(url)
        ws.closed = next(sockets)
        return ws

    session = FakeSession(get=lambda _url: next(script), ws_factory=_ws)
    client = env.client(session)

    client.start()
    await env.advance_until(
        lambda: len(session.sockets) == 2 and env.status() == "connected", step=5
    )

    assert len(session.requests) == 7
    dead, live = session.sockets
    assert dead.sent == []  # never joined a socket the server already closed
    assert live.sent[:2] == [JOIN, SNAPSHOT_REQUEST]
    # No session delivered a payload, so the backoff kept growing.
    assert env.clock.sleeps[:6] == [5, 10, 30, 120, 300, 300]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert [r.getMessage().split(" over ")[0] for r in warnings] == [
        "WS: handshake failed 5 times"
    ]
    assert "status=599" in caplog.text  # handshake timeout
    assert "status=598" in caplog.text  # transport error
    assert "status=591" in caplog.text  # non-numeric heartbeat timeout
    assert TOKEN not in caplog.text  # URLs are redacted in every log line

    await _stop(client)
    assert live.closed
    assert env.status() == "stopped"


async def test_session_frames_reach_store_and_keepalives_run(
    env: Env, cloud: FakeCloud, caplog: pytest.LogCaptureFixture
) -> None:
    """Snapshot, batched pushes, junk frames and keep-alives over two sessions."""
    session = FakeSession(get=_ok_handshake)
    client = env.client(session)
    cloud.get_rtc_time.reset_mock()
    cloud.get_rtc_time.side_effect = [aiohttp.ClientError("rtc down"), {}, {}, {}, {}]
    prog = [0, 1, 2] * 56

    client.start()
    await until(lambda: len(session.sockets) == 1, "first socket")
    ws = session.sockets[0]
    await until(lambda: env.status() == "connected", "connected")

    ws.feed(
        _event("dev_handshake", {"dev_id": DEV_ID, "fw": "1"}),
        f"5::{NS}:{{not json",
        _event("no_such_event", {}),
        _event(
            "dev_data",
            {"nodes": {"htr": {"settings": {"1": {"mode": "auto", "stemp": "20.0"}}}}},
        ),
    )
    # REST seeded stemp 21.0; the websocket snapshot replaces it.
    await until(lambda: env.heater()["stemp"] == "20.0", "snapshot applied")
    assert env.status() == "healthy"
    assert len(session.sockets) == 1  # the junk frame did not end the session

    ws.feed(
        _event(
            "data",
            [
                {"path": "/htr/1/settings", "body": {"stemp": "22.5", "prog": prog}},
                {"path": "/mgr/nodes", "body": {"nodes": [{"type": "htr", "addr": 1}]}},
                {
                    "path": f"/api/v2/devs/{DEV_ID}/htr_system/power_limit",
                    "body": {"power_limit": "1500"},
                },
            ],
        )
    )
    await until(lambda: env.heater()["stemp"] == "22.5", "batched update")
    heater = env.heater()
    assert heater["mode"] == "auto"
    assert heater["prog"] == prog
    assert env.runtime.coordinator.domain_view.get_power_limit() == 1500

    # Client heartbeat after hb_timeout * 0.45 = 27 s; RTC keep-alive every 30 s
    # survives the first failing poll.
    await env.clock.advance(30)
    assert ws.sent.count("2::") == 1
    assert cloud.get_rtc_time.await_count == 2
    assert "RTC keep-alive failed (ClientError: rtc down)" in caplog.text

    # The server closes the stream: reconnect after backoff, state survives.
    ws.server_close(code=1001)
    await env.advance_until(lambda: len(session.sockets) == 2)
    assert "websocket payload stream ended code=1001" in caplog.text
    ws2 = session.sockets[1]
    await until(lambda: SNAPSHOT_REQUEST in ws2.sent, "second snapshot request")
    assert env.heater()["stemp"] == "22.5"

    # The transport dies silently: the next heartbeat fails without noise and
    # the session is replaced once the socket reports the closure.
    ws2.closed = True
    await env.clock.advance(30)
    assert "2::" not in ws2.sent
    ws2.drop()
    await env.advance_until(lambda: len(session.sockets) == 3)

    await _stop(client)


async def test_idle_session_is_restarted(env: Env) -> None:
    """Payload silence beyond the idle window recycles the websocket."""
    session = FakeSession(get=_ok_handshake)
    client = env.client(session)
    client.start()
    await until(lambda: len(session.sockets) == 1, "socket")
    ws = session.sockets[0]
    snapshot = {"nodes": {"htr": {"settings": {"1": {"mode": "auto"}}}}}
    ws.feed(_event("dev_data", snapshot))
    await until(lambda: env.status() == "healthy", "healthy")

    # Regular pushes keep the session alive well past the idle window.
    for _ in range(3):
        await env.clock.advance(100)
        ws.feed(_event("update", {"path": "/htr/1/settings", "body": {"mode": "off"}}))
    last_payload = env.clock.now
    await env.clock.advance(100)
    assert len(session.sockets) == 1

    # Server heartbeats alone do not count as payloads.
    while len(session.sockets) == 1:
        assert env.clock.now - last_payload < 400
        ws.feed("2::")
        await env.clock.advance(30)
    assert env.clock.now - last_payload >= 240

    assert ws.close_calls[0] == (aiohttp.WSCloseCode.GOING_AWAY, b"idle restart")
    assert env.runtime.ws_state[DEV_ID]["idle_restart_pending"] is False
    assert env.heater()["mode"] == "off"

    await _stop(client)


async def test_write_after_idle_restarts_only_that_gateway(
    env: Env, hass: HomeAssistant, write_mock: AsyncMock
) -> None:
    """A heater write after a silent period restarts that gateway's socket only."""
    session_a = FakeSession(get=_ok_handshake)
    session_b = FakeSession(get=_ok_handshake)
    inventory_b = Inventory(
        DEV_B,
        build_node_inventory({"nodes": [{"type": "htr", "addr": "1", "name": "B"}]}),
    )
    client_a = env.client(session_a)
    client_b = env.client(session_b, dev_id=DEV_B, inventory=inventory_b)
    snapshot = {"nodes": {"htr": {"settings": {"1": {"mode": "auto"}}}}}

    client_a.start()
    client_b.start()
    await until(lambda: session_a.sockets and session_b.sockets, "both sockets")
    ws_a, ws_b = session_a.sockets[0], session_b.sockets[0]
    ws_a.feed(_event("dev_data", snapshot))
    ws_b.feed(_event("dev_data", snapshot))
    await until(lambda: env.status() == "healthy", "gateway A healthy")

    async def _set_temperature() -> None:
        await hass.services.async_call(
            "climate",
            "set_temperature",
            {"entity_id": "climate.living_room", "temperature": 23},
            blocking=True,
        )
        for _ in range(5):  # the debounced write runs in a background task
            await settle()
            await hass.async_block_till_done()

    # A write while payloads are fresh changes nothing.
    await env.clock.advance(10)
    await _set_temperature()
    assert write_mock.await_count == 1
    assert not ws_a.closed

    # Gateway B keeps talking; gateway A falls silent past the 240 s window
    # (but before its own 60 s idle check notices).
    await env.clock.advance(200)
    ws_b.feed(_event("update", {"path": "/htr/1/settings", "body": {"mode": "off"}}))
    await env.clock.advance(35)
    await _set_temperature()

    assert write_mock.await_count == 2
    assert write_mock.await_args.args[0] == DEV_ID
    assert ws_a.close_calls == [(aiohttp.WSCloseCode.GOING_AWAY, b"idle restart")]
    assert not ws_b.closed
    await env.advance_until(lambda: len(session_a.sockets) == 2)
    assert len(session_b.sockets) == 1

    await client_b.stop()
    await _stop(client_a)


async def test_silent_first_session_is_restarted(env: Env) -> None:
    """A session that never delivers a payload is recycled after the idle window."""
    session = FakeSession(get=_ok_handshake)
    client = env.client(session)
    client.start()
    await until(lambda: len(session.sockets) == 1, "socket")
    ws = session.sockets[0]
    await until(lambda: env.status() == "connected", "connected")
    connected_at = env.clock.now

    # No snapshot reply, no pushes: only server heartbeats arrive.
    while len(session.sockets) == 1:
        assert env.clock.now - connected_at < 600, "silent session never restarted"
        ws.feed("2::")
        await env.clock.advance(10)

    assert ws.close_calls[0] == (aiohttp.WSCloseCode.GOING_AWAY, b"idle restart")
    # Restarted at the first idle check at or after the 240 s window.
    assert 240 <= env.clock.now - connected_at <= 240 + 60 + 30

    await _stop(client)
