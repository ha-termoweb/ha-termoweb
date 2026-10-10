"""Ducaheat Engine.IO v3 sessions driven end to end on real Home Assistant.

The real ``DucaheatWSClient`` performs the polling handshake, the websocket
probe/upgrade and the Socket.IO namespace join against a scripted aiohttp
transport, with a virtual clock (see ``ws_harness``). Frames land in the real
``DomainStateStore`` of a Ducaheat config entry.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Iterable
import logging
from typing import Any
from unittest.mock import AsyncMock, patch

import aiohttp
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.termoweb.backend import ducaheat_ws
from custom_components.termoweb.backend.ducaheat import (
    DucaheatBackend,
    DucaheatRESTClient,
)
from custom_components.termoweb.backend.ducaheat_ws import DucaheatWSClient
from custom_components.termoweb.const import (
    BRAND_DUCAHEAT,
    CONF_BRAND,
    DOMAIN,
    WS_NAMESPACE,
    signal_ws_status,
)
from custom_components.termoweb.domain import state_to_dict
from tests.fakes.cloud import DEV_ID, PASSWORD, USERNAME, FakeCloud
from tests.fakes.ws_harness import (
    FakeResponse,
    FakeSession,
    VirtualClock,
    eio_polling_body,
    install_clock,
    leaked_tasks,
    settle,
    sio_event,
    until,
)

TOKEN = "tkn-secret"  # synthetic
NS = WS_NAMESPACE
NS_ACK = f"40{NS}"
OPEN = '0{"sid":"S1","pingInterval":25000,"pingTimeout":60000}'
SUBSCRIBES = [
    sio_event(NS, "subscribe", "/htr/1/samples"),
    sio_event(NS, "subscribe", "/htr/1/status"),
]
DEV_DATA_REQUEST = sio_event(NS, "dev_data")
SNAPSHOT = {"nodes": {"htr": {"status": {"1": {"mode": "auto", "stemp": "20.0"}}}}}


def _event(name: str, *args: Any) -> str:
    return sio_event(NS, name, *args)


def _update(stemp: str) -> str:
    return _event("update", {"path": "/htr/1/status", "body": {"stemp": stemp}})


def _polling_get(url: str) -> FakeResponse:
    """Serve the Engine.IO OPEN packet, then the drained namespace ack."""
    if "sid=" not in url:
        return FakeResponse(body=eio_polling_body(OPEN))
    return FakeResponse(body=eio_polling_body("40"))


def _session(ws_frames: Callable[[int], Iterable[str]]) -> FakeSession:
    return FakeSession(
        get=_polling_get, post=lambda _url: FakeResponse(body="ok"), ws_frames=ws_frames
    )


class Env:
    """Ducaheat entry set up for real, plus websocket client helpers."""

    def __init__(
        self, hass: HomeAssistant, entry: MockConfigEntry, clock: VirtualClock
    ) -> None:
        """Record websocket status signals for the entry."""
        self.hass = hass
        self.entry = entry
        self.clock = clock
        self.statuses: list[str] = []

        @callback
        def _on_status(payload: dict[str, Any]) -> None:
            self.statuses.append(payload["status"])

        entry.async_on_unload(
            async_dispatcher_connect(hass, signal_ws_status(entry.entry_id), _on_status)
        )

    @property
    def runtime(self) -> Any:
        """Return the entry runtime."""
        return self.entry.runtime_data

    @property
    def ws_state(self) -> dict[str, Any]:
        """Return the gateway's websocket diagnostics bucket."""
        return self.runtime.ws_state[DEV_ID]

    def client(self, session: FakeSession) -> DucaheatWSClient:
        """Build the production client exactly as the backend factory does."""
        return DucaheatWSClient(
            self.hass,
            entry_id=self.entry.entry_id,
            dev_id=DEV_ID,
            api_client=self.runtime.client,
            coordinator=self.runtime.coordinator,
            session=session,
            namespace=NS,
            inventory=self.runtime.inventory,
        )

    def stemp(self) -> str | None:
        """Return the stored heater setpoint."""
        state = self.runtime.coordinator.domain_view.get_heater_state("htr", "1")
        return state_to_dict(state)["stemp"] if state is not None else None

    def status(self) -> str | None:
        """Return the last websocket status published."""
        return self.statuses[-1] if self.statuses else None

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
async def env(
    hass: HomeAssistant,
    cloud: FakeCloud,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> AsyncGenerator[Env]:
    """Set up a Ducaheat entry and give the websocket modules a virtual clock."""
    caplog.set_level(logging.DEBUG, logger="custom_components.termoweb")
    clock = install_clock(monkeypatch, ducaheat_ws)
    entry = MockConfigEntry(
        domain=DOMAIN,
        unique_id=f"ducaheat:{USERNAME}",
        data={"username": USERNAME, "password": PASSWORD, CONF_BRAND: BRAND_DUCAHEAT},
    )
    with (
        patch.object(DucaheatRESTClient, "get_node_settings", cloud.get_node_settings),
        patch.object(DucaheatRESTClient, "get_node_samples", cloud.get_node_samples),
        patch.object(
            DucaheatRESTClient,
            "authed_headers",
            AsyncMock(return_value={"Authorization": f"Bearer {TOKEN}"}),
        ),
        patch.object(
            DucaheatBackend,
            "create_ws_client",
            lambda _self, hass, *a, **kw: cloud.create_ws_client(hass, *a, **kw),
        ),
    ):
        entry.add_to_hass(hass)
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        yield Env(hass, entry, clock)


async def _stop(client: DucaheatWSClient) -> None:
    """Stop the client and prove nothing it started is left running."""
    await client.stop()
    await settle()
    assert client._task is None  # noqa: SLF001
    assert leaked_tasks("DucaheatWSClient") == []


async def _join(env: Env, session: FakeSession, index: int) -> Any:
    """Wait for socket ``index`` to open its namespace, then ack it."""
    await env.advance_until(
        lambda: len(session.sockets) > index and NS_ACK in session.sockets[index].sent
    )
    ws = session.sockets[index]
    ws.feed(NS_ACK)
    await until(lambda: DEV_DATA_REQUEST in ws.sent, "snapshot request")
    return ws


async def test_probe_and_upgrade_handshake(
    env: Env, caplog: pytest.LogCaptureFixture
) -> None:
    """An unanswered probe retries; pings during probe and upgrade are answered."""
    frames = {
        0: ["2", "unexpected"],  # never acks the probe
        1: ["2", "3probe", "2", "6"],  # ping, ack, ping and a noop while draining
    }
    session = _session(lambda index: frames[index])
    client = env.client(session)

    client.start()
    await env.advance_until(lambda: len(session.sockets) == 2)

    first, second = session.sockets
    assert first.sent == ["2probe", "3"]
    assert first.close_calls  # the failed attempt released its socket
    assert "status=408, detail=probe ack timeout" in caplog.text
    assert "unexpected probe frame: 'unexpected'" in caplog.text
    assert 5 in env.clock.sleeps  # reconnect backoff

    await until(lambda: NS_ACK in second.sent, "namespace open")
    assert second.sent == ["2probe", "3", "5", "3", NS_ACK]
    assert "discarding frame after upgrade: '6'" in caplog.text
    assert [method for method, _ in session.requests] == ["GET", "POST", "GET"] * 2
    assert all(TOKEN in url for _, url in session.requests)

    second.feed(NS_ACK)
    await until(lambda: SUBSCRIBES[-1] in second.sent, "subscriptions")
    assert second.sent[5:] == [DEV_DATA_REQUEST, *SUBSCRIBES]
    second.feed(_event("dev_data", SNAPSHOT))
    await until(lambda: env.stemp() == "20.0", "snapshot applied")
    assert env.status() == "healthy"

    await _stop(client)
    assert second.closed
    assert env.status() == "stopped"


async def test_repeated_parse_errors_force_reconnect(
    env: Env, caplog: pytest.LogCaptureFixture
) -> None:
    """Sporadic junk frames are tolerated; a burst of them recycles the socket."""
    session = _session(lambda _index: ["3probe"])
    client = env.client(session)
    client.start()
    ws = await _join(env, session, 0)
    ws.feed(_event("dev_data", SNAPSHOT))
    await until(lambda: env.stemp() == "20.0", "snapshot applied")

    bad_json = f"42{NS},{{oops"
    bad_shape = "42[]"
    ws.feed(bad_json, bad_shape)
    await env.clock.advance(31)  # the first two age out of the 30 s window
    ws.feed(bad_json, _update("21.0"))
    await until(lambda: env.stemp() == "21.0", "update after junk")
    assert len(session.sockets) == 1
    assert env.ws_state["parse_errors_total"] == 3
    assert "repeated parse errors" not in caplog.text

    ws.feed(bad_json, bad_json, _update("25.0"))
    await env.advance_until(lambda: len(session.sockets) == 2)

    assert "WS: repeated parse errors (5 in 30s); reconnecting" in caplog.text
    assert env.stemp() == "21.0"  # frames after the burst were dropped
    assert ws.close_calls == [(aiohttp.WSCloseCode.GOING_AWAY, b"loop")]

    await _join(env, session, 1)
    await _stop(client)


async def test_backoff_resets_after_idle_escalation_of_healthy_session(
    env: Env,
) -> None:
    """A session that delivered payloads resets backoff even after idle escalation."""
    # The first attempt never acks the probe, which advances the backoff.
    session = _session(lambda index: ["unexpected"] if index == 0 else ["3probe"])
    client = env.client(session)
    client.start()
    ws = await _join(env, session, 1)
    ws.feed(_event("dev_data", SNAPSHOT))
    await until(lambda: env.status() == "healthy", "healthy")

    # Silence until idle recovery gives up and reconnects.
    await env.advance_until(lambda: ws.closed, limit=600)
    assert ws.close_calls[0] == (
        aiohttp.WSCloseCode.GOING_AWAY,
        b"idle_recovery_failed",
    )
    closed_at = env.clock.now
    await env.advance_until(lambda: len(session.sockets) == 3)
    assert env.clock.now - closed_at == pytest.approx(5)  # reset, not the 2nd step

    await _join(env, session, 2)
    await _stop(client)


async def test_empty_namespace_events_are_parse_errors(
    env: Env, caplog: pytest.LogCaptureFixture
) -> None:
    """Event frames without a usable body count as parse errors, not pings."""
    session = _session(lambda _index: ["3probe"])
    client = env.client(session)
    client.start()
    ws = await _join(env, session, 0)
    ws.feed(_event("dev_data", SNAPSHOT))
    await until(lambda: env.stemp() == "20.0", "snapshot applied")
    sent_before = len(ws.sent)

    ws.feed(f"42{NS},[]", f"42{NS}", _update("21.0"))
    await until(lambda: env.stemp() == "21.0", "update after junk")

    assert f"3{NS}" not in ws.sent[sent_before:]  # not answered as pings
    assert env.ws_state["parse_errors_total"] == 2
    assert "parse error (shape) count=1" in caplog.text
    assert "parse error (namespace) count=2" in caplog.text
    assert len(session.sockets) == 1

    # A third one inside the 30 s window recycles the socket.
    ws.feed(f"42{NS}")
    await env.advance_until(lambda: len(session.sockets) == 2)
    assert "WS: repeated parse errors (3 in 30s); reconnecting" in caplog.text

    await _join(env, session, 1)
    await _stop(client)


async def test_stale_payloads_trigger_recovery_then_reconnect(env: Env) -> None:
    """Silence marks the feed stale, recovery re-requests data, then reconnects."""
    session = _session(lambda _index: ["3probe"])
    client = env.client(session)
    client.start()
    ws = await _join(env, session, 0)
    ws.feed(_event("dev_data", SNAPSHOT))
    await until(lambda: env.status() == "healthy", "healthy")

    # Updates every 100 s keep the 240 s payload window fresh.
    for stemp in ("21.0", "22.0", "23.0"):
        await env.clock.advance(100)
        ws.feed(_update(stemp))
    await env.clock.advance(1)
    assert env.stemp() == "23.0"
    assert env.status() == "healthy"
    sent_before = len(ws.sent)

    # Then the server falls silent (engine pings still flow both ways).
    await env.advance_until(lambda: env.status() == "connected", limit=300)
    assert env.ws_state["payload_stale"] is True

    await env.advance_until(lambda: len(session.sockets) == 2, limit=300)

    # One early lease refresh at 80 % of the window, then three recoveries.
    recovery = [frame for frame in ws.sent[sent_before:] if frame != "2"]
    assert recovery == [DEV_DATA_REQUEST, *SUBSCRIBES] * 4
    assert env.ws_state["recovery_attempts_total"] == 3
    assert ws.close_calls[0] == (
        aiohttp.WSCloseCode.GOING_AWAY,
        b"idle_recovery_failed",
    )

    # The new session replays the cached subscriptions and recovers.
    ws2 = await _join(env, session, 1)
    await until(lambda: SUBSCRIBES[-1] in ws2.sent, "subscriptions replayed")
    ws2.feed(_event("dev_data", SNAPSHOT))
    await until(lambda: env.status() == "healthy", "healthy again")
    assert env.stemp() == "20.0"

    await _stop(client)
