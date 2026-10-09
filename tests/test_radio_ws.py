"""Tests for RadioListener: station duties, push deltas, health and reconnect."""

from __future__ import annotations

import asyncio
from datetime import datetime
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from conftest import build_entry_runtime
from fake_radio_link import (
    CLOCK_ACCEPTED,
    HEATER,
    NET,
    POWER_REQUEST,
    PROGRAM_HOURLY,
    REGISTRATION,
    STATUS_SHORT,
    FakeRadioLink,
    received,
    received_ack,
)

from custom_components.termoweb.backend import radio_client as rc, radio_ws
from custom_components.termoweb.backend.radio import protocol as p
from custom_components.termoweb.backend.radio.link import RadioLinkError
from custom_components.termoweb.backend.radio_client import RadioClient
from custom_components.termoweb.backend.radio_ws import RadioListener
from custom_components.termoweb.domain.ids import NodeType
from custom_components.termoweb.inventory import Inventory, build_node_inventory

DEV_ID = "aabbcc001122"
NODES = [
    {"type": "htr", "addr": "6", "name": "Living room"},
    {"type": "acm", "addr": "7", "name": "Hall"},
]
DAY = [1] * 5 + [2] * 16 + [1] * 3
REFRESH = 120.0
STATUS_E6 = bytes.fromhex("B921252A0300CC2C2CF0100300FF")


class Coordinator:
    """Records deltas and gateway connection updates."""

    def __init__(self) -> None:
        self.deltas: list[Any] = []
        self.connections: list[dict[str, Any]] = []

    def handle_ws_deltas(self, dev_id: str, deltas, *, replace: bool = False) -> None:
        assert dev_id == DEV_ID and not replace
        self.deltas.extend(deltas)

    def update_gateway_connection(self, **kwargs: Any) -> None:
        self.connections.append(kwargs)

    def changes_for(self, addr: str) -> list[dict[str, Any]]:
        return [dict(d.changes) for d in self.deltas if d.node_id.addr == addr]


class Sleeper:
    """Injected sleep that parks until the test releases it."""

    def __init__(self) -> None:
        self.calls: list[float] = []
        self.pending: list[asyncio.Future[None]] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        future = asyncio.get_running_loop().create_future()
        self.pending.append(future)
        await future

    def release(self) -> None:
        pending, self.pending = self.pending, []
        for future in pending:
            if not future.done():
                future.set_result(None)


async def settle(rounds: int = 30) -> None:
    """Let background tasks run."""

    for _ in range(rounds):
        await asyncio.sleep(0)


def build(nodes=NODES):
    """Return (listener, client, links, coordinator, sleeper, runtime, dispatched)."""

    links: list[FakeRadioLink] = []

    def factory(host, port, dialect, **kw):
        link = FakeRadioLink(host, port, dialect, **kw)
        link.reply(0xB8, STATUS_SHORT)
        link.reply(0xB0, PROGRAM_HOURLY)
        link.reply(0x51, CLOCK_ACCEPTED)
        link.reply(0x52, CLOCK_ACCEPTED)
        links.append(link)
        return link

    client = RadioClient(
        "radio.local", 2323, "B", nodes, network_id=NET, link_factory=factory
    )
    client.reply_timeout = 0.01
    hass = SimpleNamespace(data={})
    inventory = Inventory(DEV_ID, build_node_inventory(nodes))
    runtime = build_entry_runtime(
        hass=hass, entry_id="entry", dev_id=DEV_ID, inventory=inventory, brand="radio"
    )
    coordinator = Coordinator()
    sleeper = Sleeper()
    listener = RadioListener(
        hass,
        entry_id="entry",
        dev_id=DEV_ID,
        client=client,
        coordinator=coordinator,
        inventory=inventory,
        refresh_interval=REFRESH,
        sleep=sleeper,
    )
    dispatched = MagicMock()
    listener._dispatcher_mock = dispatched  # noqa: SLF001
    return listener, client, links, coordinator, sleeper, runtime, dispatched


@pytest.fixture(autouse=True)
def _fixed_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the station clock."""

    monkeypatch.setattr(rc, "_local_now", lambda: datetime(2026, 10, 9, 16, 52, 9))


def test_requires_inventory() -> None:
    """The listener walks the immutable inventory, never its own node list."""

    with pytest.raises(TypeError, match="Inventory"):
        RadioListener(
            SimpleNamespace(data={}),
            entry_id="e",
            dev_id=DEV_ID,
            client=MagicMock(),
            coordinator=Coordinator(),
            inventory=None,
        )


@pytest.mark.asyncio
async def test_start_connects_refreshes_and_reports_health() -> None:
    """Start: connect, keepalive + status per heater, deltas, healthy tracker."""

    listener, client, links, coordinator, sleeper, runtime, dispatched = build()
    task = listener.start()
    assert listener.start() is task
    await settle()

    assert listener.is_running()
    link = links[0]
    assert link.sent[:3] == [
        (6, bytes.fromhex("521A0A0905103409")),
        (6, b"\xb8"),
        (6, b"\xb0"),
    ]
    assert (7, bytes.fromhex("521A0A0905103409")) in link.sent
    assert coordinator.changes_for("6") == [
        {
            "units": "C",
            "ptemp": ["16.5", "18.5", "21.0"],
            "mode": "manual",
            "prog": DAY * 7,
        }
    ]
    assert coordinator.deltas[1].node_id.node_type is NodeType.ACCUMULATOR
    assert sleeper.calls == [REFRESH]

    tracker = runtime.ws_trackers[DEV_ID]
    assert tracker.status == "healthy"
    assert tracker.payload_stale_after == 3 * REFRESH
    assert tracker.last_heartbeat_at is not None
    assert coordinator.connections[-1]["connected"] is True
    statuses = [call.args[2]["status"] for call in dispatched.call_args_list]
    assert statuses[0] == "connected" and "healthy" in statuses
    assert any(call.args[2].get("health_changed") for call in dispatched.call_args_list)

    await listener.stop()
    assert not listener.is_running()
    assert link.closes == 1 and link.listeners == []
    assert tracker.status == "stopped"
    assert coordinator.connections[-1]["connected"] is False


@pytest.mark.asyncio
async def test_registration_power_request_and_reports() -> None:
    """Unsolicited heater frames get their station replies and become deltas."""

    listener, client, links, coordinator, sleeper, runtime, _ = build()
    listener.start()
    await settle()
    link = links[0]
    link.sent.clear()
    coordinator.deltas.clear()

    link.deliver(received(HEATER, REGISTRATION))
    link.deliver(received(HEATER, POWER_REQUEST))
    link.deliver(received(HEATER, bytes([p.OP_REPORT]) + STATUS_E6))
    wire = p._pack_slots([0] * 24 + DAY * 6)  # noqa: SLF001 - Sunday cold
    link.deliver(received(HEATER, bytes([p.OP_REPORT, 0xB1]) + wire))
    await settle()

    assert link.sent == [
        (6, bytes.fromhex("511A0A0905103409")),
        (6, b"\xbf\x01"),
        (6, b"\x57\x55"),
        (6, b"\x57\x55"),
    ]
    changes = coordinator.changes_for("6")
    assert changes[0] == {"max_power": 1143.3}
    assert changes[1]["mode"] == "modified_auto" and changes[1]["mtemp"] == "20.4"
    assert changes[2] == {"prog": DAY * 6 + [0] * 24}
    # The noted power fills in later status reads that lack it.
    assert (await client.get_node_settings(DEV_ID, ("htr", "6")))["max_power"] == 1143.3

    await listener.stop()


@pytest.mark.asyncio
async def test_frames_without_usable_content() -> None:
    """Undecodable reports are confirmed but push nothing; noise is ignored."""

    listener, client, links, coordinator, sleeper, runtime, _ = build()
    listener.start()
    await settle()
    link = links[0]
    link.sent.clear()
    coordinator.deltas.clear()

    link.deliver(received(HEATER, bytes([p.OP_REPORT, 0xB9, 1, 2])))  # short report
    split = [2] * (48 * 7)
    split[1] = 0
    link.deliver(
        received(HEATER, bytes([p.OP_REPORT, 0xB1]) + p._pack_slots(split))  # noqa: SLF001
    )
    link.deliver(received(HEATER, bytes([p.OP_POWER_REQUEST, 1, 2, 3, 4])))  # odd BE
    link.deliver(received(HEATER, b"", tag=p.ROUTE_PROBE_TAG))  # route probe
    link.deliver(received(HEATER, b"\xc2\xf0\x39"))  # unknown unsolicited
    link.deliver(received_ack(HEATER))  # link ack
    link.deliver(received(HEATER, REGISTRATION, dst=2))  # for another station
    link.deliver(received(9, REGISTRATION))  # unknown node
    await settle()

    assert link.sent == [(6, b"\x57\x55"), (6, b"\x57\x55")]
    assert coordinator.deltas == []

    await listener.stop()


@pytest.mark.asyncio
async def test_frame_before_connect_is_ignored() -> None:
    """A frame can only be handled once the client has a link."""

    listener, *_ = build()
    listener._on_frame(received(HEATER, REGISTRATION))  # noqa: SLF001


@pytest.mark.asyncio
async def test_failed_station_reply_is_logged_not_raised(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A heater that misses the BF grant just repeats its request."""

    listener, client, links, coordinator, sleeper, runtime, _ = build()
    listener.start()
    await settle()
    links[0].no_ack.add(HEATER)
    with caplog.at_level(logging.DEBUG, logger=radio_ws.__name__):
        links[0].deliver(received(HEATER, POWER_REQUEST))
        await settle()
    assert "Radio station reply failed" in caplog.text
    await listener.stop()


@pytest.mark.asyncio
async def test_stop_cancels_pending_station_replies() -> None:
    """Replies still waiting for a heater are cancelled on stop."""

    listener, client, links, coordinator, sleeper, runtime, _ = build()
    listener.start()
    await settle()
    client.reply_timeout = 30.0
    links[0].replies.pop(0x51)
    links[0].deliver(received(HEATER, REGISTRATION))
    await settle()
    assert len(listener._jobs) == 1  # noqa: SLF001

    await listener.stop()
    assert listener._jobs == set()  # noqa: SLF001


@pytest.mark.asyncio
async def test_disconnect_reports_down_then_reconnects_with_backoff() -> None:
    """A dropped gateway is reported at once and reconnected after a backoff."""

    listener, client, links, coordinator, sleeper, runtime, _ = build()
    listener.start()
    await settle()
    link = links[0]

    link.drop()
    tracker = runtime.ws_trackers[DEV_ID]
    assert tracker.status == "disconnected" and tracker.healthy_since is None
    assert coordinator.connections[-1]["connected"] is False

    sleeper.release()  # end of the refresh sleep: the loop sees the drop
    await settle()
    assert sleeper.calls == [REFRESH, 5.0]
    assert link.listeners == []

    sleeper.release()  # end of the backoff: reconnect
    await settle()
    assert link.connects == 2
    assert tracker.status == "healthy"
    assert sleeper.calls == [REFRESH, 5.0, REFRESH]
    await listener.stop()


@pytest.mark.asyncio
async def test_connect_failures_back_off_progressively() -> None:
    """Connect errors back off 5 s, 10 s, ... and reset after a success."""

    listener, client, links, coordinator, sleeper, runtime, _ = build()
    link = await client.async_connect()
    link.drop()
    link.connect_errors = [RadioLinkError("down"), RadioLinkError("down")]

    listener.start()
    await settle()
    sleeper.release()
    await settle()
    sleeper.release()
    await settle()

    assert sleeper.calls == [5.0, 10.0, REFRESH]
    assert listener._backoff_idx == 0  # noqa: SLF001
    assert [listener._next_backoff() for _ in range(6)] == [  # noqa: SLF001
        5.0,
        10.0,
        30.0,
        120.0,
        300.0,
        300.0,
    ]
    await listener.stop()


@pytest.mark.asyncio
async def test_refresh_skips_bad_addresses_and_silent_heaters(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A bad inventory address is skipped; a silent heater pushes nothing."""

    nodes = [*NODES, {"type": "htr", "addr": "300"}]
    listener, client, links, coordinator, sleeper, runtime, _ = build(nodes)
    await client.async_connect()
    links[0].no_ack.add(7)
    with caplog.at_level(logging.DEBUG, logger=radio_ws.__name__):
        listener.start()
        await settle()

    assert "bad radio address" in caplog.text
    assert "Keepalive clock sync to heater 7 failed" in caplog.text
    assert coordinator.changes_for("7") == []
    assert coordinator.changes_for("6") != []
    await listener.stop()


@pytest.mark.asyncio
async def test_stop_without_start() -> None:
    """Stopping an idle listener is safe."""

    listener, client, links, *_ = build()
    await listener.stop()
    assert links == []
    assert not listener.is_running()


def test_node_type_prefers_heater_for_shared_addresses() -> None:
    """An address listed under several heater types resolves to htr."""

    listener, *_ = build([{"type": "acm", "addr": "6"}, {"type": "htr", "addr": "6"}])
    assert listener._node_type_for("6") == "htr"  # noqa: SLF001
    assert listener._node_type_for("8") is None  # noqa: SLF001
