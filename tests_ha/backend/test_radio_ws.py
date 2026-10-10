"""RadioListener on real Home Assistant: station duties, push deltas, health.

The listener runs against a real ``StateCoordinator`` (its delta handler is
wrapped to record what was pushed), a real config entry runtime and the real
dispatcher. Only the radio link is faked (``tests_ha.fakes.radio_link``).
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
import pytest

from custom_components.termoweb.backend import radio_client as rc, radio_ws
from custom_components.termoweb.backend.radio import protocol as p
from custom_components.termoweb.backend.radio.link import RadioLinkError
from custom_components.termoweb.backend.radio_client import RadioClient
from custom_components.termoweb.backend.radio_ws import RadioListener
from custom_components.termoweb.const import BRAND_RADIO, signal_ws_status
from custom_components.termoweb.coordinator import StateCoordinator
from custom_components.termoweb.domain.ids import NodeType
from custom_components.termoweb.inventory import Inventory, build_node_inventory
from custom_components.termoweb.runtime import EntryRuntime
from tests_ha.fakes.radio_link import (
    CLOCK_ACCEPTED,
    HEATER,
    NET,
    POWER_RECORD_HEATING,
    POWER_RECORD_IDLE,
    POWER_REQUEST,
    PROGRAM_HOURLY,
    REGISTRATION,
    STATUS_SHORT,
    FakeRadioLink,
    received,
    received_ack,
)
from tests_ha.fakes.runtime import build_entry_runtime

DEV_ID = "aabbcc001122"
ENTRY_ID = "entry-radio"
NODES = [
    {"type": "htr", "addr": "6", "name": "Living room"},
    {"type": "acm", "addr": "7", "name": "Hall"},
]
DAY = [1] * 5 + [2] * 16 + [1] * 3
REFRESH = 120.0
STATUS_E6 = bytes.fromhex("B921252A0300CC2C2CF0100300FF")


class Sleeper:
    """Injected sleep that parks until the test releases it."""

    def __init__(self) -> None:
        """Start with no sleeps."""
        self.calls: list[float] = []
        self.pending: list[asyncio.Future[None]] = []

    async def __call__(self, seconds: float) -> None:
        """Record the sleep and park until released."""
        self.calls.append(seconds)
        future = asyncio.get_running_loop().create_future()
        self.pending.append(future)
        await future

    def release(self) -> None:
        """Wake every parked sleep."""
        pending, self.pending = self.pending, []
        for future in pending:
            if not future.done():
                future.set_result(None)


class Radio(SimpleNamespace):
    """The listener under test and everything around it."""

    listener: RadioListener
    client: RadioClient
    links: list[FakeRadioLink]
    coordinator: StateCoordinator
    deltas: list[Any]
    sleeper: Sleeper
    runtime: EntryRuntime
    statuses: list[dict[str, Any]]

    def changes_for(self, addr: str) -> list[dict[str, Any]]:
        """Return the settings changes pushed for ``addr``."""
        return [dict(d.changes) for d in self.deltas if d.node_id.addr == addr]

    def connected(self) -> bool:
        """Return the gateway connection flag in the domain state."""
        return self.coordinator.gateway_connected


async def settle(rounds: int = 30) -> None:
    """Let background tasks run."""

    for _ in range(rounds):
        await asyncio.sleep(0)


def pending_replies() -> list[asyncio.Task[Any]]:
    """Return station replies still running in the background."""
    return [
        task
        for task in asyncio.all_tasks()
        if not task.done()
        and getattr(task.get_coro(), "__qualname__", "") == "RadioListener._reply"
    ]


def build(hass: HomeAssistant, nodes: list[dict[str, str]] = NODES) -> Radio:
    """Return a listener on a real coordinator and config entry runtime."""

    links: list[FakeRadioLink] = []

    def factory(host, port, dialect, **kw):
        link = FakeRadioLink(host, port, dialect, **kw)
        link.reply(0xB8, STATUS_SHORT)
        link.reply(0xB0, PROGRAM_HOURLY)
        link.reply(0xBC, POWER_RECORD_IDLE)
        link.reply(0x51, CLOCK_ACCEPTED)
        link.reply(0x52, CLOCK_ACCEPTED)
        links.append(link)
        return link

    client = RadioClient(
        "radio.local", 2323, "B", nodes, network_id=NET, link_factory=factory
    )
    client.reply_timeout = 0.01
    inventory = Inventory(DEV_ID, build_node_inventory(nodes))
    coordinator = StateCoordinator(
        hass, client, 30, DEV_ID, {"dev_id": DEV_ID}, inventory, brand=BRAND_RADIO
    )
    deltas: list[Any] = []
    real_handle = coordinator.handle_ws_deltas

    def record(dev_id: str, batch, *, replace: bool = False) -> None:
        batch = tuple(batch)
        assert dev_id == DEV_ID and not replace
        deltas.extend(batch)
        real_handle(dev_id, batch, replace=replace)

    coordinator.handle_ws_deltas = record
    runtime = build_entry_runtime(
        hass=hass,
        entry_id=ENTRY_ID,
        dev_id=DEV_ID,
        inventory=inventory,
        coordinator=coordinator,
        brand=BRAND_RADIO,
    )
    sleeper = Sleeper()
    listener = RadioListener(
        hass,
        entry_id=ENTRY_ID,
        dev_id=DEV_ID,
        client=client,
        coordinator=coordinator,
        inventory=inventory,
        refresh_interval=REFRESH,
        sleep=sleeper,
        grant_settle_s=0,
    )
    statuses: list[dict[str, Any]] = []

    @callback
    def _on_status(payload: dict[str, Any]) -> None:
        statuses.append(payload)

    async_dispatcher_connect(hass, signal_ws_status(ENTRY_ID), _on_status)
    return Radio(
        listener=listener,
        client=client,
        links=links,
        coordinator=coordinator,
        deltas=deltas,
        sleeper=sleeper,
        runtime=runtime,
        statuses=statuses,
    )


@pytest.fixture
async def radio(hass: HomeAssistant):
    """Return the default listener harness; it is stopped afterwards."""
    harness = build(hass)
    yield harness
    await harness.listener.stop()


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
            coordinator=MagicMock(),
            inventory=None,
        )


async def test_start_connects_refreshes_and_reports_health(radio: Radio) -> None:
    """Start: connect, keepalive + status per heater, deltas, healthy tracker."""

    task = radio.listener.start()
    assert radio.listener.start() is task
    await settle()

    assert not task.done()
    link = radio.links[0]
    assert link.sent[:3] == [
        (6, bytes.fromhex("521A0A0905103409")),
        (6, b"\xb8"),
        (6, b"\xb0"),
    ]
    assert (7, bytes.fromhex("521A0A0905103409")) in link.sent
    assert radio.changes_for("6") == [
        {
            "units": "C",
            "ptemp": ["16.5", "18.5", "21.0"],
            "mode": "manual",
            "prog": DAY * 7,
            "state": "off",
            "mtemp": "22.0",
            "stemp": "22.0",
            "priority": 0,
        }
    ]
    assert radio.deltas[1].node_id.node_type is NodeType.ACCUMULATOR
    heater = radio.coordinator.domain_view.get_heater_state("htr", "6")
    assert heater is not None and heater.stemp == "22.0"
    assert radio.sleeper.calls == [REFRESH]

    tracker = radio.runtime.ws_trackers[DEV_ID]
    assert tracker.status == "healthy"
    assert tracker.payload_stale_after == 3 * REFRESH
    assert tracker.last_heartbeat_at is not None
    assert radio.connected()
    statuses = [payload["status"] for payload in radio.statuses]
    assert statuses[0] == "connected" and "healthy" in statuses
    assert any(payload.get("health_changed") for payload in radio.statuses)

    await radio.listener.stop()
    assert task.done()
    assert link.closes == 1 and link.listeners == []
    assert tracker.status == "stopped"
    assert not radio.connected()


async def test_registration_power_request_and_reports(radio: Radio) -> None:
    """Unsolicited heater frames get their station replies and become deltas."""

    radio.listener.start()
    await settle()
    link = radio.links[0]
    link.sent.clear()
    radio.deltas.clear()

    link.deliver(received(HEATER, REGISTRATION))
    link.deliver(received(HEATER, POWER_REQUEST))  # dialect B: granted, no power
    link.deliver(received(HEATER, bytes([p.OP_POWER_REQUEST, 0x1D, 0x66])))  # A form
    link.deliver(received(HEATER, bytes([p.OP_REPORT]) + STATUS_E6))
    wire = p._pack_slots([0] * 24 + DAY * 6)  # noqa: SLF001 - Sunday cold
    link.deliver(received(HEATER, bytes([p.OP_REPORT, 0xB1]) + wire))
    await settle()

    sent = [payload for _addr, payload in link.sent]
    assert sent.count(bytes.fromhex("511A0A0905103409")) == 1
    assert sent.count(b"\xbf\x01") == 2  # both power requests granted
    assert sent.count(b"\xbc") == 2  # then each heating state is read back
    assert sent.count(b"\x57\x55") == 2
    changes = radio.changes_for("6")
    assert {"max_power": 752.6} in changes
    assert {"state": "off", "mtemp": "22.0", "stemp": "22.0"} in changes
    assert any(
        c.get("mode") == "modified_auto" and c["mtemp"] == "20.4" for c in changes
    )
    assert {"prog": DAY * 6 + [0] * 24} in changes
    # The noted power fills in later status reads that lack it.
    settings = await radio.client.get_node_settings(DEV_ID, ("htr", "6"))
    assert settings["max_power"] == 752.6


async def test_granted_heater_reports_heating(radio: Radio) -> None:
    """After BF 01 the heater's power record flags heating; that becomes state on."""

    radio.listener.start()
    await settle()
    link = radio.links[0]
    link.reply(0xBC, POWER_RECORD_HEATING)
    radio.deltas.clear()

    link.deliver(received(HEATER, POWER_REQUEST))
    await settle()

    assert radio.changes_for("6") == [{"state": "on", "mtemp": "21.9", "stemp": "22.0"}]
    link.replies.pop(0xBC)
    radio.deltas.clear()
    link.deliver(received(HEATER, POWER_REQUEST))  # record read fails: no push
    await settle(60)
    assert radio.changes_for("6") == []


async def test_power_request_over_the_limit_switches_the_heater_off(
    radio: Radio,
) -> None:
    """A heater that heats over the power limit is acked, read, then switched off."""

    radio.listener.start()
    await settle()
    link = radio.links[0]
    await radio.client.set_power_limit(DEV_ID, power_limit=1000)
    radio.client.note_max_power(6, 1500.0)
    link.reply(0xBC, POWER_RECORD_HEATING)
    link.reply(0xB6, b"\xb7\x55")
    link.sent.clear()

    link.deliver(received(HEATER, POWER_REQUEST))
    await settle(60)

    assert [payload for _a, payload in link.sent] == [
        b"\xbf\x01",
        b"\xbc",
        b"\xb8",
        b"\xb8",
        bytes.fromhex("B621252A04"),
    ]
    assert radio.client.power.shed() == {6: 2}


async def test_energy_estimate_is_pushed_to_the_energy_coordinator(
    radio: Radio,
) -> None:
    """Each power-record read forwards the estimated Wh counter like a WS sample."""

    handler = radio.runtime.energy_coordinator.handle_ws_samples
    radio.client.note_max_power(6, 1500.0)
    radio.listener.start()
    await settle()
    link = radio.links[0]
    link.reply(0xBC, POWER_RECORD_HEATING)
    link.deliver(received(HEATER, POWER_REQUEST))
    await settle(60)

    assert handler.called
    dev_id, updates = handler.call_args.args[:2]
    assert dev_id == DEV_ID
    assert set(updates) == {"htr"} and "counter" in updates["htr"]["6"]


async def test_frames_without_usable_content(radio: Radio) -> None:
    """Undecodable reports are confirmed but push nothing; noise is ignored."""

    radio.listener.start()
    await settle()
    link = radio.links[0]
    link.sent.clear()
    radio.deltas.clear()

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
    # Same ids on a neighbouring network: no clock sync, grant or confirmation.
    foreign = bytes.fromhex("5678")
    link.deliver(received(HEATER, REGISTRATION, network_id=foreign))
    link.deliver(received(HEATER, POWER_REQUEST, network_id=foreign))
    link.deliver(received(HEATER, bytes([p.OP_REPORT]) + STATUS_E6, network_id=foreign))
    await settle()

    assert link.sent == [(6, b"\x57\x55"), (6, b"\x57\x55")]
    assert radio.deltas == []


async def test_shared_address_reports_as_heater(hass: HomeAssistant) -> None:
    """An address listed under several heater types pushes htr deltas."""

    harness = build(hass, [{"type": "acm", "addr": "6"}, {"type": "htr", "addr": "6"}])
    harness.listener.start()
    await settle()
    harness.deltas.clear()

    harness.links[0].deliver(received(HEATER, bytes([p.OP_REPORT]) + STATUS_E6))
    await settle()

    assert harness.deltas
    assert {d.node_id.node_type for d in harness.deltas} == {NodeType.HEATER}
    await harness.listener.stop()


async def test_failed_station_reply_is_logged_not_raised(
    radio: Radio, caplog: pytest.LogCaptureFixture
) -> None:
    """A heater that misses the BF grant just repeats its request."""

    task = radio.listener.start()
    await settle()
    radio.links[0].no_ack.add(HEATER)
    with caplog.at_level(logging.DEBUG, logger=radio_ws.__name__):
        radio.links[0].deliver(received(HEATER, POWER_REQUEST))
        await settle()
    assert "Radio station reply failed" in caplog.text
    assert not task.done()
    assert pending_replies() == []


async def test_stop_cancels_pending_station_replies(radio: Radio) -> None:
    """Replies still waiting for a heater are cancelled on stop."""

    radio.listener.start()
    await settle()
    radio.client.reply_timeout = 30.0
    radio.links[0].replies.pop(0x51)
    radio.links[0].deliver(received(HEATER, REGISTRATION))
    await settle()
    assert len(pending_replies()) == 1

    await radio.listener.stop()
    assert pending_replies() == []


async def test_disconnect_reports_down_then_reconnects_with_backoff(
    radio: Radio,
) -> None:
    """A dropped gateway is reported at once and reconnected after a backoff."""

    radio.listener.start()
    await settle()
    link = radio.links[0]

    link.drop()
    tracker = radio.runtime.ws_trackers[DEV_ID]
    assert tracker.status == "disconnected" and tracker.healthy_since is None
    assert not radio.connected()

    radio.sleeper.release()  # end of the refresh sleep: the loop sees the drop
    await settle()
    assert radio.sleeper.calls == [REFRESH, 5.0]
    assert link.listeners == []

    radio.sleeper.release()  # end of the backoff: reconnect
    await settle()
    assert link.connects == 2
    assert tracker.status == "healthy"
    assert radio.connected()
    assert radio.sleeper.calls == [REFRESH, 5.0, REFRESH]


async def test_connect_failures_back_off_progressively(radio: Radio) -> None:
    """Connect errors back off 5 s, 10 s, ...; a success resets the backoff."""

    link = await radio.client.async_connect()
    link.drop()
    link.connect_errors = [RadioLinkError("down"), RadioLinkError("down")]

    radio.listener.start()
    await settle()
    radio.sleeper.release()
    await settle()
    radio.sleeper.release()
    await settle()
    assert radio.sleeper.calls == [5.0, 10.0, REFRESH]

    link.drop()  # a later drop starts again at the first backoff step
    radio.sleeper.release()
    await settle()
    assert radio.sleeper.calls == [5.0, 10.0, REFRESH, 5.0]


async def test_refresh_skips_bad_addresses_and_silent_heaters(
    hass: HomeAssistant, caplog: pytest.LogCaptureFixture
) -> None:
    """A bad inventory address is skipped; a silent heater pushes nothing."""

    harness = build(hass, [*NODES, {"type": "htr", "addr": "300"}])
    await harness.client.async_connect()
    harness.links[0].no_ack.add(7)
    with caplog.at_level(logging.DEBUG, logger=radio_ws.__name__):
        harness.listener.start()
        await settle()

    assert "bad radio address" in caplog.text
    assert "Keepalive clock sync to heater 7 failed" in caplog.text
    assert harness.changes_for("7") == []
    assert harness.changes_for("6") != []
    await harness.listener.stop()


async def test_stop_without_start(radio: Radio) -> None:
    """Stopping an idle listener is safe and opens no link."""

    await radio.listener.stop()
    assert radio.links == []


async def test_unexpected_refresh_error_does_not_stop_the_loop(
    radio: Radio, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A bug in one heater's refresh or in the power check is logged; the loop goes on."""

    client = radio.client
    real_settings = client.get_node_settings

    async def broken_settings(dev_id, node):
        if node[1] == "6":
            raise ValueError("boom")
        return await real_settings(dev_id, node)

    async def broken_balance() -> None:
        raise ValueError("bang")

    monkeypatch.setattr(client, "get_node_settings", broken_settings)
    monkeypatch.setattr(client, "async_balance_power", broken_balance)
    with caplog.at_level(logging.ERROR, logger=radio_ws.__name__):
        task = radio.listener.start()
        await settle()
    assert "refresh of heater 6 failed unexpectedly" in caplog.text
    assert "power limit check failed unexpectedly" in caplog.text
    assert radio.changes_for("7") != []  # the next heater still refreshed
    assert not task.done()
    assert radio.sleeper.calls == [REFRESH]

    radio.links[0].sent.clear()
    radio.sleeper.release()  # the next iteration still runs its keepalives
    await settle()
    assert not task.done()
    assert radio.sleeper.calls == [REFRESH, REFRESH]
    assert (6, bytes.fromhex("521A0A0905103409")) in radio.links[0].sent


async def test_unexpected_grant_error_is_logged_and_the_loop_goes_on(
    radio: Radio, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A bug in a power grant is logged at ERROR; refreshes keep running."""

    task = radio.listener.start()
    await settle()

    async def broken_record(addr):
        raise ValueError("boom")

    monkeypatch.setattr(radio.client, "read_power_record", broken_record)
    with caplog.at_level(logging.ERROR, logger=radio_ws.__name__):
        radio.links[0].deliver(received(HEATER, POWER_REQUEST))
        await settle()
    assert "Radio station reply failed unexpectedly" in caplog.text
    assert pending_replies() == []

    radio.sleeper.release()
    await settle()
    assert not task.done()
    assert radio.sleeper.calls == [REFRESH, REFRESH]


async def test_cancellation_inside_a_refresh_still_stops_the_loop(
    radio: Radio, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CancelledError is not swallowed by the per-heater guard."""

    async def cancelled(dev_id, node):
        raise asyncio.CancelledError

    monkeypatch.setattr(radio.client, "get_node_settings", cancelled)
    task = radio.listener.start()
    await settle()
    assert task.cancelled()
