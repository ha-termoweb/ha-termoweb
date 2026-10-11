"""Radio heaters put into pairing mode are paired by themselves inside the window."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import timedelta
import logging
from types import SimpleNamespace
from typing import Any

from homeassistant.components import persistent_notification
from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.entity import EntityCategory
from homeassistant.util import dt as dt_util
import pytest
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_fire_time_changed,
)

from custom_components.termoweb import (
    binary_sensor as binary_sensor_platform,
    radio_autopair,
    radio_discovery,
)
from custom_components.termoweb.backend import radio_client as radio_client_module
from custom_components.termoweb.backend.radio import (
    DIALECT_A,
    DIALECT_B,
    ReceivedFrame,
    build_frame,
    decode,
    protocol,
)
from custom_components.termoweb.backend.radio.pairing import (
    NoFreeAddressError,
    PairedHeater,
)
from custom_components.termoweb.backend.radio_client import RadioCommandError
from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.identifiers import build_gateway_entity_unique_id
from custom_components.termoweb.radio_pairing import async_site_network_id
from custom_components.termoweb.runtime import require_runtime
from tests.fakes.radio_link import (
    NET,
    REGISTRATION,
    STATUS_SHORT,
    FakeRadioLink,
    received,
)
from tests.fakes.radio_setup import record_reloads
from tests.fakes.sensors import fake_radio_links, setup_entry

# Synthetic identifiers only.
KNOWN = {"type": "htr", "addr": "6", "name": "Heater S000006", "serial": "S000006"}
IDENTITY_A = bytes(range(0x10, 0x1C))  # synthetic 12-byte dialect-A identity
TOKEN_A = IDENTITY_A[-4:].hex()  # its serial token
VERDICT_B4 = bytes.fromhex("B555")
VERDICT_B6 = bytes.fromhex("B755")


def _identity(serial: str) -> bytes:
    """Return a dialect-B short identity reply carrying ``serial``."""
    return bytes.fromhex("5B550105010D004100400F") + serial.encode()


class Heaters:
    """Heaters on the air: stored ones answer; one in pairing mode takes its id."""

    def __init__(
        self, serials: dict[int, str | None], pairing: list[str | None]
    ) -> None:
        """``serials``: answering heaters; ``pairing``: serials of heaters to pair."""
        self.serials = dict(serials)
        self.pairing = list(pairing)
        self.assigned: list[int] = []

    def __call__(self, link: FakeRadioLink) -> None:
        """Script ``link``: answer reads, take id assignments."""

        def _on_send(dst: int, payload: bytes) -> None:
            for opcode in (0x5A, 0xB8, 0xB4, 0xB6):
                link.replies.pop(opcode, None)
            if dst == 0xFF and len(payload) == 1:  # tag-04 id assignment
                self.assigned.append(payload[0])
                self.serials[payload[0]] = self.pairing.pop(0)
                return
            if dst not in self.serials:
                return
            link.reply(0xB8, STATUS_SHORT)
            link.reply(0xB4, VERDICT_B4)  # dialect A mode write
            link.reply(0xB6, VERDICT_B6)
            if (serial := self.serials[dst]) is not None:
                link.reply(0x5A, _identity(serial))

        link.on_send = _on_send


def announce_b(dst: int = 1) -> Any:
    """Return a dialect-B pairing sweep frame from FF on net 0000."""
    return received(0xFF, b"\x55", dst=dst, tag=0x03, network_id=b"\x00\x00")


def announce_a(dst: int = 1, *, src: int = 0xFF) -> ReceivedFrame:
    """Return a dialect-A identity announcement (src != FF: a relayed copy)."""
    air = build_frame(
        DIALECT_A,
        src,
        dst,
        b"\x77" + IDENTITY_A,
        path=(0xFF, dst, 1, 1, 1),
        network_id=b"\x00\x00" if src == 0xFF else NET,  # unjoined: 00 00
    )
    return ReceivedFrame(decode(DIALECT_A, air), -60.0, 0, 1000)


def _entry(
    *, dialect: str = "B", nodes: list[Any] | None = None, **options: Any
) -> MockConfigEntry:
    """Return a radio entry with ``nodes`` (default: one known heater)."""
    data = {
        "brand": "radio",
        "host": "10.0.0.5",
        "port": 2323,
        "dialect": dialect,
        "network_id": "1234",
        "nodes": [KNOWN] if nodes is None else nodes,
    }
    return MockConfigEntry(
        domain=DOMAIN,
        minor_version=2,
        unique_id="radio:test",
        data=data,
        options=options,
    )


def _age(entry: MockConfigEntry) -> None:
    """Make ``entry`` an old one: its first-setup window is long over."""
    object.__setattr__(entry, "created_at", dt_util.utcnow() - timedelta(hours=1))


def _notifications(hass: HomeAssistant) -> dict[str, Any]:
    """Return the persistent notifications by id."""
    return persistent_notification._async_get_or_create_notifications(hass)


def _gateway_entity(hass: HomeAssistant, entry: Any, domain: str, key: str) -> str:
    """Return the entity id of the gateway entity ``key``."""
    dev_id = require_runtime(hass, entry.entry_id).dev_id
    entity_id = er.async_get(hass).async_get_entity_id(
        domain, DOMAIN, build_gateway_entity_unique_id(dev_id, key)
    )
    assert entity_id is not None
    return entity_id


async def _settle(hass: HomeAssistant) -> None:
    """Wait for background work (real-clock radio timeouts included)."""
    async with asyncio.timeout(10):
        while hass._background_tasks:
            await asyncio.sleep(0.01)
    await hass.async_block_till_done()


async def _until(check: Callable[[], bool]) -> None:
    """Wait until ``check()`` is true."""
    async with asyncio.timeout(5):
        while not check():
            await asyncio.sleep(0.005)


async def _after_debounce(hass: HomeAssistant) -> None:
    """Let the found-heater debounce pass and the batch finish."""
    async_fire_time_changed(
        hass, dt_util.utcnow() + timedelta(seconds=radio_discovery.FOUND_DEBOUNCE_S + 1)
    )
    await _settle(hass)


PAIR_RUNS: list[int] = []  # one entry per pairing exchange started


async def _quiet(link: FakeRadioLink, client: Any) -> None:
    """Wait until the listener's first refresh has let go of the radio."""
    async with asyncio.timeout(5):
        while True:
            sent = len(link.sent)
            await asyncio.sleep(0.05)
            if not client._exchange_lock.locked() and len(link.sent) == sent:
                return


async def _pair(hass: HomeAssistant, link: FakeRadioLink, announce: Any) -> None:
    """A heater sweeps: the exchange starts, then the frame to us is answered."""
    runs = len(PAIR_RUNS)
    link.deliver(announce(0x18))
    await _until(lambda: len(PAIR_RUNS) > runs)
    link.deliver(announce(1))
    await _settle(hass)


async def _unload(hass: HomeAssistant, entry: MockConfigEntry) -> None:
    """Unload the entry so no radio task outlives the test."""
    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()


class Later:
    """Moves the pairing window's clock forward (radio timeouts keep real time)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Patch the window module's clock."""
        self.offset = timedelta()
        monkeypatch.setattr(
            radio_autopair,
            "dt_util",
            SimpleNamespace(utcnow=lambda: dt_util.utcnow() + self.offset),
        )

    async def tick(self, hass: HomeAssistant, delta: timedelta) -> None:
        """Move ``delta`` forward and fire the timers that are due."""
        self.offset += delta
        async_fire_time_changed(hass, dt_util.utcnow() + self.offset)
        await hass.async_block_till_done()


@pytest.fixture
def later(monkeypatch: pytest.MonkeyPatch) -> Later:
    """Return the window clock mover."""
    return Later(monkeypatch)


@pytest.fixture(autouse=True)
def _no_scan_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake links are instant: no on-air gap between scanned addresses."""
    monkeypatch.setattr(radio_discovery, "SCAN_GAP_S", 0.0001)


@pytest.fixture(autouse=True)
def _count_pairing_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Note each pairing exchange as it starts listening (under the exchange lock)."""
    real = radio_client_module.pair_heaters
    PAIR_RUNS.clear()

    async def _pair_heaters(*args: Any, **kwargs: Any) -> Any:
        PAIR_RUNS.append(1)
        return await real(*args, **kwargs)

    monkeypatch.setattr(radio_client_module, "pair_heaters", _pair_heaters)


# --- the pairing window -------------------------------------------------------------


async def test_window_is_open_for_30_minutes_after_the_entry_is_created(
    hass: HomeAssistant, later: Later
) -> None:
    """A new entry pairs heaters for 30 minutes; the sensor then turns off."""
    entry = _entry()
    with fake_radio_links(Heaters({6: "S000006"}, [])):
        await setup_entry(hass, entry)
        sensor = _gateway_entity(hass, entry, "binary_sensor", "pairing_open")
        state = hass.states.get(sensor)
        assert state.state == STATE_ON
        end = entry.created_at + timedelta(minutes=30)
        assert state.attributes["ends_at"] == end.isoformat()

        await later.tick(hass, timedelta(minutes=30, seconds=1))

        state = hass.states.get(sensor)
        assert state.state == STATE_OFF and state.attributes["ends_at"] is None
        await _unload(hass, entry)


async def test_button_opens_the_window_for_10_minutes(
    hass: HomeAssistant, later: Later
) -> None:
    """Pressing "Pair new heaters" opens the window; it closes 10 minutes later."""
    entry = _entry()
    _age(entry)
    with fake_radio_links(Heaters({6: "S000006"}, [])):
        await setup_entry(hass, entry)
        sensor = _gateway_entity(hass, entry, "binary_sensor", "pairing_open")
        button = _gateway_entity(hass, entry, "button", "pair_heaters")
        registry = er.async_get(hass)
        assert registry.async_get(button).entity_category is EntityCategory.CONFIG
        assert hass.states.get(sensor).state == STATE_OFF

        await hass.services.async_call(
            "button", "press", {"entity_id": button}, blocking=True
        )
        state = hass.states.get(sensor)
        assert state.state == STATE_ON
        end = dt_util.parse_datetime(state.attributes["ends_at"])
        assert end - dt_util.utcnow() > timedelta(minutes=9, seconds=59)

        await later.tick(hass, timedelta(minutes=10, seconds=1))
        assert hass.states.get(sensor).state == STATE_OFF
        await _unload(hass, entry)


async def test_unload_cancels_the_window_timer(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The timer that closes the window does not outlive the entry."""
    timers: list[dict[str, bool]] = []
    real_track = binary_sensor_platform.async_track_point_in_utc_time

    def _track(*args: Any) -> Callable[[], None]:
        unsub = real_track(*args)
        timers.append({"cancelled": False})
        record = timers[-1]

        def _cancel() -> None:
            record["cancelled"] = True
            unsub()

        return _cancel

    monkeypatch.setattr(binary_sensor_platform, "async_track_point_in_utc_time", _track)
    entry = _entry()
    with fake_radio_links(Heaters({6: "S000006"}, [])):
        await setup_entry(hass, entry)
        assert timers == [{"cancelled": False}]
        await _unload(hass, entry)
    assert timers == [{"cancelled": True}]


async def test_listen_only_and_cloud_entries_have_no_pairing_entities(
    hass: HomeAssistant,
) -> None:
    """Only a radio entry that may transmit gets the button and the sensor."""
    runtime = SimpleNamespace(
        backend=SimpleNamespace(capabilities=SimpleNamespace(heater_discovery=True)),
        client=SimpleNamespace(listen_only=True),
    )
    assert not radio_autopair.pairing_supported(runtime)
    runtime.client = SimpleNamespace(listen_only=False)
    assert radio_autopair.pairing_supported(runtime)
    runtime.backend.capabilities.heater_discovery = False
    assert not radio_autopair.pairing_supported(runtime)


async def test_removing_the_entry_forgets_its_window(hass: HomeAssistant) -> None:
    """A removed entry leaves no pairing window behind."""
    entry = _entry()
    _age(entry)
    entry.add_to_hass(hass)
    radio_autopair.async_open_pairing_window(hass, entry)
    assert radio_autopair.pairing_window_end(hass, entry) is not None
    await hass.config_entries.async_remove(entry.entry_id)
    assert entry.entry_id not in hass.data[radio_autopair.PAIRING_WINDOWS]


# --- pairing on a running entry --------------------------------------------------


async def test_heater_announcing_in_the_window_is_paired_and_added_dialect_b(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dialect B: lowest free id, clock sent, node stored, one reload, a note."""
    reloads = record_reloads(monkeypatch, hass)
    entry = _entry()
    heaters = Heaters({6: "S000006"}, ["S000002"])
    with fake_radio_links(heaters) as links:
        await setup_entry(hass, entry)
        await _pair(hass, links[0], announce_b)

        assert heaters.assigned == [2]
        to_2 = [payload for dst, payload in links[0].sent if dst == 2]
        assert to_2[0] == b"\xb8" and to_2[1] == b"\x5a"  # confirm, identity
        clock = protocol.sync_clock(dt_util.now(), True, DIALECT_B)
        assert to_2[2][:2] == clock[:2]  # restore: the clock
        assert reloads == []  # the batch waits for more heaters

        await _after_debounce(hass)
        assert entry.data["nodes"] == [
            KNOWN,
            {"type": "htr", "addr": "2", "name": "Heater S000002", "serial": "S000002"},
        ]
        assert reloads == [entry.entry_id]
        note = _notifications(hass)[f"{DOMAIN}_{entry.entry_id}_heaters_found"]
        assert "Found TermoWeb heater S000002" in note["message"]
        await _unload(hass, entry)


async def test_heaters_paired_back_to_back_get_their_own_ids_and_one_reload(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An id handed out since setup is not handed out again before the reload."""
    reloads = record_reloads(monkeypatch, hass)
    monkeypatch.setattr(radio_autopair, "RETRY_PAUSE_S", 0.0)
    entry = _entry()
    heaters = Heaters({6: "S000006"}, ["S000002", "S000003"])
    with fake_radio_links(heaters) as links:
        await setup_entry(hass, entry)
        await _pair(hass, links[0], announce_b)
        await _pair(hass, links[0], announce_b)
        await _after_debounce(hass)

        assert heaters.assigned == [2, 3]
        assert [node["addr"] for node in entry.data["nodes"]] == ["6", "2", "3"]
        assert reloads == [entry.entry_id]
        await _unload(hass, entry)


async def test_heater_announcing_in_the_window_is_paired_and_added_dialect_a(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dialect A: the serial comes from the announcement; relayed copies are ignored."""
    reloads = record_reloads(monkeypatch, hass)
    entry = _entry(dialect="A")
    heaters = Heaters({6: "S000006"}, [None])
    with fake_radio_links(heaters) as links:
        await setup_entry(hass, entry)
        links[0].deliver(announce_a(src=9))  # relayed by a paired heater
        await _settle(hass)
        assert heaters.assigned == []

        await _pair(hass, links[0], announce_a)
        await _after_debounce(hass)

        assert heaters.assigned == [2]
        assert entry.data["nodes"][-1] == {
            "type": "htr",
            "addr": "2",
            "name": f"Heater {TOKEN_A}",
            "serial": TOKEN_A,
        }
        assert reloads == [entry.entry_id]
        await _unload(hass, entry)


async def test_announcement_outside_the_window_is_ignored(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A neighbour's heater in pairing mode is never answered outside the window."""
    caplog.set_level(logging.DEBUG, logger="custom_components.termoweb")
    reloads = record_reloads(monkeypatch, hass)
    entry = _entry()
    _age(entry)
    with fake_radio_links(Heaters({6: "S000006"}, ["S000002"])) as links:
        await setup_entry(hass, entry)
        links[0].deliver(announce_b(0x18))
        links[0].deliver(announce_b(1))
        await _after_debounce(hass)

        assert not [dst for dst, _payload in links[0].sent if dst in (0xFF, 2)]
        assert entry.data["nodes"] == [KNOWN] and reloads == []
        assert "the pairing window is closed" in caplog.text
        await _unload(hass, entry)


async def test_known_dialect_a_heater_is_paired_back_to_its_address(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stored heater re-entering pairing mode keeps its id; its settings return."""
    reloads = record_reloads(monkeypatch, hass)
    known = {"type": "htr", "addr": "6", "name": "Lounge", "serial": TOKEN_A}
    entry = _entry(dialect="A", nodes=[known], radio_restore={"6": {"mode": "off"}})
    heaters = Heaters({}, [None])
    with fake_radio_links(heaters) as links:
        await setup_entry(hass, entry)
        await _pair(hass, links[0], announce_a)
        await _after_debounce(hass)

        assert heaters.assigned == [6]
        assert any(dst == 6 and p[0] == 0xB4 for dst, p in links[0].sent)
        assert entry.data["nodes"] == [known]  # no duplicate
        assert entry.options["radio_restore"] == {}
        assert reloads == []
        await _unload(hass, entry)


async def test_known_dialect_b_heater_moves_to_its_new_id_with_its_settings(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dialect B has no identity first: the stored node moves, nothing is added."""
    reloads = record_reloads(monkeypatch, hass)
    entry = _entry(radio_restore={"6": {"mode": "off"}})
    heaters = Heaters({}, ["S000006"])
    with fake_radio_links(heaters) as links:
        await setup_entry(hass, entry)
        await _pair(hass, links[0], announce_b)
        await _after_debounce(hass)

        assert heaters.assigned == [2]
        assert any(dst == 2 and p[0] == 0xB6 for dst, p in links[0].sent)
        assert entry.data["nodes"] == [{**KNOWN, "addr": "2"}]
        assert entry.options["radio_restore"] == {}
        assert reloads == [entry.entry_id]
        assert f"{DOMAIN}_{entry.entry_id}_heaters_found" not in _notifications(hass)
        await _unload(hass, entry)


async def test_window_survives_the_reload_so_heaters_pair_one_after_another(
    hass: HomeAssistant,
) -> None:
    """Button pressed once: two heaters paired with a real reload in between."""
    entry = _entry()
    _age(entry)
    heaters = Heaters({6: "S000006"}, ["S000002", "S000003"])
    with fake_radio_links(heaters) as links:
        await setup_entry(hass, entry)
        button = _gateway_entity(hass, entry, "button", "pair_heaters")
        await hass.services.async_call(
            "button", "press", {"entity_id": button}, blocking=True
        )
        await _pair(hass, links[0], announce_b)
        await _after_debounce(hass)  # stores heater 2 and reloads
        await _settle(hass)
        assert entry.state is ConfigEntryState.LOADED and len(links) == 2

        sensor = _gateway_entity(hass, entry, "binary_sensor", "pairing_open")
        assert hass.states.get(sensor).state == STATE_ON
        await _pair(hass, links[1], announce_b)
        await _after_debounce(hass)
        await _settle(hass)

        assert heaters.assigned == [2, 3]
        assert [node["serial"] for node in entry.data["nodes"]] == [
            "S000006",
            "S000002",
            "S000003",
        ]
        await _unload(hass, entry)


async def test_pairing_holds_the_exchange_lock_only_for_one_short_attempt(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Listening is passive; an unanswered attempt ends soon and pauses retries."""
    monkeypatch.setattr(radio_autopair, "PAIR_ATTEMPT_S", 0.2)
    entry = _entry()
    with fake_radio_links(Heaters({6: "S000006"}, ["S000002"])) as links:
        await setup_entry(hass, entry)
        client = require_runtime(hass, entry.entry_id).client
        lock = client._exchange_lock
        await _quiet(links[0], client)
        assert not lock.locked()  # the listener hears announcements without it

        loop = asyncio.get_running_loop()
        links[0].deliver(announce_b(0x18))  # the sweep never reaches us
        await _until(lock.locked)
        start = loop.time()
        await _until(lambda: not lock.locked())
        assert loop.time() - start < 1.0
        assert radio_autopair.PAIR_ATTEMPT_S * 4 < 120  # far below the keepalive
        await _settle(hass)

        listeners = len(links[0].listeners)
        links[0].deliver(announce_b(0x18))  # within the retry pause: ignored
        await hass.async_block_till_done()
        assert len(links[0].listeners) == listeners and not lock.locked()
        assert not [dst for dst, _payload in links[0].sent if dst == 0xFF]
        await _unload(hass, entry)


async def test_manual_pairing_or_a_network_move_keeps_auto_pairing_away(
    hass: HomeAssistant,
) -> None:
    """While a pairing run holds the entry's pairing lock, announcements wait."""
    entry = _entry()
    with fake_radio_links(Heaters({6: "S000006"}, ["S000002"])) as links:
        await setup_entry(hass, entry)
        runtime = require_runtime(hass, entry.entry_id)
        async with runtime.radio_pairing_lock:
            links[0].deliver(announce_b(1))
            await hass.async_block_till_done()
        assert PAIR_RUNS == []
        assert not [dst for dst, _payload in links[0].sent if dst == 0xFF]
        await _unload(hass, entry)


# --- the pairer's edge cases ----------------------------------------------------------


class _Finder:
    """Records what the pairer hands over."""

    def __init__(self) -> None:
        self.paired: list[PairedHeater] = []

    def async_paired(self, heater: PairedHeater) -> None:
        self.paired.append(heater)


class _Client:
    """A radio client whose pairing and restore outcomes are scripted."""

    def __init__(self, pair: Any, restore: Exception | None = None) -> None:
        self.pair, self.restore_error = pair, restore
        self.calls: list[Any] = []

    async def async_pair(self, window_s: float, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self.pair, Exception):
            raise self.pair
        return self.pair

    async def async_restore(self, addr: int, **snapshot: Any) -> None:
        if self.restore_error is not None:
            raise self.restore_error

    def manual_setpoint(self, addr: int) -> None:
        return None


def _pairer(hass: HomeAssistant, client: _Client, **options: Any) -> Any:
    """Return a pairer for a stored entry with heater 6 (synthetic)."""
    entry = _entry(**options)
    entry.add_to_hass(hass)
    runtime = SimpleNamespace(
        config_entry=entry,
        client=client,
        radio_pairing_lock=asyncio.Lock(),
        coordinator=SimpleNamespace(
            domain_view=SimpleNamespace(get_heater_state=lambda *_args: None)
        ),
    )
    finder = _Finder()
    return radio_autopair.HeaterPairer(hass, runtime, finder), finder, entry


async def test_failed_pairing_logs_and_pauses_before_answering_again(
    hass: HomeAssistant, caplog: pytest.LogCaptureFixture
) -> None:
    """A pairing error is logged; the next announcement waits for the pause."""
    client = _Client(NoFreeAddressError("no free radio id"))
    pairer, finder, _entry_ = _pairer(hass, client)
    announcement = SimpleNamespace(identity=None)
    pairer.async_announced(announcement)
    await hass.async_block_till_done()
    pairer.async_announced(announcement)
    await hass.async_block_till_done()
    assert len(client.calls) == 1 and finder.paired == []
    assert "Pairing the heater failed: no free radio id" in caplog.text


async def test_restore_failure_keeps_the_settings_for_the_new_id(
    hass: HomeAssistant, caplog: pytest.LogCaptureFixture
) -> None:
    """Settings that could not be restored stay saved, under the heater's new id."""
    heater = PairedHeater(2, None, serial="S000006")
    client = _Client([heater], restore=RadioCommandError("no ack"))
    pairer, finder, entry = _pairer(hass, client, radio_restore={"6": {"mode": "off"}})
    pairer.async_announced(SimpleNamespace(identity=None))
    await hass.async_block_till_done()
    assert finder.paired == [heater]
    assert entry.options["radio_restore"] == {"2": {"mode": "off"}}
    assert "Restoring heater 2 failed" in caplog.text


async def test_stopped_pairer_and_unknown_serials(hass: HomeAssistant) -> None:
    """A stopped pairer ignores announcements; odd stored addresses are skipped."""
    client = _Client([])
    pairer, _finder, entry = _pairer(hass, client)
    pairer.async_stop()
    pairer.async_announced(SimpleNamespace(identity=None))
    await hass.async_block_till_done()
    assert client.calls == []
    bad = _entry(nodes=[{"type": "htr", "addr": "x", "serial": "S000009"}])
    assert radio_autopair._stored_addr(bad, "S000009") is None
    assert radio_autopair._stored_addr(entry, "S000099") is None
    assert radio_autopair._stored_addr(entry, None) is None


async def test_restore_failure_of_a_new_heater_saves_nothing(
    hass: HomeAssistant,
) -> None:
    """A new heater has no saved settings: a failed clock sync changes no options."""
    heater = PairedHeater(2, None, serial="S000002")
    client = _Client([heater], restore=RadioCommandError("no ack"))
    pairer, finder, entry = _pairer(hass, client)
    pairer.async_announced(SimpleNamespace(identity=None))
    await hass.async_block_till_done()
    assert finder.paired == [heater]
    assert "radio_restore" not in entry.options


# --- an entry that has no heater network yet ------------------------------------------


def _learning_entry() -> MockConfigEntry:
    """Return a new radio entry that has not learned its heater network yet."""
    return MockConfigEntry(
        domain=DOMAIN,
        minor_version=6,
        unique_id="radio:test",
        title="Radio gateway (10.0.0.5)",
        data={"brand": "radio", "host": "10.0.0.5", "port": 2323, "nodes": []},
    )


async def test_new_gateway_pairs_a_heater_of_another_installation(
    hass: HomeAssistant,
) -> None:
    """Owner's case: no network yet, the only heater is idle on another network.

    The entry still runs (button and window present) on the site network; the
    heater put into pairing mode joins it, the network is stored, and the
    heater is added with its serial.
    """
    entry = _learning_entry()
    _age(entry)  # an entry added long ago: the learning setup reopens the window
    heaters = Heaters({}, ["S000002"])
    with fake_radio_links(heaters) as links:
        await setup_entry(hass, entry)
        assert entry.state is ConfigEntryState.LOADED
        site = await async_site_network_id(hass, "test")
        assert links[0].network_id == site and links[0].dialect is DIALECT_B
        assert "network_id" not in entry.data
        sensor = _gateway_entity(hass, entry, "binary_sensor", "pairing_open")
        assert hass.states.get(sensor).state == STATE_ON
        button = _gateway_entity(hass, entry, "button", "pair_heaters")
        await hass.services.async_call(
            "button", "press", {"entity_id": button}, blocking=True
        )
        # The idle heater talks once on its old network: not enough to adopt it.
        foreign = bytes.fromhex("5678")
        links[0].deliver(received(9, REGISTRATION, network_id=foreign))
        await hass.async_block_till_done()
        assert "network_id" not in entry.data
        assert not [dst for dst, _payload in links[0].sent if dst != 0xFF]  # no scan

        await _pair(hass, links[0], announce_b)
        assert heaters.assigned == [2]
        assert entry.data["dialect"] == "B"
        assert entry.data["network_id"] == site.hex().upper()

        await _after_debounce(hass)  # stores the heater and reloads
        await _settle(hass)
        assert entry.state is ConfigEntryState.LOADED and len(links) == 2
        assert links[1].network_id == site
        assert entry.data["nodes"] == [
            {"type": "htr", "addr": "2", "name": "Heater S000002", "serial": "S000002"}
        ]
        assert hass.states.get("climate.heater_s000002") is not None
        note = _notifications(hass)[f"{DOMAIN}_{entry.entry_id}_heaters_found"]
        assert "Found TermoWeb heater S000002" in note["message"]
        await _unload(hass, entry)
