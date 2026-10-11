"""Radio heaters are found in the background and added to the entry by themselves."""

from __future__ import annotations

import asyncio
from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from homeassistant.components import persistent_notification
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
import pytest
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_fire_time_changed,
)

from custom_components.termoweb import radio_discovery
from custom_components.termoweb.backend.radio import DIALECT_B, RadioLinkError
from custom_components.termoweb.backend.radio.discovery import NetworkSighting
from custom_components.termoweb.backend.radio.pairing import PairedHeater
from custom_components.termoweb.backend.radio.survey import analyse
from custom_components.termoweb.config_flow_radio import (
    RadioSetupError,
    UnknownDialectError,
)
from custom_components.termoweb.const import DOMAIN
from tests.fakes.cloud import VERSION
from tests.fakes.radio_link import (
    NET,
    REGISTRATION,
    STATUS_SHORT,
    FakeRadioLink,
    received,
)
from tests.fakes.radio_setup import record_reloads
from tests.fakes.sensors import fake_radio_links, radio_entry, setup_entry

# Synthetic 7-character serials, one per heater address.
SERIALS = {4: "S000004", 6: "S000006", 7: "S000007", 9: "S000009"}
KNOWN = {"type": "htr", "addr": "6", "name": "Heater S000006", "serial": "S000006"}


def _identity(serial: str) -> bytes:
    """Return a dialect-B short identity reply carrying ``serial``."""
    return bytes.fromhex("5B550105010D004100400F") + serial.encode()


def _heaters(*present: int) -> Any:
    """Return a link script: heaters ``present`` answer status and identity reads."""

    def _script(link: FakeRadioLink) -> None:
        def _on_send(dst: int, payload: bytes) -> None:
            answer = dst in present
            link.replies.pop(0x5A, None)
            link.replies.pop(0xB8, None)
            if answer:
                link.reply(0x5A, _identity(SERIALS[dst]))
                link.reply(0xB8, STATUS_SHORT)

        link.on_send = _on_send

    return _script


def _notifications(hass: HomeAssistant) -> dict[str, Any]:
    """Return the persistent notifications by id."""
    return persistent_notification._async_get_or_create_notifications(hass)


def _pending_entry(**data: Any) -> MockConfigEntry:
    """Return a radio entry that has not learned its heater network yet."""
    return MockConfigEntry(
        domain=DOMAIN,
        minor_version=6,
        unique_id="radio:test",
        title="Radio gateway (10.0.0.5)",
        data={"brand": "radio", "host": "10.0.0.5", "port": 2323, "nodes": [], **data},
    )


async def _unload(hass: HomeAssistant, entry: MockConfigEntry) -> None:
    """Unload the entry so no radio task outlives the test."""
    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()


async def _settle(hass: HomeAssistant) -> None:
    """Wait for background work, then for the tasks it started.

    Unanswered reads time out on the real clock, and a task inside such a
    timeout counts as cancelling, which async_block_till_done skips.
    """
    async with asyncio.timeout(10):
        while hass._background_tasks:
            await asyncio.sleep(0.01)
    await hass.async_block_till_done()


async def _setup(hass: HomeAssistant, entry: MockConfigEntry) -> None:
    """Set the entry up and let its background search or scan finish."""
    await setup_entry(hass, entry)
    await _settle(hass)


async def _after_debounce(hass: HomeAssistant) -> None:
    """Let the debounce window pass and the identity reads finish."""
    async_fire_time_changed(
        hass, dt_util.utcnow() + timedelta(seconds=radio_discovery.FOUND_DEBOUNCE_S + 1)
    )
    await _settle(hass)


@pytest.fixture(autouse=True)
def _no_scan_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scan with a tiny on-air gap between addresses (fake links are instant)."""
    monkeypatch.setattr(radio_discovery, "SCAN_GAP_S", 0.0001)


# --- an entry that does not know its heater network yet ---------------------------


async def test_setup_does_not_wait_for_the_network_and_unload_stops_the_search(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setup returns while listening runs; unloading cancels the listening."""
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def _listen_forever(*_args: Any, **_kwargs: Any) -> Any:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(radio_discovery, "find_network", _listen_forever)
    entry = _pending_entry()

    await setup_entry(hass, entry)

    assert entry.state is ConfigEntryState.LOADED
    assert started.is_set()
    await _unload(hass, entry)
    assert cancelled.is_set()
    assert entry.state is ConfigEntryState.NOT_LOADED


async def test_heard_network_is_stored_and_the_entry_reloads(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The network heard in the background is stored; a reload then finds heaters."""
    calls: list[tuple[Any, ...]] = []
    outcomes: list[Any] = [
        RadioSetupError("no_traffic"),
        RadioLinkError("gateway busy"),
        RuntimeError("bug"),
        NetworkSighting(DIALECT_B, NET, frozenset({7})),
    ]

    async def _find(*args: Any, **kwargs: Any) -> Any:
        calls.append((*args, kwargs["link_factory"]))
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(radio_discovery, "find_network", _find)
    monkeypatch.setattr(radio_discovery, "NETWORK_RETRY_S", 0.0)
    reloads = record_reloads(monkeypatch, hass)
    entry = _pending_entry(dialect="B")

    await _setup(hass, entry)

    assert entry.data["dialect"] == "B" and entry.data["network_id"] == "1234"
    assert reloads == [entry.entry_id]
    assert len(calls) == 4  # silence, a busy gateway and a bug are retried
    assert calls[0][:4] == ("10.0.0.5", 2323, "B", None)
    await _unload(hass, entry)


async def test_unknown_dialect_saves_a_report_and_tells_the_user(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Bursts no known dialect decodes: a report is saved, listening stops."""
    hass.config.config_dir = str(tmp_path)
    report = analyse([])
    calls: list[Any] = []

    async def _find(*_args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        assert (await kwargs["analyse_survey"]([])).verdict == "silent"  # executor
        raise UnknownDialectError(
            report.__class__(**{**report.__dict__, "verdict": "undecodable"})
        )

    monkeypatch.setattr(radio_discovery, "find_network", _find)
    entry = _pending_entry()

    await _setup(hass, entry)

    assert len(calls) == 1
    files = list(tmp_path.glob("termoweb_radio_survey_setup_*.json"))
    saved = json.loads(files[0].read_text())
    assert saved["integration_version"] == VERSION
    assert saved["radio_type"] is None and saved["configured_dialect"] == "auto"
    assert saved["report"]["verdict"] == "undecodable"
    note = _notifications(hass)[f"{DOMAIN}_{entry.entry_id}_unknown_dialect"]
    assert str(files[0]) in note["message"]
    assert "network_id" not in entry.data
    await _unload(hass, entry)


async def test_unknown_dialect_report_that_cannot_be_saved(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When the report cannot be written the notification says so."""
    hass.config.config_dir = str(tmp_path / "missing")

    async def _find(*_args: Any, **_kwargs: Any) -> Any:
        raise UnknownDialectError(analyse([]))

    monkeypatch.setattr(radio_discovery, "find_network", _find)
    entry = _pending_entry()
    await _setup(hass, entry)
    note = _notifications(hass)[f"{DOMAIN}_{entry.entry_id}_unknown_dialect"]
    assert "could not be saved" in note["message"]
    await _unload(hass, entry)


# --- an entry that knows its network ------------------------------------------------


async def test_entry_without_heaters_scans_and_adds_the_heaters_that_answer(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No heaters yet: every usual address is asked; answering heaters are added."""
    reloads = record_reloads(monkeypatch, hass)
    entry = radio_entry("radio", [])
    with fake_radio_links(_heaters(4)) as links:
        await _setup(hass, entry)

        assert entry.state is ConfigEntryState.LOADED
        asked = sorted({dst for dst, payload in links[0].sent if payload == b"\x5a"})
        assert asked == list(range(2, 33))
        assert entry.data["nodes"] == [
            {"type": "htr", "addr": "4", "name": "Heater S000004", "serial": "S000004"}
        ]
        assert reloads == [entry.entry_id]
        note = _notifications(hass)[f"{DOMAIN}_{entry.entry_id}_heaters_found"]
        assert "Found TermoWeb heater S000004" in note["message"]
        await _unload(hass, entry)


async def test_entry_with_heaters_does_not_scan(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A set-up entry only asks heaters it hears, never the address range."""
    reloads = record_reloads(monkeypatch, hass)
    entry = radio_entry("radio", [KNOWN])
    with fake_radio_links(_heaters(6, 4)) as links:
        await setup_entry(hass, entry)
        await _after_debounce(hass)
        assert 4 not in {dst for dst, _payload in links[0].sent}
        assert reloads == []
        await _unload(hass, entry)


async def test_heater_heard_on_our_network_is_added_with_its_serial(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unknown heater's frame: identity read, node stored, one reload, a note."""
    caplog.set_level("INFO")
    reloads = record_reloads(monkeypatch, hass)
    entry = radio_entry("radio", [KNOWN])
    with fake_radio_links(_heaters(6, 7)) as links:
        await setup_entry(hass, entry)
        links[0].deliver(received(7, REGISTRATION))
        links[0].deliver(received(7, REGISTRATION))  # repeats are one heater
        await hass.async_block_till_done()
        assert reloads == []  # waits for more heaters first

        await _after_debounce(hass)

        assert entry.data["nodes"] == [
            KNOWN,
            {"type": "htr", "addr": "7", "name": "Heater S000007", "serial": "S000007"},
        ]
        assert [p for dst, p in links[0].sent if dst == 7] == [b"\x5a"]  # no reply
        assert reloads == [entry.entry_id]
        note = _notifications(hass)[f"{DOMAIN}_{entry.entry_id}_heaters_found"]
        assert note["message"] == (
            f"Found TermoWeb heater S000007. It was added to {entry.title}."
        )
        assert "S000007" not in caplog.text  # serials are masked in logs

        links[0].deliver(received(9, REGISTRATION))  # the entry is about to reload
        await _after_debounce(hass)
        assert reloads == [entry.entry_id]
        await _unload(hass, entry)


async def test_heaters_heard_together_cause_one_reload(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two new heaters within the debounce window: both added, one reload."""
    reloads = record_reloads(monkeypatch, hass)
    entry = radio_entry("radio", [KNOWN])
    with fake_radio_links(_heaters(6, 7, 9)) as links:
        await setup_entry(hass, entry)
        links[0].deliver(received(9, REGISTRATION))
        links[0].deliver(received(7, REGISTRATION))
        await _after_debounce(hass)

        assert [node["addr"] for node in entry.data["nodes"]] == ["6", "7", "9"]
        assert reloads == [entry.entry_id]
        note = _notifications(hass)[f"{DOMAIN}_{entry.entry_id}_heaters_found"]
        assert "S000007" in note["message"] and "S000009" in note["message"]
        await _unload(hass, entry)


async def test_frames_from_a_foreign_network_are_ignored(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A neighbour's heater with the same short id is never asked or added."""
    reloads = record_reloads(monkeypatch, hass)
    entry = radio_entry("radio", [KNOWN])
    with fake_radio_links(_heaters(6, 7)) as links:
        await setup_entry(hass, entry)
        links[0].deliver(received(7, REGISTRATION, network_id=bytes.fromhex("5678")))
        links[0].deliver(received(9, REGISTRATION, dst=2))  # not to our station
        await _after_debounce(hass)

        assert not [dst for dst, _payload in links[0].sent if dst in (7, 9)]
        assert entry.data["nodes"] == [KNOWN]
        assert reloads == []
        await _unload(hass, entry)


async def test_known_heater_at_a_new_address_moves_instead_of_duplicating(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A heater paired again at a new address keeps its node, now at that address."""
    reloads = record_reloads(monkeypatch, hass)
    entry = radio_entry("radio", [KNOWN])
    monkeypatch.setitem(SERIALS, 9, "S000006")  # heater 6, paired again as 9
    with fake_radio_links(_heaters(6, 9)) as links:
        await setup_entry(hass, entry)
        links[0].deliver(received(9, REGISTRATION))
        await _after_debounce(hass)

        assert entry.data["nodes"] == [{**KNOWN, "addr": "9"}]
        assert reloads == [entry.entry_id]
        assert f"{DOMAIN}_{entry.entry_id}_heaters_found" not in _notifications(hass)
        await _unload(hass, entry)


async def test_silent_heater_is_asked_again_only_after_a_pause(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A heater that does not report its identity is not asked at every frame."""
    reloads = record_reloads(monkeypatch, hass)
    entry = radio_entry("radio", [KNOWN])
    with fake_radio_links(_heaters(6)) as links:
        await setup_entry(hass, entry)
        for _ in range(2):
            links[0].deliver(received(7, REGISTRATION))
            await _after_debounce(hass)

        assert [p for dst, p in links[0].sent if dst == 7] == [b"\x5a"]
        assert entry.data["nodes"] == [KNOWN]
        assert reloads == []
        await _unload(hass, entry)


async def test_unload_cancels_a_pending_identification(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A heater heard just before unload is never asked once the entry is gone."""
    reloads = record_reloads(monkeypatch, hass)
    entry = radio_entry("radio", [KNOWN])
    with fake_radio_links(_heaters(6, 7)) as links:
        await setup_entry(hass, entry)
        links[0].deliver(received(7, REGISTRATION))
        await _unload(hass, entry)
        await _after_debounce(hass)

        assert not [dst for dst, _payload in links[0].sent if dst == 7]
        assert entry.data["nodes"] == [KNOWN]
        assert reloads == []


async def test_finder_ignores_invalid_addresses_and_stops_when_done(
    hass: HomeAssistant,
) -> None:
    """Addresses outside 1-254 are ignored; a stopped finder stores nothing."""
    entry = _pending_entry(dialect="B", network_id="1234")
    entry.add_to_hass(hass)

    class _Client:
        async def async_read_identity(self, addr: int) -> Any:
            raise AssertionError("must not be asked")

    finder = radio_discovery.HeaterFinder(hass, entry, _Client())
    finder.async_heard(0)
    finder.async_heard(255)
    finder.async_stop()
    finder.async_heard(7)
    await finder.async_add([7])
    await _after_debounce(hass)
    assert entry.data["nodes"] == []


async def test_finder_stores_nothing_when_every_heater_is_known(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A heater that is already stored at its address changes nothing."""
    reloads = record_reloads(monkeypatch, hass)
    entry = _pending_entry(dialect="B", network_id="1234", nodes=[KNOWN])
    entry.add_to_hass(hass)

    class _Client:
        async def async_read_identity(self, addr: int) -> Any:
            return SimpleNamespace(tail=b"", serial=SERIALS[addr])

    await radio_discovery.HeaterFinder(hass, entry, _Client()).async_add([6])
    assert entry.data["nodes"] == [KNOWN]
    assert reloads == []


async def test_paired_heaters_join_the_batch_without_a_second_identity_read(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A heater paired here is stored with the batch; one without a serial is read."""
    reloads = record_reloads(monkeypatch, hass)
    entry = _pending_entry(dialect="B", network_id="1234", nodes=[KNOWN])
    entry.add_to_hass(hass)
    asked: list[int] = []

    class _Client:
        async def async_read_identity(self, addr: int) -> Any:
            asked.append(addr)
            return SimpleNamespace(tail=b"", serial=SERIALS[addr])

    finder = radio_discovery.HeaterFinder(hass, entry, _Client())
    finder.async_paired(PairedHeater(7, None))  # its identity read failed
    finder.async_paired(PairedHeater(4, None, serial="S000004"))
    finder.async_heard(4)  # its registration: not read again
    await _after_debounce(hass)

    assert asked == [7]
    assert [node["addr"] for node in entry.data["nodes"]] == ["6", "4", "7"]
    assert reloads == [entry.entry_id]
    finder.async_paired(PairedHeater(9, None, serial="S000009"))  # done: ignored
    await _after_debounce(hass)
    assert len(entry.data["nodes"]) == 3


async def test_new_gateway_finds_its_network_then_its_heaters(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end: network heard, reload, scan, heater added, reload, heater shown."""

    async def _find(*_args: Any, **_kwargs: Any) -> Any:
        return NetworkSighting(DIALECT_B, NET)

    monkeypatch.setattr(radio_discovery, "find_network", _find)
    entry = _pending_entry()
    with fake_radio_links(_heaters(4)):
        await _setup(hass, entry)  # stores the network and reloads
        await _settle(hass)  # scan, reload
        await _settle(hass)

        assert entry.state is ConfigEntryState.LOADED
        assert entry.data["network_id"] == "1234"
        assert [node["serial"] for node in entry.data["nodes"]] == ["S000004"]
        assert hass.states.get("climate.heater_s000004") is not None
        await _unload(hass, entry)
