"""Tests for the radio backend and its power estimate."""

from __future__ import annotations

from datetime import datetime
import importlib
import logging
from types import SimpleNamespace

import pytest

from custom_components.termoweb.backend.radio_power import (
    MAX_INTEGRATION_GAP_S,
    EnergyEstimator,
    PowerManager,
)
from custom_components.termoweb.const import BRAND_RADIO
from custom_components.termoweb.inventory import Inventory, build_node_inventory
from tests.fakes.radio_link import NET, FakeRadioLink


def _mod(name: str):
    """Import a backend module at test time (other suites reload these modules)."""

    return importlib.import_module(f"custom_components.termoweb.backend{name}")


NODES = [{"type": "htr", "addr": "6", "name": "Living room"}]


def make_backend():
    """Return a radio backend over a fake-linked client."""

    client = _mod(".radio_client").RadioClient(
        "radio.local", 2323, "B", NODES, network_id=NET, link_factory=FakeRadioLink
    )
    backend = _mod(".factory").create_backend(brand=BRAND_RADIO, client=client)
    assert isinstance(backend, _mod(".radio_backend").RadioBackend)
    return backend


def test_create_radio_client_helper() -> None:
    """The factory helper builds a lazily connecting client."""

    client = _mod("").create_radio_client("10.0.0.5", 2323, "A", NODES, None)
    assert isinstance(client, _mod(".radio_client").RadioClient)
    assert client.dialect.name == "A"
    assert client.link is None
    client = _mod("").create_radio_client("10.0.0.5", 2323, "B", NODES, NET)
    assert client.dialect.name == "B"
    with pytest.raises(ValueError, match="explicit network_id"):
        _mod("").create_radio_client("10.0.0.5", 2323, "B", NODES, None)


def test_create_ws_client_returns_listener() -> None:
    """The websocket slot is filled by the radio listener."""

    backend = make_backend()
    inventory = Inventory("aabbcc001122", build_node_inventory(NODES))
    listener = backend.create_ws_client(
        SimpleNamespace(data={}),
        "entry",
        "aabbcc001122",
        SimpleNamespace(),
        inventory=inventory,
    )
    assert isinstance(listener, _mod(".radio_ws").RadioListener)
    assert listener.dev_id == "aabbcc001122" and listener.entry_id == "entry"


def test_create_ws_client_requires_radio_client() -> None:
    """A radio backend paired with another client type is a wiring bug."""

    radio_backend = _mod(".radio_backend").RadioBackend
    backend = radio_backend(brand=BRAND_RADIO, client=SimpleNamespace())
    with pytest.raises(TypeError, match="RadioClient"):
        backend.create_ws_client(
            SimpleNamespace(data={}),
            "e",
            "d",
            SimpleNamespace(),
            inventory=Inventory("d", build_node_inventory(NODES)),
        )


@pytest.mark.asyncio
async def test_fetch_hourly_samples_is_empty() -> None:
    """Radio heaters keep no energy history."""

    backend = make_backend()
    result = await backend.fetch_hourly_samples(
        "aabbcc001122", [("htr", "6")], datetime(2026, 1, 1), datetime(2026, 1, 2)
    )
    assert result == {}


@pytest.mark.asyncio
async def test_base_backend_delegates_writes_to_radio_client() -> None:
    """Backend.set_node_settings and friends reach the radio client."""

    backend = make_backend()
    link = await backend.client.async_connect()
    link.reply(0xB8, bytes.fromhex("B921252A02"))
    link.reply(0xB6, b"\xb7\x55")
    link.reply(0xBA, b"\xbb\x55")
    await backend.set_node_settings("dev", ("htr", "6"), mode="auto")
    await backend.set_node_lock("dev", ("htr", "6"), lock=True)
    assert link.payloads() == [
        b"\xb8",
        bytes.fromhex("B621252A01"),
        b"\xba\x01",
    ]


MANUAL, AUTO, OFF = 2, 1, 4


def manager(**settings) -> tuple[PowerManager, dict]:
    """Return a manager backed by a plain dict, like the entry options."""

    store: dict = {"radio_power": dict(settings)}

    def save(new: dict) -> None:
        store["radio_power"] = new

    return PowerManager(lambda: store["radio_power"], save), store


def test_without_a_limit_nothing_is_switched_off() -> None:
    """No limit: no shedding, and every shed heater is restored."""

    power, _ = manager(rated_power={"6": 1500}, shed={"7": AUTO})
    power.note_heating(6, True)
    assert power.power_limit is None
    assert power.plan() == ([], [(7, AUTO)])


def test_over_the_limit_the_lowest_priority_heater_goes_off(caplog) -> None:
    """Lowest priority first, until the rest fit; unknown power is never shed."""

    power, _ = manager(
        power_limit=2000,
        rated_power={"5": 1000, "6": 1500, "7": 800},
        priority={"5": 9, "6": 1, "7": 5},
    )
    for addr in (5, 6, 7, 8):  # 8: power unknown
        power.note_heating(addr, True)
    with caplog.at_level(logging.INFO):
        assert power.plan() == ([6], [])  # 3300 W -> 1800 W after heater 6 goes off
    assert "switching heater 6" in caplog.text


def test_a_single_heater_above_the_limit_stays_off() -> None:
    """A limit below a heater's own power keeps it switched off."""

    power, store = manager(power_limit=1000, rated_power={"6": 1500})
    power.note_heating(6, True)
    assert power.plan() == ([6], [])
    power.mark_shed(6, MANUAL)
    assert store["radio_power"]["shed"] == {"6": MANUAL}
    assert power.plan() == ([], [])  # 1500 W still does not fit: stays off


def test_shed_heaters_return_by_priority_when_they_fit() -> None:
    """Room again: the highest-priority shed heater whose power fits comes back."""

    power, _ = manager(
        power_limit=2000,
        rated_power={"5": 1200, "6": 1500, "7": 1000},
        priority={"6": 8, "7": 2},
        shed={"6": MANUAL, "7": AUTO},
    )
    power.note_heating(5, True)  # 1200 W drawing, 800 W free
    assert power.plan() == ([], [])
    power.note_heating(5, False)  # 2000 W free: 6 (1500) fits, then 7 does not
    assert power.plan() == ([], [(6, MANUAL)])
    power.clear_shed(6)
    power.clear_shed(6)  # forgetting twice is harmless
    assert power.plan() == ([], [(7, AUTO)])


def test_shed_heater_with_unknown_power_comes_back() -> None:
    """If a shed heater's power is removed from the options, it is restored."""

    power, _ = manager(power_limit=500, shed={"6": MANUAL})
    assert power.plan() == ([], [(6, MANUAL)])


def test_settings_round_trip_through_the_store() -> None:
    """Limit and priority writes go through save; rated power can be learned."""

    power, store = manager(rated_power={"6": "bad", "x": 5})
    power.set_power_limit(1800)
    power.set_priority(6, 3)
    assert store["radio_power"]["power_limit"] == 1800
    assert store["radio_power"]["priority"] == {"6": 3}
    assert power.priority(6) == 3 and power.priority(7) == 0
    assert power.rated_power(6) is None
    power.note_reported_power(6, 752.6)
    assert power.rated_power(6) == 752.6
    power.set_power_limit(0)
    assert power.power_limit is None
    with pytest.raises(ValueError, match=">= 0"):
        power.set_power_limit(-1)
    store["radio_power"] = {"power_limit": "garbage"}
    assert power.power_limit is None


def test_in_memory_settings_without_callbacks() -> None:
    """A manager without callbacks keeps its own settings."""

    power = PowerManager()
    power.set_power_limit(1200)
    assert power.power_limit == 1200
    assert PowerManager(lambda: None).power_limit is None


def test_energy_estimate_integrates_rated_power_times_duty() -> None:
    """Wh grows by rated x duty while heating; idle, unknown power and gaps are bounded."""

    rated = {6: 1500.0}
    energy = EnergyEstimator(rated.get)
    assert energy.counter_wh(6) is None
    energy.observe(6, True, 12, 0.0)  # first sight: counter starts at 0
    assert energy.counter_wh(6) == 0.0
    energy.observe(6, False, 0, 3600.0)  # heated at 12 % for the hour before
    assert energy.counter_wh(6) == pytest.approx(
        1500 * 0.12 * MAX_INTEGRATION_GAP_S / 3600
    )
    energy.observe(6, True, 0, 3700.0)  # idle interval adds nothing
    before = energy.counter_wh(6)
    energy.observe(6, True, 0, 3800.0)  # heating without a duty byte: full power
    assert energy.counter_wh(6) == pytest.approx(before + 1500 * 100 / 3600)
    energy.observe(7, True, 50, 0.0)
    energy.observe(7, True, 50, 100.0)
    assert energy.counter_wh(7) is None  # power unknown: no counter
