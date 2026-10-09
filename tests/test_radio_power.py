"""Tests for the radio gateway's local power manager."""

from __future__ import annotations

import logging

import pytest

from custom_components.termoweb.backend.radio_power import PowerManager

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
