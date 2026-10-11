"""Tests for the store of advanced setups written to radio heaters."""

from __future__ import annotations

from custom_components.termoweb.backend.radio.protocol import AdvancedSetup
from custom_components.termoweb.backend.radio_advanced import AdvancedSetupStore

DEFAULTS = {
    "control_mode": 4,
    "units": 0,
    "offset_tenths": 0,
    "away_mode": 0,
    "away_offset": 0,
    "modified_auto_span": 0,
    "window_mode": 0,
    "true_radiant": 0,
}


def test_records_round_trip_through_load_and_save() -> None:
    """put/get/forget go through the callbacks, keyed by the address as a string."""

    stored: dict = {}

    def save(records: dict) -> None:
        stored.clear()
        stored.update(records)

    store = AdvancedSetupStore(lambda: stored, save)
    assert store.get(6) is None
    store.put(6, AdvancedSetup(offset_tenths=-10))
    assert stored["6"] == {**DEFAULTS, "offset_tenths": -10}
    assert store.get(6) == AdvancedSetup(offset_tenths=-10)
    store.put(7, AdvancedSetup())
    store.forget(6)
    assert set(stored) == {"7"}
    store.forget(6)  # nothing stored: nothing saved
    assert set(stored) == {"7"}


def test_bad_stored_records_are_ignored() -> None:
    """A record with missing fields or non-numbers reads as never written."""

    records = {
        "1": {"control_mode": 4},
        "2": {**DEFAULTS, "units": "F"},
        "3": "nonsense",
        "4": dict(DEFAULTS),
    }
    store = AdvancedSetupStore(lambda: records)
    assert [store.get(addr) for addr in (1, 2, 3)] == [None, None, None]
    assert store.get(4) == AdvancedSetup()
    assert AdvancedSetupStore(lambda: None).get(1) is None


def test_without_callbacks_records_live_in_memory() -> None:
    """A store with no config entry keeps its records itself."""

    store = AdvancedSetupStore()
    store.put(6, AdvancedSetup(window_mode=1))
    assert store.get(6) == AdvancedSetup(window_mode=1)
    store.forget(6)
    assert store.get(6) is None
