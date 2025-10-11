"""Tests for StateCoordinator._collect_previous_settings."""

from custom_components.termoweb.coordinator import StateCoordinator


def test_collect_previous_settings_normalises_and_ingests_sections() -> None:
    """Ensure previous settings are normalised and extended from extra sections."""

    prev_dev = {
        "dev_id": "dev-1",
        "name": "Boiler room",
        "raw": {"ignored": True},
        "connected": True,
        "settings": {
            "htr": {
                "01": {"target": 21},
                2: {"target": 19},
                "": {"target": 15},
                None: {"target": 12},
            },
            "thm": ["not", "a", "mapping"],
        },
        "status": {
            "addrs": [" 05 ", "05"],
            "settings": {" 05 ": {"online": True}},
        },
        "prog": {
            "settings": {6: {"schedule": "weekday"}},
        },
        "inventory": {"ignored": True},
        "nodes": {},
    }
    addr_map = {
        "htr": ["01", "02", "03"],
        "status": ["05"],
        "prog": ["06"],
    }

    result = StateCoordinator._collect_previous_settings(prev_dev, addr_map)

    assert result == {
        "htr": {
            "01": {"target": 21},
            "2": {"target": 19},
        },
        "status": {
            "05": {"online": True},
        },
        "prog": {
            "6": {"schedule": "weekday"},
        },
    }
