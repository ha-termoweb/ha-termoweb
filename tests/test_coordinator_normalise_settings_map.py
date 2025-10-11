"""Tests for ``StateCoordinator._normalise_settings_map``."""

from __future__ import annotations

from typing import Any

from custom_components.termoweb.coordinator import StateCoordinator


def test_normalise_settings_map_normalises_and_copies() -> None:
    """Test normalisation of node types, addresses, and payload copies."""

    original_settings: dict[str, dict[Any, Any]] = {
        " HTR ": {
            " 01 ": {"foo": ["bar"]},
            2: {"nested": {"value": 1}},
        },
        " ThM ": {
            " 7 ": {"inner": {"value": 5}},
        },
    }

    normalised = StateCoordinator._normalise_settings_map(original_settings)

    assert normalised == {
        "htr": {
            "01": {"foo": ["bar"]},
            "2": {"nested": {"value": 1}},
        },
        "thm": {
            "7": {"inner": {"value": 5}},
        },
    }

    normalised["htr"]["01"]["foo"].append("baz")
    normalised["thm"]["7"]["inner"]["value"] = 9

    assert original_settings[" HTR "][" 01 "] == {"foo": ["bar"]}
    assert original_settings[" ThM "][" 7 "] == {"inner": {"value": 5}}
