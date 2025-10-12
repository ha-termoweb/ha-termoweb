"""Tests for StateCoordinator._collect_previous_settings."""

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from custom_components.termoweb.coordinator import StateCoordinator
from custom_components.termoweb.inventory import AccumulatorNode, HeaterNode


def test_collect_previous_settings_respects_inventory_addresses(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None],
        Any,
    ]
) -> None:
    """Previous settings should be carried forward using inventory metadata."""

    inventory = inventory_builder(
        "dev-1",
        {},
        [
            HeaterNode(name="Hall", addr="01"),
            AccumulatorNode(name="Store", addr="07"),
        ],
    )
    prev_dev = {
        "dev_id": "dev-1",
        "settings": {
            "htr": {
                "01": {"target": 21},
                2: {"target": 19},
            },
            "acm": {"07": {"mode": "auto"}},
            "thm": {"01": {"temperature": 19}},
        },
        "status": {
            "addrs": ["05"],
            "settings": {"05": {"online": True}},
        },
    }

    result = StateCoordinator._collect_previous_settings(prev_dev, inventory)

    assert result == {
        "htr": {"01": {"target": 21}},
        "acm": {"07": {"mode": "auto"}},
        "thm": {"01": {"temperature": 19}},
    }

    # Inventory should ensure heater-compatible types always exist in the map.
    inventory_empty = inventory_builder(
        "dev-2",
        {},
        [AccumulatorNode(name="Store", addr="09")],
    )
    preserved = StateCoordinator._collect_previous_settings({}, inventory_empty)
    assert preserved == {"acm": {}}


def test_collect_previous_settings_handles_missing_inventory() -> None:
    """Previous settings should be copied when inventory metadata is absent."""

    prev_dev = {"settings": {"htr": {"01": {"target": 21}}, "pmo": {"1": {"w": 900}}}}

    result = StateCoordinator._collect_previous_settings(prev_dev, None)

    assert result == {
        "htr": {"01": {"target": 21}},
        "pmo": {"1": {"w": 900}},
    }
