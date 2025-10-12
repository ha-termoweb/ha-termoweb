"""Tests for ``StateCoordinator._assemble_device_record``."""

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from homeassistant.core import HomeAssistant

from custom_components.termoweb.coordinator import StateCoordinator
from custom_components.termoweb.inventory import AccumulatorNode, HeaterNode


def test_assemble_device_record_uses_inventory_maps(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None],
        Any,
    ]
) -> None:
    """Device records should expose inventory-derived metadata."""

    hass = HomeAssistant()
    inventory = inventory_builder(
        "dev",
        {},
        [
            HeaterNode(name="Hall", addr="01"),
            AccumulatorNode(name="Store", addr="07"),
        ],
    )
    coordinator = StateCoordinator(
        hass,
        client=None,  # type: ignore[arg-type]
        base_interval=30,
        dev_id="dev",
        device={},
        nodes=inventory.payload,
        inventory=inventory,
    )

    settings = {
        "htr": {"01": {"target": 21}},
        "acm": {"07": {"mode": "auto"}},
        "extra": {"Z": {"raw": True}},
    }

    record = coordinator._assemble_device_record(
        inventory=inventory,
        settings_by_type=settings,
        name="Device dev",
    )

    assert record["addresses_by_type"] == inventory.addresses_by_type
    assert record["heater_address_map"]["forward"] == inventory.heater_address_map[0]
    assert record["power_monitor_address_map"]["forward"] == (
        inventory.power_monitor_address_map[0]
    )
    assert record["settings"] == settings
