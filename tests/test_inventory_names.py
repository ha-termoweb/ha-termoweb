"""Friendly-name fallbacks of ``Inventory.resolve_heater_name``."""

from __future__ import annotations

from custom_components.termoweb.inventory import Inventory, build_node_inventory


def test_unknown_heater_address_uses_the_default_name() -> None:
    """An address the inventory does not name falls back to the factory name."""
    inventory = Inventory(
        "dev", build_node_inventory({"nodes": [{"type": "htr", "addr": 1}]})
    )

    assert inventory.resolve_heater_name("htr", "9") == "Heater 9"
    assert (
        inventory.resolve_heater_name("htr", "9", default_factory="Radiator {}".format)
        == "Radiator 9"
    )
