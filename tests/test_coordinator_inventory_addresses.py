"""Tests for coordinator inventory address cloning."""

from __future__ import annotations

from unittest.mock import AsyncMock

from homeassistant.core import HomeAssistant

from custom_components.termoweb.coordinator import StateCoordinator
from custom_components.termoweb.inventory import Inventory, build_node_inventory


def test_inventory_addresses_by_type_returns_copy() -> None:
    """StateCoordinator should clone inventory address mappings."""

    raw_nodes = {
        "nodes": [
            {"type": "htr", "addr": "1"},
            {"type": "acm", "addr": "2"},
            {"type": "pmo", "addr": "9"},
            {"type": "thm", "addr": "5"},
        ]
    }
    inventory_nodes = build_node_inventory(raw_nodes)
    inventory = Inventory("device", raw_nodes, inventory_nodes)

    hass = HomeAssistant()
    coordinator = StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="device",
        device={},
        nodes=raw_nodes,
        inventory=inventory,
    )

    expected = inventory.addresses_by_type
    result = coordinator._inventory_addresses_by_type()

    assert result == expected
    assert result is not expected

    for node_type, addrs in expected.items():
        assert result[node_type] == addrs
        assert result[node_type] is not addrs

    result["htr"].append("extra")
    fresh_expected = inventory.addresses_by_type
    assert "extra" not in fresh_expected["htr"]
