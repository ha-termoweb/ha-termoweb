"""Tests for coordinator inventory references."""

from __future__ import annotations

from unittest.mock import AsyncMock

from homeassistant.core import HomeAssistant

from custom_components.termoweb.coordinator import StateCoordinator
from custom_components.termoweb.inventory import Inventory, Node


def test_device_record_reuses_inventory_instance() -> None:
    """Ensure the coordinator stores the provided inventory without cloning."""

    payload = {"nodes": [{"addr": "1", "type": "htr"}]}
    nodes = [Node(name="Heater", addr="1", node_type="htr")]
    inventory = Inventory("dev-123", payload, nodes)

    coordinator = StateCoordinator(
        HomeAssistant(),
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev-123",
        device={},
        nodes=inventory.payload,
        inventory=inventory,
    )

    record = coordinator._assemble_device_record(
        inventory=inventory,
        settings_by_type={"htr": {"2": {"mode": "auto"}}},
        name="Device",
    )

    assert record["inventory"] is inventory
    assert record["addresses_by_type"] == inventory.addresses_by_type

    forward, reverse = inventory.heater_address_map
    assert record["heater_address_map"] == {"forward": forward, "reverse": reverse}

    power_forward, power_reverse = inventory.power_monitor_address_map
    assert record["power_monitor_address_map"] == {
        "forward": power_forward,
        "reverse": power_reverse,
    }

    assert record["settings"] == {"htr": {"2": {"mode": "auto"}}}
