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
    assert "addresses_by_type" not in record
    assert "heater_address_map" not in record
    assert "power_monitor_address_map" not in record

    assert record["settings"] == {"htr": {"2": {"mode": "auto"}}}
