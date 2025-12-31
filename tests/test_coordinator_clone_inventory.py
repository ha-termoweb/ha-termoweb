"""Tests for coordinator inventory references."""

from __future__ import annotations

from custom_components.termoweb.domain.ids import NodeId, NodeType
from custom_components.termoweb.domain.legacy_view import (
    store_to_legacy_coordinator_data,
)
from custom_components.termoweb.domain.state import DomainStateStore
from custom_components.termoweb.inventory import Inventory, Node


def test_device_record_reuses_inventory_instance() -> None:
    """Ensure the coordinator stores the provided inventory without cloning."""

    payload = {"nodes": [{"addr": "1", "type": "htr"}]}
    nodes = [Node(name="Heater", addr="1", node_type="htr")]
    inventory = Inventory("dev-123", payload, nodes)

    store = DomainStateStore([NodeId(NodeType.HEATER, "2")])
    store.apply_full_snapshot("htr", "2", {"mode": "auto"})

    record = store_to_legacy_coordinator_data(
        "dev-123",
        store,
        inventory,
        device_name="Device",
        device_details={},
    )

    device = record["dev-123"]
    assert device["inventory"] is inventory
    assert "addresses_by_type" not in device
    assert "heater_address_map" not in device
    assert "power_monitor_address_map" not in device

    assert device["settings"] == {"htr": {"2": {"mode": "auto"}}}
