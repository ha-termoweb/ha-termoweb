"""Tests for ``StateCoordinator._clone_inventory``."""

from __future__ import annotations

from custom_components.termoweb.coordinator import StateCoordinator
from custom_components.termoweb.inventory import Inventory, Node


def test_clone_inventory_creates_independent_instance() -> None:
    """Ensure cloning returns a matching but independent inventory."""

    payload = {"nodes": [{"addr": "1"}, {"addr": "2"}]}
    nodes = [
        Node(name="Heater", addr="1", node_type="htr"),
        Node(name="Thermostat", addr="2", node_type="thm"),
    ]
    source = Inventory("dev-123", payload, nodes)

    clone = StateCoordinator._clone_inventory(source)

    assert clone is not source
    assert isinstance(clone, Inventory)
    assert clone.dev_id == source.dev_id
    assert clone.payload == source.payload
    assert clone.nodes == source.nodes

    clone._heater_name_map_cache[1] = {"sentinel": "clone"}
    clone._heater_name_map_factories[1] = lambda addr: f"Clone {addr}"

    assert source._heater_name_map_cache == {}
    assert source._heater_name_map_factories == {}
