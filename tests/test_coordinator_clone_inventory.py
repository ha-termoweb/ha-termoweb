"""Tests for coordinator inventory references."""

from __future__ import annotations

from custom_components.termoweb.domain.ids import NodeId, NodeType
from custom_components.termoweb.domain.state import DomainStateStore, state_to_dict
from custom_components.termoweb.inventory import Inventory, Node


def test_device_record_reuses_inventory_instance() -> None:
    """Ensure the coordinator stores the provided inventory without cloning."""

    nodes = [Node(name="Heater", addr="1", node_type="htr")]
    inventory = Inventory("dev-123", nodes)

    store = DomainStateStore([NodeId(NodeType.HEATER, "2")])
    store.apply_full_snapshot("htr", "2", {"mode": "auto"})

    assert inventory.nodes_by_type["htr"][0] is nodes[0]

    state_map = {
        node_id.addr: state_to_dict(state)
        for node_id, state in store.iter_states()
        if node_id.node_type is NodeType.HEATER
    }
    assert state_map == {"2": {"mode": "auto"}}
