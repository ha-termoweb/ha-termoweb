"""Unit tests for domain inventory models."""

from __future__ import annotations

import pytest

from custom_components.termoweb.domain.ids import NodeId, NodeType
from custom_components.termoweb.domain.inventory import (
    InstallationInventory,
    NodeInventory,
)


def test_node_inventory_is_frozen() -> None:
    """NodeInventory should be immutable."""

    node_id = NodeId(NodeType.HEATER, 1)
    inventory = NodeInventory(
        node_id=node_id, name="Heater 1", capabilities=frozenset({"on_off"})
    )

    with pytest.raises(AttributeError):
        inventory.name = "Changed"  # type: ignore[misc]


def test_installation_inventory_holds_mapping() -> None:
    """InstallationInventory should expose mapping keyed by NodeId."""

    node_id = NodeId(NodeType.THERMOSTAT, 2)
    node_inv = NodeInventory(
        node_id=node_id, name="Thermostat", capabilities=frozenset()
    )
    installation = InstallationInventory(dev_id="abcd", nodes={node_id: node_inv})

    assert installation.nodes[node_id] is node_inv
    with pytest.raises(TypeError):
        installation.nodes[node_id] = node_inv  # type: ignore[index]
