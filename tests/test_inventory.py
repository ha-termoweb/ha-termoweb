"""Tests for the inventory container."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, List

import pytest

from custom_components.termoweb.inventory import Inventory


class DummyNode:
    """Represent a lightweight node stub."""

    def __init__(self, name: str, metadata: Any | None = None) -> None:
        """Store identifying fields for the stub."""

        self.name = name
        self.metadata = metadata


@pytest.fixture
def sample_payload() -> dict[str, Any]:
    """Return a representative payload mapping."""

    return {"nodes": [{"addr": 1}, {"addr": 2}]}


@pytest.fixture
def sample_nodes() -> List[DummyNode]:
    """Return a list of dummy node instances."""

    return [DummyNode("alpha"), DummyNode("beta")]


@pytest.fixture
def inventory(sample_payload: dict[str, Any], sample_nodes: List[DummyNode]) -> Inventory:
    """Build an inventory instance for tests."""

    return Inventory("abc123", sample_payload, sample_nodes)


def test_inventory_properties(
    inventory: Inventory, sample_payload: dict[str, Any], sample_nodes: List[DummyNode]
) -> None:
    """Validate accessor properties and tuple conversion."""

    assert inventory.dev_id == "abc123"
    assert inventory.payload is sample_payload
    assert inventory.nodes == tuple(sample_nodes)
    assert isinstance(inventory.nodes, tuple)


def test_inventory_rejects_mutation(inventory: Inventory) -> None:
    """Ensure mutation attempts raise frozen instance errors."""

    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        inventory.dev_id = "xyz987"
    with pytest.raises(TypeError):
        inventory.nodes[0] = DummyNode("gamma")


def test_inventory_nodes_are_independent(
    sample_payload: dict[str, Any], sample_nodes: List[DummyNode]
) -> None:
    """Confirm node collections are copied into the tuple."""

    inventory = Inventory("abc123", sample_payload, sample_nodes)
    sample_nodes.append(DummyNode("gamma"))
    assert len(inventory.nodes) == 2
