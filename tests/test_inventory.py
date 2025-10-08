"""Tests for the inventory container."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any, List

import pytest

from custom_components.termoweb.inventory import Inventory, Node


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


@pytest.fixture
def heater_inventory(sample_payload: dict[str, Any]) -> Inventory:
    """Return an inventory containing heater and ancillary nodes."""

    nodes = [
        Node(name=" Lounge ", addr="1", node_type="HTR"),
        Node(name="", addr="2", node_type="htr"),
        Node(name="Storage", addr="2", node_type="ACM"),
        Node(name=None, addr=" 2 ", node_type="acm"),
        SimpleNamespace(type="htr", addr="", name="Ignored"),
        Node(name="Thermostat", addr="9", node_type="thm"),
    ]
    return Inventory("abc123", sample_payload, nodes)


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


def test_inventory_nodes_by_type_caches_results(heater_inventory: Inventory) -> None:
    """Nodes by type should be cached and immune to caller mutation."""

    assert heater_inventory._nodes_by_type_cache is None
    first = heater_inventory.nodes_by_type
    assert heater_inventory._nodes_by_type_cache is not None
    first.setdefault("htr", []).append("sentinel")

    second = heater_inventory.nodes_by_type

    assert "sentinel" not in second["htr"]
    assert len(second["htr"]) == 3


def test_inventory_heater_metadata_properties(heater_inventory: Inventory) -> None:
    """Heater-specific properties should expose canonical metadata."""

    nodes = heater_inventory.nodes
    expected_nodes = tuple(
        node
        for node in nodes
        if getattr(node, "type", "").strip().lower() in {"htr", "acm"}
    )
    actual_pairs = {
        (
            getattr(node, "type", "").strip().lower(),
            str(getattr(node, "addr", "")).strip(),
        )
        for node in heater_inventory.heater_nodes
    }
    expected_pairs = {
        (
            getattr(node, "type", "").strip().lower(),
            str(getattr(node, "addr", "")).strip(),
        )
        for node in expected_nodes
    }
    assert actual_pairs == expected_pairs

    assert heater_inventory.explicit_heater_names == {("htr", "1"), ("acm", "2")}

    forward, reverse = heater_inventory.heater_address_map

    assert forward == {"htr": ["1", "2"], "acm": ["2"]}
    assert reverse == {"1": {"htr"}, "2": {"htr", "acm"}}

    # Mutating the returned maps must not corrupt cached state.
    forward["htr"].append("bogus")
    reverse.setdefault("3", set()).add("htr")

    cached_forward, cached_reverse = heater_inventory.heater_address_map
    assert cached_forward == {"htr": ["1", "2"], "acm": ["2"]}
    assert cached_reverse == {"1": {"htr"}, "2": {"htr", "acm"}}


def test_build_heater_inventory_details_wraps_inventory(
    heater_inventory: Inventory,
) -> None:
    """Legacy helper should return data derived from the ``Inventory`` wrapper."""

    from custom_components.termoweb.inventory import (  # noqa: PLC0415
        build_heater_inventory_details,
    )

    details = build_heater_inventory_details(heater_inventory.nodes)
    forward, reverse = heater_inventory.heater_address_map

    assert details.nodes_by_type == heater_inventory.nodes_by_type
    assert details.explicit_name_pairs == heater_inventory.explicit_heater_names
    assert details.address_map == forward
    assert details.reverse_address_map == reverse
