"""Tests for the inventory container."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any, Iterable, List, Mapping

import pytest

import custom_components.termoweb.inventory as inventory_module
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


def test_inventory_heater_sample_address_map_caches(heater_inventory: Inventory) -> None:
    """Normalised heater sample addresses should cache and defend against mutation."""

    assert heater_inventory._heater_sample_address_cache is None

    forward, compat = heater_inventory.heater_sample_address_map

    assert heater_inventory._heater_sample_address_cache is not None
    assert forward == {"htr": ["1", "2"], "acm": ["2"]}
    assert compat == {"htr": "htr"}

    forward["htr"].append("bogus")
    compat["heater"] = "htr"

    cached_forward, cached_compat = heater_inventory.heater_sample_address_map

    assert cached_forward == {"htr": ["1", "2"], "acm": ["2"]}
    assert cached_compat == {"htr": "htr"}


def test_inventory_heater_sample_targets_cache_and_order(heater_inventory: Inventory) -> None:
    """Sample targets should be cached and maintain canonical ordering."""

    assert heater_inventory._heater_sample_targets_cache is None

    targets = heater_inventory.heater_sample_targets

    assert heater_inventory._heater_sample_targets_cache is not None
    assert targets == [("htr", "1"), ("htr", "2"), ("acm", "2")]

    targets.append(("acm", "extra"))

    cached = heater_inventory.heater_sample_targets
    assert cached == [("htr", "1"), ("htr", "2"), ("acm", "2")]
    assert cached is not targets


def test_inventory_heater_sample_alias_handling(heater_inventory: Inventory) -> None:
    """Alias types should normalise to heater targets for sample metadata."""

    object.__setattr__(
        heater_inventory,
        "_heater_address_map_cache",
        (
            {"heater": ("10",), "acm": ("2",)},
            {"10": frozenset({"heater"}), "2": frozenset({"acm"})},
        ),
    )
    object.__setattr__(heater_inventory, "_heater_sample_address_cache", None)
    object.__setattr__(heater_inventory, "_heater_sample_targets_cache", None)

    forward, compat = heater_inventory.heater_sample_address_map

    assert forward == {"htr": ["10"], "acm": ["2"]}
    assert compat == {"heater": "htr", "htr": "htr"}


def test_inventory_heater_name_map_caches_by_factory(
    heater_inventory: Inventory,
) -> None:
    """Name map computations should be cached per default factory."""

    def prefixed(addr: str) -> str:
        return f"Prefixed {addr}"

    first = heater_inventory.heater_name_map(prefixed)
    second = heater_inventory.heater_name_map(prefixed)

    assert first is second
    assert first[("htr", "1")] == "Lounge"
    assert first[("htr", "2")] == "Prefixed 2"

    def alternate(addr: str) -> str:
        return f"Alternate {addr}"

    third = heater_inventory.heater_name_map(alternate)
    fourth = heater_inventory.heater_name_map(alternate)

    assert third is fourth
    assert third is not first
    assert third[("htr", "2")] == "Alternate 2"


def test_inventory_heater_name_map_supports_default_factory_optional(
    sample_payload: dict[str, Any],
) -> None:
    """Calling without a factory should use the built-in heater fallback names."""

    nodes = [
        Node(name=None, addr="1", node_type="HTR"),
        Node(name="Storage", addr="2", node_type="acm"),
    ]
    inventory = Inventory("dev", sample_payload, nodes)

    first = inventory.heater_name_map()
    second = inventory.heater_name_map()

    assert first is second
    assert first[("htr", "1")] == "Heater 1"
    assert first[("acm", "2")] == "Storage"

def test_inventory_heater_sample_targets_filters_invalid(
    heater_inventory: Inventory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sample targets property should discard invalid entries from helpers."""

    def fake_targets(_: Mapping[Any, Iterable[Any]] | Iterable[Any] | None) -> list[Any]:
        return [
            None,
            "bad",
            ("htr", " 1 "),
            ("acm", None),
            ("acm", ""),
            ["acm", "2 "],
        ]

    monkeypatch.setattr(
        inventory_module,
        "heater_sample_subscription_targets",
        fake_targets,
    )
    object.__setattr__(heater_inventory, "_heater_sample_targets_cache", None)

    targets = heater_inventory.heater_sample_targets
    assert targets == [("htr", "1")]
    assert ("acm", "2") not in targets


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
