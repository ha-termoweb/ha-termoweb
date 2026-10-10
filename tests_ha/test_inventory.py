"""Tests for the inventory container."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any, Callable, Iterable, List, Mapping
from unittest.mock import MagicMock

import pytest

import custom_components.termoweb.inventory as inventory_module
from custom_components.termoweb.inventory import (
    Inventory,
    Node,
    PowerMonitorNode,
    normalize_power_monitor_addresses,
)


class DummyNode:
    """Represent a lightweight node stub."""

    def __init__(self, name: str, metadata: Any | None = None) -> None:
        """Store identifying fields for the stub."""

        self.name = name
        self.metadata = metadata


@pytest.fixture
def sample_nodes() -> List[DummyNode]:
    """Return a list of dummy node instances."""

    return [DummyNode("alpha"), DummyNode("beta")]


@pytest.fixture
def inventory(sample_nodes: List[DummyNode]) -> Inventory:
    """Build an inventory instance for tests."""

    return Inventory("abc123", sample_nodes)


@pytest.fixture
def heater_inventory() -> Inventory:
    """Return an inventory containing heater and ancillary nodes."""

    nodes = [
        Node(name=" Lounge ", addr="1", node_type="HTR"),
        Node(name="", addr="2", node_type="htr"),
        Node(name="Storage", addr="2", node_type="ACM"),
        Node(name=None, addr=" 2 ", node_type="acm"),
        SimpleNamespace(type="htr", addr="", name="Ignored"),
        Node(name="Thermostat", addr="9", node_type="thm"),
        PowerMonitorNode(name="Monitor", addr="3"),
        Node(name="Second Monitor", addr="4", node_type="pmo"),
    ]
    return Inventory("abc123", nodes)


def test_inventory_properties(
    inventory: Inventory, sample_nodes: List[DummyNode]
) -> None:
    """Validate accessor properties and tuple conversion."""

    assert inventory.dev_id == "abc123"
    assert inventory.nodes == tuple(sample_nodes)
    assert isinstance(inventory.nodes, tuple)
    assert not hasattr(inventory, "payload")


def test_inventory_rejects_mutation(inventory: Inventory) -> None:
    """Ensure mutation attempts raise frozen instance errors."""

    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        inventory.dev_id = "xyz987"
    with pytest.raises(TypeError):
        inventory.nodes[0] = DummyNode("gamma")


def test_inventory_nodes_are_independent(sample_nodes: List[DummyNode]) -> None:
    """Confirm node collections are copied into the tuple."""

    inventory = Inventory("abc123", sample_nodes)
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
        if getattr(node, "type", "").strip().lower() in {"htr", "acm", "thm"}
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

    assert heater_inventory.explicit_heater_names == {
        ("htr", "1"),
        ("acm", "2"),
        ("thm", "9"),
    }

    forward, reverse = heater_inventory.heater_address_map

    assert forward == {"htr": ["1", "2"], "acm": ["2"], "thm": ["9"]}
    assert reverse == {
        "1": {"htr"},
        "2": {"htr", "acm"},
        "9": {"thm"},
    }

    # Mutating the returned maps must not corrupt cached state.
    forward["htr"].append("bogus")
    reverse.setdefault("3", set()).add("htr")

    cached_forward, cached_reverse = heater_inventory.heater_address_map
    assert cached_forward == {"htr": ["1", "2"], "acm": ["2"], "thm": ["9"]}
    assert cached_reverse == {
        "1": {"htr"},
        "2": {"htr", "acm"},
        "9": {"thm"},
    }


def test_inventory_heater_sample_address_map_caches(
    heater_inventory: Inventory,
) -> None:
    """Normalised heater sample addresses should cache and defend against mutation."""

    assert heater_inventory._heater_sample_address_cache is None

    forward, compat = heater_inventory.heater_sample_address_map

    assert heater_inventory._heater_sample_address_cache is not None
    assert forward == {"htr": ["1", "2"], "acm": ["2"], "thm": ["9"]}
    assert compat == {"htr": "htr"}

    forward["htr"].append("bogus")
    compat["heater"] = "htr"

    cached_forward, cached_compat = heater_inventory.heater_sample_address_map

    assert cached_forward == {"htr": ["1", "2"], "acm": ["2"], "thm": ["9"]}
    assert cached_compat == {"htr": "htr"}


def test_inventory_heater_sample_targets_cache_and_order(
    heater_inventory: Inventory,
) -> None:
    """Sample targets should be cached and maintain canonical ordering."""

    assert heater_inventory._heater_sample_targets_cache is None

    targets = heater_inventory.heater_sample_targets

    assert heater_inventory._heater_sample_targets_cache is not None
    assert targets == [("htr", "1"), ("htr", "2"), ("acm", "2")]

    targets.append(("acm", "extra"))

    cached = heater_inventory.heater_sample_targets
    assert cached == [("htr", "1"), ("htr", "2"), ("acm", "2")]
    assert cached is not targets


def test_inventory_energy_sample_types_excludes_thermostats(
    heater_inventory: Inventory,
) -> None:
    """Thermostat nodes should not appear in the energy sample type set."""

    types = heater_inventory.energy_sample_types

    assert "htr" in types
    assert "acm" in types
    assert "pmo" in types
    assert "thm" not in types


def test_inventory_heater_sample_ignores_legacy_type_alias(
    heater_inventory: Inventory,
) -> None:
    """The legacy ``heater`` type is not a wire type and is not aliased to htr."""

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

    assert forward == {"htr": [], "acm": ["2"]}
    assert compat == {"htr": "htr"}


def test_inventory_power_monitor_metadata(heater_inventory: Inventory) -> None:
    """Power monitor caches should expose canonical metadata and aliases."""

    forward, reverse = heater_inventory.power_monitor_address_map

    assert forward == {"pmo": ["3", "4"]}
    assert reverse == {"3": {"pmo"}, "4": {"pmo"}}

    forward["pmo"].append("bogus")
    reverse.setdefault("5", set()).add("pmo")

    cached_forward, cached_reverse = heater_inventory.power_monitor_address_map
    assert cached_forward == {"pmo": ["3", "4"]}
    assert cached_reverse == {"3": {"pmo"}, "4": {"pmo"}}

    sample_map, compat = heater_inventory.power_monitor_sample_address_map
    assert sample_map == {"pmo": ["3", "4"]}
    assert compat == {"pmo": "pmo"}

    sample_map["pmo"].append("ignored")
    compat["alias"] = "pmo"

    cached_map, cached_compat = heater_inventory.power_monitor_sample_address_map
    assert cached_map == {"pmo": ["3", "4"]}
    assert cached_compat == {"pmo": "pmo"}

    targets = heater_inventory.power_monitor_sample_targets
    assert targets == [("pmo", "3"), ("pmo", "4")]
    targets.append(("pmo", "extra"))

    cached_targets = heater_inventory.power_monitor_sample_targets
    assert cached_targets == [("pmo", "3"), ("pmo", "4")]


def test_inventory_power_monitor_targets_filter_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Power monitor targets should drop malformed subscription entries."""

    called: list[Any] = []

    def _fake_targets(_map: Any) -> list[Any]:
        called.append(_map)
        return [
            None,
            "bad",
            ("pmo", " 3 "),
            ("pmo", None),
            ("pmo", "4"),
            ("pmo", " 4 "),
        ]

    monkeypatch.setattr(
        inventory_module,
        "power_monitor_sample_subscription_targets",
        _fake_targets,
    )
    inventory = Inventory(
        "dev",
        [
            PowerMonitorNode(name="One", addr="3"),
            PowerMonitorNode(name="Two", addr="4"),
        ],
    )

    assert inventory.power_monitor_sample_targets == [("pmo", "3"), ("pmo", "4")]
    assert called


def test_inventory_power_monitor_targets_deduplicate_and_strip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Power monitor targets should deduplicate and sanitise entries."""

    def _fake_targets(_map: Any) -> list[Any]:
        return [("pmo", " 3 "), ("pmo", "3"), ("pmo", " 4 ")]

    monkeypatch.setattr(
        inventory_module,
        "power_monitor_sample_subscription_targets",
        _fake_targets,
    )
    inventory = Inventory(
        "dev",
        [
            PowerMonitorNode(name="One", addr="3"),
            PowerMonitorNode(name="Two", addr="4"),
        ],
    )

    assert inventory.power_monitor_sample_targets == [("pmo", "3"), ("pmo", "4")]


def test_normalize_power_monitor_addresses_variants() -> None:
    """normalise helper should handle None and iterables, without type aliases."""

    empty_map, compat = normalize_power_monitor_addresses(None)
    assert empty_map == {"pmo": []}
    assert compat == {"pmo": "pmo"}

    list_map, _ = normalize_power_monitor_addresses(["A", "A", " "])
    assert list_map == {"pmo": ["A"]}

    alias_map, alias_compat = normalize_power_monitor_addresses(
        {"power_monitor": ["B"]}
    )
    assert alias_map == {"pmo": []}
    assert alias_compat == {"pmo": "pmo"}

    string_map, _ = normalize_power_monitor_addresses("C")
    assert string_map == {"pmo": ["C"]}

    ignored_map, ignored_compat = normalize_power_monitor_addresses({"ignored": ["D"]})
    assert ignored_map == {"pmo": []}
    assert "ignored" not in ignored_compat


def test_inventory_heater_names_by_type_supports_default_factory_optional() -> None:
    """Calling without a factory should use the built-in heater fallback names."""

    nodes = [
        Node(name=None, addr="1", node_type="HTR"),
        Node(name="Storage", addr="2", node_type="acm"),
    ]
    inventory = Inventory("dev", nodes)

    first = inventory.heater_names_by_type()

    assert first == inventory.heater_names_by_type()
    assert first["htr"]["1"] == "Heater 1"
    assert first["acm"]["2"] == "Storage"


def test_inventory_resolve_heater_name_uses_overrides() -> None:
    """Resolved names should prioritize explicit overrides."""

    nodes = [
        Node(name="Custom", addr="1", node_type="htr"),
        Node(name="", addr="2", node_type="htr"),
        Node(name=None, addr="3", node_type="acm"),
        Node(name="Tank", addr="4", node_type="acm"),
    ]
    inventory = Inventory("dev", nodes)

    assert inventory.resolve_heater_name("htr", "1") == "Custom"
    assert inventory.resolve_heater_name("htr", "2") == "Heater 2"
    assert inventory.resolve_heater_name("acm", "3") == "Accumulator 3"
    assert inventory.resolve_heater_name("acm", "4") == "Tank"


def test_inventory_heater_names_by_type_shape_is_stable(
    heater_inventory: Inventory,
) -> None:
    """Name maps should only contain per-type dictionaries."""

    mapping = heater_inventory.heater_names_by_type()

    assert set(mapping) == {"htr", "acm", "thm"}
    assert all(isinstance(key, str) for key in mapping)
    assert all(isinstance(value, dict) for value in mapping.values())
    assert all(
        isinstance(addr, str) and isinstance(name, str)
        for bucket in mapping.values()
        for addr, name in bucket.items()
    )


def test_inventory_heater_sample_targets_filters_invalid(
    heater_inventory: Inventory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sample targets property should discard invalid entries from helpers."""

    def fake_targets(
        _: Mapping[Any, Iterable[Any]] | Iterable[Any] | None,
    ) -> list[Any]:
        return [
            None,
            "bad",
            ("htr", " 1 "),
            ("acm", None),
            ("acm", ""),
            ["acm", "2 "],
            ("htr", "1"),
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


def test_inventory_heater_sample_targets_deduplicate_and_strip(
    heater_inventory: Inventory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sample targets should strip whitespace and deduplicate pairs."""

    def fake_targets(
        _: Mapping[Any, Iterable[Any]] | Iterable[Any] | None,
    ) -> list[Any]:
        return [("htr", " 1 "), ("htr", "1"), ("acm", " 2 ")]

    monkeypatch.setattr(
        inventory_module,
        "heater_sample_subscription_targets",
        fake_targets,
    )
    object.__setattr__(heater_inventory, "_heater_sample_targets_cache", None)

    assert heater_inventory.heater_sample_targets == [("htr", "1"), ("acm", "2")]
