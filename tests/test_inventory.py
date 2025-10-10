"""Tests for the inventory container."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any, Iterable, List, Mapping

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
        PowerMonitorNode(name="Monitor", addr="3"),
        Node(name="Alias Monitor", addr="4", node_type="power_monitor"),
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
    assert compat.get("pmo") == "pmo"
    assert compat.get("power_monitor") == "pmo"

    sample_map["pmo"].append("ignored")
    compat["alias"] = "pmo"

    cached_map, cached_compat = heater_inventory.power_monitor_sample_address_map
    assert cached_map == {"pmo": ["3", "4"]}
    assert cached_compat.get("power_monitor") == "pmo"

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
        return [None, "bad", ("pmo", " 3 "), ("pmo", None), ("pmo", "4")]

    monkeypatch.setattr(
        inventory_module,
        "power_monitor_sample_subscription_targets",
        _fake_targets,
    )
    inventory = Inventory(
        "dev",
        {"nodes": []},
        [PowerMonitorNode(name="One", addr="3"), PowerMonitorNode(name="Two", addr="4")],
    )

    assert inventory.power_monitor_sample_targets == [("pmo", "3"), ("pmo", "4")]
    assert called


def test_normalize_power_monitor_addresses_variants() -> None:
    """normalise helper should handle None, iterables and aliases."""

    empty_map, compat = normalize_power_monitor_addresses(None)
    assert empty_map == {"pmo": []}
    assert compat["power_monitor"] == "pmo"

    list_map, _ = normalize_power_monitor_addresses(["A", "A", " "])
    assert list_map == {"pmo": ["A"]}

    alias_map, alias_compat = normalize_power_monitor_addresses({"power_monitor": ["B"]})
    assert alias_map == {"pmo": ["B"]}
    assert alias_compat["power_monitor"] == "pmo"

    string_map, _ = normalize_power_monitor_addresses("C")
    assert string_map == {"pmo": ["C"]}

    ignored_map, ignored_compat = normalize_power_monitor_addresses({"ignored": ["D"]})
    assert ignored_map == {"pmo": []}
    assert "ignored" not in ignored_compat


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


def test_resolve_record_inventory_prefers_existing_container() -> None:
    """Resolution should return stored inventory without rebuilding."""

    raw_nodes = {"nodes": [{"type": "htr", "addr": "1"}]}
    container = Inventory(
        "device",
        raw_nodes,
        inventory_module.build_node_inventory(raw_nodes),
    )
    record: dict[str, Any] = {"inventory": container}

    resolution = inventory_module.resolve_record_inventory(record)

    assert resolution.inventory is container
    assert resolution.source == "inventory"
    assert resolution.filtered_count == 1
    assert resolution.raw_count == 1


def test_resolve_record_inventory_uses_cached_node_list() -> None:
    """Node lists should be normalised when no container is cached."""

    nodes = [
        Node(name="Heater", addr="1", node_type="htr"),
        SimpleNamespace(type="htr", addr="", name="ignored"),
        SimpleNamespace(as_dict=lambda: {}, type=" ", addr=""),
    ]
    record: dict[str, Any] = {
        "dev_id": 101,
        "node_inventory": nodes,
        "nodes": {"nodes": []},
    }

    resolution = inventory_module.resolve_record_inventory(record)

    assert resolution.source == "node_inventory"
    assert resolution.filtered_count == 1
    assert record["inventory"] is resolution.inventory
    assert isinstance(resolution.inventory, Inventory)
    assert resolution.inventory.dev_id == "101"
    assert record["node_inventory"] is nodes


def test_resolve_record_inventory_falls_back_to_snapshot() -> None:
    """Snapshots should be converted into inventory containers when needed."""

    raw_nodes = {"nodes": [{"type": "htr", "addr": "2"}]}
    snapshot_nodes = inventory_module.build_node_inventory(raw_nodes)
    snapshot = SimpleNamespace(
        dev_id="snapshot-dev",
        raw_nodes=raw_nodes,
        inventory=list(snapshot_nodes),
    )
    record: dict[str, Any] = {"snapshot": snapshot}

    resolution = inventory_module.resolve_record_inventory(record)

    assert resolution.source == "snapshot"
    assert resolution.filtered_count == 1
    assert record["inventory"] is resolution.inventory
    assert resolution.inventory.dev_id == "snapshot-dev"
    assert "node_inventory" not in record


def test_resolve_record_inventory_builds_from_raw_nodes() -> None:
    """Raw node payloads should be normalised when no cache exists."""

    raw_nodes = {"nodes": [{"type": "htr", "addr": "5"}]}
    record: dict[str, Any] = {"dev_id": "dev-raw", "nodes": raw_nodes}

    resolution = inventory_module.resolve_record_inventory(record)

    assert resolution.source == "raw_nodes"
    assert resolution.filtered_count == 1
    assert record["inventory"] is resolution.inventory
    assert "node_inventory" not in record
    assert resolution.inventory.heater_address_map[0] == {"htr": ["5"]}


def test_resolve_record_inventory_handles_missing_data() -> None:
    """Resolution should return a fallback result when metadata is absent."""

    resolution = inventory_module.resolve_record_inventory(None)

    assert resolution.inventory is None
    assert resolution.source == "fallback"
    assert resolution.filtered_count == 0


def test_resolve_record_inventory_handles_null_nodes() -> None:
    """Records with null node payloads should fall back gracefully."""

    record = {"dev_id": "dev-null", "nodes": None}

    resolution = inventory_module.resolve_record_inventory(record)

    assert resolution.inventory is None
    assert resolution.source == "fallback"
    assert resolution.filtered_count == 0


def test_resolve_record_inventory_detects_mismatched_inventory() -> None:
    """Cached inventory should be rebuilt when cached entries are invalid."""

    invalid = SimpleNamespace(as_dict=lambda: {}, type=" ", addr="")
    cached = Inventory("dev", {}, [invalid])
    record: dict[str, Any] = {
        "inventory": cached,
        "node_inventory": [invalid],
    }

    resolution = inventory_module.resolve_record_inventory(record)

    assert resolution.source == "node_inventory"
    assert resolution.filtered_count == 0
    assert record["inventory"] is resolution.inventory
    assert record["node_inventory"] == [invalid]


def test_heater_platform_details_default_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default naming should activate when name map is not a mapping."""

    raw_nodes = {
        "nodes": [
            {"type": "htr", "addr": "1"},
            {"type": "acm", "addr": "2"},
        ]
    }
    inventory = Inventory(
        "dev-name",
        raw_nodes,
        inventory_module.build_node_inventory(raw_nodes),
    )

    monkeypatch.setattr(
        inventory_module.Inventory,
        "heater_name_map",
        lambda self, _factory: ["invalid"],
    )

    nodes_by_type, addrs_by_type, resolver = (
        inventory_module.heater_platform_details_from_inventory(
            inventory,
            default_name_simple=lambda addr: f"Heater {addr}",
        )
    )

    assert set(nodes_by_type) == {"htr", "acm"}
    assert addrs_by_type == {"htr": ["1"], "acm": ["2"]}
    assert resolver("htr", "1") == "Heater 1"
    assert resolver("acm", "2") == "Accumulator 2"
