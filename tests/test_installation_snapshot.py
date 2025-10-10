from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from custom_components.termoweb.installation import (
    InstallationSnapshot,
    ensure_snapshot,
)
from custom_components.termoweb.inventory import (
    Inventory,
    build_node_inventory,
    heater_sample_subscription_targets,
)


def _make_snapshot(
    nodes: list[dict[str, Any]],
    *,
    dev_id: str = "dev",
    include_inventory: bool = True,
) -> InstallationSnapshot:
    """Return an installation snapshot preloaded with ``nodes`` payload."""

    payload = {"nodes": nodes}
    inventory_nodes = build_node_inventory(payload)
    if include_inventory:
        inventory = Inventory(dev_id, payload, inventory_nodes)
        return InstallationSnapshot(dev_id=dev_id, raw_nodes=payload, inventory=inventory)
    return InstallationSnapshot(dev_id=dev_id, raw_nodes=payload)


def test_snapshot_properties_and_inventory_cache() -> None:
    """Properties should expose cached metadata and computed inventory."""

    nodes = [{"type": "HTR", "addr": "1", "name": "Heater"}]
    snapshot = _make_snapshot(nodes)

    assert snapshot.dev_id == "dev"
    assert snapshot.raw_nodes == {"nodes": nodes}
    assert [node.addr for node in snapshot.inventory] == ["1"]

    reference = Inventory(snapshot.dev_id, snapshot.raw_nodes, snapshot.inventory)
    assert snapshot.nodes_by_type == reference.nodes_by_type
    assert snapshot.explicit_heater_names == reference.explicit_heater_names
    assert snapshot.heater_address_map == reference.heater_address_map
    first_map = snapshot.heater_sample_address_map
    second_map = snapshot.heater_sample_address_map
    assert first_map == second_map


def test_snapshot_sample_targets_and_name_map_caching() -> None:
    """Sample targets and name maps should reuse cached computations."""

    nodes = [
        {"type": "htr", "addr": "1", "name": ""},
        {"type": "acm", "addr": "2", "name": None},
    ]
    snapshot = _make_snapshot(nodes, include_inventory=False)

    targets_first = snapshot.heater_sample_targets
    targets_second = snapshot.heater_sample_targets
    expected_targets = heater_sample_subscription_targets(
        snapshot.heater_sample_address_map[0]
    )

    assert targets_first == expected_targets
    assert targets_second == expected_targets

    def _factory(name: str) -> str:
        return f"prefixed {name}"

    first_map = snapshot.heater_name_map(_factory)
    second_map = snapshot.heater_name_map(_factory)

    assert first_map is second_map
    assert "htr" in first_map
    assert isinstance(first_map["htr"], dict)


def test_snapshot_sample_targets_filter_invalid_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sample targets should discard invalid or empty subscription entries."""

    nodes = [
        {"type": "htr", "addr": "1"},
        {"type": "acm", "addr": "2"},
    ]
    snapshot = _make_snapshot(nodes)

    def _fake_targets(_map):
        return [
            None,
            "bad",
            ("htr", " 1 "),
            ("acm", None),
            ("acm", ""),
            ["acm", "2 "],
        ]

    monkeypatch.setattr(
        "custom_components.termoweb.installation.heater_sample_subscription_targets",
        _fake_targets,
    )

    assert snapshot.heater_sample_targets == [("htr", "1"), ("acm", "2")]


def test_snapshot_update_nodes_accepts_inventory() -> None:
    """Providing an inventory container should refresh cached nodes."""

    snapshot = InstallationSnapshot(dev_id="dev", raw_nodes={"nodes": []})
    payload = {"nodes": [{"type": "htr", "addr": "1", "name": "Heater"}]}
    inventory_nodes = build_node_inventory(payload)
    container = Inventory("dev", payload, inventory_nodes)

    snapshot.update_nodes(payload, inventory=container)

    inventory = snapshot.inventory
    assert inventory is not None
    assert [node.addr for node in inventory] == ["1"]


def test_snapshot_nodes_by_type_skips_unknown() -> None:
    """Nodes without valid types should be ignored in type maps."""

    payload = {
        "nodes": [
            {"type": "", "addr": "1", "name": ""},
            {"type": "htr", "addr": "2", "name": ""},
        ]
    }
    inventory_nodes = build_node_inventory(payload)
    snapshot = InstallationSnapshot(
        dev_id="dev",
        raw_nodes={"nodes": []},
        inventory=Inventory("dev", payload, inventory_nodes),
    )
    record = {"snapshot": snapshot}

    assert ensure_snapshot(record) is snapshot

    mapping = snapshot.nodes_by_type

    assert "htr" in mapping
    assert all(node.addr == "2" for node in mapping["htr"])


def test_snapshot_update_nodes_rebuilds_from_payload() -> None:
    """update_nodes should rebuild inventory when none is provided."""

    payload = {"nodes": [{"type": "htr", "addr": "5", "name": "Living"}]}
    snapshot = InstallationSnapshot(dev_id="dev", raw_nodes={})

    snapshot.update_nodes(payload)

    assert [node.addr for node in snapshot.inventory] == ["5"]


def test_snapshot_rejects_invalid_inventory() -> None:
    """Passing non-Inventory instances should raise ``TypeError``."""

    with pytest.raises(TypeError):
        InstallationSnapshot(dev_id="dev", raw_nodes={}, inventory=object())

    snapshot = InstallationSnapshot(dev_id="dev", raw_nodes={})
    with pytest.raises(TypeError):
        snapshot.update_nodes({}, inventory=object())


def test_ensure_snapshot_handles_missing() -> None:
    """ensure_snapshot should return None when no snapshot is present."""

    assert ensure_snapshot(None) is None
    assert ensure_snapshot(SimpleNamespace(snapshot=None)) is None
    assert ensure_snapshot({"snapshot": "ignored"}) is None
    assert ensure_snapshot({}) is None
