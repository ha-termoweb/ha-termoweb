"""Unit tests for node models."""

from __future__ import annotations

import copy
import logging
from types import MappingProxyType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from conftest import _install_stubs

_install_stubs()

import custom_components.termoweb.coordinator as coordinator_module
import custom_components.termoweb.installation as installation_module
import custom_components.termoweb.inventory as inventory_module
import custom_components.termoweb.nodes as nodes_module
from custom_components.termoweb.inventory import (
    AccumulatorNode,
    _existing_nodes_map,
    build_node_inventory,
    HeaterNode,
    Node,
    PowerMonitorNode,
    ThermostatNode,
    normalize_node_addr,
    normalize_node_type,
)
from custom_components.termoweb.nodes import heater_sample_subscription_targets
from homeassistant.core import HomeAssistant


@pytest.fixture
def ducaheat_dev_data() -> dict[str, Any]:
    """Return a snapshot-style nodes payload produced by Ducaheat websocket."""

    nodes = {
        "htr": {
            "addrs": ["1", "2"],
            "settings": {
                "1": {"name": "Living Room"},
                "2": {"setup": {"name": "Bedroom Heater"}},
            },
            "samples": {"1": {}, "2": {}},
        },
        "acm": {
            "addrs": ["10", "11"],
            "settings": {"10": {"setup": {"name": "Storage Tank"}}},
            "advanced": {"11": {"label": "Garage Reserve"}},
        },
    }

    nodes_copy = copy.deepcopy(nodes)
    snapshot: dict[str, Any] = {"nodes": nodes_copy}
    nodes_by_type = {key: value for key, value in nodes_copy.items()}
    snapshot["nodes_by_type"] = nodes_by_type
    snapshot.update(nodes_by_type)
    return snapshot


def _make_state_coordinator(
    hass: HomeAssistant,
    nodes: Any,
    *,
    inventory: list[nodes_module.Node] | None = None,
) -> coordinator_module.StateCoordinator:
    """Construct a coordinator with predictable defaults for tests."""

    client = SimpleNamespace(get_node_settings=AsyncMock())
    return coordinator_module.StateCoordinator(
        hass,
        client,
        30,
        "dev",
        {"name": "Device"},
        nodes,  # type: ignore[arg-type]
        inventory,
    )


def test_heater_node_normalises_inputs() -> None:
    node = HeaterNode(name=" Living ", addr=2)

    assert node.name == "Living"
    assert node.addr == "2"
    assert node.type == "htr"
    assert node.supports_boost() is False


def test_accumulator_node_defaults() -> None:
    node = AccumulatorNode(name=None, addr="007")

    assert node.name == ""
    assert node.addr == "007"
    assert node.type == "acm"
    assert node.supports_boost() is True


def test_accumulator_supports_boost() -> None:
    node = AccumulatorNode(name="Storage", addr=3)

    assert node.supports_boost() is True


def test_power_monitor_stub() -> None:
    node = PowerMonitorNode(name="Monitor", addr="P1")

    with pytest.raises(NotImplementedError):
        node.power_level()


def test_thermostat_stub() -> None:
    node = ThermostatNode(name="Thermostat", addr="T1")

    with pytest.raises(NotImplementedError):
        node.capabilities()


def test_node_does_not_expose_brand_attribute() -> None:
    node = HeaterNode(name="Living", addr=1)

    assert not hasattr(node, "brand")


def test_node_requires_type() -> None:
    class BareNode(Node):
        __slots__ = ()

    with pytest.raises(ValueError):
        BareNode(name="Bare", addr=1)


def test_node_requires_addr() -> None:
    with pytest.raises(ValueError):
        HeaterNode(name="Living", addr="  ")


def test_node_updates_entity_attr_name() -> None:
    class EntityNode(HeaterNode):
        __slots__ = ("_attr_name",)

    node = EntityNode(name="First", addr=1)
    node._attr_name = "Legacy"  # attribute provided by HA entity mixin

    node.name = "Updated"

    assert node._attr_name == "Updated"
    assert node.name == "Updated"


def test_node_as_dict() -> None:
    node = HeaterNode(name="Kitchen", addr=5)

    assert node.as_dict() == {
        "name": "Kitchen",
        "addr": "5",
        "type": "htr",
    }


def test_existing_nodes_map_collects_sections() -> None:
    nodes = {
        "nodes_by_type": {
            "htr": {"addrs": ["1"], "settings": {"1": {"mode": "auto"}}},
            "thm": {"addrs": [], "settings": {}},
        },
        "htr": {"addrs": ["1"], "settings": {"1": {"mode": "auto"}}},
        "thm": {"addrs": [], "settings": {}},
        "dev_id": "dev",
        "connected": True,
    }

    sections = _existing_nodes_map(nodes)

    assert sections == {
        "htr": {"addrs": ["1"], "settings": {"1": {"mode": "auto"}}},
        "thm": {"addrs": [], "settings": {}},
    }


def test_existing_nodes_map_handles_non_mapping_input() -> None:
    assert _existing_nodes_map(None) == {}
    assert _existing_nodes_map({"nodes_by_type": []}) == {}


def test_iter_snapshot_sections_skips_invalid_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: set[tuple[str, str]] = set()
    sections = {
        1: MappingProxyType({"settings": {}}),
        "  ": {"settings": {}},
        "acm": {"settings": {}},
    }

    monkeypatch.setattr(
        inventory_module,
        "_iter_snapshot_section",
        lambda node_type, section: ({"addr": 5}, {"addr": None}),
    )

    assert list(inventory_module._iter_snapshot_sections(sections, seen)) == []


def test_collect_snapshot_addresses_handles_mixed_values() -> None:
    section = {
        "addrs": [None, "1"],
        "settings": {"1": {"name": "Living"}, "  ": {"name": "Ignored"}},
        "extra": {"2": {"label": "Garage"}, "3": "Loft"},
    }

    addresses = inventory_module._collect_snapshot_addresses(section)

    assert sorted(addresses) == ["1", "2", "3", "None"]
    assert addresses["1"][0]["name"] == "Living"
    assert addresses["2"][0]["label"] == "Garage"
    assert addresses["3"][0] == {"name": "Loft"}
    assert addresses["None"] == []


def test_extract_snapshot_name_handles_repeated_payloads() -> None:
    shared: dict[str, Any] = {}
    payloads = [shared, shared, {"title": "Kitchen"}]

    result = inventory_module._extract_snapshot_name(payloads)

    assert result == "Kitchen"


def test_build_node_inventory_handles_mixed_types(caplog: pytest.LogCaptureFixture) -> None:
    payload = {
        "nodes": [
            {"type": "htr", "addr": 1, "name": "Heater"},
            {"type": "ACM", "addr": "2", "name": "Accumulator"},
            {"type": "pmo", "addr": "3"},
            {"type": "foo", "addr": 4, "name": "Unknown"},
        ]
    }

    with caplog.at_level(logging.DEBUG):
        nodes = build_node_inventory(payload)

    assert [type(node) for node in nodes] == [
        HeaterNode,
        AccumulatorNode,
        PowerMonitorNode,
        Node,
    ]
    assert [node.addr for node in nodes] == ["1", "2", "3", "4"]
    assert any("Unsupported node type" in message for message in caplog.messages)


def test_build_node_inventory_handles_list_payload(caplog: pytest.LogCaptureFixture) -> None:
    payload = [
        {"type": "htr", "addr": "01", "name": "Heater"},
        {"addr": "02", "name": "Missing type"},
    ]

    with caplog.at_level(logging.DEBUG):
        nodes = build_node_inventory(payload)

    assert [node.addr for node in nodes] == ["01"]
    assert any("Skipping node with missing type" in message for message in caplog.messages)


def test_build_node_inventory_skips_none_type() -> None:
    payload = [
        {"type": None, "addr": "03", "name": "Null type"},
    ]

    assert build_node_inventory(payload) == []


def test_build_node_inventory_handles_ducaheat_snapshot(
    ducaheat_dev_data: dict[str, Any]
) -> None:
    nodes = build_node_inventory(ducaheat_dev_data)

    indexed = {(node.type, node.addr): node for node in nodes}

    assert set(indexed) == {
        ("htr", "1"),
        ("htr", "2"),
        ("acm", "10"),
        ("acm", "11"),
    }

    assert isinstance(indexed[("htr", "1")], HeaterNode)
    assert isinstance(indexed[("htr", "2")], HeaterNode)
    assert isinstance(indexed[("acm", "10")], AccumulatorNode)
    assert isinstance(indexed[("acm", "11")], AccumulatorNode)

    assert indexed[("htr", "1")].name == "Living Room"
    assert indexed[("htr", "2")].name == "Bedroom Heater"
    assert indexed[("acm", "10")].name == "Storage Tank"
    assert indexed[("acm", "11")].name == "Garage Reserve"


def test_build_node_inventory_tolerates_empty_payload() -> None:
    assert build_node_inventory({"nodes": []}) == []


def test_build_node_inventory_falls_back_to_node_type_field() -> None:
    payload = {"nodes": [{"type": "", "node_type": " THM ", "addr": "05"}]}

    nodes = build_node_inventory(payload)

    assert len(nodes) == 1
    assert nodes[0].type == "thm"
    assert nodes[0].addr == "05"


def test_build_node_inventory_falls_back_to_address_field() -> None:
    payload = {"nodes": [{"type": "HTR", "addr": " ", "address": " 09 "}]}

    nodes = build_node_inventory(payload)

    assert len(nodes) == 1
    assert nodes[0].addr == "09"


def test_state_coordinator_handles_none_nodes_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    hass = HomeAssistant()

    with caplog.at_level(logging.DEBUG):
        coordinator = _make_state_coordinator(hass, None)

    assert coordinator._nodes == {}
    assert coordinator._node_inventory == []
    assert sum(
        "Ignoring unexpected nodes payload" in message for message in caplog.messages
    )
    assert coordinator._nodes_by_type == {}


def test_state_coordinator_logs_once_for_invalid_nodes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    hass = HomeAssistant()
    coordinator = _make_state_coordinator(hass, {})

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        coordinator.update_nodes(["bad"])
        coordinator.update_nodes("also bad")

    assert coordinator._nodes == {}
    assert sum(
        "Ignoring unexpected nodes payload" in message for message in caplog.messages
    ) == 1


def test_utils_normalization_matches_node_inventory() -> None:
    payload = {"nodes": [{"type": " HTR ", "addr": " 01 "}]}

    nodes = build_node_inventory(payload)
    assert len(nodes) == 1
    node = nodes[0]

    assert normalize_node_type(" HTR ") == node.type
    assert normalize_node_addr(" 01 ") == node.addr
    assert (
        normalize_node_type(None, default="htr", use_default_when_falsey=True) == "htr"
    )


def test_node_init_uses_normalization_helpers() -> None:
    class DerivedNode(Node):
        __slots__ = ()
        NODE_TYPE = "ACM"

    node = DerivedNode(name=" Normalised ", addr=" 42 ", node_type=None)

    assert node.type == normalize_node_type(
        None,
        default="ACM",
        use_default_when_falsey=True,
    )
    assert node.addr == normalize_node_addr(" 42 ")
    assert node.name == "Normalised"


def test_ensure_node_inventory_sets_empty_cache() -> None:
    record: dict[str, object] = {}
    result = nodes_module.ensure_node_inventory(record)
    assert result == []
    assert record["node_inventory"] == []


def test_collect_heater_sample_addresses_fallbacks() -> None:
    """Ensure coordinator address fallbacks populate heater mapping."""

    class DummyCoordinator:
        def _addrs(self) -> list[str]:
            return [" 2 ", "", "01", "2"]

    record: dict[str, Any] = {"nodes": []}

    inventory, addr_map, compat = nodes_module.collect_heater_sample_addresses(
        record,
        coordinator=DummyCoordinator(),
    )

    assert inventory == []
    assert addr_map == {"htr": ["2", "01"]}
    assert compat["htr"] == "htr"

    proxy_record = MappingProxyType({"nodes": []})
    _, proxy_map, _ = nodes_module.collect_heater_sample_addresses(
        proxy_record,
        coordinator=DummyCoordinator(),
    )
    assert proxy_map == {"htr": ["2", "01"]}


def test_collect_heater_sample_addresses_uses_snapshot_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_nodes = {"nodes": [{"type": "htr", "addr": "1"}]}
    call_count = 0

    def fake_build(nodes: Any) -> list[Any]:
        nonlocal call_count
        call_count += 1
        return build_node_inventory(raw_nodes)

    monkeypatch.setattr(installation_module, "build_node_inventory", fake_build)

    snapshot = installation_module.InstallationSnapshot(
        dev_id="dev",
        raw_nodes=raw_nodes,
    )
    record = {"snapshot": snapshot}

    _, first_map, _ = nodes_module.collect_heater_sample_addresses(record)
    _, second_map, _ = nodes_module.collect_heater_sample_addresses(record)

    assert call_count <= 1
    expected_inventory = [
        (node.type, node.addr) for node in build_node_inventory(raw_nodes)
    ]
    actual_inventory = [(node.type, node.addr) for node in snapshot.inventory]
    assert actual_inventory == expected_inventory
    cached_map_one, _ = snapshot.heater_sample_address_map
    cached_map_two, _ = snapshot.heater_sample_address_map
    assert first_map == second_map
    assert cached_map_one == cached_map_two


def test_collect_heater_sample_addresses_prefers_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Collect heater sample addresses directly from a stored snapshot."""

    import importlib

    installation = importlib.import_module("custom_components.termoweb.installation")
    nodes = importlib.import_module("custom_components.termoweb.nodes")

    base_cls = installation.InstallationSnapshot
    call_order: list[str] = []

    class TrackingSnapshot(base_cls):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)

        @property
        def inventory(self) -> list[Any]:
            call_order.append("inventory")
            return super().inventory

        @property
        def heater_sample_address_map(self) -> tuple[dict[str, list[str]], dict[str, str]]:
            call_order.append("sample_map")
            return super().heater_sample_address_map

    monkeypatch.setattr(installation, "InstallationSnapshot", TrackingSnapshot)

    snapshot = installation.InstallationSnapshot(
        dev_id="dev",
        raw_nodes={"nodes": [{"type": "htr", "addr": "1"}]},
    )
    record = {"snapshot": snapshot}

    assert installation.ensure_snapshot(record) is snapshot

    inventory, addr_map, compat = nodes.collect_heater_sample_addresses(record)
    recorded_calls = list(call_order)

    expected_inventory = [(node.type, node.addr) for node in snapshot.inventory]
    expected_map, expected_compat = snapshot.heater_sample_address_map

    assert [(node.type, node.addr) for node in inventory] == expected_inventory
    assert addr_map == expected_map
    assert compat == expected_compat
    assert recorded_calls == ["inventory", "sample_map"]


def test_heater_sample_subscription_targets_orders_types() -> None:
    targets = heater_sample_subscription_targets({"acm": ["2"], "htr": ["1", "3"]})

    assert targets == [("htr", "1"), ("htr", "3"), ("acm", "2")]


def test_heater_sample_subscription_targets_handles_empty() -> None:
    assert heater_sample_subscription_targets({}) == []
    assert heater_sample_subscription_targets(None) == []
