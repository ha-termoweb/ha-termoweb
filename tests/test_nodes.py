"""Unit tests for node models."""

from __future__ import annotations

import logging
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

import custom_components.termoweb.nodes as nodes_module
from custom_components.termoweb.nodes import (
    AccumulatorNode,
    HeaterNode,
    Node,
    PowerMonitorNode,
    ThermostatNode,
    build_node_inventory,
    heater_sample_subscription_targets,
    normalize_node_addr,
    normalize_node_type,
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


def test_heater_sample_subscription_targets_orders_types() -> None:
    targets = heater_sample_subscription_targets({"acm": ["2"], "htr": ["1", "3"]})

    assert targets == [("htr", "1"), ("htr", "3"), ("acm", "2")]


def test_heater_sample_subscription_targets_handles_empty() -> None:
    assert heater_sample_subscription_targets({}) == []
    assert heater_sample_subscription_targets(None) == []
