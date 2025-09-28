"""Unit tests for node models."""

from __future__ import annotations

import logging

import pytest

from custom_components.termoweb.nodes import (
    AccumulatorNode,
    HeaterNode,
    Node,
    PowerMonitorNode,
    ThermostatNode,
    build_node_inventory,
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


def test_build_node_inventory_tolerates_empty_payload() -> None:
    assert build_node_inventory({"nodes": []}) == []
