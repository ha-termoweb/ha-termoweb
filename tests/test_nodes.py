"""Unit tests for node models."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from conftest import _install_stubs

_install_stubs()

import custom_components.termoweb.inventory as inventory_module
from custom_components.termoweb.inventory import (
    AccumulatorNode,
    build_node_inventory,
    HeaterNode,
    Node,
    PowerMonitorNode,
    ThermostatNode,
    normalize_node_addr,
    normalize_node_type,
)
from custom_components.termoweb.inventory import heater_sample_subscription_targets


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


def test_power_monitor_default_name_and_sample_target() -> None:
    node = PowerMonitorNode(name="Monitor", addr="P1")

    assert node.sample_target() == ("pmo", "P1")
    assert node.default_name() == "Monitor"

    node.name = ""
    assert node.default_name() == "Power Monitor P1"


def test_thermostat_node_type() -> None:
    assert ThermostatNode(name="Thermostat", addr="T1").type == "thm"


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


def test_node_as_dict() -> None:
    node = HeaterNode(name="Kitchen", addr=5)

    assert node.as_dict() == {
        "name": "Kitchen",
        "addr": "5",
        "type": "htr",
    }


def test_build_node_inventory_handles_mixed_types(
    caplog: pytest.LogCaptureFixture,
) -> None:
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


def test_build_node_inventory_skips_missing_or_none_type(
    caplog: pytest.LogCaptureFixture,
) -> None:
    payload = {
        "nodes": [
            {"type": "htr", "addr": "01", "name": "Heater"},
            {"addr": "02", "name": "Missing type"},
            {"type": None, "addr": "03", "name": "Null type"},
        ]
    }

    with caplog.at_level(logging.DEBUG):
        nodes = build_node_inventory(payload)

    assert [node.addr for node in nodes] == ["01"]
    assert any(
        "Skipping node with missing type" in message for message in caplog.messages
    )
    assert [node.addr for node in build_node_inventory(payload["nodes"])] == ["01"]


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {"nodes": {"htr": {"addrs": ["1"], "settings": {"1": {"name": "Hall"}}}}},
        {"htr": {"addrs": ["1"], "settings": {"1": {"name": "Hall"}}}},
        {"nodes_by_type": {"htr": {"addrs": ["1"]}}},
    ],
)
def test_build_node_inventory_ignores_non_api_shapes(payload: Any) -> None:
    """Legacy section-shaped payloads (v1 coordinator dicts) yield no nodes."""

    assert build_node_inventory(payload) == []


def test_build_node_inventory_ignores_legacy_key_aliases() -> None:
    """``node_type``/``address``/``title`` aliases are not wire fields."""

    payload = {
        "nodes": [
            {"node_type": "htr", "addr": "1"},
            {"type": "htr", "address": "2"},
            {"type": "htr", "addr": "3", "title": "Hall"},
            {"type": "power_monitor", "addr": "4"},
        ]
    }

    nodes = build_node_inventory(payload)

    assert [(node.type, node.addr, node.name) for node in nodes] == [
        ("htr", "3", ""),
        ("power_monitor", "4", ""),
    ]
    assert type(nodes[1]) is Node


def test_normalize_node_addr_rejects_none() -> None:
    """``None`` must not become the address string ``"None"``."""

    with pytest.raises(ValueError):
        normalize_node_addr(None)
    with pytest.raises(ValueError):
        normalize_node_addr(None, default="1")
    assert normalize_node_addr(None, use_default_when_falsey=True) == ""
    assert normalize_node_addr(None, default="1", use_default_when_falsey=True) == "1"


def test_node_rejects_none_addr() -> None:
    with pytest.raises(ValueError):
        HeaterNode(name="Living", addr=None)  # type: ignore[arg-type]


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


def test_heater_sample_subscription_targets_orders_types() -> None:
    targets = heater_sample_subscription_targets({"acm": ["2"], "htr": ["1", "3"]})

    assert targets == [("htr", "1"), ("htr", "3"), ("acm", "2")]


def test_heater_sample_subscription_targets_handles_empty() -> None:
    assert heater_sample_subscription_targets({}) == []
    assert heater_sample_subscription_targets(None) == []


def test_heater_sample_subscription_targets_skips_thermostats() -> None:
    targets = heater_sample_subscription_targets({"thm": ["5"], "htr": ["1"]})

    assert targets == [("htr", "1")]


def test_heater_node_types_include_thermostat() -> None:
    assert "thm" in inventory_module.HEATER_NODE_TYPES
