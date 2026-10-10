"""Tests for node models and identifier normalisation (nodes.py)."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from custom_components.termoweb.nodes import (
    AccumulatorNode,
    HeaterNode,
    Node,
    PowerMonitorNode,
    ThermostatNode,
    build_node_inventory,
    normalize_node_addr,
    normalize_node_type,
)


@pytest.mark.parametrize(
    ("value", "default", "use_default_when_falsey", "expected"),
    [
        (" HTR ", "htr", False, "htr"),
        ("AcM", "htr", False, "acm"),
        (None, "htr", True, "htr"),
        ("  ", "htr", False, "htr"),
        (None, "", False, "none"),
    ],
)
def test_normalize_node_type_cases(
    value: object, default: str, use_default_when_falsey: bool, expected: str
) -> None:
    """Node types are trimmed and lower-cased, with an optional default."""
    assert (
        normalize_node_type(
            value, default=default, use_default_when_falsey=use_default_when_falsey
        )
        == expected
    )


@pytest.mark.parametrize(
    ("value", "default", "use_default_when_falsey", "expected"),
    [
        (" 01 ", "", False, "01"),
        ("  ", "fallback", False, "fallback"),
        (None, "", True, ""),
        ("none", "", False, "none"),
    ],
)
def test_normalize_node_addr_cases(
    value: object, default: str, use_default_when_falsey: bool, expected: str
) -> None:
    """Addresses are trimmed strings, with an optional default."""
    assert (
        normalize_node_addr(
            value, default=default, use_default_when_falsey=use_default_when_falsey
        )
        == expected
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
