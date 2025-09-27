from __future__ import annotations

import types
from typing import Any

import pytest

from custom_components.termoweb.nodes import build_node_inventory

from custom_components.termoweb.utils import (
    HEATER_NODE_TYPES,
    addresses_by_node_type,
    addresses_by_type,
    ensure_node_inventory,
    float_or_none,
)


def test_addresses_by_type_filters_and_deduplicates() -> None:
    inventory = build_node_inventory(
        {
            "nodes": [
                {"type": "htr", "addr": "A"},
                {"type": "foo", "addr": "B"},
                {"type": "acm", "addr": 1},
                {"type": "HTR", "addr": "A"},
            ]
        }
    )

    assert addresses_by_type(inventory, HEATER_NODE_TYPES) == ["A", "1"]


def test_addresses_by_type_handles_missing_types() -> None:
    inventory = build_node_inventory({"nodes": [{"type": "htr", "addr": "A"}]})

    assert addresses_by_type(inventory, [None]) == []


def test_ensure_node_inventory_returns_cached_copy() -> None:
    raw = {"nodes": [{"type": "htr", "addr": "A"}]}
    cached = build_node_inventory(raw)
    record = {"node_inventory": cached, "nodes": raw}

    result = ensure_node_inventory(record)

    assert result == cached
    assert result is not cached


def test_ensure_node_inventory_filters_invalid_cached_entries() -> None:
    raw = {"nodes": [{"type": "htr", "addr": "A"}]}
    cached = build_node_inventory(raw)
    cached.append(types.SimpleNamespace(type="", addr="bad"))
    record = {"node_inventory": cached, "nodes": {}}

    result = ensure_node_inventory(record)

    assert [node.addr for node in result] == ["A"]
    assert record.get("node_inventory") == result


def test_ensure_node_inventory_builds_and_caches() -> None:
    raw = {"nodes": [{"type": "acm", "addr": "B"}]}
    record: dict[str, Any] = {"nodes": raw}
    result = ensure_node_inventory(record)

    assert [node.addr for node in result] == ["B"]
    assert [node.addr for node in record.get("node_inventory", [])] == ["B"]

    record["nodes"] = object()
    cached = ensure_node_inventory(record)
    assert cached == result


def test_ensure_node_inventory_falls_back_to_record_nodes() -> None:
    arg_nodes = {"nodes": []}
    record_nodes = {"nodes": [{"type": "acm", "addr": "C"}]}
    record: dict[str, Any] = {"nodes": record_nodes}
    result = ensure_node_inventory(record, nodes=arg_nodes)

    assert [node.addr for node in result] == ["C"]
    assert [node.addr for node in record.get("node_inventory", [])] == ["C"]


def test_addresses_by_node_type_skips_invalid_entries() -> None:
    nodes = [
        types.SimpleNamespace(type=" ", addr="skip"),
        types.SimpleNamespace(type="acm", addr=""),
        types.SimpleNamespace(type="acm", addr="B"),
        types.SimpleNamespace(type="acm", addr="B"),
    ]

    mapping, unknown = addresses_by_node_type(nodes, known_types=["htr"])
    assert mapping == {"acm": ["B"]}
    assert unknown == {"acm"}


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        ("abc", None),
        ("123", 123.0),
        (5, 5.0),
        ("   ", None),
        (float("nan"), None),
        (float("inf"), None),
    ],
)
def test_float_or_none(value, expected) -> None:
    assert float_or_none(value) == expected


@pytest.mark.parametrize("value", ["nan", "inf"])
def test_float_or_none_non_finite_strings(value) -> None:
    assert float_or_none(value) is None


def test_get_brand_api_base_fallback() -> None:
    from custom_components.termoweb import const

    assert const.get_brand_api_base("unknown-brand") == const.API_BASE
