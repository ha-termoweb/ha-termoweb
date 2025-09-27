from __future__ import annotations

import types

import pytest

from custom_components.termoweb.nodes import build_node_inventory

from custom_components.termoweb.utils import (
    HEATER_NODE_TYPES,
    addresses_by_node_type,
    addresses_by_type,
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
    inventory = build_node_inventory(
        {"nodes": [{"type": "htr", "addr": "A"}]}
    )

    assert addresses_by_type(inventory, [None]) == []


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
