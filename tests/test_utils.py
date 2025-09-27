from __future__ import annotations

import types
from typing import Any

import pytest

from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.nodes import build_node_inventory

from custom_components.termoweb.utils import (
    HEATER_NODE_TYPES,
    _entry_gateway_record,
    addresses_by_node_type,
    build_gateway_device_info,
    ensure_node_inventory,
    float_or_none,
    normalize_heater_addresses,
)


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


def test_ensure_node_inventory_sets_empty_cache_when_missing() -> None:
    class LazyDict(dict):
        def get(self, key, default=None):
            if key == "node_inventory" and key not in self:
                return []
            return super().get(key, default)

    record = LazyDict()
    result = ensure_node_inventory(record, nodes=None)

    assert result == []
    assert record["node_inventory"] == []


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


def test_entry_gateway_record_handles_invalid_sources() -> None:
    assert _entry_gateway_record(None, "entry") is None

    hass = types.SimpleNamespace(data={DOMAIN: {}})
    assert _entry_gateway_record(hass, None) is None

    hass = types.SimpleNamespace(data={DOMAIN: []})
    assert _entry_gateway_record(hass, "entry") is None

    hass = types.SimpleNamespace(data={DOMAIN: {"entry": []}})
    assert _entry_gateway_record(hass, "entry") is None


def test_build_gateway_device_info_defaults_without_entry() -> None:
    hass = types.SimpleNamespace(data={})

    info = build_gateway_device_info(hass, "entry", "dev")

    assert info["identifiers"] == {(DOMAIN, "dev")}
    assert info["manufacturer"] == "TermoWeb"
    assert "sw_version" not in info


def test_build_gateway_device_info_uses_brand_and_version() -> None:
    hass = types.SimpleNamespace(
        data={DOMAIN: {"entry": {"brand": "  Ducaheat  ", "version": 7}}}
    )

    info = build_gateway_device_info(hass, "entry", "dev")

    assert info["manufacturer"] == "Ducaheat"
    assert info["sw_version"] == "7"


def test_normalize_heater_addresses_with_none() -> None:
    mapping, aliases = normalize_heater_addresses(None)

    assert mapping == {"htr": []}
    assert aliases == {"htr": "htr"}


def test_get_brand_api_base_fallback() -> None:
    from custom_components.termoweb import const

    assert const.get_brand_api_base("unknown-brand") == const.API_BASE
