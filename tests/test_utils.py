from __future__ import annotations

import types
from typing import Any

import pytest

from custom_components.termoweb import nodes as nodes_module
from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.nodes import (
    HEATER_NODE_TYPES,
    addresses_by_node_type,
    build_heater_energy_unique_id,
    build_node_inventory,
    ensure_node_inventory,
    normalize_heater_addresses,
    normalize_node_addr,
    normalize_node_type,
    parse_heater_energy_unique_id,
)
from custom_components.termoweb.utils import (
    _entry_gateway_record,
    build_gateway_device_info,
    extract_heater_addrs,
    float_or_none,
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


def test_ensure_node_inventory_skips_node_like_entries_with_missing_fields() -> None:
    class FakeNode:
        def __init__(self) -> None:
            self.type = ""
            self.addr = "valid"

        def as_dict(self) -> dict[str, Any]:  # pragma: no cover - minimal stub
            return {}

    raw = {"nodes": [{"type": "htr", "addr": "A"}]}
    cached = build_node_inventory(raw)
    cached.append(FakeNode())
    record = {"node_inventory": cached, "nodes": {}}

    result = ensure_node_inventory(record)

    assert [node.addr for node in result] == ["A"]


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


@pytest.mark.parametrize(
    "value,default,use_default_when_falsey,expected",
    [
        (" HTR ", "htr", False, "htr"),
        ("AcM", "htr", False, "acm"),
        (None, "htr", True, "htr"),
        ("  ", "htr", False, "htr"),
        (None, "", False, "none"),
    ],
)
def test_normalize_node_type_cases(
    value: Any, default: str, use_default_when_falsey: bool, expected: str
) -> None:
    assert (
        normalize_node_type(
            value,
            default=default,
            use_default_when_falsey=use_default_when_falsey,
        )
        == expected
    )


@pytest.mark.parametrize(
    "value,default,use_default_when_falsey,expected",
    [
        (" 01 ", "", False, "01"),
        ("  ", "fallback", False, "fallback"),
        (None, "", True, ""),
        (None, "fallback", False, "None"),
        ("none", "", False, "none"),
    ],
)
def test_normalize_node_addr_cases(
    value: Any, default: str, use_default_when_falsey: bool, expected: str
) -> None:
    assert (
        normalize_node_addr(
            value,
            default=default,
            use_default_when_falsey=use_default_when_falsey,
        )
        == expected
    )


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


def test_build_heater_energy_unique_id_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object, dict[str, Any]]] = []

    original_normalize_type = nodes_module.normalize_node_type
    original_normalize_addr = nodes_module.normalize_node_addr

    def _record_type(value, **kwargs):
        calls.append(("type", value, kwargs))
        return original_normalize_type(value, **kwargs)

    def _record_addr(value, **kwargs):
        calls.append(("addr", value, kwargs))
        return original_normalize_addr(value, **kwargs)

    monkeypatch.setattr(nodes_module, "normalize_node_type", _record_type)
    monkeypatch.setattr(nodes_module, "normalize_node_addr", _record_addr)

    unique_id = build_heater_energy_unique_id(" dev ", " ACM ", " 01 ")

    assert unique_id == f"{DOMAIN}:dev:acm:01:energy"
    assert parse_heater_energy_unique_id(unique_id) == ("dev", "acm", "01")
    assert calls == [
        ("addr", " dev ", {}),
        ("type", " ACM ", {}),
        ("addr", " 01 ", {}),
    ]


@pytest.mark.parametrize(
    "dev_id, node_type, addr",
    [
        ("", "htr", "01"),
        ("dev", "", "01"),
        ("dev", "htr", ""),
        ("dev", " ", "01"),
        ("dev", "htr", "  "),
    ],
)
def test_build_heater_energy_unique_id_requires_components(
    dev_id: str, node_type: str, addr: str
) -> None:
    with pytest.raises(ValueError):
        build_heater_energy_unique_id(dev_id, node_type, addr)


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "not-domain:dev:htr:01:energy",
        f"{DOMAIN}:dev:htr:01:power",
        f"{DOMAIN}:dev:htr:energy",
        f"{DOMAIN}:dev:htr",
        f"{DOMAIN}:dev::01:energy",
    ],
)
def test_parse_heater_energy_unique_id_invalid(value) -> None:
    assert parse_heater_energy_unique_id(value) is None


def test_extract_heater_addrs_filters_duplicates_and_types() -> None:
    payload = {
        "nodes": [
            {"type": "htr", "addr": " A "},
            {"type": "acm", "addr": "B"},
            {"type": "pmo", "addr": "p1"},
            {"type": "htr", "addr": "a"},
            {"type": "thm", "addr": "t"},
            {"type": "htr", "addr": "  "},
        ]
    }

    assert extract_heater_addrs(payload) == ["A", "B"]


@pytest.mark.parametrize("payload", [None, [], "invalid", {"nodes": "bad"}])
def test_extract_heater_addrs_handles_invalid_payload(payload: Any) -> None:
    assert extract_heater_addrs(payload) == []
