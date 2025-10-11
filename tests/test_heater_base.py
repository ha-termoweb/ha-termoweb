from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any, Callable
import pytest

from conftest import _install_stubs, make_ws_payload

_install_stubs()

from custom_components.termoweb import heater as heater_module
from custom_components.termoweb import identifiers as identifiers_module
from custom_components.termoweb.inventory import (
    HeaterNode,
    Inventory,
    build_node_inventory,
)
from homeassistant.core import HomeAssistant

HeaterNodeBase = heater_module.HeaterNodeBase
build_heater_name_map = heater_module.build_heater_name_map
iter_heater_nodes = heater_module.iter_heater_nodes
iter_boostable_heater_nodes = heater_module.iter_boostable_heater_nodes
heater_platform_details_for_entry = heater_module.heater_platform_details_for_entry


def test_build_heater_entity_unique_id_normalises_inputs() -> None:
    """Helper should normalise identifiers and enforce required fields."""

    uid = identifiers_module.build_heater_entity_unique_id(
        " 0A1B ",
        " ACM ",
        " 07 ",
        "boost",
    )
    assert uid == "termoweb:0A1B:acm:07:boost"

    uid_with_colon = identifiers_module.build_heater_entity_unique_id(
        "0a1b",
        "acm",
        "07",
        ":boost",
    )
    assert uid_with_colon == "termoweb:0a1b:acm:07:boost"

    with pytest.raises(ValueError):
        identifiers_module.build_heater_entity_unique_id("", "acm", "07")


def _make_heater(coordinator: SimpleNamespace) -> HeaterNodeBase:
    return HeaterNodeBase(coordinator, "entry", "dev", "A", "Heater A")


def test_heater_hass_accessors_fall_back_to_coordinator() -> None:
    """Heater nodes should defer hass references to the coordinator when unset."""

    hass = HomeAssistant()
    coordinator = SimpleNamespace(hass=hass)
    heater = _make_heater(coordinator)

    if hasattr(heater, "_hass"):
        delattr(heater, "_hass")

    assert heater.hass is hass
    assert heater._hass_for_runtime() is hass

    override = object()
    heater.hass = override

    assert heater.hass is override
    assert heater._hass_for_runtime() is override


def test_heater_platform_details_for_entry_groups_nodes() -> None:
    raw_nodes = {
        "nodes": [
            {"type": "HTR", "addr": "1", "name": " Lounge "},
            {"type": "acm", "addr": "2"},
            {"type": "thm", "addr": "3"},
            {"type": "htr", "addr": "4"},
            {"type": "HTR", "addr": "4"},
            {"type": "ACM", "addr": "2"},
        ]
    }
    inventory_nodes = build_node_inventory(raw_nodes)
    container = Inventory("dev", raw_nodes, inventory_nodes)
    entry_data = {"inventory": container}

    details = heater_platform_details_for_entry(
        entry_data,
        default_name_simple=lambda addr: f"Heater {addr}",
    )

    assert details.inventory is container
    nodes_by_type = details.nodes_by_type
    addrs_by_type = details.addrs_by_type
    htr_nodes = nodes_by_type.get("htr", [])
    assert [node.addr for node in htr_nodes] == ["1", "4", "4"]
    assert all(hasattr(node, "addr") for node in htr_nodes)
    assert addrs_by_type["htr"] == ["1", "4"]
    assert len(addrs_by_type["htr"]) == len(set(addrs_by_type["htr"]))
    acm_nodes = nodes_by_type.get("acm", [])
    assert [node.addr for node in acm_nodes] == ["2", "2"]
    assert addrs_by_type["acm"] == ["2"]
    assert len(addrs_by_type["acm"]) == len(set(addrs_by_type["acm"]))
    reference = Inventory("dev", raw_nodes, inventory_nodes)
    helper_map, helper_reverse = reference.heater_address_map
    assert addrs_by_type == {
        node_type: helper_map.get(node_type, [])
        for node_type in heater_module.HEATER_NODE_TYPES
    }
    assert helper_reverse == {"1": {"htr"}, "2": {"acm"}, "4": {"htr"}}
    resolve_name = details.resolve_name
    assert resolve_name("htr", "1") == "Lounge"
    assert resolve_name("htr", "4") == "Heater 4"
    assert resolve_name("acm", "2") == "Accumulator 2"


def test_heater_platform_details_for_entry_skips_blank_types() -> None:
    nodes = [
        SimpleNamespace(type="  ", addr="5"),
        SimpleNamespace(type="htr", addr="6"),
    ]
    container = Inventory("dev", {"nodes": nodes}, nodes)

    details = heater_platform_details_for_entry(
        {"inventory": container},
        default_name_simple=lambda addr: f"Heater {addr}",
    )

    assert details.inventory is container
    assert [node.addr for node in details.nodes_by_type.get("htr", [])] == ["6"]
    assert details.addrs_by_type["htr"] == ["6"]


def test_heater_platform_details_for_entry_passes_inventory_to_name_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_nodes = {
        "nodes": [
            {"type": "htr", "addr": "9"},
            {"type": "htr", "addr": "7"},
            {"type": "htr", "addr": "6"},
            {"type": "acm", "addr": "5", "name": "Heater 5"},
            {"type": "acm", "addr": "8"},
        ]
    }
    inventory_nodes = build_node_inventory(raw_nodes)
    container = Inventory("dev", raw_nodes, inventory_nodes)

    calls: list[tuple[Inventory, Callable[[str], str] | None]] = []
    custom_map: dict[Any, Any] = {
        "by_type": {
            "htr": {"9": "By Type 9"},
            "acm": {"5": "Heater 5", "8": "Heater 8"},
        },
        ("htr", "7"): "Pair 7",
        "htr": {"6": "Legacy 6"},
    }

    def fake_name_map(
        self: Inventory, default_factory: Callable[[str], str] | None = None
    ) -> dict[Any, Any]:
        calls.append((self, default_factory))
        return custom_map

    monkeypatch.setattr(Inventory, "heater_name_map", fake_name_map)

    details = heater_platform_details_for_entry(
        {"inventory": container},
        default_name_simple=lambda addr: f"Heater {addr}",
    )

    assert details.inventory is container
    assert calls
    recorded_inventory, recorded_factory = calls[0]
    assert recorded_inventory is container
    assert callable(recorded_factory)
    assert recorded_factory("9") == "Heater 9"

    resolve_name = details.resolve_name
    assert resolve_name("htr", "9") == "By Type 9"
    assert resolve_name("htr", "7") == "Pair 7"
    assert resolve_name("htr", "6") == "Legacy 6"
    assert resolve_name("acm", "5") == "Heater 5"
    assert resolve_name("acm", "8") == "Accumulator 8"
    assert resolve_name("foo", "3") == "Heater 3"


def test_build_heater_name_map_handles_invalid_entries() -> None:
    nodes = {
        "nodes": [
            123,
            {"type": "HTR", "addr": None, "name": "Ignored"},
            {"type": "foo", "addr": "B", "name": "Skip"},
            {"type": "htr", "addr": 5, "name": "  "},
            {"type": "htr", "addr": "6", "name": None},
            {"type": "htr", "addr": " None ", "name": "Skip None"},
        ]
    }

    result = build_heater_name_map(nodes, lambda addr: f"Heater {addr}")

    assert result.get(("htr", "5")) == "Heater 5"
    assert result.get(("htr", "6")) == "Heater 6"
    assert result.get("htr") == {"5": "Heater 5", "6": "Heater 6"}
    assert result.get("by_type", {}).get("htr") == {"5": "Heater 5", "6": "Heater 6"}


def test_build_heater_name_map_accepts_iterables_of_dicts() -> None:
    nodes_iter = (
        {"type": "htr", "addr": "1"},
        {"type": "acm", "addr": "2"},
    )

    result = build_heater_name_map(nodes_iter, lambda addr: f"Heater {addr}")

    assert result.get(("acm", "2")) == "Heater 2"
    assert result.get("htr", {}).get("1") == "Heater 1"


def test_heater_platform_details_for_entry_resolves_normalized_inputs() -> None:
    raw_nodes = {
        "nodes": [
            {"type": " hTr ", "addr": " 8 ", "name": "Hall"},
        ]
    }
    container = Inventory("dev", raw_nodes, build_node_inventory(raw_nodes))

    resolve_name = heater_platform_details_for_entry(
        {"inventory": container},
        default_name_simple=lambda addr: f"Heater {addr}",
    ).resolve_name

    assert resolve_name(" HTR ", " 8 ") == "Hall"


def test_heater_platform_details_for_entry_prefers_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_nodes = {
        "nodes": [
            {"type": "htr", "addr": "1", "name": " Lounge "},
            {"type": "acm", "addr": "2"},
        ]
    }
    container = Inventory("dev", raw_nodes, build_node_inventory(raw_nodes))

    details = heater_platform_details_for_entry(
        {"inventory": container},
        default_name_simple=lambda addr: f"Heater {addr}",
    )

    assert details.nodes_by_type == container.nodes_by_type
    assert details.addrs_by_type == {
        node_type: list(container.heater_address_map[0].get(node_type, []))
        for node_type in heater_module.HEATER_NODE_TYPES
    }
    resolve_name = details.resolve_name
    assert resolve_name("htr", "1") == "Lounge"
    assert resolve_name("acm", "2") == "Accumulator 2"


def test_heater_platform_details_for_entry_requires_inventory() -> None:
    with pytest.raises(ValueError):
        heater_platform_details_for_entry(
            {"dev_id": "missing"},
            default_name_simple=lambda addr: f"Heater {addr}",
        )


def test_heater_platform_details_for_entry_rejects_non_mapping_entry() -> None:
    with pytest.raises(ValueError):
        heater_platform_details_for_entry(
            SimpleNamespace(dev_id="ignored"),
            default_name_simple=lambda addr: f"Heater {addr}",
        )


def test_heater_platform_details_for_entry_uses_coordinator_inventory() -> None:
    raw_nodes = {"nodes": [{"type": "htr", "addr": "1", "name": " Lounge "}]}
    container = Inventory("dev", raw_nodes, build_node_inventory(raw_nodes))
    coordinator = SimpleNamespace(inventory=container)

    details = heater_platform_details_for_entry(
        {"coordinator": coordinator},
        default_name_simple=lambda addr: f"Heater {addr}",
    )

    assert details.inventory is container
    assert details.nodes_by_type["htr"][0].name == "Lounge"
    assert details.addrs_by_type["htr"] == ["1"]
    assert details.resolve_name("htr", "1") == "Lounge"


def test_heater_platform_details_for_entry_logs_missing_inventory(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("ERROR"):
        with pytest.raises(ValueError):
            heater_platform_details_for_entry(
                {"dev_id": "dev"},
                default_name_simple=lambda addr: f"Heater {addr}",
            )

    assert any("missing inventory" in message for message in caplog.messages)


def test_extract_inventory_accepts_mapping_inventory() -> None:
    raw_nodes = {"nodes": [{"type": "htr", "addr": "7"}, {"type": "acm", "addr": "8"}]}
    container = Inventory("dev", raw_nodes, build_node_inventory(raw_nodes))
    entry_data = {"inventory": container}

    inventory = heater_module._extract_inventory(entry_data)

    assert inventory is container


def test_extract_inventory_handles_non_mapping_input() -> None:
    assert heater_module._extract_inventory(None) is None
    assert heater_module._extract_inventory(object()) is None


def test_extract_inventory_uses_hass_data() -> None:
    raw_nodes = {"nodes": [{"type": "htr", "addr": "1"}]}
    container = Inventory("dev", raw_nodes, build_node_inventory(raw_nodes))
    hass = SimpleNamespace(data={heater_module.DOMAIN: {"entry": {"inventory": container}}})

    entry_data = {"hass": hass, "entry_id": "entry"}

    inventory = heater_module._extract_inventory(entry_data)

    assert inventory is container




def test_iter_heater_maps_deduplicates_sections() -> None:
    htr_settings = {"1": {"mode": "auto"}}
    acm_settings = {"2": {"mode": "charge"}}
    cache = {
        "settings": {"htr": htr_settings, "acm": acm_settings},
        "htr": {"settings": htr_settings},
        "acm": {"settings": acm_settings},
    }

    results = list(
        heater_module.iter_heater_maps(
            cache,
            map_key="settings",
            node_types=["", "htr", "acm", "htr"],
        )
    )

    assert len(results) == 2
    assert results[0] is htr_settings
    assert results[1] is acm_settings


def test_iter_heater_maps_accepts_string_node_type() -> None:
    cache = {"settings": {"htr": {"1": {"mode": "auto"}}}}

    results = list(
        heater_module.iter_heater_maps(
            cache,
            map_key="settings",
            node_types="htr",
        )
    )

    assert len(results) == 1
    assert results[0] == {"1": {"mode": "auto"}}


def test_iter_heater_maps_requires_truthy_key() -> None:
    cache = {"settings": {"htr": {"1": {"mode": "auto"}}}}

    assert list(heater_module.iter_heater_maps(cache, map_key="")) == []


@pytest.mark.parametrize(
    ("node_type", "expected_type"),
    [("htr", "htr"), ("ACM", "acm")],
)
def test_heater_base_unique_id_includes_node_type(
    node_type: str, expected_type: str
) -> None:
    coordinator = SimpleNamespace(hass=None)
    heater = HeaterNodeBase(
        coordinator,
        "entry",
        "dev",
        "A",
        "Heater A",
        node_type=node_type,
    )

    expected = f"{heater_module.DOMAIN}:dev:{expected_type}:{heater._addr}"
    assert heater._attr_unique_id == expected
    assert heater._node_type == expected_type
    if hasattr(heater, "unique_id"):
        assert heater.unique_id == expected


def test_payload_matching_honours_node_type() -> None:
    coordinator = SimpleNamespace(hass=None)
    heater = HeaterNodeBase(
        coordinator,
        "entry",
        "dev",
        "A",
        "Accumulator A",
        node_type="acm",
    )

    assert heater._payload_matches_heater(make_ws_payload("dev", "A", node_type="acm"))
    assert heater._payload_matches_heater(make_ws_payload("dev", "A", node_type="ACM"))
    assert heater._payload_matches_heater(make_ws_payload("dev", None, node_type="acm"))
    assert not heater._payload_matches_heater(
        make_ws_payload("dev", "A", node_type="htr")
    )
    assert not heater._payload_matches_heater(
        make_ws_payload("dev", "B", node_type="acm")
    )
    assert not heater._payload_matches_heater(
        make_ws_payload("other", "A", node_type="acm")
    )






class _FakeDict(dict):
    """Dictionary that exposes a non-callable ``get`` attribute."""

    get = "not-callable"


def test_device_record_fallback_dict() -> None:
    hass = HomeAssistant()
    coordinator = SimpleNamespace(hass=hass, data=_FakeDict({"dev": {"nodes": "ok"}}))
    heater = _make_heater(coordinator)

    assert heater._device_record() == {"nodes": "ok"}


def test_device_record_unknown_structure() -> None:
    hass = HomeAssistant()
    coordinator = SimpleNamespace(hass=hass, data=SimpleNamespace())
    heater = _make_heater(coordinator)

    assert heater._device_record() is None




def test_heater_settings_missing_mapping() -> None:
    hass = HomeAssistant()
    coordinator = SimpleNamespace(hass=hass, data={"dev": {"htr": {"settings": []}}})
    heater = _make_heater(coordinator)

    assert heater.heater_settings() is None


def test_supports_boost_helper_handles_variants(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify the boost helper supports callables, booleans, and errors."""

    true_node = SimpleNamespace(supports_boost=True)
    assert heater_module.supports_boost(true_node) is True

    callable_node = SimpleNamespace(supports_boost=lambda: " yes ")
    assert heater_module.supports_boost(callable_node) is True

    false_node = SimpleNamespace(supports_boost=False)
    assert heater_module.supports_boost(false_node) is False

    class RaisingSupports:
        def __call__(self) -> bool:
            raise RuntimeError("boom")

    caplog.set_level("DEBUG")
    error_node = SimpleNamespace(supports_boost=RaisingSupports(), addr="boom")
    assert heater_module.supports_boost(error_node) is False
    assert "boost support probe failure" in caplog.text
