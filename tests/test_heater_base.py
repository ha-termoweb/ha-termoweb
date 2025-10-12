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
    AccumulatorNode,
    HeaterNode,
    Inventory,
    build_node_inventory,
)
from homeassistant.core import HomeAssistant

HeaterNodeBase = heater_module.HeaterNodeBase
iter_boostable_heater_nodes = heater_module.iter_boostable_heater_nodes
heater_platform_details_for_entry = heater_module.heater_platform_details_for_entry
resolve_entry_inventory = heater_module.resolve_entry_inventory


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
    assert not calls
    resolve_name = details.resolve_name
    assert resolve_name("htr", "9") == "By Type 9"
    assert calls
    recorded_inventory, recorded_factory = calls[0]
    assert recorded_inventory is container
    assert callable(recorded_factory)
    assert recorded_factory("9") == "Heater 9"

    assert resolve_name("htr", "7") == "Pair 7"
    assert resolve_name("htr", "6") == "Legacy 6"
    assert resolve_name("acm", "5") == "Heater 5"
    assert resolve_name("acm", "8") == "Accumulator 8"
    assert resolve_name("foo", "3") == "Heater 3"


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


def test_heater_platform_details_iter_metadata_exposes_nodes() -> None:
    raw_nodes = {
        "nodes": [
            {"type": "htr", "addr": "1", "name": "Lounge"},
            {"type": "acm", "addr": "2"},
        ]
    }
    container = Inventory("dev", raw_nodes, build_node_inventory(raw_nodes))

    details = heater_platform_details_for_entry(
        {"inventory": container},
        default_name_simple=lambda addr: f"Heater {addr}",
    )

    metadata = list(details.iter_metadata())

    assert metadata == [
        ("htr", container.nodes_by_type["htr"][0], "1", "Lounge"),
        ("acm", container.nodes_by_type["acm"][0], "2", "Accumulator 2"),
    ]


def test_iter_boostable_heater_nodes_yields_accumulators() -> None:
    raw_nodes = {
        "nodes": [
            HeaterNode(name="Heater", addr="1"),
            AccumulatorNode(name="Storage", addr="2"),
        ]
    }
    container = Inventory("dev", raw_nodes, raw_nodes["nodes"])
    details = heater_platform_details_for_entry(
        {"inventory": container},
        default_name_simple=lambda addr: f"Heater {addr}",
    )

    results = list(iter_boostable_heater_nodes(details))

    assert results == [
        ("acm", container.nodes_by_type["acm"][0], "2", "Storage")
    ]


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


def test_resolve_entry_inventory_prefers_mapping() -> None:
    raw_nodes = {"nodes": [{"type": "htr", "addr": "1"}]}
    container = Inventory("dev", raw_nodes, build_node_inventory(raw_nodes))

    resolved = resolve_entry_inventory({"inventory": container})

    assert resolved is container


def test_resolve_entry_inventory_uses_coordinator() -> None:
    raw_nodes = {"nodes": [{"type": "acm", "addr": "2"}]}
    container = Inventory("dev", raw_nodes, build_node_inventory(raw_nodes))
    coordinator = SimpleNamespace(inventory=container)

    resolved = resolve_entry_inventory({"coordinator": coordinator})

    assert resolved is container


def test_resolve_entry_inventory_rejects_invalid_input() -> None:
    assert resolve_entry_inventory(None) is None
    assert resolve_entry_inventory(object()) is None


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
