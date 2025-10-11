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
prepare_heater_platform_data = heater_module.prepare_heater_platform_data
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


def test_prepare_heater_platform_data_groups_nodes() -> None:
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

    inventory, nodes_by_type, addrs_by_type, resolve_name = (
        prepare_heater_platform_data(
            entry_data,
            default_name_simple=lambda addr: f"Heater {addr}",
        )
    )

    assert inventory == container.nodes
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
    assert resolve_name("htr", "1") == "Lounge"
    assert resolve_name("htr", "4") == "Heater 4"
    assert resolve_name("acm", "2") == "Accumulator 2"

def test_prepare_heater_platform_data_skips_blank_types() -> None:
    nodes = [
        SimpleNamespace(type="  ", addr="5"),
        SimpleNamespace(type="htr", addr="6"),
    ]
    container = Inventory("dev", {"nodes": nodes}, nodes)

    inventory, nodes_by_type, addrs_by_type, _ = prepare_heater_platform_data(
        {"inventory": container},
        default_name_simple=lambda addr: f"Heater {addr}",
    )

    assert inventory == container.nodes
    assert [node.addr for node in nodes_by_type.get("htr", [])] == ["6"]
    assert addrs_by_type["htr"] == ["6"]


def test_prepare_heater_platform_data_passes_inventory_to_name_map(
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

    inventory, _, _, resolve_name = prepare_heater_platform_data(
        {"inventory": container},
        default_name_simple=lambda addr: f"Heater {addr}",
    )

    assert inventory == container.nodes
    assert calls
    recorded_inventory, recorded_factory = calls[0]
    assert recorded_inventory is container
    assert callable(recorded_factory)
    assert recorded_factory("9") == "Heater 9"

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


def test_prepare_heater_platform_data_resolves_normalized_inputs() -> None:
    raw_nodes = {
        "nodes": [
            {"type": " hTr ", "addr": " 8 ", "name": "Hall"},
        ]
    }
    container = Inventory("dev", raw_nodes, build_node_inventory(raw_nodes))

    _, _, _, resolve_name = prepare_heater_platform_data(
        {"inventory": container},
        default_name_simple=lambda addr: f"Heater {addr}",
    )

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

    def _fail_prepare(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("prepare_heater_platform_data should not run")

    monkeypatch.setattr(
        heater_module,
        "prepare_heater_platform_data",
        _fail_prepare,
    )

    nodes_by_type, addrs_by_type, resolve_name = heater_platform_details_for_entry(
        {"inventory": container},
        default_name_simple=lambda addr: f"Heater {addr}",
    )

    assert nodes_by_type == container.nodes_by_type
    assert addrs_by_type == {
        node_type: list(container.heater_address_map[0].get(node_type, []))
        for node_type in heater_module.HEATER_NODE_TYPES
    }
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


def test_prepare_heater_platform_data_uses_coordinator_inventory() -> None:
    raw_nodes = {"nodes": [{"type": "htr", "addr": "1", "name": " Lounge "}]}
    container = Inventory("dev", raw_nodes, build_node_inventory(raw_nodes))
    coordinator = SimpleNamespace(inventory=container)

    inventory, nodes_by_type, addrs_by_type, resolve_name = (
        prepare_heater_platform_data(
            {"coordinator": coordinator},
            default_name_simple=lambda addr: f"Heater {addr}",
        )
    )

    assert inventory == container.nodes
    assert nodes_by_type["htr"][0].name == "Lounge"
    assert addrs_by_type["htr"] == ["1"]
    assert resolve_name("htr", "1") == "Lounge"


def test_prepare_heater_platform_data_handles_missing_inventory(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("ERROR"):
        with pytest.raises(ValueError):
            prepare_heater_platform_data(
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


def test_prepare_heater_platform_data_handles_non_mapping_name_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_nodes = {"nodes": [{"type": "htr", "addr": "4"}]}
    container = Inventory("dev", raw_nodes, build_node_inventory(raw_nodes))

    monkeypatch.setattr(Inventory, "heater_name_map", lambda self, _: "invalid")

    _, _, _, resolve_name = prepare_heater_platform_data(
        {"inventory": container},
        default_name_simple=lambda addr: f"Heater {addr}",
    )

    assert resolve_name("htr", "4") == "Heater 4"


def test_log_skipped_nodes_defaults_platform_name(
    caplog: pytest.LogCaptureFixture,
) -> None:
    nodes_by_type = {"thm": [SimpleNamespace(addr="7")]}
    details = (
        nodes_by_type,
        {"thm": ["7"]},
        lambda node_type, addr: f"{node_type}-{addr}",
    )

    with caplog.at_level(logging.DEBUG):
        heater_module.log_skipped_nodes("", details, skipped_types=["thm"])

    messages = [record.message for record in caplog.records]
    assert any("platform" in message for message in messages)


def test_iter_nodes_yields_existing_node_objects() -> None:
    nodes = [HeaterNode(name="Living", addr="1")]

    yielded = list(heater_module._iter_nodes(nodes))
    assert yielded == nodes


def test_iter_heater_nodes_filters_addresses() -> None:
    nodes_by_type = {
        "htr": [SimpleNamespace(addr="1", type="htr"), SimpleNamespace(addr=" ")],
        "acm": [SimpleNamespace(addr="2", type="acm"), SimpleNamespace(addr=None)],
        "pmo": [SimpleNamespace(addr="3", type="pmo")],
    }
    details = (nodes_by_type, {"htr": ["1"], "acm": ["2"]}, lambda nt, addr: f"{nt}-{addr}")

    resolved: list[tuple[str, str]] = []

    def fake_resolve(node_type: str, addr: str) -> str:
        resolved.append((node_type, addr))
        return f"{node_type}-{addr}"

    yielded = list(iter_heater_nodes(details, fake_resolve))

    assert sorted([(node_type, addr) for node_type, _node, addr, _ in yielded]) == [
        ("acm", "2"),
        ("htr", "1"),
    ]
    assert sorted(resolved) == [("acm", "2"), ("htr", "1")]
    resolved.clear()

    resolved_acm: list[tuple[str, str]] = []

    def fake_resolve_acm(node_type: str, addr: str) -> str:
        resolved_acm.append((node_type, addr))
        return f"{node_type}-{addr}"

    only_acm = list(
        iter_heater_nodes(details, fake_resolve_acm, node_types=["acm", "thm"])
    )

    assert [(node_type, addr) for node_type, _node, addr, _ in only_acm] == [
        ("acm", "2")
    ]
    assert resolved_acm == [("acm", "2")]

    extra_resolved: list[tuple[str, str]] = []

    def extra_resolve(node_type: str, addr: str) -> str:
        extra_resolved.append((node_type, addr))
        return f"{node_type}-{addr}"

    mapping_nodes = {
        "htr": {"first": SimpleNamespace(addr="5")},
        "acm": SimpleNamespace(addr="6"),
        "pmo": [SimpleNamespace(addr=" None ")],
        "thm": "ignored",
    }
    mapping_details = (
        mapping_nodes,
        {"htr": ["5"], "acm": ["6"]},
        extra_resolve,
    )

    extra_results = list(
        iter_heater_nodes(
            mapping_details,
            extra_resolve,
            node_types=["htr", "acm", "pmo", "thm"],
        )
    )

    assert extra_results == [
        ("htr", mapping_nodes["htr"]["first"], "5", "htr-5"),
        ("acm", mapping_nodes["acm"], "6", "acm-6"),
    ]
    assert extra_resolved == [("htr", "5"), ("acm", "6")]

    blank_nodes: dict[str, list[SimpleNamespace] | None] = {"htr": None, "acm": []}
    blank_details = (blank_nodes, {"htr": [], "acm": []}, fake_resolve)
    assert list(iter_heater_nodes(blank_details, fake_resolve)) == []

    assert list(
        iter_heater_nodes(details, fake_resolve, node_types=["", "acm"])
    ) == [("acm", nodes_by_type["acm"][0], "2", "acm-2")]


def test_iter_boostable_heater_nodes_filters_support_and_types() -> None:
    nodes_by_type = {
        "htr": [
            SimpleNamespace(addr="1", supports_boost=True),
            SimpleNamespace(addr="2", supports_boost=False),
        ],
        "acm": [SimpleNamespace(addr="3", supports_boost=lambda: True)],
        "thm": [SimpleNamespace(addr="4", supports_boost=True)],
    }

    def resolve(node_type: str, addr: str) -> str:
        return f"{node_type}-{addr}"

    details = (
        nodes_by_type,
        {"htr": ["1", "2"], "acm": ["3"], "thm": ["4"]},
        resolve,
    )

    yielded = list(iter_boostable_heater_nodes(details, resolve))
    assert sorted((node_type, addr) for node_type, _node, addr, _ in yielded) == sorted(
        [("htr", "1"), ("acm", "3")]
    )

    accumulators_only = list(
        iter_boostable_heater_nodes(details, resolve, accumulators_only=True)
    )
    assert [(node_type, addr) for node_type, _node, addr, _ in accumulators_only] == [
        ("acm", "3")
    ]

    assert (
        list(
            iter_boostable_heater_nodes(
                details,
                resolve,
                node_types=["htr"],
                accumulators_only=True,
            )
        )
        == []
    )


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


def test_heater_base_async_added_without_hass() -> None:
    async def _run() -> None:
        coordinator = SimpleNamespace(hass=None)
        heater = _make_heater(coordinator)

        assert heater.hass is None
        await heater.async_added_to_hass()
        assert not heater._ws_subscription.is_connected

    asyncio.run(_run())


def test_device_available_accepts_inventory_metadata() -> None:
    coordinator = SimpleNamespace(hass=None)
    heater = _make_heater(coordinator)
    inventory = Inventory("dev", {}, [])

    assert not heater._device_available(None)
    assert not heater._device_available({})
    assert not heater._device_available({"inventory": object()})
    assert heater._device_available({"inventory": inventory})
    assert not heater._device_available({"nodes_by_type": {}})
    assert heater._device_available({"heater_address_map": {"forward": {"htr": ["A"]}}})
    assert heater._device_available({"addresses_by_type": {}})
    assert not heater._device_available({"settings": {"htr": {"A": {}}}})


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


def test_heater_section_handles_missing_device() -> None:
    hass = HomeAssistant()
    coordinator = SimpleNamespace(hass=hass, data={})
    heater = _make_heater(coordinator)

    assert heater._heater_section() == {}


def test_heater_section_falls_back_to_legacy_data() -> None:
    hass = HomeAssistant()
    coordinator = SimpleNamespace(
        hass=hass,
        data={"dev": {"htr": {"settings": {"A": {"mode": "auto"}}}}},
    )
    heater = HeaterNodeBase(
        coordinator, "entry", "dev", "A", "Heater A", node_type="acm"
    )

    section = heater._heater_section()
    assert section == {
        "addrs": ["A"],
        "settings": {"A": {"mode": "auto"}},
    }


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
