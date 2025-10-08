from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from conftest import _install_stubs, make_ws_payload

_install_stubs()

from custom_components.termoweb import heater as heater_module
from custom_components.termoweb import installation as installation_module
from custom_components.termoweb.installation import InstallationSnapshot
from custom_components.termoweb.heater_inventory import build_heater_inventory_details
from custom_components.termoweb.nodes import (
    HeaterNode,
    build_node_inventory,
)
from homeassistant.core import HomeAssistant

HeaterNodeBase = heater_module.HeaterNodeBase
build_heater_name_map = heater_module.build_heater_name_map
iter_heater_nodes = heater_module.iter_heater_nodes
iter_boostable_heater_nodes = heater_module.iter_boostable_heater_nodes
prepare_heater_platform_data = heater_module.prepare_heater_platform_data


def test_build_heater_entity_unique_id_normalises_inputs() -> None:
    """Helper should normalise identifiers and enforce required fields."""

    uid = heater_module.build_heater_entity_unique_id(
        " 0A1B ",
        " ACM ",
        " 07 ",
        "boost",
    )
    assert uid == "termoweb:0A1B:acm:07:boost"

    uid_with_colon = heater_module.build_heater_entity_unique_id(
        "0a1b",
        "acm",
        "07",
        ":boost",
    )
    assert uid_with_colon == "termoweb:0a1b:acm:07:boost"

    with pytest.raises(ValueError):
        heater_module.build_heater_entity_unique_id("", "acm", "07")


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
    inventory = build_node_inventory(raw_nodes)
    snapshot = InstallationSnapshot(
        dev_id="dev",
        raw_nodes=raw_nodes,
        node_inventory=inventory,
    )
    entry_data = {"snapshot": snapshot}

    inventory, nodes_by_type, addrs_by_type, resolve_name = (
        prepare_heater_platform_data(
            entry_data,
            default_name_simple=lambda addr: f"Heater {addr}",
        )
    )

    assert inventory is snapshot.inventory
    htr_nodes = nodes_by_type.get("htr", [])
    assert [node.addr for node in htr_nodes] == ["1", "4", "4"]
    assert all(hasattr(node, "addr") for node in htr_nodes)
    assert addrs_by_type["htr"] == ["1", "4"]
    assert len(addrs_by_type["htr"]) == len(set(addrs_by_type["htr"]))
    acm_nodes = nodes_by_type.get("acm", [])
    assert [node.addr for node in acm_nodes] == ["2", "2"]
    assert addrs_by_type["acm"] == ["2"]
    assert len(addrs_by_type["acm"]) == len(set(addrs_by_type["acm"]))
    details = build_heater_inventory_details(inventory)
    helper_map = details.address_map
    helper_reverse = details.reverse_address_map
    assert addrs_by_type == {
        node_type: helper_map.get(node_type, [])
        for node_type in heater_module.HEATER_NODE_TYPES
    }
    assert helper_reverse == {"1": {"htr"}, "2": {"acm"}, "4": {"htr"}}
    assert resolve_name("htr", "1") == "Lounge"
    assert resolve_name("htr", "4") == "Heater 4"
    assert resolve_name("acm", "2") == "Accumulator 2"

    cached_inventory, *_ = prepare_heater_platform_data(
        {"node_inventory": list(inventory)},
        default_name_simple=lambda addr: f"Heater {addr}",
    )
    assert cached_inventory == list(inventory)

    legacy_entry = {
        "nodes": {"nodes": [{"type": "htr", "addr": "9", "name": " Kitchen "}]},
        "node_inventory": build_node_inventory(
            {"nodes": [{"type": "acm", "addr": "8"}]}
        ),
    }

    _, legacy_nodes_by_type, _, legacy_resolve = prepare_heater_platform_data(
        legacy_entry,
        default_name_simple=lambda addr: f"Heater {addr}",
    )

    assert legacy_nodes_by_type.get("htr") is None or not legacy_nodes_by_type.get(
        "htr"
    )
    assert legacy_resolve("htr", "9") == "Heater 9"
    assert legacy_resolve("foo", "9") == "Heater 9"


def test_prepare_heater_platform_data_skips_blank_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blank_node = SimpleNamespace(type="  ", addr="5")
    valid_node = SimpleNamespace(type="htr", addr="6")

    def fake_ensure(record: dict[str, Any], *, nodes: Any | None = None) -> list[Any]:
        return [blank_node, valid_node]

    monkeypatch.setattr(heater_module, "ensure_node_inventory", fake_ensure)

    entry_data: dict[str, Any] = {}

    inventory, nodes_by_type, addrs_by_type, _ = prepare_heater_platform_data(
        entry_data,
        default_name_simple=lambda addr: f"Heater {addr}",
    )

    assert [node.addr for node in nodes_by_type.get("htr", [])] == ["6"]
    assert addrs_by_type["htr"] == ["6"]
    details = build_heater_inventory_details(inventory)
    helper_map = details.address_map
    assert helper_map == {"htr": ["6"]}


def test_prepare_heater_platform_data_passes_inventory_to_name_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_nodes = {"nodes": [{"type": "htr", "addr": "9", "name": "Heater"}]}
    expected_inventory = build_node_inventory(raw_nodes)

    def fake_ensure(record: dict[str, Any], *, nodes: Any | None = None) -> list[Any]:
        return list(expected_inventory)

    calls: list[tuple[Any, Callable[[str], str]]] = []

    def fake_name_map(
        nodes: Any, default_factory: Callable[[str], str]
    ) -> dict[Any, Any]:
        calls.append((nodes, default_factory))
        return {"htr": {}, "by_type": {}}

    monkeypatch.setattr(heater_module, "ensure_node_inventory", fake_ensure)
    monkeypatch.setattr(heater_module, "build_heater_name_map", fake_name_map)

    snapshot = InstallationSnapshot(dev_id="dev", raw_nodes=raw_nodes, node_inventory=expected_inventory)
    inventory, *_ = prepare_heater_platform_data(
        {"snapshot": snapshot},
        default_name_simple=lambda addr: f"Heater {addr}",
    )

    assert calls
    recorded_nodes, recorded_factory = calls[0]
    assert recorded_nodes is inventory
    assert recorded_factory("9") == "Heater 9"


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
    snapshot = InstallationSnapshot(
        dev_id="dev",
        raw_nodes=raw_nodes,
        node_inventory=build_node_inventory(raw_nodes),
    )

    _, _, _, resolve_name = prepare_heater_platform_data(
        {"snapshot": snapshot},
        default_name_simple=lambda addr: f"Heater {addr}",
    )

    assert resolve_name(" HTR ", " 8 ") == "Hall"


def test_prepare_heater_platform_data_uses_snapshot_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_nodes = {"nodes": [{"type": "htr", "addr": "1"}]}
    call_count = 0

    def fake_build(nodes: Any) -> list[Any]:
        nonlocal call_count
        call_count += 1
        return build_node_inventory(raw_nodes)

    monkeypatch.setattr(installation_module, "build_node_inventory", fake_build)

    snapshot = InstallationSnapshot(dev_id="dev", raw_nodes=raw_nodes)
    entry_data = {"snapshot": snapshot}

    prepare_heater_platform_data(
        entry_data,
        default_name_simple=lambda addr: f"Heater {addr}",
    )
    prepare_heater_platform_data(
        entry_data,
        default_name_simple=lambda addr: f"Heater {addr}",
    )

    assert call_count == 1


def test_log_skipped_nodes_defaults_platform_name(
    caplog: pytest.LogCaptureFixture,
) -> None:
    nodes_by_type = {"thm": [SimpleNamespace(addr="7")]}

    with caplog.at_level(logging.DEBUG):
        heater_module.log_skipped_nodes("", nodes_by_type, skipped_types=["thm"])

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

    resolved: list[tuple[str, str]] = []

    def fake_resolve(node_type: str, addr: str) -> str:
        resolved.append((node_type, addr))
        return f"{node_type}-{addr}"

    yielded = list(iter_heater_nodes(nodes_by_type, fake_resolve))

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
        iter_heater_nodes(nodes_by_type, fake_resolve_acm, node_types=["acm", "thm"])
    )

    assert [(node_type, addr) for node_type, _node, addr, _ in only_acm] == [
        ("acm", "2")
    ]
    assert resolved_acm == [("acm", "2")]

    mapping_nodes = {
        "htr": {"first": SimpleNamespace(addr="5")},
        "acm": SimpleNamespace(addr="6"),
        "pmo": [SimpleNamespace(addr=" None ")],
        "thm": "ignored",
    }

    extra_resolved: list[tuple[str, str]] = []

    def extra_resolve(node_type: str, addr: str) -> str:
        extra_resolved.append((node_type, addr))
        return f"{node_type}-{addr}"

    extra_results = list(
        iter_heater_nodes(
            mapping_nodes,
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
    assert list(iter_heater_nodes(blank_nodes, fake_resolve)) == []

    assert list(
        iter_heater_nodes(nodes_by_type, fake_resolve, node_types=["", "acm"])
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

    yielded = list(iter_boostable_heater_nodes(nodes_by_type, resolve))
    assert sorted((node_type, addr) for node_type, _node, addr, _ in yielded) == sorted(
        [("htr", "1"), ("acm", "3")]
    )

    accumulators_only = list(
        iter_boostable_heater_nodes(nodes_by_type, resolve, accumulators_only=True)
    )
    assert [(node_type, addr) for node_type, _node, addr, _ in accumulators_only] == [
        ("acm", "3")
    ]

    assert (
        list(
            iter_boostable_heater_nodes(
                nodes_by_type,
                resolve,
                node_types=["htr"],
                accumulators_only=True,
            )
        )
        == []
    )


def test_iter_heater_maps_deduplicates_sections() -> None:
    htr_section = {"settings": {"1": {"mode": "auto"}}}
    acm_section = {"settings": {"2": {"mode": "charge"}}}
    cache = {
        "nodes_by_type": {
            "htr": htr_section,
            "acm": acm_section,
        },
        "htr": htr_section,
    }

    results = list(
        heater_module.iter_heater_maps(
            cache,
            map_key="settings",
            node_types=["", "htr", "acm", "htr"],
        )
    )

    assert len(results) == 2
    assert results[0] is htr_section["settings"]
    assert results[1] is acm_section["settings"]


def test_iter_heater_maps_accepts_string_node_type() -> None:
    cache = {"nodes_by_type": {"htr": {"settings": {"1": {"mode": "auto"}}}}}

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
    cache = {"nodes_by_type": {"htr": {"settings": {"1": {"mode": "auto"}}}}}

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


def test_device_available_requires_nodes_section() -> None:
    coordinator = SimpleNamespace(hass=None)
    heater = _make_heater(coordinator)

    assert not heater._device_available(None)
    assert not heater._device_available({})
    assert heater._device_available({"nodes": []})


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
    assert section == {"settings": {"A": {"mode": "auto"}}}


def test_heater_settings_missing_mapping() -> None:
    hass = HomeAssistant()
    coordinator = SimpleNamespace(hass=hass, data={"dev": {"htr": {"settings": []}}})
    heater = _make_heater(coordinator)

    assert heater.heater_settings() is None


def test_supports_boost_helper_handles_variants(caplog: pytest.LogCaptureFixture) -> None:
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
