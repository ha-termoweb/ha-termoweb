"""Tests for identifiers and unique ids (identifiers.py)."""

from __future__ import annotations

import pytest

from custom_components.termoweb import identifiers as identifiers_module
from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.identifiers import (
    build_gateway_entity_unique_id,
    build_heater_energy_unique_id,
    build_heater_unique_id,
    build_installation_entity_unique_id,
    build_node_device_identifier,
    build_power_monitor_energy_unique_id,
    build_power_monitor_power_unique_id,
    build_power_monitor_unique_id,
    node_key,
    rekey_unique_id,
)
from custom_components.termoweb.inventory import Inventory
from custom_components.termoweb.nodes import build_node_inventory


def test_build_heater_unique_id_prefixes_suffix_with_colon() -> None:
    """Suffixes should gain a leading colon when missing."""

    unique_id = build_heater_unique_id(" dev ", " htr ", " 01 ", suffix="status")

    assert unique_id == f"{DOMAIN}:dev:htr:01:status"


def test_power_monitor_helpers_defers_to_heater_unique_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Power-monitor helpers should pass through to the heater ID builder."""

    calls: list[tuple[object, ...]] = []

    def _record(
        dev_id: object, node_type: object, addr: object, *, suffix: object | None = None
    ) -> str:
        calls.append((dev_id, node_type, addr, suffix))
        return "uid"

    monkeypatch.setattr(identifiers_module, "build_heater_unique_id", _record)

    assert build_power_monitor_unique_id("dev", "03", suffix="daily") == "uid"
    assert build_power_monitor_unique_id("dev", "03") == "uid"
    assert build_power_monitor_energy_unique_id("dev", "03") == "uid"
    assert build_power_monitor_power_unique_id("dev", "03") == "uid"
    assert calls == [
        ("dev", "pmo", "03", "daily"),
        ("dev", "pmo", "03", None),
        ("dev", "pmo", "03", ":energy"),
        ("dev", "pmo", "03", ":power"),
    ]


@pytest.mark.parametrize(
    "dev_id, node_type, addr",
    [
        ("", "htr", "01"),
        ("dev", "", "01"),
        ("dev", "htr", ""),
    ],
)
def test_build_heater_unique_id_requires_all_components(
    dev_id: str, node_type: str, addr: str
) -> None:
    """Missing components should raise ValueError."""

    with pytest.raises(ValueError):
        build_heater_unique_id(dev_id, node_type, addr)


def test_build_gateway_entity_unique_id_rejects_missing_dev_id() -> None:
    """A missing dev_id must not become the literal string "None"."""

    with pytest.raises(ValueError):
        build_gateway_entity_unique_id(None, "power_limit")


def test_heater_energy_unique_id_is_normalised() -> None:
    """Energy unique IDs trim and lower-case their parts."""
    assert (
        build_heater_energy_unique_id(" dev ", " ACM ", " 01 ")
        == f"{DOMAIN}:dev:acm:01:energy"
    )


@pytest.mark.parametrize(
    ("dev_id", "node_type", "addr"),
    [("", "htr", "01"), ("dev", " ", "01"), ("dev", "htr", "  ")],
)
def test_heater_energy_unique_id_requires_all_components(
    dev_id: str, node_type: str, addr: str
) -> None:
    """Blank components are rejected."""
    with pytest.raises(ValueError):
        build_heater_energy_unique_id(dev_id, node_type, addr)


def test_installation_entity_unique_id_is_site_scoped() -> None:
    """Site entity IDs carry the trimmed gateway id and a ``site`` segment."""
    assert (
        build_installation_entity_unique_id(" dev ", "power_limit")
        == f"{DOMAIN}:dev:site:power_limit"
    )
    for dev_id in ("", "   "):
        with pytest.raises(ValueError):
            build_installation_entity_unique_id(dev_id, "info")


def test_node_key_is_the_serial_when_known_else_the_address() -> None:
    """Radio heaters with a serial are keyed by it; everything else by address."""

    nodes = build_node_inventory(
        [
            {"type": "htr", "addr": "2", "serial": "S000002"},
            {"type": "htr", "addr": "3", "serial": " "},
            {"type": "acm", "addr": "4"},
        ]
    )
    inventory = Inventory("dev", nodes)
    assert node_key(inventory, "htr", "2") == "S000002"
    assert node_key(inventory, "htr", 3) == "3"
    assert node_key(inventory, "acm", "4") == "4"
    assert node_key(inventory, "thm", "9") == "9"  # not in the inventory
    assert node_key(None, "htr", " 5 ") == "5"
    assert build_node_device_identifier("dev", "S000002") == (DOMAIN, "dev", "S000002")


def test_rekey_unique_id_moves_only_that_node() -> None:
    """Only IDs of the old node key move; a longer address is another node."""

    assert (
        rekey_unique_id(f"{DOMAIN}:dev:htr:6:climate", "dev", "htr", "6", "S1")
        == f"{DOMAIN}:dev:htr:S1:climate"
    )
    assert rekey_unique_id(f"{DOMAIN}:dev:htr:6", "dev", "htr", "6", "S1") == (
        f"{DOMAIN}:dev:htr:S1"
    )
    assert rekey_unique_id(f"{DOMAIN}:dev:htr:60:temp", "dev", "htr", "6", "S1") is None
    assert rekey_unique_id(f"{DOMAIN}:dev:acm:6:temp", "dev", "htr", "6", "S1") is None
