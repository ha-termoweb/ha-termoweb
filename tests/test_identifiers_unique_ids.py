"""Tests for identifier builders focusing on unique ID helpers."""

from __future__ import annotations

import pytest

from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb import identifiers as identifiers_module
from custom_components.termoweb.identifiers import (
    build_heater_entity_unique_id,
    build_heater_unique_id,
    build_power_monitor_energy_unique_id,
    build_power_monitor_power_unique_id,
    build_power_monitor_unique_id,
)


def test_build_heater_unique_id_prefixes_suffix_with_colon() -> None:
    """Suffixes should gain a leading colon when missing."""

    unique_id = build_heater_unique_id(" dev ", " htr ", " 01 ", suffix="status")

    assert unique_id == f"{DOMAIN}:dev:htr:01:status"


def test_build_heater_entity_unique_id_defers_to_base(monkeypatch: pytest.MonkeyPatch) -> None:
    """The entity helper should delegate to the heater unique ID builder."""

    calls: list[tuple[object, ...]] = []

    def _record(dev_id: object, node_type: object, addr: object, *, suffix: object | None = None) -> str:
        calls.append((dev_id, node_type, addr, suffix))
        return "recorded"

    monkeypatch.setattr(identifiers_module, "build_heater_unique_id", _record)

    assert build_heater_entity_unique_id("dev", "acm", "02", suffix="energy") == "recorded"
    assert calls == [("dev", "acm", "02", "energy")]


def test_power_monitor_helpers_defers_to_heater_unique_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Power-monitor helpers should pass through to the heater ID builder."""

    calls: list[tuple[object, ...]] = []

    def _record(dev_id: object, node_type: object, addr: object, *, suffix: object | None = None) -> str:
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
