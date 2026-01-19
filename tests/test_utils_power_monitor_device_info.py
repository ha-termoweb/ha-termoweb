"""Tests for ``build_power_monitor_device_info`` helper."""

from __future__ import annotations

import types

from conftest import build_entry_runtime
from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.utils import build_power_monitor_device_info


def test_build_power_monitor_device_info_defaults_when_missing_entry() -> None:
    """Ensure default manufacturer and name are used without entry data."""

    hass = types.SimpleNamespace(data={})

    info = build_power_monitor_device_info(hass, "entry-id", "gateway", "01")

    assert info["manufacturer"] == "TermoWeb"
    assert info["name"] == "Power Monitor 01"


def test_build_power_monitor_device_info_uses_brand_override() -> None:
    """Ensure brand overrides manufacturer while keeping fallback name."""

    entry_id = "entry-id"
    hass = types.SimpleNamespace(data={DOMAIN: {}})
    build_entry_runtime(
        hass=hass,
        entry_id=entry_id,
        dev_id="gateway",
        brand="Ducaheat",
    )

    info = build_power_monitor_device_info(hass, entry_id, "gateway", "02")

    assert info["manufacturer"] == "Ducaheat"
    assert info["name"] == "Power Monitor 02"


def test_build_power_monitor_device_info_trims_custom_name() -> None:
    """Ensure custom names are trimmed while keeping identifiers and defaults."""

    hass = types.SimpleNamespace(data={})

    info = build_power_monitor_device_info(
        hass,
        "entry-id",
        "gateway-42",
        "07",
        name="  Kitchen Meter  ",
    )

    assert info["name"] == "Kitchen Meter"
    assert info["manufacturer"] == "TermoWeb"
    assert info["identifiers"] == {(DOMAIN, "gateway-42", "pmo", "07")}


def test_build_power_monitor_device_info_trimmed_name_with_brand_override() -> None:
    """Ensure trimmed names persist when brand overrides the manufacturer."""

    entry_id = "entry-id"
    hass = types.SimpleNamespace(data={DOMAIN: {}})
    build_entry_runtime(
        hass=hass,
        entry_id=entry_id,
        dev_id="gateway-99",
        brand="  Tevolve  ",
    )

    info = build_power_monitor_device_info(
        hass,
        entry_id,
        "gateway-99",
        "08",
        name="  Kitchen Meter  ",
    )

    assert info["name"] == "Kitchen Meter"
    assert info["manufacturer"] == "Tevolve"


def test_build_power_monitor_device_info_sets_via_device_reference() -> None:
    """The via-device reference should point back to the owning gateway."""

    hass = types.SimpleNamespace(data={})

    info = build_power_monitor_device_info(
        hass,
        "entry-id",
        "gateway-23",
        "P4",
        name="Storage Monitor",
    )

    assert info["via_device"] == (DOMAIN, "gateway-23")
    assert info["name"] == "Storage Monitor"
