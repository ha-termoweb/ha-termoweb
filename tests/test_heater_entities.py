from __future__ import annotations

from types import SimpleNamespace

from typing import Any

import pytest

from conftest import _install_stubs, make_ws_payload

_install_stubs()

from custom_components.termoweb import heater as heater_module
from custom_components.termoweb.const import DOMAIN
from homeassistant.core import HomeAssistant

HeaterNodeBase = heater_module.HeaterNodeBase


def test_heater_node_base_normalizes_address(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, dict]] = []

    original = heater_module.normalize_node_addr

    def _record_normalize(value, **kwargs):
        calls.append((value, kwargs))
        return original(value, **kwargs)

    monkeypatch.setattr(heater_module, "normalize_node_addr", _record_normalize)

    coordinator = SimpleNamespace(hass=None)
    heater = HeaterNodeBase(coordinator, "entry", "dev", " 01 ", " Heater 01 ")

    assert heater._addr == "01"
    assert heater.device_info["identifiers"] == {
        (heater_module.DOMAIN, "dev", "01")
    }
    assert heater._attr_unique_id == f"{heater_module.DOMAIN}:dev:htr:01"
    assert calls == [(" 01 ", {})]


def test_heater_node_base_payload_matching_normalizes_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, dict]] = []

    original = heater_module.normalize_node_addr

    def _record_normalize(value, **kwargs):
        calls.append((value, kwargs))
        return original(value, **kwargs)

    monkeypatch.setattr(heater_module, "normalize_node_addr", _record_normalize)

    coordinator = SimpleNamespace(hass=None)
    heater = HeaterNodeBase(coordinator, "entry", "dev", " 01 ", "Heater 01")

    assert heater._payload_matches_heater(make_ws_payload("dev", " 01 "))
    assert not heater._payload_matches_heater(make_ws_payload("dev", "02"))
    assert not heater._payload_matches_heater(make_ws_payload("dev", "  "))
    assert calls == [(" 01 ", {}), (" 01 ", {}), ("02", {}), ("  ", {})]


def test_boost_runtime_storage_roundtrip() -> None:
    """Ensure boost runtime helpers normalise addresses and defaults."""

    hass = HomeAssistant()
    entry_id = "entry-store"
    hass.data = {DOMAIN: {entry_id: {}}}

    assert (
        heater_module.get_boost_runtime_minutes(hass, entry_id, "acm", "01")
        is None
    )

    heater_module.set_boost_runtime_minutes(hass, entry_id, "acm", "01", 120)
    assert (
        heater_module.get_boost_runtime_minutes(hass, entry_id, "ACM", " 01 ")
        == 120
    )
    assert (
        heater_module.resolve_boost_runtime_minutes(
            hass,
            entry_id,
            "ACM",
            "01",
        )
        == 120
    )

    heater_module.set_boost_runtime_minutes(hass, entry_id, "acm", "01", None)
    assert (
        heater_module.get_boost_runtime_minutes(hass, entry_id, "acm", "01")
        is None
    )
    assert (
        heater_module.resolve_boost_runtime_minutes(
            hass,
            entry_id,
            "acm",
            "01",
        )
        == heater_module.DEFAULT_BOOST_DURATION
    )


def test_coerce_boost_minutes_edge_cases() -> None:
    """Exercise defensive conversions for boost duration inputs."""

    coerce = heater_module._coerce_boost_minutes
    assert coerce(None) is None
    assert coerce(True) is None
    assert coerce(0) is None
    assert coerce(-5) is None
    assert coerce("   ") is None
    assert coerce("invalid") is None
    assert coerce("90") == 90
    assert coerce(120.7) == 120


def test_boost_runtime_store_handles_non_mapping() -> None:
    """Verify boost runtime store creation tolerates unexpected inputs."""

    assert heater_module._boost_runtime_store(None, create=False) == {}
    entry_data: dict[str, Any] = {}
    assert heater_module._boost_runtime_store(entry_data, create=False) == {}
    created = heater_module._boost_runtime_store(entry_data, create=True)
    assert created == {}
    assert entry_data[heater_module._BOOST_RUNTIME_KEY] is created


def test_boost_runtime_helpers_guard_invalid_structures() -> None:
    """Ensure get/set helpers short-circuit when data is malformed."""

    hass = HomeAssistant()
    entry_id = "entry-invalid"

    # Domain data missing prevents persistence.
    heater_module.set_boost_runtime_minutes(hass, entry_id, "acm", "01", 30)
    assert (
        heater_module.get_boost_runtime_minutes(hass, entry_id, "acm", "01")
        is None
    )

    # Non-mapping entry data is ignored for get/set operations.
    hass.data = {DOMAIN: {entry_id: []}}
    assert (
        heater_module.get_boost_runtime_minutes(hass, entry_id, "acm", "01")
        is None
    )
    heater_module.set_boost_runtime_minutes(hass, entry_id, "acm", "01", 45)

    # Missing identifiers or invalid minutes are ignored.
    hass.data[DOMAIN][entry_id] = {}
    heater_module.set_boost_runtime_minutes(hass, entry_id, "", "01", 45)
    heater_module.set_boost_runtime_minutes(hass, entry_id, "acm", "", 45)
    heater_module.set_boost_runtime_minutes(hass, entry_id, "acm", "01", -10)

    # Stored garbage values should not be returned.
    hass.data[DOMAIN][entry_id][heater_module._BOOST_RUNTIME_KEY] = {
        "acm": {"01": "oops"}
    }
    assert (
        heater_module.get_boost_runtime_minutes(hass, entry_id, "acm", "01")
        is None
    )
    assert (
        heater_module.resolve_boost_runtime_minutes(
            hass,
            entry_id,
            "",
            "",
        )
        == heater_module.DEFAULT_BOOST_DURATION
    )
