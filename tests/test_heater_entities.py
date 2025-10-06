from __future__ import annotations

from types import SimpleNamespace

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
