from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from conftest import _install_stubs, make_ws_payload

_install_stubs()

from custom_components.termoweb import heater as heater_module
from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.binary_sensor import HeaterBoostActiveBinarySensor
from custom_components.termoweb.sensor import (
    HeaterBoostEndSensor,
    HeaterBoostMinutesRemainingSensor,
)
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

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


def test_derive_boost_state_uses_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify boost metadata derivation favours the coordinator resolver."""

    base_now = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(dt_util, "now", lambda: base_now)

    expected_end = base_now + timedelta(hours=1)

    class _Coordinator(SimpleNamespace):
        def resolve_boost_end(self, day, minute, *, now=None):
            return expected_end, 60

    settings = {"mode": "boost", "boost_end_day": 1, "boost_end_min": 60}
    state = heater_module.derive_boost_state(settings, _Coordinator())

    assert state.active is True
    assert state.minutes_remaining == 60
    assert state.end_datetime == expected_end
    assert state.end_iso == expected_end.isoformat()


def test_derive_boost_state_from_remaining(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure boost metadata uses remaining minutes when resolver unavailable."""

    base_now = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(dt_util, "now", lambda: base_now)

    settings = {"boost_remaining": "15"}
    state = heater_module.derive_boost_state(settings, SimpleNamespace())

    assert state.active is False
    assert state.minutes_remaining == 15
    assert state.end_iso == (base_now + timedelta(minutes=15)).isoformat()
    assert state.end_datetime == base_now + timedelta(minutes=15)


def test_derive_boost_state_parses_string_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """Boost metadata should parse ISO formatted end timestamps."""

    base_now = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(dt_util, "now", lambda: base_now)

    iso = "2024-01-01T00:30:00+00:00"
    settings = {"boost_active": True, "boost_end": iso}
    state = heater_module.derive_boost_state(settings, SimpleNamespace())

    assert state.active is True
    assert state.minutes_remaining is None
    assert state.end_iso == iso
    assert state.end_datetime == datetime(2024, 1, 1, 0, 30, tzinfo=timezone.utc)


def test_boost_entities_expose_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure boost binary and sensor entities expose derived metadata."""

    base_now = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(dt_util, "now", lambda: base_now)

    expected_end = base_now + timedelta(minutes=30)
    settings = {"mode": "boost", "boost_end_day": 1, "boost_end_min": 30}
    coordinator = SimpleNamespace(
        data={
            "dev": {
                "nodes_by_type": {
                    "acm": {"settings": {"1": settings}},
                }
            }
        },
        resolve_boost_end=lambda day, minute, *, now=None: (expected_end, 30),
    )

    boost_binary = HeaterBoostActiveBinarySensor(
        coordinator,
        "entry",
        "dev",
        "1",
        "Accumulator Boost Active",
        f"{DOMAIN}:dev:acm:1:boost_active",
        device_name="Accumulator",
        node_type="acm",
    )
    minutes_sensor = HeaterBoostMinutesRemainingSensor(
        coordinator,
        "entry",
        "dev",
        "1",
        "Accumulator Boost Minutes Remaining",
        f"{DOMAIN}:dev:acm:1:boost:minutes_remaining",
        device_name="Accumulator",
        node_type="acm",
    )
    end_sensor = HeaterBoostEndSensor(
        coordinator,
        "entry",
        "dev",
        "1",
        "Accumulator Boost End",
        f"{DOMAIN}:dev:acm:1:boost:end",
        device_name="Accumulator",
        node_type="acm",
    )

    assert boost_binary.is_on is True
    assert minutes_sensor.native_value == 30
    assert end_sensor.native_value == expected_end

    assert boost_binary.extra_state_attributes["boost_minutes_remaining"] == 30
    assert boost_binary.extra_state_attributes["boost_end"] == expected_end.isoformat()
    assert minutes_sensor.extra_state_attributes["boost_end"] == expected_end.isoformat()
    assert end_sensor.extra_state_attributes["boost_minutes_remaining"] == 30


def test_boost_entities_handle_missing_data() -> None:
    """Boost entities should gracefully handle missing settings."""

    coordinator = SimpleNamespace(
        data={
            "dev": {
                "nodes_by_type": {
                    "acm": {"settings": {"1": {}}},
                }
            }
        }
    )

    boost_binary = HeaterBoostActiveBinarySensor(
        coordinator,
        "entry",
        "dev",
        "1",
        "Accumulator Boost Active",
        f"{DOMAIN}:dev:acm:1:boost_active",
        device_name="Accumulator",
        node_type="acm",
    )
    minutes_sensor = HeaterBoostMinutesRemainingSensor(
        coordinator,
        "entry",
        "dev",
        "1",
        "Accumulator Boost Minutes Remaining",
        f"{DOMAIN}:dev:acm:1:boost:minutes_remaining",
        device_name="Accumulator",
        node_type="acm",
    )
    end_sensor = HeaterBoostEndSensor(
        coordinator,
        "entry",
        "dev",
        "1",
        "Accumulator Boost End",
        f"{DOMAIN}:dev:acm:1:boost:end",
        device_name="Accumulator",
        node_type="acm",
    )

    assert boost_binary.is_on is False
    assert minutes_sensor.native_value is None
    assert end_sensor.native_value is None
    assert boost_binary.extra_state_attributes["boost_end"] is None
    assert minutes_sensor.extra_state_attributes["boost_end"] is None
    assert end_sensor.extra_state_attributes["boost_minutes_remaining"] is None
