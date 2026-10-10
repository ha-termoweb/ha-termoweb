"""Tests for the TermoWeb codec and wire models."""

from __future__ import annotations

import logging

import pytest

from custom_components.termoweb.codecs.termoweb_codec import (
    build_boost_payload,
    build_extra_options_payload,
    build_settings_payload,
    decode_devs_payload,
    decode_node_settings,
    decode_nodes_payload,
    decode_samples,
)
from custom_components.termoweb.codecs.termoweb_models import (
    HeaterSettingsPayload,
    ThermostatSettingsPayload,
)
from custom_components.termoweb.domain import (
    AccumulatorState,
    DomainStateStore,
    HeaterState,
    NodeId,
    NodeType,
    ThermostatState,
)
from custom_components.termoweb.domain.commands import (
    SetExtraOptions,
    SetMode,
    SetPresetTemps,
    SetProgram,
    SetSetpoint,
    SetUnits,
    StartBoost,
)


def test_decode_devs_payload_list_filters_non_dicts() -> None:
    raw = [{"dev_id": "abc"}, "ignore", {"name": "ok"}]

    result = decode_devs_payload(raw)

    assert result == [{"dev_id": "abc"}, {"name": "ok"}]


def test_decode_devs_payload_dict_variants() -> None:
    raw_devs = {"devs": [{"id": 1}, "bad"]}  # legacy shape
    raw_devices = {"devices": [{"dev_id": "abc"}, 123]}  # alternate shape

    assert decode_devs_payload(raw_devs) == [{"id": 1}]
    assert decode_devs_payload(raw_devices) == [{"dev_id": "abc"}]


def test_decode_devs_payload_unexpected_shape() -> None:
    assert decode_devs_payload("oops") == []
    assert decode_devs_payload({"weird": []}) == []


def test_decode_nodes_payload_dict_normalizes_addresses() -> None:
    raw = {"nodes": [{"type": "htr", "addr": 2, "name": "Heater"}]}

    decoded = decode_nodes_payload(raw)

    assert decoded == {"nodes": [{"type": "htr", "addr": "2", "name": "Heater"}]}


def test_decode_nodes_payload_list_passthrough() -> None:
    raw = [{"type": "htr", "addr": "1"}]

    assert decode_nodes_payload(raw) is raw


# Documented GET /htr/{addr}/settings sample (docs/termoweb_api.md).
HTR_SETTINGS_SAMPLE = {
    "name": "Guest bedroom ",
    "priority": 0,
    "prog": [0, 1, 2] * 56,
    "units": "C",
    "ptemp": ["10.0", "16.0", "21.0"],
    "mtemp": "25.7",
    "stemp": "10.0",
    "mode": "off",
    "max_power": "974",
    "state": "off",
    "true_radiant_active": False,
    "window_state_active": False,
    "sync_status": "ok",
}

ACM_SETTINGS_SAMPLE = {
    "mode": "auto",
    "stemp": "18.0",
    "units": "C",
    "max_power": "1500",
    "state": "on",
    "lock": False,
    "status": {
        "charging": True,
        "current_charge_per": 50,
        "target_charge_per": 80,
        "boost_active": True,
        "boost_end_day": 3,
        "boost_end_min": 945,
        "boost_remaining": 30,
    },
}


def _store_state(node_type: NodeType, decoded: dict) -> object:
    """Apply decoded settings to a one-node domain store and return its state."""
    store = DomainStateStore([NodeId(node_type, "1")])
    store.apply_full_snapshot(node_type, "1", decoded)
    return store.get_state(node_type, "1")


def test_decode_htr_settings_keeps_state_and_max_power() -> None:
    decoded = decode_node_settings("htr", HTR_SETTINGS_SAMPLE)

    assert decoded["state"] == "off"
    assert decoded["max_power"] == "974"
    assert decoded["priority"] == 0

    state = _store_state(NodeType.HEATER, decoded)
    assert isinstance(state, HeaterState)
    assert state.state == "off"
    assert state.max_power == 974.0


def test_decode_acm_settings_keeps_charge_boost_and_lock() -> None:
    decoded = decode_node_settings("acm", ACM_SETTINGS_SAMPLE)

    assert decoded == {
        "mode": "auto",
        "stemp": "18.0",
        "units": "C",
        "max_power": "1500",
        "state": "on",
        "lock": False,
        "charging": True,
        "current_charge_per": 50,
        "target_charge_per": 80,
        "boost_active": True,
        "boost_end_day": 3,
        "boost_end_min": 945,
        "boost_remaining": 30,
    }

    state = _store_state(NodeType.ACCUMULATOR, decoded)
    assert isinstance(state, AccumulatorState)
    assert state.state == "on"
    assert state.max_power == 1500.0
    assert state.lock is False
    assert state.charging is True
    assert state.current_charge_per == 50
    assert state.target_charge_per == 80
    assert state.boost_end_day == 3
    assert state.boost_end_min == 945
    assert state.boost_remaining == 30


def test_decode_thm_settings_normalises_and_keeps_fields() -> None:
    raw = {
        "mode": "manual",
        "stemp": 21,
        "mtemp": "19.25",
        "temp": 20,
        "prog": ["0", 1, "x"],
        "ptemp": [7, "17", None],
        "units": "C",
        "state": "on",
        "batt_level": 4,
        "lock": True,
        "priority": 1,
    }

    decoded = decode_node_settings("thm", raw)

    assert decoded == {
        "mode": "manual",
        "stemp": "21.0",
        "mtemp": "19.2",
        "temp": "20.0",
        "prog": [0, 1, "x"],
        "ptemp": ["7.0", "17.0", None],
        "units": "C",
        "state": "on",
        "batt_level": 4,
        "lock": True,
        "priority": 1,
    }

    state = _store_state(NodeType.THERMOSTAT, decoded)
    assert isinstance(state, ThermostatState)
    assert state.batt_level == 4
    assert state.lock is True
    assert state.state == "on"


def test_decode_node_settings_formats_temperatures() -> None:
    raw = {
        "mode": "Auto",
        "stemp": 21,
        "mtemp": " 19.25 ",
        "ptemp": [18, "20", None],
    }

    decoded = decode_node_settings("htr", raw)

    assert decoded["mode"] == "Auto"
    assert decoded["stemp"] == "21.0"
    assert decoded["mtemp"] == "19.2"
    assert decoded["ptemp"] == ["18.0", "20.0", None]


def test_decode_node_settings_handles_status_block() -> None:
    raw = {
        "status": {
            "mode": "off",
            "stemp": 19,
            "prog": [0, 1, 2],
            "ptemp": ["7", "17.5", "21.0"],
        }
    }

    decoded = decode_node_settings("acm", raw)

    assert decoded["mode"] == "off"
    assert decoded["stemp"] == "19.0"
    assert decoded["prog"] == [0, 1, 2]
    assert decoded["ptemp"] == ["7.0", "17.5", "21.0"]
    assert "status" not in decoded


def test_decode_node_settings_preserves_short_prog() -> None:
    raw = {"prog": [0, 1, 2]}

    decoded = decode_node_settings("htr", raw)

    assert decoded["prog"] == [0, 1, 2]


def test_decode_node_settings_strips_vendor_blobs() -> None:
    raw = {
        "mode": "auto",
        "capabilities": {"raw": True},
        "status": {"mode": "manual", "extra": "ignored"},
    }

    decoded = decode_node_settings("htr", raw)

    assert decoded == {"mode": "auto"}


def test_decode_node_settings_falls_back_on_validation_error() -> None:
    raw = {"mode": "manual", "status": object(), "temp": "19"}

    decoded = decode_node_settings("htr", raw)

    assert decoded == {"mode": "manual", "temp": "19"}


def test_decode_samples_filters_invalid_items(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    raw = {
        "samples": [
            {"t": 1000, "counter": 1.5},
            {"timestamp": 2000, "value": 5},
            {"t": "bad", "counter": 3},
            {"t": 3000},
        ]
    }

    decoded = decode_samples(raw)

    assert decoded == [
        {"t": 1000, "counter": "1.5"},
        {"t": 2000, "counter": "5"},
    ]
    assert any("Unexpected htr sample shape" in rec.message for rec in caplog.records)


def test_build_settings_payload_formats_fields() -> None:
    """Ensure temperatures, modes and units are normalised."""

    commands = [
        SetUnits("f"),
        SetMode("Heat"),
        SetSetpoint(21.24),
        SetProgram([0, 1, 2] * 56),
        SetPresetTemps([18, 19.5, "20"]),
    ]

    payload = build_settings_payload(commands)

    assert payload == {
        "mode": "manual",
        "stemp": "21.2",
        "prog": [0, 1, 2] * 56,
        "ptemp": ["18.0", "19.5", "20.0"],
        "units": "F",
    }


def test_build_settings_payload_invalid_program_length() -> None:
    """Reject programs that are not full-week lists."""

    with pytest.raises(ValueError, match="prog must be a list of 168 integers"):
        build_settings_payload(
            [
                SetUnits("C"),
                SetProgram([0] * 24),
            ],
        )


def test_build_settings_payload_preserves_modified_auto_mode() -> None:
    """Ensure modified_auto survives mode normalisation unchanged."""

    payload = build_settings_payload([SetMode(" modified_auto ")])

    assert payload == {"mode": "modified_auto"}


def test_build_extra_options_payload_requires_values() -> None:
    """Ensure extra options payloads need at least one field."""

    with pytest.raises(ValueError, match="must be provided"):
        build_extra_options_payload(SetExtraOptions())


def test_build_boost_payload_validates_minutes() -> None:
    """Validate boost duration before encoding payload."""

    with pytest.raises(ValueError, match="boost_time must be one of"):
        build_boost_payload(StartBoost(boost_time=30))


def test_build_boost_payload_formats_units_and_temps() -> None:
    """Format boost payload values as the API expects."""

    payload = build_boost_payload(StartBoost(boost_time=120, stemp="22", units=" c "))

    assert payload == {
        "boost": True,
        "boost_time": 120,
        "stemp": "22.0",
        "units": "C",
    }


def test_heater_temperatures_use_safe_temperature() -> None:
    """Numbers format to one decimal; text is stripped; junk and nan become None."""

    payload = HeaterSettingsPayload.model_validate(
        {
            "stemp": 21,
            "mtemp": " warm ",
            "temp": float("nan"),
            "ptemp": ["5", "", 21.25],
        }
    )

    assert payload.stemp == "21.0"
    assert payload.mtemp == "warm"
    assert payload.temp is None
    assert payload.ptemp == ["5.0", None, "21.2"]


def test_thermostat_non_scalar_temperature_is_dropped() -> None:
    """A non-scalar temperature no longer passes through unchanged."""

    payload = ThermostatSettingsPayload.model_validate({"stemp": [1], "mtemp": None})

    assert payload.stemp is None
    assert payload.mtemp is None
