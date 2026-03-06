"""Tests for Ducaheat codec encoding helpers."""

from __future__ import annotations

import pytest

from custom_components.termoweb.codecs.ducaheat_codec import (
    encode_boost_command,
    encode_extra_options_command,
    encode_preset_temps_command,
    encode_program_command,
    encode_setpoint_command,
)
from custom_components.termoweb.domain.commands import (
    SetExtraOptions,
    SetPresetTemps,
    SetProgram,
    SetSetpoint,
    StartBoost,
)


def test_encode_setpoint_formats_temperature_and_units() -> None:
    """Ensure setpoint payloads normalise temperatures and units."""

    payload = encode_setpoint_command(SetSetpoint(21.234), units=" f ")

    assert payload == {"stemp": "21.2", "units": "F"}


def test_encode_preset_temps_requires_three_values() -> None:
    """Reject preset payloads that are not full triplets."""

    with pytest.raises(ValueError, match="presets must contain"):
        encode_preset_temps_command(SetPresetTemps([15, 18]), units="C")


def test_encode_program_command_splits_weekly_prog() -> None:
    """Ensure weekly programs are keyed by day with 24 slots each."""

    payload = encode_program_command(SetProgram([0, 1, 2] * 56))

    assert set(payload) == {"prog"}
    assert set(payload["prog"]) == {"0", "1", "2", "3", "4", "5", "6"}
    assert payload["prog"]["0"] == [0, 0, 1, 1, 2, 2] * 8
    assert len(payload["prog"]["6"]) == 48


def test_encode_extra_options_requires_values() -> None:
    """Require at least one setup extra_options value."""

    with pytest.raises(ValueError, match="extra_options must include"):
        encode_extra_options_command(SetExtraOptions())


def test_encode_boost_command_validates_minutes_and_formats() -> None:
    """Validate boost minutes and formatting for accumulator boost payloads."""

    with pytest.raises(ValueError, match="boost_time must be one of"):
        encode_boost_command(StartBoost(boost_time=30))

    payload = encode_boost_command(StartBoost(boost_time=120, stemp=20, units=" c "))

    assert payload == {"boost": True, "boost_time": 120, "stemp": "20.0", "units": "C"}


# ---------------------------------------------------------------------------
# decode_settings tests
# ---------------------------------------------------------------------------

from custom_components.termoweb.codecs.ducaheat_codec import (
    decode_settings,
    encode_lock_command,
    encode_mode_command,
    encode_priority_command,
    encode_units_command,
    infer_status_endpoint,
)
from custom_components.termoweb.domain.commands import (
    SetLock,
    SetMode,
    SetPriority,
    SetUnits,
    StopBoost,
)
from custom_components.termoweb.domain.ids import NodeType


def test_decode_settings_non_dict_returns_empty() -> None:
    """Non-dict payloads return empty dict."""

    assert decode_settings("not a dict", node_type=NodeType.HEATER) == {}


def test_decode_settings_thermostat_mode() -> None:
    """Thermostat payloads are decoded through DucaheatThermostatSettings."""

    result = decode_settings(
        {"mode": "AUTO", "stemp": 21.0, "mtemp": 19.5},
        node_type=NodeType.THERMOSTAT,
    )
    assert result.get("mode") == "auto"


def test_decode_settings_heater_segmented() -> None:
    """Heater payloads go through segmented settings decoding."""

    result = decode_settings(
        {"status": {"mode": "MANUAL", "stemp": "22.0"}},
        node_type=NodeType.HEATER,
    )
    assert result.get("mode") == "manual"


def test_decode_settings_accumulator_with_setup() -> None:
    """Accumulator payloads include setup segment data."""

    result = decode_settings(
        {
            "status": {"mode": "auto", "stemp": "21.0", "charging": True},
            "setup": {"boost_time": 120, "current_charge_per": 50},
        },
        node_type=NodeType.ACCUMULATOR,
    )
    assert result.get("mode") == "auto"


def test_decode_settings_unknown_node_type_passthrough() -> None:
    """Unknown node types pass through canonicalize_settings_payload directly."""

    result = decode_settings(
        {"mode": "manual", "stemp": "20.0"},
        node_type=NodeType.POWER_MONITOR,
    )
    # Should still pass through canonicalize without crash
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# encode_mode_command
# ---------------------------------------------------------------------------

def test_encode_mode_command_basic() -> None:
    """Encode mode command produces correct payload."""

    payload = encode_mode_command(SetMode(mode="auto"))
    assert payload["mode"] == "auto"


# ---------------------------------------------------------------------------
# encode_setpoint_command edge cases
# ---------------------------------------------------------------------------

def test_encode_setpoint_with_boost_context() -> None:
    """Setpoint encoding includes boost flag and boost_time when provided."""

    payload = encode_setpoint_command(
        SetSetpoint(21.5),
        boost=True,
        boost_time=120,
        mode="manual",
    )
    assert payload["stemp"] == "21.5"
    assert payload["boost"] is True
    assert payload["boost_time"] == 120
    assert payload["mode"] == "manual"


# ---------------------------------------------------------------------------
# encode_units_command
# ---------------------------------------------------------------------------

def test_encode_units_command() -> None:
    """Units command wraps units string."""

    payload = encode_units_command(SetUnits(units="F"))
    assert payload == {"units": "F"}


# ---------------------------------------------------------------------------
# encode_program_command error paths
# ---------------------------------------------------------------------------

def test_encode_program_command_wrong_length() -> None:
    """Reject programs that are not exactly 168 items."""

    with pytest.raises(ValueError, match="prog must be a list of 168"):
        encode_program_command(SetProgram([0] * 100))


def test_encode_program_command_non_integer_values() -> None:
    """Reject programs with non-integer values."""

    with pytest.raises(ValueError, match="prog contains non-integer"):
        encode_program_command(SetProgram(["x"] * 168))


def test_encode_program_command_invalid_range() -> None:
    """Reject programs with values outside 0-2 range."""

    with pytest.raises(ValueError, match="prog values must be 0, 1, or 2"):
        encode_program_command(SetProgram([5] * 168))


# ---------------------------------------------------------------------------
# encode_lock_command
# ---------------------------------------------------------------------------

def test_encode_lock_command() -> None:
    """Lock command serialises lock state."""

    payload = encode_lock_command(SetLock(lock=True))
    assert payload == {"lock": True}


# ---------------------------------------------------------------------------
# encode_priority_command
# ---------------------------------------------------------------------------

def test_encode_priority_command() -> None:
    """Priority command serialises priority level."""

    payload = encode_priority_command(SetPriority(priority=3))
    assert payload == {"priority": 3}


# ---------------------------------------------------------------------------
# infer_status_endpoint routing
# ---------------------------------------------------------------------------

def test_infer_status_endpoint_prog() -> None:
    assert infer_status_endpoint(NodeType.HEATER, SetProgram([0] * 168)) == "prog"


def test_infer_status_endpoint_mode() -> None:
    assert infer_status_endpoint(NodeType.HEATER, SetMode("auto")) == "mode"


def test_infer_status_endpoint_setpoint() -> None:
    assert infer_status_endpoint(NodeType.HEATER, SetSetpoint(21.0)) == "status"


def test_infer_status_endpoint_preset_temps() -> None:
    assert infer_status_endpoint(NodeType.HEATER, SetPresetTemps([15, 18, 21])) == "status"


def test_infer_status_endpoint_units() -> None:
    assert infer_status_endpoint(NodeType.HEATER, SetUnits("C")) == "status"


def test_infer_status_endpoint_extra_options() -> None:
    assert infer_status_endpoint(NodeType.ACCUMULATOR, SetExtraOptions(boost_time=120)) == "setup"


def test_infer_status_endpoint_boost_start() -> None:
    assert infer_status_endpoint(NodeType.ACCUMULATOR, StartBoost()) == "boost"


def test_infer_status_endpoint_boost_stop() -> None:
    assert infer_status_endpoint(NodeType.ACCUMULATOR, StopBoost()) == "boost"


def test_infer_status_endpoint_lock() -> None:
    assert infer_status_endpoint(NodeType.HEATER, SetLock(lock=True)) == "lock"


def test_infer_status_endpoint_priority() -> None:
    assert infer_status_endpoint(NodeType.HEATER, SetPriority(priority=1)) == "setup"


def test_infer_status_endpoint_unsupported_raises() -> None:
    from custom_components.termoweb.domain.commands import BaseCommand
    with pytest.raises(TypeError, match="Unsupported command type"):
        infer_status_endpoint(NodeType.HEATER, BaseCommand())


def test_encode_preset_temps_with_units() -> None:
    """Preset temps encoding includes units when provided."""

    payload = encode_preset_temps_command(
        SetPresetTemps([7.0, 16.0, 21.0]), units=" c "
    )
    assert payload["cold"] == "7.0"
    assert payload["night"] == "16.0"
    assert payload["day"] == "21.0"
    assert payload["units"] == "C"


def test_encode_extra_options_with_values() -> None:
    """Extra options encoding with valid boost_time and boost_temp."""

    payload = encode_extra_options_command(
        SetExtraOptions(boost_time=120, boost_temp=25.0)
    )
    assert "extra_options" in payload
    assert payload["extra_options"]["boost_time"] == 120
