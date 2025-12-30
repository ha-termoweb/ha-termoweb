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
