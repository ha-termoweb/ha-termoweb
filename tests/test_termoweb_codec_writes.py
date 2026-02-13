"""Tests for TermoWeb write encoding helpers."""

from __future__ import annotations

import pytest

from custom_components.termoweb.codecs.termoweb_codec import (
    build_boost_payload,
    build_extra_options_payload,
    build_settings_payload,
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


def test_build_settings_payload_formats_fields() -> None:
    """Ensure temperatures, modes and units are normalised."""

    commands = [
        SetUnits("f"),
        SetMode("Heat"),
        SetSetpoint(21.24),
        SetProgram([0, 1, 2] * 56),
        SetPresetTemps([18, 19.5, "20"]),
    ]

    payload = build_settings_payload("htr", commands)

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
            "htr",
            [
                SetUnits("C"),
                SetProgram([0] * 24),
            ],
        )


def test_build_settings_payload_preserves_modified_auto_mode() -> None:
    """Ensure modified_auto survives mode normalisation unchanged."""

    payload = build_settings_payload("htr", [SetMode(" modified_auto ")])

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
