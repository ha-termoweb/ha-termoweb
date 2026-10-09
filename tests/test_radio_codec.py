"""Tests for the radio codec and planner (canonical dict <-> radio records)."""

from __future__ import annotations

import pytest
from fake_radio_link import POWER_REQUEST, PROGRAM_HOURLY, STATUS_SHORT

from custom_components.termoweb.backend.radio import protocol as p
from custom_components.termoweb.codecs import radio_codec as codec
from custom_components.termoweb.domain.commands import (
    SetLock,
    SetMode,
    SetPresetTemps,
    SetProgram,
    SetSetpoint,
    SetUnits,
)
from custom_components.termoweb.planner import radio_planner as planner

# E6 full status: presets 16.5/18.5/21.0, mode 03, room 20.4, setpoint 22.0,
# power 1150.4 W, duty 0x10, flags heating|locked.
STATUS_E6 = bytes.fromhex("B921252A0300CC2C2CF0100300FF")
# Hourly day used by the reference heater: night 0-4, day 5-20, night 21-23.
DAY = [1] * 5 + [2] * 16 + [1] * 3


def test_short_status_maps_only_what_it_carries() -> None:
    """The dialect-B short record yields mode, presets and units, nothing else."""

    settings = codec.settings_from_status(p.decode_status(STATUS_SHORT))

    assert settings == {
        "units": "C",
        "ptemp": ["16.5", "18.5", "21.0"],
        "mode": "manual",
    }


def test_full_status_maps_room_setpoint_power_state_and_lock() -> None:
    """The E6 record adds mtemp, stemp, max_power, state and lock."""

    settings = codec.settings_from_status(p.decode_status(STATUS_E6))

    assert settings == {
        "units": "C",
        "ptemp": ["16.5", "18.5", "21.0"],
        "mode": "modified_auto",
        "mtemp": "20.4",
        "stemp": "22.0",
        "max_power": 1150.4,
        "state": "on",
        "lock": True,
    }
    idle = STATUS_E6[:-3] + bytes([0x00, 0x00, 0xFF])
    assert codec.settings_from_status(p.decode_status(idle))["state"] == "off"


@pytest.mark.parametrize(
    ("code", "mode"),
    [(1, "auto"), (2, "manual"), (3, "modified_auto"), (4, "off"), (9, None)],
)
def test_mode_mapping(code: int, mode: str | None) -> None:
    """Radio mode codes map to the canonical cloud mode strings; unknown is omitted."""

    settings = codec.settings_from_status(
        p.decode_status(STATUS_SHORT[:4] + bytes([code]))
    )

    assert settings.get("mode") == mode


def test_program_is_monday_first_168_ints() -> None:
    """The Sunday-first wire program becomes the cloud's Monday-first prog."""

    record = p.decode_program(PROGRAM_HOURLY)
    prog = codec.prog_from_program(record)

    assert prog == DAY * 7
    # Wire day 0 is Sunday; make Sunday distinct and check where it lands.
    sunday = [0] * 24
    wire = p._pack_slots(sunday + DAY * 6)  # noqa: SLF001 - build a test vector
    prog = codec.prog_from_program(p.decode_program(b"\xb1" + wire))
    assert prog[:24] == DAY  # Monday
    assert prog[-24:] == sunday  # Sunday is the last day


def test_program_with_split_half_hours_is_omitted() -> None:
    """A half-hourly program with a split hour has no faithful hourly form."""

    slots = [2] * (48 * 7)
    slots[1] = 0
    record = p.decode_program(b"\xb1" + p._pack_slots(slots))  # noqa: SLF001

    assert codec.prog_from_program(record) is None


def test_power_request_maps_max_power() -> None:
    """A dialect-B BE power request carries the full-load power in deciwatts."""

    request = p.decode_power_request(POWER_REQUEST)

    assert codec.settings_from_power_request(request) == {"max_power": 1143.3}


@pytest.mark.parametrize(
    ("command", "payload"),
    [
        (SetMode("auto"), "B401"),
        (SetMode("manual"), "B402"),
        (SetMode("Heat"), "B402"),
        (SetMode("off"), "B404"),
        (SetSetpoint(21.5, mode="manual"), "B4022B"),
        (SetSetpoint("19.0"), "B40226"),
        (SetSetpoint(22, mode="modified_auto"), "B4032C"),
        (SetPresetTemps([7, "16.5", 21.0]), "B60E212A"),
        (SetLock(True), "BA01"),
        (SetLock(False), "BA00"),
    ],
)
def test_encode_command(command, payload: str) -> None:
    """Canonical commands encode to the documented radio payloads."""

    assert codec.encode_command(command) == bytes.fromhex(payload)


def test_encode_program_rotates_to_sunday_first() -> None:
    """A Monday-first prog is written Sunday-first with 48 slots a day."""

    prog = DAY * 6 + [0] * 24  # Sunday all cold
    payload = codec.encode_command(SetProgram(prog))

    assert payload[0] == p.OP_PROGRAM_WRITE and len(payload) == 85
    decoded = p.decode_program(b"\xb1" + payload[1:])
    assert codec.prog_from_program(decoded) == prog
    assert decoded.hourly[:24] == tuple([0] * 24)  # wire day 0 = Sunday


@pytest.mark.parametrize(
    ("command", "message"),
    [
        (SetMode("modified_auto"), "needs a setpoint"),
        (SetMode("boost"), "not supported"),
        (SetSetpoint(20, mode="auto"), "cannot be written"),
        (SetSetpoint("warm"), "invalid setpoint"),
        (SetSetpoint(float("nan")), "invalid setpoint"),
        (SetSetpoint(40), "outside"),
        (SetPresetTemps([10, 12]), "three values"),
        (SetPresetTemps("101"), "three values"),
        (SetPresetTemps([10, 9, 12]), "presets must satisfy"),
        (SetProgram([0] * 167), "168 values"),
        (SetProgram("0" * 168), "168 values"),
        (SetProgram([0] * 167 + ["x"]), "non-integer"),
        (SetProgram([0] * 167 + [3]), "must be 0, 1 or 2"),
    ],
)
def test_encode_command_rejects_invalid_input(command, message: str) -> None:
    """Invalid writes fail before anything is sent."""

    with pytest.raises(ValueError, match=message):
        codec.encode_command(command)


def test_encode_command_rejects_unknown_command() -> None:
    """Commands the radio has no payload for raise TypeError."""

    with pytest.raises(TypeError, match="SetUnits"):
        codec.encode_command(SetUnits("C"))


def test_planner_orders_presets_program_then_mode() -> None:
    """Presets and program go first so the final frame sets the mode."""

    commands = planner.plan_settings(
        mode="manual", stemp=21.0, prog=DAY * 7, ptemp=[7.0, 16.0, 21.0]
    )
    planned = planner.plan_commands(commands)

    assert [type(c) for c in commands] == [SetPresetTemps, SetProgram, SetSetpoint]
    assert [w.opcode for w in planned] == [0xB6, 0xB2, 0xB4]
    assert planned[-1].payload == bytes.fromhex("B4022A")
    assert planner.plan_settings(mode="off") == [SetMode("off")]
    assert planner.plan_settings() == []


def test_planner_rejects_fahrenheit_and_bad_units() -> None:
    """The radio only speaks Celsius."""

    with pytest.raises(ValueError, match="Celsius"):
        planner.plan_settings(mode="auto", units="F")
    with pytest.raises(ValueError, match="Invalid units"):
        planner.plan_settings(mode="auto", units="K")
