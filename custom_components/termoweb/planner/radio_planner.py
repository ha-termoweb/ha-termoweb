"""Planner for radio writes: canonical settings → ordered radio payloads."""

from __future__ import annotations

from dataclasses import dataclass

from custom_components.termoweb.backend.radio import protocol
from custom_components.termoweb.backend.radio.dialect import DIALECT_A, Dialect
from custom_components.termoweb.codecs.common import validate_units
from custom_components.termoweb.codecs.radio_codec import (
    RADIO_UNITS,
    encode_command,
    encode_preset_mode_write,
)
from custom_components.termoweb.domain.commands import (
    BaseCommand,
    SetMode,
    SetPresetTemps,
    SetProgram,
    SetSetpoint,
)


@dataclass(frozen=True, slots=True)
class PlannedRadioWrite:
    """One radio payload to send; the heater answers ``<opcode+1> 55|56``."""

    payload: bytes

    @property
    def opcode(self) -> int:
        """Return the write opcode, the first payload byte."""

        return self.payload[0]


def plan_settings(
    *,
    mode: str | None = None,
    stemp: float | None = None,
    prog: list[int] | None = None,
    ptemp: list[float] | None = None,
    units: str = RADIO_UNITS,
) -> list[BaseCommand]:
    """Return the commands for a settings write: presets, program, then mode."""

    if validate_units(units, trim=True) != RADIO_UNITS:
        raise ValueError("the radio only accepts Celsius temperatures")
    commands: list[BaseCommand] = []
    if ptemp is not None:
        commands.append(SetPresetTemps(list(ptemp)))
    if prog is not None:
        commands.append(SetProgram(list(prog)))
    if stemp is not None:
        commands.append(SetSetpoint(stemp, mode=mode))
    elif mode is not None:
        commands.append(SetMode(mode))
    return commands


def needs_status(commands: list[BaseCommand], dialect: Dialect) -> bool:
    """Return True when ``dialect`` must re-send current values for ``commands``."""

    return dialect.mode_in_preset_write and any(
        isinstance(command, (SetPresetTemps, SetMode, SetSetpoint))
        for command in commands
    )


def validate_commands(commands: list[BaseCommand], dialect: Dialect) -> None:
    """Raise ValueError if any of ``commands`` cannot be encoded for ``dialect``."""

    for command in commands:
        encode_command(command, dialect)


def plan_commands(
    commands: list[BaseCommand],
    dialect: Dialect = DIALECT_A,
    status: protocol.StatusRecord | None = None,
) -> list[PlannedRadioWrite]:
    """Return the radio payloads for ``commands``, validating every one first."""

    if not needs_status(commands, dialect):
        return [
            PlannedRadioWrite(encode_command(command, dialect)) for command in commands
        ]
    if status is None:
        raise ValueError("preset and mode writes need the heater's current status")
    presets = next((c.presets for c in commands if isinstance(c, SetPresetTemps)), None)
    mode = next((c.mode for c in commands if isinstance(c, SetMode)), None)
    setpoint_command = next((c for c in commands if isinstance(c, SetSetpoint)), None)
    setpoint = None
    if setpoint_command is not None:
        setpoint, mode = setpoint_command.setpoint, setpoint_command.mode
    planned = [
        PlannedRadioWrite(
            encode_preset_mode_write(
                status, presets=presets, mode=mode, setpoint=setpoint
            )
        )
    ]
    planned.extend(
        PlannedRadioWrite(encode_command(command, dialect))
        for command in commands
        if not isinstance(command, (SetPresetTemps, SetMode, SetSetpoint))
    )
    return planned


__all__ = [
    "PlannedRadioWrite",
    "needs_status",
    "plan_commands",
    "plan_settings",
    "validate_commands",
]
