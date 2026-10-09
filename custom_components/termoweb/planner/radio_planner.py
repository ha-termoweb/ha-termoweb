"""Planner for radio writes: canonical settings → ordered radio payloads."""

from __future__ import annotations

from dataclasses import dataclass

from custom_components.termoweb.codecs.common import validate_units
from custom_components.termoweb.codecs.radio_codec import RADIO_UNITS, encode_command
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


def plan_commands(commands: list[BaseCommand]) -> list[PlannedRadioWrite]:
    """Return the radio payloads for ``commands``, validating every one first."""

    return [PlannedRadioWrite(encode_command(command)) for command in commands]


__all__ = ["PlannedRadioWrite", "plan_commands", "plan_settings"]
