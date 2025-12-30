"""Command placeholder types for future writes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class BaseCommand:
    """Base type for node commands."""


@dataclass(slots=True)
class SetMode(BaseCommand):
    """Change the operating mode for a node."""

    mode: str
    boost_time: int | None = None


@dataclass(slots=True)
class SetSetpoint(BaseCommand):
    """Update the target setpoint for manual mode."""

    setpoint: float | str
    mode: str | None = None
    boost_time: int | None = None


@dataclass(slots=True)
class SetProgram(BaseCommand):
    """Replace the full weekly program."""

    program: list[int]


@dataclass(slots=True)
class SetPresetTemps(BaseCommand):
    """Update preset temperatures."""

    presets: list[float | str]


@dataclass(slots=True)
class SetUnits(BaseCommand):
    """Specify the temperature unit for a payload."""

    units: str


@dataclass(slots=True)
class AccumulatorCommand(BaseCommand):
    """Command envelope for accumulator-specific actions."""


@dataclass(slots=True)
class SetExtraOptions(AccumulatorCommand):
    """Persist accumulator extra options such as boost defaults."""

    boost_time: int | None = None
    boost_temp: float | str | None = None


@dataclass(slots=True)
class StartBoost(AccumulatorCommand):
    """Start an accumulator boost session."""

    boost_time: int | None = None
    stemp: float | str | None = None
    units: str | None = None


@dataclass(slots=True)
class StopBoost(AccumulatorCommand):
    """Stop an accumulator boost session."""

    boost_time: int | None = None
    stemp: float | str | None = None
    units: str | None = None
