"""Domain runtime state objects."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class HeaterState:
    """Runtime state for a heater node."""


@dataclass(slots=True)
class AccumulatorState(HeaterState):
    """Runtime state for an accumulator node."""

    charge_level: float | None = None
    boost_active: bool = False


@dataclass(slots=True)
class ThermostatState:
    """Runtime state for a thermostat node."""


@dataclass(slots=True)
class PowerMonitorState:
    """Runtime state for a power monitor node."""
