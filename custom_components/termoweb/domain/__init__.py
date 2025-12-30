"""Domain-layer primitives for TermoWeb integration."""

from .commands import (
    AccumulatorCommand,
    BaseCommand,
    SetExtraOptions,
    SetMode,
    SetPresetTemps,
    SetProgram,
    SetSetpoint,
    SetUnits,
    StartBoost,
    StopBoost,
)
from .ids import NodeId, NodeType, normalize_node_type
from .inventory import InstallationInventory, NodeInventory
from .state import AccumulatorState, HeaterState, PowerMonitorState, ThermostatState

__all__ = [
    "AccumulatorCommand",
    "AccumulatorState",
    "BaseCommand",
    "HeaterState",
    "InstallationInventory",
    "NodeId",
    "NodeInventory",
    "NodeType",
    "PowerMonitorState",
    "SetExtraOptions",
    "SetMode",
    "SetPresetTemps",
    "SetProgram",
    "SetSetpoint",
    "SetUnits",
    "StartBoost",
    "StopBoost",
    "ThermostatState",
    "normalize_node_type",
]
