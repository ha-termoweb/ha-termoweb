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
from .legacy_view import store_to_legacy_coordinator_data
from .state import (
    AccumulatorState,
    DomainStateStore,
    HeaterState,
    PowerMonitorState,
    ThermostatState,
)

__all__ = [
    "AccumulatorCommand",
    "AccumulatorState",
    "BaseCommand",
    "DomainStateStore",
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
    "store_to_legacy_coordinator_data",
    "normalize_node_type",
]
