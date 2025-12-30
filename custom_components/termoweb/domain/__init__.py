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
    NodeDelta,
    NodeSamplesDelta,
    NodeSettingsDelta,
    NodeStatusDelta,
    PowerMonitorState,
    ThermostatState,
    build_state_from_payload,
)
from .view import DomainStateView

__all__ = [
    "AccumulatorCommand",
    "AccumulatorState",
    "BaseCommand",
    "DomainStateStore",
    "DomainStateView",
    "HeaterState",
    "InstallationInventory",
    "NodeDelta",
    "NodeId",
    "NodeInventory",
    "NodeSamplesDelta",
    "NodeSettingsDelta",
    "NodeStatusDelta",
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
    "build_state_from_payload",
    "normalize_node_type",
    "store_to_legacy_coordinator_data",
]
