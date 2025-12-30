"""Domain-layer primitives for TermoWeb integration."""

from .commands import BaseCommand
from .ids import NodeId, NodeType, normalize_node_type
from .inventory import InstallationInventory, NodeInventory
from .state import HeaterState, PowerMonitorState, ThermostatState

__all__ = [
    "BaseCommand",
    "HeaterState",
    "InstallationInventory",
    "NodeId",
    "NodeInventory",
    "NodeType",
    "PowerMonitorState",
    "ThermostatState",
    "normalize_node_type",
]
