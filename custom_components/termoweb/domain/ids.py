"""Identifiers for domain objects."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class NodeType(str, Enum):
    """Supported node types."""

    HEATER = "htr"
    ACCUMULATOR = "acm"
    THERMOSTAT = "thm"
    POWER_MONITOR = "pmo"


NodeTypeLiteral = Literal["htr", "acm", "thm", "pmo"]


def normalize_node_type(node_type: NodeType | NodeTypeLiteral | str) -> NodeType:
    """Normalize assorted node type inputs to ``NodeType``."""

    try:
        return NodeType(node_type)
    except ValueError as err:
        raise ValueError(f"Unknown node type: {node_type}") from err


@dataclass(frozen=True, slots=True)
class NodeId:
    """Identifier for a node consisting of type and address."""

    node_type: NodeType
    addr: str

    def __post_init__(self) -> None:
        """Normalise the node address to a non-empty string."""

        addr_str = str(self.addr).strip()
        if not addr_str:
            msg = "addr must not be empty"
            raise ValueError(msg)
        object.__setattr__(self, "addr", addr_str)
