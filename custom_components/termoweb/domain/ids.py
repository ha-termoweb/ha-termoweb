"""Identifiers for domain objects."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    """Supported node types."""

    HEATER = "htr"
    ACCUMULATOR = "acm"
    THERMOSTAT = "thm"
    POWER_MONITOR = "pmo"

    @classmethod
    def coerce(cls, value: Any) -> NodeType | None:
        """Return the NodeType for ``value`` (case-insensitive, stripped) or None."""

        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            return None
        try:
            return cls(value.strip().lower())
        except ValueError:
            return None


# Canonical node-type groups (plain strings; compare with normalised type strings).
HEATER_NODE_TYPES: frozenset[str] = frozenset({"htr", "acm", "thm"})
"""Node types with a heater-style climate entity."""

HEATING_NODE_TYPES: frozenset[str] = frozenset({"htr", "acm"})
"""Node types with their own heating element (lock, flash, total energy)."""

ENERGY_NODE_TYPES: frozenset[str] = frozenset({"htr", "acm", "pmo"})
"""Node types that report energy samples."""


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
