"""Domain inventory models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from .ids import NodeId


@dataclass(frozen=True, slots=True)
class NodeInventory:
    """Static metadata for a node."""

    node_id: NodeId
    name: str
    capabilities: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True, slots=True)
class InstallationInventory:
    """Static inventory for a TermoWeb installation."""

    dev_id: str
    nodes: Mapping[NodeId, NodeInventory]

    def __post_init__(self) -> None:
        """Ensure nodes mapping is immutable."""

        object.__setattr__(
            self,
            "nodes",
            MappingProxyType(dict(self.nodes)),
        )
