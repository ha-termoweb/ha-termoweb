"""Typed energy domain models for coordinator snapshots."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from .ids import NodeId, NodeType, normalize_node_type


@dataclass(frozen=True, slots=True)
class EnergySamplePoint:
    """Represent a single energy counter reading."""

    t: float
    counter: float


@dataclass(frozen=True, slots=True)
class EnergyNodeMetrics:
    """Derived energy metrics for a node at a point in time."""

    energy_kwh: float | None
    power_w: float | None
    source: str
    ts: float


@dataclass(frozen=True, slots=True)
class EnergySnapshot:
    """Immutable energy snapshot published by the coordinator."""

    dev_id: str
    metrics: Mapping[NodeId, EnergyNodeMetrics]
    updated_at: float
    ws_deadline: float | None

    def metrics_for_type(
        self, node_type: NodeType | str
    ) -> dict[str, EnergyNodeMetrics]:
        """Return metrics keyed by address for ``node_type`` when known."""

        try:
            normalized_type = normalize_node_type(node_type)
        except ValueError:
            return {}

        return {
            node_id.addr: metrics
            for node_id, metrics in self.metrics.items()
            if node_id.node_type is normalized_type
        }

    def iter_metrics(self) -> Iterator[tuple[NodeId, EnergyNodeMetrics]]:
        """Yield stored metrics keyed by node identifier."""

        yield from self.metrics.items()


def build_empty_snapshot(
    dev_id: str, *, ws_deadline: float | None = None
) -> EnergySnapshot:
    """Return an empty snapshot placeholder for ``dev_id``."""

    return EnergySnapshot(
        dev_id=dev_id,
        metrics={},
        updated_at=0.0,
        ws_deadline=ws_deadline,
    )


def coerce_snapshot(candidate: Any) -> EnergySnapshot | None:
    """Return ``candidate`` when it behaves like an :class:`EnergySnapshot`."""

    return candidate if isinstance(candidate, EnergySnapshot) else None
