"""Read-only façade for accessing domain state."""

from __future__ import annotations

from .energy import EnergyNodeMetrics, EnergySnapshot
from .ids import NodeId, NodeType, normalize_node_type
from .state import (
    DomainState,
    DomainStateStore,
    GatewayConnectionState,
    PowerMonitorState,
)


class DomainStateView:
    """Provide read-only access to domain state."""

    def __init__(
        self,
        dev_id: str,
        store: DomainStateStore | None,
    ) -> None:
        """Initialise the view for ``dev_id`` using ``store`` when available."""

        self._dev_id = str(dev_id)
        self._store = store

    def update_store(self, store: DomainStateStore | None) -> None:
        """Refresh the backing domain store reference."""

        self._store = store

    def _build_state(self, node_type: str | NodeType, addr: str) -> DomainState | None:
        """Return a domain state object using the store."""

        if self._store is not None:
            state = self._store.get_state(node_type, addr)
            if state is not None:
                return state

        return None

    def get_heater_state(
        self, node_type: str | NodeType, addr: str
    ) -> DomainState | None:
        """Return the heater/thermostat/accumulator state for ``addr``."""

        return self._build_state(node_type, addr)

    def get_power_monitor_state(self, addr: str) -> PowerMonitorState | None:
        """Return the power monitor state for ``addr`` when present."""

        state = self._build_state(NodeType.POWER_MONITOR, addr)
        return state if isinstance(state, PowerMonitorState) else None

    def get_gateway_connection_state(self) -> GatewayConnectionState:
        """Return the gateway connection state when available."""

        if self._store is None:
            return GatewayConnectionState()
        return self._store.get_gateway_connection_state()

    def get_energy_snapshot(self) -> EnergySnapshot | None:
        """Return the latest energy snapshot for this gateway."""

        if self._store is None:
            return None
        snapshot = self._store.get_energy_snapshot()
        if snapshot is None or snapshot.dev_id != self._dev_id:
            return None
        return snapshot

    def get_energy_metric(
        self, node_type: NodeType | str, addr: str
    ) -> EnergyNodeMetrics | None:
        """Return metrics for ``(node_type, addr)`` when available."""

        snapshot = self.get_energy_snapshot()
        if snapshot is None:
            return None
        try:
            normalized_type = normalize_node_type(node_type)
        except ValueError:
            return None
        try:
            node_id = NodeId(normalized_type, addr)
        except ValueError:
            return None
        return snapshot.metrics.get(node_id)

    def get_energy_metrics_for_type(
        self, node_type: NodeType | str
    ) -> dict[str, EnergyNodeMetrics]:
        """Return metrics keyed by address for ``node_type``."""

        snapshot = self.get_energy_snapshot()
        if snapshot is None:
            return {}
        return snapshot.metrics_for_type(node_type)
