"""Tests for the domain state view façade."""

from __future__ import annotations

from custom_components.termoweb.domain.ids import NodeId, NodeType
from custom_components.termoweb.domain.state import (
    DomainStateStore,
    GatewayConnectionState,
)
from custom_components.termoweb.domain.view import DomainStateView


def test_domain_state_view_prefers_store_data() -> None:
    """DomainStateView should return store-backed state when available."""

    store = DomainStateStore([NodeId(NodeType.HEATER, "01")])
    store.apply_full_snapshot("htr", "01", {"mode": "auto"})
    view = DomainStateView("dev", store)

    state = view.get_heater_state("htr", "01")

    assert state is not None
    assert state.mode == "auto"


def test_domain_state_view_without_store_returns_none() -> None:
    """DomainStateView should return None when no store is available."""

    view = DomainStateView("dev", None)

    assert view.get_heater_state("htr", "01") is None


def test_domain_state_view_gateway_connection_state() -> None:
    """DomainStateView should return gateway connection state defaults."""

    store = DomainStateStore([])
    store.set_gateway_connection_state(
        GatewayConnectionState(status="connected", connected=True)
    )
    view = DomainStateView("dev", store)

    state = view.get_gateway_connection_state()

    assert state.status == "connected"
    assert state.connected is True

    empty_view = DomainStateView("dev", None)
    empty_state = empty_view.get_gateway_connection_state()
    assert empty_state.connected is False


# ---------------------------------------------------------------------------
# Power monitor state tests
# ---------------------------------------------------------------------------

from custom_components.termoweb.domain.state import PowerMonitorState


def test_get_power_monitor_state_returns_none_for_non_power_monitor() -> None:
    """get_power_monitor_state returns None when the state is not PowerMonitorState."""

    store = DomainStateStore([NodeId(NodeType.HEATER, "01")])
    store.apply_full_snapshot("htr", "01", {"mode": "auto"})
    view = DomainStateView("dev", store)

    result = view.get_power_monitor_state("01")
    assert result is None


def test_get_power_monitor_state_returns_state_when_present() -> None:
    """get_power_monitor_state returns the state when it is a PowerMonitorState."""

    store = DomainStateStore([NodeId(NodeType.POWER_MONITOR, "01")])
    store.apply_full_snapshot("pmo", "01", {"power": 1500})
    view = DomainStateView("dev", store)

    result = view.get_power_monitor_state("01")
    assert isinstance(result, PowerMonitorState)
    assert result.power == 1500


def test_get_power_monitor_state_no_store_returns_none() -> None:
    """get_power_monitor_state returns None when no store is set."""

    view = DomainStateView("dev", None)
    assert view.get_power_monitor_state("01") is None


# ---------------------------------------------------------------------------
# Energy snapshot tests
# ---------------------------------------------------------------------------

from custom_components.termoweb.domain.energy import EnergyNodeMetrics, EnergySnapshot


def test_get_energy_snapshot_returns_none_without_store() -> None:
    """get_energy_snapshot returns None when no store is set."""

    view = DomainStateView("dev", None)
    assert view.get_energy_snapshot() is None


def test_get_energy_snapshot_returns_none_when_dev_id_mismatch() -> None:
    """get_energy_snapshot returns None when the snapshot dev_id does not match."""

    store = DomainStateStore([])
    snapshot = EnergySnapshot(
        dev_id="other-dev",
        metrics={},
        updated_at=1.0,
        ws_deadline=None,
    )
    store.set_energy_snapshot(snapshot)
    view = DomainStateView("dev", store)

    assert view.get_energy_snapshot() is None


def test_get_energy_snapshot_returns_snapshot_when_matched() -> None:
    """get_energy_snapshot returns the snapshot when dev_id matches."""

    store = DomainStateStore([])
    snapshot = EnergySnapshot(
        dev_id="dev",
        metrics={},
        updated_at=1.0,
        ws_deadline=None,
    )
    store.set_energy_snapshot(snapshot)
    view = DomainStateView("dev", store)

    result = view.get_energy_snapshot()
    assert result is not None
    assert result.dev_id == "dev"


# ---------------------------------------------------------------------------
# Energy metric tests
# ---------------------------------------------------------------------------


def test_get_energy_metric_returns_none_without_snapshot() -> None:
    """get_energy_metric returns None when no snapshot exists."""

    view = DomainStateView("dev", None)
    assert view.get_energy_metric(NodeType.HEATER, "01") is None


def test_get_energy_metric_returns_none_for_invalid_node_type() -> None:
    """get_energy_metric returns None for an invalid node type string."""

    store = DomainStateStore([])
    snapshot = EnergySnapshot(
        dev_id="dev",
        metrics={},
        updated_at=1.0,
        ws_deadline=None,
    )
    store.set_energy_snapshot(snapshot)
    view = DomainStateView("dev", store)

    assert view.get_energy_metric("invalid_type", "01") is None


def test_get_energy_metric_returns_none_for_empty_addr() -> None:
    """get_energy_metric returns None for an empty address."""

    store = DomainStateStore([])
    snapshot = EnergySnapshot(
        dev_id="dev",
        metrics={},
        updated_at=1.0,
        ws_deadline=None,
    )
    store.set_energy_snapshot(snapshot)
    view = DomainStateView("dev", store)

    assert view.get_energy_metric(NodeType.HEATER, "") is None


def test_get_energy_metric_returns_metric_when_present() -> None:
    """get_energy_metric returns the metric for existing node."""

    node_id = NodeId(NodeType.HEATER, "01")
    metric = EnergyNodeMetrics(
        energy_kwh=1.5,
        power_w=500.0,
        source="ws",
        ts=1000.0,
    )
    store = DomainStateStore([node_id])
    snapshot = EnergySnapshot(
        dev_id="dev",
        metrics={node_id: metric},
        updated_at=1.0,
        ws_deadline=None,
    )
    store.set_energy_snapshot(snapshot)
    view = DomainStateView("dev", store)

    result = view.get_energy_metric(NodeType.HEATER, "01")
    assert result is not None
    assert result.energy_kwh == 1.5


# ---------------------------------------------------------------------------
# Energy metrics for type
# ---------------------------------------------------------------------------


def test_get_energy_metrics_for_type_no_snapshot() -> None:
    """get_energy_metrics_for_type returns empty dict without snapshot."""

    view = DomainStateView("dev", None)
    assert view.get_energy_metrics_for_type(NodeType.HEATER) == {}


def test_get_energy_metrics_for_type_returns_filtered() -> None:
    """get_energy_metrics_for_type filters by node type."""

    htr_id = NodeId(NodeType.HEATER, "01")
    pmo_id = NodeId(NodeType.POWER_MONITOR, "02")
    htr_metric = EnergyNodeMetrics(energy_kwh=1.0, power_w=100, source="ws", ts=1.0)
    pmo_metric = EnergyNodeMetrics(energy_kwh=2.0, power_w=200, source="ws", ts=1.0)

    store = DomainStateStore([htr_id, pmo_id])
    snapshot = EnergySnapshot(
        dev_id="dev",
        metrics={htr_id: htr_metric, pmo_id: pmo_metric},
        updated_at=1.0,
        ws_deadline=None,
    )
    store.set_energy_snapshot(snapshot)
    view = DomainStateView("dev", store)

    result = view.get_energy_metrics_for_type(NodeType.HEATER)
    assert "01" in result
    assert "02" not in result


# ---------------------------------------------------------------------------
# update_store
# ---------------------------------------------------------------------------


def test_update_store_changes_backing_store() -> None:
    """update_store refreshes the view's backing store reference."""

    view = DomainStateView("dev", None)
    assert view.get_heater_state("htr", "01") is None

    store = DomainStateStore([NodeId(NodeType.HEATER, "01")])
    store.apply_full_snapshot("htr", "01", {"mode": "manual"})
    view.update_store(store)

    state = view.get_heater_state("htr", "01")
    assert state is not None
    assert state.mode == "manual"
