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
