"""Tests for the domain state view façade."""

from __future__ import annotations

from custom_components.termoweb.domain.ids import NodeId, NodeType
from custom_components.termoweb.domain.state import DomainStateStore
from custom_components.termoweb.domain.view import DomainStateView


def _legacy_payload() -> dict[str, dict[str, dict[str, dict[str, str]]]]:
    """Return a representative legacy settings payload."""

    return {"settings": {"htr": {"01": {"mode": "manual"}}}}


def test_domain_state_view_prefers_store_data() -> None:
    """DomainStateView should return store-backed state when available."""

    store = DomainStateStore([NodeId(NodeType.HEATER, "01")])
    store.apply_full_snapshot("htr", "01", {"mode": "auto"})
    view = DomainStateView("dev", store, _legacy_payload)

    state = view.get_heater_state("htr", "01")

    assert state is not None
    assert state.mode == "auto"


def test_domain_state_view_falls_back_to_legacy_payload() -> None:
    """DomainStateView should return state built from legacy data when needed."""

    view = DomainStateView("dev", None, _legacy_payload)

    state = view.get_heater_state("htr", "01")

    assert state is not None
    assert state.mode == "manual"
