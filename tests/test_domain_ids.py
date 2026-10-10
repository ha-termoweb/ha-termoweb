"""Unit tests for domain IDs."""

import pytest

from custom_components.termoweb.domain.energy import EnergyNodeMetrics, EnergySnapshot
from custom_components.termoweb.domain.ids import (
    ENERGY_NODE_TYPES,
    HEATER_NODE_TYPES,
    HEATING_NODE_TYPES,
    NodeId,
    NodeType,
)
from custom_components.termoweb.domain.state import DomainStateStore
from custom_components.termoweb.domain.view import DomainStateView


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("htr", NodeType.HEATER),
        ("acm", NodeType.ACCUMULATOR),
        ("thm", NodeType.THERMOSTAT),
        ("pmo", NodeType.POWER_MONITOR),
        ("HTR", NodeType.HEATER),
        (" Acm ", NodeType.ACCUMULATOR),
        (NodeType.THERMOSTAT, NodeType.THERMOSTAT),
    ],
)
def test_coerce_accepts_known_types_case_insensitively(
    value: object, expected: NodeType
) -> None:
    """NodeType.coerce strips and lowercases before matching."""

    assert NodeType.coerce(value) is expected


@pytest.mark.parametrize("value", ["xyz", "", "   ", None, 7, b"htr", object()])
def test_coerce_returns_none_for_unknown(value: object) -> None:
    """NodeType.coerce returns None for anything that is not a node type."""

    assert NodeType.coerce(value) is None


def test_constructor_stays_strict() -> None:
    """The enum constructor is the strict, raising form."""

    with pytest.raises(ValueError):
        NodeType("HTR")


def test_node_type_sets_are_canonical_strings() -> None:
    """The node-type sets are plain strings drawn from NodeType values."""

    values = {member.value for member in NodeType}
    for group in (ENERGY_NODE_TYPES, HEATER_NODE_TYPES, HEATING_NODE_TYPES):
        assert group <= values
        assert all(type(item) is str for item in group)
    assert ENERGY_NODE_TYPES == {"htr", "acm", "pmo"}
    assert HEATER_NODE_TYPES == {"htr", "acm", "thm"}
    assert HEATING_NODE_TYPES == {"htr", "acm"}


def _view_with_metric() -> tuple[DomainStateView, EnergyNodeMetrics]:
    """Return a view whose snapshot holds one heater metric."""

    node_id = NodeId(NodeType.HEATER, "1")
    metric = EnergyNodeMetrics(energy_kwh=1.0, power_w=2.0, source="ws", ts=3.0)
    store = DomainStateStore([node_id])
    store.set_energy_snapshot(
        EnergySnapshot(
            dev_id="dev", metrics={node_id: metric}, updated_at=1.0, ws_deadline=None
        )
    )
    return DomainStateView("dev", store), metric


def test_energy_lookups_are_case_insensitive() -> None:
    """Energy lookups accept the same node-type spellings as state lookups."""

    view, metric = _view_with_metric()

    assert view.get_energy_metric("HTR", "1") is metric
    assert view.get_energy_metrics_for_type("HTR") == {"1": metric}
    snapshot = view.get_energy_snapshot()
    assert snapshot is not None
    assert snapshot.metrics_for_type(" Htr ") == {"1": metric}
    assert view.get_energy_metric("nope", "1") is None
    assert snapshot.metrics_for_type("nope") == {}


def test_node_id_equality_and_hash() -> None:
    """NodeId equality and hashing are based on contents."""

    node_a = NodeId(NodeType.HEATER, "1")
    node_b = NodeId(NodeType.HEATER, "1")
    node_c = NodeId(NodeType.THERMOSTAT, "1")

    assert node_a == node_b
    assert hash(node_a) == hash(node_b)
    assert node_a != node_c
