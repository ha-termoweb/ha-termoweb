"""Unit tests for the Ducaheat command planner."""

from __future__ import annotations

import pytest

from custom_components.termoweb.domain.commands import SetMode, SetPriority, StopBoost
from custom_components.termoweb.domain.ids import NodeId, NodeType
from custom_components.termoweb.planner.ducaheat_planner import plan_command


def test_plan_command_returns_write_call_only() -> None:
    """Ensure write planning only returns the segmented mutation call."""

    node_id = NodeId(NodeType.HEATER, "01")

    plan = plan_command("dev123", node_id, SetMode("Auto"))

    assert len(plan) == 1
    assert plan[0].path.endswith("/htr/01/status")
    assert plan[0].json == {"mode": "auto"}


def test_plan_command_rejects_boost_for_non_accumulators() -> None:
    """Guard boost commands to accumulator nodes."""

    node_id = NodeId(NodeType.HEATER, "01")

    with pytest.raises(ValueError, match="only supported for accumulators"):
        plan_command("dev123", node_id, StopBoost())


def test_plan_command_targets_accumulator_boost_endpoint() -> None:
    """Route boost commands to the accumulator boost segment."""

    node_id = NodeId(NodeType.ACCUMULATOR, "02")

    plan = plan_command("dev123", node_id, StopBoost())

    assert plan[0].path.endswith("/acm/02/boost")
    assert plan[0].json == {"boost": False}


@pytest.mark.parametrize("priority", [0, 7, 30])
def test_plan_command_writes_priority_to_the_setup_segment(priority: int) -> None:
    """Priority goes to the node's setup segment."""
    plan = plan_command("dev123", NodeId(NodeType.HEATER, "01"), SetPriority(priority))

    assert len(plan) == 1
    assert plan[0].path.endswith("/htr/01/setup")
    assert plan[0].json == {"priority": priority}


@pytest.mark.parametrize("priority", [-1, 31])
def test_plan_command_rejects_out_of_range_priority(priority: int) -> None:
    """Priorities outside 0-30 never reach the wire."""
    with pytest.raises(ValueError, match="priority"):
        plan_command("dev123", NodeId(NodeType.HEATER, "01"), SetPriority(priority))
