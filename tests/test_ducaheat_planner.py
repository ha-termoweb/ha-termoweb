"""Unit tests for the Ducaheat command planner."""

from __future__ import annotations

import pytest

from custom_components.termoweb.planner.ducaheat_planner import plan_command
from custom_components.termoweb.domain.commands import SetMode, StopBoost
from custom_components.termoweb.domain.ids import NodeId, NodeType


def test_plan_command_returns_write_call_only() -> None:
    """Ensure write planning only returns the segmented mutation call."""

    node_id = NodeId(NodeType.HEATER, "01")

    plan = plan_command("dev123", node_id, SetMode("Auto"))

    assert [call.method for call in plan] == ["POST"]
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
