"""Decode/plan contract of the Ducaheat codec and planner public functions."""

from __future__ import annotations

import pytest

from custom_components.termoweb.codecs.ducaheat_codec import decode_settings
from custom_components.termoweb.domain.commands import SetExtraOptions
from custom_components.termoweb.domain.ids import NodeId, NodeType
from custom_components.termoweb.planner.ducaheat_planner import plan_command


@pytest.mark.parametrize(
    ("node_type", "payload"),
    [
        (
            NodeType.HEATER,
            {"status": {"mode": None, "state": None, "units": None, "stemp": "20.0"}},
        ),
        (
            NodeType.ACCUMULATOR,
            {"status": {"mode": None, "state": None, "units": None, "stemp": "20.0"}},
        ),
        (
            NodeType.THERMOSTAT,
            {"mode": None, "state": None, "units": None, "stemp": "20.0"},
        ),
    ],
)
def test_null_status_fields_decode_as_absent(
    node_type: NodeType, payload: dict
) -> None:
    """Fields the schema declares nullable decode to absent, not to "none"."""
    decoded = decode_settings(payload, node_type=node_type)

    assert not {"mode", "state", "units"} & set(decoded)
    assert decoded["stemp"] in ("20.0", 20.0)


def test_empty_program_section_yields_no_program() -> None:
    """An empty ``prog`` mapping does not produce a weekly schedule."""
    decoded = decode_settings(
        {"status": {"mode": "auto"}, "prog": {}}, node_type=NodeType.HEATER
    )

    assert decoded == {"mode": "auto"}


def test_planner_rejects_commands_without_a_segment() -> None:
    """Accumulator extra options have no planner segment; planning them fails."""
    with pytest.raises(TypeError, match="SetExtraOptions"):
        plan_command(
            "dev", NodeId(NodeType.ACCUMULATOR, "1"), SetExtraOptions(boost_time=60)
        )
