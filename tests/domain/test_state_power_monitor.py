"""Power monitor readings flow from the GET payload into the domain state."""

from __future__ import annotations

from custom_components.termoweb.codecs.termoweb_codec import decode_node_settings
from custom_components.termoweb.domain.ids import NodeId, NodeType
from custom_components.termoweb.domain.state import DomainStateStore


def test_power_monitor_status_readings_reach_the_state() -> None:
    """Power, voltage, current and energy in ``status`` are all stored."""
    node_id = NodeId(NodeType.POWER_MONITOR, "1")
    store = DomainStateStore([node_id])
    payload = {
        "status": {"power": "150", "voltage": "230", "current": "0.65", "energy": "5"}
    }

    store.apply_full_snapshot("pmo", "1", decode_node_settings("pmo", payload))

    state = store.get_state("pmo", "1")
    assert (state.power, state.voltage, state.current, state.energy) == (
        150,
        230,
        0.65,
        5,
    )
