"""Tests for ``StateCoordinator._assemble_device_record``."""

from custom_components.termoweb.domain.ids import NodeId, NodeType
from custom_components.termoweb.domain.state import DomainStateStore, state_to_dict


def test_domain_store_snapshots_match_expected_payload() -> None:
    """Domain state store should expose typed settings snapshots."""

    settings = {
        "htr": {"01": {"stemp": 21}},
        "acm": {"07": {"mode": "auto"}},
    }

    store = DomainStateStore(
        [NodeId(NodeType.HEATER, "01"), NodeId(NodeType.ACCUMULATOR, "07")]
    )
    for node_type, bucket in settings.items():
        if not isinstance(bucket, dict):
            continue
        for addr, payload in bucket.items():
            store.apply_full_snapshot(node_type, addr, payload)

    record = {
        (node_id.node_type.value, node_id.addr): state_to_dict(state)
        for node_id, state in store.iter_states()
    }
    assert record[("htr", "01")] == {"stemp": 21}
    assert record[("acm", "07")] == {"mode": "auto"}
