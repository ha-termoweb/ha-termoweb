"""Unit tests for domain IDs."""

from custom_components.termoweb.domain.ids import NodeId, NodeType, normalize_node_type


def test_normalize_node_type_accepts_literals() -> None:
    """normalize_node_type should return matching NodeType for literals."""

    assert normalize_node_type("htr") is NodeType.HEATER
    assert normalize_node_type("acm") is NodeType.ACCUMULATOR
    assert normalize_node_type("thm") is NodeType.THERMOSTAT
    assert normalize_node_type("pmo") is NodeType.POWER_MONITOR


def test_normalize_node_type_rejects_unknown() -> None:
    """normalize_node_type should reject unsupported values."""

    try:
        normalize_node_type("xyz")
    except ValueError as err:
        assert "Unknown node type" in str(err)
    else:
        assert False, "normalize_node_type did not raise"


def test_node_id_equality_and_hash() -> None:
    """NodeId equality and hashing are based on contents."""

    node_a = NodeId(NodeType.HEATER, "1")
    node_b = NodeId(NodeType.HEATER, "1")
    node_c = NodeId(NodeType.THERMOSTAT, "1")

    assert node_a == node_b
    assert hash(node_a) == hash(node_b)
    assert node_a != node_c
