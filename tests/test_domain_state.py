"""Unit tests for domain state models."""

from typing import Any

import pytest

from custom_components.termoweb.domain.state import (
    AccumulatorState,
    GatewayConnectionState,
    HeaterState,
    NodeDelta,
    NodeSettingsDelta,
    NodeStatusDelta,
    PowerMonitorState,
    ThermostatState,
    _build_heater_state,
    _build_accumulator_state,
    _build_power_monitor_state,
    _build_state,
    _build_thermostat_state,
    _copy_sequence,
    _merge_state,
    _normalize_node_type,
    _populate_heater_state,
    _populate_power_monitor_state,
    canonicalize_settings_payload,
    clone_gateway_connection_state,
    clone_state,
    state_to_dict,
)
from custom_components.termoweb.domain.ids import NodeId, NodeType


def test_accumulator_inherits_heater_state() -> None:
    """AccumulatorState should inherit from HeaterState and add fields."""

    state = AccumulatorState()

    assert isinstance(state, HeaterState)
    assert state.charge_level is None
    assert state.boost_active is None


# ---------------------------------------------------------------------------
# _copy_sequence: cover tuple branch (line 20)
# ---------------------------------------------------------------------------


def test_copy_sequence_converts_tuple_to_list() -> None:
    """_copy_sequence should convert a tuple to a list."""

    result = _copy_sequence((1, 2, 3))
    assert result == [1, 2, 3]
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# canonicalize_settings_payload: non-Mapping branch (line 137)
# ---------------------------------------------------------------------------


def test_canonicalize_settings_payload_rejects_non_mapping() -> None:
    """canonicalize_settings_payload should return empty dict for non-Mapping."""

    assert canonicalize_settings_payload("not a mapping") == {}  # type: ignore[arg-type]
    assert canonicalize_settings_payload(42) == {}  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# NodeDelta.payload returns empty dict (line 167)
# ---------------------------------------------------------------------------


def test_node_delta_base_payload_returns_empty() -> None:
    """NodeDelta base class should return an empty payload mapping."""

    delta = NodeDelta(node_id=NodeId(NodeType.HEATER, "1"))
    assert delta.payload == {}


# ---------------------------------------------------------------------------
# _populate_heater_state: various coercion branches
# ---------------------------------------------------------------------------


def test_populate_heater_state_mode_none(  # line 205
) -> None:
    """Setting mode to None should store None."""

    state = _build_heater_state({"mode": None})
    assert state.mode is None


def test_populate_heater_state_mode_non_string() -> None:
    """Non-string mode values should be stringified (line 209)."""

    state = _build_heater_state({"mode": 42})
    assert state.mode == "42"


def test_populate_heater_state_temp_field() -> None:
    """'temp' field should be stored directly (line 215)."""

    state = _build_heater_state({"temp": 22.5})
    assert state.temp == 22.5


def test_populate_heater_state_units_none() -> None:
    """Setting units to None should store None (line 223)."""

    state = _build_heater_state({"units": None})
    assert state.units is None


def test_populate_heater_state_units_non_string() -> None:
    """Non-string units should be stringified (line 227)."""

    state = _build_heater_state({"units": 1})
    assert state.units == "1"


def test_populate_heater_state_state_none() -> None:
    """Setting state to None should store None (line 231)."""

    state = _build_heater_state({"state": None})
    assert state.state is None


def test_populate_heater_state_state_non_string() -> None:
    """Non-string state should be stringified (line 235)."""

    state = _build_heater_state({"state": 99})
    assert state.state == "99"


def test_populate_heater_state_lock_int_float() -> None:
    """Numeric lock values should be coerced to bool (lines 246-248)."""

    state = _build_heater_state({"lock": 1})
    assert state.lock is True

    state2 = _build_heater_state({"lock": 0.0})
    assert state2.lock is False


def test_populate_heater_state_lock_unknown_string() -> None:
    """Unrecognized lock strings should result in None (lines 256-258)."""

    state = _build_heater_state({"lock": "maybe"})
    assert state.lock is None


def test_populate_heater_state_lock_non_bool_non_number_non_string() -> None:
    """Non-bool, non-number, non-string lock should result in None (line 258)."""

    state = _build_heater_state({"lock": [1, 2]})
    assert state.lock is None


# ---------------------------------------------------------------------------
# _populate_accumulator_fields: charging coercion
# ---------------------------------------------------------------------------


def test_accumulator_charging_int_float() -> None:
    """Numeric charging values should coerce to bool (line 284)."""

    state = _build_accumulator_state({"charging": 1})
    assert state.charging is True

    state2 = _build_accumulator_state({"charging": 0})
    assert state2.charging is False


def test_accumulator_charging_non_coercible() -> None:
    """Non-coercible charging values should result in None (line 296)."""

    state = _build_accumulator_state({"charging": "maybe"})
    assert state.charging is None


def test_accumulator_boost_temp_and_time_fields() -> None:
    """Accumulator should accept boost_temp and boost_time (lines 296-298)."""

    state = _build_accumulator_state({"boost_temp": 30, "boost_time": 60})
    assert state.boost_temp == 30
    assert state.boost_time == 60


# ---------------------------------------------------------------------------
# _build_power_monitor_state and _populate_power_monitor_state
# ---------------------------------------------------------------------------


def test_build_power_monitor_state_basic() -> None:
    """Build a power monitor state with basic fields (lines 325-326)."""

    state = _build_power_monitor_state({"power": 100, "voltage": 230})
    assert isinstance(state, PowerMonitorState)
    assert state.power == 100
    assert state.voltage == 230


def test_power_monitor_state_status_fallback() -> None:
    """Power monitor fields in status should fill in when top-level is missing (lines 334-352)."""

    state = _build_power_monitor_state({
        "status": {
            "power": 150,
            "voltage": 220,
            "current": 0.68,
            "energy": 5.0,
        }
    })
    assert state.power == 150
    assert state.voltage == 220
    assert state.current == 0.68
    assert state.energy == 5.0


def test_power_monitor_state_top_level_overrides_status() -> None:
    """Top-level fields should take precedence over status fields."""

    state = _build_power_monitor_state({
        "power": 200,
        "status": {"power": 100, "voltage": 230}
    })
    assert state.power == 200
    assert state.voltage == 230


# ---------------------------------------------------------------------------
# _build_state and _merge_state dispatch
# ---------------------------------------------------------------------------


def test_build_state_thermostat() -> None:
    """_build_state should build ThermostatState for THERMOSTAT (line 363)."""

    state = _build_state(NodeType.THERMOSTAT, {"mode": "auto"})
    assert isinstance(state, ThermostatState)
    assert state.mode == "auto"


def test_merge_state_accumulator() -> None:
    """_merge_state should merge into AccumulatorState (line 371)."""

    state = AccumulatorState(mode="auto")
    merged = _merge_state(state, {"charge_level": 50})
    assert isinstance(merged, AccumulatorState)
    assert merged.charge_level == 50


def test_merge_state_thermostat() -> None:
    """_merge_state should merge into ThermostatState (line 373)."""

    state = ThermostatState(mode="manual")
    merged = _merge_state(state, {"stemp": "22.0"})
    assert isinstance(merged, ThermostatState)
    assert merged.stemp == "22.0"


def test_merge_state_power_monitor() -> None:
    """_merge_state should merge into PowerMonitorState (line 375)."""

    state = PowerMonitorState(power=100)
    merged = _merge_state(state, {"voltage": 230})
    assert isinstance(merged, PowerMonitorState)
    assert merged.voltage == 230
    assert merged.power == 100


def test_merge_state_heater_fallback() -> None:
    """_merge_state on plain HeaterState goes through the fallback (line 376)."""

    state = HeaterState(mode="auto")
    merged = _merge_state(state, {"stemp": "20.0"})
    assert isinstance(merged, HeaterState)
    assert merged.stemp == "20.0"


# ---------------------------------------------------------------------------
# DomainStateStore edge cases
# ---------------------------------------------------------------------------


def test_store_reset_nodes_filters_non_node_id() -> None:
    """reset_nodes should skip non-NodeId entries (line 399)."""

    from custom_components.termoweb.domain.state import DomainStateStore

    store = DomainStateStore([NodeId(NodeType.HEATER, "1")])
    store.reset_nodes(["not-a-node-id", NodeId(NodeType.HEATER, "1")])  # type: ignore[list-item]
    assert store.addresses_by_type == {"htr": ("1",)}


def test_store_prune_energy_snapshot_empty_metrics() -> None:
    """Pruning an empty-metrics snapshot returns it unchanged (line 420)."""

    from custom_components.termoweb.domain.energy import EnergySnapshot
    from custom_components.termoweb.domain.state import DomainStateStore

    store = DomainStateStore([NodeId(NodeType.HEATER, "1")])
    snapshot = EnergySnapshot(dev_id="dev", metrics={}, updated_at=1.0, ws_deadline=None)
    result = store._prune_energy_snapshot(snapshot)
    assert result is snapshot  # unchanged, returned as-is


def test_store_resolve_node_id_invalid_type() -> None:
    """_resolve_node_id should return None for unknown type strings (line 440)."""

    from custom_components.termoweb.domain.state import DomainStateStore

    store = DomainStateStore([NodeId(NodeType.HEATER, "1")])
    assert store._resolve_node_id("unknown_type", "1") is None


def test_store_resolve_node_id_invalid_addr() -> None:
    """_resolve_node_id should return None for invalid addr (line 444-445)."""

    from custom_components.termoweb.domain.state import DomainStateStore

    store = DomainStateStore([NodeId(NodeType.HEATER, "1")])
    assert store._resolve_node_id("htr", "") is None


def test_store_apply_payload_non_mapping() -> None:
    """_apply_payload should skip non-Mapping payloads (line 459)."""

    from custom_components.termoweb.domain.state import DomainStateStore

    store = DomainStateStore([NodeId(NodeType.HEATER, "1")])
    node_id = NodeId(NodeType.HEATER, "1")
    store._apply_payload(node_id, "not-a-mapping", replace=False)  # type: ignore[arg-type]
    assert store.get_state("htr", "1") is None


def test_store_apply_payload_empty_no_replace_skips() -> None:
    """_apply_payload with empty normalized and replace=False should skip (line 466)."""

    from custom_components.termoweb.domain.state import DomainStateStore

    store = DomainStateStore([NodeId(NodeType.HEATER, "1")])
    node_id = NodeId(NodeType.HEATER, "1")
    store._apply_payload(node_id, {"unknown_field": "value"}, replace=False)
    assert store.get_state("htr", "1") is None


def test_store_apply_full_snapshot_non_mapping_skips() -> None:
    """apply_full_snapshot should skip None decoded_settings (line 483)."""

    from custom_components.termoweb.domain.state import DomainStateStore

    store = DomainStateStore([NodeId(NodeType.HEATER, "1")])
    store.apply_full_snapshot("htr", "1", None)
    assert store.get_state("htr", "1") is None


def test_store_apply_full_snapshot_unknown_node_skips() -> None:
    """apply_full_snapshot for unregistered node should skip (line 487)."""

    from custom_components.termoweb.domain.state import DomainStateStore

    store = DomainStateStore([NodeId(NodeType.HEATER, "1")])
    store.apply_full_snapshot("htr", "99", {"mode": "auto"})
    assert store.get_state("htr", "99") is None


def test_store_apply_patch_non_mapping_skips() -> None:
    """apply_patch should skip None delta (line 500)."""

    from custom_components.termoweb.domain.state import DomainStateStore

    store = DomainStateStore([NodeId(NodeType.HEATER, "1")])
    store.apply_patch("htr", "1", None)
    assert store.get_state("htr", "1") is None


def test_store_apply_patch_unknown_node_skips() -> None:
    """apply_patch for unregistered node should skip (line 504)."""

    from custom_components.termoweb.domain.state import DomainStateStore

    store = DomainStateStore([NodeId(NodeType.HEATER, "1")])
    store.apply_patch("htr", "99", {"mode": "auto"})
    assert store.get_state("htr", "99") is None


def test_store_apply_delta_non_delta_skips() -> None:
    """apply_delta should skip non-NodeDelta inputs (line 512)."""

    from custom_components.termoweb.domain.state import DomainStateStore

    store = DomainStateStore([NodeId(NodeType.HEATER, "1")])
    store.apply_delta(None)
    store.apply_delta("not-a-delta")  # type: ignore[arg-type]
    assert store.get_state("htr", "1") is None


def test_store_apply_delta_unknown_node_skips() -> None:
    """apply_delta with unregistered node should skip (line 516)."""

    from custom_components.termoweb.domain.state import DomainStateStore

    store = DomainStateStore([NodeId(NodeType.HEATER, "1")])
    delta = NodeSettingsDelta(
        node_id=NodeId(NodeType.HEATER, "99"),
        changes={"mode": "auto"},
    )
    store.apply_delta(delta)


def test_store_get_state_unknown_type() -> None:
    """get_state should return None for unknown type (line 525)."""

    from custom_components.termoweb.domain.state import DomainStateStore

    store = DomainStateStore([NodeId(NodeType.HEATER, "1")])
    assert store.get_state("zzz", "1") is None


def test_store_set_energy_snapshot_non_snapshot() -> None:
    """set_energy_snapshot should return False for non-EnergySnapshot (line 537)."""

    from custom_components.termoweb.domain.state import DomainStateStore

    store = DomainStateStore([])
    assert store.set_energy_snapshot("not-a-snapshot") is False  # type: ignore[arg-type]


def test_store_set_energy_snapshot_unchanged() -> None:
    """set_energy_snapshot returns False when unchanged (line 540)."""

    from custom_components.termoweb.domain.energy import EnergySnapshot
    from custom_components.termoweb.domain.state import DomainStateStore

    store = DomainStateStore([])
    snapshot = EnergySnapshot(dev_id="dev", metrics={}, updated_at=1.0, ws_deadline=None)
    assert store.set_energy_snapshot(snapshot) is True
    assert store.set_energy_snapshot(snapshot) is False


def test_store_set_gateway_connection_rejects_non_instance() -> None:
    """set_gateway_connection_state should skip non-GatewayConnectionState (line 562)."""

    from custom_components.termoweb.domain.state import DomainStateStore

    store = DomainStateStore([])
    store.set_gateway_connection_state("not-a-state")  # type: ignore[arg-type]
    gw = store.get_gateway_connection_state()
    assert gw.connected is False


# ---------------------------------------------------------------------------
# replace_state edge cases
# ---------------------------------------------------------------------------


def test_replace_state_accumulator_and_thermostat() -> None:
    """replace_state should match expected types for accumulator and thermostat (lines 590, 592)."""

    from custom_components.termoweb.domain.state import DomainStateStore

    store = DomainStateStore([
        NodeId(NodeType.ACCUMULATOR, "1"),
        NodeId(NodeType.THERMOSTAT, "2"),
        NodeId(NodeType.POWER_MONITOR, "3"),
    ])
    store.replace_state("acm", "1", AccumulatorState(mode="boost"))
    assert store.get_state("acm", "1").mode == "boost"

    store.replace_state("thm", "2", ThermostatState(mode="auto"))
    assert store.get_state("thm", "2").mode == "auto"

    store.replace_state("pmo", "3", PowerMonitorState(power=200))
    assert store.get_state("pmo", "3").power == 200

    with pytest.raises(TypeError, match="does not match"):
        store.replace_state("acm", "1", HeaterState())


# ---------------------------------------------------------------------------
# _normalize_node_type
# ---------------------------------------------------------------------------


def test_normalize_node_type_case_insensitive() -> None:
    """_normalize_node_type should handle case-insensitive strings (lines 614-618)."""

    assert _normalize_node_type(NodeType.HEATER) is NodeType.HEATER
    assert _normalize_node_type("htr") is NodeType.HEATER
    assert _normalize_node_type("HTR") is NodeType.HEATER
    assert _normalize_node_type("unknown") is None


# ---------------------------------------------------------------------------
# clone_state for None
# ---------------------------------------------------------------------------


def test_clone_state_none() -> None:
    """clone_state should return None when given None (line 655)."""

    assert clone_state(None) is None


# ---------------------------------------------------------------------------
# clone_gateway_connection_state for None
# ---------------------------------------------------------------------------


def test_clone_gateway_connection_state_none() -> None:
    """clone_gateway_connection_state should return defaults for None (line 671)."""

    result = clone_gateway_connection_state(None)
    assert isinstance(result, GatewayConnectionState)
    assert result.connected is False


# ---------------------------------------------------------------------------
# state_to_dict with None state
# ---------------------------------------------------------------------------


def test_state_to_dict_none() -> None:
    """state_to_dict should return empty dict for None (line 639)."""

    assert state_to_dict(None) == {}
