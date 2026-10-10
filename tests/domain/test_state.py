"""Tests for domain state, the state store and its read-only view."""

from __future__ import annotations

import datetime as dt
from typing import Any, Callable, Iterable, Iterator, Mapping

import pytest

from custom_components.termoweb.domain.ids import NodeId, NodeType
from custom_components.termoweb.domain.state import (
    AccumulatorState,
    DomainStateStore,
    GatewayConnectionState,
    HeaterState,
    NodeDelta,
    NodeSettingsDelta,
    PowerMonitorState,
    ThermostatState,
    _build_accumulator_state,
    _build_heater_state,
    _build_power_monitor_state,
    _build_state,
    _copy_sequence,
    _merge_state,
    canonicalize_settings_payload,
    clone_gateway_connection_state,
    clone_state,
    state_to_dict,
)
from custom_components.termoweb.domain.view import DomainStateView
from custom_components.termoweb.inventory import build_node_inventory


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

    state = _build_power_monitor_state(
        {
            "status": {
                "power": 150,
                "voltage": 220,
                "current": 0.68,
                "energy": 5.0,
            }
        }
    )
    assert state.power == 150
    assert state.voltage == 220
    assert state.current == 0.68
    assert state.energy == 5.0


def test_power_monitor_state_top_level_overrides_status() -> None:
    """Top-level fields should take precedence over status fields."""

    state = _build_power_monitor_state(
        {"power": 200, "status": {"power": 100, "voltage": 230}}
    )
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


def test_store_init_filters_non_node_id() -> None:
    """The store only admits NodeId entries."""

    from custom_components.termoweb.domain.state import DomainStateStore

    store = DomainStateStore(["not-a-node-id", NodeId(NodeType.HEATER, "1")])  # type: ignore[list-item]
    assert store.addresses_by_type == {"htr": ("1",)}


def test_store_prune_energy_snapshot_empty_metrics() -> None:
    """Pruning an empty-metrics snapshot returns it unchanged (line 420)."""

    from custom_components.termoweb.domain.energy import EnergySnapshot
    from custom_components.termoweb.domain.state import DomainStateStore

    store = DomainStateStore([NodeId(NodeType.HEATER, "1")])
    snapshot = EnergySnapshot(
        dev_id="dev", metrics={}, updated_at=1.0, ws_deadline=None
    )
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
    assert store.get_state("htr", "99") is None
    assert store.get_state("htr", "1") is None


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
    snapshot = EnergySnapshot(
        dev_id="dev", metrics={}, updated_at=1.0, ws_deadline=None
    )
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

    store = DomainStateStore(
        [
            NodeId(NodeType.ACCUMULATOR, "1"),
            NodeId(NodeType.THERMOSTAT, "2"),
            NodeId(NodeType.POWER_MONITOR, "3"),
        ]
    )
    store.replace_state("acm", "1", AccumulatorState(mode="boost"))
    assert store.get_state("acm", "1").mode == "boost"

    store.replace_state("thm", "2", ThermostatState(mode="auto"))
    assert store.get_state("thm", "2").mode == "auto"

    store.replace_state("pmo", "3", PowerMonitorState(power=200))
    assert store.get_state("pmo", "3").power == 200

    with pytest.raises(TypeError, match="does not match"):
        store.replace_state("acm", "1", HeaterState())


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


class CountingList(list):
    """Track iteration counts to detect redundant copies."""

    def __init__(self, values: Iterable[Any]) -> None:
        """Initialise the counting list with ``values``."""
        super().__init__(values)
        self.iterations = 0

    def __iter__(self) -> Iterator[Any]:
        """Iterate while incrementing the counter."""
        self.iterations += 1
        return super().__iter__()


def test_domain_state_store_applies_snapshots_and_patches() -> None:
    """DomainStateStore should persist snapshots and merge patches."""

    store = DomainStateStore(
        [NodeId(NodeType.HEATER, "1"), NodeId(NodeType.ACCUMULATOR, "2")]
    )
    store.apply_full_snapshot(
        "htr",
        "1",
        {
            "mode": "manual",
            "stemp": "21.0",
            "prog": [0, 1, 2],
            "state": "heating",
            "max_power": 1000,
            "batt_level": "4",
            "unexpected": "ignored",
        },
    )
    store.apply_full_snapshot(
        "acm",
        "2",
        {
            "mode": "auto",
            "charge_level": 75,
            "boost_active": False,
            "boost_end_datetime": dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc),
            "charging": True,
            "current_charge_per": "45.5",
            "target_charge_per": 95,
            "boost_end": {"day": 7, "minute": 15},
            "boost_end_day": 7,
            "boost_end_min": 15,
            "boost_remaining": 10,
            "extra": {"raw": "data"},
        },
    )
    settings = {
        node_id.node_type.value: {node_id.addr: state_to_dict(state)}
        for node_id, state in store.iter_states()
    }
    assert settings["htr"]["1"]["mode"] == "manual"
    assert settings["htr"]["1"]["stemp"] == "21.0"
    assert settings["htr"]["1"]["state"] == "heating"
    assert settings["htr"]["1"]["max_power"] == 1000
    assert settings["htr"]["1"]["batt_level"] == 4
    assert "unexpected" not in settings["htr"]["1"]
    assert settings["acm"]["2"]["charge_level"] == 75
    assert settings["acm"]["2"]["boost_active"] is False
    assert settings["acm"]["2"]["charging"] is True
    assert settings["acm"]["2"]["current_charge_per"] == 45.5
    assert settings["acm"]["2"]["target_charge_per"] == 95
    assert "boost_end" not in settings["acm"]["2"]
    assert settings["acm"]["2"]["boost_end_day"] == 7
    assert settings["acm"]["2"]["boost_end_min"] == 15
    assert settings["acm"]["2"]["boost_remaining"] == 10
    assert "extra" not in settings["acm"]["2"]
    assert all(
        not isinstance(value, Mapping) for value in settings["acm"]["2"].values()
    )

    store.apply_delta(
        NodeSettingsDelta(
            node_id=NodeId(NodeType.HEATER, "1"), changes={"stemp": "19.5"}
        )
    )
    patched = {
        node_id.node_type.value: {node_id.addr: state_to_dict(state)}
        for node_id, state in store.iter_states()
    }
    assert patched["htr"]["1"]["stemp"] == "19.5"
    assert patched["acm"]["2"]["charge_level"] == 75


def test_domain_state_store_applies_deltas() -> None:
    """Typed deltas should merge into the domain state store."""

    store = DomainStateStore([NodeId(NodeType.HEATER, "1")])
    store.apply_delta(
        NodeSettingsDelta(
            node_id=NodeId(NodeType.HEATER, "1"),
            changes={"mode": "auto", "stemp": "20.0"},
        )
    )
    legacy = {
        node_id.node_type.value: {node_id.addr: state_to_dict(state)}
        for node_id, state in store.iter_states()
    }
    assert legacy["htr"]["1"]["mode"] == "auto"
    assert legacy["htr"]["1"]["stemp"] == "20.0"
    assert "status" not in legacy["htr"]["1"]


def test_domain_state_store_gateway_connection_state() -> None:
    """Gateway connection state should be stored and cloned safely."""

    store = DomainStateStore([])
    state = GatewayConnectionState(
        status="healthy",
        connected=True,
        last_event_at=12.0,
        healthy_since=10.0,
        healthy_minutes=2.0,
        last_payload_at=11.0,
        last_heartbeat_at=11.5,
        payload_stale=False,
        payload_stale_after=120.0,
        idle_restart_pending=False,
    )
    store.set_gateway_connection_state(state)

    fetched = store.get_gateway_connection_state()

    assert fetched == state
    assert fetched is not state


def test_domain_state_store_strips_raw_status_and_capabilities() -> None:
    """Status payloads should be canonicalised without retaining raw blobs."""

    store = DomainStateStore([NodeId(NodeType.HEATER, "1")])
    store.apply_full_snapshot(
        "htr",
        "1",
        {
            "status": {
                "mode": "auto",
                "stemp": "19.5",
                "capabilities": {"nested": True},
                "mystery": {"raw": "data"},
            },
            "capabilities": {"ignored": True},
        },
    )
    legacy = {
        node_id.node_type.value: {node_id.addr: state_to_dict(state)}
        for node_id, state in store.iter_states()
    }
    snapshot = legacy["htr"]["1"]
    assert snapshot == {"mode": "auto", "stemp": "19.5"}
    assert "status" not in snapshot
    assert "capabilities" not in snapshot


def test_replace_state_validates_types_and_inventory() -> None:
    """Replacing state should enforce inventory and expected types."""

    store = DomainStateStore([NodeId(NodeType.HEATER, "1")])
    store.replace_state("htr", "1", None)
    assert store.get_state("htr", "1") is None

    with pytest.raises(ValueError):
        store.replace_state("htr", "2", HeaterState(mode="manual"))

    with pytest.raises(TypeError):
        store.replace_state("htr", "1", AccumulatorState())

    state = HeaterState(mode="auto")
    store.replace_state("htr", "1", state)
    assert store.get_state("htr", "1") is state


def test_store_iter_states_includes_inventory_nodes(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], Any
    ],
) -> None:
    """Domain store iteration should align with the immutable inventory."""

    nodes = {"nodes": [{"type": "htr", "addr": "1"}, {"type": "acm", "addr": "2"}]}
    node_inventory = build_node_inventory(nodes)
    inventory = inventory_builder("dev", nodes, node_inventory)
    store = DomainStateStore(
        [NodeId(NodeType.HEATER, "1"), NodeId(NodeType.ACCUMULATOR, "2")]
    )
    store.apply_full_snapshot("htr", "1", {"mode": "manual"})
    store.apply_full_snapshot("acm", "2", {"boost_minutes_delta": 15})

    states = {
        (node_id.node_type.value, node_id.addr): state_to_dict(state)
        for node_id, state in store.iter_states()
    }
    assert ("htr", "1") in states
    assert ("acm", "2") in states
    assert states[("acm", "2")]["boost_minutes_delta"] == 15
    assert states[("htr", "1")]["mode"] == "manual"


def test_state_to_dict_copies_mutable_fields_once() -> None:
    """state_to_dict should shallow-copy mutable fields without asdict churn."""

    prog = CountingList([1, 2, 3])
    raw_temp = {"raw": True}
    state = AccumulatorState(mode="auto", prog=prog)
    state.temp = raw_temp

    payload = state_to_dict(state, include_none=True)

    assert payload["mode"] == "auto"
    assert payload["prog"] == [1, 2, 3]
    assert payload["prog"] is not prog
    assert prog.iterations == 1
    assert payload["temp"] == raw_temp
    assert payload["temp"] is not raw_temp
    assert "boost_minutes_delta" in payload
    assert payload["boost_minutes_delta"] is None


def test_clone_state_returns_independent_copy() -> None:
    """clone_state should detach mutable fields."""

    prog = CountingList([1, 2])
    ptemp = [3, 4]
    raw_temp = {"raw": False}
    state = HeaterState(mode="manual", prog=prog, ptemp=ptemp, temp=raw_temp)

    clone = clone_state(state)

    assert isinstance(clone, HeaterState)
    assert clone is not state
    assert clone.mode == "manual"
    assert clone.prog is not prog
    assert clone.ptemp is not ptemp
    assert clone.temp is not raw_temp

    clone.mode = "auto"
    clone.prog.append(5)
    clone.ptemp.append(6)
    clone.temp["raw"] = True

    assert state.mode == "manual"
    assert state.prog == [1, 2]
    assert state.ptemp == [3, 4]
    assert raw_temp == {"raw": False}
    assert prog.iterations == 1


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


def test_domain_state_view_power_limit() -> None:
    """DomainStateView should expose the stored gateway power limit."""

    store = DomainStateStore([])
    view = DomainStateView("dev", store)
    assert view.get_power_limit() is None

    assert store.set_power_limit("2500") is True
    assert store.set_power_limit(2500) is False
    assert store.set_power_limit("junk") is False
    assert view.get_power_limit() == 2500

    assert DomainStateView("dev", None).get_power_limit() is None


# ---------------------------------------------------------------------------
# Power monitor state tests
# ---------------------------------------------------------------------------


def test_get_power_monitor_state_returns_none_for_non_power_monitor() -> None:
    """get_power_monitor_state returns None when the state is not PowerMonitorState."""

    store = DomainStateStore([NodeId(NodeType.HEATER, "01")])
    store.apply_full_snapshot("htr", "01", {"mode": "auto"})
    view = DomainStateView("dev", store)

    result = view.get_power_monitor_state("01")
    assert result is None


def test_get_power_monitor_state_returns_state_when_present() -> None:
    """get_power_monitor_state returns the state when it is a PowerMonitorState."""

    store = DomainStateStore([NodeId(NodeType.POWER_MONITOR, "01")])
    store.apply_full_snapshot("pmo", "01", {"power": 1500})
    view = DomainStateView("dev", store)

    result = view.get_power_monitor_state("01")
    assert isinstance(result, PowerMonitorState)
    assert result.power == 1500


def test_get_power_monitor_state_no_store_returns_none() -> None:
    """get_power_monitor_state returns None when no store is set."""

    view = DomainStateView("dev", None)
    assert view.get_power_monitor_state("01") is None


# ---------------------------------------------------------------------------
# Energy snapshot tests
# ---------------------------------------------------------------------------

from custom_components.termoweb.domain.energy import EnergyNodeMetrics, EnergySnapshot


def test_get_energy_snapshot_returns_none_without_store() -> None:
    """get_energy_snapshot returns None when no store is set."""

    view = DomainStateView("dev", None)
    assert view.get_energy_snapshot() is None


def test_get_energy_snapshot_returns_none_when_dev_id_mismatch() -> None:
    """get_energy_snapshot returns None when the snapshot dev_id does not match."""

    store = DomainStateStore([])
    snapshot = EnergySnapshot(
        dev_id="other-dev",
        metrics={},
        updated_at=1.0,
        ws_deadline=None,
    )
    store.set_energy_snapshot(snapshot)
    view = DomainStateView("dev", store)

    assert view.get_energy_snapshot() is None


def test_get_energy_snapshot_returns_snapshot_when_matched() -> None:
    """get_energy_snapshot returns the snapshot when dev_id matches."""

    store = DomainStateStore([])
    snapshot = EnergySnapshot(
        dev_id="dev",
        metrics={},
        updated_at=1.0,
        ws_deadline=None,
    )
    store.set_energy_snapshot(snapshot)
    view = DomainStateView("dev", store)

    result = view.get_energy_snapshot()
    assert result is not None
    assert result.dev_id == "dev"


# ---------------------------------------------------------------------------
# Energy metric tests
# ---------------------------------------------------------------------------


def test_get_energy_metric_returns_none_without_snapshot() -> None:
    """get_energy_metric returns None when no snapshot exists."""

    view = DomainStateView("dev", None)
    assert view.get_energy_metric(NodeType.HEATER, "01") is None


def test_get_energy_metric_returns_none_for_invalid_node_type() -> None:
    """get_energy_metric returns None for an invalid node type string."""

    store = DomainStateStore([])
    snapshot = EnergySnapshot(
        dev_id="dev",
        metrics={},
        updated_at=1.0,
        ws_deadline=None,
    )
    store.set_energy_snapshot(snapshot)
    view = DomainStateView("dev", store)

    assert view.get_energy_metric("invalid_type", "01") is None


def test_get_energy_metric_returns_none_for_empty_addr() -> None:
    """get_energy_metric returns None for an empty address."""

    store = DomainStateStore([])
    snapshot = EnergySnapshot(
        dev_id="dev",
        metrics={},
        updated_at=1.0,
        ws_deadline=None,
    )
    store.set_energy_snapshot(snapshot)
    view = DomainStateView("dev", store)

    assert view.get_energy_metric(NodeType.HEATER, "") is None


def test_get_energy_metric_returns_metric_when_present() -> None:
    """get_energy_metric returns the metric for existing node."""

    node_id = NodeId(NodeType.HEATER, "01")
    metric = EnergyNodeMetrics(
        energy_kwh=1.5,
        power_w=500.0,
        source="ws",
        ts=1000.0,
    )
    store = DomainStateStore([node_id])
    snapshot = EnergySnapshot(
        dev_id="dev",
        metrics={node_id: metric},
        updated_at=1.0,
        ws_deadline=None,
    )
    store.set_energy_snapshot(snapshot)
    view = DomainStateView("dev", store)

    result = view.get_energy_metric(NodeType.HEATER, "01")
    assert result is not None
    assert result.energy_kwh == 1.5


# ---------------------------------------------------------------------------
# Energy metrics for type
# ---------------------------------------------------------------------------


def test_get_energy_metrics_for_type_no_snapshot() -> None:
    """get_energy_metrics_for_type returns empty dict without snapshot."""

    view = DomainStateView("dev", None)
    assert view.get_energy_metrics_for_type(NodeType.HEATER) == {}


def test_get_energy_metrics_for_type_returns_filtered() -> None:
    """get_energy_metrics_for_type filters by node type."""

    htr_id = NodeId(NodeType.HEATER, "01")
    pmo_id = NodeId(NodeType.POWER_MONITOR, "02")
    htr_metric = EnergyNodeMetrics(energy_kwh=1.0, power_w=100, source="ws", ts=1.0)
    pmo_metric = EnergyNodeMetrics(energy_kwh=2.0, power_w=200, source="ws", ts=1.0)

    store = DomainStateStore([htr_id, pmo_id])
    snapshot = EnergySnapshot(
        dev_id="dev",
        metrics={htr_id: htr_metric, pmo_id: pmo_metric},
        updated_at=1.0,
        ws_deadline=None,
    )
    store.set_energy_snapshot(snapshot)
    view = DomainStateView("dev", store)

    result = view.get_energy_metrics_for_type(NodeType.HEATER)
    assert "01" in result
    assert "02" not in result
