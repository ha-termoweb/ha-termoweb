from __future__ import annotations

import datetime as dt
from typing import Any, Callable, Iterable, Iterator, Mapping

import pytest

from custom_components.termoweb.domain import (
    AccumulatorState,
    clone_state,
    DomainStateStore,
    GatewayConnectionState,
    HeaterState,
    NodeId,
    NodeSettingsDelta,
    NodeStatusDelta,
    NodeType,
    state_to_dict,
)
from custom_components.termoweb.inventory import build_node_inventory


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

    store.apply_patch("htr", "1", {"stemp": "19.5"})
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
    store.apply_delta(
        NodeStatusDelta(
            node_id=NodeId(NodeType.HEATER, "1"),
            status={"stemp": "20.5", "online": True},
        )
    )
    legacy = {
        node_id.node_type.value: {node_id.addr: state_to_dict(state)}
        for node_id, state in store.iter_states()
    }
    assert legacy["htr"]["1"]["mode"] == "auto"
    assert legacy["htr"]["1"]["stemp"] == "20.5"
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
