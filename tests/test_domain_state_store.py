from __future__ import annotations

import datetime as dt
from typing import Any, Callable, Mapping

import pytest

from custom_components.termoweb.domain import (
    AccumulatorState,
    DomainStateStore,
    HeaterState,
    NodeId,
    NodeSettingsDelta,
    NodeStatusDelta,
    NodeType,
    ThermostatState,
    build_state_from_payload,
    state_to_dict,
)
from custom_components.termoweb.inventory import build_node_inventory


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
            "boost": True,
            "boost_active": False,
            "boost_end_datetime": dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc),
            "charging": True,
            "current_charge_per": "45.5",
            "target_charge_per": 95,
            "boost_end": {"day": 7, "minute": 15},
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
    assert settings["acm"]["2"]["boost"] is True
    assert settings["acm"]["2"]["boost_end"] == {"day": 7, "minute": 15}
    assert settings["acm"]["2"]["boost_remaining"] == 10
    assert "extra" not in settings["acm"]["2"]

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


def test_build_state_from_payload_ignores_unknown_fields() -> None:
    """Domain state instances should only retain explicit fields."""

    state = build_state_from_payload(
        "thm",
        {
            "mode": "auto",
            "state": "idle",
            "batt_level": "5",
            "mystery": {"raw": "payload"},
        },
    )
    assert isinstance(state, ThermostatState)
    legacy = state_to_dict(state)
    assert legacy["mode"] == "auto"
    assert legacy["state"] == "idle"
    assert legacy["batt_level"] == 5
    assert "capabilities" not in legacy
    assert "mystery" not in legacy
    assert not hasattr(state, "status")


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
