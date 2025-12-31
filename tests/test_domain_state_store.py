from __future__ import annotations

import datetime as dt
from typing import Any, Callable, Mapping

from custom_components.termoweb.domain import (
    ThermostatState,
    build_state_from_payload,
    DomainStateStore,
    NodeId,
    NodeSettingsDelta,
    NodeStatusDelta,
    NodeType,
    store_to_legacy_coordinator_data,
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
    settings = store.legacy_view()
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
    patched = store.legacy_view()
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
    legacy = store.legacy_view()
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
    legacy = state.to_legacy()
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
    legacy = store.legacy_view()
    snapshot = legacy["htr"]["1"]
    assert snapshot == {"mode": "auto", "stemp": "19.5"}
    assert "status" not in snapshot
    assert "capabilities" not in snapshot


def test_store_to_legacy_coordinator_data_matches_schema(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], Any
    ],
) -> None:
    """Legacy adapter should expose coordinator data with expected shape."""

    nodes = {"nodes": [{"type": "htr", "addr": "1"}, {"type": "acm", "addr": "2"}]}
    node_inventory = build_node_inventory(nodes)
    inventory = inventory_builder("dev", nodes, node_inventory)
    store = DomainStateStore(
        [NodeId(NodeType.HEATER, "1"), NodeId(NodeType.ACCUMULATOR, "2")]
    )
    store.apply_full_snapshot("htr", "1", {"mode": "manual"})
    store.apply_full_snapshot("acm", "2", {"boost_minutes_delta": 15})

    legacy = store_to_legacy_coordinator_data(
        "dev",
        store,
        inventory,
        device_name="Device dev",
        device_details={"dev_id": "dev", "model": "Controller"},
    )
    record = legacy["dev"]
    assert record["dev_id"] == "dev"
    assert record["name"] == "Device dev"
    assert record["model"] == "Controller"
    assert record["inventory"] is inventory
    assert record["settings"]["acm"]["2"]["boost_minutes_delta"] == 15
    assert record["settings"]["htr"]["1"]["mode"] == "manual"
