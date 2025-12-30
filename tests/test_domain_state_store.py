from __future__ import annotations

import datetime as dt
from typing import Any, Callable, Mapping

from custom_components.termoweb.domain import (
    DomainStateStore,
    NodeId,
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
        {"mode": "manual", "stemp": "21.0", "prog": [0, 1, 2]},
    )
    store.apply_full_snapshot(
        "acm",
        "2",
        {
            "mode": "auto",
            "charge_level": 75,
            "boost_active": False,
            "boost_end_datetime": dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc),
        },
    )
    settings = store.legacy_view()
    assert settings["htr"]["1"]["mode"] == "manual"
    assert settings["htr"]["1"]["stemp"] == "21.0"
    assert settings["acm"]["2"]["charge_level"] == 75
    assert settings["acm"]["2"]["boost_active"] is False

    store.apply_patch("htr", "1", {"stemp": "19.5"})
    patched = store.legacy_view()
    assert patched["htr"]["1"]["stemp"] == "19.5"
    assert patched["acm"]["2"]["charge_level"] == 75


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
        device_raw={"dev_id": "dev"},
    )
    record = legacy["dev"]
    assert record["dev_id"] == "dev"
    assert record["name"] == "Device dev"
    assert record["inventory"] is inventory
    assert record["settings"]["acm"]["2"]["boost_minutes_delta"] == 15
    assert record["settings"]["htr"]["1"]["mode"] == "manual"
