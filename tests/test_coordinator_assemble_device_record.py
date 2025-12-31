"""Tests for ``StateCoordinator._assemble_device_record``."""

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from custom_components.termoweb.domain.ids import NodeId, NodeType
from custom_components.termoweb.domain.legacy_view import (
    store_to_legacy_coordinator_data,
)
from custom_components.termoweb.domain.state import DomainStateStore
from custom_components.termoweb.inventory import AccumulatorNode, HeaterNode


def test_assemble_device_record_uses_inventory_maps(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None],
        Any,
    ],
) -> None:
    """Device records should expose inventory-derived metadata."""

    inventory = inventory_builder(
        "dev",
        {},
        [
            HeaterNode(name="Hall", addr="01"),
            AccumulatorNode(name="Store", addr="07"),
        ],
    )

    settings = {
        "htr": {"01": {"stemp": 21}},
        "acm": {"07": {"mode": "auto"}},
    }

    store = DomainStateStore(
        [NodeId(NodeType.HEATER, "01"), NodeId(NodeType.ACCUMULATOR, "07")]
    )
    for node_type, bucket in settings.items():
        if not isinstance(bucket, Mapping):
            continue
        for addr, payload in bucket.items():
            store.apply_full_snapshot(node_type, addr, payload)

    record = store_to_legacy_coordinator_data(
        "dev",
        store,
        inventory,
        device_name="Device dev",
        device_details={"name": "Device dev"},
    )

    device = record["dev"]
    assert device["inventory"] is inventory
    assert "addresses_by_type" not in device
    assert "heater_address_map" not in device
    assert "power_monitor_address_map" not in device
    assert device["settings"] == settings
