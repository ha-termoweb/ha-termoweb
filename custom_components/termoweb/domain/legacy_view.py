"""Adapters to expose domain state in legacy coordinator shapes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from custom_components.termoweb.inventory import Inventory

from .state import DomainStateStore


def store_to_legacy_coordinator_data(
    dev_id: str,
    store: DomainStateStore,
    inventory: Inventory,
    *,
    device_name: str,
    device_details: Mapping[str, Any] | None = None,
    include_nodes_by_type: bool = True,
) -> dict[str, dict[str, Any]]:
    """Return a coordinator data mapping using the legacy dict schema."""

    model: str | None = None
    if isinstance(device_details, Mapping):
        model_value = device_details.get("model")
        if model_value not in (None, ""):
            model = str(model_value)

    settings = store.legacy_view()
    device_record: dict[str, Any] = {
        "dev_id": dev_id,
        "name": device_name,
        "model": model,
        "connected": True,
        "inventory": inventory,
        "settings": settings,
    }
    nodes_by_type: dict[str, Any] = {}
    for node_type, bucket in settings.items():
        device_record[node_type] = {"settings": dict(bucket)}
        nodes_by_type[node_type] = {
            "settings": dict(bucket),
            "addrs": list(bucket),
        }
    if include_nodes_by_type:
        device_record["nodes_by_type"] = nodes_by_type

    return {dev_id: device_record}
