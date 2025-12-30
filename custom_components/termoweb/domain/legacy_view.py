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
    device_raw: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Return a coordinator data mapping using the legacy dict schema."""

    return {
        dev_id: {
            "dev_id": dev_id,
            "name": device_name,
            "raw": device_raw or {},
            "connected": True,
            "inventory": inventory,
            "settings": store.legacy_view(),
        }
    }
