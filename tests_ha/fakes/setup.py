"""Build a real ``StateCoordinator`` for tests that need gateway metadata."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any
from unittest.mock import AsyncMock

from homeassistant.core import HomeAssistant

from custom_components.termoweb.const import BRAND_TERMOWEB
from custom_components.termoweb.coordinator import StateCoordinator
from custom_components.termoweb.inventory import Inventory, build_node_inventory


def state_coordinator(
    hass: HomeAssistant,
    dev_id: str = "dev",
    device: Mapping[str, Any] | None = None,
    nodes: Iterable[Mapping[str, Any]] = (),
    *,
    brand: str = BRAND_TERMOWEB,
) -> StateCoordinator:
    """Return a real coordinator over ``nodes``; its REST client is a mock."""
    inventory = Inventory(dev_id, build_node_inventory({"nodes": list(nodes)}))
    return StateCoordinator(
        hass, AsyncMock(), 30, dev_id, device, inventory, brand=brand
    )
