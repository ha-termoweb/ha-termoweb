"""Tests for coordinator inventory address integration."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from homeassistant.core import HomeAssistant

from custom_components.termoweb import inventory as inventory_module
from custom_components.termoweb.coordinator import StateCoordinator
from custom_components.termoweb.inventory import Inventory, build_node_inventory


def _build_inventory() -> Inventory:
    """Return inventory metadata for a heater-only device."""

    raw_nodes = {"nodes": [{"type": "htr", "addr": "1"}]}
    inventory_nodes = build_node_inventory(raw_nodes)
    return Inventory("device", raw_nodes, inventory_nodes)


@pytest.mark.asyncio
async def test_async_update_data_consults_inventory_addresses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coordinator polling should obtain addresses from the inventory."""

    hass = HomeAssistant()
    client = AsyncMock()
    client.get_node_settings = AsyncMock(return_value={})
    inventory = _build_inventory()
    coordinator = StateCoordinator(
        hass,
        client=client,
        base_interval=30,
        dev_id="device",
        device={},
        nodes=inventory.payload,
        inventory=inventory,
    )

    original_property = inventory_module.Inventory.addresses_by_type
    call_count = 0

    def fake_addresses(self: Inventory) -> dict[str, list[str]]:
        nonlocal call_count
        call_count += 1
        return original_property.fget(self)

    monkeypatch.setattr(
        inventory_module.Inventory,
        "addresses_by_type",
        property(fake_addresses),
    )

    result = await coordinator._async_update_data()

    assert call_count == 1
    assert result["device"]["addresses_by_type"]["htr"] == ["1"]


@pytest.mark.asyncio
async def test_async_refresh_heater_consults_inventory_addresses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Manual heater refresh should read addresses directly from inventory."""

    hass = HomeAssistant()
    client = AsyncMock()
    client.get_node_settings = AsyncMock(return_value={})
    inventory = _build_inventory()
    coordinator = StateCoordinator(
        hass,
        client=client,
        base_interval=30,
        dev_id="device",
        device={},
        nodes=inventory.payload,
        inventory=inventory,
    )

    original_property = inventory_module.Inventory.addresses_by_type
    call_count = 0

    def fake_addresses(self: Inventory) -> dict[str, list[str]]:
        nonlocal call_count
        call_count += 1
        return original_property.fget(self)

    monkeypatch.setattr(
        inventory_module.Inventory,
        "addresses_by_type",
        property(fake_addresses),
    )

    await coordinator.async_refresh_heater(("htr", "1"))

    assert call_count == 1
    assert coordinator.data["device"]["addresses_by_type"]["htr"] == ["1"]
