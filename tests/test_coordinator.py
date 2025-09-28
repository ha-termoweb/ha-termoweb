from __future__ import annotations

from types import MappingProxyType
from typing import Any
from unittest.mock import AsyncMock

import pytest

from homeassistant.core import HomeAssistant

from custom_components.termoweb import coordinator as coord_module
from custom_components.termoweb.nodes import HeaterNode


def test_device_display_name_helper() -> None:
    """Helpers should trim names and fall back to the device id."""

    assert coord_module._device_display_name({"name": " Device "}, "dev") == "Device"
    assert coord_module._device_display_name({"name": ""}, "dev") == "Device dev"
    assert coord_module._device_display_name({}, "dev") == "Device dev"
    assert coord_module._device_display_name({"name": 1234}, "dev") == "1234"

    proxy_device: MappingProxyType[str, str] = MappingProxyType({"name": " Proxy "})
    assert coord_module._device_display_name(proxy_device, "dev") == "Proxy"


def test_ensure_heater_section_helper() -> None:
    """The helper must reuse existing sections or insert defaults."""

    nodes_by_type: dict[str, dict[str, Any]] = {
        "htr": {"addrs": ["1"], "settings": {"1": {}}}
    }
    existing = coord_module._ensure_heater_section(nodes_by_type, lambda: {})
    assert existing is nodes_by_type["htr"]

    proxy_nodes = MappingProxyType({"addrs": ("2",), "settings": {"2": {}}})
    nodes_by_type = {"htr": proxy_nodes}  # type: ignore[assignment]
    converted = coord_module._ensure_heater_section(nodes_by_type, lambda: {})
    assert converted == {"addrs": ["2"], "settings": {"2": {}}}
    assert nodes_by_type["htr"] == converted

    nodes_by_type = {}
    created = coord_module._ensure_heater_section(
        nodes_by_type,
        lambda: MappingProxyType(
            {"addrs": ("A",), "settings": {"A": {"mode": "auto"}}}
        ),
    )
    assert created == {"addrs": ["A"], "settings": {"A": {"mode": "auto"}}}
    assert nodes_by_type["htr"] == created


@pytest.mark.asyncio
async def test_refresh_skips_pending_settings_merge() -> None:
    """Heater refresh should defer merging stale payloads while pending."""

    hass = HomeAssistant()
    client = AsyncMock()
    nodes = {"nodes": [{"type": "htr", "addr": "1", "name": "Heater"}]}
    inventory = [HeaterNode(name="Heater", addr="1")]
    coordinator = coord_module.StateCoordinator(
        hass,
        client=client,
        base_interval=30,
        dev_id="dev",
        device={"name": "Device"},
        nodes=nodes,
        node_inventory=inventory,
    )
    initial = {
        "dev": {
            "dev_id": "dev",
            "name": "Device",
            "raw": {"name": "Device"},
            "connected": True,
            "nodes": nodes,
            "nodes_by_type": {
                "htr": {
                    "addrs": ["1"],
                    "settings": {"1": {"mode": "manual", "stemp": "21.0"}},
                }
            },
            "htr": {
                "addrs": ["1"],
                "settings": {"1": {"mode": "manual", "stemp": "21.0"}},
            },
        }
    }
    coordinator.data = initial

    client.get_node_settings = AsyncMock(return_value={"mode": "auto", "stemp": "20.0"})
    coordinator.register_pending_setting(
        "htr", "1", mode="manual", stemp=21.0, ttl=60.0
    )

    await coordinator.async_refresh_heater(("htr", "1"))

    settings = coordinator.data["dev"]["nodes_by_type"]["htr"]["settings"]["1"]
    assert settings == {"mode": "manual", "stemp": "21.0"}
    assert ("htr", "1") in coordinator._pending_settings

    client.get_node_settings.return_value = {"mode": "manual", "stemp": "21.0"}

    await coordinator.async_refresh_heater(("htr", "1"))

    assert ("htr", "1") not in coordinator._pending_settings
    assert client.get_node_settings.await_count == 2


@pytest.mark.asyncio
async def test_poll_skips_pending_settings_merge() -> None:
    """Polling should defer merges until pending settings match."""

    hass = HomeAssistant()
    client = AsyncMock()
    nodes = {"nodes": [{"type": "htr", "addr": "1", "name": "Heater"}]}
    inventory = [HeaterNode(name="Heater", addr="1")]
    coordinator = coord_module.StateCoordinator(
        hass,
        client=client,
        base_interval=30,
        dev_id="dev",
        device={"name": "Device"},
        nodes=nodes,
        node_inventory=inventory,
    )
    initial = {
        "dev": {
            "dev_id": "dev",
            "name": "Device",
            "raw": {"name": "Device"},
            "connected": True,
            "nodes": nodes,
            "nodes_by_type": {
                "htr": {
                    "addrs": ["1"],
                    "settings": {"1": {"mode": "manual", "stemp": "21.0"}},
                }
            },
            "htr": {
                "addrs": ["1"],
                "settings": {"1": {"mode": "manual", "stemp": "21.0"}},
            },
        }
    }
    coordinator.data = initial

    client.get_node_settings = AsyncMock(return_value={"mode": "auto", "stemp": "20.0"})
    coordinator.register_pending_setting(
        "htr", "1", mode="manual", stemp=21.0, ttl=60.0
    )

    result = await coordinator._async_update_data()

    settings = result["dev"]["nodes_by_type"]["htr"]["settings"]["1"]
    assert settings == {"mode": "manual", "stemp": "21.0"}
    assert ("htr", "1") in coordinator._pending_settings

    coordinator.data = result
    client.get_node_settings.return_value = {"mode": "manual", "stemp": "21.0"}

    result_second = await coordinator._async_update_data()

    assert ("htr", "1") not in coordinator._pending_settings
    assert client.get_node_settings.await_count == 2
    settings_second = result_second["dev"]["nodes_by_type"]["htr"]["settings"]["1"]
    assert settings_second == {"mode": "manual", "stemp": "21.0"}
