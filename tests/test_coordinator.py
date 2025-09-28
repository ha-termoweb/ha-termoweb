from __future__ import annotations

from types import MappingProxyType
from typing import Any

from custom_components.termoweb import coordinator as coord_module


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
