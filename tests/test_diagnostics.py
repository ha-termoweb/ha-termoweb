"""Tests for the TermoWeb diagnostics helper."""

from __future__ import annotations

import asyncio
import platform
import sys
import types
from typing import Any

import pytest

from conftest import _install_stubs

_install_stubs()

diagnostics_stub = types.ModuleType("diagnostics")


async def _async_passthrough(data: Any, _keys: set[str]) -> Any:
    return data


diagnostics_stub.async_redact_data = _async_passthrough
components_pkg = sys.modules.setdefault(
    "homeassistant.components", types.ModuleType("homeassistant.components")
)
setattr(components_pkg, "diagnostics", diagnostics_stub)
sys.modules["homeassistant.components.diagnostics"] = diagnostics_stub

from custom_components.termoweb.const import BRAND_DUCAHEAT, CONF_BRAND, DOMAIN
from custom_components.termoweb.diagnostics import async_get_config_entry_diagnostics
from custom_components.termoweb.installation import InstallationSnapshot
from custom_components.termoweb.inventory import Node
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant


def _flatten(data: Any) -> list[str]:
    if isinstance(data, dict):
        keys: list[str] = list(data)
        for value in data.values():
            keys.extend(_flatten(value))
        return keys
    if isinstance(data, list):
        keys: list[str] = []
        for item in data:
            keys.extend(_flatten(item))
        return keys
    return []


def test_diagnostics_with_cached_inventory() -> None:
    """Diagnostics return cached inventory and redact sensitive keys."""

    hass = HomeAssistant()
    hass.version = "2025.5.0"
    hass.config.time_zone = "Europe/London"

    entry = ConfigEntry(
        "entry-one",
        data={CONF_BRAND: BRAND_DUCAHEAT},
    )

    nodes = [
        Node(name="Heater One", addr="1", node_type="htr"),
        Node(name="Monitor", addr="2", node_type="pmo"),
    ]

    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = {
        "version": "1.2.3",
        "brand": BRAND_DUCAHEAT,
        "node_inventory": list(nodes),
        "dev_id": "secret-dev",
        "username": "user@example.com",
    }

    diagnostics = asyncio.run(async_get_config_entry_diagnostics(hass, entry))

    assert diagnostics["integration"]["version"] == "1.2.3"
    assert diagnostics["integration"]["brand"] == "Ducaheat"
    assert diagnostics["home_assistant"]["version"] == "2025.5.0"
    assert diagnostics["home_assistant"]["python_version"] == platform.python_version()
    assert diagnostics["home_assistant"]["time_zone"] == "Europe/London"
    assert diagnostics["installation"]["node_inventory"] == [
        {"name": "Heater One", "addr": "1", "type": "htr"},
        {"name": "Monitor", "addr": "2", "type": "pmo"},
    ]

    flattened = _flatten(diagnostics)
    assert "dev_id" not in flattened
    assert "username" not in flattened


def test_diagnostics_uses_snapshot_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Diagnostics fall back to snapshot inventory and helper version."""

    hass = HomeAssistant()
    hass.version = "2025.5.1"

    entry = ConfigEntry(
        "entry-two",
        data={CONF_BRAND: "termoweb"},
    )

    raw_nodes = [
        {"name": "Heater Two", "addr": "5", "type": "htr"},
    ]
    snapshot = InstallationSnapshot(dev_id="dev-123", raw_nodes=raw_nodes)

    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = {
        "snapshot": snapshot,
    }

    diagnostics = asyncio.run(async_get_config_entry_diagnostics(hass, entry))

    assert hass.integration_requests == [DOMAIN]
    assert diagnostics["integration"]["version"] == "test-version"
    assert diagnostics["integration"]["brand"] == "TermoWeb"
    assert diagnostics["installation"]["node_inventory"] == [
        {"name": "Heater Two", "addr": "5", "type": "htr"},
    ]
    assert "dev_id" not in _flatten(diagnostics)
    assert "time_zone" not in diagnostics["home_assistant"]


def test_diagnostics_without_record(monkeypatch: pytest.MonkeyPatch) -> None:
    """Diagnostics handle missing registry records with safe defaults."""

    hass = HomeAssistant()
    entry = ConfigEntry("entry-three", data={})

    diagnostics = asyncio.run(async_get_config_entry_diagnostics(hass, entry))

    assert hass.integration_requests == [DOMAIN]
    assert diagnostics["integration"]["version"] == "test-version"
    assert diagnostics["integration"]["brand"] == "TermoWeb"
    assert diagnostics["home_assistant"]["version"] == "unknown"
    assert "time_zone" not in diagnostics["home_assistant"]
    assert diagnostics["installation"]["node_inventory"] == []
