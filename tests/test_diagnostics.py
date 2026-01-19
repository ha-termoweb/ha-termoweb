"""Tests for the TermoWeb diagnostics helper."""

from __future__ import annotations

import asyncio
import logging
import platform
import sys
import types
import time
from typing import Any, Callable

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
from custom_components.termoweb.backend.ws_health import WsHealthTracker
from custom_components.termoweb.diagnostics import async_get_config_entry_diagnostics
from custom_components.termoweb.inventory import (
    Inventory,
    InventorySnapshot,
    build_node_inventory,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant


@pytest.fixture
def diagnostics_record(
    inventory_builder: Callable[[str, dict[str, Any], list[Any]], Inventory],
    runtime_factory: Callable[..., Any],
) -> Callable[..., tuple[Any, Inventory]]:
    """Return helper to build diagnostics records with cached inventory."""

    def _factory(
        nodes: list[dict[str, Any]],
        *,
        dev_id: str,
        entry_id: str = "entry-one",
        version: str | None = None,
        brand: str | None = None,
        **extra: Any,
    ) -> tuple[Any, Inventory]:
        payload = {"nodes": list(nodes)}
        inventory = inventory_builder(
            dev_id,
            payload,
            build_node_inventory(payload),
        )
        runtime = runtime_factory(
            entry_id=entry_id,
            dev_id=dev_id,
            inventory=inventory,
            version=version or "0.0.0",
            brand=brand or "termoweb",
        )
        if version is None:
            runtime.version = None  # type: ignore[assignment]
        if extra:
            for key, value in extra.items():
                runtime[key] = value
        return runtime, inventory

    return _factory


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


def test_diagnostics_with_cached_inventory(
    caplog: pytest.LogCaptureFixture,
    diagnostics_record: Callable[..., tuple[Any, Inventory]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Diagnostics return cached inventory and redact sensitive keys."""

    hass = HomeAssistant()
    hass.version = "2025.5.0"
    hass.config.time_zone = "Europe/London"

    entry = ConfigEntry(
        "entry-one",
        data={CONF_BRAND: BRAND_DUCAHEAT},
    )

    record, inventory = diagnostics_record(
        [
            {"name": "Heater One", "addr": "1", "type": "htr"},
            {"name": "Monitor", "addr": "2", "type": "pmo"},
        ],
        dev_id="secret-dev",
        entry_id=entry.entry_id,
        version="1.2.3",
        brand=BRAND_DUCAHEAT,
        username="user@example.com",
    )

    record.config_entry = entry
    record.version = "1.2.3"
    record.brand = BRAND_DUCAHEAT
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = record

    captured_snapshots: list[InventorySnapshot] = []

    original_snapshot = Inventory.snapshot

    def _capture_snapshot(self: Inventory) -> InventorySnapshot:
        snapshot = original_snapshot(self)
        captured_snapshots.append(snapshot)
        return snapshot

    monkeypatch.setattr(Inventory, "snapshot", _capture_snapshot, raising=True)

    with caplog.at_level(logging.DEBUG):
        diagnostics = asyncio.run(async_get_config_entry_diagnostics(hass, entry))

    assert diagnostics["integration"]["version"] == "1.2.3"
    assert diagnostics["integration"]["brand"] == "Ducaheat"
    assert diagnostics["home_assistant"]["version"] == "2025.5.0"
    assert diagnostics["home_assistant"]["python_version"] == platform.python_version()
    assert diagnostics["home_assistant"]["time_zone"] == "Europe/London"
    expected_inventory = [
        {"name": "Heater One", "addr": "1", "type": "htr"},
        {"name": "Monitor", "addr": "2", "type": "pmo"},
    ]
    assert diagnostics["installation"]["node_inventory"] == expected_inventory

    assert len(captured_snapshots) == 1
    assert list(captured_snapshots[0].node_inventory) == expected_inventory

    flattened = _flatten(diagnostics)
    assert "dev_id" not in flattened
    assert "username" not in flattened

    assert "Diagnostics inventory cache for entry-one: raw=2, filtered=2" in caplog.text
    assert not any(record.levelno >= logging.ERROR for record in caplog.records)


def test_diagnostics_include_websocket_clients(
    caplog: pytest.LogCaptureFixture,
    diagnostics_record: Callable[..., tuple[Any, Inventory]],
) -> None:
    """Diagnostics should expose websocket state and health snapshots."""

    hass = HomeAssistant()
    entry = ConfigEntry(
        "entry-ws",
        data={CONF_BRAND: BRAND_DUCAHEAT},
    )

    dev_id = "abcdef123456"
    record, inventory = diagnostics_record(
        [
            {"name": "Heater One", "addr": "1", "type": "htr"},
        ],
        dev_id=dev_id,
        entry_id=entry.entry_id,
    )

    ws_state = {"status": "healthy", "last_payload_at": 10.0}
    tracker = WsHealthTracker(dev_id=dev_id)
    now = time.time()
    tracker.update_status("healthy", healthy_since=now - 60)
    tracker.mark_payload(timestamp=now, stale_after=1_000_000.0)

    record.ws_state = {dev_id: ws_state}
    record.ws_trackers = {dev_id: tracker}

    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = record

    with caplog.at_level(logging.DEBUG):
        diagnostics = asyncio.run(async_get_config_entry_diagnostics(hass, entry))

    websocket = diagnostics.get("websocket")
    assert websocket is not None
    clients = websocket.get("clients") if isinstance(websocket, dict) else None
    assert isinstance(clients, list)
    assert len(clients) == 1
    client = clients[0]
    assert client["device"] == "abcdef...3456"
    assert client["state"] == ws_state
    assert "health" in client and client["health"]["status"] == "healthy"

    assert "Diagnostics inventory cache for entry-ws: raw=1, filtered=1" in caplog.text
    assert not any(record.levelno >= logging.ERROR for record in caplog.records)


def test_diagnostics_with_inventory_missing_version(
    caplog: pytest.LogCaptureFixture,
    diagnostics_record: Callable[..., tuple[Any, Inventory]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Diagnostics rely on stored inventory and fetch helper version."""

    hass = HomeAssistant()
    hass.version = "2025.5.1"

    entry = ConfigEntry(
        "entry-two",
        data={CONF_BRAND: "termoweb"},
    )

    record, _ = diagnostics_record(
        [
            {"name": "Heater Two", "addr": "5", "type": "htr"},
        ],
        dev_id="dev-two",
        entry_id=entry.entry_id,
    )

    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = record

    captured_snapshots: list[InventorySnapshot] = []
    original_snapshot = Inventory.snapshot

    def _capture_snapshot(self: Inventory) -> InventorySnapshot:
        snapshot = original_snapshot(self)
        captured_snapshots.append(snapshot)
        return snapshot

    monkeypatch.setattr(Inventory, "snapshot", _capture_snapshot, raising=True)

    with caplog.at_level(logging.DEBUG):
        diagnostics = asyncio.run(async_get_config_entry_diagnostics(hass, entry))

    assert hass.integration_requests == [DOMAIN]
    assert diagnostics["integration"]["version"] == "test-version"
    assert diagnostics["integration"]["brand"] == "TermoWeb"
    expected_inventory = [
        {"name": "Heater Two", "addr": "5", "type": "htr"},
    ]
    assert diagnostics["installation"]["node_inventory"] == expected_inventory
    assert len(captured_snapshots) == 1
    assert list(captured_snapshots[0].node_inventory) == expected_inventory
    assert "dev_id" not in _flatten(diagnostics)
    assert "time_zone" not in diagnostics["home_assistant"]

    assert "Diagnostics inventory cache for entry-two: raw=1, filtered=1" in caplog.text
    assert not any(record.levelno >= logging.ERROR for record in caplog.records)
