"""Tests for websocket heater address helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping, MutableMapping
from unittest.mock import MagicMock

from homeassistant.core import HomeAssistant

from conftest import build_entry_runtime
from custom_components.termoweb.backend.ducaheat_ws import DucaheatWSClient
from custom_components.termoweb.inventory import Inventory, build_node_inventory


class DummyREST:
    """Provide the minimal REST client contract for the websocket client."""

    def __init__(self) -> None:
        self._session = SimpleNamespace()

    async def authed_headers(self) -> Mapping[str, str]:
        """Return headers containing an access token."""

        return {"Authorization": "Bearer token"}


class DummyCoordinator:
    """Expose coordinator storage accessed by the websocket client."""

    def __init__(self) -> None:
        self.data: MutableMapping[str, Any] = {"device": {}}
        self.update_nodes = MagicMock()


def _make_client(*, inventory: Inventory | None = None) -> DucaheatWSClient:
    """Instantiate a websocket client with stub dependencies."""

    hass = HomeAssistant()
    coordinator = DummyCoordinator()
    build_entry_runtime(
        hass=hass,
        entry_id="entry",
        dev_id="device",
        coordinator=coordinator,
        inventory=inventory,
    )
    client = DucaheatWSClient(
        hass,
        entry_id="entry",
        dev_id="device",
        api_client=DummyREST(),
        coordinator=coordinator,
        session=SimpleNamespace(),  # type: ignore[arg-type]
    )
    return client


def _build_inventory_payload() -> dict[str, Any]:
    """Return a representative raw node payload."""

    return {
        "nodes": [
            {"type": "htr", "addr": "1"},
            {"type": "acm", "addr": "02"},
            {"type": "pmo", "addr": "A1"},
        ]
    }


def test_dispatch_nodes_includes_inventory_metadata() -> None:
    """Inventory metadata should populate dispatch payloads."""

    inventory_payload = _build_inventory_payload()
    inventory = Inventory(
        "device",
        build_node_inventory(inventory_payload),
    )
    client = _make_client(inventory=inventory)
    client._dispatcher = MagicMock()
    client._inventory = inventory

    client._dispatch_nodes({"htr": {"settings": {"1": {"target_temp": 21}}}})

    assert client._dispatcher.call_count == 1
    dispatched = client._dispatcher.call_args[0][2]
    assert dispatched["inventory"] is inventory
