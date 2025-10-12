"""Tests for websocket heater address helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterable, Mapping, MutableMapping
from unittest.mock import MagicMock

from homeassistant.core import HomeAssistant

from custom_components.termoweb.backend.ducaheat_ws import DucaheatWSClient, DOMAIN
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


def _make_client() -> DucaheatWSClient:
    """Instantiate a websocket client with stub dependencies."""

    hass = HomeAssistant()
    hass.data.setdefault(DOMAIN, {})["entry"] = {}
    coordinator = DummyCoordinator()
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


def test_apply_heater_addresses_updates_state() -> None:
    """Heater address normalisation should update entry and coordinator state."""

    client = _make_client()
    hass = client.hass
    energy_coordinator = SimpleNamespace(update_addresses=MagicMock())
    hass.data[DOMAIN]["entry"]["energy_coordinator"] = energy_coordinator

    raw_nodes = {
        "nodes": [
            {"type": "htr", "addr": "1"},
            {"type": "acm", "addr": "2"},
            {"type": "pmo", "addr": "A1"},
        ]
    }
    inventory = Inventory("device", raw_nodes, build_node_inventory(raw_nodes))

    normalized_map: Mapping[Any, Iterable[Any]] = {
        "htr": [" 1 ", "01"],
        "heater": ["003"],
        "acm": ("2", "02", "2"),
        "pmo": ["A1", "a1", "B2"],
        "thm": ["9"],
    }

    cleaned = client._apply_heater_addresses(normalized_map, inventory=inventory)

    heater_map, heater_aliases = inventory.heater_sample_address_map
    power_map, power_aliases = inventory.power_monitor_sample_address_map

    assert cleaned["htr"] == heater_map["htr"]
    assert "003" not in cleaned["htr"]
    assert cleaned["acm"] == heater_map["acm"]
    assert cleaned["pmo"] == power_map.get("pmo", [])
    assert "thm" not in cleaned

    assert hass.data[DOMAIN]["entry"]["inventory"] is inventory
    assert client._inventory is inventory
    energy_coordinator.update_addresses.assert_called_once_with(inventory)

    sample_aliases = hass.data[DOMAIN]["entry"].get("sample_aliases")
    assert sample_aliases is not None
    for alias_map in (heater_aliases, power_aliases):
        for alias, target in alias_map.items():
            assert sample_aliases.get(alias) == target


def test_dispatch_nodes_uses_inventory_addresses() -> None:
    """Inventory address caches should populate dispatch payloads."""

    client = _make_client()
    client._dispatcher = MagicMock()

    inventory_payload = _build_inventory_payload()
    inventory = Inventory(
        "device",
        inventory_payload,
        build_node_inventory(inventory_payload),
    )

    record = client.hass.data[DOMAIN][client.entry_id]
    record["inventory"] = inventory
    client._inventory = inventory

    client._dispatch_nodes({"htr": {"settings": {"1": {"target_temp": 21}}}})

    assert client._dispatcher.call_count == 1
    dispatched = client._dispatcher.call_args[0][2]
    assert dispatched["addr_map"]["htr"] == ["1"]
    assert dispatched["addresses_by_type"]["htr"] == ["1"]
    sample_aliases = client.hass.data[DOMAIN][client.entry_id].get("sample_aliases")
    assert sample_aliases is not None
    assert sample_aliases.get("htr") == "htr"

