"""Tests for websocket heater address helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterable, Mapping, MutableMapping
from unittest.mock import MagicMock

from homeassistant.core import HomeAssistant

from custom_components.termoweb.backend.ducaheat_ws import DucaheatWSClient, DOMAIN
from custom_components.termoweb.inventory import (
    Inventory,
    build_node_inventory,
    normalize_node_addr,
)


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


def _expected_addresses(values: Iterable[Any]) -> list[str]:
    """Return normalised, deduplicated addresses for ``values``."""

    seen: set[str] = set()
    result: list[str] = []
    for candidate in values:
        addr = normalize_node_addr(candidate)
        if not addr or addr in seen:
            continue
        seen.add(addr)
        result.append(addr)
    return result


def test_ensure_type_bucket_registers_defaults() -> None:
    """Buckets should expose default sections and update the device map."""

    client = _make_client()
    nodes_by_type: dict[str, dict[str, Any]] = {}
    dev_map: dict[str, Any] = {"addresses_by_type": {}, "settings": {}}

    bucket = client._ensure_type_bucket(nodes_by_type, "htr", dev_map=dev_map)

    assert nodes_by_type["htr"] is bucket
    for section in ("settings", "samples", "status", "advanced"):
        assert isinstance(bucket[section], dict)
    assert bucket["addrs"] == []

    assert dev_map["nodes_by_type"]["htr"] is bucket
    assert dev_map["addresses_by_type"]["htr"] == []
    assert dev_map["settings"]["htr"] == {}


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

    assert cleaned["htr"] == _expected_addresses([" 1 ", "01"])
    assert normalize_node_addr("003") not in cleaned["htr"]
    assert cleaned["acm"] == _expected_addresses(["2", "02"])
    assert cleaned["pmo"] == inventory.power_monitor_address_map[0]["pmo"]
    assert "thm" not in cleaned

    assert hass.data[DOMAIN]["entry"]["inventory"] is inventory
    assert client._inventory is inventory
    energy_coordinator.update_addresses.assert_called_once_with(cleaned)

