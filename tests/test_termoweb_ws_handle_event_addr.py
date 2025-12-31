from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from homeassistant.core import HomeAssistant

from custom_components.termoweb.backend import termoweb_ws as module
from custom_components.termoweb.domain import NodeSettingsDelta
from custom_components.termoweb.inventory import build_node_inventory
from tests.test_termoweb_ws_protocol import DummyREST


def _make_hass() -> HomeAssistant:
    """Return a Home Assistant instance with a synchronous loop stub."""

    hass = HomeAssistant()
    hass.loop = SimpleNamespace(
        call_soon_threadsafe=lambda cb, *args: cb(*args),
        is_running=lambda: False,
    )
    return hass


def test_handle_event_routes_updates_to_deltas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Update events should translate into domain deltas without raw caches."""

    hass = _make_hass()
    hass.data.setdefault(module.DOMAIN, {})["entry"] = {}
    handle_ws_deltas = MagicMock()
    coordinator = SimpleNamespace(data={}, handle_ws_deltas=handle_ws_deltas)

    monkeypatch.setattr(module.TermoWebWSClient, "_install_write_hook", lambda self: None)

    client = module.TermoWebWSClient(
        hass,
        entry_id="entry",
        dev_id="device",
        api_client=DummyREST(),
        coordinator=coordinator,
        session=SimpleNamespace(closed=False),
    )

    inventory_payload = {"nodes": [{"type": "htr", "addr": "2"}]}
    client._inventory = module.Inventory(
        client.dev_id,
        inventory_payload,
        build_node_inventory(inventory_payload),
    )
    hass.data[module.DOMAIN]["entry"]["inventory"] = client._inventory
    hass.data[module.DOMAIN]["entry"]["coordinator"] = coordinator

    event_payload: dict[str, Any] = {
        "name": "update",
        "args": [
            {
                "nodes": {
                    "htr": {
                        "settings": {
                            "2": {"mode": "eco"},
                        }
                    }
                }
            }
        ],
    }

    client._handle_event(event_payload)

    handle_ws_deltas.assert_called_once()
    args, kwargs = handle_ws_deltas.call_args
    assert args[0] == client.dev_id
    deltas = args[1]
    assert isinstance(deltas, tuple)
    assert len(deltas) == 1
    delta = deltas[0]
    assert isinstance(delta, NodeSettingsDelta)
    assert delta.node_id.addr == "2"
    assert delta.node_id.node_type.value == "htr"
    assert delta.changes == {"mode": "eco"}
    assert kwargs["replace"] is False
    assert coordinator.data == {}
