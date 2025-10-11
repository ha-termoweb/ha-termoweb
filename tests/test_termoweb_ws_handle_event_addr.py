from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping
from unittest.mock import MagicMock

import pytest

from homeassistant.core import HomeAssistant

from custom_components.termoweb.backend import termoweb_ws as module
from tests.test_termoweb_ws_protocol import DummyREST


def _make_hass() -> HomeAssistant:
    """Return a Home Assistant instance with a synchronous loop stub."""

    hass = HomeAssistant()
    hass.loop = SimpleNamespace(
        call_soon_threadsafe=lambda cb, *args: cb(*args),
        is_running=lambda: False,
    )
    return hass


def test_handle_event_normalises_addr_and_records_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Events should normalise addresses for dispatch and clone raw payloads."""

    hass = _make_hass()
    hass.data.setdefault(module.DOMAIN, {})["entry"] = {}
    coordinator = SimpleNamespace(data={}, update_nodes=MagicMock())

    dispatcher = MagicMock()
    monkeypatch.setattr(module, "async_dispatcher_send", dispatcher)
    monkeypatch.setattr(module.TermoWebWSClient, "_install_write_hook", lambda self: None)
    monkeypatch.setattr(module.TermoWebWSClient, "_dispatch_nodes", lambda self, payload: {"htr": ["2"]})

    client = module.TermoWebWSClient(
        hass,
        entry_id="entry",
        dev_id="device",
        api_client=DummyREST(),
        coordinator=coordinator,
        session=SimpleNamespace(closed=False),
    )

    nodes_body: Mapping[str, Any] = {"htr": {"settings": {" 2 ": {"mode": "auto"}}}}
    settings_body: dict[str, Any] = {"mode": "manual", "flags": ["eco"]}
    event_payload = {
        "name": "data",
        "args": [
            [
                {"path": "/devs/device/mgr/nodes", "body": nodes_body},
                {"path": "/devs/device/htr/ 2 /settings", "body": settings_body},
            ]
        ],
    }

    client._handle_event(event_payload)

    payloads = [call.args[2] for call in dispatcher.call_args_list]
    assert payloads
    settings_payload = next(
        (payload for payload in payloads if payload.get("kind") == "htr_settings"),
        None,
    )
    assert settings_payload is not None
    assert settings_payload["addr_map"] == {"htr": ["2"]}

    raw_store = client._nodes_raw
    assert raw_store["htr"]["settings"][" 2 "] == {"mode": "manual", "flags": ["eco"]}
    assert raw_store["htr"]["settings"][" 2 "] is not settings_body

    settings_body["flags"].append("boost")
    assert raw_store["htr"]["settings"][" 2 "]["flags"] == ["eco"]
