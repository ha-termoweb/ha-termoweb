"""Tests for mapping-derived payload window hints."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from custom_components.termoweb.backend import ducaheat_ws
from homeassistant.core import HomeAssistant


class DummyREST:
    """Provide the minimal REST interface required by the websocket client."""

    def __init__(self) -> None:
        self._session = SimpleNamespace()


def _make_client() -> ducaheat_ws.DucaheatWSClient:
    """Return a websocket client with stubbed dependencies."""

    hass = HomeAssistant()
    hass.data.setdefault(ducaheat_ws.DOMAIN, {})["entry"] = {}
    coordinator = SimpleNamespace(
        update_nodes=MagicMock(),
        data={
            "device": {
                "nodes_by_type": {},
                "addr_map": {},
                "settings": {},
                "addresses_by_type": {},
            }
        },
    )
    return ducaheat_ws.DucaheatWSClient(
        hass,
        entry_id="entry",
        dev_id="device",
        api_client=DummyREST(),
        coordinator=coordinator,
    )


def test_update_payload_window_applies_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure mapping cadence candidates trigger window hint updates."""

    client = _make_client()
    monkeypatch.setattr(
        client,
        "_extract_cadence_candidates",
        MagicMock(return_value=[45, 90]),
    )
    wrapped = MagicMock(wraps=client._apply_payload_window_hint)
    monkeypatch.setattr(client, "_apply_payload_window_hint", wrapped)

    client._update_payload_window_from_mapping(
        {"status": {"lease_seconds": 45}},
        source="snapshot",
    )

    wrapped.assert_called_once_with(source="snapshot", candidates=[45, 90])


def test_update_payload_window_ignores_empty_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure missing cadence hints keep the payload window unchanged."""

    client = _make_client()
    monkeypatch.setattr(
        client,
        "_extract_cadence_candidates",
        MagicMock(return_value=[]),
    )
    wrapped = MagicMock(wraps=client._apply_payload_window_hint)
    monkeypatch.setattr(client, "_apply_payload_window_hint", wrapped)

    client._update_payload_window_from_mapping({}, source="poll")

    wrapped.assert_not_called()
