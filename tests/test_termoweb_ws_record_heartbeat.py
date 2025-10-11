"""Tests for websocket heartbeat recording helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from custom_components.termoweb.backend import termoweb_ws as module
from custom_components.termoweb.backend.termoweb_ws import TermoWebWSClient
from homeassistant.core import HomeAssistant

from tests.test_termoweb_ws_protocol import DummyREST


def test_record_heartbeat_updates_timestamps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure ``_record_heartbeat`` updates event timestamps consistently."""

    hass = HomeAssistant()
    hass.loop = SimpleNamespace(
        call_soon_threadsafe=lambda cb, *args: cb(*args),
        is_running=lambda: False,
    )
    hass.data.setdefault(module.DOMAIN, {})["entry"] = {}
    coordinator = SimpleNamespace(data={}, update_nodes=MagicMock())

    monkeypatch.setattr(TermoWebWSClient, "_install_write_hook", lambda self: None)

    client = TermoWebWSClient(
        hass,
        entry_id="entry",
        dev_id="device",
        api_client=DummyREST(),
        coordinator=coordinator,
        session=SimpleNamespace(closed=False),
    )

    captured: list[float] = []

    def _capture_mark(*, timestamp: float) -> None:
        captured.append(timestamp)

    monkeypatch.setattr(client, "_mark_ws_heartbeat", _capture_mark, raising=False)

    client._record_heartbeat(source="socketio")

    assert len(captured) == 1
    recorded_ts = captured[0]
    assert client._stats.last_event_ts == recorded_ts
    assert client._last_event_at == recorded_ts
    assert client._last_heartbeat_at == recorded_ts
