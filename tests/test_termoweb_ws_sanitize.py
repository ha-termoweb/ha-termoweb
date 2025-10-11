"""Tests for TermoWeb websocket sanitization helpers."""

from __future__ import annotations

import asyncio
import types

import pytest

from custom_components.termoweb.backend import termoweb_ws


class FakeRESTClient:
    """Minimal stub of the REST client used by the websocket client."""

    def __init__(self) -> None:
        """Initialise default attributes expected by the websocket client."""
        self._session = types.SimpleNamespace(closed=True)
        self._is_ducaheat = False
        self.user_agent = None
        self.requested_with = None


def test_sanitise_placeholder_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Spy on sanitisation calls and verify known/unknown keys."""

    loop = asyncio.new_event_loop()
    try:
        hass = types.SimpleNamespace(loop=loop)
        ws_client = termoweb_ws.WebSocketClient(
            hass,
            entry_id="entry",
            dev_id="device",
            api_client=FakeRESTClient(),
            coordinator=types.SimpleNamespace(),
        )

        calls: list[str | None] = []

        def spy(value: str | None) -> str | None:
            calls.append(value)
            return value

        monkeypatch.setitem(
            termoweb_ws._SENSITIVE_PLACEHOLDERS, "token", ("{token}", spy)
        )

        placeholder = ws_client._sanitise_placeholder("token", "abc12345")
        assert placeholder == "{token}"
        assert calls == ["abc12345"]

        assert ws_client._sanitise_placeholder("unknown", "value") is None
    finally:
        loop.close()
