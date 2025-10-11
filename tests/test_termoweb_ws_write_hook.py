"""Tests for the TermoWeb websocket write hook installation."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from custom_components.termoweb.backend.termoweb_ws import TermoWebWSClient
from homeassistant.core import HomeAssistant


class DummyREST:
    """Provide a minimal REST client where writes echo their inputs."""

    def __init__(self) -> None:
        """Initialise mock REST client helpers used by the websocket client."""

        self._session = SimpleNamespace(closed=False)
        self._ensure_token = AsyncMock()
        self.authed_headers = AsyncMock(return_value={"Authorization": "Bearer token"})
        self.api_base = "https://api.termoweb"
        self.user_agent = "agent"
        self.requested_with = "requested"

    async def set_node_settings(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Return the received arguments to make verification straightforward."""

        return {"args": args, "kwargs": kwargs}


@pytest.mark.asyncio
async def test_write_hook_dispatches_once_and_is_idempotent() -> None:
    """Ensure the write hook dispatches once and installation remains idempotent."""

    hass = HomeAssistant()
    hass.loop = asyncio.get_running_loop()
    coordinator = SimpleNamespace(data={}, update_nodes=AsyncMock())
    rest_client = DummyREST()

    client = TermoWebWSClient(
        hass,
        entry_id="entry-id",
        dev_id="dev-123",
        api_client=rest_client,
        coordinator=coordinator,
        session=rest_client._session,
    )

    restart_mock = AsyncMock()
    client.maybe_restart_after_write = restart_mock  # type: ignore[assignment]

    node = SimpleNamespace(type="htr", addr="1")
    payload = {"mode": "auto"}

    result = await rest_client.set_node_settings("dev-123", node, **payload)

    assert result == {"args": ("dev-123", node), "kwargs": payload}
    assert restart_mock.await_count == 1

    watchers = getattr(rest_client, "_tw_ws_write_watchers")
    device_watchers = watchers.get("dev-123")
    assert device_watchers is not None
    assert client in device_watchers
    assert len(list(device_watchers)) == 1

    original_watchers = device_watchers
    original_wrapper = rest_client.set_node_settings

    client._install_write_hook()

    assert getattr(rest_client, "_tw_ws_write_watchers")["dev-123"] is original_watchers
    assert rest_client.set_node_settings is original_wrapper
    assert len(list(original_watchers)) == 1

    second_result = await rest_client.set_node_settings("dev-123", node, **payload)

    assert second_result == result
    assert restart_mock.await_count == 2
    assert len(list(getattr(rest_client, "_tw_ws_write_watchers")["dev-123"])) == 1
