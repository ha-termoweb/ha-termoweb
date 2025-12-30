"""Tests for the TermoWeb websocket RTC keepalive loop."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from custom_components.termoweb.backend import termoweb_ws as module
from homeassistant.core import HomeAssistant
from tests.test_termoweb_ws_protocol import DummyREST


@pytest.mark.asyncio
async def test_rtc_keepalive_loop_retries_and_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RTC keepalive loop should retry after errors until closing."""

    hass = HomeAssistant()
    hass.loop = SimpleNamespace(
        call_soon_threadsafe=lambda cb, *args: cb(*args),
        is_running=lambda: False,
    )
    coordinator = SimpleNamespace(data={}, update_nodes=MagicMock())

    monkeypatch.setattr(
        module.TermoWebWSClient, "_install_write_hook", lambda self: None
    )
    client = module.TermoWebWSClient(
        hass,
        entry_id="entry",
        dev_id="device",
        api_client=DummyREST(),
        coordinator=coordinator,
        session=SimpleNamespace(closed=False),
    )
    client._closing = False
    client._rtc_keepalive_interval = 1.0

    sleep_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(module.asyncio, "sleep", sleep_mock)

    steps = ["success", "error", "stop"]
    events: list[str] = []

    async def rtc_side_effect(dev_id: str) -> None:
        step = steps[len(events)]
        events.append(step)
        assert dev_id == client.dev_id
        if step == "error":
            raise RuntimeError("boom")
        if step == "stop":
            client._closing = True

    rtc_mock = AsyncMock(side_effect=rtc_side_effect)
    monkeypatch.setattr(client._client, "get_rtc_time", rtc_mock, raising=False)

    await asyncio.wait_for(client._rtc_keepalive_loop(), timeout=1.0)

    assert events == ["success", "error", "stop"]
    assert rtc_mock.await_args_list == [call(client.dev_id)] * 3
    assert sleep_mock.await_args_list == [call(client._rtc_keepalive_interval)] * 3
    assert rtc_mock.await_count == 3
    assert sleep_mock.await_count == 3
