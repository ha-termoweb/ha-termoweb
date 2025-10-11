"""Tests for the websocket heartbeat loop."""

from __future__ import annotations

from functools import partial
from unittest.mock import AsyncMock

import pytest

from custom_components.termoweb.backend.termoweb_ws import TermoWebWSClient


@pytest.mark.asyncio
async def test_heartbeat_loop_invokes_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the heartbeat loop passes the interval and sender to the runner."""

    client: TermoWebWSClient = TermoWebWSClient.__new__(TermoWebWSClient)
    client._hb_send_interval = 12.0  # type: ignore[attr-defined]
    send_mock = AsyncMock()
    client._send_text = send_mock  # type: ignore[attr-defined]

    run_mock = AsyncMock()
    monkeypatch.setattr(client, "_run_heartbeat", run_mock, raising=False)

    await client._heartbeat_loop()

    assert run_mock.await_count == 1
    interval_arg, sender_arg = run_mock.await_args.args
    assert interval_arg == client._hb_send_interval
    assert isinstance(sender_arg, partial)
    assert sender_arg.func is send_mock
    assert sender_arg.args == ("2::",)

    await sender_arg()
    send_mock.assert_awaited_once_with("2::")


@pytest.mark.asyncio
async def test_heartbeat_loop_swallows_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure the heartbeat loop ignores runtime errors from the runner."""

    client: TermoWebWSClient = TermoWebWSClient.__new__(TermoWebWSClient)
    client._hb_send_interval = 5.0  # type: ignore[attr-defined]
    client._send_text = AsyncMock()  # type: ignore[attr-defined]

    run_mock = AsyncMock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr(client, "_run_heartbeat", run_mock, raising=False)

    await client._heartbeat_loop()

    assert run_mock.await_count == 1
