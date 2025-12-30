"""Tests covering legacy websocket handshake failure tracking."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

from custom_components.termoweb.backend.termoweb_ws import TermoWebWSClient
from custom_components.termoweb.backend.ws_client import HandshakeError


@pytest.mark.asyncio
async def test_handshake_failures_reset_after_threshold(monkeypatch, caplog) -> None:
    """Ensure repeated handshake failures trigger warnings and reset counters."""

    monkeypatch.setattr(TermoWebWSClient, "_install_write_hook", lambda self: None)

    hass = SimpleNamespace(loop=asyncio.get_running_loop())
    session = SimpleNamespace()
    api_client = SimpleNamespace(_session=session)

    client = TermoWebWSClient(
        hass,
        entry_id="entry-id",
        dev_id="device-id",
        api_client=api_client,
        coordinator=object(),
        session=session,
        handshake_fail_threshold=3,
    )

    async def _immediate_sleep(delay: float, *args, **kwargs) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _immediate_sleep)

    attempts = 0

    async def _failing_handshake(self) -> tuple[str, int]:
        nonlocal attempts
        attempts += 1
        if attempts == client._hs_fail_threshold:
            client._closing = True
        raise HandshakeError(
            status=503,
            url="https://unit.test/socket",
            detail=f"failure-{attempts}",
            response_snippet="failure",
        )

    monkeypatch.setattr(TermoWebWSClient, "_handshake", _failing_handshake)

    with caplog.at_level(logging.INFO):
        await client._run_socketio_09()

    assert attempts == client._hs_fail_threshold
    assert client._closing is True
    assert client._hs_fail_count == 0
    assert client._hs_fail_start == 0.0

    warnings = [
        record.message for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert any("WS: handshake failed" in message for message in warnings)
    assert any("3 times" in message for message in warnings)
