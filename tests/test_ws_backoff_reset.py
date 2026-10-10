"""Regression tests for websocket reconnect backoff reset and status ordering."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from conftest import DummyREST, build_entry_runtime
import pytest

from custom_components.termoweb.backend import ducaheat_ws, termoweb_ws

SESSIONS = 5
ESCALATING = [5, 10, 30, 120, 300]


def _stub_loop() -> SimpleNamespace:
    """Return a loop stub whose tasks never run."""

    def _create_task(coro: Any, **_kwargs: Any) -> SimpleNamespace:
        coro.close()
        return SimpleNamespace(done=lambda: False, cancel=lambda: None)

    return SimpleNamespace(
        create_task=_create_task,
        call_soon_threadsafe=lambda cb, *args: cb(*args),
    )


def _install_sleep_recorder(
    monkeypatch: pytest.MonkeyPatch, client: Any
) -> list[tuple[float, str]]:
    """Record backoff sleeps with the tracker status seen during each one."""

    sleeps: list[tuple[float, str]] = []

    async def _sleep(delay: float, *_args: Any, **_kwargs: Any) -> None:
        if delay <= 0:
            return
        sleeps.append((delay, client._ws_health_tracker().status))
        if len(sleeps) >= SESSIONS:
            raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    client._connect_limiter._sleep = AsyncMock()
    return sleeps


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("healthy_sessions", "expected"),
    [(True, [5] * SESSIONS), (False, ESCALATING)],
)
async def test_ducaheat_backoff_resets_after_healthy_session(
    monkeypatch: pytest.MonkeyPatch,
    healthy_sessions: bool,
    expected: list[int],
) -> None:
    """Ducaheat restarts backoff after a healthy session and reports disconnect first."""

    hass = SimpleNamespace(loop=_stub_loop(), data={termoweb_ws.DOMAIN: {}})
    build_entry_runtime(hass=hass, entry_id="entry", dev_id="device")
    client = ducaheat_ws.DucaheatWSClient(
        hass,
        entry_id="entry",
        dev_id="device",
        api_client=DummyREST(is_ducaheat=True),
        coordinator=SimpleNamespace(update_nodes=MagicMock()),
        session=SimpleNamespace(),
    )
    client._dispatcher_mock = MagicMock()  # type: ignore[attr-defined]
    sleeps = _install_sleep_recorder(monkeypatch, client)

    async def _read_loop_ws() -> None:
        if healthy_sessions:
            client._mark_ws_payload(timestamp=time.time())
            client._update_status("healthy")
        raise RuntimeError("websocket closed")

    monkeypatch.setattr(client, "_connect_once", AsyncMock())
    monkeypatch.setattr(client, "_read_loop_ws", _read_loop_ws)

    with pytest.raises(asyncio.CancelledError):
        await client._runner()

    assert [delay for delay, _ in sleeps] == expected
    assert all(status == "disconnected" for _, status in sleeps)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("healthy_sessions", "expected"),
    [(True, [5] * SESSIONS), (False, ESCALATING)],
)
async def test_termoweb_backoff_resets_after_healthy_session(
    monkeypatch: pytest.MonkeyPatch,
    healthy_sessions: bool,
    expected: list[int],
) -> None:
    """TermoWeb restarts backoff after a healthy session and reports disconnect first."""

    monkeypatch.setattr(termoweb_ws.random, "uniform", lambda _a, _b: 1.0)
    session = SimpleNamespace(closed=False)
    api_client = DummyREST()
    api_client._session = session
    hass = SimpleNamespace(loop=_stub_loop(), data={termoweb_ws.DOMAIN: {}})
    build_entry_runtime(hass=hass, entry_id="entry", dev_id="device")
    client = termoweb_ws.TermoWebWSClient(
        hass,
        entry_id="entry",
        dev_id="device",
        api_client=api_client,
        coordinator=SimpleNamespace(),
        session=session,
    )
    client._dispatcher_mock = MagicMock()  # type: ignore[attr-defined]
    sleeps = _install_sleep_recorder(monkeypatch, client)

    async def _connect_ws(_sid: str) -> None:
        client._ws = SimpleNamespace(
            closed=False, send_str=AsyncMock(), close=AsyncMock()
        )

    async def _read_loop() -> None:
        if healthy_sessions:
            client._mark_event(paths=None, count_event=True)
        raise RuntimeError("server disconnect")

    monkeypatch.setattr(client, "_handshake", AsyncMock(return_value=("sid", 60)))
    monkeypatch.setattr(client, "_connect_ws", _connect_ws)
    monkeypatch.setattr(client, "_read_loop", _read_loop)

    with pytest.raises(asyncio.CancelledError):
        await client._run_socketio_09()

    assert [delay for delay, _ in sleeps] == expected
    assert all(status == "disconnected" for _, status in sleeps)
