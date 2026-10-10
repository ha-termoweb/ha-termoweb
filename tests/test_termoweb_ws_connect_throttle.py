"""Regression tests for TermoWeb websocket handshake retry throttling."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from conftest import DummyREST
import pytest

from custom_components.termoweb.backend import termoweb_ws as module
from custom_components.termoweb.backend.ws_client import ConnectionRateLimiter

BACKOFF_FLOOR = 5 * 0.8  # first backoff step with minimum jitter


class _FakeClock:
    """Monotonic fake clock advanced only by fake sleeps."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, delay: float, *_args: Any, **_kwargs: Any) -> None:
        self.now += max(0.0, float(delay))


class _Resp:
    """Minimal aiohttp response returning HTTP 401."""

    status = 401

    async def text(self) -> str:
        return "unauthorized"

    async def __aenter__(self) -> _Resp:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None


@pytest.mark.asyncio
async def test_handshake_retries_use_limiter_and_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed TermoWeb handshakes must be rate-limited and spaced by backoff."""

    clock = _FakeClock()
    monkeypatch.setattr(module.asyncio, "sleep", clock.sleep)
    monkeypatch.setattr(module.time, "time", clock)

    attempts: list[float] = []
    session = SimpleNamespace(closed=False)

    def _get(url: str, **_kwargs: Any) -> _Resp:
        attempts.append(clock.now)
        if len(attempts) >= 4:
            client._closing = True
        return _Resp()

    session.get = _get
    api_client = DummyREST()
    api_client._session = session
    hass = SimpleNamespace(loop=asyncio.get_running_loop(), data={module.DOMAIN: {}})

    client = module.TermoWebWSClient(
        hass,
        entry_id="entry",
        dev_id="device",
        api_client=api_client,
        coordinator=SimpleNamespace(),
        session=session,
    )
    client._dispatcher_mock = MagicMock()  # type: ignore[attr-defined]

    limiter = client._connect_limiter
    assert isinstance(limiter, ConnectionRateLimiter)
    limiter._clock = clock
    limiter._sleep = clock.sleep
    spy = AsyncMock(wraps=limiter.wait_for_slot)
    monkeypatch.setattr(limiter, "wait_for_slot", spy)

    await client._run_socketio_09()

    assert len(attempts) == 4
    assert spy.await_count == 4
    gaps = [later - earlier for earlier, later in zip(attempts, attempts[1:])]
    assert all(gap >= BACKOFF_FLOOR for gap in gaps), gaps
    # Every 401 forces a token refresh, so spacing bounds token endpoint load too.
    assert api_client._ensure_token.await_count == 4
