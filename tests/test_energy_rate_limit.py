"""Tests for the shared rate limiter helpers."""

from __future__ import annotations

import asyncio

import pytest

from custom_components.termoweb.throttle import (
    MonotonicRateLimiter,
    default_samples_rate_limit_state,
    reset_samples_rate_limit_state,
)


def test_default_samples_rate_limit_state_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shared rate limiter is reused, throttles calls and can be reset."""

    current = 0.4

    def fake_monotonic() -> float:
        return current

    sleep_calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        nonlocal current
        sleep_calls.append(delay)
        current += delay

    limiter = default_samples_rate_limit_state()
    assert isinstance(limiter, MonotonicRateLimiter)
    assert isinstance(limiter.lock, asyncio.Lock)
    assert default_samples_rate_limit_state() is limiter

    monkeypatch.setattr(limiter, "monotonic", fake_monotonic)
    monkeypatch.setattr(limiter, "sleep", fake_sleep)
    reset_samples_rate_limit_state()

    asyncio.run(limiter.async_throttle())
    assert sleep_calls == [pytest.approx(0.6)]

    current = 1.8
    asyncio.run(limiter.async_throttle())
    assert sleep_calls == [pytest.approx(0.6), pytest.approx(0.2)]

    reset_samples_rate_limit_state()
    asyncio.run(limiter.async_throttle())
    assert sleep_calls == [pytest.approx(0.6), pytest.approx(0.2)]

    reset_samples_rate_limit_state()


def test_async_throttle_invokes_on_wait_callback() -> None:
    """Rate limiter should report the computed delay to on_wait callback."""

    monotonic_values = iter([0.2, 1.2])

    def fake_monotonic() -> float:
        return next(monotonic_values)

    sleep_calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    on_wait_calls: list[float] = []

    def on_wait(delay: float) -> None:
        on_wait_calls.append(delay)

    limiter = MonotonicRateLimiter(
        lock=asyncio.Lock(),
        monotonic=fake_monotonic,
        sleep=fake_sleep,
        min_interval=1.0,
    )

    result = asyncio.run(limiter.async_throttle(on_wait=on_wait))

    assert result == pytest.approx(0.8)
    assert on_wait_calls == [pytest.approx(0.8)]
    assert sleep_calls == [pytest.approx(0.8)]
