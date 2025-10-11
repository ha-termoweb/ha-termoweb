"""Tests for the shared rate limiter helpers."""

from __future__ import annotations

import asyncio
import types

import pytest

from custom_components.termoweb.throttle import (
    MonotonicRateLimiter,
    default_samples_rate_limit_state,
    reset_samples_rate_limit_state,
)


def test_default_samples_rate_limit_state_round_trip() -> None:
    """Shared rate limiter should reuse the same lock and manage timestamps."""

    current = 0.4

    def fake_monotonic() -> float:
        return current

    sleep_calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        nonlocal current
        sleep_calls.append(delay)
        current += delay

    time_module = types.SimpleNamespace(monotonic=fake_monotonic)

    reset_samples_rate_limit_state(time_module=time_module, sleep=fake_sleep)

    limiter = default_samples_rate_limit_state()
    assert isinstance(limiter, MonotonicRateLimiter)
    assert isinstance(limiter.lock, asyncio.Lock)

    same_limiter = default_samples_rate_limit_state()
    assert limiter is same_limiter

    asyncio.run(limiter.async_throttle())
    assert sleep_calls == [pytest.approx(0.6)]
    assert limiter.last_timestamp() == pytest.approx(1.0)

    current = 1.8
    asyncio.run(limiter.async_throttle())
    assert sleep_calls == [pytest.approx(0.6), pytest.approx(0.2)]
    assert limiter.last_timestamp() == pytest.approx(2.0)

    reset_samples_rate_limit_state()
    assert limiter.last_timestamp() == 0.0


def test_default_samples_rate_limit_state_overrides_time_and_sleep() -> None:
    """Existing limiter should accept new monotonic and sleep callables."""

    limiter = default_samples_rate_limit_state()

    current = 3.2
    monotonic_calls: list[float] = []

    def new_monotonic() -> float:
        monotonic_calls.append(current)
        return current

    sleep_calls: list[float] = []

    async def new_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    new_time = types.SimpleNamespace(monotonic=new_monotonic)

    reset_samples_rate_limit_state(time_module=new_time, sleep=new_sleep)

    updated = default_samples_rate_limit_state(time_module=new_time, sleep=new_sleep)

    assert updated is limiter
    assert updated.monotonic is new_monotonic
    assert updated.sleep is new_sleep

    asyncio.run(updated.async_throttle())

    assert monotonic_calls == [pytest.approx(current)]
    assert not sleep_calls

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


def test_set_last_timestamp_short_circuits_future_sleep() -> None:
    """Future timestamp override should bypass sleeping and update state."""

    def fake_monotonic() -> float:
        return 15.0

    sleep_calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    limiter = MonotonicRateLimiter(
        lock=asyncio.Lock(),
        monotonic=fake_monotonic,
        sleep=fake_sleep,
        min_interval=1.0,
    )

    limiter.set_last_timestamp(10.0)
    assert limiter.last_timestamp() == pytest.approx(10.0)

    result = asyncio.run(limiter.async_throttle())

    assert result == pytest.approx(0.0)
    assert not sleep_calls
    assert limiter.last_timestamp() == pytest.approx(15.0)
