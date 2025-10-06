"""Tests for the shared rate limiter helpers."""

from __future__ import annotations

import asyncio

from custom_components.termoweb.energy import (
    RateLimitState,
    default_samples_rate_limit_state,
    reset_samples_rate_limit_state,
)


def test_default_samples_rate_limit_state_round_trip() -> None:
    """Shared rate limiter should reuse the same lock and manage timestamps."""

    reset_samples_rate_limit_state()

    state1 = default_samples_rate_limit_state()
    assert isinstance(state1, RateLimitState)
    assert isinstance(state1.lock, asyncio.Lock)

    state2 = default_samples_rate_limit_state()
    assert state1.lock is state2.lock

    state1.set_last_query(123.45)
    assert state1.get_last_query() == 123.45
    assert state2.get_last_query() == 123.45

    reset_samples_rate_limit_state()
    assert state1.get_last_query() == 0.0
