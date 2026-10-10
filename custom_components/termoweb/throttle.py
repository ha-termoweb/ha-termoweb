"""Shared throttling helpers for the TermoWeb integration."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import time
from typing import Any

SleepCallable = Callable[[float], Awaitable[Any]]
MonotonicCallable = Callable[[], float]


@dataclass(slots=True)
class MonotonicRateLimiter:
    """Enforce a minimum interval between asynchronous calls."""

    lock: asyncio.Lock
    monotonic: MonotonicCallable
    sleep: SleepCallable
    min_interval: float
    _last_monotonic: float = 0.0

    async def async_throttle(
        self, *, on_wait: Callable[[float], None] | None = None
    ) -> float:
        """Sleep if required to honour ``min_interval`` seconds between calls."""

        async with self.lock:
            now = self.monotonic()
            wait = self.min_interval - (now - self._last_monotonic)
            if wait > 0:
                if on_wait is not None:
                    on_wait(wait)
                await self.sleep(wait)
                now = self.monotonic()
            self._last_monotonic = now
            return max(wait, 0.0)

    def reset(self) -> None:
        """Reset the stored timestamp so the next call executes immediately."""

        self._last_monotonic = 0.0


_SAMPLES_RATE_LIMITER: MonotonicRateLimiter | None = None
_SAMPLES_INTERVAL = 1.0


def default_samples_rate_limit_state() -> MonotonicRateLimiter:
    """Return the shared rate limiter for heater samples requests."""

    global _SAMPLES_RATE_LIMITER  # noqa: PLW0603
    if _SAMPLES_RATE_LIMITER is None:
        _SAMPLES_RATE_LIMITER = MonotonicRateLimiter(
            lock=asyncio.Lock(),
            monotonic=time.monotonic,
            sleep=asyncio.sleep,
            min_interval=_SAMPLES_INTERVAL,
        )
    return _SAMPLES_RATE_LIMITER


def reset_samples_rate_limit_state() -> None:
    """Reset the shared samples rate limiter to its initial state."""

    default_samples_rate_limit_state().reset()
