"""Shared throttling helpers for the TermoWeb integration."""

from __future__ import annotations

"""Shared throttling helpers for the TermoWeb integration."""

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

    def last_timestamp(self) -> float:
        """Return the timestamp of the most recent throttled call."""

        return self._last_monotonic

    def set_last_timestamp(self, value: float) -> None:
        """Update the stored timestamp for the last throttled call."""

        self._last_monotonic = value


_SAMPLES_RATE_LIMITER: MonotonicRateLimiter | None = None
_SAMPLES_INTERVAL = 1.0


def _new_samples_rate_limiter(
    *, time_module: Any | None = None, sleep: SleepCallable | None = None
) -> MonotonicRateLimiter:
    """Return a freshly constructed rate limiter for heater samples."""

    time_mod = time_module or time
    return MonotonicRateLimiter(
        lock=asyncio.Lock(),
        monotonic=time_mod.monotonic,
        sleep=sleep or asyncio.sleep,
        min_interval=_SAMPLES_INTERVAL,
    )


def default_samples_rate_limit_state(
    *, time_module: Any | None = None, sleep: SleepCallable | None = None
) -> MonotonicRateLimiter:
    """Return the shared rate limiter for heater samples requests."""

    global _SAMPLES_RATE_LIMITER
    if _SAMPLES_RATE_LIMITER is None:
        _SAMPLES_RATE_LIMITER = _new_samples_rate_limiter(
            time_module=time_module,
            sleep=sleep,
        )
    else:
        if time_module is not None:
            _SAMPLES_RATE_LIMITER.monotonic = time_module.monotonic
        if sleep is not None:
            _SAMPLES_RATE_LIMITER.sleep = sleep
    return _SAMPLES_RATE_LIMITER


def reset_samples_rate_limit_state(
    *, time_module: Any | None = None, sleep: SleepCallable | None = None
) -> None:
    """Reset the shared samples rate limiter to its initial state."""

    global _SAMPLES_RATE_LIMITER
    limiter = default_samples_rate_limit_state(
        time_module=time_module, sleep=sleep
    )
    limiter.reset()
