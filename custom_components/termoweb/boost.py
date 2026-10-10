"""Helpers for parsing boost metadata shared across TermoWeb modules."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import logging
from typing import Any, Final

from homeassistant.util import dt as dt_util

from .coerce import as_bool, as_int

_LOGGER = logging.getLogger(__name__)


def coerce_boost_minutes(value: Any) -> int | None:
    """Return ``value`` as a positive integer minute count when possible."""

    if value is None or isinstance(value, bool):
        return None

    minutes = as_int(value)
    if minutes is None or minutes <= 0:
        return None

    return minutes


def supports_boost(node: Any) -> bool:
    """Return ``True`` when ``node`` exposes boost controls."""

    candidate = getattr(node, "supports_boost", None)

    if isinstance(candidate, bool):
        return candidate

    if callable(candidate):
        try:
            candidate = candidate()
        except Exception:  # noqa: BLE001 - defensive
            node_ref = getattr(node, "addr", node)
            _LOGGER.debug(
                "Ignoring boost support probe failure for node %r",
                node_ref,
                exc_info=True,
            )
            return False

    result = as_bool(candidate)
    if result is not None:
        return result

    return False


def resolve_boost_end_from_fields(
    boost_end_day: Any,
    boost_end_min: Any,
    *,
    now: datetime | None = None,
) -> tuple[datetime | None, int | None]:
    """Translate boost end ``day``/``minute`` fields into a timestamp."""

    day = as_int(boost_end_day)
    minute = as_int(boost_end_min)
    if day is None or minute is None or minute < 0:
        return None, None

    now_dt = now or dt_util.now()
    tzinfo = now_dt.tzinfo or UTC

    candidates: list[datetime] = []

    if 0 < day <= 400:
        for year_offset in (-1, 0, 1):
            year = now_dt.year + year_offset
            try:
                start = datetime(year, 1, 1, tzinfo=tzinfo)
            except ValueError:  # pragma: no cover - defensive
                continue
            candidate = start + timedelta(days=day - 1, minutes=minute)
            candidates.append(candidate)

    if day >= 0:
        epoch_candidate = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(
            days=day,
            minutes=minute,
        )
        candidates.append(epoch_candidate.astimezone(tzinfo))

    if not candidates:
        return None, None

    window = 7 * 24 * 3600
    filtered = [
        candidate
        for candidate in candidates
        if abs((candidate - now_dt).total_seconds()) <= window
    ]
    if filtered:
        candidates = filtered

    def _candidate_key(candidate: datetime) -> tuple[int, float]:
        delta_seconds = (candidate - now_dt).total_seconds()
        is_future = 0 if delta_seconds >= 0 else 1
        return is_future, abs(delta_seconds)

    selected = min(candidates, key=_candidate_key)
    delta_seconds = (selected - now_dt).total_seconds()
    minutes_remaining = int(max(0.0, delta_seconds) // 60)

    return selected, minutes_remaining


ALLOWED_BOOST_MINUTES: Final[tuple[int, ...]] = tuple(range(60, 601, 60))
"""Valid boost durations (in minutes) supported by TermoWeb heaters."""

ALLOWED_BOOST_MINUTES_SET: Final[frozenset[int]] = frozenset(ALLOWED_BOOST_MINUTES)
ALLOWED_BOOST_MINUTES_MESSAGE: Final[str] = ", ".join(
    str(option) for option in ALLOWED_BOOST_MINUTES
)


def validate_boost_minutes(value: int | None) -> int | None:
    """Return a validated boost duration in minutes or ``None``."""

    if value is None:
        return None
    try:
        minutes = int(value)
    except (TypeError, ValueError) as err:  # pragma: no cover - defensive
        raise ValueError(f"Invalid boost_time value: {value!r}") from err
    if minutes not in ALLOWED_BOOST_MINUTES_SET:
        raise ValueError(f"boost_time must be one of: {ALLOWED_BOOST_MINUTES_MESSAGE}")
    return minutes
