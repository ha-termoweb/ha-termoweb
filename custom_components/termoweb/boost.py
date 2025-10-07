"""Helpers for parsing boost metadata shared across TermoWeb modules."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import math
from typing import Any

from homeassistant.util import dt as dt_util


def coerce_int(value: Any) -> int | None:
    """Return ``value`` as ``int`` when possible, else ``None``."""

    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return int(value)
    try:
        candidate = str(value).strip()
    except Exception:  # noqa: BLE001 - defensive
        return None
    if not candidate:
        return None
    try:
        return int(float(candidate))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def coerce_boost_bool(value: Any) -> bool | None:
    """Return ``value`` as a boolean when possible."""

    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
    try:
        text = str(value).strip().lower()
    except Exception:  # noqa: BLE001 - defensive
        return None
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off"}:
        return False
    return None


def coerce_boost_minutes(value: Any) -> int | None:
    """Return ``value`` as positive minutes when possible."""

    if value is None or isinstance(value, bool):
        return None
    try:
        if isinstance(value, (int, float)):
            minutes = int(value)
        else:
            text = str(value).strip()
            if not text:
                return None
            minutes = int(float(text))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None
    return minutes if minutes > 0 else None


def coerce_boost_remaining_minutes(value: Any) -> int | None:
    """Return ``value`` as a positive integer minute count when possible."""

    if value is None or isinstance(value, bool):
        return None

    candidate: int | None
    try:
        if isinstance(value, (int, float)):
            candidate = int(value)
        else:
            text = str(value).strip()
            if not text:
                return None
            candidate = int(float(text))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None

    if candidate is None or candidate <= 0:
        return None

    return candidate


def resolve_boost_end_from_fields(
    boost_end_day: Any,
    boost_end_min: Any,
    *,
    now: datetime | None = None,
) -> tuple[datetime | None, int | None]:
    """Translate boost end ``day``/``minute`` fields into a timestamp."""

    day = coerce_int(boost_end_day)
    minute = coerce_int(boost_end_min)
    if day is None or minute is None or minute < 0:
        return None, None

    now_dt = now or dt_util.now()
    tzinfo = now_dt.tzinfo or getattr(dt_util, "UTC", UTC)

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
        epoch_timezone = getattr(dt_util, "UTC", UTC)
        epoch_candidate = datetime(1970, 1, 1, tzinfo=epoch_timezone) + timedelta(
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

