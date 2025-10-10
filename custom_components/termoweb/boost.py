"""Helpers for parsing boost metadata shared across TermoWeb modules."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import math
from typing import Any

from homeassistant.util import dt as dt_util

from .inventory import Inventory, Node, normalize_node_addr, normalize_node_type


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


@dataclass(frozen=True, slots=True)
class InventoryHeaterMetadata:
    """Metadata describing a heater node sourced from an inventory."""

    node_type: str
    addr: str
    name: str
    node: Node
    supports_boost: bool


def iter_inventory_heater_metadata(
    inventory: Inventory | None,
    *,
    default_name_simple: Callable[[str], str] | None = None,
) -> Iterator[InventoryHeaterMetadata]:
    """Yield metadata for heater nodes described by ``inventory``."""

    if not isinstance(inventory, Inventory):
        return

    default_factory = default_name_simple or (lambda addr: f"Heater {addr}")
    name_map = inventory.heater_name_map(default_factory)
    explicit_names = inventory.explicit_heater_names

    if isinstance(name_map, dict):
        names_by_type = name_map.get("by_type", {})  # type: ignore[assignment]
        legacy_names = name_map.get("htr", {})  # type: ignore[assignment]
        name_lookup = name_map
    else:  # pragma: no cover - defensive
        names_by_type = {}
        legacy_names = {}
        name_lookup = {}

    def _resolve_name(node_type: str, addr: str) -> str:
        node_type_norm = normalize_node_type(
            node_type,
            use_default_when_falsey=True,
        )
        addr_norm = normalize_node_addr(
            addr,
            use_default_when_falsey=True,
        )
        if not node_type_norm or not addr_norm:
            return default_factory(addr_norm or addr)  # pragma: no cover - defensive

        default_simple = default_factory(addr_norm)

        def _candidate(value: Any) -> str | None:
            if not isinstance(value, str) or not value:
                return None
            if (
                node_type_norm == "acm"
                and value == default_simple
                and (node_type_norm, addr_norm) not in explicit_names
            ):
                return None
            return value

        per_type = (
            names_by_type.get(node_type_norm, {})
            if isinstance(names_by_type, dict)
            else {}
        )
        for candidate in (
            per_type.get(addr_norm) if isinstance(per_type, dict) else None,
            name_lookup.get((node_type_norm, addr_norm)),
            legacy_names.get(addr_norm) if isinstance(legacy_names, dict) else None,
        ):
            resolved = _candidate(candidate)
            if resolved:
                return resolved

        if node_type_norm == "acm":
            return f"Accumulator {addr_norm}"
        return default_simple  # pragma: no cover - defensive

    for node in inventory.heater_nodes:
        node_type = normalize_node_type(
            getattr(node, "type", None),
            use_default_when_falsey=True,
        )
        addr = normalize_node_addr(
            getattr(node, "addr", None),
            use_default_when_falsey=True,
        )
        if not node_type or not addr:
            continue  # pragma: no cover - defensive

        name = _resolve_name(node_type, addr)
        candidate = getattr(node, "supports_boost", None)
        if callable(candidate):
            try:
                result = candidate()
            except Exception:  # noqa: BLE001 - defensive  # pragma: no cover - defensive
                result = None
        else:
            result = candidate  # pragma: no cover - defensive
        supports = coerce_boost_bool(result)
        supports_boost = bool(supports) if supports is not None else False

        yield InventoryHeaterMetadata(
            node_type=node_type,
            addr=addr,
            name=name,
            node=node,
            supports_boost=supports_boost,
        )

