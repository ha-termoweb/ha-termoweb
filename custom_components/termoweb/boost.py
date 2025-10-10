"""Helpers for parsing boost metadata shared across TermoWeb modules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Callable, Iterator

from homeassistant.util import dt as dt_util

from .inventory import (
    Inventory,
    Node,
    heater_platform_details_from_inventory,
    normalize_node_addr,
    normalize_node_type,
)


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
    nodes_by_type, addresses_by_type, resolve_name = (
        heater_platform_details_from_inventory(
            inventory,
            default_name_simple=default_factory,
        )
    )

    for node_type_raw, addresses in addresses_by_type.items():
        node_type = normalize_node_type(
            node_type_raw,
            use_default_when_falsey=True,
        )
        if not node_type or not addresses:
            continue

        nodes = nodes_by_type.get(node_type, [])
        if not isinstance(nodes, list):  # pragma: no cover - defensive
            nodes = list(nodes)

        node_lookup: dict[str, Node] = {}
        for node in nodes:
            addr = normalize_node_addr(
                getattr(node, "addr", None),
                use_default_when_falsey=True,
            )
            if addr and addr not in node_lookup:
                node_lookup[addr] = node

        for addr_raw in addresses:
            addr = normalize_node_addr(
                addr_raw,
                use_default_when_falsey=True,
            )
            if addr:
                node = node_lookup.get(addr)
                if node is not None:
                    name = resolve_name(node_type, addr)
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

