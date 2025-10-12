"""Helpers for parsing boost metadata shared across TermoWeb modules."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from datetime import UTC, datetime, timedelta
import logging
import math
from typing import Any, Final, cast

from homeassistant.util import dt as dt_util

from .inventory import (
    Inventory,
    Node,
    heater_platform_details_from_inventory,
    normalize_node_addr,
    normalize_node_type,
)

_LOGGER = logging.getLogger(__name__)


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

    minutes = coerce_int(value)
    if minutes is None or minutes <= 0:
        return None

    return minutes


def coerce_boost_remaining_minutes(value: Any) -> int | None:
    """Return ``value`` as a positive integer minute count when possible."""

    if value is None or isinstance(value, bool):
        return None

    candidate = coerce_int(value)
    if candidate is None or candidate <= 0:
        return None

    return candidate


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

    result = coerce_boost_bool(candidate)
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


type HeaterInventoryEntry = tuple[str, str, str, Node]
"""Type alias describing ``(node_type, addr, name, node)`` tuples."""


def iter_inventory_boostable_metadata(
    inventory: Inventory | None,
) -> Iterator[tuple[str, str, str]]:
    """Yield canonical boostable heater metadata from ``inventory``."""

    for node_type, addr, base_name, node in iter_inventory_heater_metadata(inventory):
        if not supports_boost(node):
            continue
        canonical_type = normalize_node_type(
            node_type,
            use_default_when_falsey=True,
        )
        canonical_addr = normalize_node_addr(
            addr,
            use_default_when_falsey=True,
        )
        if not canonical_type or not canonical_addr:
            continue
        yield canonical_type, canonical_addr, base_name


def _iter_node_container(nodes: Iterable[Any] | Mapping[Any, Any] | Any) -> Iterator[Node]:
    """Yield ``Node`` instances from ``nodes`` regardless of container shape."""

    if isinstance(nodes, Mapping):
        values = nodes.values()
    elif isinstance(nodes, Iterable) and not isinstance(nodes, (str, bytes)):
        values = nodes
    else:
        values = (nodes,)

    for node in values:
        if isinstance(node, Node):
            yield node
            continue
        if hasattr(node, "addr") and hasattr(node, "type"):
            yield cast(Node, node)


def iter_inventory_heater_metadata(
    inventory: Inventory | None,
    *,
    default_name_simple: Callable[[str], str] | None = None,
) -> Iterator[HeaterInventoryEntry]:
    """Yield ``(node_type, addr, name, node)`` tuples for heater nodes."""

    if not isinstance(inventory, Inventory):
        return

    default_factory = default_name_simple or (lambda addr: f"Heater {addr}")
    nodes_by_type, addresses_by_type, resolve_name = (
        heater_platform_details_from_inventory(
            inventory,
            default_name_simple=default_factory,
        )
    )

    for node_type, addresses in addresses_by_type.items():
        if not addresses:
            continue

        candidates = list(_iter_node_container(nodes_by_type.get(node_type, ())))
        if not candidates:
            continue

        for raw_addr in addresses:
            if not isinstance(raw_addr, str) or not raw_addr:
                continue

            matched: Node | None = None
            for index, node in enumerate(candidates):
                if node.addr == raw_addr:
                    matched = node
                    break

            if matched is None:
                continue

            yield (
                node_type,
                raw_addr,
                resolve_name(node_type, raw_addr),
                matched,
            )

ALLOWED_BOOST_MINUTES: Final[tuple[int, ...]] = tuple(range(60, 601, 60))
"""Valid boost durations (in minutes) supported by TermoWeb heaters."""

