"""Helpers for parsing boost metadata shared across TermoWeb modules."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import logging
import math
from typing import Any, Final

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

    heater_nodes = getattr(inventory, "heater_nodes", ())
    node_lookup: dict[tuple[str, str], Node] = {}
    for node in heater_nodes:
        node_type = normalize_node_type(
            getattr(node, "type", None),
            use_default_when_falsey=True,
        )
        addr = normalize_node_addr(
            getattr(node, "addr", None),
            use_default_when_falsey=True,
        )
        if not node_type or not addr:
            continue
        key = (node_type, addr)
        if key not in node_lookup:
            node_lookup[key] = node

    if not node_lookup:
        for node_type_raw, nodes in nodes_by_type.items():
            node_type = normalize_node_type(
                node_type_raw,
                use_default_when_falsey=True,
            )
            if not node_type:
                continue
            candidates: Iterable[Any]
            if isinstance(nodes, Mapping):
                candidates = nodes.values()
            elif isinstance(nodes, Iterable) and not isinstance(nodes, (str, bytes)):
                candidates = nodes
            else:
                candidates = (nodes,)
            for node in candidates:
                addr = normalize_node_addr(
                    getattr(node, "addr", None),
                    use_default_when_falsey=True,
                )
                if not addr:
                    continue
                key = (node_type, addr)
                node_lookup.setdefault(key, node)

    for node_type_raw, addresses in addresses_by_type.items():
        node_type = normalize_node_type(
            node_type_raw,
            use_default_when_falsey=True,
        )
        if not node_type or not addresses:
            continue

        for addr_raw in addresses:
            addr = normalize_node_addr(
                addr_raw,
                use_default_when_falsey=True,
            )
            if not addr:
                continue
            node = node_lookup.get((node_type, addr))
            if node is None:
                continue
            name = resolve_name(node_type, addr)
            yield InventoryHeaterMetadata(
                node_type=node_type,
                addr=addr,
                name=name,
                node=node,
                supports_boost=supports_boost(node),
            )

ALLOWED_BOOST_MINUTES: Final[tuple[int, ...]] = tuple(range(60, 601, 60))
"""Valid boost durations (in minutes) supported by TermoWeb heaters."""

