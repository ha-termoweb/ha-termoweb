from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from .nodes import Node

HEATER_NODE_TYPES: frozenset[str] = frozenset({"htr", "acm"})


def addresses_by_type(nodes: Iterable[Node], node_types: Iterable[str]) -> list[str]:
    """Return unique addresses for nodes whose ``type`` matches ``node_types``."""

    valid_types: set[str] = set()
    for node_type in node_types:
        if node_type is None:
            continue
        valid_types.add(str(node_type).strip().lower())
    result: list[str] = []
    seen: set[str] = set()

    if not valid_types:
        return result

    for node in nodes:
        node_type = str(getattr(node, "type", "")).strip().lower()
        if node_type not in valid_types:
            continue
        addr = str(getattr(node, "addr", "")).strip()
        if not addr or addr in seen:
            continue
        seen.add(addr)
        result.append(addr)

    return result


def addresses_by_node_type(
    nodes: Iterable[Node],
    *,
    known_types: Iterable[str] | None = None,
) -> tuple[dict[str, list[str]], set[str]]:
    """Return mapping of node type to address list, tracking unknown types."""

    known: set[str] | None = None
    if known_types is not None:
        known = {str(node_type).strip().lower() for node_type in known_types if node_type}

    result: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    unknown: set[str] = set()

    for node in nodes:
        node_type = str(getattr(node, "type", "")).strip().lower()
        if not node_type:
            continue
        addr = str(getattr(node, "addr", "")).strip()
        if not addr:
            continue
        type_seen = seen.setdefault(node_type, set())
        if addr in type_seen:
            continue
        type_seen.add(addr)
        result.setdefault(node_type, []).append(addr)
        if known is not None and node_type not in known:
            unknown.add(node_type)

    return result, unknown


def float_or_none(value: Any) -> float | None:
    """Return value as ``float`` if possible, else ``None``.

    Converts integers, floats, and numeric strings to ``float`` while safely
    handling ``None`` and non-numeric inputs.
    """
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            num = float(value)
        else:
            string_val = str(value).strip()
            if not string_val:
                return None
            num = float(string_val)
        return num if math.isfinite(num) else None
    except Exception:
        return None
