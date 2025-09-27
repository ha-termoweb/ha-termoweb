from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
import math
from typing import Any, cast

from .nodes import Node, build_node_inventory

HEATER_NODE_TYPES: frozenset[str] = frozenset({"htr", "acm"})


def ensure_node_inventory(
    record: Mapping[str, Any], *, nodes: Any | None = None
) -> list[Node]:
    """Return cached node inventory, rebuilding and caching when missing."""

    cacheable = isinstance(record, MutableMapping)
    cached = record.get("node_inventory")
    if isinstance(cached, list) and cached:
        cached_nodes = [cast(Node, node) for node in cached if isinstance(node, Node)]
        if cached_nodes:
            if cacheable and len(cached_nodes) != len(cached):
                record["node_inventory"] = list(cached_nodes)
            return list(cached_nodes)

    payloads: list[Any] = []
    if nodes is not None:
        payloads.append(nodes)

    record_nodes = record.get("nodes")
    if record_nodes is not None and (not payloads or record_nodes is not payloads[0]):
        payloads.append(record_nodes)

    last_index = len(payloads) - 1
    for index, raw_nodes in enumerate(payloads):
        try:
            inventory = build_node_inventory(raw_nodes)
        except Exception:  # pragma: no cover - defensive
            inventory = []

        if cacheable and (inventory or index == last_index):
            record["node_inventory"] = list(inventory)

        if inventory:
            return list(inventory)

    if isinstance(cached, list):
        if cacheable and "node_inventory" not in record:
            record["node_inventory"] = []
        return []

    if cacheable and "node_inventory" not in record:
        record["node_inventory"] = []

    return []


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
