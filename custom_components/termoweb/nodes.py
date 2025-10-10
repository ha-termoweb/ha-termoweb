"""Node helpers for TermoWeb devices."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
import logging
from typing import Any, cast

from .inventory import (
    HEATER_NODE_TYPES as _HEATER_NODE_TYPES,
    Inventory,
    Node,
    build_node_inventory,
    normalize_node_addr,
    normalize_node_type,
)

HEATER_NODE_TYPES = _HEATER_NODE_TYPES

_LOGGER = logging.getLogger(__name__)


def ensure_node_inventory(
    record: Mapping[str, Any], *, nodes: Any | None = None
) -> list[Node]:
    """Return cached node inventory, rebuilding and caching when missing."""

    mutable_record: MutableMapping[str, Any] | None = None
    if isinstance(record, MutableMapping):
        mutable_record = record

    cacheable = mutable_record is not None
    cached = record.get("node_inventory")
    if isinstance(cached, list) and cached:
        cached_nodes: list[Node] = []
        for node in cached:
            if not hasattr(node, "as_dict"):
                continue
            node_type = normalize_node_type(getattr(node, "type", ""))
            addr = normalize_node_addr(getattr(node, "addr", ""))
            if not node_type or not addr:
                continue
            cached_nodes.append(cast(Node, node))
        if cached_nodes:
            if cacheable and len(cached_nodes) != len(cached) and mutable_record is not None:
                mutable_record["node_inventory"] = list(cached_nodes)
            return list(cached_nodes)

    inventory_container = record.get("inventory")
    if isinstance(inventory_container, Inventory):
        inventory_nodes = list(inventory_container.nodes)
        if inventory_nodes:
            if cacheable and mutable_record is not None:
                mutable_record.setdefault("node_inventory", list(inventory_nodes))
            return list(inventory_nodes)

    payloads: list[Any] = []
    if nodes is not None:
        payloads.append(nodes)

    last_index = len(payloads) - 1
    for index, raw_nodes in enumerate(payloads):
        try:
            inventory = build_node_inventory(raw_nodes)
        except Exception:  # pragma: no cover - defensive  # noqa: BLE001
            inventory = []

        if cacheable and (inventory or index == last_index) and mutable_record is not None:
            mutable_record["node_inventory"] = list(inventory)

        if inventory:
            return list(inventory)

    if isinstance(cached, list):
        if cacheable and "node_inventory" not in record and mutable_record is not None:
            mutable_record["node_inventory"] = []
        return []

    if cacheable and "node_inventory" not in record and mutable_record is not None:
        mutable_record["node_inventory"] = []

    return []
