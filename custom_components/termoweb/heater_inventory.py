"""Shared helpers for deriving heater inventory metadata."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from .inventory import (
    Node,
    build_heater_address_map,
    normalize_node_addr,
    normalize_node_type,
)
from .nodes import HEATER_NODE_TYPES


@dataclass(slots=True)
class HeaterInventoryDetails:
    """Describe cached metadata derived from heater nodes."""

    nodes_by_type: dict[str, list[Node]]
    explicit_name_pairs: set[tuple[str, str]]
    address_map: dict[str, list[str]]
    reverse_address_map: dict[str, set[str]]


def build_heater_inventory_details(
    nodes: Iterable[Node],
) -> HeaterInventoryDetails:
    """Return derived heater metadata for ``nodes``."""

    inventory = list(nodes)

    nodes_by_type: dict[str, list[Node]] = defaultdict(list)
    explicit_names: set[tuple[str, str]] = set()

    for node in inventory:
        node_type = normalize_node_type(getattr(node, "type", ""))
        if not node_type:
            continue
        nodes_by_type[node_type].append(node)
        addr = normalize_node_addr(getattr(node, "addr", ""))
        if addr and getattr(node, "name", "").strip():
            explicit_names.add((node_type, addr))

    forward, reverse = build_heater_address_map(inventory)

    filtered_forward = {
        node_type: list(addresses)
        for node_type, addresses in forward.items()
        if node_type in HEATER_NODE_TYPES and addresses
    }

    filtered_reverse = {
        addr: set(node_types)
        for addr, node_types in reverse.items()
        if node_types
    }

    return HeaterInventoryDetails(
        nodes_by_type={k: list(v) for k, v in nodes_by_type.items()},
        explicit_name_pairs=explicit_names,
        address_map=filtered_forward,
        reverse_address_map=filtered_reverse,
    )

