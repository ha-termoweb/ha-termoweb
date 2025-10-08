"""Node helpers for TermoWeb devices."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
import logging
from typing import Any, cast

from .inventory import (
    HEATER_NODE_TYPES,
    Node,
    addresses_by_node_type,
    build_node_inventory,
    normalize_heater_addresses,
    normalize_node_addr,
    normalize_node_type,
)

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


def collect_heater_sample_addresses(
    record: Mapping[str, Any] | None,
    *,
    coordinator: Any | None = None,
) -> tuple[list[Node], dict[str, list[str]], dict[str, str]]:
    """Return inventory and canonical heater sample subscription addresses."""

    from .installation import InstallationSnapshot, ensure_snapshot  # noqa: PLC0415

    snapshot = ensure_snapshot(record)
    if isinstance(snapshot, InstallationSnapshot):
        inventory = snapshot.inventory
        normalized_map, compat = snapshot.heater_sample_address_map
    else:
        nodes_payload: Any | None = None
        cache_record: MutableMapping[str, Any] | None = None

        if isinstance(record, MutableMapping):
            cache_record = cast(MutableMapping[str, Any], record)
            nodes_payload = record.get("nodes")
        elif isinstance(record, Mapping):
            nodes_payload = record.get("nodes")

        inventory = ensure_node_inventory(cache_record or {}, nodes=nodes_payload)

        raw_map, _ = addresses_by_node_type(
            inventory,
            known_types=HEATER_NODE_TYPES,
        )
        addr_map: dict[str, list[str]] = {
            node_type: list(addresses)
            for node_type, addresses in raw_map.items()
            if node_type in HEATER_NODE_TYPES and addresses
        }

        normalized_map, compat = normalize_heater_addresses(addr_map)

    if (not normalized_map.get("htr")) and coordinator is not None:
        fallback: Iterable[Any] | None = None
        if hasattr(coordinator, "_addrs"):
            try:
                fallback = coordinator._addrs()  # type: ignore[attr-defined]  # noqa: SLF001
            except Exception:  # pragma: no cover - defensive  # noqa: BLE001
                fallback = None
        if fallback:
            normalised: list[str] = list(normalized_map.get("htr", []))
            seen = set(normalised)
            for candidate in fallback:
                addr = normalize_node_addr(candidate)
                if not addr or addr in seen:
                    continue
                seen.add(addr)
                normalised.append(addr)
            if normalised:
                normalized_map = dict(normalized_map)
                normalized_map["htr"] = normalised

    return list(inventory), normalized_map, compat
