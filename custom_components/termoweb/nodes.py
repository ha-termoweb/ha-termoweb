"""Node helpers for TermoWeb devices."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
import logging
from typing import Any, cast

from .const import DOMAIN
from .inventory import (
    Node,
    build_node_inventory,
    normalize_node_addr,
    normalize_node_type,
)

_LOGGER = logging.getLogger(__name__)


HEATER_NODE_TYPES: frozenset[str] = frozenset({"htr", "acm"})


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


def parse_heater_energy_unique_id(unique_id: str) -> tuple[str, str, str] | None:
    """Parse a heater energy sensor unique ID into its components."""

    if not isinstance(unique_id, str):
        return None
    stripped = unique_id.strip()
    if not stripped or not stripped.startswith(f"{DOMAIN}:"):
        return None
    try:
        domain, dev, node, address, metric = stripped.split(":", 4)
    except ValueError:
        return None
    if domain != DOMAIN or metric != "energy":
        return None
    if not dev or not node or not address:
        return None
    return dev, node, address


def addresses_by_node_type(
    nodes: Iterable[Node],
    *,
    known_types: Iterable[str] | None = None,
) -> tuple[dict[str, list[str]], set[str]]:
    """Return mapping of node type to address list, tracking unknown types."""

    known: set[str] | None = None
    if known_types is not None:
        known = {normalize_node_type(node_type) for node_type in known_types if node_type}

    result: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    unknown: set[str] = set()

    for node in nodes:
        node_type = normalize_node_type(getattr(node, "type", ""))
        if not node_type:
            continue
        addr = normalize_node_addr(getattr(node, "addr", ""))
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


def build_heater_address_map(
    nodes: Iterable[Any],
    *,
    heater_types: Iterable[str] | None = None,
) -> tuple[dict[str, list[str]], dict[str, set[str]]]:
    """Return mapping of heater node types to addresses and reverse lookup."""

    allowed_types: set[str]
    if heater_types is None:
        allowed_types = set(HEATER_NODE_TYPES)
    else:
        allowed_types = {
            normalize_node_type(node_type, use_default_when_falsey=True)
            for node_type in heater_types
            if normalize_node_type(node_type, use_default_when_falsey=True)
        }  # pragma: no cover - exercised indirectly in integration

    if not allowed_types:
        return {}, {}  # pragma: no cover - defensive

    by_type_raw, _ = addresses_by_node_type(
        nodes,
        known_types=allowed_types,
    )  # pragma: no cover - exercised via higher level integration tests

    by_type: dict[str, list[str]] = {
        node_type: list(addresses)
        for node_type, addresses in by_type_raw.items()
        if node_type in allowed_types and addresses
    }

    reverse: dict[str, set[str]] = {}
    for node_type, addresses in by_type.items():
        for address in addresses:
            reverse.setdefault(address, set()).add(node_type)

    return by_type, reverse


def normalize_heater_addresses(
    addrs: Iterable[Any] | Mapping[Any, Iterable[Any]] | None,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Return canonical heater addresses and compatibility aliases."""

    cleaned_map: dict[str, list[str]] = {}
    compat_aliases: dict[str, str] = {}

    if addrs is None:
        sources: Iterable[tuple[Any, Iterable[Any] | Any]] = []
    elif isinstance(addrs, Mapping):
        sources = addrs.items()
    else:
        sources = [("htr", addrs)]

    for raw_type, values in sources:
        node_type = normalize_node_type(
            raw_type,
            use_default_when_falsey=True,
        )
        if not node_type:
            continue

        alias_target: str | None = None
        if node_type in {"heater", "heaters", "htr"}:
            alias_target = "htr"
        if alias_target is not None and node_type != alias_target:
            compat_aliases[node_type] = alias_target
            node_type = alias_target

        if node_type not in HEATER_NODE_TYPES:
            continue

        if isinstance(values, str) or not isinstance(values, Iterable):
            candidates = [values]
        else:
            candidates = list(values)

        bucket = cleaned_map.setdefault(node_type, [])
        seen: set[str] = set(bucket)
        for candidate in candidates:
            addr = normalize_node_addr(
                candidate,
                use_default_when_falsey=True,
            )
            if not addr or addr in seen:
                continue
            seen.add(addr)
            bucket.append(addr)

    cleaned_map.setdefault("htr", [])
    compat_aliases["htr"] = "htr"

    return cleaned_map, compat_aliases


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


def heater_sample_subscription_targets(
    addrs: Mapping[Any, Iterable[Any]] | Iterable[Any] | None,
) -> list[tuple[str, str]]:
    """Return canonical heater sample subscription target pairs."""

    normalized_map, _ = normalize_heater_addresses(addrs)
    if not any(normalized_map.values()):
        return []

    other_types = sorted(node_type for node_type in normalized_map if node_type != "htr")
    order = ["htr", *other_types]
    return [
        (node_type, addr)
        for node_type in order
        for addr in normalized_map.get(node_type, []) or []
    ]
