"""Inventory helpers and node model abstractions for TermoWeb."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping
from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Any, cast

from .const import DOMAIN

RawNodePayload = Any
PrebuiltNode = Any

_NODE_SECTION_IGNORE_KEYS = frozenset(
    {"dev_id", "name", "raw", "connected", "nodes", "nodes_by_type"}
)

_SNAPSHOT_NAME_CANDIDATE_KEYS = (
    "name",
    "label",
    "title",
    "display_name",
    "device_name",
    "friendly_name",
    "heater_name",
    "alias",
    "room",
)


_NODE_TYPE_ALIASES: dict[str, str] = {
    "power_monitor": "pmo",
    "power_monitors": "pmo",
}


def _default_heater_name(addr: str) -> str:
    """Return the default fallback name for a heater address."""

    return f"Heater {addr}"


__all__ = [
    "HEATER_NODE_TYPES",
    "NODE_CLASS_BY_TYPE",
    "AccumulatorNode",
    "HeaterNode",
    "Inventory",
    "InventoryResolution",
    "Node",
    "NodeDescriptor",
    "PowerMonitorNode",
    "ThermostatNode",
    "_normalize_node_identifier",
    "addresses_by_node_type",
    "boostable_accumulator_details_for_entry",
    "build_heater_address_map",
    "build_node_inventory",
    "heater_platform_details_from_inventory",
    "heater_sample_subscription_targets",
    "normalize_heater_addresses",
    "normalize_node_addr",
    "normalize_node_type",
    "normalize_power_monitor_addresses",
    "parse_heater_energy_unique_id",
    "power_monitor_sample_subscription_targets",
    "resolve_record_inventory",
]


_LOGGER = logging.getLogger(__name__)


HEATER_NODE_TYPES: frozenset[str] = frozenset({"htr", "acm"})


if TYPE_CHECKING:
    from .heater import HeaterPlatformDetails


@dataclass(frozen=True, slots=True)
class InventoryResolution:
    """Describe the origin of an ``Inventory`` resolution attempt."""

    inventory: Inventory | None
    source: str
    raw_count: int
    filtered_count: int


@dataclass(frozen=True, slots=True)
class Inventory:
    """Represent immutable node inventory details."""

    _dev_id: str
    _payload: RawNodePayload
    _nodes: tuple[PrebuiltNode, ...]
    _addresses_by_type_cache: dict[str, tuple[str, ...]] | None
    _nodes_by_type_cache: dict[str, tuple[PrebuiltNode, ...]] | None
    _heater_nodes_cache: tuple[PrebuiltNode, ...] | None
    _explicit_name_pairs_cache: frozenset[tuple[str, str]] | None
    _heater_address_map_cache: (
        tuple[dict[str, tuple[str, ...]], dict[str, frozenset[str]]] | None
    )
    _heater_sample_address_cache: (
        tuple[dict[str, tuple[str, ...]], dict[str, str]] | None
    )
    _heater_sample_targets_cache: tuple[tuple[str, str], ...] | None
    _power_monitor_address_map_cache: (
        tuple[dict[str, tuple[str, ...]], dict[str, frozenset[str]]] | None
    )
    _power_monitor_sample_address_cache: (
        tuple[dict[str, tuple[str, ...]], dict[str, str]] | None
    )
    _power_monitor_sample_targets_cache: tuple[tuple[str, str], ...] | None
    _heater_name_map_cache: dict[int, dict[Any, Any]]
    _heater_name_map_factories: dict[int, Callable[[str], str]]

    def __init__(
        self,
        dev_id: str,
        payload: RawNodePayload,
        nodes: Iterable[PrebuiltNode],
    ) -> None:
        """Initialize the inventory container."""

        object.__setattr__(self, "_dev_id", dev_id)
        object.__setattr__(self, "_payload", payload)
        object.__setattr__(self, "_nodes", tuple(nodes))
        object.__setattr__(self, "_addresses_by_type_cache", None)
        object.__setattr__(self, "_nodes_by_type_cache", None)
        object.__setattr__(self, "_heater_nodes_cache", None)
        object.__setattr__(self, "_explicit_name_pairs_cache", None)
        object.__setattr__(self, "_heater_address_map_cache", None)
        object.__setattr__(self, "_heater_sample_address_cache", None)
        object.__setattr__(self, "_heater_sample_targets_cache", None)
        object.__setattr__(self, "_power_monitor_address_map_cache", None)
        object.__setattr__(self, "_power_monitor_sample_address_cache", None)
        object.__setattr__(self, "_power_monitor_sample_targets_cache", None)
        object.__setattr__(self, "_heater_name_map_cache", {})
        object.__setattr__(self, "_heater_name_map_factories", {})

    @property
    def dev_id(self) -> str:
        """Get the device identifier."""

        return self._dev_id

    @property
    def payload(self) -> RawNodePayload:
        """Get the raw node payload."""

        return self._payload

    @property
    def nodes(self) -> tuple[PrebuiltNode, ...]:
        """Get the immutable tuple of node objects."""

        return self._nodes

    def _ensure_addresses_by_type_cache(self) -> dict[str, tuple[str, ...]]:
        """Return cached node addresses grouped by normalised type."""

        cached = self._addresses_by_type_cache
        if cached is not None:
            return cached

        grouped, _ = addresses_by_node_type(self._nodes)
        normalized = {key: tuple(values) for key, values in grouped.items()}
        object.__setattr__(self, "_addresses_by_type_cache", normalized)
        return normalized

    @property
    def addresses_by_type(self) -> dict[str, list[str]]:
        """Return mapping of node type to normalised addresses."""

        cached = self._ensure_addresses_by_type_cache()
        return {key: list(values) for key, values in cached.items()}

    def _ensure_nodes_by_type_cache(self) -> dict[str, tuple[PrebuiltNode, ...]]:
        """Return cached node groupings keyed by normalised type."""

        cached = self._nodes_by_type_cache
        if cached is not None:
            return cached

        grouped: dict[str, list[PrebuiltNode]] = defaultdict(list)
        for node in self._nodes:
            node_type = normalize_node_type(getattr(node, "type", ""))
            if not node_type:
                continue
            grouped[node_type].append(node)

        normalised = {key: tuple(values) for key, values in grouped.items()}
        object.__setattr__(self, "_nodes_by_type_cache", normalised)
        return normalised

    @property
    def nodes_by_type(self) -> dict[str, list[PrebuiltNode]]:
        """Return mapping of node type to node instances."""

        cached = self._ensure_nodes_by_type_cache()
        return {key: list(values) for key, values in cached.items()}

    @property
    def heater_nodes(self) -> tuple[PrebuiltNode, ...]:
        """Return tuple of nodes belonging to heater-compatible types."""

        cached = self._heater_nodes_cache
        if cached is None:
            grouped = self._ensure_nodes_by_type_cache()
            heater_list: list[PrebuiltNode] = []
            for node_type in HEATER_NODE_TYPES:
                heater_list.extend(grouped.get(node_type, ()))
            cached = tuple(heater_list)
            object.__setattr__(self, "_heater_nodes_cache", cached)
        return cached

    @property
    def explicit_heater_names(self) -> set[tuple[str, str]]:
        """Return node type/address pairs that have explicit user-defined names."""

        cached = self._explicit_name_pairs_cache
        if cached is None:
            pairs: set[tuple[str, str]] = set()
            grouped = self._ensure_nodes_by_type_cache()
            for node_type, nodes in grouped.items():
                if node_type not in HEATER_NODE_TYPES:
                    continue
                for node in nodes:
                    addr = normalize_node_addr(getattr(node, "addr", ""))
                    if not addr:
                        continue
                    raw_name = getattr(node, "name", "")
                    if isinstance(raw_name, str) and raw_name.strip():
                        pairs.add((node_type, addr))
            cached = frozenset(pairs)
            object.__setattr__(self, "_explicit_name_pairs_cache", cached)
        return set(cached)

    @property
    def heater_address_map(self) -> tuple[dict[str, list[str]], dict[str, set[str]]]:
        """Return forward and reverse heater address mappings."""

        cached = self._heater_address_map_cache
        if cached is None:
            forward_raw, reverse_raw = build_heater_address_map(self._nodes)
            filtered_forward = {
                node_type: tuple(addresses)
                for node_type, addresses in forward_raw.items()
                if node_type in HEATER_NODE_TYPES and addresses
            }
            filtered_reverse = {
                addr: frozenset(node_types)
                for addr, node_types in reverse_raw.items()
                if node_types
            }
            cached = (filtered_forward, filtered_reverse)
            object.__setattr__(self, "_heater_address_map_cache", cached)

        forward_cache, reverse_cache = cached
        return (
            {
                node_type: list(addresses)
                for node_type, addresses in forward_cache.items()
            },
            {addr: set(node_types) for addr, node_types in reverse_cache.items()},
        )

    def iter_heater_platform_metadata(
        self,
        default_name_simple: Callable[[str], str],
    ) -> Iterator[tuple[str, Node, str, str]]:
        """Yield heater metadata derived from cached inventory details."""

        forward_map, _ = self.heater_address_map
        if not forward_map:
            return

        grouped = self._ensure_nodes_by_type_cache()

        for node_type, addresses in forward_map.items():
            if not addresses:
                continue

            nodes = grouped.get(node_type, ())
            if not nodes:
                continue

            for raw_addr in addresses:
                if not isinstance(raw_addr, str) or not raw_addr:
                    continue

                matched: Any | None = None
                for candidate in nodes:
                    addr = normalize_node_addr(
                        getattr(candidate, "addr", None),
                        use_default_when_falsey=True,
                    )
                    if addr == raw_addr:
                        matched = candidate
                        break

                if matched is None:
                    continue

                yield (
                    node_type,
                    cast("Node", matched),
                    raw_addr,
                    self.resolve_heater_name(
                        node_type,
                        raw_addr,
                        default_factory=default_name_simple,
                    ),
                )

    @property
    def power_monitor_address_map(
        self,
    ) -> tuple[dict[str, list[str]], dict[str, set[str]]]:
        """Return forward and reverse power monitor address mappings."""

        cached = self._power_monitor_address_map_cache
        if cached is None:
            forward_raw, _ = addresses_by_node_type(
                self._nodes,
                known_types=("pmo",),
            )
            filtered_forward: dict[str, tuple[str, ...]] = {}
            reverse_map: dict[str, set[str]] = {}
            for node_type, addresses in forward_raw.items():
                if node_type != "pmo":
                    continue
                normalized = tuple(addr for addr in addresses if addr)
                filtered_forward[node_type] = normalized
                for addr in normalized:
                    reverse_map.setdefault(addr, set()).add(node_type)
            filtered_forward.setdefault("pmo", ())
            cached = (
                filtered_forward,
                {addr: frozenset(node_types) for addr, node_types in reverse_map.items()},
            )
            object.__setattr__(self, "_power_monitor_address_map_cache", cached)

        forward_cache, reverse_cache = cached
        return (
            {node_type: list(addresses) for node_type, addresses in forward_cache.items()},
            {addr: set(node_types) for addr, node_types in reverse_cache.items()},
        )

    def resolve_heater_name(
        self,
        node_type: str,
        addr: str,
        *,
        default_factory: Callable[[str], str] | None = None,
    ) -> str:
        """Return the friendly name for ``(node_type, addr)``."""

        node_type_norm = normalize_node_type(
            node_type,
            use_default_when_falsey=True,
        )
        addr_norm = normalize_node_addr(
            addr,
            use_default_when_falsey=True,
        )

        factory = default_factory or _default_heater_name
        if not node_type_norm or not addr_norm:
            return factory(normalize_node_addr(addr) or str(addr))

        name_map = self.heater_name_map(factory)
        names_by_type: Mapping[str, Any]
        legacy_names: Mapping[str, Any]
        if isinstance(name_map, Mapping):
            names_by_type = name_map.get("by_type", {})
            if not isinstance(names_by_type, Mapping):
                names_by_type = {}
            legacy_names = name_map.get("htr", {})
            if not isinstance(legacy_names, Mapping):
                legacy_names = {}
            pair_lookup = name_map
        else:
            names_by_type = {}
            legacy_names = {}
            pair_lookup = {}

        default_simple = factory(addr_norm)
        explicit_names = self.explicit_heater_names

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

        per_type_raw = names_by_type.get(node_type_norm, {})
        per_type = per_type_raw if isinstance(per_type_raw, Mapping) else {}

        for candidate_value in (
            per_type.get(addr_norm),
            pair_lookup.get((node_type_norm, addr_norm)),
            legacy_names.get(addr_norm),
        ):
            candidate = _candidate(candidate_value)
            if candidate:
                return candidate

        if node_type_norm == "acm":
            return f"Accumulator {addr_norm}"
        return factory(addr_norm)

    def _ensure_heater_sample_addresses(
        self,
    ) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
        """Return cached normalised heater address data for samples."""

        cached = self._heater_sample_address_cache
        if cached is None:
            forward_map, _ = self.heater_address_map
            normalized_map, compat = normalize_heater_addresses(forward_map)
            cached = (
                {
                    node_type: tuple(addresses)
                    for node_type, addresses in normalized_map.items()
                },
                dict(compat),
            )
            object.__setattr__(self, "_heater_sample_address_cache", cached)
        return cached

    def _ensure_power_monitor_sample_addresses(
        self,
    ) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
        """Return cached normalised power monitor address data for samples."""

        cached = self._power_monitor_sample_address_cache
        if cached is None:
            forward_map, _ = self.power_monitor_address_map
            normalized_map, compat = normalize_power_monitor_addresses(forward_map)
            cached = (
                {
                    node_type: tuple(addresses)
                    for node_type, addresses in normalized_map.items()
                },
                dict(compat),
            )
            object.__setattr__(self, "_power_monitor_sample_address_cache", cached)
        return cached

    @property
    def heater_sample_address_map(
        self,
    ) -> tuple[dict[str, list[str]], dict[str, str]]:
        """Return normalised heater addresses and compatibility aliases."""

        forward_cache, compat_cache = self._ensure_heater_sample_addresses()
        return (
            {
                node_type: list(addresses)
                for node_type, addresses in forward_cache.items()
            },
            dict(compat_cache),
        )

    @property
    def heater_sample_targets(self) -> list[tuple[str, str]]:
        """Return ordered ``(node_type, addr)`` sample subscription targets."""

        cached = self._heater_sample_targets_cache
        if cached is None:
            normalized_map, _ = self._ensure_heater_sample_addresses()
            raw_targets = heater_sample_subscription_targets(normalized_map)
            validated: list[tuple[str, str]] = []
            for item in raw_targets:
                if not isinstance(item, tuple) or len(item) != 2:
                    continue
                node_type, addr = item
                if not isinstance(node_type, str) or not isinstance(addr, str):
                    continue
                node_clean = node_type.strip()
                addr_clean = addr.strip()
                if node_clean and addr_clean:
                    validated.append((node_clean, addr_clean))
            cached = tuple(validated)
            object.__setattr__(self, "_heater_sample_targets_cache", cached)
        return [tuple(pair) for pair in cached]

    @property
    def power_monitor_sample_address_map(
        self,
    ) -> tuple[dict[str, list[str]], dict[str, str]]:
        """Return normalised power monitor addresses and compatibility aliases."""

        forward_cache, compat_cache = self._ensure_power_monitor_sample_addresses()
        return (
            {
                node_type: list(addresses)
                for node_type, addresses in forward_cache.items()
            },
            dict(compat_cache),
        )

    @property
    def power_monitor_sample_targets(self) -> list[tuple[str, str]]:
        """Return ordered power monitor sample subscription targets."""

        cached = self._power_monitor_sample_targets_cache
        if cached is None:
            normalized_map, _ = self._ensure_power_monitor_sample_addresses()
            raw_targets = power_monitor_sample_subscription_targets(normalized_map)
            validated: list[tuple[str, str]] = []
            for item in raw_targets:
                if not isinstance(item, tuple) or len(item) != 2:
                    continue
                node_type, addr = item
                if not isinstance(node_type, str) or not isinstance(addr, str):
                    continue
                node_clean = node_type.strip()
                addr_clean = addr.strip()
                if node_clean and addr_clean:
                    validated.append((node_clean, addr_clean))
            cached = tuple(validated)
            object.__setattr__(self, "_power_monitor_sample_targets_cache", cached)
        return [tuple(pair) for pair in cached]

    def heater_name_map(
        self, default_factory: Callable[[str], str] | None = None
    ) -> dict[Any, Any]:
        """Return cached heater name mapping for ``default_factory``."""

        factory = default_factory or _default_heater_name
        key = id(factory)
        cached = self._heater_name_map_cache.get(key)
        if cached is not None and self._heater_name_map_factories.get(key) is factory:
            return cached

        nodes: tuple[PrebuiltNode, ...] = self._nodes
        sanitized_nodes: tuple[Any, ...]
        if nodes:
            sanitized: list[Any] = []
            for node in nodes:
                as_dict = getattr(node, "as_dict", None)
                if callable(as_dict):
                    try:
                        payload = as_dict()
                    except Exception:  # pragma: no cover - defensive  # noqa: BLE001
                        payload = None
                    else:
                        if isinstance(payload, Mapping):
                            sanitized.append(dict(payload))
                            continue

                sanitized.append(
                    {
                        "type": getattr(node, "type", None),
                        "addr": getattr(node, "addr", None),
                        "name": getattr(node, "name", None),
                    }
                )
            sanitized_nodes = tuple(sanitized)
        else:
            sanitized_nodes = nodes

        def _node_value(candidate: Any, key: str) -> Any:
            """Return attribute ``key`` for ``candidate`` regardless of type."""

            if isinstance(candidate, Mapping):
                return candidate.get(key)
            return getattr(candidate, key, None)

        by_type: dict[str, dict[str, str]] = {}
        by_node: dict[tuple[str, str], str] = {}

        for node in sanitized_nodes:
            node_type = normalize_node_type(_node_value(node, "type"))
            if node_type not in HEATER_NODE_TYPES:
                continue
            addr = normalize_node_addr(_node_value(node, "addr"))
            if not addr or addr.lower() == "none":
                continue
            raw_name = _node_value(node, "name")
            if isinstance(raw_name, str) and raw_name.strip():
                resolved = raw_name.strip()
            else:
                resolved = factory(addr)
            by_node[(node_type, addr)] = resolved
            bucket = by_type.setdefault(node_type, {})
            bucket[addr] = resolved

        mapping: dict[Any, Any] = {"htr": dict(by_type.get("htr", {}))}
        if by_type:
            mapping["by_type"] = {k: dict(v) for k, v in by_type.items()}
        mapping.update(by_node)

        self._heater_name_map_cache[key] = mapping
        self._heater_name_map_factories[key] = factory
        return mapping


def _normalize_node_iterable(
    nodes: Iterable[Any] | None,
) -> tuple[tuple[Node, ...], int] | None:
    """Return filtered node tuples and raw length for ``nodes``."""

    if nodes is None:
        return None

    raw_list = list(nodes)
    filtered: list[Node] = []
    for node in raw_list:
        if not hasattr(node, "as_dict"):
            continue
        node_type = normalize_node_type(getattr(node, "type", ""))
        addr = normalize_node_addr(getattr(node, "addr", ""))
        if not node_type or not addr:
            continue
        filtered.append(cast(Node, node))

    return (tuple(filtered), len(raw_list))


def resolve_record_inventory(
    record: Mapping[str, Any] | None,
    *,
    dev_id: str | None = None,
    nodes_payload: Any | None = None,
    node_list: Iterable[Node] | None = None,
) -> InventoryResolution:
    """Return the ``Inventory`` for ``record`` and describe its origin."""

    mapping: Mapping[str, Any] | None = record if isinstance(record, Mapping) else None
    mutable: MutableMapping[str, Any] | None
    if isinstance(record, MutableMapping):
        mutable = record
    else:
        mutable = None

    resolved_dev_id = str(dev_id or "")

    if mapping is not None:
        record_dev = mapping.get("dev_id")
        if not resolved_dev_id:
            if isinstance(record_dev, str) and record_dev.strip():
                resolved_dev_id = record_dev.strip()
            elif record_dev not in (None, ""):
                resolved_dev_id = str(record_dev)

    payload_candidate = nodes_payload

    nodes_candidate: Iterable[Any] | None = node_list

    def _node_pairs(items: Iterable[Any]) -> set[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()
        for node in items:
            node_type = normalize_node_type(getattr(node, "type", ""))
            addr = normalize_node_addr(getattr(node, "addr", ""))
            if not node_type or not addr:
                continue
            pairs.add((node_type, addr))
        return pairs

    node_info = _normalize_node_iterable(nodes_candidate)

    candidate = mapping.get("inventory") if mapping is not None else None
    if isinstance(candidate, Inventory):
        if node_info is None:
            node_count = len(candidate.nodes)
            return InventoryResolution(candidate, "inventory", node_count, node_count)
        nodes_tuple, _ = node_info
        candidate_pairs = _node_pairs(candidate.nodes)
        override_pairs = _node_pairs(nodes_tuple)
        if candidate_pairs == override_pairs and len(candidate.nodes) == len(nodes_tuple):
            node_count = len(candidate.nodes)
            return InventoryResolution(candidate, "inventory", node_count, node_count)

    def _finalize(
        nodes_tuple: tuple[Node, ...],
        *,
        source: str,
        raw_count: int,
        payload: Any | None,
        override_dev_id: str | None = None,
    ) -> InventoryResolution | None:
        nonlocal resolved_dev_id

        effective_dev_id = override_dev_id or resolved_dev_id
        effective_dev_id = str(effective_dev_id or "")
        payload_obj = payload if payload is not None else {}

        try:
            container = Inventory(effective_dev_id, payload_obj, nodes_tuple)
        except Exception:  # pragma: no cover - defensive  # noqa: BLE001
            _LOGGER.debug("Failed to construct inventory from %s data", source, exc_info=True)
            return None

        if mutable is not None:
            mutable["inventory"] = container

        resolved_dev_id = container.dev_id
        return InventoryResolution(container, source, raw_count, len(container.nodes))

    if node_info is not None:
        nodes_tuple, raw_count = node_info
        result = _finalize(
            nodes_tuple,
            source="node_list",
            raw_count=raw_count,
            payload=payload_candidate,
        )
        if result is not None:
            return result

    snapshot = mapping.get("snapshot") if mapping is not None else None
    if snapshot is not None:
        snapshot_nodes = getattr(snapshot, "inventory", None)
        node_info = _normalize_node_iterable(snapshot_nodes)
        if node_info is not None:
            nodes_tuple, raw_count = node_info
            snapshot_payload = payload_candidate
            if snapshot_payload is None:
                snapshot_payload = getattr(snapshot, "raw_nodes", None)
            snapshot_dev_id = getattr(snapshot, "dev_id", None)
            result = _finalize(
                nodes_tuple,
                source="snapshot",
                raw_count=raw_count,
                payload=snapshot_payload,
                override_dev_id=str(snapshot_dev_id) if snapshot_dev_id is not None else None,
            )
            if result is not None:
                return result

    payload_for_build = payload_candidate

    if payload_for_build is not None:
        try:
            built_nodes = build_node_inventory(payload_for_build)
        except Exception:  # pragma: no cover - defensive  # noqa: BLE001
            _LOGGER.debug("Failed to normalise raw nodes for inventory", exc_info=True)
        else:
            result = _finalize(
                tuple(built_nodes),
                source="raw_nodes",
                raw_count=len(built_nodes),
                payload=payload_for_build,
            )
            if result is not None:
                return result

    return InventoryResolution(None, "fallback", 0, 0)


def _normalize_node_identifier(
    value: Any,
    *,
    default: str = "",
    use_default_when_falsey: bool = False,
    lowercase: bool,
) -> str:
    """Return ``value`` as a normalised node identifier string."""

    raw = value
    if use_default_when_falsey and not raw:
        raw = default

    try:
        normalized = str(raw).strip()
    except Exception:  # pragma: no cover - defensive  # noqa: BLE001
        normalized = ""
    else:
        if lowercase:
            normalized = normalized.lower()

    if normalized:
        return normalized

    if default and not use_default_when_falsey:
        try:
            normalized_default = str(default).strip()
        except Exception:  # pragma: no cover - defensive  # noqa: BLE001
            return ""
        if lowercase:
            normalized_default = normalized_default.lower()
        return normalized_default

    return ""


def normalize_node_type(
    value: Any,
    *,
    default: str = "",
    use_default_when_falsey: bool = False,
) -> str:
    """Return ``value`` as a normalised node type string."""

    normalized = _normalize_node_identifier(
        value,
        default=default,
        use_default_when_falsey=use_default_when_falsey,
        lowercase=True,
    )
    return _NODE_TYPE_ALIASES.get(normalized, normalized)


def normalize_node_addr(
    value: Any,
    *,
    default: str = "",
    use_default_when_falsey: bool = False,
) -> str:
    """Return ``value`` as a normalised node address string."""

    return _normalize_node_identifier(
        value,
        default=default,
        use_default_when_falsey=use_default_when_falsey,
        lowercase=False,
    )


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


def heater_platform_details_from_inventory(
    inventory: Inventory,
    *,
    default_name_simple: Callable[[str], str],
) -> tuple[
    dict[str, list[Node]],
    dict[str, list[str]],
    Callable[[str, str], str],
]:
    """Return heater platform metadata derived from ``inventory``."""

    nodes_by_type = inventory.nodes_by_type
    explicit_names = inventory.explicit_heater_names
    forward_map, _ = inventory.heater_address_map
    addrs_by_type = {
        node_type: list(forward_map.get(node_type, []))
        for node_type in HEATER_NODE_TYPES
    }

    def resolve_name(node_type: str, addr: str) -> str:
        """Resolve friendly names for heater nodes."""

        return inventory.resolve_heater_name(
            node_type,
            addr,
            default_factory=default_name_simple,
        )

    return nodes_by_type, addrs_by_type, resolve_name


def boostable_accumulator_details_for_entry(
    entry_data: Mapping[str, Any] | None,
    *,
    default_name_simple: Callable[[str], str],
    platform_name: str,
    logger: logging.Logger | None = None,
    accumulators_only: bool = True,
) -> tuple[HeaterPlatformDetails, list[tuple[str, str, str]]]:
    """Return boostable accumulator metadata for a config entry."""

    from .heater import (
        heater_platform_details_for_entry,
        iter_boostable_heater_nodes,
        log_skipped_nodes,
    )

    details = heater_platform_details_for_entry(
        entry_data,
        default_name_simple=default_name_simple,
    )

    metadata: list[tuple[str, str, str]] = [
        (node_type, addr_str, base_name)
        for node_type, _node, addr_str, base_name in iter_boostable_heater_nodes(
            details,
            accumulators_only=accumulators_only,
        )
    ]

    log_skipped_nodes(platform_name, details, logger=logger)

    return details, metadata


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


def normalize_power_monitor_addresses(
    addrs: Iterable[Any] | Mapping[Any, Iterable[Any]] | None,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Return canonical power monitor addresses and compatibility aliases."""

    if addrs is None:
        sources: Iterable[tuple[Any, Iterable[Any] | Any]] = []
    elif isinstance(addrs, Mapping):
        sources = addrs.items()
    else:
        sources = [("pmo", addrs)]

    cleaned_map: dict[str, list[str]] = {}
    compat_aliases: dict[str, str] = {"pmo": "pmo"}
    for alias, target in _NODE_TYPE_ALIASES.items():
        if target == "pmo":
            compat_aliases.setdefault(alias, "pmo")

    for raw_type, values in sources:
        raw_type_normalized = _normalize_node_identifier(
            raw_type,
            use_default_when_falsey=True,
            lowercase=True,
        )
        node_type = normalize_node_type(
            raw_type,
            use_default_when_falsey=True,
        )
        if not node_type or node_type != "pmo":
            continue

        if raw_type_normalized and raw_type_normalized != "pmo":
            compat_aliases[raw_type_normalized] = "pmo"

        if isinstance(values, str) or not isinstance(values, Iterable):
            candidates = [values]
        else:
            candidates = list(values)

        bucket = cleaned_map.setdefault("pmo", [])
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

    cleaned_map.setdefault("pmo", [])

    return cleaned_map, compat_aliases


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


def power_monitor_sample_subscription_targets(
    addrs: Mapping[Any, Iterable[Any]] | Iterable[Any] | None,
) -> list[tuple[str, str]]:
    """Return canonical power monitor sample subscription target pairs."""

    normalized_map, _ = normalize_power_monitor_addresses(addrs)
    return [("pmo", addr) for addr in normalized_map.get("pmo", []) if addr]


class Node:
    """Base representation of a TermoWeb node."""

    __slots__ = ("_node_name", "addr", "type")
    NODE_TYPE = ""

    def __init__(
        self,
        *,
        name: str | None,
        addr: str | int,
        node_type: str | None = None,
    ) -> None:
        """Initialise a node with normalised metadata."""

        resolved_type = normalize_node_type(
            node_type,
            default=self.NODE_TYPE,
            use_default_when_falsey=True,
        )
        if not resolved_type:
            msg = "node_type must be provided"
            raise ValueError(msg)

        addr_str = normalize_node_addr(addr)
        if not addr_str:
            msg = "addr must be provided"
            raise ValueError(msg)

        self.addr = addr_str
        self.type = resolved_type
        self._node_name = ""
        self.name = name if name is not None else ""

    @property
    def name(self) -> str:
        """Return the friendly name for the node."""

        attr_name = getattr(self, "_attr_name", None)
        if isinstance(attr_name, str) and attr_name.strip():
            return attr_name
        return self._node_name

    @name.setter
    def name(self, value: str | None) -> None:
        """Update the stored friendly name for the node."""
        cleaned = str(value or "").strip()
        self._node_name = cleaned
        if hasattr(self, "_attr_name"):
            self._attr_name = cleaned

    def as_dict(self) -> dict[str, Any]:
        """Return a serialisable snapshot of core node metadata."""

        return {
            "name": self.name,
            "addr": self.addr,
            "type": self.type,
        }


NodeDescriptor = Node | tuple[str, str | int]


class HeaterNode(Node):
    """Heater node (type ``htr``)."""

    __slots__ = ()
    NODE_TYPE = "htr"

    def __init__(self, *, name: str | None, addr: str | int) -> None:
        """Initialise a heater node."""

        super().__init__(name=name, addr=addr)

    def supports_boost(self) -> bool:
        """Return whether the node natively exposes boost/runback control."""

        return False


class AccumulatorNode(HeaterNode):
    """Storage heater / accumulator node (type ``acm``)."""

    __slots__ = ()
    NODE_TYPE = "acm"

    def supports_boost(self) -> bool:
        """Return whether the accumulator exposes boost/runback."""

        return True


class PowerMonitorNode(Node):
    """Power monitor node (type ``pmo``)."""

    __slots__ = ()
    NODE_TYPE = "pmo"

    def __init__(self, *, name: str | None, addr: str | int) -> None:
        """Initialise a power monitor node."""

        super().__init__(name=name, addr=addr)

    def power_level(self) -> float:
        """Return the reported power level (stub)."""

        raise NotImplementedError

    def default_name(self) -> str:
        """Return the fallback friendly name for the power monitor."""

        return self.name or f"Power Monitor {self.addr}"

    def sample_target(self) -> tuple[str, str]:
        """Return the canonical ``(node_type, addr)`` sample target."""

        return (self.type, self.addr)


class ThermostatNode(Node):
    """Thermostat node (type ``thm``)."""

    __slots__ = ()
    NODE_TYPE = "thm"

    def __init__(self, *, name: str | None, addr: str | int) -> None:
        """Initialise a thermostat node."""

        super().__init__(name=name, addr=addr)

    def capabilities(self) -> dict[str, Any]:
        """Return thermostat capabilities (stub)."""

        raise NotImplementedError


NODE_CLASS_BY_TYPE: dict[str, type[Node]] = {
    HeaterNode.NODE_TYPE: HeaterNode,
    AccumulatorNode.NODE_TYPE: AccumulatorNode,
    PowerMonitorNode.NODE_TYPE: PowerMonitorNode,
    ThermostatNode.NODE_TYPE: ThermostatNode,
}


def _existing_nodes_map(source: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Return a mapping of node type sections extracted from ``source``."""

    if not isinstance(source, Mapping):
        return {}

    sections: dict[str, dict[str, Any]] = {}

    raw_existing = source.get("nodes_by_type")
    if isinstance(raw_existing, Mapping):
        for node_type, section in raw_existing.items():
            if isinstance(section, Mapping):
                sections[node_type] = dict(section)

    for key, value in source.items():
        if key in _NODE_SECTION_IGNORE_KEYS:
            continue
        if isinstance(value, Mapping):
            sections.setdefault(key, dict(value))

    return sections


def _iter_snapshot_sections(
    sections: Mapping[str, Any],
    seen: set[tuple[str, str]],
) -> Iterable[dict[str, Any]]:
    """Yield node payloads derived from snapshot-style ``sections``."""

    for node_type, payload in sections.items():
        if not isinstance(node_type, str):
            continue
        normalized_type = node_type.strip()
        if not normalized_type or not isinstance(payload, Mapping):
            continue
        for entry in _iter_snapshot_section(normalized_type, payload):
            addr = entry.get("addr")
            if not isinstance(addr, str):
                continue
            key = (normalized_type, addr)
            if key in seen:
                continue
            seen.add(key)
            yield entry


def _iter_snapshot_section(
    node_type: str, section: Mapping[str, Any]
) -> Iterable[dict[str, Any]]:
    """Yield node dictionaries for a single node type section."""

    addresses = _collect_snapshot_addresses(section)
    for addr in sorted(addresses):
        entry: dict[str, Any] = {"type": node_type, "addr": addr}
        name = _extract_snapshot_name(addresses[addr])
        if name:
            entry["name"] = name
        yield entry


def _collect_snapshot_addresses(
    section: Mapping[str, Any]
) -> dict[str, list[Mapping[str, Any]]]:
    """Return mapping of addresses to candidate payloads from ``section``."""

    addresses: dict[str, list[Mapping[str, Any]]] = {}

    addrs = section.get("addrs")
    if isinstance(addrs, (list, tuple, set)):
        for candidate in addrs:
            addr = normalize_node_addr(candidate)
            if addr:
                addresses.setdefault(addr, [])

    for value in section.values():
        if not isinstance(value, Mapping):
            continue
        for addr_key, payload in value.items():
            addr = normalize_node_addr(addr_key)
            if not addr:
                continue
            bucket = addresses.setdefault(addr, [])
            if isinstance(payload, Mapping):
                bucket.append(payload)
            elif isinstance(payload, str):
                bucket.append({"name": payload})

    return addresses


def _extract_snapshot_name(payloads: Iterable[Mapping[str, Any]]) -> str:
    """Return best candidate name extracted from ``payloads``."""

    queue = [payload for payload in payloads if isinstance(payload, Mapping)]
    seen: set[int] = set()

    while queue:
        payload = queue.pop(0)
        payload_id = id(payload)
        if payload_id in seen:
            continue
        seen.add(payload_id)

        for key in _SNAPSHOT_NAME_CANDIDATE_KEYS:
            value = payload.get(key)
            if isinstance(value, str):
                candidate = value.strip()
                if candidate:
                    return candidate

        queue.extend(
            nested for nested in payload.values() if isinstance(nested, Mapping)
        )

    return ""


def _iter_node_payload(raw_nodes: Any) -> Iterable[dict[str, Any]]:
    """Yield node dictionaries from a payload returned by the API."""

    if isinstance(raw_nodes, dict):
        node_list = raw_nodes.get("nodes")
        if isinstance(node_list, list):
            for entry in node_list:
                if isinstance(entry, dict):
                    yield entry
            return

        seen: set[tuple[str, str]] = set()

        if isinstance(node_list, dict):
            yield from _iter_snapshot_sections(node_list, seen)

        sections = _existing_nodes_map(raw_nodes)
        if sections:
            yield from _iter_snapshot_sections(sections, seen)
            if seen:
                return

    if isinstance(raw_nodes, list):
        for entry in raw_nodes:
            if isinstance(entry, dict):
                yield entry


def _resolve_node_class(node_type: str) -> type[Node]:
    """Return the most appropriate node class for ``node_type``."""

    return NODE_CLASS_BY_TYPE.get(node_type, Node)


def _normalise_with_fallback(
    normalizer: Callable[..., str],
    *candidates: Any,
) -> str:
    """Return the first non-empty normalised value from ``candidates``."""

    for candidate in candidates:
        normalized = normalizer(candidate, use_default_when_falsey=True)
        if normalized:
            return normalized
    return ""


def build_node_inventory(raw_nodes: Any) -> list[Node]:
    """Return a list of :class:`Node` instances for the provided payload."""

    inventory: list[Node] = []
    for index, payload in enumerate(_iter_node_payload(raw_nodes)):
        node_type = _normalise_with_fallback(
            normalize_node_type,
            payload.get("type"),
            payload.get("node_type"),
        )
        if not node_type:
            _LOGGER.debug(
                "Skipping node with missing type at index %s: %s",
                index,
                payload,
            )
            continue

        name = payload.get("name") or payload.get("title") or payload.get("label")
        addr = _normalise_with_fallback(
            normalize_node_addr,
            payload.get("addr"),
            payload.get("address"),
        )

        node_cls = _resolve_node_class(node_type)
        if node_cls is Node:
            _LOGGER.debug("Unsupported node type '%s' encountered", node_type)

        try:
            if node_cls is Node:
                node = node_cls(name=name, addr=addr, node_type=node_type)
            else:
                node = node_cls(name=name, addr=addr)
        except (TypeError, ValueError) as err:  # pragma: no cover - defensive
            _LOGGER.debug(
                "Failed to initialise node %s at index %s: %s",
                payload,
                index,
                err,
            )
            continue

        inventory.append(node)

    return inventory
