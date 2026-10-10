"""Immutable gateway inventory: node lookups, address maps and sample targets."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
import typing
from typing import Any, cast

from .domain.ids import HEATER_NODE_TYPES
from .nodes import Node, normalize_node_addr, normalize_node_type

PrebuiltNode = Any


def _default_heater_name(addr: str) -> str:
    """Return the default fallback name for a heater address."""

    return f"Heater {addr}"


__all__ = [
    "HEATER_NODE_TYPES",
    "Inventory",
    "InventoryNodeMetadata",
    "InventorySnapshot",
    "addresses_by_node_type",
    "build_heater_address_map",
    "heater_sample_subscription_targets",
    "normalize_heater_addresses",
    "normalize_power_monitor_addresses",
    "power_monitor_sample_subscription_targets",
]


@dataclass(frozen=True, slots=True)
class InventoryNodeMetadata:
    """Describe a node and its resolved display metadata."""

    node_type: str
    node: Node
    addr: str
    name: str


@dataclass(frozen=True, slots=True)
class InventorySnapshot:
    """Represent a serialisable snapshot of inventory node metadata."""

    raw_count: int
    node_inventory: tuple[dict[str, str], ...]

    @property
    def filtered_count(self) -> int:
        """Return the number of nodes included in the snapshot."""

        return len(self.node_inventory)


@dataclass(frozen=True, slots=True)
class Inventory:
    """Represent immutable node inventory details."""

    _dev_id: str
    _nodes: tuple[PrebuiltNode, ...]
    _addresses_by_type_cache: dict[str, tuple[str, ...]] | None
    _nodes_by_type_cache: dict[str, tuple[PrebuiltNode, ...]] | None
    _explicit_name_pairs_cache: frozenset[tuple[str, str]] | None
    _heater_address_map_cache: (
        tuple[dict[str, tuple[str, ...]], dict[str, frozenset[str]]] | None
    )
    _heater_sample_address_cache: (
        tuple[dict[str, tuple[str, ...]], dict[str, str]] | None
    )
    _heater_sample_targets_cache: tuple[tuple[str, str], ...] | None
    _energy_sample_types_cache: frozenset[str] | None
    _power_monitor_address_map_cache: (
        tuple[dict[str, tuple[str, ...]], dict[str, frozenset[str]]] | None
    )
    _power_monitor_sample_address_cache: (
        tuple[dict[str, tuple[str, ...]], dict[str, str]] | None
    )
    _power_monitor_sample_targets_cache: tuple[tuple[str, str], ...] | None

    def __init__(
        self,
        dev_id: str,
        nodes: Iterable[PrebuiltNode],
    ) -> None:
        """Initialize the inventory container."""

        object.__setattr__(self, "_dev_id", dev_id)
        object.__setattr__(self, "_nodes", tuple(nodes))
        object.__setattr__(self, "_addresses_by_type_cache", None)
        object.__setattr__(self, "_nodes_by_type_cache", None)
        object.__setattr__(self, "_explicit_name_pairs_cache", None)
        object.__setattr__(self, "_heater_address_map_cache", None)
        object.__setattr__(self, "_heater_sample_address_cache", None)
        object.__setattr__(self, "_heater_sample_targets_cache", None)
        object.__setattr__(self, "_power_monitor_address_map_cache", None)
        object.__setattr__(self, "_power_monitor_sample_address_cache", None)
        object.__setattr__(self, "_power_monitor_sample_targets_cache", None)
        object.__setattr__(self, "_energy_sample_types_cache", None)

    @property
    def dev_id(self) -> str:
        """Get the device identifier."""

        return self._dev_id

    @property
    def nodes(self) -> tuple[PrebuiltNode, ...]:
        """Get the immutable tuple of node objects."""

        return self._nodes

    def has_node(self, node_type: Any, addr: Any) -> bool:
        """Return ``True`` when ``node_type``/``addr`` exists in the inventory."""

        normalized_type = normalize_node_type(
            node_type,
            use_default_when_falsey=True,
        )
        normalized_addr = normalize_node_addr(
            addr,
            use_default_when_falsey=True,
        )
        addresses = self._ensure_addresses_by_type_cache()
        return normalized_addr in addresses.get(normalized_type, ())

    def iter_known_entries(
        self, entries: Iterable[Any]
    ) -> Iterator[tuple[str, str, Mapping[str, typing.Any]]]:
        """Yield ``(node_type, addr, entry)`` for payload entries in the inventory."""

        for entry in entries:
            if not isinstance(entry, Mapping):
                continue

            node_type = normalize_node_type(
                entry.get("type"),
                use_default_when_falsey=True,
            )
            addr = normalize_node_addr(
                entry.get("addr"),
                use_default_when_falsey=True,
            )

            if not self.has_node(node_type, addr):
                continue

            yield node_type, addr, entry

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
            grouped[node.type].append(node)

        normalised = {key: tuple(values) for key, values in grouped.items()}
        object.__setattr__(self, "_nodes_by_type_cache", normalised)
        return normalised

    @property
    def nodes_by_type(self) -> dict[str, list[PrebuiltNode]]:
        """Return mapping of node type to node instances."""

        cached = self._ensure_nodes_by_type_cache()
        return {key: list(values) for key, values in cached.items()}

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

        for metadata in self.iter_nodes_metadata(
            node_types=HEATER_NODE_TYPES,
            default_name_simple=default_name_simple,
        ):
            yield metadata.node_type, metadata.node, metadata.addr, metadata.name

    def iter_nodes_metadata(
        self,
        *,
        node_types: Iterable[str] | None = None,
        default_name_simple: Callable[[str], str] | None = None,
    ) -> Iterator[InventoryNodeMetadata]:
        """Yield canonical node metadata for ``node_types``."""

        factory = default_name_simple or _default_heater_name

        if node_types is None:
            allowed_types: set[str] | None = None
        else:
            allowed_types = {
                normalize_node_type(candidate, use_default_when_falsey=True)
                for candidate in node_types
            }

        grouped = self._ensure_nodes_by_type_cache()

        for node_type, nodes in grouped.items():
            if allowed_types is not None and node_type not in allowed_types:
                continue

            for candidate in nodes:
                addr = normalize_node_addr(
                    getattr(candidate, "addr", None),
                    use_default_when_falsey=True,
                )
                if not addr:
                    continue

                if node_type in HEATER_NODE_TYPES:
                    name = self.resolve_heater_name(
                        node_type,
                        addr,
                        default_factory=factory,
                    )
                else:
                    name = candidate.name or f"{node_type.upper()} {addr}"

                yield InventoryNodeMetadata(
                    node_type=node_type,
                    node=cast("Node", candidate),
                    addr=addr,
                    name=name,
                )

    def snapshot(self) -> InventorySnapshot:
        """Return a serialisable snapshot of node metadata."""

        entries = [
            {
                "name": metadata.name,
                "addr": metadata.addr,
                "type": metadata.node_type,
            }
            for metadata in self.iter_nodes_metadata()
        ]

        entries.sort(key=lambda item: (item["addr"], item["type"]))
        return InventorySnapshot(
            raw_count=len(self._nodes),
            node_inventory=tuple(entries),
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
                {
                    addr: frozenset(node_types)
                    for addr, node_types in reverse_map.items()
                },
            )
            object.__setattr__(self, "_power_monitor_address_map_cache", cached)

        forward_cache, reverse_cache = cached
        return (
            {
                node_type: list(addresses)
                for node_type, addresses in forward_cache.items()
            },
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
        names_by_type = self.heater_names_by_type(factory)

        default_simple = factory(addr_norm)
        explicit_names = self.explicit_heater_names

        def _candidate(value: Any) -> str | None:
            if not isinstance(value, str) or not value:
                return None
            if (
                node_type_norm in ("acm", "thm")
                and value == default_simple
                and (node_type_norm, addr_norm) not in explicit_names
            ):
                return None
            return value

        per_type = names_by_type.get(node_type_norm, {})
        candidate = _candidate(per_type.get(addr_norm))
        if candidate:
            return candidate

        if node_type_norm == "acm":
            return f"Accumulator {addr_norm}"
        if node_type_norm == "thm":
            return f"Thermostat {addr_norm}"
        return factory(addr_norm)

    def _ensure_sample_addresses(
        self,
        *,
        cache_attr: str,
        forward_map_factory: Callable[[], tuple[dict[str, list[str]], Any]],
        normalizer: Callable[
            [Mapping[Any, Iterable[Any]] | Iterable[Any] | None],
            tuple[dict[str, list[str]], dict[str, str]],
        ],
    ) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
        """Return cached normalised sample addresses for the provided inputs."""

        cached = cast(
            "tuple[dict[str, tuple[str, ...]], dict[str, str]] | None",
            getattr(self, cache_attr),
        )
        if cached is None:
            forward_map, _ = forward_map_factory()
            normalized_map, compat = normalizer(forward_map)
            cached = (
                {
                    node_type: tuple(addresses)
                    for node_type, addresses in normalized_map.items()
                },
                dict(compat),
            )
            object.__setattr__(self, cache_attr, cached)
        return cached

    def _ensure_heater_sample_addresses(
        self,
    ) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
        """Return cached normalised heater address data for samples."""

        return self._ensure_sample_addresses(
            cache_attr="_heater_sample_address_cache",
            forward_map_factory=lambda: self.heater_address_map,
            normalizer=normalize_heater_addresses,
        )

    def _ensure_power_monitor_sample_addresses(
        self,
    ) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
        """Return cached normalised power monitor address data for samples."""

        return self._ensure_sample_addresses(
            cache_attr="_power_monitor_sample_address_cache",
            forward_map_factory=lambda: self.power_monitor_address_map,
            normalizer=normalize_power_monitor_addresses,
        )

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
            cached = self._validate_sample_targets(raw_targets)
            object.__setattr__(self, "_heater_sample_targets_cache", cached)
        return [tuple(pair) for pair in cached]

    @property
    def energy_sample_types(self) -> frozenset[str]:
        """Return canonical node types that expose energy or power samples."""

        cached = self._energy_sample_types_cache
        if cached is None:
            cached = frozenset(
                node_type
                for node_type, _ in (
                    *self.heater_sample_targets,
                    *self.power_monitor_sample_targets,
                )
            )
            object.__setattr__(self, "_energy_sample_types_cache", cached)
        return cached

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
            cached = self._validate_sample_targets(raw_targets)
            object.__setattr__(self, "_power_monitor_sample_targets_cache", cached)
        return [tuple(pair) for pair in cached]

    def _validate_sample_targets(
        self, targets: Iterable[Any]
    ) -> tuple[tuple[str, str], ...]:
        """Return deduplicated and sanitised sample target pairs."""

        validated: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for item in targets:
            if not isinstance(item, tuple) or len(item) != 2:
                continue
            node_type, addr = item
            if not isinstance(node_type, str) or not isinstance(addr, str):
                continue
            node_clean = node_type.strip()
            addr_clean = addr.strip()
            if not node_clean or not addr_clean:
                continue
            pair = (node_clean, addr_clean)
            if pair in seen:
                continue
            seen.add(pair)
            validated.append(pair)
        return tuple(validated)

    def sample_alias_map(
        self,
        *,
        base_aliases: Mapping[str, str] | None = None,
        include_types: Iterable[str] | None = None,
        restrict_to: Iterable[str] | None = None,
    ) -> dict[str, str]:
        """Return compatibility aliases for sample payload processing."""

        heater_forward, heater_aliases = self.heater_sample_address_map
        power_forward, power_aliases = self.power_monitor_sample_address_map

        alias_map: dict[str, str] = {**(base_aliases or {})}
        alias_map.update(heater_aliases)
        alias_map.update(power_aliases)
        for node_type in (*heater_forward, *power_forward, *(include_types or ())):
            alias_map.setdefault(node_type, node_type)

        if restrict_to is not None:
            allowed = set(restrict_to)
            alias_map = {
                key: value
                for key, value in alias_map.items()
                if key in allowed or value in allowed
            }

        return alias_map

    def heater_names_by_type(
        self, default_factory: Callable[[str], str] | None = None
    ) -> dict[str, dict[str, str]]:
        """Return heater names by type, using ``default_factory`` for unnamed nodes."""

        factory = default_factory or _default_heater_name
        by_type: dict[str, dict[str, str]] = {}
        for node_type, nodes in self._ensure_nodes_by_type_cache().items():
            if node_type not in HEATER_NODE_TYPES:
                continue
            for node in nodes:
                addr = normalize_node_addr(getattr(node, "addr", ""))
                if not addr:
                    continue
                raw_name = getattr(node, "name", None)
                if isinstance(raw_name, str) and raw_name.strip():
                    resolved = raw_name.strip()
                else:
                    resolved = factory(addr)
                by_type.setdefault(node_type, {})[addr] = resolved
        return by_type


def addresses_by_node_type(
    nodes: Iterable[Node],
    *,
    known_types: Iterable[str] | None = None,
) -> tuple[dict[str, list[str]], set[str]]:
    """Return mapping of node type to address list, tracking unknown types."""

    known: set[str] | None = None
    if known_types is not None:
        known = {
            normalize_node_type(node_type) for node_type in known_types if node_type
        }

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
) -> tuple[dict[str, list[str]], dict[str, set[str]]]:
    """Return mapping of heater node types to addresses and reverse lookup."""

    by_type_raw, _ = addresses_by_node_type(nodes, known_types=HEATER_NODE_TYPES)

    by_type: dict[str, list[str]] = {
        node_type: list(addresses)
        for node_type, addresses in by_type_raw.items()
        if node_type in HEATER_NODE_TYPES and addresses
    }

    reverse: dict[str, set[str]] = {}
    for node_type, addresses in by_type.items():
        for address in addresses:
            reverse.setdefault(address, set()).add(node_type)

    return by_type, reverse


def normalize_heater_addresses(
    addrs: Mapping[Any, Iterable[Any]] | None,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Return canonical heater addresses and compatibility aliases."""

    cleaned_map: dict[str, list[str]] = {}
    compat_aliases: dict[str, str] = {}

    sources = addrs.items() if addrs is not None else ()

    for raw_type, values in sources:
        node_type = normalize_node_type(
            raw_type,
            use_default_when_falsey=True,
        )
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

    for raw_type, values in sources:
        node_type = normalize_node_type(
            raw_type,
            use_default_when_falsey=True,
        )
        if node_type != "pmo":
            continue

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

    other_types = [
        node_type
        for node_type in sorted(normalized_map)
        if node_type not in {"htr", "thm"}
    ]
    order = ["htr", *other_types]
    return [
        (node_type, addr)
        for node_type in order
        for addr in normalized_map.get(node_type, []) or []
        if addr
    ]


def power_monitor_sample_subscription_targets(
    addrs: Mapping[Any, Iterable[Any]] | Iterable[Any] | None,
) -> list[tuple[str, str]]:
    """Return canonical power monitor sample subscription target pairs."""

    normalized_map, _ = normalize_power_monitor_addresses(addrs)
    return [("pmo", addr) for addr in normalized_map.get("pmo", []) if addr]
