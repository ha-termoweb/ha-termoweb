"""Helpers describing a TermoWeb installation snapshot."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any

from .nodes import (
    Node,
    build_heater_address_map,
    build_node_inventory,
    heater_sample_subscription_targets,
    normalize_heater_addresses,
    normalize_node_addr,
    normalize_node_type,
)


class InstallationSnapshot:
    """Cache metadata derived from node inventory for reuse."""

    __slots__ = (
        "_compat_aliases",
        "_dev_id",
        "_explicit_name_pairs",
        "_heater_address_map",
        "_heater_address_reverse",
        "_inventory",
        "_name_map_cache",
        "_name_map_factories",
        "_nodes_by_type",
        "_raw_nodes",
        "_sample_address_map",
        "_sample_targets",
    )

    def __init__(
        self,
        *,
        dev_id: str,
        raw_nodes: Any,
        node_inventory: Iterable[Node] | None = None,
    ) -> None:
        """Initialise the snapshot with backend payload metadata."""

        self._dev_id = str(dev_id)
        self._raw_nodes = raw_nodes
        self._inventory = list(node_inventory) if node_inventory is not None else None
        self._nodes_by_type: dict[str, list[Node]] | None = None
        self._explicit_name_pairs: set[tuple[str, str]] | None = None
        self._heater_address_map: dict[str, list[str]] | None = None
        self._heater_address_reverse: dict[str, set[str]] | None = None
        self._sample_address_map: dict[str, list[str]] | None = None
        self._compat_aliases: dict[str, str] | None = None
        self._sample_targets: list[tuple[str, str]] | None = None
        self._name_map_cache: dict[int, dict[Any, Any]] = {}
        self._name_map_factories: dict[int, Callable[[str], str]] = {}

    @property
    def dev_id(self) -> str:
        """Return the backend device identifier for the snapshot."""

        return self._dev_id

    @property
    def raw_nodes(self) -> Any:
        """Return the raw nodes payload associated with the snapshot."""

        return self._raw_nodes

    def update_nodes(
        self, raw_nodes: Any, *, node_inventory: Iterable[Node] | None = None
    ) -> None:
        """Update cached payload data and invalidate derived caches."""

        self._raw_nodes = raw_nodes
        self._inventory = list(node_inventory) if node_inventory is not None else None
        self._invalidate_caches()

    def _invalidate_caches(self) -> None:
        """Reset derived cache state so it can be lazily recomputed."""

        self._nodes_by_type = None
        self._explicit_name_pairs = None
        self._heater_address_map = None
        self._heater_address_reverse = None
        self._sample_address_map = None
        self._compat_aliases = None
        self._sample_targets = None
        self._name_map_cache.clear()
        self._name_map_factories.clear()

    def _ensure_inventory(self) -> list[Node]:
        """Return the cached node inventory, rebuilding if required."""

        if self._inventory is None:
            self._inventory = build_node_inventory(self._raw_nodes)
        return self._inventory

    @property
    def inventory(self) -> list[Node]:
        """Expose the normalised node inventory for the installation."""

        return self._ensure_inventory()

    def _ensure_nodes_by_type(self) -> None:
        """Populate caches describing heater nodes by type."""

        if self._nodes_by_type is not None:
            return

        nodes_by_type: dict[str, list[Node]] = defaultdict(list)
        explicit_names: set[tuple[str, str]] = set()
        for node in self._ensure_inventory():
            node_type = normalize_node_type(getattr(node, "type", ""))
            if not node_type:
                continue
            nodes_by_type[node_type].append(node)
            addr = normalize_node_addr(getattr(node, "addr", ""))
            if addr and getattr(node, "name", "").strip():
                explicit_names.add((node_type, addr))

        self._nodes_by_type = {k: list(v) for k, v in nodes_by_type.items()}
        self._explicit_name_pairs = explicit_names

    @property
    def nodes_by_type(self) -> dict[str, list[Node]]:
        """Return a mapping of node type to ``Node`` instances."""

        self._ensure_nodes_by_type()
        return dict(self._nodes_by_type or {})

    @property
    def explicit_heater_names(self) -> set[tuple[str, str]]:
        """Return node type/address pairs with explicit names."""

        self._ensure_nodes_by_type()
        return set(self._explicit_name_pairs or set())

    def _ensure_address_maps(self) -> None:
        """Populate cached heater address maps from the inventory."""

        if self._heater_address_map is not None:
            return

        forward, reverse = build_heater_address_map(self._ensure_inventory())
        self._heater_address_map = {k: list(v) for k, v in forward.items()}
        self._heater_address_reverse = {
            addr: set(node_types) for addr, node_types in reverse.items()
        }

    @property
    def heater_address_map(self) -> tuple[dict[str, list[str]], dict[str, set[str]]]:
        """Return forward and reverse heater address maps."""

        self._ensure_address_maps()
        forward = {k: list(v) for k, v in (self._heater_address_map or {}).items()}
        reverse = {
            addr: set(types)
            for addr, types in (self._heater_address_reverse or {}).items()
        }
        return forward, reverse

    def _ensure_sample_addresses(self) -> None:
        """Calculate canonical heater sample address data."""

        if self._sample_address_map is not None:
            return

        forward, _ = self.heater_address_map
        normalized_map, compat = normalize_heater_addresses(forward)
        self._sample_address_map = {
            node_type: list(addrs) for node_type, addrs in normalized_map.items()
        }
        self._compat_aliases = dict(compat)

    @property
    def heater_sample_address_map(self) -> tuple[dict[str, list[str]], dict[str, str]]:
        """Return normalised heater addresses suitable for samples."""

        self._ensure_sample_addresses()
        return (
            {k: list(v) for k, v in (self._sample_address_map or {}).items()},
            dict(self._compat_aliases or {}),
        )

    @property
    def heater_sample_targets(self) -> list[tuple[str, str]]:
        """Return ordered ``(node_type, addr)`` sample subscription targets."""

        if self._sample_targets is None:
            normalized_map, _ = self.heater_sample_address_map
            self._sample_targets = heater_sample_subscription_targets(normalized_map)
        return list(self._sample_targets)

    def heater_name_map(
        self, default_factory: Callable[[str], str]
    ) -> dict[Any, Any]:
        """Return cached heater name mapping for ``default_factory``."""

        key = id(default_factory)
        cached = self._name_map_cache.get(key)
        if cached is not None and self._name_map_factories.get(key) is default_factory:
            return cached

        from .heater import (  # noqa: PLC0415
            build_heater_name_map as _build_heater_name_map,
        )

        mapping = _build_heater_name_map(self._ensure_inventory(), default_factory)
        self._name_map_cache[key] = mapping
        self._name_map_factories[key] = default_factory
        return mapping


def ensure_snapshot(record: Any) -> InstallationSnapshot | None:
    """Return the installation snapshot stored in ``record`` when present."""

    if isinstance(record, dict):
        snapshot = record.get("snapshot")
        if isinstance(snapshot, InstallationSnapshot):
            return snapshot
    return None
