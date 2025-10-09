"""Helpers describing a TermoWeb installation snapshot."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from .inventory import (
    Inventory,
    Node,
    build_node_inventory,
    heater_sample_subscription_targets as _heater_sample_subscription_targets,
)

# Backwards compatibility for legacy monkeypatches/tests relying on module-level helpers.
heater_sample_subscription_targets = _heater_sample_subscription_targets


class InstallationSnapshot:
    """Cache metadata derived from node inventory for reuse."""

    __slots__ = (
        "_compat_aliases",
        "_dev_id",
        "_inventory",
        "_name_map_cache",
        "_name_map_factories",
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
        self._inventory: Inventory | None = (
            Inventory(self._dev_id, self._raw_nodes, node_inventory)
            if node_inventory is not None
            else None
        )
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
        self._inventory = (
            Inventory(self._dev_id, self._raw_nodes, node_inventory)
            if node_inventory is not None
            else None
        )
        self._invalidate_caches()

    def _invalidate_caches(self) -> None:
        """Reset derived cache state so it can be lazily recomputed."""

        self._sample_address_map = None
        self._compat_aliases = None
        self._sample_targets = None
        self._name_map_cache.clear()
        self._name_map_factories.clear()

    def _ensure_inventory(self) -> Inventory:
        """Return the cached node inventory container, rebuilding if required."""

        if self._inventory is None:
            nodes = build_node_inventory(self._raw_nodes)
            self._inventory = Inventory(self._dev_id, self._raw_nodes, nodes)
        return self._inventory

    @property
    def inventory(self) -> tuple[Node, ...]:
        """Expose the normalised node inventory for the installation."""

        return self._ensure_inventory().nodes

    @property
    def nodes_by_type(self) -> dict[str, list[Node]]:
        """Return a mapping of node type to ``Node`` instances."""

        inventory = self._ensure_inventory()
        return inventory.nodes_by_type

    @property
    def explicit_heater_names(self) -> set[tuple[str, str]]:
        """Return node type/address pairs with explicit names."""

        inventory = self._ensure_inventory()
        return inventory.explicit_heater_names

    @property
    def heater_address_map(self) -> tuple[dict[str, list[str]], dict[str, set[str]]]:
        """Return forward and reverse heater address maps."""

        inventory = self._ensure_inventory()
        return inventory.heater_address_map

    def _ensure_sample_addresses(self) -> None:
        """Calculate canonical heater sample address data."""

        if self._sample_address_map is not None:
            return

        inventory = self._ensure_inventory()
        normalized_map, compat = inventory.heater_sample_address_map
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
            inventory = self._ensure_inventory()
            self._sample_targets = inventory.heater_sample_targets
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

        mapping = _build_heater_name_map(self._ensure_inventory().nodes, default_factory)
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
