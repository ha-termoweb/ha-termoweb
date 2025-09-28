"""Coordinator helpers for the TermoWeb integration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import timedelta
import logging
import time
from typing import Any

from aiohttp import ClientError
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .api import BackendAuthError, BackendRateLimitError, RESTClient
from .const import HTR_ENERGY_UPDATE_INTERVAL, MIN_POLL_INTERVAL
from .nodes import Node, build_node_inventory
from .utils import build_heater_address_map, float_or_none, normalize_heater_addresses

_LOGGER = logging.getLogger(__name__)

# How many heater settings to fetch per device per cycle (keep gentle)
HTR_SETTINGS_PER_CYCLE = 1


class StateCoordinator(
    DataUpdateCoordinator[dict[str, dict[str, Any]]]
):  # dev_id -> per-device data
    """Polls TermoWeb and exposes a per-device dict used by platforms."""

    def __init__(
        self,
        hass: HomeAssistant,
        client: RESTClient,
        base_interval: int,
        dev_id: str,
        device: dict[str, Any],
        nodes: dict[str, Any],
        node_inventory: list[Node] | None = None,
    ) -> None:
        """Initialize the TermoWeb device coordinator."""
        super().__init__(
            hass,
            logger=_LOGGER,
            name="termoweb",
            update_interval=timedelta(seconds=max(base_interval, MIN_POLL_INTERVAL)),
        )
        self.client = client
        self._base_interval = max(base_interval, MIN_POLL_INTERVAL)
        self._backoff = 0  # seconds
        self._rr_index: dict[str, int] = {}
        self._dev_id = dev_id
        self._device = device or {}
        self._nodes = nodes or {}
        self._node_inventory: list[Node] = list(node_inventory or [])
        self._nodes_by_type: dict[str, list[str]] = {}
        self._addr_lookup: dict[str, set[str]] = {}
        self._refresh_node_cache()

    def _set_inventory_from_nodes(
        self,
        nodes: Mapping[str, Any] | None,
        provided: Iterable[Node] | None = None,
    ) -> list[Node]:
        """Populate the cached inventory from ``provided`` or ``nodes``."""

        if provided is not None:
            inventory = list(provided)
        elif nodes:
            try:
                inventory = build_node_inventory(nodes)
            except ValueError as err:  # pragma: no cover - defensive
                _LOGGER.debug(
                    "Failed to build node inventory for %s: %s",
                    self._dev_id,
                    err,
                    exc_info=err,
                )
                inventory = []
        else:
            inventory = []

        self._node_inventory = inventory
        return inventory

    def _ensure_inventory(self) -> list[Node]:
        """Return cached node inventory, rebuilding when necessary."""
        if not self._node_inventory:
            self._set_inventory_from_nodes(self._nodes)
        self._refresh_node_cache()
        return self._node_inventory

    def _refresh_node_cache(self) -> None:
        """Rebuild cached mappings of node types to addresses."""

        inventory = self._node_inventory
        forward_map, reverse_map = build_heater_address_map(inventory)

        nodes_by_type: dict[str, list[str]] = {
            node_type: list(addresses) for node_type, addresses in forward_map.items()
        }
        addr_lookup: dict[str, set[str]] = {
            str(addr): set(node_types) for addr, node_types in reverse_map.items()
        }

        for node in inventory:
            node_type = str(getattr(node, "type", "")).strip().lower()
            addr = str(getattr(node, "addr", "")).strip()
            if not node_type or not addr:
                continue
            bucket = nodes_by_type.setdefault(node_type, [])
            nodes_by_type[node_type] = list(dict.fromkeys([*bucket, addr]))
            addr_lookup.setdefault(addr, set()).add(node_type)

        self._nodes_by_type = nodes_by_type
        self._addr_lookup = addr_lookup

    def _merge_address_payload(
        self, payload: Mapping[str, Iterable[Any]] | None
    ) -> None:
        """Merge normalized heater addresses into cached lookups."""

        if not payload:
            return

        cleaned_map, _ = normalize_heater_addresses(payload)

        for node_type, addrs in cleaned_map.items():
            if not addrs:
                continue
            bucket = self._nodes_by_type.setdefault(node_type, [])
            for addr in addrs:
                if addr not in bucket:
                    bucket.append(addr)
                existing = self._addr_lookup.get(addr)
                if isinstance(existing, set):
                    lookup = existing
                else:
                    lookup = set()
                    if isinstance(existing, str) and existing:
                        lookup.add(existing)
                    self._addr_lookup[addr] = lookup
                lookup.add(node_type)

    def _register_node_address(self, node_type: str, addr: str) -> None:
        """Add ``addr`` to the cached map for ``node_type`` if missing."""

        self._merge_address_payload({node_type: [addr]})

    @staticmethod
    def _normalise_type_section(
        node_type: str,
        section: Any,
        default_addrs: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Return a standard mapping for a node type section."""

        addrs: list[str] = []
        if default_addrs is not None:
            addrs = [str(addr).strip() for addr in default_addrs if str(addr).strip()]

        settings: dict[str, Any] = {}

        if isinstance(section, dict):
            raw_addrs = section.get("addrs")
            if isinstance(raw_addrs, Iterable) and not isinstance(
                raw_addrs, (str, bytes)
            ):
                addrs = [str(item).strip() for item in raw_addrs if str(item).strip()]
            raw_settings = section.get("settings")
            if isinstance(raw_settings, dict):
                settings = {
                    str(addr).strip(): data
                    for addr, data in raw_settings.items()
                    if str(addr).strip()
                }

        # Ensure addrs contains any addresses present in settings
        for addr in settings:
            if addr not in addrs:
                addrs.append(addr)

        return {"addrs": addrs, "settings": settings}

    def update_nodes(
        self,
        nodes: dict[str, Any],
        node_inventory: list[Node] | None = None,
    ) -> None:
        """Update cached node payload and inventory."""

        self._nodes = nodes or {}
        self._set_inventory_from_nodes(self._nodes, provided=node_inventory)
        self._refresh_node_cache()

    async def async_refresh_heater(self, node: str | tuple[str, str]) -> None:
        """Refresh settings for a specific node and push the update to listeners."""

        dev_id = self._dev_id
        success = False
        if isinstance(node, tuple) and len(node) == 2:
            raw_type, raw_addr = node
            node_type = str(raw_type or "").strip().lower()
            addr = str(raw_addr or "").strip()
        else:
            node_type = ""
            addr = str(node or "").strip()

        _LOGGER.info(
            "Refreshing heater settings for device %s node_type=%s addr=%s",
            dev_id,
            node_type or "<auto>",
            addr,
        )
        resolved_type = node_type or "htr"

        try:
            if not addr:
                _LOGGER.error(
                    "Cannot refresh heater settings without an address for device %s",
                    dev_id,
                )
                return

            self._ensure_inventory()
            reverse = {
                address: set(types) for address, types in self._addr_lookup.items()
            }
            addr_types = reverse.get(addr)
            resolved_type = node_type or (
                next(iter(addr_types)) if addr_types else "htr"
            )

            payload = await self.client.get_node_settings(dev_id, (resolved_type, addr))

            if not isinstance(payload, dict):
                _LOGGER.debug(
                    "Ignoring unexpected heater settings payload for device %s %s %s: %s",
                    dev_id,
                    resolved_type,
                    addr,
                    payload,
                )
                return

            current = self.data or {}
            new_data: dict[str, dict[str, Any]] = dict(current)
            dev_data = dict(new_data.get(dev_id) or {})

            if not dev_data:
                dev_name = (self._device.get("name") or f"Device {dev_id}").strip()
                dev_data = {
                    "dev_id": dev_id,
                    "name": dev_name,
                    "raw": self._device,
                    "connected": True,
                    "nodes": self._nodes,
                }
            else:
                dev_data.setdefault("dev_id", dev_id)
                if "name" not in dev_data:
                    dev_data["name"] = (
                        self._device.get("name") or f"Device {dev_id}"
                    ).strip()
                dev_data.setdefault("raw", self._device)
                dev_data.setdefault("nodes", self._nodes)
                dev_data.setdefault("connected", True)

            node_type = resolved_type

            existing_nodes = {}
            raw_existing = dev_data.get("nodes_by_type")
            if isinstance(raw_existing, dict):
                existing_nodes.update(raw_existing)

            for key, value in dev_data.items():
                if key in {
                    "dev_id",
                    "name",
                    "raw",
                    "connected",
                    "nodes",
                    "nodes_by_type",
                }:
                    continue
                if isinstance(value, dict):
                    existing_nodes.setdefault(key, value)

            cache_map = dict(self._nodes_by_type)
            if node_type not in cache_map:
                cache_map[node_type] = []

            all_types = set(existing_nodes) | set(cache_map)
            nodes_by_type: dict[str, dict[str, Any]] = {}
            for n_type in all_types:
                default_addrs = cache_map.get(n_type, [])
                section = existing_nodes.get(n_type)
                nodes_by_type[n_type] = self._normalise_type_section(
                    n_type, section, default_addrs
                )

            bucket = nodes_by_type.setdefault(
                node_type,
                self._normalise_type_section(
                    node_type, {}, cache_map.get(node_type, [])
                ),
            )
            if addr not in bucket["addrs"]:
                bucket["addrs"].append(addr)
            bucket.setdefault("settings", {})[addr] = payload

            self._register_node_address(node_type, addr)
            cached_addrs = self._nodes_by_type.get(node_type, [])
            merged_addrs: list[str] = []
            for candidate in [*cached_addrs, *bucket["addrs"]]:
                if candidate not in merged_addrs:
                    merged_addrs.append(candidate)
            bucket["addrs"] = merged_addrs

            for n_type, cached in self._nodes_by_type.items():
                section = nodes_by_type.setdefault(
                    n_type,
                    self._normalise_type_section(n_type, {}, cached),
                )
                merged: list[str] = []
                for candidate in [*cached, *section["addrs"]]:
                    if candidate not in merged:
                        merged.append(candidate)
                section["addrs"] = merged

            dev_data["nodes"] = self._nodes
            dev_data["nodes_by_type"] = {
                n_type: {
                    "addrs": list(section["addrs"]),
                    "settings": dict(section["settings"]),
                }
                for n_type, section in nodes_by_type.items()
            }

            for n_type, section in dev_data["nodes_by_type"].items():
                dev_data[n_type] = section

            heater_section = dev_data["nodes_by_type"].get("htr")
            if heater_section is None:
                heater_section = self._normalise_type_section(
                    "htr", {}, self._nodes_by_type.get("htr", [])
                )
                dev_data["nodes_by_type"]["htr"] = heater_section
            dev_data["htr"] = heater_section
            new_data[dev_id] = dev_data
            self.async_set_updated_data(new_data)
            success = True

        except TimeoutError as err:
            _LOGGER.error(
                "Timeout refreshing heater settings for device %s %s %s",
                dev_id,
                node_type or resolved_type,
                addr,
                exc_info=err,
            )
        except (ClientError, BackendRateLimitError, BackendAuthError) as err:
            _LOGGER.error(
                "Failed to refresh heater settings for device %s %s %s: %s",
                dev_id,
                node_type or resolved_type,
                addr,
                err,
                exc_info=err,
            )
        finally:
            _LOGGER.info(
                "Finished heater settings refresh for device %s %s %s (success=%s)",
                dev_id,
                node_type or resolved_type,
                addr,
                success,
            )

    async def _async_update_data(self) -> dict[str, dict[str, Any]]:
        """Fetch heater settings for a subset of addresses on each poll."""
        dev_id = self._dev_id
        self._ensure_inventory()
        addr_map = dict(self._nodes_by_type)
        reverse = {
            address: set(types) for address, types in self._addr_lookup.items()
        }
        addrs = [addr for addrs in addr_map.values() for addr in addrs]
        try:
            prev_dev = (self.data or {}).get(dev_id, {})
            prev_by_type: dict[str, dict[str, Any]] = {}

            existing_nodes = prev_dev.get("nodes_by_type")
            if isinstance(existing_nodes, dict):
                for node_type, section in existing_nodes.items():
                    normalised = self._normalise_type_section(
                        node_type,
                        section,
                        addr_map.get(node_type, []),
                    )
                    if normalised["settings"]:
                        prev_by_type[node_type] = dict(normalised["settings"])

            for key, value in prev_dev.items():
                if key in {
                    "dev_id",
                    "name",
                    "raw",
                    "connected",
                    "nodes",
                    "nodes_by_type",
                }:
                    continue
                if not isinstance(value, dict):
                    continue
                normalised = self._normalise_type_section(
                    key,
                    value,
                    addr_map.get(key, []),
                )
                if normalised["settings"]:
                    bucket = prev_by_type.setdefault(key, {})
                    bucket.update(normalised["settings"])

            all_types = set(addr_map) | set(prev_by_type)
            settings_by_type: dict[str, dict[str, Any]] = {
                node_type: dict(prev_by_type.get(node_type, {}))
                for node_type in all_types
            }

            if addrs:
                start_index = self._rr_index.get(dev_id, 0) % len(addrs)
                count = min(HTR_SETTINGS_PER_CYCLE, len(addrs))
                for k in range(count):
                    idx = (start_index + k) % len(addrs)
                    addr = addrs[idx]
                    addr_types = reverse.get(addr)
                    node_type = next(iter(addr_types)) if addr_types else "htr"
                    js = await self.client.get_node_settings(
                        dev_id, (node_type, addr)
                    )
                    if isinstance(js, dict):
                        bucket = settings_by_type.setdefault(node_type, {})
                        bucket[addr] = js
                self._rr_index[dev_id] = (start_index + count) % len(addrs)

            dev_name = (self._device.get("name") or f"Device {dev_id}").strip()

            for node_type, settings in settings_by_type.items():
                for addr in settings:
                    self._register_node_address(node_type, str(addr))

            addr_map = dict(self._nodes_by_type)

            combined_types = set(addr_map) | set(settings_by_type)
            nodes_by_type: dict[str, dict[str, Any]] = {}
            for node_type in combined_types:
                cached_addrs = addr_map.get(node_type, [])
                settings = dict(settings_by_type.get(node_type, {}))
                final_addrs: list[str] = []
                for candidate in [*cached_addrs, *settings.keys()]:
                    addr_str = str(candidate).strip()
                    if not addr_str or addr_str in final_addrs:
                        continue
                    final_addrs.append(addr_str)
                nodes_by_type[node_type] = {
                    "addrs": final_addrs,
                    "settings": settings,
                }

            heater_section = nodes_by_type.get("htr")
            if heater_section is None:
                heater_section = {
                    "addrs": list(addr_map.get("htr", [])),
                    "settings": dict(settings_by_type.get("htr", {})),
                }
                nodes_by_type["htr"] = heater_section

            result = {
                dev_id: {
                    "dev_id": dev_id,
                    "name": dev_name,
                    "raw": self._device,
                    "connected": True,
                    "nodes": self._nodes,
                    "nodes_by_type": nodes_by_type,
                },
            }

            for node_type, section in nodes_by_type.items():
                result[dev_id][node_type] = section

            result[dev_id]["htr"] = nodes_by_type.get(
                "htr", {"addrs": [], "settings": {}}
            )

        except TimeoutError as err:
            raise UpdateFailed("API timeout") from err
        except BackendRateLimitError as err:
            self._backoff = min(
                max(self._base_interval, (self._backoff or self._base_interval) * 2),
                3600,
            )
            self.update_interval = timedelta(seconds=self._backoff)
            raise UpdateFailed(
                f"Rate limited; backing off to {self._backoff}s"
            ) from err
        except (ClientError, BackendAuthError) as err:
            raise UpdateFailed(f"API error: {err}") from err
        else:
            if self._backoff:
                self._backoff = 0
                self.update_interval = timedelta(seconds=self._base_interval)

            return result


class EnergyStateCoordinator(
    DataUpdateCoordinator[dict[str, dict[str, Any]]]
):  # dev_id -> per-device data
    """Polls heater energy counters and exposes energy and power per heater."""

    def __init__(
        self,
        hass: HomeAssistant,
        client: RESTClient,
        dev_id: str,
        addrs: Iterable[str] | Mapping[str, Iterable[str]],
    ) -> None:
        """Initialize the heater energy coordinator."""
        super().__init__(
            hass,
            logger=_LOGGER,
            name="termoweb-htr-energy",
            update_interval=HTR_ENERGY_UPDATE_INTERVAL,
        )
        self.client = client
        self._dev_id = dev_id
        self._addresses_by_type: dict[str, list[str]] = {}
        self._addr_lookup: dict[str, str] = {}
        self._addrs: list[str] = []
        self._compat_aliases: dict[str, str] = {}
        self._last: dict[tuple[str, str], tuple[float, float]] = {}
        self.update_addresses(addrs)

    def update_addresses(
        self, addrs: Iterable[str] | Mapping[str, Iterable[str]]
    ) -> None:
        """Replace the tracked heater addresses with ``addrs``."""

        cleaned_map, compat_aliases = normalize_heater_addresses(addrs)
        self._addresses_by_type = cleaned_map
        self._compat_aliases = compat_aliases
        self._addr_lookup = {
            addr: node_type
            for node_type, addrs_for_type in cleaned_map.items()
            for addr in addrs_for_type
        }
        self._addrs = [
            addr for addrs_for_type in cleaned_map.values() for addr in addrs_for_type
        ]

        valid_keys = {
            (node_type, addr)
            for node_type, addrs_for_type in cleaned_map.items()
            for addr in addrs_for_type
        }
        self._last = {
            key: value for key, value in self._last.items() if key in valid_keys
        }

    async def _async_update_data(self) -> dict[str, dict[str, Any]]:
        """Fetch recent heater energy samples and derive totals and power."""
        dev_id = self._dev_id
        try:
            energy_by_type: dict[str, dict[str, float]] = {}
            power_by_type: dict[str, dict[str, float]] = {}

            for node_type, addrs_for_type in self._addresses_by_type.items():
                for addr in addrs_for_type:
                    now = time.time()
                    start = now - 3600  # fetch recent samples
                    try:
                        samples = await self.client.get_node_samples(
                            dev_id, (node_type, addr), start, now
                        )
                    except (ClientError, BackendRateLimitError, BackendAuthError):
                        samples = []

                    if not samples:
                        _LOGGER.debug(
                            "No energy samples for device %s node %s:%s",
                            dev_id,
                            node_type,
                            addr,
                        )
                        continue

                    last = samples[-1]
                    counter = float_or_none(last.get("counter"))
                    t = float_or_none(last.get("t"))
                    if counter is None or t is None:
                        _LOGGER.debug(
                            "Latest sample missing 't' or 'counter' for device %s node %s:%s",
                            dev_id,
                            node_type,
                            addr,
                        )
                        continue

                    kwh = counter / 1000.0
                    energy_bucket = energy_by_type.setdefault(node_type, {})
                    energy_bucket[addr] = kwh

                    key = (node_type, addr)
                    prev = self._last.get(key)
                    if prev:
                        prev_t, prev_kwh = prev
                        if kwh < prev_kwh or t <= prev_t:
                            self._last[key] = (t, kwh)
                            continue
                        dt_hours = (t - prev_t) / 3600
                        if dt_hours > 0:
                            delta_kwh = kwh - prev_kwh
                            power = delta_kwh / dt_hours * 1000
                            power_bucket = power_by_type.setdefault(node_type, {})
                            power_bucket[addr] = power
                        self._last[key] = (t, kwh)
                    else:
                        self._last[key] = (t, kwh)

            dev_data: dict[str, Any] = {"dev_id": dev_id}
            nodes_by_type: dict[str, dict[str, Any]] = {}

            for node_type, addrs_for_type in self._addresses_by_type.items():
                bucket = {
                    "energy": dict(energy_by_type.get(node_type, {})),
                    "power": dict(power_by_type.get(node_type, {})),
                    "addrs": list(addrs_for_type),
                }
                dev_data[node_type] = bucket
                nodes_by_type[node_type] = bucket

            heater_data = nodes_by_type.get("htr")
            if heater_data is None:
                heater_data = {
                    "energy": dict(energy_by_type.get("htr", {})),
                    "power": dict(power_by_type.get("htr", {})),
                    "addrs": list(self._addresses_by_type.get("htr", [])),
                }
                nodes_by_type["htr"] = heater_data
                dev_data["htr"] = heater_data

            for alias, canonical in self._compat_aliases.items():
                canonical_bucket = nodes_by_type.get(canonical)
                if canonical_bucket is None:
                    canonical_bucket = {
                        "energy": {},
                        "power": {},
                        "addrs": list(self._addresses_by_type.get(canonical, [])),
                    }
                    nodes_by_type[canonical] = canonical_bucket
                    dev_data[canonical] = canonical_bucket
                dev_data[alias] = canonical_bucket
                if alias not in nodes_by_type:
                    nodes_by_type[alias] = canonical_bucket

            dev_data["nodes_by_type"] = nodes_by_type

            result: dict[str, dict[str, Any]] = {dev_id: dev_data}

        except TimeoutError as err:
            raise UpdateFailed("API timeout") from err
        except (ClientError, BackendRateLimitError, BackendAuthError) as err:
            raise UpdateFailed(f"API error: {err}") from err
        else:
            return result
