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
from .utils import HEATER_NODE_TYPES, addresses_by_node_type, float_or_none

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
        self._addr_map: dict[str, list[str]] | None = None

    def _ensure_inventory(self) -> list[Node]:
        """Return cached node inventory, rebuilding when necessary."""
        if not self._node_inventory and self._nodes:
            try:
                self._node_inventory = build_node_inventory(self._nodes)
            except ValueError as err:  # pragma: no cover - defensive
                _LOGGER.debug(
                    "Failed to build node inventory for %s: %s",
                    self._dev_id,
                    err,
                    exc_info=err,
                )
                self._node_inventory = []
        return self._node_inventory

    def _addr_lookup(self) -> tuple[dict[str, list[str]], dict[str, str]]:
        """Return mapping of node types to addresses and reverse lookup."""

        if self._addr_map is None:
            inventory = self._ensure_inventory()
            mapping: dict[str, list[str]] = {}
            if inventory:
                grouped, _unknown = addresses_by_node_type(
                    inventory, known_types=HEATER_NODE_TYPES
                )
                for node_type, addrs in grouped.items():
                    if node_type in HEATER_NODE_TYPES:
                        mapping[node_type] = list(addrs)
            self._addr_map = mapping

        addr_map = self._addr_map
        reverse: dict[str, str] = {}
        for node_type, addrs in addr_map.items():
            for addr in addrs:
                reverse.setdefault(addr, node_type)

        return addr_map, reverse

    def update_nodes(
        self,
        nodes: dict[str, Any],
        node_inventory: list[Node] | None = None,
    ) -> None:
        """Update cached node payload and inventory."""

        self._nodes = nodes or {}
        if node_inventory is not None:
            self._node_inventory = list(node_inventory)
        else:
            self._node_inventory = []
            if self._nodes:
                try:
                    self._node_inventory = build_node_inventory(self._nodes)
                except ValueError as err:  # pragma: no cover - defensive
                    _LOGGER.debug(
                        "Failed to build node inventory for %s: %s",
                        self._dev_id,
                        err,
                        exc_info=err,
                    )
        self._addr_map = None

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

            addr_map, reverse = self._addr_lookup()
            resolved_type = node_type or reverse.get(addr, "htr")

            if resolved_type == "htr":
                payload = await self.client.get_htr_settings(dev_id, addr)
            else:
                payload = await self.client.get_node_settings(
                    dev_id, (resolved_type, addr)
                )

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

            nodes_by_type: dict[str, dict[str, Any]] = {}
            existing_nodes = dev_data.get("nodes_by_type")
            if isinstance(existing_nodes, dict):
                for n_type, section in existing_nodes.items():
                    if not isinstance(section, dict):
                        continue
                    addrs_list = section.get("addrs")
                    if isinstance(addrs_list, list):
                        addrs_copy = [
                            str(item).strip() for item in addrs_list if str(item).strip()
                        ]
                    else:
                        addrs_copy = list(addr_map.get(n_type, []))
                    settings_map = section.get("settings")
                    settings_copy = (
                        dict(settings_map) if isinstance(settings_map, dict) else {}
                    )
                    nodes_by_type[n_type] = {
                        "addrs": addrs_copy,
                        "settings": settings_copy,
                    }

            for n_type, addrs_for_type in addr_map.items():
                nodes_by_type.setdefault(
                    n_type,
                    {"addrs": list(addrs_for_type), "settings": {}},
                )

            section = nodes_by_type.setdefault(
                node_type, {"addrs": [], "settings": {}}
            )
            if addr not in section["addrs"]:
                section["addrs"].append(addr)
            section_settings = section.setdefault("settings", {})
            section_settings[addr] = payload
            nodes_by_type[node_type] = {
                "addrs": list(section["addrs"]),
                "settings": dict(section_settings),
            }

            legacy = nodes_by_type.get("htr")
            if legacy is None:
                legacy = {"addrs": list(addr_map.get("htr", [])), "settings": {}}
                nodes_by_type["htr"] = legacy

            dev_data["nodes_by_type"] = nodes_by_type
            dev_data["htr"] = legacy
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
        addr_map, reverse = self._addr_lookup()
        addrs = [addr for addrs in addr_map.values() for addr in addrs]
        try:
            prev_dev = (self.data or {}).get(dev_id, {})
            prev_by_type: dict[str, dict[str, Any]] = {}
            existing_nodes = prev_dev.get("nodes_by_type")
            if isinstance(existing_nodes, dict):
                for node_type, section in existing_nodes.items():
                    if not isinstance(section, dict):
                        continue
                    settings = section.get("settings")
                    if isinstance(settings, dict):
                        prev_by_type[node_type] = dict(settings)
            prev_htr = prev_dev.get("htr") or {}
            if isinstance(prev_htr, dict):
                legacy_settings = prev_htr.get("settings")
                if isinstance(legacy_settings, dict):
                    bucket = prev_by_type.setdefault("htr", {})
                    bucket.update(legacy_settings)

            settings_by_type: dict[str, dict[str, Any]] = {}
            for node_type, addrs_for_type in addr_map.items():
                settings_by_type[node_type] = dict(
                    prev_by_type.get(node_type, {})
                )

            if addrs:
                start = self._rr_index.get(dev_id, 0) % len(addrs)
                count = min(HTR_SETTINGS_PER_CYCLE, len(addrs))
                for k in range(count):
                    idx = (start + k) % len(addrs)
                    addr = addrs[idx]
                    node_type = reverse.get(addr, "htr")
                    if node_type == "htr":
                        js = await self.client.get_htr_settings(dev_id, addr)
                    else:
                        js = await self.client.get_node_settings(
                            dev_id, (node_type, addr)
                        )
                    if isinstance(js, dict):
                        bucket = settings_by_type.setdefault(node_type, {})
                        bucket[addr] = js
                self._rr_index[dev_id] = (start + count) % len(addrs)

            dev_name = (self._device.get("name") or f"Device {dev_id}").strip()

            nodes_by_type = {
                node_type: {
                    "addrs": list(addrs_for_type),
                    "settings": dict(settings_by_type.get(node_type, {})),
                }
                for node_type, addrs_for_type in addr_map.items()
            }

            legacy = nodes_by_type.get("htr")
            if legacy is None:
                legacy = {"addrs": [], "settings": {}}
                nodes_by_type["htr"] = legacy

            result = {
                dev_id: {
                    "dev_id": dev_id,
                    "name": dev_name,
                    "raw": self._device,
                    "connected": True,
                    "nodes": self._nodes,
                    "nodes_by_type": nodes_by_type,
                    "htr": legacy,
                }
            }

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
        self._last: dict[tuple[str, str, str], tuple[float, float]] = {}
        self.update_addresses(addrs)

    def update_addresses(
        self, addrs: Iterable[str] | Mapping[str, Iterable[str]]
    ) -> None:
        """Replace the tracked heater addresses with ``addrs``."""

        cleaned_map: dict[str, list[str]] = {}

        if isinstance(addrs, Mapping):
            sources = addrs.items()
        else:
            sources = [("htr", addrs)]

        for node_type, values in sources:
            node_type_str = str(node_type or "").strip().lower()
            if not node_type_str:
                continue
            if node_type_str not in HEATER_NODE_TYPES:
                continue
            bucket = cleaned_map.setdefault(node_type_str, [])
            seen: set[str] = set(bucket)
            for addr in values or []:
                addr_str = str(addr).strip()
                if not addr_str or addr_str in seen:
                    continue
                seen.add(addr_str)
                bucket.append(addr_str)

        if "htr" not in cleaned_map:
            cleaned_map["htr"] = []

        self._addresses_by_type = cleaned_map
        self._addr_lookup = {
            addr: node_type for node_type, addrs_for_type in cleaned_map.items() for addr in addrs_for_type
        }
        self._addrs = [addr for addrs_for_type in cleaned_map.values() for addr in addrs_for_type]

        valid_keys = {
            (self._dev_id, node_type, addr)
            for node_type, addrs_for_type in cleaned_map.items()
            for addr in addrs_for_type
        }
        self._last = {
            key: value for key, value in self._last.items() if key in valid_keys
        }

    async def _async_update_data(self) -> dict[str, dict[str, Any]]:
        """Fetch recent heater energy samples and derive totals and power."""
        dev_id = self._dev_id
        addrs = self._addrs
        try:
            energy_by_type: dict[str, dict[str, float]] = {
                node_type: {} for node_type in self._addresses_by_type
            }
            power_by_type: dict[str, dict[str, float]] = {
                node_type: {} for node_type in self._addresses_by_type
            }

            for addr in addrs:
                node_type = self._addr_lookup.get(addr, "htr")
                now = time.time()
                start = now - 3600  # fetch recent samples
                try:
                    if node_type == "htr":
                        samples = await self.client.get_htr_samples(
                            dev_id, addr, start, now
                        )
                    else:
                        samples = await self.client.get_node_samples(
                            dev_id, (node_type, addr), start, now
                        )
                except (ClientError, BackendRateLimitError, BackendAuthError):
                    samples = []

                if not samples:
                    _LOGGER.debug(
                        "No energy samples for device %s heater %s", dev_id, addr
                    )
                    continue

                last = samples[-1]
                counter = float_or_none(last.get("counter"))
                t = float_or_none(last.get("t"))
                if counter is None or t is None:
                    _LOGGER.debug(
                        "Latest sample missing 't' or 'counter' for device %s heater %s",
                        dev_id,
                        addr,
                    )
                    continue

                kwh = counter / 1000.0
                energy_bucket = energy_by_type.setdefault(node_type, {})
                energy_bucket[addr] = kwh

                key = (dev_id, node_type, addr)
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

            nodes_by_type = {}
            for node_type, addrs_for_type in self._addresses_by_type.items():
                nodes_by_type[node_type] = {
                    "energy": dict(energy_by_type.get(node_type, {})),
                    "power": dict(power_by_type.get(node_type, {})),
                    "addrs": list(addrs_for_type),
                }

            legacy = nodes_by_type.get("htr")
            if legacy is None:
                legacy = {"energy": {}, "power": {}, "addrs": []}
                nodes_by_type["htr"] = legacy

            result: dict[str, dict[str, Any]] = {
                dev_id: {
                    "dev_id": dev_id,
                    "nodes_by_type": nodes_by_type,
                    "htr": legacy,
                }
            }

        except TimeoutError as err:
            raise UpdateFailed("API timeout") from err
        except (ClientError, BackendRateLimitError, BackendAuthError) as err:
            raise UpdateFailed(f"API error: {err}") from err
        else:
            return result
