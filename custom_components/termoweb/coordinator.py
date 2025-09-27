"""Coordinator helpers for the TermoWeb integration."""

from __future__ import annotations

from datetime import timedelta
import logging
import time
from typing import Any

from aiohttp import ClientError
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .api import TermoWebAuthError, TermoWebClient, TermoWebRateLimitError
from .const import HTR_ENERGY_UPDATE_INTERVAL, MIN_POLL_INTERVAL
from .utils import extract_heater_addrs, float_or_none

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
        client: TermoWebClient,
        base_interval: int,
        dev_id: str,
        device: dict[str, Any],
        nodes: dict[str, Any],
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
        self._addr_map: dict[str, list[str]] | None = None
        self._addr_reverse: dict[str, str] | None = None

    # ----------------- Inventory helpers -----------------
    def _ensure_inventory(self) -> None:
        if self._addr_map is not None and self._addr_reverse is not None:
            return

        nodes_by_type: dict[str, list[str]] = {}

        def _record(node_type: str | None, addr: Any) -> None:
            addr_str = str(addr).strip() if addr is not None else ""
            if not addr_str:
                return
            key = (node_type or "htr").lower()
            addrs = nodes_by_type.setdefault(key, [])
            if addr_str not in addrs:
                addrs.append(addr_str)

        nodes = self._nodes
        if isinstance(nodes, dict):
            items = nodes.get("nodes")
            if isinstance(items, list):
                for node in items:
                    if isinstance(node, dict):
                        _record(node.get("type"), node.get("addr"))
            for key, value in nodes.items():
                if key == "nodes":
                    continue
                if isinstance(value, list):
                    for node in value:
                        if isinstance(node, dict):
                            _record(node.get("type") or key.rstrip("s"), node.get("addr"))
                        else:
                            _record(key.rstrip("s"), node)
                elif isinstance(value, dict):
                    addrs = value.get("addrs")
                    if isinstance(addrs, list):
                        for addr in addrs:
                            _record(key.rstrip("s"), addr)
        elif isinstance(nodes, list):
            for node in nodes:
                if isinstance(node, dict):
                    _record(node.get("type"), node.get("addr"))

        htr_addrs = [str(addr) for addr in extract_heater_addrs(nodes)]
        if htr_addrs:
            addrs = nodes_by_type.setdefault("htr", [])
            for addr in htr_addrs:
                if addr not in addrs:
                    addrs.append(addr)
        nodes_by_type.setdefault("htr", [])

        reverse: dict[str, str] = {}
        for node_type, addrs in nodes_by_type.items():
            for addr in addrs:
                reverse.setdefault(addr, node_type)

        self._addr_map = nodes_by_type
        self._addr_reverse = reverse

    def _addr_lookup(self) -> tuple[dict[str, list[str]], dict[str, str]]:
        self._ensure_inventory()
        assert self._addr_map is not None
        assert self._addr_reverse is not None
        return dict(self._addr_map), dict(self._addr_reverse)

    @staticmethod
    def _merge_addrs(*addr_lists: list[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for addrs in addr_lists:
            for addr in addrs:
                addr_str = str(addr)
                if not addr_str or addr_str in seen:
                    continue
                seen.add(addr_str)
                merged.append(addr_str)
        return merged

    def _normalise_type_section(
        self, node_type: str, section: Any, expected_addrs: list[str]
    ) -> dict[str, Any]:
        addrs: list[str] = []
        settings: dict[str, Any] = {}

        if isinstance(section, dict):
            raw_addrs = section.get("addrs")
            if isinstance(raw_addrs, list):
                addrs = [str(a).strip() for a in raw_addrs if str(a).strip()]
            elif isinstance(raw_addrs, (tuple, set)):
                addrs = [str(a).strip() for a in raw_addrs if str(a).strip()]
            elif isinstance(raw_addrs, str) and raw_addrs.strip():
                addrs = [raw_addrs.strip()]
            raw_settings = section.get("settings")
            if isinstance(raw_settings, dict):
                for key, value in raw_settings.items():
                    if value is None:
                        continue
                    settings[str(key)] = value
        elif isinstance(section, list):
            addrs = [str(a).strip() for a in section if str(a).strip()]

        expected = [str(a) for a in expected_addrs if str(a)]
        addrs = self._merge_addrs(expected, addrs)
        return {"addrs": addrs, "settings": settings}

    def _addrs(self) -> list[str]:
        addr_map, _ = self._addr_lookup()
        return list(addr_map.get("htr", []))

    async def _fetch_settings(self, node_type: str, addr: str) -> dict[str, Any] | None:
        getter = getattr(self.client, "get_node_settings", None)
        if callable(getter):
            result = await getter(self._dev_id, (node_type, addr))
            return result if isinstance(result, dict) else None

        if node_type == "htr":
            htr_getter = getattr(self.client, "get_htr_settings", None)
            if callable(htr_getter):
                result = await htr_getter(self._dev_id, addr)
                return result if isinstance(result, dict) else None
        return None

    async def async_refresh_heater(self, addr: Any) -> None:
        """Refresh settings for a specific node and push the update to listeners."""

        node_type = "htr"
        node_addr: Any = addr
        if isinstance(addr, (tuple, list)) and len(addr) >= 2:
            node_type = str(addr[0] or "htr").lower()
            node_addr = addr[1]

        dev_id = self._dev_id
        addr_str = str(node_addr).strip() if node_addr is not None else ""
        success = False
        _LOGGER.info(
            "Refreshing heater settings for device %s node %s:%s",
            dev_id,
            node_type,
            addr_str,
        )
        try:
            if not addr_str:
                _LOGGER.error(
                    "Cannot refresh heater settings without an address for device %s",
                    dev_id,
                )
                return

            payload = await self._fetch_settings(node_type, addr_str)

            if not isinstance(payload, dict):
                _LOGGER.debug(
                    "Ignoring unexpected heater settings payload for device %s node %s:%s: %s",
                    dev_id,
                    node_type,
                    addr_str,
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

            nodes_by_type = dict(dev_data.get("nodes_by_type") or {})
            addr_map, _ = self._addr_lookup()
            expected = addr_map.get(node_type, [])
            section = self._normalise_type_section(
                node_type, nodes_by_type.get(node_type), expected
            )
            section["addrs"] = self._merge_addrs(section["addrs"], [addr_str])
            settings = dict(section.get("settings") or {})
            settings[addr_str] = payload
            section["settings"] = settings
            nodes_by_type[node_type] = section

            # Ensure htr section is always present for backwards compatibility
            htr_section = self._normalise_type_section(
                "htr", nodes_by_type.get("htr"), addr_map.get("htr", [])
            )
            nodes_by_type["htr"] = htr_section

            dev_data["nodes_by_type"] = nodes_by_type
            dev_data["htr"] = {
                "addrs": htr_section["addrs"],
                "settings": dict(htr_section.get("settings") or {}),
            }

            new_data[dev_id] = dev_data
            self.async_set_updated_data(new_data)
            success = True

        except TimeoutError as err:
            _LOGGER.error(
                "Timeout refreshing heater settings for device %s node %s:%s",
                dev_id,
                node_type,
                addr_str,
                exc_info=err,
            )
        except (ClientError, TermoWebRateLimitError, TermoWebAuthError) as err:
            _LOGGER.error(
                "Failed to refresh heater settings for device %s node %s:%s: %s",
                dev_id,
                node_type,
                addr_str,
                err,
                exc_info=err,
            )
        finally:
            _LOGGER.info(
                "Finished heater settings refresh for device %s node %s:%s (success=%s)",
                dev_id,
                node_type,
                addr_str,
                success,
            )

    async def _async_update_data(self) -> dict[str, dict[str, Any]]:
        dev_id = self._dev_id
        self._ensure_inventory()
        addr_map, reverse = self._addr_lookup()
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
                if key in {"dev_id", "name", "raw", "connected", "nodes", "nodes_by_type"}:
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
                node_type: dict(prev_by_type.get(node_type, {})) for node_type in all_types
            }

            if addrs:
                start_index = self._rr_index.get(dev_id, 0) % len(addrs)
                count = min(HTR_SETTINGS_PER_CYCLE, len(addrs))
                for k in range(count):
                    idx = (start_index + k) % len(addrs)
                    addr = addrs[idx]
                    node_type = reverse.get(addr, "htr")
                    try:
                        js = await self._fetch_settings(node_type, addr)
                        if isinstance(js, dict):
                            bucket = settings_by_type.setdefault(node_type, {})
                            bucket[addr] = js
                    except (
                        ClientError,
                        TermoWebRateLimitError,
                        TermoWebAuthError,
                    ) as err:
                        _LOGGER.debug(
                            "Error fetching settings for %s %s: %s",
                            node_type,
                            addr,
                            err,
                            exc_info=err,
                        )
                        # keep previous settings on error
                self._rr_index[dev_id] = (start_index + count) % len(addrs)

            dev_name = (self._device.get("name") or f"Device {dev_id}").strip()

            nodes_by_type_result: dict[str, dict[str, Any]] = {}
            ordered_types: list[str] = []
            for node_type in addr_map:
                if node_type not in ordered_types:
                    ordered_types.append(node_type)
            for node_type in settings_by_type:
                if node_type not in ordered_types:
                    ordered_types.append(node_type)

            for node_type in ordered_types:
                default_addrs = addr_map.get(node_type, [])
                section = self._normalise_type_section(
                    node_type,
                    prev_dev.get("nodes_by_type", {}).get(node_type)
                    if isinstance(prev_dev.get("nodes_by_type"), dict)
                    else None,
                    default_addrs,
                )
                bucket = settings_by_type.get(node_type, {})
                section["settings"] = dict(bucket)
                section["addrs"] = self._merge_addrs(
                    section["addrs"], list(bucket.keys())
                )
                nodes_by_type_result[node_type] = section

            htr_section = nodes_by_type_result.get("htr") or {
                "addrs": addr_map.get("htr", []),
                "settings": settings_by_type.get("htr", {}),
            }

            result = {
                dev_id: {
                    "dev_id": dev_id,
                    "name": dev_name,
                    "raw": self._device,
                    "connected": True,
                    "nodes": self._nodes,
                    "nodes_by_type": nodes_by_type_result,
                    "htr": {
                        "addrs": list(htr_section.get("addrs", [])),
                        "settings": dict(htr_section.get("settings", {})),
                    },
                }
            }

        except TimeoutError as err:
            raise UpdateFailed("API timeout") from err
        except TermoWebRateLimitError as err:
            self._backoff = min(
                max(self._base_interval, (self._backoff or self._base_interval) * 2),
                3600,
            )
            self.update_interval = timedelta(seconds=self._backoff)
            raise UpdateFailed(
                f"Rate limited; backing off to {self._backoff}s"
            ) from err
        except (ClientError, TermoWebAuthError) as err:
            raise UpdateFailed(f"API error: {err}") from err
        else:
            if self._backoff:
                self._backoff = 0
                self.update_interval = timedelta(seconds=self._base_interval)

            return result


# Backwards compatibility: retain historical class name used across the integration
TermoWebCoordinator = StateCoordinator


class TermoWebHeaterEnergyCoordinator(
    DataUpdateCoordinator[dict[str, dict[str, Any]]]
):  # dev_id -> per-device data
    """Polls heater energy counters and exposes energy and power per heater."""

    def __init__(
        self,
        hass: HomeAssistant,
        client: TermoWebClient,
        dev_id: str,
        addrs: list[str],
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
        self._addrs = addrs
        self._last: dict[tuple[str, str], tuple[float, float]] = {}

    async def _async_update_data(self) -> dict[str, dict[str, Any]]:
        dev_id = self._dev_id
        addrs = self._addrs
        try:
            energy_map: dict[str, float] = {}
            power_map: dict[str, float] = {}

            for addr in addrs:
                now = time.time()
                start = now - 3600  # fetch recent samples
                try:
                    samples = await self.client.get_htr_samples(
                        dev_id, addr, start, now
                    )
                except (ClientError, TermoWebRateLimitError, TermoWebAuthError):
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
                energy_map[addr] = kwh

                prev = self._last.get((dev_id, addr))
                if prev:
                    prev_t, prev_kwh = prev
                    if kwh < prev_kwh or t <= prev_t:
                        self._last[(dev_id, addr)] = (t, kwh)
                        continue
                    dt_hours = (t - prev_t) / 3600
                    if dt_hours > 0:
                        delta_kwh = kwh - prev_kwh
                        power = delta_kwh / dt_hours * 1000
                        power_map[addr] = power
                    self._last[(dev_id, addr)] = (t, kwh)
                else:
                    self._last[(dev_id, addr)] = (t, kwh)

            result: dict[str, dict[str, Any]] = {
                dev_id: {
                    "dev_id": dev_id,
                    "htr": {
                        "energy": energy_map,
                        "power": power_map,
                    },
                }
            }

        except TimeoutError as err:
            raise UpdateFailed("API timeout") from err
        except (ClientError, TermoWebRateLimitError, TermoWebAuthError) as err:
            raise UpdateFailed(f"API error: {err}") from err
        else:
            return result
