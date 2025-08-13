from __future__ import annotations

import logging
import time
from datetime import timedelta
from typing import Any, Dict, List, Optional

from aiohttp import ClientError, ClientResponseError
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.core import callback

from .api import TermoWebClient, TermoWebRateLimitError, TermoWebAuthError
from .const import MIN_POLL_INTERVAL, signal_ws_data

_LOGGER = logging.getLogger(__name__)

# How many heater settings to fetch per device per cycle (keep gentle)
HTR_SETTINGS_PER_CYCLE = 1


def _as_float(value: Any) -> Optional[float]:
    """Safely convert a value to float."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None
    return None


class TermoWebCoordinator(DataUpdateCoordinator[Dict[str, Dict[str, Any]]]):  # dev_id -> per-device data
    """Polls TermoWeb and exposes a per-device dict used by platforms."""

    def __init__(self, hass: HomeAssistant, client: TermoWebClient, base_interval: int) -> None:
        super().__init__(
            hass,
            logger=_LOGGER,
            name="termoweb",
            update_interval=timedelta(seconds=max(base_interval, MIN_POLL_INTERVAL)),
        )
        self.client = client
        self._base_interval = max(base_interval, MIN_POLL_INTERVAL)
        self._backoff = 0  # seconds
        self._rr_index: dict[str, int] = {}  # per-device round-robin index for heater settings

    async def _async_update_data(self) -> Dict[str, Dict[str, Any]]:
        try:
            devices: List[Dict[str, Any]] = await self.client.list_devices()
            if not isinstance(devices, list):
                devices = []

            result: Dict[str, Dict[str, Any]] = {}
            for dev in devices:
                if not isinstance(dev, dict):
                    continue

                dev_id = str(dev.get("dev_id") or dev.get("id") or dev.get("serial_id") or "").strip()
                if not dev_id:
                    continue

                # Fetch nodes; tolerate failures
                try:
                    nodes = await self.client.get_nodes(dev_id)
                except Exception:
                    nodes = None

                # Prepare carry-over cache of heater settings for this device
                prev_dev = (self.data or {}).get(dev_id, {})
                prev_htr = prev_dev.get("htr") or {}
                settings_map: Dict[str, Any] = dict(prev_htr.get("settings") or {})

                # Determine heater addresses for this device
                addrs: list[str] = []
                node_list = nodes.get("nodes") if isinstance(nodes, dict) else None
                if isinstance(node_list, list):
                    for n in node_list:
                        if isinstance(n, dict) and (n.get("type") or "").lower() == "htr":
                            addrs.append(str(n.get("addr")))

                # Round-robin fetch: at most HTR_SETTINGS_PER_CYCLE items
                if addrs:
                    start = self._rr_index.get(dev_id, 0) % len(addrs)
                    count = min(HTR_SETTINGS_PER_CYCLE, len(addrs))
                    for k in range(count):
                        idx = (start + k) % len(addrs)
                        addr = addrs[idx]
                        try:
                            js = await self.client.get_htr_settings(dev_id, addr)
                            if isinstance(js, dict):
                                settings_map[addr] = js
                        except (ClientError, TermoWebRateLimitError, TermoWebAuthError):
                            # On error, keep old cached settings for that addr
                            pass
                    # advance pointer
                    self._rr_index[dev_id] = (start + count) % len(addrs)

                # Build device entry
                dev_name = (dev.get("name") or f"Device {dev_id}").strip()
                connected: Optional[bool] = True if nodes is not None else None

                result[dev_id] = {
                    "dev_id": dev_id,
                    "name": dev_name,
                    "raw": dev,
                    "connected": connected,
                    "nodes": nodes,
                    "htr": {
                        "addrs": addrs,
                        "settings": settings_map,  # addr -> HtrSettings JSON
                    },
                }

            # Reset backoff on success
            if self._backoff:
                self._backoff = 0
                self.update_interval = timedelta(seconds=self._base_interval)

            return result

        except TermoWebRateLimitError as err:
            # Exponential backoff up to 1 hour
            self._backoff = min(max(self._base_interval, (self._backoff or self._base_interval) * 2), 3600)
            self.update_interval = timedelta(seconds=self._backoff)
            raise UpdateFailed(f"Rate limited; backing off to {self._backoff}s") from err
        except (ClientError, TermoWebAuthError) as err:
            raise UpdateFailed(f"API error: {err}") from err


class TermoWebPmoPowerCoordinator(
    DataUpdateCoordinator[Dict[str, Dict[str, Any]]]
):
    """Coordinator polling real-time power for PMO nodes."""

    def __init__(
        self,
        hass: HomeAssistant,
        client: TermoWebClient,
        base: TermoWebCoordinator,
        entry_id: str,
    ) -> None:
        super().__init__(
            hass,
            logger=_LOGGER,
            name="termoweb_pmo_power",
            update_interval=timedelta(seconds=60),
        )
        self.client = client
        self._base = base
        self.addr_set: dict[str, set[str]] = {}
        self._unsub = async_dispatcher_connect(
            hass, signal_ws_data(entry_id), self._on_ws_data
        )
        self._unsupported: set[tuple[str, str]] = set()

    async def _async_update_data(self) -> Dict[str, Dict[str, Any]]:
        base_data = self._base.data or {}
        data: Dict[str, Dict[str, Any]] = dict(self.data or {})
        for dev_id, dev in base_data.items():
            nodes = dev.get("nodes") or {}
            node_list = nodes.get("nodes") if isinstance(nodes, dict) else None
            pmo_addrs: list[str] = []
            if isinstance(node_list, list):
                for node in node_list:
                    if (
                        isinstance(node, dict)
                        and (node.get("type") or "").lower() == "pmo"
                    ):
                        pmo_addrs.append(str(node.get("addr")))
            if not pmo_addrs:
                pmo_addrs = dev.get("htr", {}).get("addrs") or []
            for addr in pmo_addrs:
                self.addr_set.setdefault(dev_id, set()).add(addr)
                if (dev_id, addr) in self._unsupported:
                    continue
                try:
                    power = await self.client.get_pmo_power(dev_id, addr)
                except ClientResponseError as err:
                    _LOGGER.debug(
                        "PMO power error for %s/%s: %s", dev_id, addr, err
                    )
                    continue
                except Exception:
                    continue
                if power is None:
                    _LOGGER.debug("PMO power unsupported for %s/%s", dev_id, addr)
                    self._unsupported.add((dev_id, addr))
                    continue
                val = _as_float(power)
                if val is None:
                    continue
                dev_map = (
                    data.setdefault(dev_id, {})
                    .setdefault("pmo", {})
                    .setdefault("power", {})
                )
                dev_map[addr] = val
        return data

    @callback
    def _on_ws_data(self, payload: dict) -> None:
        if payload.get("kind") != "pmo_power":
            return
        dev_id = payload.get("dev_id")
        addr = payload.get("addr")
        val = _as_float(payload.get("value"))
        if val is None:
            return
        data: Dict[str, Dict[str, Any]] = dict(self.data or {})
        dev_map = data.setdefault(dev_id, {}).setdefault("pmo", {}).setdefault("power", {})
        dev_map[addr] = val
        if dev_id is not None and addr is not None:
            self.addr_set.setdefault(dev_id, set()).add(addr)
            _LOGGER.debug("WS set PMO power %s/%s to %s", dev_id, addr, val)
        self.async_set_updated_data(data)


class TermoWebPmoEnergyCoordinator(
    DataUpdateCoordinator[Dict[str, Dict[str, Any]]],
):
    """Coordinator polling cumulative energy for PMO nodes."""

    def __init__(
        self,
        hass: HomeAssistant,
        client: TermoWebClient,
        base: TermoWebCoordinator,
    ) -> None:
        super().__init__(
            hass,
            logger=_LOGGER,
            name="termoweb_pmo_energy",
            update_interval=timedelta(minutes=15),
        )
        self.client = client
        self._base = base
        self.addr_set: dict[str, set[str]] = {}

    async def _async_update_data(self) -> Dict[str, Dict[str, Any]]:
        base_data = self._base.data or {}
        data: Dict[str, Dict[str, Any]] = dict(self.data or {})
        end = int(time.time())
        start = end - 3600
        for dev_id, dev in base_data.items():
            nodes = dev.get("nodes") or {}
            node_list = nodes.get("nodes") if isinstance(nodes, dict) else None
            pmo_addrs: list[str] = []
            if isinstance(node_list, list):
                for node in node_list:
                    if (
                        isinstance(node, dict)
                        and (node.get("type") or "").lower() == "pmo"
                    ):
                        pmo_addrs.append(str(node.get("addr")))
            if not pmo_addrs:
                pmo_addrs = dev.get("htr", {}).get("addrs") or []
            for addr in pmo_addrs:
                self.addr_set.setdefault(dev_id, set()).add(addr)
                try:
                    samples = await self.client.get_pmo_samples(dev_id, addr, start, end)
                except ClientResponseError as err:
                    _LOGGER.debug("PMO energy error for %s/%s: %s", dev_id, addr, err)
                    continue
                except Exception:
                    continue
                dev_map = (
                    data.setdefault(dev_id, {})
                    .setdefault("pmo", {})
                    .setdefault("energy", {})
                )
                if not samples:
                    _LOGGER.debug(
                        "PMO energy samples empty for %s/%s", dev_id, addr
                    )
                    dev_map[addr] = None
                    continue
                latest = max(samples, key=lambda s: s.get("t") or 0)
                val = _as_float(latest.get("counter"))
                if val is None:
                    continue
                dev_map[addr] = val
        return data
