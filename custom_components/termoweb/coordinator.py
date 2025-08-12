from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional

from aiohttp import ClientError
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .api import TermoWebAuthError, TermoWebClient, TermoWebRateLimitError
from .const import MIN_POLL_INTERVAL

_LOGGER = logging.getLogger(__name__)

# How many heater settings to fetch per device per cycle (keep gentle)
HTR_SETTINGS_PER_CYCLE = 1


def _as_float(val: Any) -> Optional[float]:
    try:
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        s = str(val).strip()
        if not s:
            return None
        return float(s)
    except Exception:
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


class TermoWebPmoEnergyCoordinator(
    DataUpdateCoordinator[dict[str, dict[str, Any]]]
):  # dev_id -> pmo energy
    """Poll PMO energy counters and expose kWh totals."""

    def __init__(self, hass: HomeAssistant, client: TermoWebClient) -> None:
        super().__init__(
            hass,
            logger=_LOGGER,
            name="termoweb_pmo_energy",
            update_interval=timedelta(minutes=15),
        )
        self.client = client

    async def _async_update_data(self) -> dict[str, dict[str, Any]]:
        try:
            devices: list[dict[str, Any]] = await self.client.list_devices()
            if not isinstance(devices, list):
                devices = []

            result: dict[str, dict[str, Any]] = {}
            for dev in devices:
                if not isinstance(dev, dict):
                    continue

                dev_id = str(
                    dev.get("dev_id")
                    or dev.get("id")
                    or dev.get("serial_id")
                    or ""
                ).strip()
                if not dev_id:
                    continue

                prev_dev = (self.data or {}).get(dev_id, {})
                prev_energy = (
                    (prev_dev.get("pmo") or {}).get("energy_total") or {}
                )
                energy_map: dict[str, float] = dict(prev_energy)

                try:
                    nodes = await self.client.get_nodes(dev_id)
                except Exception:
                    nodes = None

                node_list = nodes.get("nodes") if isinstance(nodes, dict) else None
                if isinstance(node_list, list):
                    for node in node_list:
                        if not isinstance(node, dict):
                            continue
                        if (node.get("type") or "").lower() != "pmo":
                            continue
                        addr = str(node.get("addr"))
                        prev_val = energy_map.get(addr)
                        try:
                            samples = await self.client.get_pmo_samples(dev_id, addr)
                            samp_list = (
                                samples.get("samples")
                                if isinstance(samples, dict)
                                else None
                            )
                            if isinstance(samp_list, list) and samp_list:
                                last = samp_list[-1]
                                counter_wh = _as_float(last.get("counter"))
                                if counter_wh is not None:
                                    kwh = counter_wh / 1000.0
                                    if prev_val is not None and kwh < prev_val:
                                        _LOGGER.warning(
                                            "PMO energy counter reset for %s addr %s: %s -> %s",
                                            dev_id,
                                            addr,
                                            prev_val,
                                            kwh,
                                        )
                                    energy_map[addr] = kwh
                        except (ClientError, TermoWebAuthError, TermoWebRateLimitError):
                            if prev_val is not None:
                                energy_map[addr] = prev_val

                result[dev_id] = {
                    "dev_id": dev_id,
                    "pmo": {"energy_total": energy_map},
                }

            return result
        except (ClientError, TermoWebAuthError, TermoWebRateLimitError) as err:
            raise UpdateFailed(f"API error: {err}") from err
