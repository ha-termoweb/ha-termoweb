from __future__ import annotations

import logging
from datetime import timedelta
import time
from typing import Any, Dict, Optional

from aiohttp import ClientError
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .api import TermoWebAuthError, TermoWebClient, TermoWebRateLimitError
from .const import HTR_ENERGY_UPDATE_INTERVAL, MIN_POLL_INTERVAL

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


class TermoWebCoordinator(
    DataUpdateCoordinator[Dict[str, Dict[str, Any]]]
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

    def _addrs(self) -> list[str]:
        node_list = self._nodes.get("nodes") if isinstance(self._nodes, dict) else None
        addrs: list[str] = []
        if isinstance(node_list, list):
            for n in node_list:
                if isinstance(n, dict) and (n.get("type") or "").lower() == "htr":
                    addrs.append(str(n.get("addr")))
        return addrs

    async def _async_update_data(self) -> Dict[str, Dict[str, Any]]:
        dev_id = self._dev_id
        addrs = self._addrs()
        try:
            prev_dev = (self.data or {}).get(dev_id, {})
            prev_htr = prev_dev.get("htr") or {}
            settings_map: Dict[str, Any] = dict(prev_htr.get("settings") or {})

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
                    except (ClientError, TermoWebRateLimitError, TermoWebAuthError) as err:
                        _LOGGER.debug(
                            "Error fetching settings for heater %s: %s", addr, err, exc_info=err
                        )
                        # keep previous settings on error
                        pass
                self._rr_index[dev_id] = (start + count) % len(addrs)

            dev_name = (self._device.get("name") or f"Device {dev_id}").strip()

            result = {
                dev_id: {
                    "dev_id": dev_id,
                    "name": dev_name,
                    "raw": self._device,
                    "connected": True,
                    "nodes": self._nodes,
                    "htr": {
                        "addrs": addrs,
                        "settings": settings_map,
                    },
                }
            }

            if self._backoff:
                self._backoff = 0
                self.update_interval = timedelta(seconds=self._base_interval)

            return result

        except TermoWebRateLimitError as err:
            self._backoff = min(
                max(self._base_interval, (self._backoff or self._base_interval) * 2),
                3600,
            )
            self.update_interval = timedelta(seconds=self._backoff)
            raise UpdateFailed(f"Rate limited; backing off to {self._backoff}s") from err
        except (ClientError, TermoWebAuthError) as err:
            raise UpdateFailed(f"API error: {err}") from err


class TermoWebHeaterEnergyCoordinator(
    DataUpdateCoordinator[Dict[str, Dict[str, Any]]]
):  # dev_id -> per-device data
    """Polls heater energy counters and exposes energy and power per heater."""

    def __init__(
        self,
        hass: HomeAssistant,
        client: TermoWebClient,
        dev_id: str,
        addrs: list[str],
    ) -> None:
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

    async def _async_update_data(self) -> Dict[str, Dict[str, Any]]:
        dev_id = self._dev_id
        addrs = self._addrs
        try:
            energy_map: Dict[str, float] = {}
            power_map: Dict[str, float] = {}

            for addr in addrs:
                now = time.time()
                start = now - 3600  # fetch recent samples
                try:
                    samples = await self.client.get_htr_samples(dev_id, addr, start, now)
                except (ClientError, TermoWebRateLimitError, TermoWebAuthError):
                    samples = []

                if not samples:
                    _LOGGER.debug(
                        "No energy samples for device %s heater %s", dev_id, addr
                    )
                    continue

                last = samples[-1]
                counter = _as_float(last.get("counter"))
                t = _as_float(last.get("t"))
                if counter is None or t is None:
                    _LOGGER.debug(
                        "Latest sample missing 't' or 'counter' for device %s heater %s",
                        dev_id,
                        addr,
                    )
                    continue

                energy_map[addr] = counter

                prev = self._last.get((dev_id, addr))
                if prev:
                    prev_t, prev_counter = prev
                    if counter < prev_counter or t <= prev_t:
                        self._last[(dev_id, addr)] = (t, counter)
                        continue
                    dt_hours = (t - prev_t) / 3600
                    if dt_hours > 0:
                        power = (counter - prev_counter) / dt_hours * 1000
                        power_map[addr] = power
                    self._last[(dev_id, addr)] = (t, counter)
                else:
                    self._last[(dev_id, addr)] = (t, counter)

            result: Dict[str, Dict[str, Any]] = {
                dev_id: {
                    "dev_id": dev_id,
                    "htr": {
                        "energy": energy_map,
                        "power": power_map,
                    },
                }
            }

            return result

        except (ClientError, TermoWebRateLimitError, TermoWebAuthError) as err:
            raise UpdateFailed(f"API error: {err}") from err
