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
from .utils import extract_heater_addrs

_LOGGER = logging.getLogger(__name__)

# How many heater settings to fetch per device per cycle (keep gentle)
HTR_SETTINGS_PER_CYCLE = 1


def _as_float(value: Any) -> float | None:
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

    def _addrs(self) -> list[str]:
        return extract_heater_addrs(self._nodes)

    async def async_refresh_heater(self, addr: str) -> None:
        """Refresh settings for a specific heater and push the update to listeners."""

        dev_id = self._dev_id
        success = False
        _LOGGER.info(
            "Refreshing heater settings for device %s heater %s",
            dev_id,
            addr,
        )
        try:
            if not addr:
                _LOGGER.error(
                    "Cannot refresh heater settings without an address for device %s",
                    dev_id,
                )
                return

            payload = await self.client.get_htr_settings(dev_id, addr)

            if not isinstance(payload, dict):
                _LOGGER.debug(
                    "Ignoring unexpected heater settings payload for device %s heater %s: %s",
                    dev_id,
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

            htr_data = dict(dev_data.get("htr") or {})
            existing_addrs = htr_data.get("addrs")
            if isinstance(existing_addrs, list):
                htr_data["addrs"] = list(existing_addrs)
            else:
                htr_data["addrs"] = list(self._addrs())

            settings = dict(htr_data.get("settings") or {})
            settings[addr] = payload
            htr_data["settings"] = settings
            dev_data["htr"] = htr_data
            new_data[dev_id] = dev_data
            self.async_set_updated_data(new_data)
            success = True

        except TimeoutError as err:
            _LOGGER.error(
                "Timeout refreshing heater settings for device %s heater %s",
                dev_id,
                addr,
                exc_info=err,
            )
        except (ClientError, TermoWebRateLimitError, TermoWebAuthError) as err:
            _LOGGER.error(
                "Failed to refresh heater settings for device %s heater %s: %s",
                dev_id,
                addr,
                err,
                exc_info=err,
            )
        finally:
            _LOGGER.info(
                "Finished heater settings refresh for device %s heater %s (success=%s)",
                dev_id,
                addr,
                success,
            )

    async def _async_update_data(self) -> dict[str, dict[str, Any]]:
        dev_id = self._dev_id
        addrs = self._addrs()
        try:
            prev_dev = (self.data or {}).get(dev_id, {})
            prev_htr = prev_dev.get("htr") or {}
            settings_map: dict[str, Any] = dict(prev_htr.get("settings") or {})

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
                    except (
                        ClientError,
                        TermoWebRateLimitError,
                        TermoWebAuthError,
                    ) as err:
                        _LOGGER.debug(
                            "Error fetching settings for heater %s: %s",
                            addr,
                            err,
                            exc_info=err,
                        )
                        # keep previous settings on error
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
                counter = _as_float(last.get("counter"))
                t = _as_float(last.get("t"))
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
