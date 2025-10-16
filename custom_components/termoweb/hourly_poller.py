"""Hourly REST poller for historical samples across all backends."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from contextlib import suppress
from datetime import datetime, timedelta
from itertools import chain
import logging
import random
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

try:  # pragma: no cover - optional helper on older Home Assistant cores
    from homeassistant.helpers.event import async_track_time_change
except ImportError:  # pragma: no cover - tests provide stubbed helpers
    async_track_time_change = None  # type: ignore[assignment]

from custom_components.termoweb.backend import Backend
from custom_components.termoweb.backend.sanitize import mask_identifier
from custom_components.termoweb.coordinator import EnergyStateCoordinator
from custom_components.termoweb.inventory import (
    Inventory,
    normalize_node_addr,
    normalize_node_type,
)

_LOGGER = logging.getLogger(__name__)


class HourlySamplesPoller:
    """Schedule and execute hourly REST polls for canonical sample buckets."""

    def __init__(
        self,
        hass: HomeAssistant,
        coordinator: EnergyStateCoordinator,
        backend: Backend,
        inventory: Inventory | Iterable[Inventory],
    ) -> None:
        """Initialise the poller with Home Assistant context and dependencies."""

        self._hass = hass
        self._coordinator = coordinator
        self._backend = backend
        if isinstance(inventory, Inventory):
            self._inventories: tuple[Inventory, ...] = (inventory,)
        else:
            validated: list[Inventory] = []
            for item in inventory:
                if not isinstance(item, Inventory):
                    msg = "HourlySamplesPoller requires Inventory instances"
                    raise TypeError(msg)
                validated.append(item)
            if not validated:
                msg = "At least one Inventory is required for hourly polling"
                raise ValueError(msg)
            self._inventories = tuple(validated)
        self._remove_listener: Callable[[], None] | None = None
        self._active_task: asyncio.Task[None] | None = None
        self._semaphore = asyncio.Semaphore(3)
        self._last_window_end: datetime | None = None

    async def async_setup(self) -> None:
        """Register the HH:05 local trigger and perform optional catch-up."""

        helper = async_track_time_change
        if helper is None:
            _LOGGER.debug(
                "Hourly poller: time-change helper unavailable; scheduling skipped"
            )
        elif self._remove_listener is None:
            self._remove_listener = helper(
                self._hass,
                self._on_time,
                minute=5,
                second=0,
            )

        now_local = dt_util.now()
        if now_local.minute < 15:
            await self._run_for_previous_hour(now_local)

    async def async_shutdown(self) -> None:
        """Cancel scheduled triggers and pending polling tasks."""

        if callable(self._remove_listener):
            try:
                self._remove_listener()
            except Exception:  # noqa: BLE001 - defensive logging only
                _LOGGER.debug(
                    "Hourly poller: failed to remove time listener", exc_info=True
                )
            self._remove_listener = None

        task = self._active_task
        self._active_task = None
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    def _on_time(self, now: datetime | None) -> None:
        """Callback invoked by Home Assistant's time tracker in a thread-safe way."""

        reference = dt_util.as_local(now) if now is not None else dt_util.now()

        loop = self._hass.loop

        def _schedule() -> None:
            if self._active_task is not None and not self._active_task.done():
                _LOGGER.debug(
                    "Hourly poller: skipping trigger while previous run is active"
                )
                return
            try:
                task = loop.create_task(self._run_for_previous_hour(reference))
            except RuntimeError:
                _LOGGER.exception(
                    "Hourly poller: failed to schedule run for %s", reference
                )
                return
            self._active_task = task

            def _finalise(finished: asyncio.Task[None]) -> None:
                self._active_task = None
                if finished.cancelled():
                    _LOGGER.debug("Hourly poller task cancelled")
                    return
                exception = finished.exception()
                if exception is not None:
                    _LOGGER.exception(
                        "Hourly poller run raised an exception", exc_info=exception
                    )

            task.add_done_callback(_finalise)

        loop.call_soon_threadsafe(_schedule)

    async def _run_for_previous_hour(self, now_local: datetime) -> None:
        """Fetch and merge samples for the hour preceding ``now_local``."""

        reference = dt_util.as_local(now_local)
        current_hour_start = reference.replace(minute=0, second=0, microsecond=0)
        end_local = current_hour_start
        start_local = end_local - timedelta(hours=1)

        if self._last_window_end is not None and end_local <= self._last_window_end:
            _LOGGER.debug(
                "Hourly poller: window %s already processed", end_local.isoformat()
            )
            return

        total_nodes = 0
        device_total = len(self._inventories)
        tasks: list[asyncio.Task[None]] = []
        for container in self._inventories:
            heater_targets = container.heater_sample_targets
            power_targets = container.power_monitor_sample_targets
            if not heater_targets and not power_targets:
                continue
            total_nodes += len(heater_targets) + len(power_targets)
            tasks.append(
                asyncio.create_task(
                    self._poll_device(
                        container.dev_id,
                        chain(heater_targets, power_targets),
                        start_local,
                        end_local,
                    )
                )
            )

        _LOGGER.info(
            "Hourly samples poll: window=%s–%s devices=%d nodes=%d",
            start_local.isoformat(),
            end_local.isoformat(),
            device_total,
            total_nodes,
        )

        if tasks:
            await asyncio.gather(*tasks)

        self._last_window_end = end_local

    async def _poll_device(
        self,
        dev_id: str,
        nodes: Iterable[tuple[str, str]],
        start_local: datetime,
        end_local: datetime,
    ) -> None:
        """Fetch samples for ``dev_id`` and merge them into the coordinator."""

        node_pairs = tuple(
            (
                normalize_node_type(node_type, use_default_when_falsey=True),
                normalize_node_addr(addr, use_default_when_falsey=True),
            )
            for node_type, addr in nodes
        )
        node_pairs = tuple(
            (node_type, addr)
            for node_type, addr in node_pairs
            if node_type and addr
        )

        if not node_pairs:
            return

        attempt = 1
        result: dict[tuple[str, str], list[dict[str, Any]]] | None = None
        while attempt <= 2:
            try:
                async with self._semaphore:
                    result = await self._backend.fetch_hourly_samples(
                        dev_id,
                        node_pairs,
                        start_local,
                        end_local,
                    )
            except asyncio.CancelledError:
                raise
            except Exception as err:  # noqa: BLE001 - retry once before giving up
                delay = random.uniform(0.5, 1.5)
                _LOGGER.warning(
                    "Hourly poller: fetch failed for %s (attempt %d/%d): %s",
                    mask_identifier(dev_id),
                    attempt,
                    2,
                    err,
                )
                if attempt >= 2:
                    return
                attempt += 1
                await asyncio.sleep(delay)
                continue
            break

        if not result:
            _LOGGER.debug(
                "Hourly poller: backend returned no data for %s",
                mask_identifier(dev_id),
            )
            return

        totals = sum(len(bucket) for bucket in result.values())
        _LOGGER.debug(
            "Hourly poller: %s merged nodes=%d buckets=%d",
            mask_identifier(dev_id),
            len(result),
            totals,
        )

        try:
            await self._coordinator.merge_samples_for_window(dev_id, result)
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - defensive logging
            _LOGGER.exception(
                "Hourly poller: coordinator merge failed for %s",
                mask_identifier(dev_id),
            )


__all__ = ["HourlySamplesPoller"]
