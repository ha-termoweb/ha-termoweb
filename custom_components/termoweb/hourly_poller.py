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
            _LOGGER.debug("Hourly poller: time-change helper unavailable; scheduling skipped")
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
                _LOGGER.debug("Hourly poller: failed to remove time listener", exc_info=True)
            self._remove_listener = None

        task = self._active_task
        self._active_task = None
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    def _on_time(self, now: datetime | None) -> None:
        """Callback invoked by Home Assistant's time tracker."""

        reference = dt_util.as_local(now) if now is not None else dt_util.now()
        if self._active_task is not None and not self._active_task.done():
            _LOGGER.debug("Hourly poller: skipping trigger while previous run is active")
            return
        task = self._hass.async_create_task(self._run_for_previous_hour(reference))
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

    def _enumerate_nodes(self, inventory: Inventory) -> list[tuple[str, str]]:
        """Return canonical ``(node_type, addr)`` pairs for ``inventory``."""

        nodes: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for node_type, addr in chain(
            inventory.heater_sample_targets,
            inventory.power_monitor_sample_targets,
        ):
            normalized_type = normalize_node_type(
                node_type,
                use_default_when_falsey=True,
            )
            normalized_addr = normalize_node_addr(
                addr,
                use_default_when_falsey=True,
            )
            if not normalized_type or not normalized_addr:
                continue
            pair = (normalized_type, normalized_addr)
            if pair in seen:
                continue
            seen.add(pair)
            nodes.append(pair)
        return nodes

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

        device_nodes: dict[str, list[tuple[str, str]]] = {}
        total_nodes = 0
        for container in self._inventories:
            nodes = self._enumerate_nodes(container)
            device_nodes[container.dev_id] = nodes
            total_nodes += len(nodes)

        _LOGGER.info(
            "Hourly samples poll: window=%s–%s devices=%d nodes=%d",
            start_local.isoformat(),
            end_local.isoformat(),
            len(device_nodes),
            total_nodes,
        )

        tasks = [
            asyncio.create_task(
                self._poll_device(
                    dev_id,
                    nodes,
                    start_local,
                    end_local,
                )
            )
            for dev_id, nodes in device_nodes.items()
            if nodes
        ]

        if tasks:
            await asyncio.gather(*tasks)

        self._last_window_end = end_local

    async def _poll_device(
        self,
        dev_id: str,
        nodes: list[tuple[str, str]],
        start_local: datetime,
        end_local: datetime,
    ) -> None:
        """Fetch samples for ``dev_id`` and merge them into the coordinator."""

        if not nodes:
            return

        attempt = 1
        result: dict[tuple[str, str], list[dict[str, Any]]] | None = None
        while attempt <= 2:
            try:
                async with self._semaphore:
                    result = await self._backend.fetch_hourly_samples(
                        dev_id,
                        nodes,
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
                "Hourly poller: backend returned no data for %s", mask_identifier(dev_id)
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
                "Hourly poller: coordinator merge failed for %s", mask_identifier(dev_id)
            )


__all__ = ["HourlySamplesPoller"]
