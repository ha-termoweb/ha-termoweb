"""Service wiring for energy history import."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from custom_components.termoweb import energy as energy_module
from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.energy import (
    async_import_energy_history as _async_import_energy_history_impl,
)
from custom_components.termoweb.inventory import Inventory
from custom_components.termoweb.runtime import EntryRuntime
from custom_components.termoweb.throttle import default_samples_rate_limit_state

_LOGGER = energy_module._LOGGER  # noqa: SLF001


async def async_import_energy_history_with_rate_limit(
    hass: HomeAssistant,
    entry: ConfigEntry,
    *,
    nodes: Inventory | None = None,
    node_types: Iterable[str] | None = None,
    addresses: Iterable[str] | None = None,
    day_chunk_hours: int = 24,
    reset_progress: bool = False,
    max_days: int | None = None,
) -> None:
    """Run energy history import with rate limiting and filter defaults."""

    rate_state = default_samples_rate_limit_state()
    kwargs: dict[str, Any] = {
        "reset_progress": reset_progress,
        "max_days": max_days,
        "rate_limit": rate_state,
    }

    if nodes is not None:
        kwargs["nodes"] = nodes
    if node_types is not None:
        kwargs["node_types"] = tuple(node_types)
    if addresses is not None:
        kwargs["addresses"] = tuple(addresses)
    if day_chunk_hours != 24:
        kwargs["day_chunk_hours"] = day_chunk_hours

    await _async_import_energy_history_impl(
        hass,
        entry,
        **kwargs,
    )


async def async_register_import_energy_history_service(
    hass: HomeAssistant,
    import_fn: Callable[..., Awaitable[None]],
) -> None:
    """Register the import_energy_history service if it is missing."""

    if hass.services.has_service(DOMAIN, "import_energy_history"):
        return

    logger = _LOGGER
    async_mod = asyncio

    async def _service_import_energy_history(call) -> None:
        """Handle the import_energy_history service call."""

        logger.debug("service import_energy_history called")
        reset = bool(call.data.get("reset_progress", False))
        max_days = call.data.get("max_history_retrieval")
        node_types_raw = call.data.get("node_types")
        addresses_raw = call.data.get("addresses")
        day_chunk_raw = call.data.get("day_chunk_hours", 24)

        node_types_filter: tuple[str, ...] | None
        if node_types_raw is None:
            node_types_filter = None
        elif isinstance(node_types_raw, (list, tuple, set)):
            node_types_filter = tuple(str(value) for value in node_types_raw)
        else:
            node_types_filter = (str(node_types_raw),)

        addresses_filter: tuple[str, ...] | None
        if addresses_raw is None:
            addresses_filter = None
        elif isinstance(addresses_raw, (list, tuple, set)):
            addresses_filter = tuple(str(value) for value in addresses_raw)
        else:
            addresses_filter = (str(addresses_raw),)

        try:
            day_chunk_hours = int(day_chunk_raw)
        except (TypeError, ValueError):
            logger.warning(
                "import_energy_history: invalid day_chunk_hours %s; defaulting to 24",
                day_chunk_raw,
            )
            day_chunk_hours = 24

        tasks = []
        records = hass.data.get(DOMAIN, {})
        if not isinstance(records, Mapping):
            records = {}
        for runtime in records.values():
            if not isinstance(runtime, EntryRuntime):
                continue
            ent = runtime.config_entry
            inventory = runtime.inventory
            if not isinstance(inventory, Inventory):
                entry_entry_id = getattr(ent, "entry_id", "<unknown>")
                logger.error(
                    "%s: energy import aborted; inventory missing in integration state (entry=%s)",
                    runtime.dev_id,
                    entry_entry_id,
                )
                continue
            kwargs: dict[str, Any] = {
                "reset_progress": reset,
                "max_days": max_days,
            }
            if node_types_filter is not None:
                kwargs["node_types"] = node_types_filter
            if addresses_filter is not None:
                kwargs["addresses"] = addresses_filter
            if day_chunk_hours != 24:
                kwargs["day_chunk_hours"] = day_chunk_hours

            try:
                coro = import_fn(
                    hass,
                    ent,
                    **kwargs,
                )
            except ValueError as err:
                logger.error(
                    "%s: import_energy_history rejected input: %s",
                    runtime.dev_id,
                    err,
                )
                continue
            tasks.append(coro)
        if tasks:
            logger.debug("import_energy_history: awaiting %d tasks", len(tasks))
            results = await async_mod.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, async_mod.CancelledError):
                    raise res
                if isinstance(res, Exception):
                    logger.exception("import_energy_history task failed: %s", res)

    hass.services.async_register(
        DOMAIN,
        "import_energy_history",
        _service_import_energy_history,
    )
