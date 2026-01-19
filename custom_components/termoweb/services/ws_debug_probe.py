"""Service for websocket debug probe dispatch."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Iterable, Mapping
import logging
from typing import Any

from homeassistant.core import HomeAssistant, ServiceCall

from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.inventory import Inventory
from custom_components.termoweb.runtime import EntryRuntime

_LOGGER = logging.getLogger(__name__)


async def async_register_ws_debug_probe_service(hass: HomeAssistant) -> None:
    """Register the ws_debug_probe debug helper service."""

    if hass.services.has_service(DOMAIN, "ws_debug_probe"):
        return

    async def _async_ws_debug_probe(call: ServiceCall) -> None:
        """Emit a websocket dev_data probe for debugging."""

        entry_filter = call.data.get("entry_id")
        dev_filter = call.data.get("dev_id")
        domain_records = hass.data.get(DOMAIN, {})
        if not isinstance(domain_records, Mapping):
            _LOGGER.debug("ws_debug_probe: integration data unavailable")
            return

        entries: list[tuple[str, EntryRuntime | Mapping[str, Any]]] = []
        if entry_filter:
            record = domain_records.get(entry_filter)
            if isinstance(record, (EntryRuntime, Mapping)):
                entries.append((entry_filter, record))
        else:
            entries = [
                (entry_id, rec)
                for entry_id, rec in domain_records.items()
                if isinstance(rec, (EntryRuntime, Mapping))
            ]

        if not entries:
            _LOGGER.debug("ws_debug_probe: no matching config entries")
            return

        tasks: list[Awaitable[Any]] = []
        for entry_id, runtime in entries:
            if isinstance(runtime, EntryRuntime):
                debug_enabled = runtime.debug
                clients = runtime.ws_clients
                inventory_obj = runtime.inventory
            else:
                debug_enabled = bool(runtime.get("debug", False))
                clients = runtime.get("ws_clients")
                inventory_obj = runtime.get("inventory")

            if not debug_enabled:
                _LOGGER.debug(
                    "ws_debug_probe: debug helpers disabled for entry %s",
                    entry_id,
                )
                continue
            if not clients:
                _LOGGER.debug(
                    "ws_debug_probe: no websocket clients for entry %s",
                    entry_id,
                )
                continue
            if dev_filter:
                target_dev_ids: Iterable[str] = [str(dev_filter)]
            else:
                target_dev_ids = [str(dev) for dev in clients]
            for dev_id in target_dev_ids:
                client = clients.get(dev_id)
                if client is None:
                    _LOGGER.debug(
                        "ws_debug_probe: websocket client missing for %s/%s",
                        entry_id,
                        dev_id,
                    )
                    continue
                probe = getattr(client, "debug_probe", None)
                if probe is None:
                    _LOGGER.debug(
                        "ws_debug_probe: client %s/%s has no debug_probe",
                        entry_id,
                        dev_id,
                    )
                    continue
                if not isinstance(inventory_obj, Inventory):
                    inventory_obj = None
                try:
                    result = probe(inventory_obj)
                except TypeError as err:
                    try:
                        result = probe()
                    except TypeError:
                        _LOGGER.debug(
                            "ws_debug_probe: client %s/%s probe invocation failed: %s",
                            entry_id,
                            dev_id,
                            err,
                        )
                        continue
                if asyncio.iscoroutine(result):
                    tasks.append(result)
                else:
                    _LOGGER.debug(
                        "ws_debug_probe: client %s/%s returned non-awaitable probe",
                        entry_id,
                        dev_id,
                    )

        if not tasks:
            _LOGGER.debug("ws_debug_probe: no matching websocket clients to probe")
            return

        _LOGGER.debug("ws_debug_probe: awaiting %d websocket probe(s)", len(tasks))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, asyncio.CancelledError):
                raise res
            if isinstance(res, Exception):
                _LOGGER.debug("ws_debug_probe: probe raised %s", res)

    hass.services.async_register(
        DOMAIN,
        "ws_debug_probe",
        _async_ws_debug_probe,
    )
