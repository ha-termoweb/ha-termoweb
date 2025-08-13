from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from typing import Any, Dict

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import aiohttp_client
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.loader import async_get_integration

from .api import TermoWebClient
from .const import (
    DOMAIN,
    DEFAULT_POLL_INTERVAL,
    HTR_ENERGY_UPDATE_INTERVAL,
    MIN_POLL_INTERVAL,
    STRETCHED_POLL_INTERVAL,
    signal_ws_data,
    signal_ws_status,
)
from .coordinator import TermoWebCoordinator
from .ws_client_legacy import TermoWebWSLegacyClient

_LOGGER = logging.getLogger(__name__)

PLATFORMS = ["button", "binary_sensor", "climate", "sensor"]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    session = aiohttp_client.async_get_clientsession(hass)
    username = entry.data["username"]
    password = entry.data["password"]
    base_interval = int(entry.options.get("poll_interval", entry.data.get("poll_interval", DEFAULT_POLL_INTERVAL)))

    # DRY version: read from manifest
    integration = await async_get_integration(hass, DOMAIN)
    version = integration.version or "unknown"

    client = TermoWebClient(session, username, password)
    devices = await client.list_devices()
    dev = devices[0] if isinstance(devices, list) and devices else {}
    dev_id = str(
        dev.get("dev_id") or dev.get("id") or dev.get("serial_id") or ""
    ).strip()
    nodes = await client.get_nodes(dev_id)
    addrs: list[str] = []
    node_list = nodes.get("nodes") if isinstance(nodes, dict) else None
    if isinstance(node_list, list):
        for n in node_list:
            if isinstance(n, dict) and (n.get("type") or "").lower() == "htr":
                addrs.append(str(n.get("addr")))

    coordinator = TermoWebCoordinator(hass, client, base_interval, dev_id, dev, nodes)

    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = data = {
        "client": client,
        "coordinator": coordinator,
        "dev_id": dev_id,
        "nodes": nodes,
        "htr_addrs": addrs,
        "base_poll_interval": max(base_interval, MIN_POLL_INTERVAL),
        "stretched": False,
        "ws_tasks": {},     # dev_id -> asyncio.Task
        "ws_clients": {},   # dev_id -> TermoWebWSLegacyClient
        "ws_state": {},     # dev_id -> status attrs
        "version": version,
    }

    async def _start_ws(dev_id: str) -> None:
        tasks: Dict[str, asyncio.Task] = data["ws_tasks"]
        clients: Dict[str, TermoWebWSLegacyClient] = data["ws_clients"]
        if dev_id in tasks and not tasks[dev_id].done():
            return
        ws_client = clients.get(dev_id)
        if not ws_client:
            ws_client = TermoWebWSLegacyClient(
                hass,
                entry_id=entry.entry_id,
                dev_id=dev_id,
                api_client=client,
                coordinator=coordinator,
            )
            clients[dev_id] = ws_client
        task = ws_client.start()
        tasks[dev_id] = task
        _LOGGER.info("WS: started read-only client for %s", dev_id)

    def _recalc_poll_interval() -> None:
        """Stretch polling if all running WS clients report healthy; else restore."""
        stretched = data["stretched"]
        tasks: Dict[str, asyncio.Task] = data["ws_tasks"]
        state: Dict[str, Dict[str, Any]] = data["ws_state"]

        if not tasks:
            if stretched:
                coordinator.update_interval = timedelta(seconds=data["base_poll_interval"])
                data["stretched"] = False
            return

        all_healthy = True
        for dev_id, task in tasks.items():
            if task.done():
                all_healthy = False
                break
            s = state.get(dev_id) or {}
            if s.get("status") != "healthy":
                all_healthy = False
                break

        if all_healthy and not stretched:
            coordinator.update_interval = timedelta(seconds=STRETCHED_POLL_INTERVAL)
            data["stretched"] = True
            _LOGGER.info("WS: healthy for ≥5m; stretching REST polling to %ss", STRETCHED_POLL_INTERVAL)
        elif (not all_healthy) and stretched:
            coordinator.update_interval = timedelta(seconds=data["base_poll_interval"])
            data["stretched"] = False
            _LOGGER.info("WS: no longer healthy; restoring REST polling to %ss", data["base_poll_interval"])

    data["recalc_poll"] = _recalc_poll_interval

    def _on_ws_status(_payload: dict) -> None:
        _recalc_poll_interval()

    unsub = async_dispatcher_connect(hass, signal_ws_status(entry.entry_id), _on_ws_status)
    data["unsub_ws_status"] = unsub

    def _on_ws_data(payload: dict) -> None:
        if payload.get("kind") == "htr_samples":
            energy_coordinator = data.get("energy_coordinator")
            if energy_coordinator:
                energy_coordinator.update_interval = HTR_ENERGY_UPDATE_INTERVAL
                hass.async_create_task(energy_coordinator.async_request_refresh())

    unsub_data = async_dispatcher_connect(hass, signal_ws_data(entry.entry_id), _on_ws_data)
    data["unsub_ws_data"] = unsub_data

    # First refresh (inventory etc.)
    await coordinator.async_config_entry_first_refresh()

    # Always-on push: start for all current devices
    for dev_id in (coordinator.data or {}).keys():
        hass.async_create_task(_start_ws(dev_id))

    # Start for any devices discovered later
    def _on_coordinator_updated() -> None:
        for dev_id in (coordinator.data or {}).keys():
            if dev_id not in data["ws_tasks"]:
                hass.async_create_task(_start_ws(dev_id))

    coordinator.async_add_listener(_on_coordinator_updated)

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    _LOGGER.info("TermoWeb setup complete (v%s)", version)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    domain_data = hass.data.get(DOMAIN)
    rec = domain_data.get(entry.entry_id) if domain_data else None
    if not rec:
        return True

    # Cancel WS tasks and close clients
    for dev_id, task in list(rec.get("ws_tasks", {}).items()):
        try:
            task.cancel()
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            _LOGGER.exception("WS task for %s failed to cancel cleanly", dev_id)

    for dev_id, client in list(rec.get("ws_clients", {}).items()):
        try:
            await client.stop()
        except Exception:
            _LOGGER.exception("WS client for %s failed to stop", dev_id)

    if "unsub_ws_status" in rec and callable(rec["unsub_ws_status"]):
        rec["unsub_ws_status"]()
    if "unsub_ws_data" in rec and callable(rec["unsub_ws_data"]):
        rec["unsub_ws_data"]()

    ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    if ok and domain_data:
        domain_data.pop(entry.entry_id, None)

    return ok


async def async_migrate_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    # No structured migrations needed yet
    return True


async def async_update_entry_options(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Options updated (only poll_interval remains); recompute interval if needed."""
    rec = hass.data[DOMAIN][entry.entry_id]
    rec["recalc_poll"]()
