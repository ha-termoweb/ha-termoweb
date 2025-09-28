from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta
import logging
import time
from typing import Any

from aiohttp import ClientError
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from homeassistant.helpers.dispatcher import async_dispatcher_connect

from . import energy as energy_module
from .api import BackendAuthError, BackendRateLimitError, RESTClient
from .backend import Backend, WsClientProto, create_backend
from .client import async_list_devices_with_logging, create_rest_client
from .const import (
    BRAND_DUCAHEAT,
    CONF_BRAND,
    DEFAULT_BRAND,
    DEFAULT_POLL_INTERVAL,
    DOMAIN,
    MIN_POLL_INTERVAL,
    STRETCHED_POLL_INTERVAL,
    signal_ws_status,
)
from .coordinator import StateCoordinator
from .energy import (
    DEFAULT_MAX_HISTORY_DAYS as ENERGY_DEFAULT_MAX_HISTORY_DAYS,
    OPTION_ENERGY_HISTORY_IMPORTED as ENERGY_OPTION_ENERGY_HISTORY_IMPORTED,
    OPTION_ENERGY_HISTORY_PROGRESS as ENERGY_OPTION_ENERGY_HISTORY_PROGRESS,
    OPTION_MAX_HISTORY_RETRIEVED as ENERGY_OPTION_MAX_HISTORY_RETRIEVED,
    async_import_energy_history as _async_import_energy_history_impl,
    async_register_import_energy_history_service,
    async_schedule_initial_energy_import,
    default_samples_rate_limit_state,
    reset_samples_rate_limit_state,
)
from .nodes import build_node_inventory
from .utils import (
    HEATER_NODE_TYPES as _HEATER_NODE_TYPES,
    async_get_integration_version as _async_get_integration_version,
    build_heater_address_map as _build_heater_address_map,
    ensure_node_inventory as _ensure_node_inventory,
    normalize_heater_addresses as _normalize_heater_addresses,
)

HEATER_NODE_TYPES = _HEATER_NODE_TYPES

OPTION_ENERGY_HISTORY_IMPORTED = ENERGY_OPTION_ENERGY_HISTORY_IMPORTED
OPTION_ENERGY_HISTORY_PROGRESS = ENERGY_OPTION_ENERGY_HISTORY_PROGRESS
OPTION_MAX_HISTORY_RETRIEVED = ENERGY_OPTION_MAX_HISTORY_RETRIEVED
DEFAULT_MAX_HISTORY_DAYS = ENERGY_DEFAULT_MAX_HISTORY_DAYS

build_heater_address_map = _build_heater_address_map
ensure_node_inventory = _ensure_node_inventory
normalize_heater_addresses = _normalize_heater_addresses

EVENT_HOMEASSISTANT_STARTED = energy_module.EVENT_HOMEASSISTANT_STARTED
er = energy_module.er
_iso_date = energy_module._iso_date
_store_statistics = energy_module._store_statistics
_statistics_during_period_compat = energy_module._statistics_during_period_compat
_get_last_statistics_compat = energy_module._get_last_statistics_compat
_clear_statistics_compat = energy_module._clear_statistics_compat

# Re-export legacy WS client for backward compatibility (tests may patch it).
from .ws_client import WebSocketClient as WebSocket09Client  # noqa: F401

_LOGGER = logging.getLogger(__name__)

PLATFORMS = ["button", "binary_sensor", "climate", "sensor"]

reset_samples_rate_limit_state()

async def _async_import_energy_history(
    hass: HomeAssistant,
    entry: ConfigEntry,
    nodes: Mapping[str, Iterable[str]] | Iterable[str] | None = None,
    *,
    reset_progress: bool = False,
    max_days: int | None = None,
) -> None:
    """Delegate to the energy helper with shared rate limiting."""

    rate_state = default_samples_rate_limit_state()
    await _async_import_energy_history_impl(
        hass,
        entry,
        nodes,
        reset_progress=reset_progress,
        max_days=max_days,
        rate_limit=rate_state,
    )


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up the TermoWeb integration for a config entry."""
    username = entry.data["username"]
    password = entry.data["password"]
    base_interval = int(
        entry.options.get(
            "poll_interval", entry.data.get("poll_interval", DEFAULT_POLL_INTERVAL)
        )
    )
    brand = entry.data.get(CONF_BRAND, DEFAULT_BRAND)

    version = await _async_get_integration_version(hass)

    client: RESTClient = create_rest_client(hass, username, password, brand)
    backend = create_backend(brand=brand, client=client)
    try:
        devices = await async_list_devices_with_logging(client)
    except BackendAuthError as err:
        raise ConfigEntryAuthFailed from err
    except (TimeoutError, ClientError, BackendRateLimitError) as err:
        raise ConfigEntryNotReady from err

    if not devices:
        _LOGGER.info("list_devices returned no devices")
        raise ConfigEntryNotReady

    dev = devices[0] if isinstance(devices, list) and devices else {}
    dev_id = str(
        dev.get("dev_id") or dev.get("id") or dev.get("serial_id") or ""
    ).strip()
    nodes = await client.get_nodes(dev_id)
    node_inventory = build_node_inventory(nodes)

    if node_inventory:
        type_counts = Counter(node.type for node in node_inventory)
        summary = ", ".join(f"{node_type}:{count}" for node_type, count in sorted(type_counts.items()))
    else:
        summary = "none"
    _LOGGER.info("%s: discovered node types: %s", dev_id, summary)

    coordinator = StateCoordinator(
        hass,
        client,
        base_interval,
        dev_id,
        dev,
        nodes,
        node_inventory,
    )

    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = data = {
        "backend": backend,
        "client": backend.client,
        "coordinator": coordinator,
        "dev_id": dev_id,
        "nodes": nodes,
        "node_inventory": node_inventory,
        "config_entry": entry,
        "base_poll_interval": max(base_interval, MIN_POLL_INTERVAL),
        "stretched": False,
        "ws_tasks": {},  # dev_id -> asyncio.Task
        "ws_clients": {},  # dev_id -> WS clients
        "ws_state": {},  # dev_id -> status attrs
        "version": version,
        "brand": brand,
    }

    async def _start_ws(dev_id: str) -> None:
        """Ensure a websocket client exists and is running for ``dev_id``."""
        backend: Backend = data["backend"]
        tasks: dict[str, asyncio.Task] = data["ws_tasks"]
        clients: dict[str, WsClientProto] = data["ws_clients"]
        if dev_id in tasks and not tasks[dev_id].done():
            return
        ws_client = clients.get(dev_id)
        if not ws_client:
            ws_client = backend.create_ws_client(
                hass,
                entry_id=entry.entry_id,
                dev_id=dev_id,
                coordinator=coordinator,
            )
            clients[dev_id] = ws_client
        task = ws_client.start()
        tasks[dev_id] = task
        _LOGGER.info("WS: started read-only client for %s", dev_id)

    def _recalc_poll_interval() -> None:
        """Stretch polling if all running WS clients report healthy; else restore."""
        stretched = data["stretched"]
        tasks: dict[str, asyncio.Task] = data["ws_tasks"]
        state: dict[str, dict[str, Any]] = data["ws_state"]

        if not tasks:
            if stretched:
                coordinator.update_interval = timedelta(
                    seconds=data["base_poll_interval"]
                )
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
            _LOGGER.info(
                "WS: healthy for ≥5m; stretching REST polling to %ss",
                STRETCHED_POLL_INTERVAL,
            )
        elif (not all_healthy) and stretched:
            coordinator.update_interval = timedelta(seconds=data["base_poll_interval"])
            data["stretched"] = False
            _LOGGER.info(
                "WS: no longer healthy; restoring REST polling to %ss",
                data["base_poll_interval"],
            )

    data["recalc_poll"] = _recalc_poll_interval

    def _on_ws_status(_payload: dict) -> None:
        """Recalculate polling intervals when websocket status changes."""
        _recalc_poll_interval()

    unsub = async_dispatcher_connect(
        hass, signal_ws_status(entry.entry_id), _on_ws_status
    )
    data["unsub_ws_status"] = unsub

    # First refresh (inventory etc.)
    await coordinator.async_config_entry_first_refresh()

    # Always-on push: start for all current devices
    for dev_id in (coordinator.data or {}).keys():
        hass.async_create_task(_start_ws(dev_id))

    # Start for any devices discovered later
    def _on_coordinator_updated() -> None:
        """Start websocket clients for newly discovered devices."""
        for dev_id in (coordinator.data or {}).keys():
            if dev_id not in data["ws_tasks"]:
                hass.async_create_task(_start_ws(dev_id))

    coordinator.async_add_listener(_on_coordinator_updated)

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    await async_register_import_energy_history_service(
        hass,
        _async_import_energy_history,
    )

    async_schedule_initial_energy_import(
        hass,
        entry,
        _async_import_energy_history,
    )

    _LOGGER.info("TermoWeb setup complete (v%s)", version)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry for TermoWeb."""
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

    ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    if ok and domain_data:
        domain_data.pop(entry.entry_id, None)

    return ok


async def async_migrate_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Migrate a config entry; no migrations are needed yet."""
    return True


async def async_update_entry_options(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options updates; recompute interval if needed."""
    rec = hass.data[DOMAIN][entry.entry_id]
    rec["recalc_poll"]()
