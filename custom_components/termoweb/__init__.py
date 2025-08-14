from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
import time
from typing import Any, Dict, Iterable

from homeassistant.components.recorder.statistics import (
    async_import_statistics,
    async_update_statistics_metadata,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EVENT_HOMEASSISTANT_STARTED
from homeassistant.core import HomeAssistant
from homeassistant.helpers import aiohttp_client, entity_registry as er
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.loader import async_get_integration

from .api import TermoWebClient
from .const import (
    DOMAIN,
    DEFAULT_POLL_INTERVAL,
    MIN_POLL_INTERVAL,
    STRETCHED_POLL_INTERVAL,
    signal_ws_status,
)
from .coordinator import TermoWebCoordinator
from .ws_client_legacy import TermoWebWSLegacyClient

_LOGGER = logging.getLogger(__name__)

PLATFORMS = ["button", "binary_sensor", "climate", "sensor"]

OPTION_ENERGY_HISTORY_IMPORTED = "energy_history_imported"
OPTION_ENERGY_HISTORY_PROGRESS = "energy_history_progress"
OPTION_MAX_HISTORY_RETRIEVED = "max_history_retrieved"

DEFAULT_MAX_HISTORY_DAYS = 7

# Guard htr/samples API usage
_SAMPLES_QUERY_LOCK = asyncio.Lock()
_LAST_SAMPLES_QUERY = 0.0


def _iso_date(ts: int) -> str:
    """Convert unix timestamp to ISO date."""
    return datetime.fromtimestamp(ts, timezone.utc).date().isoformat()


async def _async_import_energy_history(
    hass: HomeAssistant,
    entry: ConfigEntry,
    addrs: Iterable[str] | None = None,
    *,
    reset_progress: bool = False,
    max_days: int | None = None,
) -> None:
    """Fetch historical hourly samples and insert statistics."""
    rec = hass.data.get(DOMAIN, {}).get(entry.entry_id)
    if not rec:
        _LOGGER.debug("%s: no record found for energy import", entry.entry_id)
        return
    client: TermoWebClient = rec["client"]
    dev_id: str = rec["dev_id"]
    all_addrs: list[str] = rec.get("htr_addrs", [])
    target_addrs = all_addrs if addrs is None else [a for a in all_addrs if a in set(str(x) for x in addrs)]

    day = 24 * 3600
    now_ts = int(time.time())
    if max_days is None:
        max_days = int(entry.options.get(OPTION_MAX_HISTORY_RETRIEVED, DEFAULT_MAX_HISTORY_DAYS))
    target = now_ts - max_days * day
    progress: Dict[str, int] = dict(entry.options.get(OPTION_ENERGY_HISTORY_PROGRESS, {}))

    if reset_progress:
        if addrs is None:
            progress.clear()
        else:
            for addr in target_addrs:
                progress.pop(addr, None)
        options = dict(entry.options)
        options[OPTION_ENERGY_HISTORY_PROGRESS] = progress
        options.pop(OPTION_ENERGY_HISTORY_IMPORTED, None)
        hass.config_entries.async_update_entry(entry, options=options)
    elif entry.options.get(OPTION_ENERGY_HISTORY_IMPORTED):
        _LOGGER.debug("%s: energy history already imported", entry.entry_id)
        return

    _LOGGER.debug("%s: importing hourly samples down to %s", dev_id, _iso_date(target))

    async def _rate_limited_fetch(addr: str, start: int, stop: int) -> list[dict[str, Any]]:
        global _LAST_SAMPLES_QUERY
        async with _SAMPLES_QUERY_LOCK:
            now = time.monotonic()
            wait = 1 - (now - _LAST_SAMPLES_QUERY)
            if wait > 0:
                _LOGGER.debug(
                    "%s/%s: sleeping %.2fs before query", addr, _iso_date(start), wait
                )
                await asyncio.sleep(wait)
            _LAST_SAMPLES_QUERY = time.monotonic()
        _LOGGER.debug(
            "%s: requesting samples %s-%s", addr, _iso_date(start), _iso_date(stop)
        )
        try:
            return await client.get_htr_samples(dev_id, addr, start, stop)
        except Exception as err:  # pragma: no cover - defensive
            _LOGGER.debug("%s: error fetching samples: %s", addr, err)
            return []

    ent_reg: er.EntityRegistry | None = er.async_get(hass)
    for addr in target_addrs:
        _LOGGER.debug("%s: importing history for heater %s", dev_id, addr)
        stats: list[dict[str, Any]] = []
        start_ts = int(progress.get(addr, now_ts))
        while start_ts > target:
            chunk_start = max(start_ts - day, target)
            samples = await _rate_limited_fetch(addr, chunk_start, start_ts)

            _LOGGER.debug(
                "%s: fetched %d samples for %s-%s",
                addr,
                len(samples),
                _iso_date(chunk_start),
                _iso_date(start_ts),
            )

            for sample in samples:
                t = sample.get("t")
                counter = sample.get("counter")
                try:
                    ts = int(t)
                    kwh = float(counter) / 1000.0
                except (TypeError, ValueError):
                    _LOGGER.debug("%s: invalid sample %s", addr, sample)
                    continue
                stats.append(
                    {
                        "start": datetime.fromtimestamp(ts, timezone.utc),
                        "state": kwh,
                        "sum": kwh,
                    }
                )

            start_ts = chunk_start
            progress[addr] = start_ts
            options = dict(entry.options)
            options[OPTION_ENERGY_HISTORY_PROGRESS] = progress
            hass.config_entries.async_update_entry(entry, options=options)

        if not stats:
            _LOGGER.debug("%s: no samples fetched", addr)
            continue

        uid = f"{DOMAIN}:{dev_id}:htr:{addr}:energy"
        entity_id = ent_reg.async_get_entity_id("sensor", DOMAIN, uid) if ent_reg else None
        if not entity_id:
            _LOGGER.debug("%s: no energy sensor found", addr)
            continue
        _LOGGER.debug("%s: inserting statistics for %s", addr, entity_id)
        ent_entry = ent_reg.async_get(entity_id) if ent_reg else None
        name = getattr(ent_entry, "original_name", None) or entity_id

        metadata = {
            "source": DOMAIN,
            "statistic_id": entity_id,
            "unit_of_measurement": "kWh",
            "name": name,
            "has_sum": True,
        }
        _LOGGER.debug("%s: adding %d stats entries", addr, len(stats))
        try:
            async_update_statistics_metadata(hass, metadata)
            stat_list = [{"statistic_id": entity_id, **s} for s in stats]
            async_import_statistics(hass, stat_list)
        except Exception as err:  # pragma: no cover - log & continue
            _LOGGER.exception(
                "%s: async_import_statistics failed: %s",
                addr,
                err,
            )

    options = dict(entry.options)
    options[OPTION_ENERGY_HISTORY_PROGRESS] = progress
    if all(progress.get(addr, now_ts) <= target for addr in all_addrs):
        options[OPTION_ENERGY_HISTORY_IMPORTED] = True
    hass.config_entries.async_update_entry(entry, options=options)
    _LOGGER.debug("%s: energy import complete", entry.entry_id)


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
        "config_entry": entry,
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

    async def _service_import_energy_history(call) -> None:
        _LOGGER.debug("service import_energy_history called")
        reset = bool(call.data.get("reset_progress", False))
        max_days = call.data.get("max_history_retrieval")
        ent_ids = call.data.get("entity_id")
        tasks = []
        if ent_ids:
            ent_reg = er.async_get(hass)
            if isinstance(ent_ids, str):
                ent_ids = [ent_ids]
            entry_map: Dict[str, set[str]] = {}
            for eid in ent_ids:
                er_ent = ent_reg.async_get(eid)
                if not er_ent or er_ent.platform != DOMAIN:
                    continue
                parts = (er_ent.unique_id or "").split(":")
                if len(parts) >= 4 and parts[0] == DOMAIN and parts[2] == "htr":
                    entry_map.setdefault(er_ent.config_entry_id, set()).add(parts[3])
            for entry_id, addr_set in entry_map.items():
                ent = hass.config_entries.async_get_entry(entry_id)
                if ent:
                    tasks.append(
                        _async_import_energy_history(
                            hass,
                            ent,
                            addr_set,
                            reset_progress=reset,
                            max_days=max_days,
                        )
                    )
        else:
            for rec in hass.data.get(DOMAIN, {}).values():
                ent: ConfigEntry | None = rec.get("config_entry")
                if ent:
                    tasks.append(
                        _async_import_energy_history(
                            hass,
                            ent,
                            None,
                            reset_progress=reset,
                            max_days=max_days,
                        )
                    )
        if tasks:
            _LOGGER.debug("import_energy_history: awaiting %d tasks", len(tasks))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, Exception):
                    _LOGGER.exception("import_energy_history task failed: %s", res)

    if not hass.services.has_service(DOMAIN, "import_energy_history"):
        hass.services.async_register(
            DOMAIN, "import_energy_history", _service_import_energy_history
        )

    if not entry.options.get(OPTION_ENERGY_HISTORY_IMPORTED):
        _LOGGER.debug("%s: scheduling initial energy import", entry.entry_id)

        async def _schedule_import(_event: Any | None = None) -> None:
            await _async_import_energy_history(hass, entry)

        if hass.is_running:
            hass.async_create_task(_schedule_import())
        else:
            hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STARTED, _schedule_import)

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
