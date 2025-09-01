from __future__ import annotations

import asyncio
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta, timezone
import logging
import time
from typing import Any, Dict, List, Optional

# Import of recorder statistics helpers is deferred until runtime in
# _store_statistics to avoid ImportError on Home Assistant versions
# that do not provide async_update_statistics_metadata or
# async_import_statistics.  See _store_statistics for details.
async_import_statistics = None  # type: ignore
async_update_statistics_metadata = None  # type: ignore
from aiohttp import ClientError
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EVENT_HOMEASSISTANT_STARTED
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from homeassistant.helpers import aiohttp_client, entity_registry as er
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.loader import async_get_integration

from .api import TermoWebAuthError, TermoWebClient, TermoWebRateLimitError
from .const import (
    DEFAULT_POLL_INTERVAL,
    DOMAIN,
    MIN_POLL_INTERVAL,
    STRETCHED_POLL_INTERVAL,
    signal_ws_status,
)
from .coordinator import TermoWebCoordinator
from .utils import extract_heater_addrs
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
    return datetime.fromtimestamp(ts, UTC).date().isoformat()


def _store_statistics(
    hass: HomeAssistant, metadata: dict[str, Any], stats: list[dict[str, Any]]
) -> None:
    """Insert statistics using recorder helpers.

    This helper dynamically determines whether Home Assistant supports
    internal statistics import (`async_update_statistics_metadata` and
    `async_import_statistics`).  If these functions are available, they
    are used to import the provided statistics for the given metadata.
    Otherwise, the statistics are stored as external statistics using
    `async_add_external_statistics`.  The metadata is adjusted for
    external statistics by converting the dotted statistic_id into
    colon-separated form and setting the source equal to the domain.
    """
    # Attempt to import the internal statistics helpers at runtime.
    try:
        # On modern Home Assistant versions async_import_statistics is available
        from homeassistant.components.recorder.statistics import (
            async_import_statistics as _import_stats,
        )
    except ImportError:
        _import_stats = None

    if _import_stats:
        # Use internal statistics API.  Import statistics for the provided
        # metadata.  Metadata updates are handled internally by the recorder.
        _import_stats(hass, metadata, stats)
        return

    # Fall back to external statistics API.  Import only when needed.
    from homeassistant.components.recorder.statistics import (
        async_add_external_statistics,
    )

    stat_id: str = metadata["statistic_id"]
    domain, obj_id = stat_id.split(".", 1)
    ext_meta = dict(metadata)
    ext_meta.update(
        {
            "statistic_id": f"{domain}:{obj_id}",
            "source": domain,
        }
    )
    async_add_external_statistics(hass, ext_meta, stats)


async def _async_import_energy_history(
    hass: HomeAssistant,
    entry: ConfigEntry,
    addrs: Iterable[str] | None = None,
    *,
    reset_progress: bool = False,
    max_days: int | None = None,
) -> None:
    """Fetch historical hourly samples and insert statistics.

    This function collects hourly counter samples from TermoWeb and
    transforms them into long‑term statistics for energy sensors.  It
    computes the cumulative sum of hourly deltas rather than using the
    raw meter value directly, filters out non‑positive deltas to reduce
    the number of imported points, and ensures timestamps are aligned
    to the top of the hour.  Statistics are then stored via the
    recorder helpers.
    """
    rec = hass.data.get(DOMAIN, {}).get(entry.entry_id)
    if not rec:
        _LOGGER.debug("%s: no record found for energy import", entry.entry_id)
        return
    client: TermoWebClient = rec["client"]
    dev_id: str = rec["dev_id"]
    all_addrs: list[str] = rec.get("htr_addrs", [])
    target_addrs = (
        all_addrs
        if addrs is None
        else [a for a in all_addrs if a in {str(x) for x in addrs}]
    )

    day = 24 * 3600
    # Determine the end of the import window.  To avoid importing
    # partial data for the current day (which can cause negative
    # consumption readings in the Energy dashboard), we import only up to
    # the start of today (00:00 in UTC).  Live polling of the sensors
    # will provide today's consumption incrementally.
    now_dt = datetime.now(UTC)
    # Compute the start of today (midnight) in UTC.  We subtract one second from
    # this value when determining the end of the import window so that the
    # 00:00 sample of the current day is not included.  Without this
    # adjustment, the importer will fetch a sample at exactly midnight for
    # the current day.  Home Assistant treats this sample as part of the
    # next day's statistics, and because most meters reset at midnight, the
    # resulting `sum` becomes lower than the previous day's final `sum`,
    # producing a negative consumption bar.  By subtracting one second we
    # ensure the final range end is 23:59:59 of the previous day.
    start_of_today = now_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    now_ts = int(start_of_today.timestamp()) - 1
    if max_days is None:
        max_days = int(
            entry.options.get(OPTION_MAX_HISTORY_RETRIEVED, DEFAULT_MAX_HISTORY_DAYS)
        )
    target = now_ts - max_days * day
    progress: dict[str, int] = dict(
        entry.options.get(OPTION_ENERGY_HISTORY_PROGRESS, {})
    )

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

    async def _rate_limited_fetch(
        addr: str, start: int, stop: int
    ) -> list[dict[str, Any]]:
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
        except asyncio.CancelledError:  # pragma: no cover - allow cancellation
            raise
        except Exception as err:  # pragma: no cover - defensive
            _LOGGER.debug("%s: error fetching samples: %s", addr, err)
            return []

    ent_reg: er.EntityRegistry | None = er.async_get(hass)
    for addr in target_addrs:
        _LOGGER.debug("%s: importing history for heater %s", dev_id, addr)
        # Collect all samples across the requested range.  We will process
        # them after the loop in chronological order.
        all_samples: list[dict[str, Any]] = []
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

            # Append all samples as-is for later processing
            all_samples.extend(samples)

            start_ts = chunk_start
            progress[addr] = start_ts
            options = dict(entry.options)
            options[OPTION_ENERGY_HISTORY_PROGRESS] = progress
            hass.config_entries.async_update_entry(entry, options=options)

        if not all_samples:
            _LOGGER.debug("%s: no samples fetched", addr)
            continue

        # Sort all samples chronologically so that we can compute deltas properly.
        all_samples_sorted = sorted(all_samples, key=lambda s: s.get("t", 0))

        # Resolve the entity_id for the heater's energy sensor
        uid = f"{DOMAIN}:{dev_id}:htr:{addr}:energy"
        entity_id = (
            ent_reg.async_get_entity_id("sensor", DOMAIN, uid) if ent_reg else None
        )
        if not entity_id:
            _LOGGER.debug("%s: no energy sensor found", addr)
            continue

        # Determine existing cumulative sum before the earliest sample
        earliest_ts = int(all_samples_sorted[0].get("t", 0))
        earliest_start_dt = datetime.fromtimestamp(earliest_ts, UTC).replace(
            minute=0, second=0, microsecond=0
        )
        sum_offset = 0.0
        try:
            from homeassistant.components.recorder.statistics import (
                async_get_last_statistics,
            )

            existing = await async_get_last_statistics(
                hass, 1, [entity_id], start_time=earliest_start_dt
            )
            if existing and (vals := existing.get(entity_id)):
                sum_offset = float(vals[0].get("sum") or 0.0)
        except asyncio.CancelledError:  # pragma: no cover - allow cancellation
            raise
        except Exception as err:  # pragma: no cover - defensive
            _LOGGER.debug("%s: error fetching last statistics: %s", addr, err)

        stats: list[dict[str, Any]] = []
        sum_kwh: float = 0.0
        previous_kwh: float | None = None
        for sample in all_samples_sorted:
            t = sample.get("t")
            counter = sample.get("counter")
            try:
                ts = int(t)
                kwh = float(counter) / 1000.0
            except (TypeError, ValueError):
                _LOGGER.debug("%s: invalid sample %s", addr, sample)
                continue
            # Align start time to the top of the hour
            start_dt = datetime.fromtimestamp(ts, UTC).replace(
                minute=0, second=0, microsecond=0
            )
            if previous_kwh is None:
                previous_kwh = kwh
                continue
            delta = kwh - previous_kwh
            # Skip zero or negative deltas to avoid importing non‑consumption hours
            if delta <= 0:
                previous_kwh = kwh
                continue
            sum_kwh += delta
            stats.append(
                {"start": start_dt, "state": None, "sum": sum_kwh + sum_offset}
            )
            previous_kwh = kwh

        if not stats:
            _LOGGER.debug("%s: no positive deltas found", addr)
            continue

        _LOGGER.debug("%s: inserting statistics for %s", addr, entity_id)
        ent_entry = ent_reg.async_get(entity_id) if ent_reg else None
        name = getattr(ent_entry, "original_name", None) or entity_id

        # Metadata for internal statistics: source must be 'recorder'
        metadata = {
            "source": "recorder",
            "statistic_id": entity_id,
            "unit_of_measurement": "kWh",
            "name": name,
            "has_sum": True,
            "has_mean": False,
        }
        _LOGGER.debug("%s: adding %d stats entries", addr, len(stats))
        try:
            _store_statistics(hass, metadata, stats)
        except Exception as err:  # pragma: no cover - log & continue
            _LOGGER.exception("%s: statistics insert failed: %s", addr, err)

    options = dict(entry.options)
    options[OPTION_ENERGY_HISTORY_PROGRESS] = progress
    # If all heaters are imported down to the target date mark import as complete
    if all(progress.get(addr, now_ts) <= target for addr in all_addrs):
        options[OPTION_ENERGY_HISTORY_IMPORTED] = True
    hass.config_entries.async_update_entry(entry, options=options)
    _LOGGER.debug("%s: energy import complete", entry.entry_id)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up the TermoWeb integration for a config entry."""
    session = aiohttp_client.async_get_clientsession(hass)
    username = entry.data["username"]
    password = entry.data["password"]
    base_interval = int(
        entry.options.get(
            "poll_interval", entry.data.get("poll_interval", DEFAULT_POLL_INTERVAL)
        )
    )

    # DRY version: read from manifest
    integration = await async_get_integration(hass, DOMAIN)
    version = integration.version or "unknown"

    client = TermoWebClient(session, username, password)
    try:
        devices = await client.list_devices()
    except TermoWebAuthError as err:
        _LOGGER.info("list_devices auth error: %s", err)
        raise ConfigEntryAuthFailed from err
    except (TimeoutError, ClientError, TermoWebRateLimitError) as err:
        _LOGGER.info("list_devices connection error: %s", err)
        raise ConfigEntryNotReady from err

    if not devices:
        _LOGGER.info("list_devices returned no devices")
        raise ConfigEntryNotReady

    dev = devices[0] if isinstance(devices, list) and devices else {}
    dev_id = str(
        dev.get("dev_id") or dev.get("id") or dev.get("serial_id") or ""
    ).strip()
    nodes = await client.get_nodes(dev_id)
    addrs = extract_heater_addrs(nodes)

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
        "ws_tasks": {},  # dev_id -> asyncio.Task
        "ws_clients": {},  # dev_id -> TermoWebWSLegacyClient
        "ws_state": {},  # dev_id -> status attrs
        "version": version,
    }

    async def _start_ws(dev_id: str) -> None:
        tasks: dict[str, asyncio.Task] = data["ws_tasks"]
        clients: dict[str, TermoWebWSLegacyClient] = data["ws_clients"]
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
            entry_map: dict[str, set[str]] = {}
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
                if isinstance(res, asyncio.CancelledError):
                    raise res
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
