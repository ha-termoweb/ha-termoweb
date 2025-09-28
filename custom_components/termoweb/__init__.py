from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta
import logging
import time
from typing import Any

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

from .api import BackendAuthError, BackendRateLimitError, RESTClient
from .backend import Backend, DucaheatRESTClient, WsClientProto, create_backend
from .const import (
    BRAND_DUCAHEAT,
    CONF_BRAND,
    DEFAULT_BRAND,
    DEFAULT_POLL_INTERVAL,
    DOMAIN,
    MIN_POLL_INTERVAL,
    STRETCHED_POLL_INTERVAL,
    get_brand_api_base,
    get_brand_basic_auth,
    signal_ws_status,
)
from .coordinator import StateCoordinator
from .nodes import build_node_inventory
from .utils import (
    HEATER_NODE_TYPES as _HEATER_NODE_TYPES,
    build_heater_address_map,
    ensure_node_inventory,
    normalize_heater_addresses,
)

HEATER_NODE_TYPES = _HEATER_NODE_TYPES

# Re-export legacy WS client for backward compatibility (tests may patch it).
from .ws_client import WebSocketClient as WebSocket09Client  # noqa: F401

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
    nodes: Mapping[str, Iterable[str]] | Iterable[str] | None = None,
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
    client: RESTClient = rec["client"]
    dev_id: str = rec["dev_id"]
    inventory: list[Any] = ensure_node_inventory(rec)

    by_type, reverse_lookup = build_heater_address_map(inventory)

    requested_map: dict[str, list[str]] | None
    if nodes is None:
        requested_map = None
    else:
        normalized_map, _ = normalize_heater_addresses(nodes)
        requested_map = {k: list(v) for k, v in normalized_map.items() if v}

    selected_map: dict[str, list[str]] = {}
    if requested_map:
        available_sets: dict[str, set[str]] = {
            node_type: set(addrs) for node_type, addrs in by_type.items()
        }
        for req_type, addr_list in requested_map.items():
            if not addr_list:  # pragma: no cover - defensive guard
                continue
            if req_type == "htr":
                for addr in addr_list:
                    for actual_type in reverse_lookup.get(addr, set()):
                        selected_map.setdefault(actual_type, []).append(addr)
            else:
                available = available_sets.get(req_type)
                if not available:
                    continue
                filtered = [addr for addr in addr_list if addr in available]
                if filtered:
                    selected_map.setdefault(req_type, []).extend(filtered)

    if not selected_map:
        selected_map = {node_type: list(addrs) for node_type, addrs in by_type.items()}

    if selected_map:
        deduped_map, _ = normalize_heater_addresses(selected_map)
        selected_map = {k: list(v) for k, v in deduped_map.items() if v}

    all_pairs: list[tuple[str, str]] = [
        (node_type, addr) for node_type, addrs in by_type.items() for addr in addrs
    ]
    target_pairs: list[tuple[str, str]] = [
        (node_type, addr)
        for node_type, addrs in selected_map.items()
        for addr in addrs
    ]

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

    def _progress_value(node_type: str, addr: str) -> int:
        raw = progress.get(f"{node_type}:{addr}")
        if raw is None:
            raw = progress.get(addr)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return now_ts

    def _write_progress_options(
        progress_state: Mapping[str, int], *, imported: bool | None = None
    ) -> None:
        """Persist energy import progress to the config entry options."""

        options = dict(entry.options)
        options[OPTION_ENERGY_HISTORY_PROGRESS] = dict(progress_state)
        if imported is False:
            options.pop(OPTION_ENERGY_HISTORY_IMPORTED, None)
        elif imported:
            options[OPTION_ENERGY_HISTORY_IMPORTED] = True
        hass.config_entries.async_update_entry(entry, options=options)

    if reset_progress:
        if nodes is None:
            progress.clear()
        else:
            cleared_any = False
            for node_type, addr in target_pairs:
                progress.pop(f"{node_type}:{addr}", None)
                progress.pop(addr, None)
                cleared_any = True
            if not cleared_any and requested_map:
                for req_type, addr_list in requested_map.items():
                    for addr in addr_list:
                        progress.pop(f"{req_type}:{addr}", None)
                        progress.pop(addr, None)
        _write_progress_options(progress, imported=False)
    elif entry.options.get(OPTION_ENERGY_HISTORY_IMPORTED):
        _LOGGER.debug("%s: energy history already imported", entry.entry_id)
        return

    if not target_pairs:
        _LOGGER.debug("%s: no heater nodes selected for energy import", dev_id)
        return

    _LOGGER.debug("%s: importing hourly samples down to %s", dev_id, _iso_date(target))

    async def _rate_limited_fetch(
        node_type: str, addr: str, start: int, stop: int
    ) -> list[dict[str, Any]]:
        """Fetch heater samples while respecting the shared rate limit."""
        global _LAST_SAMPLES_QUERY
        async with _SAMPLES_QUERY_LOCK:
            now = time.monotonic()
            wait = 1 - (now - _LAST_SAMPLES_QUERY)
            if wait > 0:
                _LOGGER.debug(
                    "%s:%s/%s: sleeping %.2fs before query",
                    node_type,
                    addr,
                    _iso_date(start),
                    wait,
                )
                await asyncio.sleep(wait)
            _LAST_SAMPLES_QUERY = time.monotonic()
        _LOGGER.debug(
            "%s:%s: requesting samples %s-%s",
            node_type,
            addr,
            _iso_date(start),
            _iso_date(stop),
        )
        try:
            return await client.get_node_samples(
                dev_id, (node_type, addr), start, stop
            )
        except asyncio.CancelledError:  # pragma: no cover - allow cancellation
            raise
        except Exception as err:  # pragma: no cover - defensive
            _LOGGER.debug("%s:%s: error fetching samples: %s", node_type, addr, err)
            return []

    ent_reg: er.EntityRegistry | None = er.async_get(hass)
    for node_type, addr in target_pairs:
        _LOGGER.debug(
            "%s: importing history for %s %s", dev_id, node_type or "htr", addr
        )
        # Collect all samples across the requested range.  We will process
        # them after the loop in chronological order.
        all_samples: list[dict[str, Any]] = []
        start_ts = _progress_value(node_type, addr)
        while start_ts > target:
            chunk_start = max(start_ts - day, target)
            samples = await _rate_limited_fetch(node_type, addr, chunk_start, start_ts)

            _LOGGER.debug(
                "%s:%s: fetched %d samples for %s-%s",
                node_type,
                addr,
                len(samples),
                _iso_date(chunk_start),
                _iso_date(start_ts),
            )

            # Append all samples as-is for later processing
            all_samples.extend(samples)

            start_ts = chunk_start
            progress[f"{node_type}:{addr}"] = start_ts
            progress.pop(addr, None)
            _write_progress_options(progress)

        if not all_samples:
            _LOGGER.debug("%s: no samples fetched", addr)
            continue

        # Sort all samples chronologically so that we can compute deltas properly.
        all_samples_sorted = sorted(all_samples, key=lambda s: s.get("t", 0))

        # Resolve the entity_id for the heater's energy sensor
        uid = f"{DOMAIN}:{dev_id}:{node_type}:{addr}:energy"
        entity_id = (
            ent_reg.async_get_entity_id("sensor", DOMAIN, uid) if ent_reg else None
        )
        if not entity_id and node_type != "htr":
            legacy_uid = f"{DOMAIN}:{dev_id}:htr:{addr}:energy"
            entity_id = (
                ent_reg.async_get_entity_id("sensor", DOMAIN, legacy_uid)
                if ent_reg
                else None
            )
        if not entity_id:
            _LOGGER.debug("%s:%s: no energy sensor found", node_type, addr)
            continue

        first_ts = int(all_samples_sorted[0].get("t", 0))
        last_ts = int(all_samples_sorted[-1].get("t", 0))
        import_start_dt = datetime.fromtimestamp(first_ts, UTC).replace(
            minute=0, second=0, microsecond=0
        )
        import_end_dt = datetime.fromtimestamp(last_ts, UTC).replace(
            minute=0, second=0, microsecond=0
        )

        sum_offset = 0.0
        previous_kwh: float | None = None
        last_before: dict[str, Any] | None = None

        period_stats: dict[str, list[dict[str, Any]]] | None = None
        try:
            from homeassistant.components.recorder.statistics import (
                async_get_statistics_during_period,
            )
        except (ImportError, AttributeError):  # pragma: no cover - defensive
            async_get_statistics_during_period = None

        if async_get_statistics_during_period:
            lookback_days = max(2, max_days + 1)
            lookback_start = import_start_dt - timedelta(days=lookback_days)
            try:
                period_stats = await async_get_statistics_during_period(
                    hass,
                    lookback_start,
                    import_end_dt + timedelta(hours=1),
                    [entity_id],
                    period="hour",
                )
            except asyncio.CancelledError:  # pragma: no cover - allow cancellation
                raise
            except Exception as err:  # pragma: no cover - defensive
                _LOGGER.debug(
                    "%s: error fetching statistics window %s-%s: %s",
                    addr,
                    lookback_start,
                    import_end_dt,
                    err,
                )
            else:
                window_values = period_stats.get(entity_id) if period_stats else []
                if window_values:
                    before_values = [
                        val
                        for val in window_values
                        if isinstance(val.get("start"), datetime)
                        and val["start"] < import_start_dt
                    ]
                    if before_values:
                        last_before = before_values[-1]

        if last_before is None:
            try:
                from homeassistant.components.recorder.statistics import (
                    async_get_last_statistics,
                )

                existing = await async_get_last_statistics(hass, 1, [entity_id])
                if existing and (vals := existing.get(entity_id)):
                    candidate = vals[0]
                    start_dt = candidate.get("start")
                    if isinstance(start_dt, datetime) and start_dt < import_start_dt:
                        last_before = candidate
            except asyncio.CancelledError:  # pragma: no cover - allow cancellation
                raise
            except Exception as err:  # pragma: no cover - defensive
                _LOGGER.debug(
                    "%s: error fetching last statistics for offset: %s", addr, err
                )

        if last_before:
            try:
                sum_offset = float(last_before.get("sum") or 0.0)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                _LOGGER.debug(
                    "%s: invalid sum offset in existing statistics: %s",
                    addr,
                    last_before,
                )
                sum_offset = 0.0
            prev_state = last_before.get("state")
            if prev_state is not None:
                try:
                    previous_kwh = float(prev_state)
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    _LOGGER.debug(
                        "%s: invalid previous state in statistics: %s", addr, prev_state
                    )
                    previous_kwh = None

        overlap_exists = False
        if period_stats is not None:
            overlap_exists = any(
                isinstance(val.get("start"), datetime)
                and import_start_dt <= val["start"] <= import_end_dt
                for val in period_stats.get(entity_id, [])
            )
        else:
            try:
                from homeassistant.components.recorder.statistics import (
                    async_get_last_statistics,
                )

                overlap_stats = await async_get_last_statistics(
                    hass, 1, [entity_id], start_time=import_start_dt
                )
                overlap_exists = bool(
                    overlap_stats and overlap_stats.get(entity_id)
                )
            except asyncio.CancelledError:  # pragma: no cover - allow cancellation
                raise
            except Exception as err:  # pragma: no cover - defensive
                _LOGGER.debug(
                    "%s: error checking for overlapping statistics: %s", addr, err
                )

        if overlap_exists:
            try:
                from homeassistant.components.recorder.statistics import (
                    async_delete_statistics,
                )

                delete_args: dict[str, Any] = {
                    "start_time": import_start_dt,
                    "end_time": import_end_dt + timedelta(hours=1),
                }
                try:
                    await async_delete_statistics(hass, [entity_id], **delete_args)
                    _LOGGER.debug("%s: cleared overlapping statistics for %s", addr, entity_id)
                except TypeError:
                    await async_delete_statistics(hass, [entity_id])
                    _LOGGER.debug("%s: cleared statistics for %s", addr, entity_id)
            except asyncio.CancelledError:  # pragma: no cover - allow cancellation
                raise
            except ImportError:  # pragma: no cover - defensive
                _LOGGER.debug(
                    "%s: async_delete_statistics not available to clear overlap",
                    addr,
                )
            except Exception as err:  # pragma: no cover - defensive
                _LOGGER.debug("%s: failed to clear overlapping statistics: %s", addr, err)

        stats: list[dict[str, Any]] = []
        running_sum: float = sum_offset
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
            running_sum += delta
            stats.append({"start": start_dt, "state": kwh, "sum": running_sum})
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

    imported_flag: bool | None = None
    if all(_progress_value(node_type, addr) <= target for node_type, addr in all_pairs):
        imported_flag = True
    _write_progress_options(progress, imported=imported_flag)
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
    brand = entry.data.get(CONF_BRAND, DEFAULT_BRAND)
    api_base = get_brand_api_base(brand)
    basic_auth = get_brand_basic_auth(brand)

    # DRY version: read from manifest
    integration = await async_get_integration(hass, DOMAIN)
    version = integration.version or "unknown"

    client_cls = DucaheatRESTClient if brand == BRAND_DUCAHEAT else RESTClient
    client = client_cls(
        session,
        username,
        password,
        api_base=api_base,
        basic_auth_b64=basic_auth,
    )
    backend = create_backend(brand=brand, client=client)
    try:
        devices = await client.list_devices()
    except BackendAuthError as err:
        _LOGGER.info("list_devices auth error: %s", err)
        raise ConfigEntryAuthFailed from err
    except (TimeoutError, ClientError, BackendRateLimitError) as err:
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

    async def _service_import_energy_history(call) -> None:
        """Handle the import_energy_history service call."""
        _LOGGER.debug("service import_energy_history called")
        reset = bool(call.data.get("reset_progress", False))
        max_days = call.data.get("max_history_retrieval")
        ent_ids = call.data.get("entity_id")
        tasks = []
        if ent_ids:
            ent_reg = er.async_get(hass)
            if isinstance(ent_ids, str):
                ent_ids = [ent_ids]
            entry_map: dict[str, dict[str, set[str]]] = {}

            def _parse_energy_unique_id(unique_id: str) -> tuple[str, str] | None:
                if not unique_id or not unique_id.startswith(f"{DOMAIN}:"):
                    return None
                try:
                    remainder = unique_id.split(":", 1)[1]
                    prefix, metric = remainder.rsplit(":", 1)
                except ValueError:
                    return None
                if metric != "energy":
                    return None
                try:
                    _dev_part, node_type, addr = prefix.rsplit(":", 2)
                except ValueError:
                    return None
                return node_type, addr

            for eid in ent_ids:
                er_ent = ent_reg.async_get(eid)
                if not er_ent or er_ent.platform != DOMAIN:
                    continue
                parsed = _parse_energy_unique_id(er_ent.unique_id or "")
                if not parsed:
                    continue
                node_type, addr = parsed
                entry_id = er_ent.config_entry_id
                if not entry_id:
                    continue
                entry_map.setdefault(entry_id, {}).setdefault(node_type, set()).add(addr)

            for entry_id, addr_map in entry_map.items():
                ent = hass.config_entries.async_get_entry(entry_id)
                if not ent:
                    continue
                normalized = {
                    node_type: sorted(addrs)
                    for node_type, addrs in addr_map.items()
                    if addrs
                }
                if not normalized:  # pragma: no cover - defensive guard
                    continue
                tasks.append(
                    _async_import_energy_history(
                        hass,
                        ent,
                        normalized,
                        reset_progress=reset,
                        max_days=max_days,
                    )
                )
        else:
            for rec in hass.data.get(DOMAIN, {}).values():
                ent: ConfigEntry | None = rec.get("config_entry")
                if not ent:
                    continue
                inventory: Iterable[Any] = rec.get("node_inventory") or []
                by_type, _ = build_heater_address_map(inventory)
                tasks.append(
                    _async_import_energy_history(
                        hass,
                        ent,
                        by_type,
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
            """Kick off the initial energy history import task."""
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
