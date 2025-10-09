"""Energy history helpers for the TermoWeb integration."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import logging
from typing import Any, cast

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EVENT_HOMEASSISTANT_STARTED
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er

from .api import RESTClient
from .const import DOMAIN
from .identifiers import build_heater_energy_unique_id
from .installation import ensure_snapshot
from .inventory import (
    build_heater_address_map,
    normalize_heater_addresses,
    parse_heater_energy_unique_id,
)
from .nodes import ensure_node_inventory
from .throttle import (
    MonotonicRateLimiter,
    default_samples_rate_limit_state,
    reset_samples_rate_limit_state,
)

_LOGGER = logging.getLogger(__name__)

OPTION_ENERGY_HISTORY_IMPORTED = "energy_history_imported"
OPTION_ENERGY_HISTORY_PROGRESS = "energy_history_progress"
OPTION_MAX_HISTORY_RETRIEVED = "max_history_retrieved"

DEFAULT_MAX_HISTORY_DAYS = 7


@dataclass(slots=True)
class _RecorderStatisticsHelpers:
    """Container for recorder statistics helper callables."""

    executor: Callable[..., Awaitable[Any]] | None
    sync_target: Any | None
    sync: Callable[..., Any] | None
    async_fn: Callable[..., Awaitable[Any]] | None


@dataclass(slots=True)
class _RecorderModuleImports:
    """Container for recorder module helper imports."""

    get_instance: Callable[[HomeAssistant], Any] | None
    statistics: Any | None


_RECORDER_IMPORTS: _RecorderModuleImports | None = None


def _resolve_recorder_imports() -> _RecorderModuleImports:
    """Return cached recorder helper imports."""

    global _RECORDER_IMPORTS
    if _RECORDER_IMPORTS is not None:
        return _RECORDER_IMPORTS

    get_instance: Callable[[HomeAssistant], Any] | None = None
    statistics_mod: Any | None = None

    try:
        from homeassistant.components.recorder import (
            get_instance as _get_instance,
            statistics as _statistics_module,
        )
    except (ImportError, AttributeError):  # pragma: no cover - defensive
        try:
            from homeassistant.components.recorder import (
                statistics as _statistics_module,
            )
        except (ImportError, AttributeError):  # pragma: no cover - defensive
            _statistics_module = None
    else:
        get_instance = _get_instance

    statistics_mod = _statistics_module

    _RECORDER_IMPORTS = _RecorderModuleImports(
        get_instance=get_instance,
        statistics=statistics_mod,
    )
    return _RECORDER_IMPORTS


def _resolve_statistics_helpers(
    hass: HomeAssistant,
    sync_name: str,
    async_name: str,
    *,
    sync_uses_instance: bool = False,
) -> _RecorderStatisticsHelpers:
    """Return the recorder statistics helpers for compatibility shims."""

    imports = _resolve_recorder_imports()

    statistics_mod: Any | None = imports.statistics

    sync_helper: Callable[..., Any] | None = None
    async_helper: Callable[..., Awaitable[Any]] | None = None
    executor: Callable[..., Awaitable[Any]] | None = None
    sync_target: Any | None = None

    if statistics_mod is not None:
        sync_candidate = getattr(statistics_mod, sync_name, None)
        if callable(sync_candidate):
            sync_helper = sync_candidate

        async_candidate = getattr(statistics_mod, async_name, None)
        if callable(async_candidate):
            async_helper = async_candidate

    if sync_helper is not None and imports.get_instance is not None:
        instance = imports.get_instance(hass)
        executor = instance.async_add_executor_job
        sync_target = instance if sync_uses_instance else hass

    return _RecorderStatisticsHelpers(
        executor=executor,
        sync_target=sync_target,
        sync=sync_helper,
        async_fn=async_helper,
    )


def _iso_date(ts: int) -> str:
    """Convert unix timestamp to ISO date."""

    return datetime.fromtimestamp(ts, UTC).date().isoformat()


def _store_statistics(
    hass: HomeAssistant, metadata: dict[str, Any], stats: list[dict[str, Any]]
) -> None:
    """Insert statistics using recorder helpers."""

    _import_stats: Callable[[HomeAssistant, Mapping[str, Any], list[dict[str, Any]]], None] | None

    try:
        from homeassistant.components.recorder.statistics import (
            async_import_statistics as _async_import_statistics,
        )
    except ImportError:
        _async_import_statistics = None

    if _async_import_statistics is not None:
        _import_stats = _async_import_statistics
    else:
        _import_stats = None

    if _import_stats:
        _import_stats(hass, metadata, stats)
        return

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


def _statistics_row_get(row: Any, key: str) -> Any:
    """Read a field from a statistics row regardless of its container type."""

    if isinstance(row, dict):
        return row.get(key)
    return getattr(row, key, None)  # pragma: no cover - attribute rows rare in tests


async def _statistics_during_period_compat(  # pragma: no cover - compatibility shim
    hass: HomeAssistant,
    start_time: datetime,
    end_time: datetime,
    statistic_ids: set[str],
) -> dict[str, list[Any]] | None:
    """Fetch statistics for a period using the best available API."""

    wanted_types = {"state", "sum"}

    helpers = _resolve_statistics_helpers(
        hass,
        "statistics_during_period",
        "async_get_statistics_during_period",
    )

    if helpers.sync and helpers.executor and helpers.sync_target is not None:
        return await helpers.executor(
            helpers.sync,
            helpers.sync_target,
            start_time,
            end_time,
            statistic_ids,
            "hour",
            None,
            wanted_types,
        )

    if helpers.async_fn is None:
        return None

    return await helpers.async_fn(
        hass,
        start_time,
        end_time,
        list(statistic_ids),
        period="hour",
        types=wanted_types,
    )  # pragma: no cover - exercised when async helper available at runtime


async def _get_last_statistics_compat(  # pragma: no cover - compatibility shim
    hass: HomeAssistant,
    number_of_stats: int,
    statistic_id: str,
    *,
    types: set[str] | None = None,
    start_time: datetime | None = None,
) -> dict[str, list[Any]] | None:
    """Retrieve the last statistics row via synchronous or async helpers."""

    types = types or {"state", "sum"}

    helpers = _resolve_statistics_helpers(
        hass,
        "get_last_statistics",
        "async_get_last_statistics",
    )

    if (
        helpers.sync
        and helpers.executor
        and helpers.sync_target is not None
        and start_time is None
    ):
        return await helpers.executor(
            helpers.sync,
            helpers.sync_target,
            number_of_stats,
            statistic_id,
            types,
        )

    if helpers.async_fn is None:
        return None

    kwargs: dict[str, Any] = {"types": types}
    if start_time is not None:
        kwargs["start_time"] = start_time

    return await helpers.async_fn(
        hass,
        number_of_stats,
        [statistic_id],
        **kwargs,
    )  # pragma: no cover - dependent on runtime helper availability


async def _clear_statistics_compat(  # pragma: no cover - compatibility shim
    hass: HomeAssistant,
    statistic_id: str,
    *,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> str | None:
    """Clear statistics using whichever helper is available."""

    helpers = _resolve_statistics_helpers(
        hass,
        "clear_statistics",
        "async_delete_statistics",
        sync_uses_instance=True,
    )

    if helpers.sync and helpers.executor and helpers.sync_target is not None:
        await helpers.executor(
            helpers.sync,
            helpers.sync_target,
            [statistic_id],
        )
        return "clear"

    if helpers.async_fn is None:
        return None

    delete_args: dict[str, Any] = {}
    if start_time is not None:
        delete_args["start_time"] = start_time
    if end_time is not None:
        delete_args["end_time"] = end_time

    try:
        await helpers.async_fn(hass, [statistic_id], **delete_args)
    except TypeError:  # pragma: no cover - older signature fallback
        await helpers.async_fn(hass, [statistic_id])
    return "delete"  # pragma: no cover - dependent on async helper availability


async def async_import_energy_history(
    hass: HomeAssistant,
    entry: ConfigEntry,
    nodes: Mapping[str, Iterable[str]] | Iterable[str] | None = None,
    *,
    reset_progress: bool = False,
    max_days: int | None = None,
    rate_limit: MonotonicRateLimiter,
) -> None:
    """Fetch historical hourly samples and insert statistics."""

    logger = _LOGGER
    async_mod = asyncio
    datetime_mod = datetime
    ensure_inventory = ensure_node_inventory
    build_map = build_heater_address_map
    normalize = normalize_heater_addresses
    registry_mod = er
    store_stats = _store_statistics
    stats_period = _statistics_during_period_compat
    last_stats_fn = _get_last_statistics_compat
    clear_stats_fn = _clear_statistics_compat

    rec = hass.data.get(DOMAIN, {}).get(entry.entry_id)
    if not rec:
        logger.debug("%s: no record found for energy import", entry.entry_id)
        return
    client: RESTClient = rec["client"]
    dev_id: str = rec["dev_id"]
    inventory: list[Any] = ensure_inventory(rec)

    by_type, reverse_lookup = build_map(inventory)

    requested_map: dict[str, list[str]] | None
    if nodes is None:
        requested_map = None
    else:
        normalized_map, _ = normalize(nodes)
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
        deduped_map, _ = normalize(selected_map)
        selected_map = {k: list(v) for k, v in deduped_map.items() if v}

    all_pairs: list[tuple[str, str]] = [
        (node_type, addr) for node_type, addrs in by_type.items() for addr in addrs
    ]
    target_pairs: list[tuple[str, str]] = [
        (node_type, addr) for node_type, addrs in selected_map.items() for addr in addrs
    ]

    day = 24 * 3600
    now_dt = datetime_mod.now(UTC)
    start_of_today = now_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    now_ts = int(start_of_today.timestamp()) - 1
    if max_days is None:
        raw_max_days = entry.options.get(OPTION_MAX_HISTORY_RETRIEVED)
        try:
            max_days = int(raw_max_days)
        except (TypeError, ValueError):
            max_days = DEFAULT_MAX_HISTORY_DAYS
    target = now_ts - max_days * day
    raw_progress = entry.options.get(OPTION_ENERGY_HISTORY_PROGRESS)
    progress: dict[str, int]
    if isinstance(raw_progress, Mapping):
        progress = dict(raw_progress)
    else:
        progress = {}

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
        logger.debug("%s: energy history already imported", entry.entry_id)
        return

    if not target_pairs:
        logger.debug("Energy import: no heater nodes selected for device")
        return

    logger.debug("Energy import: fetching hourly samples down to %s", _iso_date(target))

    async def _rate_limited_fetch(
        node_type: str, addr: str, start: int, stop: int
    ) -> list[dict[str, Any]]:

        def _log_wait(wait: float) -> None:
            logger.debug(
                "%s:%s/%s: sleeping %.2fs before query",
                node_type,
                addr,
                _iso_date(start),
                wait,
            )

        await rate_limit.async_throttle(on_wait=_log_wait)
        logger.debug(
            "%s:%s: requesting samples %s-%s",
            node_type,
            addr,
            _iso_date(start),
            _iso_date(stop),
        )
        try:
            return await client.get_node_samples(dev_id, (node_type, addr), start, stop)
        except async_mod.CancelledError:  # pragma: no cover - allow cancellation
            raise
        except Exception as err:  # pragma: no cover - defensive
            logger.debug("%s:%s: error fetching samples: %s", node_type, addr, err)
            return []

    ent_reg: er.EntityRegistry | None = registry_mod.async_get(hass)
    for node_type, addr in target_pairs:
        logger.debug(
            "Energy import: importing history for %s %s",
            node_type or "htr",
            addr,
        )
        all_samples: list[dict[str, Any]] = []
        start_ts = _progress_value(node_type, addr)
        while start_ts > target:
            chunk_start = max(start_ts - day, target)
            samples = await _rate_limited_fetch(node_type, addr, chunk_start, start_ts)

            logger.debug(
                "%s:%s: fetched %d samples for %s-%s",
                node_type,
                addr,
                len(samples),
                _iso_date(chunk_start),
                _iso_date(start_ts),
            )

            all_samples.extend(samples)

            start_ts = chunk_start
            progress[f"{node_type}:{addr}"] = start_ts
            progress.pop(addr, None)
            _write_progress_options(progress)

        if not all_samples:
            logger.debug("%s: no samples fetched", addr)
            continue

        all_samples_sorted = sorted(all_samples, key=lambda s: s.get("t", 0))

        uid = build_heater_energy_unique_id(dev_id, node_type, addr)
        entity_id = (
            ent_reg.async_get_entity_id("sensor", DOMAIN, uid) if ent_reg else None
        )
        if not entity_id and node_type != "htr":
            legacy_uid = build_heater_energy_unique_id(dev_id, "htr", addr)
            entity_id = (
                ent_reg.async_get_entity_id("sensor", DOMAIN, legacy_uid)
                if ent_reg
                else None
            )
        if not entity_id:
            logger.debug("%s:%s: no energy sensor found", node_type, addr)
            continue

        first_ts = int(all_samples_sorted[0].get("t", 0))
        last_ts = int(all_samples_sorted[-1].get("t", 0))
        import_start_dt = datetime_mod.fromtimestamp(first_ts, UTC).replace(
            minute=0, second=0, microsecond=0
        )
        import_end_dt = datetime_mod.fromtimestamp(last_ts, UTC).replace(
            minute=0, second=0, microsecond=0
        )

        sum_offset = 0.0
        previous_kwh: float | None = None
        last_before: dict[str, Any] | None = None

        lookback_days = max(2, max_days + 1)
        lookback_start = import_start_dt - timedelta(days=lookback_days)

        period_stats: dict[str, list[Any]] | None = None
        try:
            period_stats = await stats_period(
                hass,
                lookback_start,
                import_end_dt + timedelta(hours=1),
                {entity_id},
            )
        except async_mod.CancelledError:  # pragma: no cover - allow cancellation
            raise
        except Exception as err:  # pragma: no cover - defensive
            logger.error(
                "%s: error fetching statistics window %s-%s: %s",
                addr,
                lookback_start,
                import_end_dt,
                err,
                exc_info=True,
            )

        if period_stats:
            window_values = period_stats.get(entity_id) or []
            if window_values:
                before_values = [
                    val
                    for val in window_values
                    if isinstance(_statistics_row_get(val, "start"), datetime)
                    and cast(datetime, _statistics_row_get(val, "start"))
                    < import_start_dt
                ]
                if before_values:
                    last_before = before_values[-1]

        if last_before is None:
            try:
                existing = await last_stats_fn(
                    hass,
                    1,
                    entity_id,
                    types={"state", "sum"},
                )
                if existing and (vals := existing.get(entity_id)):
                    candidate = vals[0]
                    start_dt = _statistics_row_get(candidate, "start")
                    if isinstance(start_dt, datetime) and start_dt < import_start_dt:
                        last_before = candidate
            except async_mod.CancelledError:  # pragma: no cover - allow cancellation
                raise
            except Exception as err:  # pragma: no cover - defensive
                logger.error(
                    "%s: error fetching last statistics for offset: %s",
                    addr,
                    err,
                    exc_info=True,
                )

        if last_before:
            try:
                sum_offset = float(_statistics_row_get(last_before, "sum") or 0.0)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                logger.debug(
                    "%s: invalid sum offset in existing statistics: %s",
                    addr,
                    last_before,
                )
                sum_offset = 0.0
            prev_state = _statistics_row_get(last_before, "state")
            if prev_state is not None:
                try:
                    previous_kwh = float(prev_state)
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    logger.debug(
                        "%s: invalid previous state in statistics: %s", addr, prev_state
                    )
                    previous_kwh = None

        overlap_exists = False
        if period_stats is not None:
            overlap_exists = any(
                isinstance(_statistics_row_get(val, "start"), datetime)
                and import_start_dt
                <= cast(datetime, _statistics_row_get(val, "start"))
                <= import_end_dt
                for val in period_stats.get(entity_id, [])
            )
        else:
            try:
                overlap_stats = await last_stats_fn(
                    hass,
                    1,
                    entity_id,
                    start_time=import_start_dt,
                )
                overlap_exists = bool(overlap_stats and overlap_stats.get(entity_id))
            except async_mod.CancelledError:  # pragma: no cover - allow cancellation
                raise
            except Exception as err:  # pragma: no cover - defensive
                logger.error(
                    "%s: error checking for overlapping statistics: %s",
                    addr,
                    err,
                    exc_info=True,
                )

        if overlap_exists:
            try:
                cleared = await clear_stats_fn(
                    hass,
                    entity_id,
                    start_time=import_start_dt,
                    end_time=import_end_dt + timedelta(hours=1),
                )
                if cleared == "clear":
                    logger.debug("%s: cleared statistics for %s", addr, entity_id)
                elif cleared == "delete":
                    logger.debug(
                        "%s: cleared overlapping statistics for %s", addr, entity_id
                    )
                else:
                    logger.debug(
                        "%s: statistics helpers unavailable to clear overlap", addr
                    )  # pragma: no cover - informational fallback
            except async_mod.CancelledError:  # pragma: no cover - allow cancellation
                raise
            except Exception as err:  # pragma: no cover - defensive
                logger.error(
                    "%s: failed to clear overlapping statistics: %s",
                    addr,
                    err,
                    exc_info=True,
                )

        stats: list[dict[str, Any]] = []
        running_sum: float = sum_offset
        for sample in all_samples_sorted:
            t_val = sample.get("t")
            counter_val = sample.get("counter")
            try:
                ts = int(t_val)
                kwh = float(counter_val) / 1000.0
            except (TypeError, ValueError):
                logger.debug("%s: invalid sample %s", addr, sample)
                continue

            start_dt = datetime_mod.fromtimestamp(ts, UTC).replace(
                minute=0, second=0, microsecond=0
            )
            if previous_kwh is None:
                previous_kwh = kwh
                continue

            delta = kwh - previous_kwh
            if delta <= 0:
                previous_kwh = kwh
                continue

            running_sum += delta
            stats.append({"start": start_dt, "state": kwh, "sum": running_sum})
            previous_kwh = kwh

        if not stats:
            logger.debug("%s: no positive deltas found", addr)
            continue

        logger.debug("%s: inserting statistics for %s", addr, entity_id)
        ent_entry = ent_reg.async_get(entity_id) if ent_reg else None
        name = getattr(ent_entry, "original_name", None) or entity_id

        metadata = {
            "source": "recorder",
            "statistic_id": entity_id,
            "unit_of_measurement": "kWh",
            "name": name,
            "has_sum": True,
            "has_mean": False,
        }
        logger.debug("%s: adding %d stats entries", addr, len(stats))
        try:
            store_stats(hass, metadata, stats)
        except Exception as err:  # pragma: no cover - log & continue
            logger.exception("%s: statistics insert failed: %s", addr, err)

    imported_flag: bool | None = None
    if all(_progress_value(node_type, addr) <= target for node_type, addr in all_pairs):
        imported_flag = True
    _write_progress_options(progress, imported=imported_flag)
    logger.debug("%s: energy import complete", entry.entry_id)


async def async_register_import_energy_history_service(
    hass: HomeAssistant,
    import_fn: Callable[..., Awaitable[None]],
) -> None:
    """Register the import_energy_history service if it is missing."""

    if hass.services.has_service(DOMAIN, "import_energy_history"):
        return

    logger = _LOGGER
    async_mod = asyncio
    build_map = build_heater_address_map
    registry_mod = er

    async def _service_import_energy_history(call) -> None:
        """Handle the import_energy_history service call."""

        logger.debug("service import_energy_history called")
        reset = bool(call.data.get("reset_progress", False))
        max_days = call.data.get("max_history_retrieval")
        ent_ids = call.data.get("entity_id")
        tasks = []
        if ent_ids:
            ent_reg = registry_mod.async_get(hass)
            if isinstance(ent_ids, str):
                ent_ids = [ent_ids]
            entry_map: dict[str, dict[str, set[str]]] = {}

            def _parse_energy_unique_id(unique_id: str) -> tuple[str, str] | None:
                parsed = parse_heater_energy_unique_id(unique_id)
                if not parsed:
                    return None
                _dev_id, node_type, addr = parsed
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
                entry_map.setdefault(entry_id, {}).setdefault(node_type, set()).add(
                    addr
                )

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
                    import_fn(
                        hass,
                        ent,
                        normalized,
                        reset_progress=reset,
                        max_days=max_days,
                    )
                )
        else:
            for rec in hass.data.get(DOMAIN, {}).values():
                if not isinstance(rec, Mapping):
                    continue
                ent: ConfigEntry | None = rec.get("config_entry")
                if not ent:
                    continue
                snapshot = ensure_snapshot(rec)
                if snapshot is not None:
                    override = rec.get("node_inventory")
                    if override is not None:
                        inventory = list(override)
                        snapshot.update_nodes(
                            snapshot.raw_nodes, node_inventory=inventory
                        )
                    else:
                        inventory = snapshot.inventory
                        rec["node_inventory"] = list(inventory)
                else:
                    inventory = rec.get("node_inventory") or []
                by_type, _ = build_map(inventory)
                tasks.append(
                    import_fn(
                        hass,
                        ent,
                        by_type,
                        reset_progress=reset,
                        max_days=max_days,
                    )
                )
        if tasks:
            logger.debug("import_energy_history: awaiting %d tasks", len(tasks))
            results = await async_mod.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, async_mod.CancelledError):
                    raise res
                if isinstance(res, Exception):
                    logger.exception("import_energy_history task failed: %s", res)

    hass.services.async_register(
        DOMAIN, "import_energy_history", _service_import_energy_history
    )


def async_schedule_initial_energy_import(
    hass: HomeAssistant,
    entry: ConfigEntry,
    import_fn: Callable[..., Awaitable[None]],
) -> None:
    """Schedule the initial energy history import for an entry."""

    if entry.options.get(OPTION_ENERGY_HISTORY_IMPORTED):
        return

    _LOGGER.debug("%s: scheduling initial energy import", entry.entry_id)

    async def _schedule_import(_event: Any | None = None) -> None:
        await import_fn(hass, entry)

    if hass.is_running:
        hass.async_create_task(_schedule_import())
    else:
        hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STARTED, _schedule_import)
