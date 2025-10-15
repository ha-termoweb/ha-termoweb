"""Energy history helpers for the TermoWeb integration."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import partial
import inspect
import logging
from typing import Any, cast

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er

from .api import RESTClient
from .const import DOMAIN
from .identifiers import build_heater_energy_unique_id
from .inventory import Inventory, normalize_node_addr, normalize_node_type
from .throttle import MonotonicRateLimiter

_LOGGER = logging.getLogger(__name__)

OPTION_ENERGY_HISTORY_IMPORTED = "energy_history_imported"
OPTION_ENERGY_HISTORY_PROGRESS = "energy_history_progress"
OPTION_MAX_HISTORY_RETRIEVED = "max_history_retrieved"

DEFAULT_MAX_HISTORY_DAYS = 7
RESET_DELTA_THRESHOLD_KWH = 0.2
SUMMARY_KEY_LAST_RUN = "last_energy_import_summary"


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


async def _store_statistics(
    hass: HomeAssistant, metadata: dict[str, Any], stats: list[dict[str, Any]]
) -> None:
    """Insert entity statistics using the recorder import helper."""

    if not stats:
        return

    from homeassistant.components.recorder.statistics import async_import_statistics

    stat_id = metadata.get("statistic_id")
    if not isinstance(stat_id, str) or "." not in stat_id:
        raise ValueError("metadata must include an entity statistic_id")

    domain, _ = stat_id.split(".", 1)
    import_metadata = dict(metadata)
    import_metadata.update({"source": domain, "statistic_id": stat_id})

    await async_import_statistics(hass, import_metadata, stats)


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

    if helpers.sync and helpers.executor and helpers.sync_target is not None:
        signature = inspect.signature(helpers.sync)
        params = list(signature.parameters.values())
        call_args: list[Any] = [
            helpers.sync_target,
            number_of_stats,
            statistic_id,
        ]
        call_kwargs: dict[str, Any] = {}
        can_call_sync = True

        for param in params[len(call_args) :]:
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            if param.name == "convert_units":
                value: Any = True
            elif param.name == "types":
                value = types
            elif param.name == "start_time":
                if start_time is None and param.default is inspect.Signature.empty:
                    value = None
                elif start_time is not None:
                    value = start_time
                else:
                    continue
            else:
                if param.default is inspect.Signature.empty:
                    can_call_sync = False
                    break
                continue

            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                call_args.append(value)
            else:
                call_kwargs[param.name] = value

        if can_call_sync:
            sync_call = partial(helpers.sync, *call_args, **call_kwargs)
            try:
                return await helpers.executor(sync_call)
            except TypeError as err:
                message = str(err)
                if not any(
                    phrase in message
                    for phrase in (
                        "unexpected keyword argument",
                        "required positional argument",
                        "positional arguments but",
                    )
                ):
                    raise

                fallback_calls = [
                    partial(
                        helpers.sync,
                        helpers.sync_target,
                        number_of_stats,
                        statistic_id,
                        types=types,
                    ),
                    partial(
                        helpers.sync,
                        helpers.sync_target,
                        number_of_stats,
                        statistic_id,
                        convert_units=True,
                        types=types,
                    ),
                    partial(
                        helpers.sync,
                        helpers.sync_target,
                        number_of_stats,
                        statistic_id,
                        types,
                        None,
                    ),
                    partial(
                        helpers.sync,
                        helpers.sync_target,
                        number_of_stats,
                        statistic_id,
                        types,
                    ),
                ]

                for call in fallback_calls:
                    try:
                        return await helpers.executor(call)
                    except TypeError:
                        continue

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


async def _collect_statistics(
    hass: HomeAssistant,
    statistic_id: str,
    start: datetime,
    end: datetime,
) -> list[Any]:
    """Return statistics rows for a single statistic id."""

    try:
        period = await _statistics_during_period_compat(
            hass,
            start,
            end,
            {statistic_id},
        )
    except Exception:  # pragma: no cover - defensive
        _LOGGER.exception(
            "%s: failed to collect statistics between %s and %s",
            statistic_id,
            start,
            end,
        )
        return []

    if not period:
        return []

    rows = period.get(statistic_id) or []
    return [
        row for row in rows if isinstance(_statistics_row_get(row, "start"), datetime)
    ]


async def _enforce_monotonic_sum(
    hass: HomeAssistant,
    entity_id: str,
    import_start_dt: datetime,
    import_end_dt: datetime,
) -> None:
    """Clamp entity statistics so sums never decrease near import seams."""

    if "." not in entity_id:
        return

    window_start = import_start_dt - timedelta(hours=1)
    window_end = import_end_dt + timedelta(hours=6)

    rows = await _collect_statistics(hass, entity_id, window_start, window_end)
    if not rows:
        return

    ordered_rows = sorted(
        rows,
        key=lambda row: cast(datetime, _statistics_row_get(row, "start")),
    )

    rewrites: list[dict[str, Any]] = []
    last_sum: float | None = None

    for row in ordered_rows:
        start_dt = cast(datetime, _statistics_row_get(row, "start"))
        sum_value_raw = _statistics_row_get(row, "sum")
        try:
            sum_value = float(sum_value_raw) if sum_value_raw is not None else None
        except (TypeError, ValueError):
            sum_value = None

        if last_sum is None:
            if sum_value is None:
                continue
            last_sum = sum_value
            continue

        if sum_value is None or sum_value < last_sum:
            rewrites.append({"start": start_dt, "sum": last_sum})
            continue

        last_sum = sum_value

    if not rewrites:
        return

    ent_reg = er.async_get(hass)
    metadata_name = entity_id
    if ent_reg:
        entry = ent_reg.async_get(entity_id)
        if entry is not None:
            metadata_name = (
                getattr(entry, "original_name", None)
                or getattr(entry, "name", None)
                or entity_id
            )

    metadata = {
        "statistic_id": entity_id,
        "name": metadata_name,
        "unit_of_measurement": "kWh",
        "has_sum": True,
        "has_mean": False,
    }

    try:
        await _store_statistics(hass, metadata, rewrites)
    except Exception:  # pragma: no cover - defensive
        _LOGGER.exception("%s: failed to rewrite non-monotonic statistics", entity_id)
        return

    _LOGGER.info("%s: enforced monotonic sum for %d hour(s)", entity_id, len(rewrites))


async def async_import_energy_history(
    hass: HomeAssistant,
    entry: ConfigEntry,
    nodes: Inventory | None = None,
    *,
    node_types: Iterable[str] | None = None,
    addresses: Iterable[str] | None = None,
    day_chunk_hours: int = 24,
    reset_progress: bool = False,
    max_days: int | None = None,
    rate_limit: MonotonicRateLimiter,
) -> None:
    """Fetch historical hourly samples and insert statistics with filters."""

    logger = _LOGGER
    async_mod = asyncio
    datetime_mod = datetime
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

    if nodes is not None and not isinstance(nodes, Inventory):
        raise TypeError(
            "async_import_energy_history nodes must be an Inventory instance"
        )

    inventory: Inventory | None
    container = rec if isinstance(rec, Mapping) else None
    try:
        inventory = Inventory.require_from_context(
            inventory=nodes,
            container=container,
        )
    except LookupError:
        logger.error(
            "%s: energy import aborted; inventory missing in integration state",
            dev_id,
        )
        return

    all_pairs: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()

    def _extend_targets(pairs: Iterable[tuple[str, str]]) -> None:
        for node_type, addr in pairs:
            normalized_type = node_type.strip() if isinstance(node_type, str) else ""
            normalized_addr = addr.strip() if isinstance(addr, str) else ""
            if not normalized_type or not normalized_addr:
                continue
            pair = (normalized_type, normalized_addr)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            all_pairs.append(pair)

    _extend_targets(inventory.heater_sample_targets)
    _extend_targets(inventory.power_monitor_sample_targets)
    _extend_targets(
        (metadata.node_type, metadata.addr)
        for metadata in inventory.iter_nodes_metadata()
    )

    if not all_pairs:
        logger.debug("Energy import: no nodes available for device")
        return

    available_types = {node_type for node_type, _ in all_pairs}
    available_addresses = {addr for _, addr in all_pairs}

    normalized_type_filters: set[str] | None = None
    if node_types is not None:
        normalized_type_filters = set()
        invalid_types: list[str] = []
        for candidate in node_types:
            normalized = normalize_node_type(
                candidate,
                use_default_when_falsey=True,
            )
            if not normalized:
                continue
            if normalized not in available_types:
                invalid_types.append(str(candidate))
                continue
            normalized_type_filters.add(normalized)
        if invalid_types:
            raise ValueError(
                "Unsupported node_types for import_energy_history: "
                + ", ".join(sorted(invalid_types))
            )
        if not normalized_type_filters:
            normalized_type_filters = None

    normalized_address_filters: set[str] | None = None
    if addresses is not None:
        normalized_address_filters = set()
        unknown_addresses: list[str] = []
        for candidate in addresses:
            normalized_addr = normalize_node_addr(
                candidate,
                use_default_when_falsey=True,
            )
            if not normalized_addr:
                continue
            if normalized_addr not in available_addresses:
                unknown_addresses.append(normalized_addr)
                continue
            normalized_address_filters.add(normalized_addr)
        if unknown_addresses:
            logger.warning(
                "%s: ignoring unknown addresses for energy import: %s",
                dev_id,
                ", ".join(sorted(set(unknown_addresses))),
            )
        if not normalized_address_filters:
            normalized_address_filters = None

    target_pairs = [
        pair
        for pair in all_pairs
        if (normalized_type_filters is None or pair[0] in normalized_type_filters)
        and (
            normalized_address_filters is None or pair[1] in normalized_address_filters
        )
    ]

    if not target_pairs:
        logger.debug("Energy import: no nodes available for device after filtering")
        return

    try:
        chunk_hours_value = int(day_chunk_hours)
    except (TypeError, ValueError):
        logger.warning(
            "%s: invalid day_chunk_hours %s; defaulting to 24",
            dev_id,
            day_chunk_hours,
        )
        chunk_hours_value = 24

    if chunk_hours_value <= 0:
        logger.warning("%s: day_chunk_hours must be positive; defaulting to 24", dev_id)
        chunk_hours_value = 24
    elif chunk_hours_value > 24:
        logger.debug(
            "%s: capping day_chunk_hours=%s to 24 hours",
            dev_id,
            chunk_hours_value,
        )
        chunk_hours_value = 24

    day = 24 * 3600
    chunk_seconds = day if chunk_hours_value >= 24 else chunk_hours_value * 3600

    now_dt = datetime_mod.now(UTC)
    run_started_iso = now_dt.isoformat()
    current_minute = now_dt.replace(second=0, microsecond=0)
    now_ts = int(current_minute.timestamp())
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
        progress.clear()
        _write_progress_options(progress, imported=False)
    elif entry.options.get(OPTION_ENERGY_HISTORY_IMPORTED):
        logger.debug("%s: energy history already imported", entry.entry_id)
        return

    logger.debug(
        "Energy import: fetching hourly samples down to %s (chunk=%sh)",
        _iso_date(target),
        chunk_hours_value,
    )

    filters_snapshot = {
        "node_types": sorted(normalized_type_filters)
        if normalized_type_filters
        else [],
        "addresses": sorted(normalized_address_filters)
        if normalized_address_filters
        else [],
        "day_chunk_hours": chunk_hours_value,
    }

    node_summaries: list[dict[str, Any]] = []
    total_nodes_processed = 0
    total_samples_fetched = 0
    total_samples_written = 0
    total_resets_detected = 0
    overlap_warning_logged = False

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
        total_nodes_processed += 1
        logger.debug(
            "Energy import: importing history for %s %s",
            node_type,
            addr,
        )
        all_samples: list[dict[str, Any]] = []
        start_ts = _progress_value(node_type, addr)
        node_raw_samples = 0

        while start_ts > target:
            chunk_start = max(start_ts - day, target)
            sub_stop = start_ts
            while sub_stop > chunk_start:
                sub_start = max(sub_stop - chunk_seconds, chunk_start)
                samples = await _rate_limited_fetch(
                    node_type,
                    addr,
                    sub_start,
                    sub_stop,
                )
                fetched = len(samples)
                if fetched:
                    logger.debug(
                        "%s:%s: fetched %d samples for %s-%s",
                        node_type,
                        addr,
                        fetched,
                        _iso_date(sub_start),
                        _iso_date(sub_stop),
                    )
                    all_samples.extend(samples)
                total_samples_fetched += fetched
                node_raw_samples += fetched
                sub_stop = sub_start

            start_ts = chunk_start
            progress[f"{node_type}:{addr}"] = start_ts
            progress.pop(addr, None)
            _write_progress_options(progress)

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
            node_summary = {
                "node_type": node_type,
                "address": addr,
                "entity_id": None,
                "first_ts": None,
                "last_ts": None,
                "raw_samples": node_raw_samples,
                "samples": 0,
                "written": 0,
                "resets": 0,
                "running_sum": 0.0,
            }
            node_summaries.append(node_summary)
            logger.info(
                "%s:%s energy import summary first=%s last=%s samples=%d "
                "written=%d resets=%d sum=%.3f",
                node_type,
                addr,
                "n/a",
                "n/a",
                0,
                0,
                0,
                0.0,
            )
            continue

        first_ts_val: int | None = None
        last_ts_val: int | None = None
        for sample in all_samples_sorted:
            try:
                ts_candidate = int(sample.get("t"))
            except (TypeError, ValueError):
                continue
            if first_ts_val is None:
                first_ts_val = ts_candidate
            last_ts_val = ts_candidate

        if first_ts_val is None or last_ts_val is None:
            node_summary = {
                "node_type": node_type,
                "address": addr,
                "entity_id": entity_id,
                "first_ts": None,
                "last_ts": None,
                "raw_samples": node_raw_samples,
                "samples": 0,
                "written": 0,
                "resets": 0,
                "running_sum": 0.0,
            }
            node_summaries.append(node_summary)
            logger.info(
                "%s:%s energy import summary first=%s last=%s samples=%d "
                "written=%d resets=%d sum=%.3f",
                node_type,
                addr,
                "n/a",
                "n/a",
                0,
                0,
                0,
                0.0,
            )
            continue

        import_start_dt = datetime_mod.fromtimestamp(first_ts_val, UTC).replace(
            minute=0, second=0, microsecond=0
        )
        import_end_dt = datetime_mod.fromtimestamp(last_ts_val, UTC).replace(
            minute=0, second=0, microsecond=0
        )

        sum_offset = 0.0
        previous_kwh: float | None = None
        last_before: dict[str, Any] | None = None
        last_before_start: datetime | None = None

        external_id: str | None = None
        stat_ids: set[str] = {entity_id}
        if "." in entity_id:
            domain, obj_id = entity_id.split(".", 1)
            external_id = f"{domain}:{obj_id}"
            stat_ids.add(external_id)

        clear_stat_ids: tuple[str, ...]
        if external_id:
            clear_stat_ids = (entity_id, external_id)
        else:
            clear_stat_ids = (entity_id,)

        lookback_days = max(2, max_days + 1)
        lookback_start = import_start_dt - timedelta(days=lookback_days)

        period_stats: dict[str, list[Any]] | None = None
        try:
            period_stats = await stats_period(
                hass,
                lookback_start,
                import_end_dt + timedelta(hours=1),
                stat_ids,
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
            window_values: list[Any] = []
            for stat_id in stat_ids:
                window_values.extend(period_stats.get(stat_id) or [])
            if window_values:
                before_values = [
                    val
                    for val in window_values
                    if isinstance(_statistics_row_get(val, "start"), datetime)
                    and cast(datetime, _statistics_row_get(val, "start"))
                    < import_start_dt
                ]
                if before_values:
                    last_before = max(
                        before_values,
                        key=lambda row: cast(
                            datetime, _statistics_row_get(row, "start")
                        ),
                    )
                    last_before_start = cast(
                        datetime, _statistics_row_get(last_before, "start")
                    )
        else:
            for stat_id in clear_stat_ids:
                try:
                    last_stats = await last_stats_fn(
                        hass,
                        1,
                        stat_id,
                        start_time=import_start_dt,
                    )
                except (
                    async_mod.CancelledError
                ):  # pragma: no cover - allow cancellation
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
                    continue
                if not last_stats:
                    continue
                vals = last_stats.get(stat_id) or []
                if not vals:
                    continue
                candidate = vals[0]
                start_dt = _statistics_row_get(candidate, "start")
                if not isinstance(start_dt, datetime):
                    continue
                if start_dt >= import_start_dt:
                    continue
                if last_before_start is None or start_dt > last_before_start:
                    last_before = candidate
                    last_before_start = start_dt

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

        clear_end = import_end_dt + timedelta(hours=1)
        for stat_id in clear_stat_ids:
            try:
                cleared = await clear_stats_fn(
                    hass,
                    stat_id,
                    start_time=import_start_dt,
                    end_time=clear_end,
                )
            except async_mod.CancelledError:  # pragma: no cover - allow cancellation
                raise
            except Exception as err:  # pragma: no cover - defensive
                logger.error(
                    "%s: failed to clear statistics for %s: %s",
                    addr,
                    stat_id,
                    err,
                    exc_info=True,
                )
                continue
            if cleared == "clear":
                logger.debug("%s: cleared statistics for %s", addr, stat_id)
            elif cleared == "delete":
                logger.debug("%s: deleted statistics for %s", addr, stat_id)
            else:
                if not overlap_warning_logged:
                    logger.info(
                        "%s: recorder statistics delete helpers unavailable; proceeding "
                        "without deleting existing statistics",
                        dev_id,
                    )
                    overlap_warning_logged = True
                logger.debug(
                    "%s: statistics helpers unavailable to clear %s", addr, stat_id
                )

        stats: list[dict[str, Any]] = []
        running_sum: float = sum_offset
        previous_ts: int | None = None
        node_samples_processed = 0
        node_resets_detected = 0

        for sample in all_samples_sorted:
            t_val = sample.get("t")
            counter_val = sample.get("counter")
            try:
                ts = int(t_val)
                kwh = float(counter_val) / 1000.0
            except (TypeError, ValueError):
                logger.debug("%s: invalid sample %s", addr, sample)
                continue

            if previous_ts is not None and ts == previous_ts:
                # Ignore duplicate raw samples that share the same timestamp.
                continue
            previous_ts = ts
            node_samples_processed += 1

            start_dt = datetime_mod.fromtimestamp(ts, UTC).replace(
                minute=0, second=0, microsecond=0
            )
            # Bucket to UTC hour boundaries so statistics align deterministically.

            if previous_kwh is None:
                previous_kwh = kwh
                continue

            delta = kwh - previous_kwh
            if delta <= 0:
                if previous_kwh - kwh >= RESET_DELTA_THRESHOLD_KWH:
                    node_resets_detected += 1
                previous_kwh = kwh
                continue

            running_sum += delta
            stats.append({"start": start_dt, "sum": running_sum})
            previous_kwh = kwh

        first_iso = datetime_mod.fromtimestamp(first_ts_val, UTC).isoformat()
        last_iso = datetime_mod.fromtimestamp(last_ts_val, UTC).isoformat()

        if not stats:
            total_resets_detected += node_resets_detected
            node_summary = {
                "node_type": node_type,
                "address": addr,
                "entity_id": entity_id,
                "first_ts": first_iso,
                "last_ts": last_iso,
                "raw_samples": node_raw_samples,
                "samples": node_samples_processed,
                "written": 0,
                "resets": node_resets_detected,
                "running_sum": running_sum,
            }
            node_summaries.append(node_summary)
            logger.info(
                "%s:%s energy import summary first=%s last=%s samples=%d "
                "written=%d resets=%d sum=%.3f",
                node_type,
                addr,
                first_iso,
                last_iso,
                node_samples_processed,
                0,
                node_resets_detected,
                running_sum,
            )
            continue

        logger.debug("%s: inserting statistics for %s", addr, entity_id)
        ent_entry = ent_reg.async_get(entity_id) if ent_reg else None
        name = getattr(ent_entry, "original_name", None) or entity_id

        domain, _ = entity_id.split(".", 1)
        metadata = {
            "source": domain,
            "statistic_id": entity_id,
            "unit_of_measurement": "kWh",
            "name": name,
            "has_sum": True,
            "has_mean": False,
        }
        logger.debug("%s: adding %d stats entries", addr, len(stats))
        store_failed = False
        try:
            await store_stats(hass, metadata, stats)
        except Exception as err:  # pragma: no cover - log & continue
            store_failed = True
            logger.exception("%s: statistics insert failed: %s", addr, err)

        written_count = 0 if store_failed else len(stats)
        if not store_failed:
            total_samples_written += len(stats)
            try:
                await _enforce_monotonic_sum(
                    hass,
                    entity_id,
                    import_start_dt,
                    import_end_dt,
                )
            except Exception:  # pragma: no cover - defensive
                logger.exception("%s: monotonic sum enforcement failed", addr)
        total_resets_detected += node_resets_detected

        node_summary = {
            "node_type": node_type,
            "address": addr,
            "entity_id": entity_id,
            "first_ts": first_iso,
            "last_ts": last_iso,
            "raw_samples": node_raw_samples,
            "samples": node_samples_processed,
            "written": written_count,
            "resets": node_resets_detected,
            "running_sum": running_sum,
        }
        node_summaries.append(node_summary)
        logger.info(
            "%s:%s energy import summary first=%s last=%s samples=%d "
            "written=%d resets=%d sum=%.3f",
            node_type,
            addr,
            first_iso,
            last_iso,
            node_samples_processed,
            written_count,
            node_resets_detected,
            running_sum,
        )

    imported_flag: bool | None = None
    if target_pairs and all(
        _progress_value(node_type, addr) <= target for node_type, addr in target_pairs
    ):
        imported_flag = True
    _write_progress_options(progress, imported=imported_flag)

    run_summary = {
        "device_id": dev_id,
        "started_at": run_started_iso,
        "completed_at": datetime_mod.now(UTC).isoformat(),
        "filters": filters_snapshot,
        "nodes_processed": total_nodes_processed,
        "samples_fetched": total_samples_fetched,
        "samples_written": total_samples_written,
        "resets_detected": total_resets_detected,
        "nodes": node_summaries,
    }

    try:
        rec[SUMMARY_KEY_LAST_RUN] = run_summary  # type: ignore[index]
    except Exception:  # pragma: no cover - defensive
        logger.debug("%s: unable to store energy import summary on record", dev_id)

    logger.info(
        "%s: energy history import summary nodes=%d samples=%d written=%d resets=%d",
        dev_id,
        total_nodes_processed,
        total_samples_fetched,
        total_samples_written,
        total_resets_detected,
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
        for entry_id, rec in records.items():
            if not isinstance(rec, Mapping):
                continue
            ent: ConfigEntry | None = rec.get("config_entry")
            if not ent:
                continue
            try:
                Inventory.require_from_context(container=rec)
            except LookupError:
                entry_entry_id = getattr(ent, "entry_id", "<unknown>")
                logger.error(
                    "%s: energy import aborted; inventory missing in integration state (entry=%s)",
                    rec.get("dev_id"),
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
                    rec.get("dev_id"),
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
        DOMAIN, "import_energy_history", _service_import_energy_history
    )
