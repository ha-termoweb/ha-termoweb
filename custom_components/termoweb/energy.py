"""Energy history helpers for the TermoWeb integration."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime, timedelta
import importlib
import importlib.util
import inspect
import logging
import sys
from typing import Any, cast

from homeassistant.components.recorder.statistics import (
    async_delete_statistics,
    async_get_last_statistics,
    async_get_statistics_during_period,
    async_import_statistics,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er

from .api import RESTClient
from .const import DOMAIN
from .identifiers import build_heater_energy_unique_id
from .inventory import Inventory, normalize_node_addr, normalize_node_type
from .runtime import require_runtime
from .throttle import MonotonicRateLimiter

_LOGGER = logging.getLogger(__name__)


class RecorderImports:
    """Container for recorder imports that may be missing at runtime."""

    def __init__(
        self, get_instance: Callable[..., Any] | None, statistics: Any
    ) -> None:
        """Initialize the recorder import references."""

        self.get_instance = get_instance
        self.statistics = statistics


OPTION_ENERGY_HISTORY_IMPORTED = "energy_history_imported"
OPTION_ENERGY_HISTORY_PROGRESS = "energy_history_progress"
OPTION_MAX_HISTORY_RETRIEVED = "max_history_retrieved"

DEFAULT_MAX_HISTORY_DAYS = 7
RESET_DELTA_THRESHOLD_KWH = 0.2
SUMMARY_KEY_LAST_RUN = "last_energy_import_summary"


def _iso_date(ts: int) -> str:
    """Convert unix timestamp to ISO date."""

    return datetime.fromtimestamp(ts, UTC).date().isoformat()


_RECORDER_IMPORTS: RecorderImports | None = None


def _resolve_recorder_imports() -> RecorderImports:
    """Return cached recorder imports for statistics helpers."""
    cached = globals().get("_RECORDER_IMPORTS")
    if isinstance(cached, RecorderImports):
        return cached

    if "homeassistant.components.recorder" in sys.modules:
        recorder_mod = sys.modules["homeassistant.components.recorder"]
    else:
        spec = importlib.util.find_spec("homeassistant.components.recorder")
        if spec is None:
            cached = RecorderImports(None, None)
            globals()["_RECORDER_IMPORTS"] = cached
            return cached
        recorder_mod = importlib.import_module("homeassistant.components.recorder")
    statistics = getattr(recorder_mod, "statistics", None)
    get_instance = getattr(recorder_mod, "get_instance", None)
    cached = RecorderImports(get_instance, statistics)
    globals()["_RECORDER_IMPORTS"] = cached
    return cached


async def _clear_statistics_compat(
    hass: HomeAssistant,
    statistic_id: str,
    *,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> str:
    """Delete statistics using recorder instance APIs when possible."""

    recorder_imports = _resolve_recorder_imports()
    instance = None
    if callable(recorder_imports.get_instance):
        instance = recorder_imports.get_instance(hass)
    if instance is not None:
        async_delete = getattr(instance, "async_delete_statistics", None)
        if callable(async_delete):
            await async_delete(
                [statistic_id],
                start_time=start_time,
                end_time=end_time,
            )
            return "delete"
        async_clear = getattr(instance, "async_clear_statistics", None)
        if callable(async_clear):
            await async_clear([statistic_id])
            return "delete"

    await _delete_statistics(
        hass,
        statistic_id,
        start_time=start_time,
        end_time=end_time,
    )
    return "delete"


def _resolve_statistics_module() -> Any:
    """Return the recorder statistics module when available."""

    if "homeassistant.components.recorder.statistics" in sys.modules:
        return sys.modules["homeassistant.components.recorder.statistics"]
    recorder_imports = _resolve_recorder_imports()
    return recorder_imports.statistics


async def _statistics_during_period(
    hass: HomeAssistant,
    start_time: datetime,
    end_time: datetime,
    statistic_ids: set[str],
) -> dict[str, list[Any]]:
    """Return recorder statistics rows for the provided period."""

    stats_mod = _resolve_statistics_module()
    stats_func = getattr(stats_mod, "async_get_statistics_during_period", None)
    if not callable(stats_func):
        stats_func = async_get_statistics_during_period
    return await stats_func(
        hass,
        start_time,
        end_time,
        statistic_ids,
        period="hour",
        types={"state", "sum"},
    )


async def _statistics_during_period_compat(
    hass: HomeAssistant,
    start_time: datetime,
    end_time: datetime,
    statistic_ids: set[str],
) -> dict[str, list[Any]]:
    """Return recorder statistics rows with compatibility shims."""

    return await _statistics_during_period(
        hass,
        start_time,
        end_time,
        statistic_ids,
    )


async def _get_last_statistics(
    hass: HomeAssistant,
    number_of_stats: int,
    statistic_id: str,
    *,
    types: set[str] | None = None,
    start_time: datetime | None = None,
) -> dict[str, list[Any]]:
    """Return the most recent recorder statistics row for ``statistic_id``."""

    types = types or {"state", "sum"}
    stats_mod = _resolve_statistics_module()
    stats_func = getattr(stats_mod, "async_get_last_statistics", None)
    if not callable(stats_func):
        stats_func = async_get_last_statistics
    return await stats_func(
        hass,
        number_of_stats,
        [statistic_id],
        types=types,
        start_time=start_time,
    )


async def _delete_statistics(
    hass: HomeAssistant,
    statistic_id: str,
    *,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> str:
    """Delete recorder statistics rows for ``statistic_id``."""

    stats_mod = _resolve_statistics_module()
    stats_func = getattr(stats_mod, "async_delete_statistics", None)
    if not callable(stats_func):
        stats_func = async_delete_statistics
    await stats_func(
        hass,
        [statistic_id],
        start_time=start_time,
        end_time=end_time,
    )
    return "delete"


async def _store_statistics(
    hass: HomeAssistant, metadata: dict[str, Any], stats: list[dict[str, Any]]
) -> None:
    """Insert entity statistics using the recorder import helper."""

    if not stats:
        return

    stat_id = metadata.get("statistic_id")
    if not isinstance(stat_id, str) or "." not in stat_id:
        raise ValueError("metadata must include an entity statistic_id")

    import_metadata = dict(metadata)
    import_metadata.update({"source": "recorder", "statistic_id": stat_id})

    stats_mod = _resolve_statistics_module()
    stats_func = getattr(stats_mod, "async_import_statistics", None)
    if not callable(stats_func):
        stats_func = async_import_statistics
    result = stats_func(hass, import_metadata, stats)
    if inspect.isawaitable(result):
        await result


def _statistics_row_get(row: Any, key: str) -> Any:
    """Read a field from a statistics row regardless of its container type."""

    if isinstance(row, dict):
        return row.get(key)
    return getattr(row, key, None)  # pragma: no cover - attribute rows rare in tests


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


async def async_import_energy_history(  # noqa: C901
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
    last_stats_fn = _get_last_statistics
    clear_stats_fn = _clear_statistics_compat

    try:
        runtime = require_runtime(hass, entry.entry_id)
    except LookupError:
        logger.debug("%s: no record found for energy import", entry.entry_id)
        return
    client: RESTClient = runtime.client  # type: ignore[assignment]
    dev_id: str = runtime.dev_id

    if nodes is not None and not isinstance(nodes, Inventory):
        raise TypeError(
            "async_import_energy_history nodes must be an Inventory instance"
        )

    if nodes is not None:
        inventory = nodes
    else:
        inventory = runtime.inventory
    if not isinstance(inventory, Inventory):
        logger.error(
            "%s: energy import aborted; inventory missing in integration state",
            dev_id,
        )
        return

    addresses_by_type = inventory.addresses_by_type

    if not any(addresses_by_type.values()):
        logger.debug("Energy import: no nodes available for device")
        return

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
            addresses = addresses_by_type.get(normalized)
            if not addresses:
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
            if not any(
                normalized_addr in bucket for bucket in addresses_by_type.values()
            ):
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

    target_pairs: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()

    def _extend_targets(pairs: Iterable[tuple[str, str]]) -> None:
        for node_type, addr in pairs:
            normalized_type = node_type.strip() if isinstance(node_type, str) else ""
            normalized_addr = addr.strip() if isinstance(addr, str) else ""
            if not normalized_type or not normalized_addr:
                continue
            if not inventory.has_node(normalized_type, normalized_addr):
                continue
            if (
                normalized_type_filters is not None
                and normalized_type not in normalized_type_filters
            ):
                continue
            if (
                normalized_address_filters is not None
                and normalized_addr not in normalized_address_filters
            ):
                continue
            pair = (normalized_type, normalized_addr)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            target_pairs.append(pair)

    _extend_targets(inventory.heater_sample_targets)
    _extend_targets(inventory.power_monitor_sample_targets)
    _extend_targets(
        (metadata.node_type, metadata.addr)
        for metadata in inventory.iter_nodes_metadata()
    )

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
        except Exception as err:  # pragma: no cover - defensive  # noqa: BLE001
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
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "%s: error fetching statistics window %s-%s",
                addr,
                lookback_start,
                import_end_dt,
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
                except Exception:  # pragma: no cover - defensive
                    logger.exception(
                        "%s: error fetching statistics window %s-%s",
                        addr,
                        lookback_start,
                        import_end_dt,
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
            except Exception:  # pragma: no cover - defensive
                logger.exception(
                    "%s: failed to clear statistics for %s",
                    addr,
                    stat_id,
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
            if stats and stats[-1]["start"] == start_dt:
                stats[-1]["sum"] = running_sum
            else:
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

        metadata = {
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
        except Exception:  # pragma: no cover - log & continue
            store_failed = True
            logger.exception("%s: statistics insert failed", addr)

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
        runtime[SUMMARY_KEY_LAST_RUN] = run_summary
    except Exception:  # pragma: no cover - defensive  # noqa: BLE001
        logger.debug("%s: unable to store energy import summary on record", dev_id)

    logger.info(
        "%s: energy history import summary nodes=%d samples=%d written=%d resets=%d",
        dev_id,
        total_nodes_processed,
        total_samples_fetched,
        total_samples_written,
        total_resets_detected,
    )
