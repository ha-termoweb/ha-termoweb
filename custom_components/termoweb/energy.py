"""Energy history import into Home Assistant long-term statistics.

The import walks each node's window forward one day at a time. Every chunk is
written with ``async_import_statistics`` (an upsert keyed by ``start``) and the
recorder commit is awaited before the chunk is recorded as done in a ``Store``,
so an interrupted import resumes without gaps. Sums continue from the statistic
before the window; once the window is complete, every later statistic is shifted
with ``async_adjust_statistics`` so there is no step at the seam.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import logging
from typing import Any

from aiohttp import ClientError
from homeassistant.components.recorder import get_instance
from homeassistant.components.recorder.models import StatisticMeanType
from homeassistant.components.recorder.statistics import (
    async_import_statistics,
    statistics_during_period,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import EnergyConverter

from .backend.rest_client import BackendAuthError, BackendRateLimitError
from .const import DOMAIN
from .identifiers import build_heater_energy_unique_id
from .runtime import EntryRuntime
from .throttle import default_samples_rate_limit_state

_LOGGER = logging.getLogger(__name__)

STORE_VERSION = 1
DEFAULT_MAX_HISTORY_DAYS = 7
MAX_HISTORY_DAYS = 3650
RESET_DELTA_THRESHOLD_KWH = 0.2
HOUR = 3600
DAY = 24 * HOUR
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_UNIT = "kWh"

LEGACY_OPTION_PROGRESS = "energy_history_progress"
LEGACY_OPTION_IMPORTED = "energy_history_imported"
LEGACY_OPTION_MAX_DAYS = "max_history_retrieved"
_LEGACY_OPTIONS = (
    LEGACY_OPTION_PROGRESS,
    LEGACY_OPTION_IMPORTED,
    LEGACY_OPTION_MAX_DAYS,
)


class EnergyImportError(HomeAssistantError):
    """A sample fetch failed; progress stays at the last written chunk."""


@dataclass(slots=True)
class NodeSummary:
    """Per-node outcome of one import run."""

    node_type: str
    address: str
    entity_id: str
    requests: int = 0
    samples: int = 0
    written: int = 0
    resets: int = 0
    sum: float = 0.0


def energy_import_store(hass: HomeAssistant, entry_id: str) -> Store[dict[str, Any]]:
    """Return the Store that holds import progress for ``entry_id``."""
    return Store(hass, STORE_VERSION, f"{DOMAIN}.energy_import.{entry_id}")


def _floor_hour(ts: float) -> int:
    """Return ``ts`` rounded down to the hour."""
    return int(ts) - int(ts) % HOUR


def _utc(ts: int) -> datetime:
    """Return a UTC datetime for a unix timestamp."""
    return datetime.fromtimestamp(ts, UTC)


async def _async_load_progress(
    hass: HomeAssistant, entry: ConfigEntry, store: Store[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Load node progress, moving legacy ``entry.options`` keys into the Store."""
    data = await store.async_load() or {}
    nodes: dict[str, dict[str, Any]] = dict(data.get("nodes", {}))
    if not any(key in entry.options for key in _LEGACY_OPTIONS):
        return nodes
    legacy = entry.options.get(LEGACY_OPTION_PROGRESS)
    # Only a completed legacy import is trusted; partial runs lost data (they
    # saved progress before writing statistics), so those nodes re-import.
    if entry.options.get(LEGACY_OPTION_IMPORTED) and isinstance(legacy, Mapping):
        for key, value in legacy.items():
            if ":" in str(key) and isinstance(value, int | float):
                nodes.setdefault(str(key), {"imported_from": _floor_hour(value)})
    await store.async_save({"nodes": nodes})
    options = {k: v for k, v in entry.options.items() if k not in _LEGACY_OPTIONS}
    hass.config_entries.async_update_entry(entry, options=options)
    _LOGGER.info("%s: moved legacy energy import progress to storage", entry.entry_id)
    return nodes


async def _async_read_sums(
    hass: HomeAssistant, statistic_id: str, end: int
) -> list[tuple[float, float]]:
    """Return ``(start_ts, sum)`` for every hourly statistic starting before ``end``."""
    rows = await get_instance(hass).async_add_executor_job(
        statistics_during_period,
        hass,
        _EPOCH,
        _utc(end),
        {statistic_id},
        "hour",
        None,
        {"sum"},
    )
    return [
        (row["start"], row["sum"])
        for row in rows.get(statistic_id, [])
        if row.get("sum") is not None
    ]


def _hour_rows(
    window: dict[str, Any],
    samples: list[dict[str, Any]],
    chunk_start: int,
    chunk_end: int,
    summary: NodeSummary,
) -> list[dict[str, Any]]:
    """Advance ``window`` over one chunk and return its hourly statistic rows.

    A sample's ``counter`` is the cumulative Wh total at ``t``, so the delta
    between samples at T and T+1h is consumption during the hour starting at T.
    Hours without samples carry the running sum forward.
    """
    points: dict[int, float] = {}
    for sample in samples:
        try:
            ts = int(sample["t"])
            kwh = float(sample["counter"]) / 1000.0
        except (KeyError, TypeError, ValueError):
            _LOGGER.debug("%s: ignoring invalid sample %s", summary.entity_id, sample)
            continue
        if chunk_start <= ts <= chunk_end:
            points[ts] = kwh

    started = window["prev_t"] is not None
    first_hour = chunk_start
    consumption: dict[int, float] = {}
    for ts in sorted(points):
        kwh = points[ts]
        if window["prev_t"] is None:
            first_hour = _floor_hour(ts)
        elif ts <= window["prev_t"]:
            continue
        else:
            delta = kwh - window["prev_kwh"]
            if delta >= 0:
                bucket = max(_floor_hour(window["prev_t"]), chunk_start)
                consumption[bucket] = consumption.get(bucket, 0.0) + delta
            elif -delta >= RESET_DELTA_THRESHOLD_KWH:
                summary.resets += 1
        window["prev_t"], window["prev_kwh"] = ts, kwh
        summary.samples += 1
        started = True

    if not started:
        return []
    rows: list[dict[str, Any]] = []
    for hour in range(first_hour, chunk_end, HOUR):
        window["sum"] += consumption.get(hour, 0.0)
        rows.append({"start": _utc(hour), "sum": window["sum"]})
    return rows


async def _async_fetch(
    runtime: EntryRuntime, node_type: str, addr: str, start: int, end: int
) -> list[dict[str, Any]]:
    """Fetch one chunk of samples through the shared samples rate limiter."""
    await default_samples_rate_limit_state().async_throttle()
    _LOGGER.debug("%s:%s: requesting samples %s-%s", node_type, addr, start, end)
    try:
        return await runtime.client.get_node_samples(
            runtime.dev_id, (node_type, addr), start, end
        )
    except (BackendAuthError, BackendRateLimitError, ClientError, TimeoutError) as err:
        raise EnergyImportError(
            f"Energy import stopped while fetching {node_type} {addr} samples "
            f"({type(err).__name__}); progress is saved, run the action again "
            "later to resume"
        ) from err


async def _async_import_window(
    hass: HomeAssistant,
    runtime: EntryRuntime,
    store: Store[dict[str, Any]],
    progress: dict[str, dict[str, Any]],
    *,
    node: dict[str, Any],
    summary: NodeSummary,
) -> None:
    """Import the node's pending window chunk by chunk, then fix later sums."""
    window = node["window"]
    # Matches the sensor platform's own metadata.
    metadata = {
        "has_sum": True,
        "mean_type": StatisticMeanType.NONE,
        "name": None,
        "source": "recorder",
        "statistic_id": summary.entity_id,
        "unit_class": EnergyConverter.UNIT_CLASS,
        "unit_of_measurement": _UNIT,
    }
    recorder = get_instance(hass)
    while window["next"] < window["end"]:
        chunk_start = window["next"]
        chunk_end = min(chunk_start + DAY, window["end"])
        samples = await _async_fetch(
            runtime, summary.node_type, summary.address, chunk_start, chunk_end
        )
        summary.requests += 1
        advanced = dict(window)
        rows = _hour_rows(advanced, samples, chunk_start, chunk_end, summary)
        if rows:
            async_import_statistics(hass, metadata, rows)
            await recorder.async_block_till_done()
            summary.written += len(rows)
        advanced["next"] = chunk_end
        node["window"] = window = advanced
        await store.async_save({"nodes": progress})

    summary.sum = window["sum"]
    offset = window["sum"] - window["ref"]
    if abs(offset) > 1e-9:
        recorder.async_adjust_statistics(
            summary.entity_id, _utc(window["end"]), offset, _UNIT
        )
        await recorder.async_block_till_done()
    imported_from = node.get("imported_from")
    node.pop("window")
    node["imported_from"] = (
        window["start"]
        if imported_from is None
        else min(imported_from, window["start"])
    )
    await store.async_save({"nodes": progress})


async def _async_import_node(
    hass: HomeAssistant,
    runtime: EntryRuntime,
    store: Store[dict[str, Any]],
    progress: dict[str, dict[str, Any]],
    summary: NodeSummary,
    *,
    requested_start: int,
    now_end: int,
) -> None:
    """Resume or start the node's import until ``requested_start`` is covered."""
    node = progress.setdefault(f"{summary.node_type}:{summary.address}", {})
    while True:
        if "window" not in node:
            end = node.get("imported_from", now_end)
            if requested_start >= end:
                return
            sums = await _async_read_sums(hass, summary.entity_id, end)
            before = [value for start, value in sums if start < requested_start]
            node["window"] = {
                "start": requested_start,
                "end": end,
                "next": requested_start,
                "sum": before[-1] if before else 0.0,
                "ref": sums[-1][1] if sums else 0.0,
                "prev_t": None,
                "prev_kwh": None,
            }
        await _async_import_window(
            hass, runtime, store, progress, node=node, summary=summary
        )


def _log_summary(summary: NodeSummary) -> None:
    """Log one node's import summary."""
    _LOGGER.info(
        "%s:%s energy import: requests=%d samples=%d written=%d resets=%d sum=%.3f",
        summary.node_type,
        summary.address,
        summary.requests,
        summary.samples,
        summary.written,
        summary.resets,
        summary.sum,
    )


async def async_import_energy_history(
    hass: HomeAssistant,
    runtime: EntryRuntime,
    *,
    max_days: int = DEFAULT_MAX_HISTORY_DAYS,
    reset_progress: bool = False,
) -> None:
    """Import up to ``max_days`` of hourly energy history for one entry."""
    entry = runtime.config_entry
    store = energy_import_store(hass, entry.entry_id)
    progress = await _async_load_progress(hass, entry, store)
    if reset_progress:
        progress.clear()
        await store.async_save({"nodes": progress})

    now_end = _floor_hour(dt_util.utcnow().timestamp())
    requested_start = now_end - max_days * DAY
    started_at = dt_util.utcnow().isoformat()
    _LOGGER.info(
        "%s: energy history import from %s (reset=%s)",
        entry.entry_id,
        _utc(requested_start).isoformat(),
        reset_progress,
    )

    inventory = runtime.inventory
    targets = dict.fromkeys(
        [*inventory.heater_sample_targets, *inventory.power_monitor_sample_targets]
    )
    ent_reg = er.async_get(hass)
    summaries: list[NodeSummary] = []
    try:
        for node_type, addr in targets:
            unique_id = build_heater_energy_unique_id(runtime.dev_id, node_type, addr)
            entity_id = ent_reg.async_get_entity_id("sensor", DOMAIN, unique_id)
            if entity_id is None:
                _LOGGER.debug("%s:%s: no energy sensor; skipping", node_type, addr)
                continue
            summary = NodeSummary(node_type, addr, entity_id)
            summaries.append(summary)
            try:
                await _async_import_node(
                    hass,
                    runtime,
                    store,
                    progress,
                    summary,
                    requested_start=requested_start,
                    now_end=now_end,
                )
            finally:
                _log_summary(summary)
    finally:
        runtime.last_energy_import_summary = {
            "started_at": started_at,
            "completed_at": dt_util.utcnow().isoformat(),
            "max_days": max_days,
            "reset_progress": reset_progress,
            "nodes": [asdict(summary) for summary in summaries],
        }
