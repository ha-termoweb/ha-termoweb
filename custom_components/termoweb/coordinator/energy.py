"""Energy coordinator: polls and decodes energy samples into snapshots."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, MutableMapping
from datetime import datetime
from itertools import chain
import logging
import time
from time import monotonic as time_mod
import typing
from typing import Any

from aiohttp import ClientError
from homeassistant.core import HomeAssistant
from homeassistant.helpers.event import async_track_time_change
from homeassistant.helpers.update_coordinator import UpdateFailed

from custom_components.termoweb.backend.base import SAMPLE_COUNTER_SCALES
from custom_components.termoweb.backend.rest_client import (
    BackendAuthError,
    BackendRateLimitError,
    RESTClient,
)
from custom_components.termoweb.coerce import as_float
from custom_components.termoweb.coordinator.base import (
    RaiseUpdateFailedCoordinator,
    wrap_logger,
)
from custom_components.termoweb.coordinator.state import StateCoordinator
from custom_components.termoweb.domain.energy import (
    EnergyNodeMetrics,
    EnergySnapshot,
    build_empty_snapshot,
)
from custom_components.termoweb.domain.ids import (
    ENERGY_NODE_TYPES,
    NodeId as DomainNodeId,
    NodeType as DomainNodeType,
)
from custom_components.termoweb.inventory import Inventory, normalize_node_addr

_LOGGER = logging.getLogger(__name__)

# Minute past each hour at which REST energy samples are polled, so the
# previous hour's final sample is available.
ENERGY_POLL_MINUTE = 5
# Upper bound on hourly energy polls skipped after a 429.
ENERGY_RATE_LIMIT_MAX_SKIP = 8


class EnergyStateCoordinator(
    RaiseUpdateFailedCoordinator[EnergySnapshot]
):  # dev_id -> per-device data
    """Polls heater energy counters and exposes energy and power per heater."""

    def __init__(
        self,
        hass: HomeAssistant,
        client: RESTClient,
        dev_id: str,
        inventory: Inventory,
        state_coordinator: StateCoordinator | None = None,
    ) -> None:
        """Initialize the heater energy coordinator."""
        super().__init__(
            hass,
            logger=wrap_logger(_LOGGER),
            name="termoweb-htr-energy",
            # REST samples are polled at HH:05 by ``async_start_hourly_poll``.
            update_interval=None,
        )
        if not isinstance(inventory, Inventory):
            msg = "Energy inventory is unavailable"
            raise TypeError(msg)
        self.client = client
        self._dev_id = dev_id
        self._inventory: Inventory = inventory
        self._last: dict[tuple[str, str], tuple[float, float]] = {}
        self._rate_limit_backoff = 0  # hourly polls to skip after the next 429
        self._skip_polls = 0
        self._ws_lease: float = 0.0
        self._ws_deadline: float | None = None
        self._ws_margin_default = 60.0
        if isinstance(state_coordinator, StateCoordinator):
            self._state_coordinator = state_coordinator
        else:
            self._state_coordinator = None
        self.data = build_empty_snapshot(dev_id)

    def _publish_to_state_store(self, snapshot: EnergySnapshot) -> None:
        """Update the canonical state store with ``snapshot``."""

        coordinator = self._state_coordinator
        if coordinator is None:
            return
        coordinator.apply_energy_snapshot(snapshot)

    def async_set_updated_data(self, data: EnergySnapshot) -> None:
        """Store ``data`` in the domain state before notifying listeners."""

        self._publish_to_state_store(data)
        super().async_set_updated_data(data)

    @staticmethod
    def _targets_by_type(inventory: Inventory) -> dict[str, list[str]]:
        """Return the inventory's energy sample addresses grouped by node type."""

        buckets: dict[str, list[str]] = {}
        for node_type, addr in chain(
            inventory.heater_sample_targets,
            inventory.power_monitor_sample_targets,
        ):
            buckets.setdefault(node_type, []).append(addr)
        return buckets

    def _prefill_energy_buckets(
        self,
        dev_id: str,
        targets_by_type: Mapping[str, list[str]],
        energy_by_type: dict[str, dict[str, float]],
        power_by_type: dict[str, dict[str, float]],
    ) -> None:
        """Seed energy and power buckets from cached coordinator state."""

        for node_id, current in self.data.iter_metrics():
            node_type = node_id.node_type.value
            energy_by_type[node_type].setdefault(node_id.addr, current.energy_kwh)
            if current.power_w is not None:
                power_by_type[node_type].setdefault(node_id.addr, current.power_w)

        for (node_type, addr), (_, kwh) in self._last.items():
            energy_by_type[node_type].setdefault(addr, kwh)

    def _build_snapshot(
        self,
        dev_id: str,
        targets_by_type: Mapping[str, list[str]],
        energy_by_type: Mapping[str, Mapping[str, float]],
        power_by_type: Mapping[str, Mapping[str, float]],
        *,
        source: str,
        updated_at: float,
        ws_deadline: float | None,
    ) -> EnergySnapshot:
        """Return an :class:`EnergySnapshot` built from metric buckets."""

        metrics: dict[DomainNodeId, EnergyNodeMetrics] = {}
        for node_type, addrs_for_type in targets_by_type.items():
            for addr in addrs_for_type:
                node_id = DomainNodeId(DomainNodeType(node_type), addr)
                energy_value = energy_by_type.get(node_type, {}).get(addr)
                power_value = power_by_type.get(node_type, {}).get(addr)
                if energy_value is None and power_value is None:
                    continue
                last_ts = self._last.get((node_type, addr), (updated_at, 0.0))[0]
                metrics[node_id] = EnergyNodeMetrics(
                    energy_kwh=energy_value,
                    power_w=power_value,
                    source=source,
                    ts=last_ts,
                )

        return EnergySnapshot(
            dev_id=dev_id,
            metrics=metrics,
            updated_at=updated_at,
            ws_deadline=ws_deadline,
        )

    async def _poll_recent_samples(
        self,
        dev_id: str,
        targets_by_type: Mapping[str, list[str]],
        energy_by_type: dict[str, dict[str, float]],
        power_by_type: dict[str, dict[str, float]],
    ) -> None:
        """Fetch recent energy samples for every tracked node; stop on a 429."""

        for node_type, addrs_for_type in targets_by_type.items():
            for addr in addrs_for_type:
                now = time.time()
                start = now - 3600  # fetch recent samples
                try:
                    samples = await self.client.get_node_samples(
                        dev_id, (node_type, addr), start, now
                    )
                except BackendRateLimitError as err:
                    self._rate_limit_backoff = min(
                        max(1, self._rate_limit_backoff * 2),
                        ENERGY_RATE_LIMIT_MAX_SKIP,
                    )
                    self._skip_polls = self._rate_limit_backoff
                    _LOGGER.warning(
                        "Energy poll rate limited (%s); skipping the next %d "
                        "hourly polls",
                        err,
                        self._skip_polls,
                    )
                    return
                except ClientError, BackendAuthError:
                    samples = []

                if not samples:
                    _LOGGER.debug(
                        "No energy samples for node_type=%s addr=%s",
                        node_type,
                        addr,
                    )
                    continue

                last = samples[-1]
                counter = as_float(last.get("counter"))
                if counter is None:
                    counter = as_float(last.get("counter_max"))
                if counter is None:
                    counter = as_float(last.get("counter_min"))
                t = as_float(last.get("t"))
                if counter is None or t is None:
                    _LOGGER.debug(
                        "Latest sample missing 't' or 'counter' for "
                        "node_type=%s addr=%s",
                        node_type,
                        addr,
                    )
                    continue

                energy_bucket = energy_by_type.setdefault(node_type, {})
                power_bucket = power_by_type.setdefault(node_type, {})
                self._process_energy_sample(
                    node_type,
                    addr,
                    t,
                    counter,
                    energy_bucket,
                    power_bucket,
                )
        self._rate_limit_backoff = 0

    def _process_energy_sample(
        self,
        node_type: str,
        addr: str,
        when: float,
        counter: float,
        energy_bucket: MutableMapping[str, float],
        power_bucket: MutableMapping[str, float],
        *,
        prune_power: bool = False,
    ) -> bool:
        """Update cached energy and derived power for ``addr``."""

        kwh = counter / SAMPLE_COUNTER_SCALES[node_type]
        key = (node_type, addr)
        prev = self._last.get(key)
        if prev and when <= prev[0]:
            # Stale or duplicate sample: never rewind the published counter.
            return False

        energy_bucket[addr] = kwh
        if prev:
            prev_t, prev_kwh = prev
            if kwh < prev_kwh:
                # Newer sample with a lower counter: genuine meter reset.
                self._last[key] = (when, kwh)
                if prune_power:
                    power_bucket.pop(addr, None)
                    return True
                return False

            delta_kwh = kwh - prev_kwh
            power_bucket[addr] = delta_kwh / ((when - prev_t) / 3600.0) * 1000.0
            self._last[key] = (when, kwh)
            return prune_power

        self._last[key] = (when, kwh)
        if prune_power:
            power_bucket.pop(addr, None)
            return True
        return False

    async def _async_update_data(self) -> EnergySnapshot:
        """Fetch recent heater energy samples and derive totals and power."""
        if self._should_skip_poll():
            _LOGGER.debug("Energy poll skipped (fresh websocket samples)")
            self._publish_to_state_store(self.data)
            return self.data

        dev_id = self._dev_id
        try:
            inventory = self._inventory
            targets_by_type = self._targets_by_type(inventory)

            energy_by_type: dict[str, dict[str, float]] = {
                node_type: {} for node_type in targets_by_type
            }
            power_by_type: dict[str, dict[str, float]] = {
                node_type: {} for node_type in targets_by_type
            }

            self._prefill_energy_buckets(
                dev_id,
                targets_by_type,
                energy_by_type,
                power_by_type,
            )

            await self._poll_recent_samples(
                dev_id,
                targets_by_type,
                energy_by_type,
                power_by_type,
            )

            now = time_mod()
            snapshot = self._build_snapshot(
                dev_id,
                targets_by_type,
                energy_by_type,
                power_by_type,
                source="rest",
                updated_at=now,
                ws_deadline=None,
            )

        except TimeoutError as err:
            raise UpdateFailed("API timeout") from err
        else:
            self._ws_deadline = None
            self._publish_to_state_store(snapshot)
            return snapshot

    def async_start_hourly_poll(self) -> Callable[[], None]:
        """Schedule the HH:05 REST sample poll; return its unsubscribe callback."""

        return async_track_time_change(
            self.hass, self._async_hourly_poll, minute=ENERGY_POLL_MINUTE, second=0
        )

    async def _async_hourly_poll(self, _now: datetime | None = None) -> None:
        """Poll REST samples for the past hour unless backing off after a 429."""

        if self._skip_polls > 0:
            self._skip_polls -= 1
            _LOGGER.debug(
                "Hourly energy poll skipped (rate-limit backoff, %d left)",
                self._skip_polls,
            )
            return
        try:
            await self.async_refresh()
        except UpdateFailed as err:
            _LOGGER.debug("Hourly energy poll failed: %s", err)

    def _should_skip_poll(self) -> bool:
        """Return True when websocket pushes keep energy data fresh."""

        return self._ws_deadline is not None and time_mod() < self._ws_deadline

    def _ws_margin_seconds(self) -> float:
        """Return the buffer to wait after a positive websocket lease expires."""

        return max(self._ws_margin_default, min(self._ws_lease * 0.25, 600.0))

    @staticmethod
    def _extract_sample_point(payload: Any) -> tuple[float, float] | None:
        """Extract ``(timestamp, counter)`` from websocket sample payloads."""

        if isinstance(payload, Mapping):
            nested = payload.get("samples")
            if nested is not None and nested is not payload:
                extracted = EnergyStateCoordinator._extract_sample_point(nested)
                if extracted:
                    return extracted
            t = as_float(payload.get("t"))
            if t is None:
                return None
            counter_raw = payload.get("counter")
            counter = as_float(counter_raw)
            counter_min = as_float(payload.get("counter_min"))
            counter_max = as_float(payload.get("counter_max"))
            if isinstance(counter_raw, Mapping):
                counter = as_float(counter_raw.get("value")) or as_float(
                    counter_raw.get("counter")
                )
                counter_min = counter_min or as_float(counter_raw.get("min"))
                counter_max = counter_max or as_float(counter_raw.get("max"))
            if counter is None:
                counter = counter_max or counter_min
            if counter is None:
                counter = as_float(payload.get("value"))
            if counter is None:
                return None
            return t, counter
        if isinstance(payload, Iterable) and not isinstance(payload, (str, bytes)):
            latest: tuple[float, float] | None = None
            for item in payload:
                extracted = EnergyStateCoordinator._extract_sample_point(item)
                if extracted is None:
                    continue
                if latest is None or extracted[0] >= latest[0]:
                    latest = extracted
            return latest
        return None

    def handle_ws_samples(
        self,
        dev_id: str,
        updates: Mapping[str, Mapping[str, typing.Any]],
        *,
        lease_seconds: float | None = None,
    ) -> None:
        """Update cached heater metrics from websocket ``samples`` payloads."""

        if dev_id != self._dev_id:
            return

        if lease_seconds is not None:
            self._ws_lease = float(lease_seconds) if lease_seconds > 0 else 0.0

        now = time_mod()
        if self._ws_lease > 0:
            # Fresh WS samples suspend the hourly REST poll until the lease
            # (plus margin) runs out; they never make REST polling more frequent.
            self._ws_deadline = now + self._ws_lease + self._ws_margin_seconds()
        else:
            self._ws_deadline = None

        inventory = self._inventory
        targets_by_type = self._targets_by_type(inventory)
        alias_map = inventory.sample_alias_map(
            include_types=ENERGY_NODE_TYPES,
            restrict_to=ENERGY_NODE_TYPES,
        )

        energy_by_type: dict[str, dict[str, float]] = {
            node_type: {} for node_type in targets_by_type
        }
        power_by_type: dict[str, dict[str, float]] = {
            node_type: {} for node_type in targets_by_type
        }
        for node_id, metrics in self.data.iter_metrics():
            node_type = node_id.node_type.value
            energy_by_type[node_type][node_id.addr] = metrics.energy_kwh
            if metrics.power_w is not None:
                power_by_type[node_type][node_id.addr] = metrics.power_w

        changed = False

        for node_type, payload in updates.items():
            canonical_type = alias_map.get(node_type, node_type)
            tracked_addrs = targets_by_type.get(canonical_type)
            if not tracked_addrs:
                continue
            energy_bucket = energy_by_type.setdefault(canonical_type, {})
            power_bucket = power_by_type.setdefault(canonical_type, {})
            for raw_addr, sample_payload in payload.items():
                addr = normalize_node_addr(raw_addr)
                if not addr or addr not in tracked_addrs:
                    continue
                point = self._extract_sample_point(sample_payload)
                if point is None:
                    continue
                sample_t, counter = point
                changed_sample = self._process_energy_sample(
                    canonical_type,
                    addr,
                    sample_t,
                    counter,
                    energy_bucket,
                    power_bucket,
                    prune_power=True,
                )
                if changed_sample:
                    changed = True

        now = time_mod()
        snapshot = self._build_snapshot(
            dev_id,
            targets_by_type,
            energy_by_type,
            power_by_type,
            source="ws",
            updated_at=now,
            ws_deadline=self._ws_deadline,
        )

        if changed or snapshot != self.data:
            self.async_set_updated_data(snapshot)
