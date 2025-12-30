"""Coordinator helpers for the TermoWeb integration."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import chain
import logging
import math
import time
from time import monotonic as time_mod
from typing import Any, TypeVar

from aiohttp import ClientError
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util

from .api import BackendAuthError, BackendRateLimitError, RESTClient
from .backend.sanitize import mask_identifier
from .boost import coerce_int, resolve_boost_end_from_fields
from .const import HTR_ENERGY_UPDATE_INTERVAL, MIN_POLL_INTERVAL
from .domain.ids import NodeId as DomainNodeId, NodeType as DomainNodeType
from .domain.legacy_view import store_to_legacy_coordinator_data
from .domain.state import DomainStateStore, NodeSettingsDelta
from .domain.view import DomainStateView
from .inventory import Inventory, normalize_node_addr, normalize_node_type
from .utils import float_or_none

_LOGGER = logging.getLogger(__name__)
_CANCELLED_ERROR = asyncio.CancelledError

_DataT = TypeVar("_DataT")


class RaiseUpdateFailedCoordinator(DataUpdateCoordinator[_DataT]):
    """Coordinator that propagates ``UpdateFailed`` to manual refresh callers."""

    async def async_refresh(self) -> None:
        """Refresh data and raise ``UpdateFailed`` when polling fails."""

        await super().async_refresh()
        exc = getattr(self, "last_exception", None)
        if not self.last_update_success and isinstance(exc, UpdateFailed):
            raise exc


# TTL for pending heater setting confirmations.
_PENDING_SETTINGS_TTL = 10.0
_SETPOINT_TOLERANCE = 0.05

ENERGY_NODE_TYPES: frozenset[str] = frozenset({"htr", "acm", "pmo"})


@dataclass
class PendingSetting:
    """Track expected heater settings awaiting confirmation."""

    mode: str | None
    stemp: float | None
    expires_at: float


@dataclass(slots=True)
class InstantPowerEntry:
    """Represent a cached instant power reading."""

    watts: float
    timestamp: float
    source: str


def _device_display_name(device: Mapping[str, Any] | None, dev_id: str) -> str:
    """Return the trimmed device name or a fallback for ``dev_id``."""

    raw_name: Any | None = None
    if isinstance(device, Mapping):
        raw_name = device.get("name")

    if raw_name is not None:
        candidate = raw_name if isinstance(raw_name, str) else str(raw_name)
        trimmed = candidate.strip()
        if trimmed:
            return trimmed

    return f"Device {dev_id}"


class StateCoordinator(
    RaiseUpdateFailedCoordinator[dict[str, dict[str, Any]]]
):  # dev_id -> per-device data
    """Polls TermoWeb and exposes a per-device dict used by platforms."""

    def __init__(
        self,
        hass: HomeAssistant,
        client: RESTClient,
        base_interval: int,
        dev_id: str,
        device: dict[str, Any],
        nodes: Mapping[str, Any] | None,
        inventory: Inventory | None = None,
    ) -> None:
        """Initialize the TermoWeb device coordinator."""
        super().__init__(
            hass,
            logger=_wrap_logger(_LOGGER),
            name="termoweb",
            update_interval=timedelta(seconds=max(base_interval, MIN_POLL_INTERVAL)),
        )
        self.client = client
        self._base_interval = max(base_interval, MIN_POLL_INTERVAL)
        self._backoff = 0  # seconds
        self._dev_id = dev_id
        self._device = device or {}
        is_ducaheat = getattr(client, "_is_ducaheat", False)
        self._is_ducaheat = bool(is_ducaheat is True)
        if not isinstance(inventory, Inventory):
            msg = "EnergyStateCoordinator requires an Inventory instance"
            raise TypeError(msg)

        self._inventory: Inventory | None = None
        self._state_store: DomainStateStore | None = None
        self._pending_settings: dict[tuple[str, str], PendingSetting] = {}
        self._rtc_reference: datetime | None = None
        self._rtc_reference_monotonic: float | None = None
        self._instant_power: dict[tuple[str, str], InstantPowerEntry] = {}
        self._domain_view = DomainStateView(dev_id, None)
        self.update_nodes(nodes, inventory=inventory)

    @property
    def domain_view(self) -> DomainStateView:
        """Return the read-only domain state view."""

        return self._domain_view

    def _include_raw_settings(self) -> bool:
        """Return True when raw settings payloads should be retained."""

        try:
            checker = getattr(_LOGGER, "isEnabledFor", None)
            if callable(checker):
                return bool(checker(logging.DEBUG))
        except (AttributeError, TypeError):  # pragma: no cover - defensive
            return False
        return False

    def _filtered_settings_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Return a defensive copy of ``payload`` without raw blobs."""

        if not isinstance(payload, Mapping):
            return {}

        if self._include_raw_settings():
            return dict(payload)

        return {key: value for key, value in payload.items() if key != "raw"}

    def _instant_power_key(self, node_type: str, addr: str) -> tuple[str, str] | None:
        """Return a normalized key for instant power tracking."""

        normalized_type = normalize_node_type(
            node_type,
            use_default_when_falsey=True,
        )
        normalized_addr = normalize_node_addr(
            addr,
            use_default_when_falsey=True,
        )
        if not normalized_type or not normalized_addr:
            return None
        return normalized_type, normalized_addr

    def _instant_power_snapshot(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Return the current instant power cache grouped by node type."""

        snapshot: dict[str, dict[str, dict[str, Any]]] = {}
        for (node_type, addr), entry in self._instant_power.items():
            bucket = snapshot.setdefault(node_type, {})
            bucket[addr] = {
                "watts": entry.watts,
                "timestamp": entry.timestamp,
                "source": entry.source,
            }
        return snapshot

    def _sync_instant_power_data(self) -> None:
        """Publish the latest instant power snapshot to listeners."""

        if self.data is not None:
            self.async_set_updated_data(self.data)

    def _record_instant_power(
        self,
        node_type: str,
        addr: str,
        watts: float,
        *,
        timestamp: float | None = None,
        source: str,
    ) -> bool:
        """Store an instant power reading and return ``True`` when changed."""

        key = self._instant_power_key(node_type, addr)
        if key is None:
            return False

        if not isinstance(watts, (int, float)):
            return False

        candidate = float(watts)
        if math.isnan(candidate) or candidate < 0:
            return False

        if timestamp is None:
            timestamp = time.time()
        current_ts = float(timestamp)

        existing = self._instant_power.get(key)
        if existing is not None:
            if existing.timestamp >= current_ts and existing.source == source:
                return False
            if (
                existing.source == "ws"
                and source == "rest"
                and existing.timestamp >= current_ts
            ):
                return False
            if (
                existing.timestamp == current_ts
                and existing.watts == candidate
                and existing.source == source
            ):
                return False

        self._instant_power[key] = InstantPowerEntry(
            watts=candidate,
            timestamp=current_ts,
            source=source,
        )
        self._sync_instant_power_data()
        return True

    def handle_instant_power_update(
        self,
        dev_id: str,
        node_type: str,
        addr: str,
        watts: float,
        *,
        timestamp: float | None = None,
    ) -> None:
        """Process a websocket instant power update."""

        if dev_id != self._dev_id:
            return

        if not isinstance(watts, (int, float)):
            return

        candidate = float(watts)
        if math.isnan(candidate) or candidate < 0:
            return

        self._record_instant_power(
            node_type,
            addr,
            candidate,
            timestamp=timestamp,
            source="ws",
        )

    def instant_power_entry(
        self, node_type: str, addr: str
    ) -> InstantPowerEntry | None:
        """Return the cached instant power entry for ``(node_type, addr)``."""

        key = self._instant_power_key(node_type, addr)
        if key is None:
            return None
        return self._instant_power.get(key)

    def instant_power_overview(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Return a diagnostics-friendly snapshot of instant power values."""

        return self._instant_power_snapshot()

    def _should_skip_rest_power(self, node_type: str, addr: str) -> bool:
        """Return ``True`` when websocket data is fresh enough to skip REST power."""

        entry = self.instant_power_entry(node_type, addr)
        if entry is None or entry.source != "ws":
            return False

        now = time.time()
        interval = self.update_interval.total_seconds()
        return now - entry.timestamp < interval

    async def _async_fetch_settings_to_store(
        self,
        dev_id: str,
        addr_map: Mapping[str, Iterable[str]],
        reverse: Mapping[str, set[str]],
        store: DomainStateStore,
        rtc_now: datetime | None,
    ) -> datetime | None:
        """Fetch settings for every address and store them in the domain cache."""

        current_rtc = rtc_now
        for node_type, addrs_for_type in addr_map.items():
            for addr in addrs_for_type:
                addr_types = reverse.get(addr)
                resolved_type = (
                    node_type
                    if node_type in (addr_types or {node_type})
                    else next(iter(addr_types))
                    if addr_types
                    else node_type
                )
                payload = await self.client.get_node_settings(
                    dev_id, (resolved_type, addr)
                )
                if not isinstance(payload, dict):
                    continue
                if self._should_defer_pending_setting(resolved_type, addr, payload):
                    _LOGGER.debug(
                        "Deferring poll merge for pending settings type=%s addr=%s",
                        resolved_type,
                        addr,
                    )
                    continue
                if resolved_type == "acm" and isinstance(payload, MutableMapping):
                    now_value: datetime | None = None
                    if self._requires_boost_resolution(payload) and current_rtc is None:
                        current_rtc = await self._async_fetch_rtc_datetime()
                    now_value = current_rtc or self._device_now_estimate()
                    self._apply_accumulator_boost_metadata(payload, now=now_value)
                store.apply_full_snapshot(
                    resolved_type,
                    addr,
                    self._filtered_settings_payload(payload),
                )
        return current_rtc

    @staticmethod
    def _normalize_mode_value(value: Any) -> str | None:
        """Return a canonical string for backend HVAC modes."""

        if value is None:
            return None
        if isinstance(value, str):
            candidate = value.strip().lower()
        else:
            candidate = str(value).strip().lower()
        return candidate or None

    def _pending_key(self, node_type: str, addr: str) -> tuple[str, str] | None:
        """Return the normalised pending settings key for a node."""

        normalized_type = normalize_node_type(
            node_type,
            use_default_when_falsey=True,
        )
        normalized_addr = normalize_node_addr(
            addr,
            use_default_when_falsey=True,
        )
        if not normalized_type or not normalized_addr:
            return None
        return normalized_type, normalized_addr

    def _prune_pending_settings(self) -> None:
        """Drop expired pending settings entries."""

        now = time_mod()
        stale = [
            key
            for key, entry in self._pending_settings.items()
            if entry.expires_at <= now
        ]
        for key in stale:
            self._pending_settings.pop(key, None)

    def register_pending_setting(
        self,
        node_type: str,
        addr: str,
        *,
        mode: str | None,
        stemp: float | None,
        ttl: float = _PENDING_SETTINGS_TTL,
    ) -> None:
        """Record expected heater settings awaiting confirmation."""

        key = self._pending_key(node_type, addr)
        if key is None:
            return

        try:
            ttl_value = float(ttl)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            ttl_value = _PENDING_SETTINGS_TTL
        expires_at = time_mod() + max(ttl_value, 0.0)
        normalized_mode = self._normalize_mode_value(mode)
        normalized_stemp = float_or_none(stemp)
        self._pending_settings[key] = PendingSetting(
            mode=normalized_mode,
            stemp=normalized_stemp,
            expires_at=expires_at,
        )
        _LOGGER.debug(
            "Registered pending settings type=%s addr=%s mode=%s stemp=%s ttl=%.1f",
            key[0],
            key[1],
            normalized_mode,
            normalized_stemp,
            ttl,
        )

    def resolve_boost_end(
        self,
        boost_end_day: Any,
        boost_end_min: Any,
        *,
        now: datetime | None = None,
    ) -> tuple[datetime | None, int | None]:
        """Return boost end metadata derived from cached day/minute fields."""

        reference = now or self._device_now_estimate()
        return resolve_boost_end_from_fields(
            boost_end_day,
            boost_end_min,
            now=reference,
        )

    @staticmethod
    def _rtc_payload_to_datetime(payload: Mapping[str, Any] | None) -> datetime | None:
        """Return a timezone-aware datetime extracted from RTC payload."""

        if not isinstance(payload, Mapping):
            return None

        year = coerce_int(payload.get("y"))
        month = coerce_int(payload.get("n"))
        day = coerce_int(payload.get("d"))
        if year is None or month is None or day is None:
            return None

        hour = coerce_int(payload.get("h"))
        minute = coerce_int(payload.get("m"))
        second = coerce_int(payload.get("s"))

        tzinfo = dt_util.now().tzinfo or getattr(dt_util, "UTC", UTC)
        try:
            return datetime(
                year,
                month,
                day,
                hour or 0,
                minute or 0,
                second or 0,
                tzinfo=tzinfo,
            )
        except ValueError:
            return None

    def _device_now_estimate(self) -> datetime | None:
        """Return the latest hub time reference adjusted by monotonic delta."""

        if self._rtc_reference is None or self._rtc_reference_monotonic is None:
            return None
        delta_seconds = time_mod() - self._rtc_reference_monotonic
        try:
            return self._rtc_reference + timedelta(seconds=delta_seconds)
        except OverflowError:  # pragma: no cover - defensive
            return self._rtc_reference

    async def _async_fetch_rtc_datetime(self) -> datetime | None:
        """Fetch the hub RTC time and update the cached reference."""

        try:
            payload = await self.client.get_rtc_time(self._dev_id)
        except TimeoutError as err:  # pragma: no cover - defensive logging
            _LOGGER.debug(
                "RTC fetch timed out for dev %s: %s",
                self._dev_id,
                err,
                exc_info=err,
            )
            return None
        except (ClientError, BackendRateLimitError, BackendAuthError) as err:
            _LOGGER.debug(
                "RTC fetch failed for dev %s: %s",
                self._dev_id,
                err,
                exc_info=err,
            )
            return None

        rtc_now = self._rtc_payload_to_datetime(payload)
        if rtc_now is not None:
            self._rtc_reference = rtc_now
            self._rtc_reference_monotonic = time_mod()
        return rtc_now

    @staticmethod
    def _requires_boost_resolution(payload: Mapping[str, Any] | None) -> bool:
        """Return True when ``payload`` exposes boost day/min metadata."""

        if not isinstance(payload, Mapping):
            return False
        day = coerce_int(payload.get("boost_end_day"))
        minute = coerce_int(payload.get("boost_end_min"))
        return day is not None or minute is not None

    def _apply_accumulator_boost_metadata(
        self,
        payload: MutableMapping[str, Any],
        *,
        now: datetime | None,
    ) -> None:
        """Store derived boost metadata on ``payload``."""

        derived_dt, minutes = self.resolve_boost_end(
            payload.get("boost_end_day"),
            payload.get("boost_end_min"),
            now=now,
        )

        if derived_dt is not None:
            payload["boost_end_datetime"] = derived_dt
        else:
            payload.pop("boost_end_datetime", None)

        if minutes is not None:
            payload["boost_minutes_delta"] = minutes
        else:
            payload.pop("boost_minutes_delta", None)

    def _apply_boost_metadata_for_settings(
        self,
        bucket: Mapping[str, Any] | None,
        *,
        now: datetime | None,
    ) -> None:
        """Apply boost metadata derivation to every settings payload."""

        if not isinstance(bucket, Mapping):
            return
        for payload in bucket.values():
            if isinstance(payload, MutableMapping):
                self._apply_accumulator_boost_metadata(payload, now=now)

    def _should_defer_pending_setting(
        self,
        node_type: str,
        addr: str,
        payload: Mapping[str, Any] | None,
    ) -> bool:
        """Return True when a pending write should defer payload merging."""

        key = self._pending_key(node_type, addr)
        if key is None:
            return False

        entry = self._pending_settings.get(key)
        if entry is None:
            return False

        now = time_mod()
        if entry.expires_at <= now:
            _LOGGER.debug("Pending settings expired type=%s addr=%s", *key)
            self._pending_settings.pop(key, None)
            return False

        mode_expected = entry.mode
        stemp_expected = entry.stemp
        if payload is None:
            _LOGGER.debug(
                "Deferring merge due to pending settings type=%s addr=%s payload=None",
                key[0],
                key[1],
            )
            return True

        mode_payload = self._normalize_mode_value(payload.get("mode"))
        stemp_payload = float_or_none(payload.get("stemp"))

        mode_matches = mode_expected is None or mode_expected == mode_payload
        stemp_matches = True
        if stemp_expected is not None:
            if stemp_payload is None:
                stemp_matches = False
            else:
                stemp_matches = (
                    abs(stemp_payload - stemp_expected) <= _SETPOINT_TOLERANCE
                )

        if mode_matches and stemp_matches:
            _LOGGER.debug("Pending settings satisfied type=%s addr=%s", *key)
            self._pending_settings.pop(key, None)
            return False

        _LOGGER.debug(
            "Deferring merge due to pending settings type=%s addr=%s "
            "expected_mode=%s expected_stemp=%s payload_mode=%s payload_stemp=%s",
            key[0],
            key[1],
            mode_expected,
            stemp_expected,
            mode_payload,
            stemp_payload,
        )
        return True

    def update_nodes(
        self,
        _nodes: Mapping[str, Any] | None = None,
        *,
        inventory: Inventory | None = None,
    ) -> None:
        """Update cached inventory metadata."""

        if isinstance(inventory, Inventory):
            self._inventory = inventory
            self._ensure_state_store(inventory)
        else:
            self._inventory = None
            self._state_store = None
            self._domain_view.update_store(None)

    def _ensure_inventory(self) -> Inventory | None:
        """Ensure cached inventory metadata is available."""

        return self._inventory

    def _node_ids_from_inventory(self, inventory: Inventory) -> list[DomainNodeId]:
        """Return domain node identifiers derived from ``inventory``."""

        node_ids: list[DomainNodeId] = []
        raw_nodes: Iterable[Any] | None = inventory.nodes
        nodes_iterable: Iterable[Any] = raw_nodes or ()
        if not nodes_iterable:
            payload = (
                inventory.payload if isinstance(inventory.payload, Mapping) else {}
            )
            nodes_section = payload.get("nodes") if isinstance(payload, Mapping) else []
            if isinstance(nodes_section, Iterable) and not isinstance(
                nodes_section, (str, bytes)
            ):
                nodes_iterable = nodes_section
            else:
                nodes_iterable = ()

        for node in nodes_iterable:
            node_type_value: Any
            addr_value: Any
            if isinstance(node, Mapping):
                node_type_value = node.get("type")
                addr_value = node.get("addr")
            else:
                node_type_value = getattr(node, "type", None)
                addr_value = getattr(node, "addr", None)

            try:
                node_type = DomainNodeType(str(node_type_value))
            except ValueError:
                try:
                    node_type = DomainNodeType(str(node_type_value).lower())
                except ValueError:
                    continue
            try:
                node_ids.append(DomainNodeId(node_type, addr_value))
            except ValueError:
                continue
        return node_ids

    def _ensure_state_store(self, inventory: Inventory) -> DomainStateStore | None:
        """Ensure a domain state store exists when applicable."""

        node_ids = self._node_ids_from_inventory(inventory)
        if self._state_store is None:
            self._state_store = DomainStateStore(node_ids)
        else:
            self._state_store.reset_nodes(node_ids)
        self._domain_view.update_store(self._state_store)
        self._seed_state_store_from_data()
        return self._state_store

    def _seed_state_store_from_data(self) -> None:
        """Populate the domain store from any existing coordinator data."""

        store = self._state_store
        if store is None:
            return

        data = self.data if isinstance(self.data, Mapping) else None
        if not isinstance(data, Mapping):
            return

        dev_state = data.get(self._dev_id)
        _ = data.get(self._dev_id)
        settings = dev_state.get("settings") if isinstance(dev_state, Mapping) else None
        if not isinstance(settings, Mapping):
            return

        for node_type, bucket in settings.items():
            if not isinstance(bucket, Mapping):
                continue
            for addr, payload in bucket.items():
                if not isinstance(addr, str) or not isinstance(payload, Mapping):
                    continue
                store.apply_full_snapshot(
                    node_type,
                    addr,
                    self._filtered_settings_payload(payload),
                )

    def handle_ws_deltas(
        self,
        dev_id: str,
        deltas: Iterable[NodeSettingsDelta],
        *,
        replace: bool = False,
    ) -> None:
        """Merge websocket-delivered deltas into the domain state store."""

        if dev_id != self._dev_id:
            return

        inventory = self._inventory
        if not isinstance(inventory, Inventory):
            return

        store = self._state_store or self._ensure_state_store(inventory)
        if store is None:
            return

        self._seed_state_store_from_data()
        applied = False
        for delta in deltas:
            if not isinstance(delta, NodeSettingsDelta):
                continue
            if replace:
                store.apply_full_snapshot(
                    delta.node_id.node_type,
                    delta.node_id.addr,
                    delta.payload,
                )
            else:
                store.apply_delta(delta)
            applied = True

        if not applied:
            return

        dev_name = _device_display_name(self._device, self._dev_id)
        device_record = store_to_legacy_coordinator_data(
            self._dev_id,
            store,
            inventory,
            device_name=dev_name,
            device_raw=self._device,
        )
        new_data = dict(self.data or {})
        new_data.update(device_record)
        self.async_set_updated_data(new_data)

    def apply_entity_patch(
        self, node_type: str, addr: str, mutator: Callable[[dict[str, Any]], None]
    ) -> bool:
        """Apply an optimistic patch through the domain store when available."""

        inventory = self._inventory
        store = self._state_store
        if store is None or not isinstance(inventory, Inventory):
            return False

        try:
            data = self.data if isinstance(self.data, Mapping) else {}
            if isinstance(data, Mapping):
                _ = data.get(self._dev_id)
            current_state = store.get_state(node_type, addr)
            base = current_state.to_legacy() if current_state else {}
            payload = dict(base)
            mutator(payload)
            store.apply_patch(node_type, addr, payload)
        except _CANCELLED_ERROR:
            raise
        except Exception as err:  # pragma: no cover - defensive  # noqa: BLE001
            _LOGGER.debug(
                "Failed to apply optimistic patch type=%s addr=%s: %s",
                node_type,
                addr,
                err,
            )
            return False

        dev_name = _device_display_name(self._device, self._dev_id)
        device_record = store_to_legacy_coordinator_data(
            self._dev_id,
            store,
            inventory,
            device_name=dev_name,
            device_raw=self._device,
        )
        new_data = dict(self.data or {})
        new_data.update(device_record)
        self.async_set_updated_data(new_data)
        return True

    async def async_refresh_heater(self, node: str | tuple[str, str]) -> None:
        """Refresh settings for a specific node and push the update to listeners."""

        dev_id = self._dev_id
        success = False
        if isinstance(node, tuple) and len(node) == 2:
            raw_type, raw_addr = node
            node_type = normalize_node_type(
                raw_type,
                use_default_when_falsey=True,
            )
            addr = normalize_node_addr(
                raw_addr,
                use_default_when_falsey=True,
            )
        else:
            node_type = ""
            addr = normalize_node_addr(node, use_default_when_falsey=True)

        _LOGGER.info(
            "Refreshing heater settings node_type=%s addr=%s",
            node_type or "<auto>",
            addr,
        )
        resolved_type = node_type or "htr"

        try:
            self._prune_pending_settings()
            if not addr:
                _LOGGER.error(
                    "Cannot refresh heater settings without an address",
                )
                return

            inventory = self._ensure_inventory()

            if inventory is None:
                _LOGGER.error(
                    "Cannot refresh heater settings without inventory metadata",
                )
                return

            forward_map, reverse = inventory.heater_address_map
            addr_types = reverse.get(addr)
            resolved_type = node_type or (
                next(iter(addr_types)) if addr_types else "htr"
            )

            payload = await self.client.get_node_settings(dev_id, (resolved_type, addr))

            if not isinstance(payload, dict):
                _LOGGER.debug(
                    "Ignoring unexpected heater settings payload for "
                    "node_type=%s addr=%s: %s",
                    resolved_type,
                    addr,
                    payload,
                )
                return

            if resolved_type == "acm":
                now_value: datetime | None = None
                if self._requires_boost_resolution(payload):
                    now_value = await self._async_fetch_rtc_datetime()
                if now_value is None:
                    now_value = self._device_now_estimate()
                if isinstance(payload, MutableMapping):
                    self._apply_accumulator_boost_metadata(payload, now=now_value)

            if self._should_defer_pending_setting(resolved_type, addr, payload):
                _LOGGER.debug(
                    "Skipping heater refresh merge for pending settings "
                    "type=%s addr=%s",
                    resolved_type,
                    addr,
                )
                success = True
                return

            store = self._state_store or self._ensure_state_store(inventory)
            if store is None:
                _LOGGER.error(
                    "Cannot refresh heater settings without a domain state store"
                )
                return

            self._seed_state_store_from_data()
            store.apply_full_snapshot(
                resolved_type,
                addr,
                self._filtered_settings_payload(payload),
            )
            dev_name = _device_display_name(self._device, dev_id)
            device_record = store_to_legacy_coordinator_data(
                dev_id,
                store,
                inventory,
                device_name=dev_name,
                device_raw=self._device,
            )
            new_data = dict(self.data or {})
            new_data.update(device_record)
            self.async_set_updated_data(new_data)
            success = True

        except TimeoutError as err:
            _LOGGER.error(
                "Timeout refreshing heater settings for node_type=%s addr=%s",
                node_type or resolved_type,
                addr,
                exc_info=err,
            )
        except (ClientError, BackendRateLimitError, BackendAuthError) as err:
            _LOGGER.error(
                "Failed to refresh heater settings for node_type=%s addr=%s: %s",
                node_type or resolved_type,
                addr,
                err,
                exc_info=err,
            )
        finally:
            _LOGGER.info(
                "Finished heater settings refresh for node_type=%s "
                "addr=%s (success=%s)",
                node_type or resolved_type,
                addr,
                success,
            )

    async def _async_update_data(self) -> dict[str, dict[str, Any]]:
        """Fetch the latest settings for every known node on each poll."""
        dev_id = self._dev_id
        inventory = self._ensure_inventory()

        if inventory is None:
            _LOGGER.debug("Skipping poll because inventory metadata is unavailable")
            return {}

        store = self._state_store or self._ensure_state_store(inventory)

        addr_map, reverse = inventory.heater_address_map
        addrs = [addr for addrs in addr_map.values() for addr in addrs]
        rtc_now: datetime | None = None
        try:
            self._prune_pending_settings()
            if store is None:
                return {}

            self._seed_state_store_from_data()
            if addrs:
                rtc_now = await self._async_fetch_settings_to_store(
                    dev_id,
                    addr_map,
                    reverse,
                    store,
                    rtc_now,
                )
            dev_name = _device_display_name(self._device, dev_id)
            result = store_to_legacy_coordinator_data(
                dev_id,
                store,
                inventory,
                device_name=dev_name,
                device_raw=self._device,
                include_nodes_by_type=False,
            )

        except TimeoutError as err:
            raise UpdateFailed("API timeout") from err
        except BackendRateLimitError as err:
            self._backoff = min(
                max(self._base_interval, (self._backoff or self._base_interval) * 2),
                3600,
            )
            self.update_interval = timedelta(seconds=self._backoff)
            raise UpdateFailed(
                f"Rate limited; backing off to {self._backoff}s"
            ) from err
        except (ClientError, BackendAuthError) as err:
            raise UpdateFailed(f"API error: {err}") from err
        else:
            if self._backoff:
                self._backoff = 0
                self.update_interval = timedelta(seconds=self._base_interval)

            return result


class EnergyStateCoordinator(
    RaiseUpdateFailedCoordinator[dict[str, dict[str, Any]]]
):  # dev_id -> per-device data
    """Polls heater energy counters and exposes energy and power per heater."""

    def __init__(
        self,
        hass: HomeAssistant,
        client: RESTClient,
        dev_id: str,
        inventory: Inventory | None,
    ) -> None:
        """Initialize the heater energy coordinator."""
        super().__init__(
            hass,
            logger=_wrap_logger(_LOGGER),
            name="termoweb-htr-energy",
            update_interval=HTR_ENERGY_UPDATE_INTERVAL,
        )
        self.client = client
        self._dev_id = dev_id
        self._inventory: Inventory | None = None
        self._last: dict[tuple[str, str], tuple[float, float]] = {}
        self._base_interval = HTR_ENERGY_UPDATE_INTERVAL
        self._base_interval_seconds = HTR_ENERGY_UPDATE_INTERVAL.total_seconds()
        self._ws_lease: float = 0.0
        self._ws_deadline: float | None = None
        self._ws_margin_default = 60.0
        self._counter_scales: dict[str, float] = {
            "htr": 1000.0,
            "acm": 1000.0,
            "pmo": 3_600_000.0,
        }
        self.update_addresses(inventory)

    def _resolve_inventory(self, candidate: Inventory | None = None) -> Inventory:
        """Return the active inventory, preferring ``candidate`` when provided."""

        if candidate is not None and not isinstance(candidate, Inventory):
            msg = "Energy inventory is unavailable"
            raise TypeError(msg)

        inventory = candidate if isinstance(candidate, Inventory) else self._inventory
        if not isinstance(inventory, Inventory):
            msg = "Energy inventory is unavailable"
            raise TypeError(msg)
        return inventory

    def _iter_energy_targets(self, inventory: Inventory) -> Iterator[tuple[str, str]]:
        """Yield normalised energy node targets from ``inventory``."""

        for node_type, addr in chain(
            inventory.heater_sample_targets,
            inventory.power_monitor_sample_targets,
        ):
            normalized_type = normalize_node_type(
                node_type,
                use_default_when_falsey=True,
            )
            normalized_addr = normalize_node_addr(
                addr,
                use_default_when_falsey=True,
            )
            if (
                not normalized_type
                or not normalized_addr
                or normalized_type not in ENERGY_NODE_TYPES
            ):
                continue
            if not inventory.has_node(normalized_type, normalized_addr):
                continue
            yield normalized_type, normalized_addr

    def _targets_by_type(self, inventory: Inventory) -> dict[str, list[str]]:
        """Return energy node addresses grouped by canonical type."""

        buckets: dict[str, list[str]] = {}
        for node_type, addr in self._iter_energy_targets(inventory):
            bucket = buckets.setdefault(node_type, [])
            if addr not in bucket:
                bucket.append(addr)
        return buckets

    def _prefill_energy_buckets(
        self,
        dev_id: str,
        targets_by_type: Mapping[str, list[str]],
        energy_by_type: dict[str, dict[str, float]],
        power_by_type: dict[str, dict[str, float]],
    ) -> None:
        """Seed energy and power buckets from cached coordinator state."""

        cached_dev: Mapping[str, Any] | None = None
        if isinstance(self.data, Mapping):
            cached = self.data.get(dev_id)
            if isinstance(cached, Mapping):
                cached_dev = cached

        if cached_dev:
            for node_type, addrs_for_type in targets_by_type.items():
                prev_node = cached_dev.get(node_type)
                if not isinstance(prev_node, Mapping):
                    continue

                prev_energy = prev_node.get("energy")
                if isinstance(prev_energy, Mapping):
                    energy_bucket = energy_by_type.setdefault(node_type, {})
                    for addr in addrs_for_type:
                        cached_energy = float_or_none(prev_energy.get(addr))
                        if cached_energy is not None:
                            energy_bucket.setdefault(addr, cached_energy)

                prev_power = prev_node.get("power")
                if isinstance(prev_power, Mapping):
                    power_bucket = power_by_type.setdefault(node_type, {})
                    for addr in addrs_for_type:
                        cached_power = float_or_none(prev_power.get(addr))
                        if cached_power is not None:
                            power_bucket.setdefault(addr, cached_power)

        for (node_type, addr), (_, kwh) in self._last.items():
            addrs_for_type = targets_by_type.get(node_type)
            if not addrs_for_type or addr not in addrs_for_type:
                continue
            energy_bucket = energy_by_type.setdefault(node_type, {})
            energy_bucket.setdefault(addr, kwh)

    async def _poll_recent_samples(
        self,
        dev_id: str,
        targets_by_type: Mapping[str, list[str]],
        energy_by_type: dict[str, dict[str, float]],
        power_by_type: dict[str, dict[str, float]],
    ) -> None:
        """Fetch recent energy samples for every tracked node."""

        for node_type, addrs_for_type in targets_by_type.items():
            for addr in addrs_for_type:
                now = time.time()
                start = now - 3600  # fetch recent samples
                try:
                    samples = await self.client.get_node_samples(
                        dev_id, (node_type, addr), start, now
                    )
                except (ClientError, BackendRateLimitError, BackendAuthError):
                    samples = []

                if not samples:
                    _LOGGER.debug(
                        "No energy samples for node_type=%s addr=%s",
                        node_type,
                        addr,
                    )
                    continue

                last = samples[-1]
                counter = float_or_none(last.get("counter"))
                if counter is None:
                    counter = float_or_none(last.get("counter_max"))
                if counter is None:
                    counter = float_or_none(last.get("counter_min"))
                t = float_or_none(last.get("t"))
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

    def update_addresses(
        self,
        inventory: Inventory | None,
    ) -> None:
        """Replace the tracked nodes using immutable inventory metadata."""

        if not isinstance(inventory, Inventory):
            msg = "Energy inventory is unavailable"
            raise TypeError(msg)

        valid_keys = set(self._iter_energy_targets(inventory))
        self._inventory = inventory
        self._last = {
            key: value for key, value in self._last.items() if key in valid_keys
        }

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

        scale = float(self._counter_scales.get(node_type, 1000.0) or 1000.0)
        if scale <= 0:
            scale = 1000.0
        kwh = counter / scale
        energy_bucket[addr] = kwh
        key = (node_type, addr)
        prev = self._last.get(key)
        if prev:
            prev_t, prev_kwh = prev
            if kwh < prev_kwh or when <= prev_t:
                self._last[key] = (when, kwh)
                if prune_power:
                    power_bucket.pop(addr, None)
                    return True
                return False

            dt_hours = (when - prev_t) / 3600.0
            if dt_hours > 0:
                delta_kwh = kwh - prev_kwh
                power_bucket[addr] = delta_kwh / dt_hours * 1000.0
            elif prune_power:
                power_bucket.pop(addr, None)

            self._last[key] = (when, kwh)
            return prune_power

        self._last[key] = (when, kwh)
        if prune_power:
            power_bucket.pop(addr, None)
            return True
        return False

    async def _async_update_data(self) -> dict[str, dict[str, Any]]:
        """Fetch recent heater energy samples and derive totals and power."""
        if self._should_skip_poll():
            existing = self.data or {}
            if not isinstance(existing, dict):
                return {}
            _LOGGER.debug("Energy poll skipped (fresh websocket samples)")
            return dict(existing)
        dev_id = self._dev_id
        try:
            inventory = self._resolve_inventory()
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

            dev_data: dict[str, Any] = {"dev_id": dev_id}

            for node_type, addrs_for_type in targets_by_type.items():
                bucket = {
                    "energy": dict(energy_by_type.get(node_type, {})),
                    "power": dict(power_by_type.get(node_type, {})),
                    "addrs": list(addrs_for_type),
                }
                dev_data[node_type] = bucket

            heater_data = dev_data.get("htr")
            if not isinstance(heater_data, dict):
                heater_data = {
                    "energy": dict(energy_by_type.get("htr", {})),
                    "power": dict(power_by_type.get("htr", {})),
                    "addrs": list(targets_by_type.get("htr", ())),
                }
                dev_data["htr"] = heater_data

            alias_map = inventory.sample_alias_map(
                include_types=ENERGY_NODE_TYPES,
                restrict_to=ENERGY_NODE_TYPES,
            )
            for alias, canonical in alias_map.items():
                canonical_bucket = dev_data.get(canonical)
                if not isinstance(canonical_bucket, dict):
                    canonical_bucket = {
                        "energy": {},
                        "power": {},
                        "addrs": list(targets_by_type.get(canonical, ())),
                    }
                    dev_data[canonical] = canonical_bucket
                dev_data[alias] = canonical_bucket

            result: dict[str, dict[str, Any]] = {dev_id: dev_data}

        except TimeoutError as err:
            raise UpdateFailed("API timeout") from err
        except (ClientError, BackendRateLimitError, BackendAuthError) as err:
            raise UpdateFailed(f"API error: {err}") from err
        else:
            self.update_interval = self._base_interval
            self._ws_deadline = None
            return result

    def _should_skip_poll(self) -> bool:
        """Return True when websocket pushes keep energy data fresh."""

        if not isinstance(self.data, dict):
            return False
        if self._ws_deadline is None:
            return False
        if time_mod() >= self._ws_deadline:
            return False
        return True

    def _ws_margin_seconds(self) -> float:
        """Return the buffer to wait after the websocket lease expires."""

        if self._ws_lease <= 0:
            return self._ws_margin_default
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
            t = float_or_none(payload.get("t"))
            if t is None:
                return None
            counter_raw = payload.get("counter")
            counter = float_or_none(counter_raw)
            counter_min = float_or_none(payload.get("counter_min"))
            counter_max = float_or_none(payload.get("counter_max"))
            if isinstance(counter_raw, Mapping):
                counter = float_or_none(counter_raw.get("value")) or float_or_none(
                    counter_raw.get("counter")
                )
                counter_min = counter_min or float_or_none(counter_raw.get("min"))
                counter_max = counter_max or float_or_none(counter_raw.get("max"))
            if counter is None:
                counter = counter_max or counter_min
            if counter is None:
                counter = float_or_none(payload.get("value"))
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
        updates: Mapping[str, Mapping[str, Any]],
        *,
        lease_seconds: float | None = None,
    ) -> None:
        """Update cached heater metrics from websocket ``samples`` payloads."""

        if dev_id != self._dev_id or not isinstance(self.data, dict):
            return
        dev_data = self.data.get(dev_id)
        if not isinstance(dev_data, dict):
            return

        if lease_seconds is not None:
            self._ws_lease = float(lease_seconds) if lease_seconds > 0 else 0.0

        now = time_mod()
        if self._ws_lease > 0:
            margin = self._ws_margin_seconds()
            wait_seconds = max(self._ws_lease + margin, 300.0)
            interval_seconds = min(self._base_interval_seconds, wait_seconds)
            new_interval = timedelta(seconds=interval_seconds)
            if self.update_interval != new_interval:
                self.update_interval = new_interval
            self._ws_deadline = now + self._ws_lease + margin
        else:
            self._ws_deadline = None

        changed = False

        try:
            inventory = self._resolve_inventory()
        except TypeError:
            return

        targets_by_type = self._targets_by_type(inventory)

        alias_map = inventory.sample_alias_map(
            include_types=ENERGY_NODE_TYPES,
            restrict_to=ENERGY_NODE_TYPES,
        )

        for raw_type, payload in updates.items():
            node_type = normalize_node_type(raw_type)
            if not node_type:
                continue
            canonical_type = alias_map.get(node_type, node_type)
            tracked_addrs = targets_by_type.get(canonical_type)
            if not tracked_addrs:
                continue
            bucket = dev_data.get(canonical_type)
            if not isinstance(bucket, dict):
                bucket = {
                    "energy": {},
                    "power": {},
                    "addrs": list(tracked_addrs),
                }
                dev_data[canonical_type] = bucket
            if node_type != canonical_type:
                dev_data[node_type] = bucket
            energy_bucket = bucket.setdefault("energy", {})
            power_bucket = bucket.setdefault("power", {})
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

        if changed:
            dev_data.setdefault("dev_id", dev_id)

    async def merge_samples_for_window(
        self,
        dev_id: str,
        samples: Mapping[tuple[str, str], Iterable[Mapping[str, Any]]],
    ) -> None:
        """Merge normalised hourly samples into the cached energy state."""

        if dev_id != self._dev_id or not isinstance(samples, Mapping):
            return

        data = self.data
        if not isinstance(data, dict):
            return

        dev_data = data.get(dev_id)
        if not isinstance(dev_data, dict):
            dev_data = {"dev_id": dev_id}
            data[dev_id] = dev_data

        try:
            inventory = self._resolve_inventory()
        except TypeError:
            return

        targets_by_type = self._targets_by_type(inventory)

        alias_map = inventory.sample_alias_map(
            include_types=ENERGY_NODE_TYPES,
            restrict_to=ENERGY_NODE_TYPES,
        )
        merge_counts: dict[tuple[str, str], int] = {}

        for descriptor, records in samples.items():
            if not isinstance(descriptor, tuple) or len(descriptor) != 2:
                continue

            raw_type, raw_addr = descriptor
            node_type = normalize_node_type(
                raw_type,
                use_default_when_falsey=True,
            )
            addr = normalize_node_addr(
                raw_addr,
                use_default_when_falsey=True,
            )
            if not node_type or not addr:
                continue

            canonical_type = alias_map.get(node_type, node_type)
            tracked_addrs = targets_by_type.get(canonical_type)
            if not tracked_addrs or addr not in tracked_addrs:
                continue

            bucket = dev_data.get(canonical_type)
            if not isinstance(bucket, dict):
                bucket = {
                    "energy": {},
                    "power": {},
                    "addrs": list(tracked_addrs),
                }
                dev_data[canonical_type] = bucket
            else:
                bucket.setdefault("energy", {})
                bucket.setdefault("power", {})
                bucket.setdefault("addrs", list(tracked_addrs))

            energy_bucket = bucket["energy"]
            power_bucket = bucket["power"]
            scale = float(self._counter_scales.get(canonical_type, 1000.0) or 1000.0)
            factor = scale / 1000.0 if scale else 1.0

            prepared: list[tuple[float, float]] = []
            for record in records:
                if not isinstance(record, Mapping):
                    continue
                ts_value = record.get("ts")
                if isinstance(ts_value, datetime):
                    when = dt_util.as_utc(ts_value).timestamp()
                else:
                    when = float_or_none(record.get("timestamp"))
                    if when is None:
                        continue
                energy_wh = float_or_none(record.get("energy_wh"))
                if energy_wh is None:
                    continue
                prepared.append((when, energy_wh * factor))

            if not prepared:
                continue

            prepared.sort(key=lambda item: item[0])
            merge_counts[(canonical_type, addr)] = len(prepared)

            for when, counter in prepared:
                self._process_energy_sample(
                    canonical_type,
                    addr,
                    when,
                    counter,
                    energy_bucket,
                    power_bucket,
                    prune_power=True,
                )

        for canonical_type, addrs in targets_by_type.items():
            bucket = dev_data.get(canonical_type)
            if isinstance(bucket, dict):
                bucket.setdefault("addrs", list(addrs))

        for alias, canonical in alias_map.items():
            if alias == canonical:
                continue
            canonical_bucket = dev_data.get(canonical)
            if canonical_bucket is not None:
                dev_data[alias] = canonical_bucket

        dev_data.setdefault("dev_id", dev_id)

        if merge_counts:
            summary = ", ".join(
                f"{node_type}:{addr}={count}"
                for (node_type, addr), count in sorted(merge_counts.items())
            )
            _LOGGER.debug(
                "Hourly samples merge complete for %s: %s",
                mask_identifier(dev_id),
                summary,
            )


def _wrap_logger(logger: Any) -> Any:
    """Return a logger proxy that exposes ``isEnabledFor`` when missing."""

    if hasattr(logger, "isEnabledFor"):
        return logger

    class _LoggerProxy:
        """Proxy that adds ``isEnabledFor`` support for stubbed loggers."""

        def __init__(self, inner: Any) -> None:
            self._inner = inner

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

        def isEnabledFor(self, _level: int) -> bool:
            return False

    return _LoggerProxy(logger)
