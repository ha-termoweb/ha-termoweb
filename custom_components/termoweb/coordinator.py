"""Coordinator helpers for the TermoWeb integration."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import logging
import time
from time import monotonic as time_mod
from typing import Any, TypeVar

from aiohttp import ClientError
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util

from .api import BackendAuthError, BackendRateLimitError, RESTClient
from .boost import coerce_int, resolve_boost_end_from_fields
from .const import HTR_ENERGY_UPDATE_INTERVAL, MIN_POLL_INTERVAL
from .inventory import (
    Node,
    _existing_nodes_map,
    build_heater_address_map,
    build_node_inventory,
    normalize_heater_addresses,
    normalize_node_addr,
    normalize_node_type,
)
from .utils import float_or_none

_LOGGER = logging.getLogger(__name__)

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
@dataclass
class PendingSetting:
    """Track expected heater settings awaiting confirmation."""

    mode: str | None
    stemp: float | None
    expires_at: float


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


def _ensure_heater_section(
    nodes_by_type: dict[str, dict[str, Any]],
    factory: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Ensure ``nodes_by_type`` contains an ``htr`` section and return it."""

    existing = nodes_by_type.get("htr")
    if isinstance(existing, dict):
        return existing
    if isinstance(existing, Mapping):
        section = dict(existing)
        addrs = section.get("addrs")
        if isinstance(addrs, Iterable) and not isinstance(addrs, (list, str, bytes)):
            section["addrs"] = list(addrs)
        nodes_by_type["htr"] = section
        return section

    candidate = factory()
    if isinstance(candidate, dict):
        heater_section = candidate
    elif isinstance(candidate, Mapping):
        heater_section = dict(candidate)
    else:  # pragma: no cover - defensive conversion
        heater_section = dict(candidate or {})
    addrs = heater_section.get("addrs")
    if isinstance(addrs, Iterable) and not isinstance(addrs, (list, str, bytes)):
        heater_section["addrs"] = list(addrs)
    nodes_by_type["htr"] = heater_section
    return heater_section


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
        nodes: dict[str, Any],
        node_inventory: list[Node] | None = None,
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
        self._nodes: dict[str, Any] = {}
        self._node_inventory: list[Node] = []
        self._nodes_by_type: dict[str, list[str]] = {}
        self._addr_lookup: dict[str, set[str]] = {}
        self._pending_settings: dict[tuple[str, str], PendingSetting] = {}
        self._invalid_nodes_logged = False
        self._rtc_reference: datetime | None = None
        self._rtc_reference_monotonic: float | None = None
        self.update_nodes(nodes, node_inventory=node_inventory)

    def _set_inventory_from_nodes(
        self,
        nodes: Mapping[str, Any] | None,
        provided: Iterable[Node] | None = None,
    ) -> list[Node]:
        """Populate the cached inventory from ``provided`` or ``nodes``."""

        if provided is not None:
            inventory = list(provided)
        elif nodes:
            try:
                inventory = build_node_inventory(nodes)
            except ValueError as err:  # pragma: no cover - defensive
                _LOGGER.debug(
                    "Failed to build node inventory: %s",
                    err,
                    exc_info=err,
                )
                inventory = []
        else:
            inventory = []

        self._node_inventory = inventory
        return inventory

    def _ensure_inventory(self) -> list[Node]:
        """Return cached node inventory, rebuilding when necessary."""
        if not self._node_inventory:
            self._set_inventory_from_nodes(self._nodes)
        self._refresh_node_cache()
        return self._node_inventory

    def _refresh_node_cache(self) -> None:
        """Rebuild cached mappings of node types to addresses."""

        inventory = self._node_inventory
        forward_map, reverse_map = build_heater_address_map(inventory)

        nodes_by_type: dict[str, list[str]] = {
            node_type: list(addresses) for node_type, addresses in forward_map.items()
        }
        addr_lookup: dict[str, set[str]] = {
            str(addr): set(node_types) for addr, node_types in reverse_map.items()
        }

        for node in inventory:
            node_type = normalize_node_type(getattr(node, "type", ""))
            addr = normalize_node_addr(getattr(node, "addr", ""))
            if not node_type or not addr:
                continue
            bucket = nodes_by_type.setdefault(node_type, [])
            nodes_by_type[node_type] = list(dict.fromkeys([*bucket, addr]))
            addr_lookup.setdefault(addr, set()).add(node_type)

        self._nodes_by_type = nodes_by_type
        self._addr_lookup = addr_lookup

    def _merge_address_payload(
        self, payload: Mapping[str, Iterable[Any]] | None
    ) -> None:
        """Merge normalized heater addresses into cached lookups."""

        if not payload:
            return

        cleaned_map, _ = normalize_heater_addresses(payload)

        for node_type, addrs in cleaned_map.items():
            if not addrs:
                continue
            bucket = self._nodes_by_type.setdefault(node_type, [])
            for addr in addrs:
                if addr not in bucket:
                    bucket.append(addr)
                existing = self._addr_lookup.get(addr)
                if isinstance(existing, set):
                    lookup = existing
                else:
                    lookup = set()
                    if isinstance(existing, str) and existing:
                        lookup.add(existing)
                    self._addr_lookup[addr] = lookup
                lookup.add(node_type)

    def _register_node_address(self, node_type: str, addr: str) -> None:
        """Add ``addr`` to the cached map for ``node_type`` if missing."""
        self._merge_address_payload({node_type: [addr]})

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

    def _apply_boost_metadata_for_section(
        self,
        section: Mapping[str, Any] | None,
        *,
        now: datetime | None,
    ) -> None:
        """Apply boost metadata derivation to every settings payload."""

        if not isinstance(section, Mapping):
            return
        settings = section.get("settings")
        if not isinstance(settings, MutableMapping):
            return
        for payload in settings.values():
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
            "Deferring merge due to pending settings type=%s addr=%s expected_mode=%s expected_stemp=%s payload_mode=%s payload_stemp=%s",
            key[0],
            key[1],
            mode_expected,
            stemp_expected,
            mode_payload,
            stemp_payload,
        )
        return True

    def _merge_nodes_by_type(
        self,
        cache_map: Mapping[str, Iterable[str]],
        current_sections: Mapping[str, Any] | None,
        new_payload: Mapping[str, Mapping[str, Any]] | None,
    ) -> dict[str, dict[str, Any]]:
        """Merge cached addresses, existing sections and payload settings."""

        sections: dict[str, Any] = {}
        if isinstance(current_sections, Mapping):
            sections = dict(current_sections)

        payload_map: dict[str, Mapping[str, Any]] = {}
        if isinstance(new_payload, Mapping):
            payload_map = dict(new_payload)

        merged_types = set(cache_map) | set(sections) | set(payload_map)
        nodes_by_type: dict[str, dict[str, Any]] = {}

        for node_type in merged_types:
            default_addrs = cache_map.get(node_type, [])
            section = sections.get(node_type)
            normalized = self._normalise_type_section(node_type, section, default_addrs)

            payload_settings = payload_map.get(node_type)
            if isinstance(payload_settings, Mapping):
                settings_bucket = normalized.setdefault("settings", {})
                for raw_addr, data in payload_settings.items():
                    addr = normalize_node_addr(raw_addr)
                    if not addr:
                        continue
                    settings_bucket[addr] = data
                    if addr not in normalized["addrs"]:
                        normalized["addrs"].append(addr)

            cache_addrs = cache_map.get(node_type)
            if cache_addrs:
                for raw_addr in cache_addrs:
                    addr = normalize_node_addr(raw_addr)
                    if addr and addr not in normalized["addrs"]:
                        normalized["addrs"].append(addr)

            nodes_by_type[node_type] = normalized

        return nodes_by_type

    @staticmethod
    def _normalise_type_section(
        node_type: str,
        section: Any,
        default_addrs: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Return a standard mapping for a node type section."""

        addrs: list[str] = []
        if default_addrs is not None:
            addrs = [
                addr
                for addr in (
                    normalize_node_addr(candidate) for candidate in default_addrs
                )
                if addr
            ]

        settings: dict[str, Any] = {}

        if isinstance(section, dict):
            raw_addrs = section.get("addrs")
            if isinstance(raw_addrs, Iterable) and not isinstance(
                raw_addrs, (str, bytes)
            ):
                addrs = [
                    addr
                    for addr in (
                        normalize_node_addr(candidate) for candidate in raw_addrs
                    )
                    if addr
                ]
            raw_settings = section.get("settings")
            if isinstance(raw_settings, dict):
                settings = {
                    normalized: data
                    for addr, data in raw_settings.items()
                    if (normalized := normalize_node_addr(addr))
                }

        # Ensure addrs contains any addresses present in settings
        for addr in settings:
            if addr not in addrs:
                addrs.append(addr)

        return {"addrs": addrs, "settings": settings}

    def update_nodes(
        self,
        nodes: Mapping[str, Any] | None,
        node_inventory: list[Node] | None = None,
    ) -> None:
        """Update cached node payload and inventory."""

        if isinstance(nodes, Mapping):
            self._nodes = dict(nodes)
            self._invalid_nodes_logged = False
        else:
            if not self._invalid_nodes_logged:
                _LOGGER.debug(
                    "Ignoring unexpected nodes payload (type=%s): %s",
                    type(nodes).__name__ if nodes is not None else "NoneType",
                    nodes,
                )
                self._invalid_nodes_logged = True
            self._nodes = {}
        self._set_inventory_from_nodes(self._nodes, provided=node_inventory)
        self._refresh_node_cache()

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

            self._ensure_inventory()
            reverse = {
                normalize_node_addr(address): set(types)
                for address, types in self._addr_lookup.items()
                if normalize_node_addr(address)
            }
            addr_types = reverse.get(addr)
            resolved_type = node_type or (
                next(iter(addr_types)) if addr_types else "htr"
            )

            payload = await self.client.get_node_settings(dev_id, (resolved_type, addr))

            if not isinstance(payload, dict):
                _LOGGER.debug(
                    "Ignoring unexpected heater settings payload for node_type=%s addr=%s: %s",
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

            current = self.data or {}
            new_data: dict[str, dict[str, Any]] = dict(current)
            dev_data = dict(new_data.get(dev_id) or {})

            if not dev_data:
                dev_data = {
                    "dev_id": dev_id,
                    "name": _device_display_name(self._device, dev_id),
                    "raw": self._device,
                    "connected": True,
                    "nodes": self._nodes,
                }
            else:
                dev_data.setdefault("dev_id", dev_id)
                if "name" not in dev_data:
                    dev_data["name"] = _device_display_name(self._device, dev_id)
                dev_data.setdefault("raw", self._device)
                dev_data.setdefault("nodes", self._nodes)
                dev_data.setdefault("connected", True)

            node_type = resolved_type

            if self._should_defer_pending_setting(node_type, addr, payload):
                _LOGGER.debug(
                    "Skipping heater refresh merge for pending settings type=%s addr=%s",
                    node_type,
                    addr,
                )
                success = True
                return

            existing_nodes = _existing_nodes_map(dev_data)

            cache_map = dict(self._nodes_by_type)
            self._register_node_address(node_type, addr)
            payload_map = {node_type: {addr: payload}}
            nodes_by_type = self._merge_nodes_by_type(
                cache_map,
                existing_nodes,
                payload_map,
            )

            dev_data["nodes"] = self._nodes
            dev_data["nodes_by_type"] = {
                n_type: {
                    "addrs": list(section["addrs"]),
                    "settings": dict(section["settings"]),
                }
                for n_type, section in nodes_by_type.items()
            }

            for n_type, section in dev_data["nodes_by_type"].items():
                dev_data[n_type] = section

            heater_section = _ensure_heater_section(
                dev_data["nodes_by_type"],
                lambda: self._normalise_type_section(
                    "htr", {}, self._nodes_by_type.get("htr", [])
                ),
            )
            dev_data["htr"] = heater_section
            new_data[dev_id] = dev_data
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
                "Finished heater settings refresh for node_type=%s addr=%s (success=%s)",
                node_type or resolved_type,
                addr,
                success,
            )

    async def _async_update_data(self) -> dict[str, dict[str, Any]]:
        """Fetch the latest settings for every known node on each poll."""
        dev_id = self._dev_id
        self._ensure_inventory()
        addr_map = dict(self._nodes_by_type)
        reverse = {address: set(types) for address, types in self._addr_lookup.items()}
        addrs = [addr for addrs in addr_map.values() for addr in addrs]
        rtc_now: datetime | None = None
        try:
            self._prune_pending_settings()
            prev_dev = (self.data or {}).get(dev_id, {})
            prev_by_type: dict[str, dict[str, Any]] = {}

            existing_nodes = prev_dev.get("nodes_by_type")
            if isinstance(existing_nodes, dict):
                for node_type, section in existing_nodes.items():
                    normalised = self._normalise_type_section(
                        node_type,
                        section,
                        addr_map.get(node_type, []),
                    )
                    if normalised["settings"]:
                        prev_by_type[node_type] = dict(normalised["settings"])

            for key, value in prev_dev.items():
                if key in {
                    "dev_id",
                    "name",
                    "raw",
                    "connected",
                    "nodes",
                    "nodes_by_type",
                }:
                    continue
                if not isinstance(value, dict):
                    continue
                normalised = self._normalise_type_section(
                    key,
                    value,
                    addr_map.get(key, []),
                )
                if normalised["settings"]:
                    bucket = prev_by_type.setdefault(key, {})
                    bucket.update(normalised["settings"])

            all_types = set(addr_map) | set(prev_by_type)
            settings_by_type: dict[str, dict[str, Any]] = {
                node_type: dict(prev_by_type.get(node_type, {}))
                for node_type in all_types
            }

            if addrs:
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
                        js = await self.client.get_node_settings(
                            dev_id, (resolved_type, addr)
                        )
                        if not isinstance(js, dict):
                            continue
                        if self._should_defer_pending_setting(resolved_type, addr, js):
                            _LOGGER.debug(
                                "Deferring poll merge for pending settings type=%s addr=%s",
                                resolved_type,
                                addr,
                            )
                            continue
                        if (
                            resolved_type == "acm"
                            and isinstance(js, MutableMapping)
                        ):
                            now_value: datetime | None = None
                            if self._requires_boost_resolution(js) and rtc_now is None:
                                rtc_now = await self._async_fetch_rtc_datetime()
                            now_value = rtc_now or self._device_now_estimate()
                            self._apply_accumulator_boost_metadata(js, now=now_value)
                        bucket = settings_by_type.setdefault(resolved_type, {})
                        bucket[addr] = js

            dev_name = _device_display_name(self._device, dev_id)

            for node_type, settings in settings_by_type.items():
                for addr in settings:
                    self._register_node_address(node_type, str(addr))

            addr_map = dict(self._nodes_by_type)

            existing_nodes = _existing_nodes_map(prev_dev)

            nodes_by_type = self._merge_nodes_by_type(
                addr_map,
                existing_nodes,
                settings_by_type,
            )

            acm_section = nodes_by_type.get("acm")
            if isinstance(acm_section, Mapping):
                now_value = rtc_now or self._device_now_estimate()
                self._apply_boost_metadata_for_section(acm_section, now=now_value)

            heater_section = _ensure_heater_section(
                nodes_by_type,
                lambda: {
                    "addrs": list(addr_map.get("htr", [])),
                    "settings": dict(settings_by_type.get("htr", {})),
                },
            )

            result = {
                dev_id: {
                    "dev_id": dev_id,
                    "name": dev_name,
                    "raw": self._device,
                    "connected": True,
                    "nodes": self._nodes,
                    "nodes_by_type": nodes_by_type,
                },
            }

            for node_type, section in nodes_by_type.items():
                result[dev_id][node_type] = section

            result[dev_id]["htr"] = heater_section

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
        addrs: Iterable[str] | Mapping[str, Iterable[str]],
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
        self._addresses_by_type: dict[str, list[str]] = {}
        self._addr_lookup: dict[str, str] = {}
        self._addrs: list[str] = []
        self._compat_aliases: dict[str, str] = {}
        self._last: dict[tuple[str, str], tuple[float, float]] = {}
        self._base_interval = HTR_ENERGY_UPDATE_INTERVAL
        self._base_interval_seconds = HTR_ENERGY_UPDATE_INTERVAL.total_seconds()
        self._ws_lease: float = 0.0
        self._ws_deadline: float | None = None
        self._ws_margin_default = 60.0
        self.update_addresses(addrs)

    def update_addresses(
        self, addrs: Iterable[str] | Mapping[str, Iterable[str]]
    ) -> None:
        """Replace the tracked heater addresses with ``addrs``."""

        cleaned_map, compat_aliases = normalize_heater_addresses(addrs)
        self._addresses_by_type = cleaned_map
        self._compat_aliases = compat_aliases
        self._addr_lookup = {
            addr: node_type
            for node_type, addrs_for_type in cleaned_map.items()
            for addr in addrs_for_type
        }
        self._addrs = [
            addr for addrs_for_type in cleaned_map.values() for addr in addrs_for_type
        ]

        valid_keys = {
            (node_type, addr)
            for node_type, addrs_for_type in cleaned_map.items()
            for addr in addrs_for_type
        }
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

        kwh = counter / 1000.0
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
            energy_by_type: dict[str, dict[str, float]] = {}
            power_by_type: dict[str, dict[str, float]] = {}

            for node_type, addrs_for_type in self._addresses_by_type.items():
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
                    t = float_or_none(last.get("t"))
                    if counter is None or t is None:
                        _LOGGER.debug(
                            "Latest sample missing 't' or 'counter' for node_type=%s addr=%s",
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

            dev_data: dict[str, Any] = {"dev_id": dev_id}
            nodes_by_type: dict[str, dict[str, Any]] = {}

            for node_type, addrs_for_type in self._addresses_by_type.items():
                bucket = {
                    "energy": dict(energy_by_type.get(node_type, {})),
                    "power": dict(power_by_type.get(node_type, {})),
                    "addrs": list(addrs_for_type),
                }
                dev_data[node_type] = bucket
                nodes_by_type[node_type] = bucket

            heater_data = _ensure_heater_section(
                nodes_by_type,
                lambda: {
                    "energy": dict(energy_by_type.get("htr", {})),
                    "power": dict(power_by_type.get("htr", {})),
                    "addrs": list(self._addresses_by_type.get("htr", [])),
                },
            )
            dev_data["htr"] = heater_data

            for alias, canonical in self._compat_aliases.items():
                canonical_bucket = nodes_by_type.get(canonical)
                if canonical_bucket is None:
                    canonical_bucket = {
                        "energy": {},
                        "power": {},
                        "addrs": list(self._addresses_by_type.get(canonical, [])),
                    }
                    nodes_by_type[canonical] = canonical_bucket
                    dev_data[canonical] = canonical_bucket
                dev_data[alias] = canonical_bucket
                if alias not in nodes_by_type:
                    nodes_by_type[alias] = canonical_bucket

            dev_data["nodes_by_type"] = nodes_by_type

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
            counter = float_or_none(payload.get("counter"))
            if t is None or counter is None:
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
        nodes_by_type = dev_data.get("nodes_by_type")
        if not isinstance(nodes_by_type, dict):
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

        for raw_type, payload in updates.items():
            node_type = normalize_node_type(raw_type)
            if not node_type:
                continue
            tracked_addrs = self._addresses_by_type.get(node_type)
            if not tracked_addrs:
                continue
            bucket = nodes_by_type.get(node_type)
            if not isinstance(bucket, dict):
                continue
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
                    node_type,
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
            dev_data["nodes_by_type"] = nodes_by_type
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
