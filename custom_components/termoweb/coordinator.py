"""Coordinator helpers for the TermoWeb integration."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, MutableMapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import logging
import time
from time import monotonic as time_mod
from typing import Any, TypeVar, cast

from aiohttp import ClientError
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util

from .api import BackendAuthError, BackendRateLimitError, RESTClient
from .boost import coerce_int, resolve_boost_end_from_fields
from .const import HTR_ENERGY_UPDATE_INTERVAL, MIN_POLL_INTERVAL
from .inventory import (
    Inventory,
    build_node_inventory,
    normalize_heater_addresses,
    normalize_node_addr,
    normalize_node_type,
    normalize_power_monitor_addresses,
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

ENERGY_NODE_TYPES: frozenset[str] = frozenset({"htr", "acm", "pmo"})


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
        self._inventory: Inventory | None = None
        self._pending_settings: dict[tuple[str, str], PendingSetting] = {}
        self._invalid_nodes_logged = False
        self._invalid_inventory_logged = False
        self._rtc_reference: datetime | None = None
        self._rtc_reference_monotonic: float | None = None
        self.update_nodes(nodes, inventory=inventory)

    @staticmethod
    def _collect_previous_settings(
        prev_dev: Mapping[str, Any],
        addr_map: Mapping[str, Iterable[str]],
    ) -> dict[str, dict[str, Any]]:
        """Return normalised settings carried over from previous poll."""

        preserved: dict[str, dict[str, Any]] = {}

        existing_settings = prev_dev.get("settings")
        if isinstance(existing_settings, Mapping):
            for node_type, bucket in existing_settings.items():
                if not isinstance(bucket, Mapping):
                    continue
                dest: dict[str, Any] = preserved.setdefault(node_type, {})
                for raw_addr, payload in bucket.items():
                    addr = normalize_node_addr(
                        raw_addr,
                        use_default_when_falsey=True,
                    )
                    if not addr:
                        continue
                    dest[addr] = payload

        def _ingest_section(node_type: str, section: Any) -> None:
            if node_type in preserved:
                return
            normalised = StateCoordinator._normalise_type_section(
                node_type,
                section,
                addr_map.get(node_type, []),
            )
            if not normalised["settings"]:
                return
            preserved[node_type] = dict(normalised["settings"])

        for key, value in prev_dev.items():
            if key in {
                "dev_id",
                "name",
                "raw",
                "connected",
                "nodes",
                "nodes_by_type",
                "settings",
                "addresses_by_type",
                "heater_address_map",
                "power_monitor_address_map",
                "inventory",
            }:
                continue
            if not isinstance(value, Mapping):
                continue
            _ingest_section(key, value)

        return preserved

    async def _async_fetch_settings_by_address(
        self,
        dev_id: str,
        addr_map: Mapping[str, Iterable[str]],
        reverse: Mapping[str, set[str]],
        settings_by_type: dict[str, dict[str, Any]],
        rtc_now: datetime | None,
    ) -> datetime | None:
        """Fetch settings for every address and update ``settings_by_type``."""

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
                bucket = settings_by_type.setdefault(resolved_type, {})
                bucket[addr] = payload
        return current_rtc

    @staticmethod
    def _normalise_settings_map(
        settings_by_type: Mapping[str, Mapping[str, Any]] | None,
    ) -> dict[str, dict[str, Any]]:
        """Return a copy of ``settings_by_type`` keyed by normalised addresses."""

        if not isinstance(settings_by_type, Mapping):
            return {}

        normalised: dict[str, dict[str, Any]] = {}
        for node_type, bucket in settings_by_type.items():
            normalized_type = normalize_node_type(
                node_type, use_default_when_falsey=True
            )
            if not normalized_type or not isinstance(bucket, Mapping):
                continue
            dest: dict[str, Any] = {}
            for raw_addr, payload in bucket.items():
                addr = normalize_node_addr(raw_addr, use_default_when_falsey=True)
                if not addr:
                    continue
                dest[addr] = deepcopy(payload)
            normalised[normalized_type] = dest
        return normalised

    def _assemble_device_record(
        self,
        *,
        inventory: Inventory,
        settings_by_type: Mapping[str, Mapping[str, Any]],
        name: str,
    ) -> dict[str, Any]:
        """Return a coordinator cache record for ``inventory`` and settings."""

        normalized_settings = self._normalise_settings_map(settings_by_type)

        addresses_by_type = inventory.addresses_by_type
        for node_type, bucket in normalized_settings.items():
            dest = addresses_by_type.setdefault(node_type, [])
            seen = set(dest)
            for addr in bucket:
                if addr in seen:
                    continue
                dest.append(addr)
                seen.add(addr)

        heater_forward, heater_reverse = inventory.heater_address_map
        power_forward, power_reverse = inventory.power_monitor_address_map

        return {
            "dev_id": self._dev_id,
            "name": name,
            "raw": self._device,
            "connected": True,
            "inventory": inventory,
            "addresses_by_type": addresses_by_type,
            "heater_address_map": {
                "forward": heater_forward,
                "reverse": heater_reverse,
            },
            "power_monitor_address_map": {
                "forward": power_forward,
                "reverse": power_reverse,
            },
            "settings": normalized_settings,
        }

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

            default_order = [
                addr
                for addr in (
                    normalize_node_addr(candidate)
                    for candidate in cache_map.get(node_type, [])
                )
                if addr
            ]
            extras = [addr for addr in normalized["addrs"] if addr not in default_order]
            normalized["addrs"] = default_order + extras

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
        inventory: Inventory | None = None,
    ) -> None:
        """Update cached node payload and inventory."""

        valid_inventory: Inventory | None
        if isinstance(inventory, Inventory):
            valid_inventory = inventory
        elif (
            inventory is not None
            and hasattr(inventory, "payload")
            and hasattr(inventory, "nodes")
            and hasattr(inventory, "heater_address_map")
        ):
            valid_inventory = cast(Inventory, inventory)
        elif inventory is None:
            valid_inventory = None
        else:
            if not self._invalid_inventory_logged:
                _LOGGER.debug(
                    "Ignoring unexpected inventory container (type=%s): %s",
                    type(inventory).__name__,
                    inventory,
                )
                self._invalid_inventory_logged = True
            valid_inventory = None

        if isinstance(nodes, Mapping):
            if valid_inventory is not None:
                self._inventory = valid_inventory
                self._invalid_inventory_logged = False
                self._invalid_nodes_logged = False
                return

            try:
                node_list = list(build_node_inventory(nodes))
            except ValueError as err:  # pragma: no cover - defensive
                if not self._invalid_nodes_logged:
                    _LOGGER.debug(
                        "Failed to rebuild inventory from nodes payload: %s",
                        err,
                        exc_info=err,
                    )
                    self._invalid_nodes_logged = True
                self._inventory = None
            else:
                self._inventory = Inventory(self._dev_id, nodes, node_list)
                self._invalid_inventory_logged = False
                self._invalid_nodes_logged = False
            return

        if nodes is not None and not isinstance(nodes, Mapping):
            if not self._invalid_nodes_logged:
                _LOGGER.debug(
                    "Ignoring unexpected nodes payload (type=%s): %s",
                    type(nodes).__name__,
                    nodes,
                )
                self._invalid_nodes_logged = True
        else:
            self._invalid_nodes_logged = False

        if valid_inventory is not None:
            self._inventory = valid_inventory
            self._invalid_inventory_logged = False
            return

        if inventory is None:
            self._invalid_inventory_logged = False

        self._inventory = None

    def _ensure_inventory(self) -> Inventory | None:
        """Ensure cached inventory metadata is available."""

        return self._inventory

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

            if self._should_defer_pending_setting(resolved_type, addr, payload):
                _LOGGER.debug(
                    "Skipping heater refresh merge for pending settings type=%s addr=%s",
                    resolved_type,
                    addr,
                )
                success = True
                return

            current = self.data or {}
            new_data: dict[str, dict[str, Any]] = dict(current)
            prev_dev = dict(new_data.get(dev_id) or {})
            prev_settings = self._collect_previous_settings(prev_dev, forward_map)

            settings_map = {
                node_type: dict(bucket) for node_type, bucket in prev_settings.items()
            }
            settings_bucket = settings_map.setdefault(resolved_type, {})
            settings_bucket[addr] = payload

            dev_name = _device_display_name(self._device, dev_id)
            device_record = self._assemble_device_record(
                inventory=inventory,
                settings_by_type=settings_map,
                name=dev_name,
            )

            new_data[dev_id] = device_record
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
        inventory = self._ensure_inventory()

        if inventory is None:
            _LOGGER.debug("Skipping poll because inventory metadata is unavailable")
            return {}

        addr_map, reverse = inventory.heater_address_map
        addrs = [addr for addrs in addr_map.values() for addr in addrs]
        rtc_now: datetime | None = None
        try:
            self._prune_pending_settings()
            prev_dev = (self.data or {}).get(dev_id, {})
            prev_by_type = self._collect_previous_settings(prev_dev, addr_map)
            all_types = set(addr_map) | set(prev_by_type)
            settings_by_type: dict[str, dict[str, Any]] = {
                node_type: dict(prev_by_type.get(node_type, {}))
                for node_type in all_types
            }

            if addrs:
                rtc_now = await self._async_fetch_settings_by_address(
                    dev_id,
                    addr_map,
                    reverse,
                    settings_by_type,
                    rtc_now,
                )

            dev_name = _device_display_name(self._device, dev_id)

            acm_settings = settings_by_type.get("acm")
            if isinstance(acm_settings, Mapping):
                now_value = rtc_now or self._device_now_estimate()
                self._apply_boost_metadata_for_settings(acm_settings, now=now_value)

            device_record = self._assemble_device_record(
                inventory=inventory,
                settings_by_type=settings_by_type,
                name=dev_name,
            )

            result = {dev_id: device_record}

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
        self._counter_scales: dict[str, float] = {
            "htr": 1000.0,
            "acm": 1000.0,
            "pmo": 3_600_000.0,
        }
        self.update_addresses(addrs)

    def update_addresses(
        self, addrs: Iterable[str] | Mapping[str, Iterable[str]]
    ) -> None:
        """Replace the tracked heater addresses with ``addrs``."""

        heater_map, heater_aliases = normalize_heater_addresses(addrs)
        power_source: Mapping[str, Iterable[str]] | None
        if isinstance(addrs, Mapping):
            power_source = addrs
        else:
            power_source = None
        power_map, power_aliases = normalize_power_monitor_addresses(power_source)

        cleaned_map: dict[str, list[str]] = {
            "htr": list(heater_map.get("htr", [])),
            "acm": list(heater_map.get("acm", [])),
            "pmo": list(power_map.get("pmo", [])),
        }
        for node_type in ENERGY_NODE_TYPES:
            cleaned_map.setdefault(node_type, [])

        compat_aliases: dict[str, str] = dict(heater_aliases)
        compat_aliases.update(power_aliases)
        for node_type in ENERGY_NODE_TYPES:
            compat_aliases.setdefault(node_type, node_type)

        self._addresses_by_type = cleaned_map
        self._compat_aliases = compat_aliases
        self._addr_lookup = {
            addr: node_type
            for node_type, addrs_for_type in cleaned_map.items()
            for addr in addrs_for_type
        }
        self._addrs = [
            addr
            for node_type in ENERGY_NODE_TYPES
            for addr in cleaned_map.get(node_type, [])
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

    def _seed_cached_energy_and_power(
        self, dev_id: str
    ) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
        """Return energy and power buckets pre-populated from cached data."""

        energy_by_type: dict[str, dict[str, float]] = {
            node_type: {} for node_type in self._addresses_by_type
        }
        power_by_type: dict[str, dict[str, float]] = {
            node_type: {} for node_type in self._addresses_by_type
        }

        cached_dev: Mapping[str, Any] | None = None
        if isinstance(self.data, Mapping):
            cached = self.data.get(dev_id)
            if isinstance(cached, Mapping):
                cached_dev = cached

        if cached_dev:
            for node_type, addrs_for_type in self._addresses_by_type.items():
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
            addrs_for_type = self._addresses_by_type.get(node_type)
            if not addrs_for_type or addr not in addrs_for_type:
                continue
            energy_bucket = energy_by_type.setdefault(node_type, {})
            energy_bucket.setdefault(addr, kwh)

        return energy_by_type, power_by_type

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
            energy_by_type, power_by_type = self._seed_cached_energy_and_power(dev_id)

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
                    if counter is None:
                        counter = float_or_none(last.get("counter_max"))
                    if counter is None:
                        counter = float_or_none(last.get("counter_min"))
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

            for node_type, addrs_for_type in self._addresses_by_type.items():
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
                    "addrs": list(self._addresses_by_type.get("htr", [])),
                }
                dev_data["htr"] = heater_data

            for alias, canonical in self._compat_aliases.items():
                canonical_bucket = dev_data.get(canonical)
                if not isinstance(canonical_bucket, dict):
                    canonical_bucket = {
                        "energy": {},
                        "power": {},
                        "addrs": list(self._addresses_by_type.get(canonical, [])),
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

        for raw_type, payload in updates.items():
            node_type = normalize_node_type(raw_type)
            if not node_type:
                continue
            canonical_type = self._compat_aliases.get(node_type, node_type)
            tracked_addrs = self._addresses_by_type.get(canonical_type)
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
