"""State coordinator: polls node settings and owns the domain state store."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import logging
from time import monotonic as time_mod
import typing
from typing import Any

from aiohttp import ClientError
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import UpdateFailed
from homeassistant.util import dt as dt_util

from custom_components.termoweb.backend.factory import backend_capabilities
from custom_components.termoweb.backend.rest_client import (
    BackendAuthError,
    BackendRateLimitError,
    RESTClient,
)
from custom_components.termoweb.backend.sanitize import mask_identifier, redact_text
from custom_components.termoweb.boost import resolve_boost_end_from_fields
from custom_components.termoweb.coerce import as_float, as_int
from custom_components.termoweb.const import BRAND_TERMOWEB, MIN_POLL_INTERVAL
from custom_components.termoweb.coordinator.base import (
    RaiseUpdateFailedCoordinator,
    wrap_logger,
)
from custom_components.termoweb.domain.energy import EnergySnapshot
from custom_components.termoweb.domain.ids import (
    NodeId as DomainNodeId,
    NodeType as DomainNodeType,
)
from custom_components.termoweb.domain.state import (
    AccumulatorState,
    DomainState,
    DomainStateStore,
    GatewayConnectionState,
    GeoData,
    HeaterState,
    NodeSettingsDelta,
    PowerMonitorState,
    ThermostatState,
    clone_state,
)
from custom_components.termoweb.domain.view import DomainStateView
from custom_components.termoweb.inventory import (
    Inventory,
    normalize_node_addr,
    normalize_node_type,
)

_LOGGER = logging.getLogger(__name__)


# TTL for pending heater setting confirmations.
_PENDING_SETTINGS_TTL = 10.0
_SETPOINT_TOLERANCE = 0.05


@dataclass
class PendingSetting:
    """Track expected heater settings awaiting confirmation."""

    mode: str | None
    stemp: float | None
    expires_at: float


@dataclass(frozen=True, slots=True)
class DeviceMetadata:
    """Represent immutable device metadata."""

    dev_id: str
    name: str
    model: str | None
    serial_id: str | None = None
    fw_version: str | None = None
    geo_data: GeoData | None = None


def _normalise_device_name(raw_name: Any, dev_id: str) -> str:
    """Return a trimmed device name or a fallback derived from ``dev_id``."""

    candidate = raw_name if isinstance(raw_name, str) else str(raw_name or "")
    trimmed = candidate.strip()
    return trimmed or f"Device {dev_id}"


def _normalise_device_model(raw_model: Any) -> str | None:
    """Return a trimmed model string when present."""

    if raw_model in (None, ""):
        return None
    candidate = raw_model if isinstance(raw_model, str) else str(raw_model)
    trimmed = candidate.strip()
    return trimmed or None


def _normalise_optional_string(raw: Any) -> str | None:
    """Return a trimmed string when present, otherwise ``None``."""

    if raw in (None, ""):
        return None
    candidate = raw if isinstance(raw, str) else str(raw)
    trimmed = candidate.strip()
    return trimmed or None


def build_device_metadata(
    dev_id: str, device: Mapping[str, typing.Any] | None
) -> DeviceMetadata:
    """Return immutable metadata derived from a device payload."""

    name = (
        _normalise_device_name(device.get("name"), dev_id)
        if isinstance(device, Mapping)
        else f"Device {dev_id}"
    )
    model = (
        _normalise_device_model(device.get("model"))
        if isinstance(device, Mapping)
        else None
    )
    serial_id = (
        _normalise_optional_string(device.get("serial_id"))
        if isinstance(device, Mapping)
        else None
    )
    fw_version = (
        _normalise_optional_string(device.get("fw_version"))
        if isinstance(device, Mapping)
        else None
    )
    return DeviceMetadata(
        dev_id=dev_id,
        name=name,
        model=model,
        serial_id=serial_id,
        fw_version=fw_version,
    )


def _device_display_name(device: DeviceMetadata, dev_id: str) -> str:
    """Return the trimmed device name or a fallback for ``dev_id``."""

    return device.name.strip() or f"Device {dev_id}"


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
        device: DeviceMetadata | Mapping[str, typing.Any] | None,
        inventory: Inventory,
        brand: str = BRAND_TERMOWEB,
    ) -> None:
        """Initialize the TermoWeb device coordinator."""
        super().__init__(
            hass,
            logger=wrap_logger(_LOGGER),
            name="termoweb",
            update_interval=timedelta(seconds=max(base_interval, MIN_POLL_INTERVAL)),
        )
        self.client = client
        self._base_interval = max(base_interval, MIN_POLL_INTERVAL)
        self._backoff = 0  # seconds
        self._dev_id = dev_id
        if isinstance(device, DeviceMetadata):
            metadata = device
        elif isinstance(device, Mapping) or device is None:
            metadata = build_device_metadata(dev_id, device)
        else:  # pragma: no cover - defensive
            msg = "StateCoordinator requires DeviceMetadata or a mapping payload"
            raise TypeError(msg)
        self._device_metadata = metadata
        self._brand = brand or BRAND_TERMOWEB
        if not isinstance(inventory, Inventory):
            msg = "StateCoordinator requires an Inventory instance"
            raise TypeError(msg)

        self._inventory: Inventory = inventory
        self._state_store = DomainStateStore(self._node_ids_from_inventory(inventory))
        self._pending_settings: dict[tuple[str, str], PendingSetting] = {}
        self._rtc_reference: datetime | None = None
        self._rtc_reference_monotonic: float | None = None
        self._domain_view = DomainStateView(dev_id, self._state_store)

    @property
    def domain_view(self) -> DomainStateView:
        """Return the read-only domain state view."""

        return self._domain_view

    @property
    def device_metadata(self) -> DeviceMetadata:
        """Return immutable metadata for this gateway."""

        return self._device_metadata

    @property
    def gateway_name(self) -> str:
        """Return the display name for this gateway."""

        return _device_display_name(self._device_metadata, self._dev_id)

    @property
    def gateway_model(self) -> str | None:
        """Return the display model name for this gateway."""

        return self._device_metadata.model

    @property
    def gateway_connected(self) -> bool:
        """Return True when the coordinator reports the gateway online."""

        connection = self._domain_view.get_gateway_connection_state()
        return bool(connection.connected)

    def update_gateway_connection(
        self,
        *,
        status: str | None,
        connected: bool,
        last_event_at: float | None,
        healthy_since: float | None,
        healthy_minutes: float | None,
        last_payload_at: float | None,
        last_heartbeat_at: float | None,
        payload_stale: bool | None,
        payload_stale_after: float | None,
        idle_restart_pending: bool | None,
    ) -> None:
        """Update the gateway connection state in the domain store."""

        state = GatewayConnectionState(
            status=status,
            connected=connected,
            last_event_at=last_event_at,
            healthy_since=healthy_since,
            healthy_minutes=healthy_minutes,
            last_payload_at=last_payload_at,
            last_heartbeat_at=last_heartbeat_at,
            payload_stale=payload_stale,
            payload_stale_after=payload_stale_after,
            idle_restart_pending=idle_restart_pending,
        )
        self._state_store.set_gateway_connection_state(state)
        self._publish_device_record()

    def apply_power_limit(self, value: Any) -> None:
        """Store a gateway power limit and notify listeners when it changed."""

        if self._state_store.set_power_limit(value):
            self._publish_device_record()

    def apply_energy_snapshot(self, snapshot: EnergySnapshot) -> None:
        """Store an energy snapshot in the domain state store."""

        if not isinstance(snapshot, EnergySnapshot):
            return
        if snapshot.dev_id != self._dev_id:
            return
        changed = self._state_store.set_energy_snapshot(snapshot)
        if changed:
            # An energy poll says nothing about the state poll: refresh
            # listeners without marking a failed state update successful.
            self.data = self._device_record()
            self.async_update_listeners()

    def _device_record(self) -> dict[str, dict[str, Any]]:
        """Return a minimal coordinator payload for this device."""

        model = self._device_metadata.model
        backend = self._brand

        record = {
            "dev_id": self._dev_id,
            "name": _device_display_name(self._device_metadata, self._dev_id),
            "model": model,
            "connected": self.gateway_connected,
            "backend": backend,
        }

        return {self._dev_id: record}

    def _publish_device_record(self) -> None:
        """Push the latest device snapshot to coordinator listeners."""

        self.async_set_updated_data(self._device_record())

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
                if resolved_type == "acm":
                    now_value: datetime | None = None
                    if self._requires_boost_resolution(payload) and current_rtc is None:
                        current_rtc = await self._async_fetch_rtc_datetime()
                    now_value = current_rtc or self._device_now_estimate()
                    self._apply_accumulator_boost_metadata(payload, now=now_value)
                store.apply_full_snapshot(
                    resolved_type,
                    addr,
                    payload,
                )
                _LOGGER.debug(
                    "Node %s/%s max_power=%s",
                    resolved_type,
                    addr,
                    store.get_state(resolved_type, addr).max_power,
                )
        return current_rtc

    @staticmethod
    def _normalize_mode_value(value: str | None) -> str | None:
        """Return a canonical string for backend HVAC modes."""

        if value is None:
            return None
        return value.strip().lower() or None

    @staticmethod
    def _node_key(node_type: str, addr: str) -> tuple[str, str] | None:
        """Return the normalised ``(node_type, addr)`` key for per-node caches."""

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

        key = self._node_key(node_type, addr)
        if key is None:
            return

        try:
            ttl_value = float(ttl)
        except TypeError, ValueError:  # pragma: no cover - defensive
            ttl_value = _PENDING_SETTINGS_TTL
        expires_at = time_mod() + max(ttl_value, 0.0)
        normalized_mode = self._normalize_mode_value(mode)
        normalized_stemp = as_float(stemp)
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
    def _rtc_payload_to_datetime(
        payload: Mapping[str, typing.Any] | None,
    ) -> datetime | None:
        """Return a timezone-aware datetime extracted from RTC payload."""

        if not isinstance(payload, Mapping):
            return None

        year = as_int(payload.get("y"))
        month = as_int(payload.get("n"))
        day = as_int(payload.get("d"))
        if year is None or month is None or day is None:
            return None

        hour = as_int(payload.get("h"))
        minute = as_int(payload.get("m"))
        second = as_int(payload.get("s"))

        tzinfo = dt_util.now().tzinfo or UTC
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
    def _requires_boost_resolution(payload: Mapping[str, typing.Any]) -> bool:
        """Return True when ``payload`` exposes boost day/min metadata."""

        day = as_int(payload.get("boost_end_day"))
        minute = as_int(payload.get("boost_end_min"))
        return day is not None or minute is not None

    def _apply_accumulator_boost_metadata(
        self,
        payload: MutableMapping[str, typing.Any],
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

    def _should_defer_pending_setting(
        self,
        node_type: str,
        addr: str,
        payload: Mapping[str, typing.Any],
    ) -> bool:
        """Return True when a pending write should defer payload merging."""

        key = self._node_key(node_type, addr)
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
        mode_payload = self._normalize_mode_value(payload.get("mode"))
        stemp_payload = as_float(payload.get("stemp"))

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

    def _node_ids_from_inventory(self, inventory: Inventory) -> list[DomainNodeId]:
        """Return domain node identifiers derived from ``inventory``."""

        node_ids: list[DomainNodeId] = []
        for node in inventory.nodes:
            node_type = DomainNodeType.coerce(node.type)
            if node_type is not None:
                node_ids.append(DomainNodeId(node_type, node.addr))
        return node_ids

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

        store = self._state_store
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

        self._publish_device_record()

    def apply_entity_patch(
        self, node_type: str, addr: str, mutator: Callable[[DomainState], None]
    ) -> bool:
        """Apply an optimistic patch through the domain store when available."""

        store = self._state_store
        normalized_type = normalize_node_type(
            node_type,
            use_default_when_falsey=True,
        )
        if not normalized_type:
            return False

        try:
            node_id = store.resolve_node_id(normalized_type, addr)
        except (TypeError, ValueError) as err:  # pragma: no cover - defensive
            _LOGGER.debug(
                "Failed to resolve node for optimistic patch type=%s addr=%s: %s",
                node_type,
                addr,
                err,
            )
            return False

        if node_id is None:
            return False

        current_state = store.get_state(node_id.node_type, node_id.addr)
        working_state = clone_state(current_state)
        if working_state is None:
            if node_id.node_type is DomainNodeType.ACCUMULATOR:
                working_state = AccumulatorState()
            elif node_id.node_type is DomainNodeType.THERMOSTAT:
                working_state = ThermostatState()
            elif node_id.node_type is DomainNodeType.POWER_MONITOR:
                working_state = PowerMonitorState()
            else:
                working_state = HeaterState()

        before_boost = (
            getattr(working_state, "boost_active", None),
            getattr(working_state, "boost_end_day", None),
            getattr(working_state, "boost_end_min", None),
        )
        try:
            mutator(working_state)
        except asyncio.CancelledError:
            raise
        except Exception as err:  # pragma: no cover - defensive  # noqa: BLE001
            _LOGGER.debug(
                "Failed to apply optimistic patch type=%s addr=%s: %s",
                node_type,
                addr,
                err,
            )
            return False
        after_boost = (
            getattr(working_state, "boost_active", None),
            getattr(working_state, "boost_end_day", None),
            getattr(working_state, "boost_end_min", None),
        )
        if before_boost != after_boost and isinstance(working_state, AccumulatorState):
            working_state.boost_end_datetime = None
            working_state.boost_minutes_delta = None

        store.replace_state(node_id.node_type, node_id.addr, working_state)
        self._publish_device_record()
        return True

    async def async_refresh_heater(self, node: tuple[str, str]) -> None:
        """Refresh settings for a specific node and push the update to listeners."""

        dev_id = self._dev_id
        success = False
        raw_type, raw_addr = node
        resolved_type = normalize_node_type(raw_type, use_default_when_falsey=True)
        addr = normalize_node_addr(raw_addr, use_default_when_falsey=True)

        _LOGGER.debug(
            "Refreshing heater settings node_type=%s addr=%s",
            resolved_type,
            addr,
        )

        try:
            self._prune_pending_settings()
            if not resolved_type or not addr:
                _LOGGER.error(
                    "Cannot refresh heater settings without a node type and address",
                )
                return

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

            self._state_store.apply_full_snapshot(
                resolved_type,
                addr,
                payload,
            )
            self._publish_device_record()
            success = True

        except TimeoutError as err:
            _LOGGER.warning(
                "Timeout refreshing heater settings for node_type=%s addr=%s: %s",
                resolved_type,
                addr,
                err or "timed out",
            )
        except (ClientError, BackendRateLimitError, BackendAuthError) as err:
            _LOGGER.warning(
                "Failed to refresh heater settings for node_type=%s addr=%s: %s",
                resolved_type,
                addr,
                redact_text(str(err)),
            )
        finally:
            _LOGGER.debug(
                "Finished heater settings refresh for node_type=%s "
                "addr=%s (success=%s)",
                resolved_type,
                addr,
                success,
            )

    async def _async_update_data(self) -> dict[str, dict[str, Any]]:
        """Fetch the latest settings for every known node on each poll."""
        dev_id = self._dev_id
        addr_map, reverse = self._inventory.heater_address_map
        addrs = [addr for addrs in addr_map.values() for addr in addrs]
        rtc_now: datetime | None = None
        try:
            self._prune_pending_settings()
            if addrs:
                rtc_now = await self._async_fetch_settings_to_store(
                    dev_id,
                    addr_map,
                    reverse,
                    self._state_store,
                    rtc_now,
                )

            # Poll the installation power limit when the backend has one.
            if backend_capabilities(self._brand).power_limit:
                try:
                    self._state_store.set_power_limit(
                        await self.client.get_power_limit(dev_id)
                    )
                except Exception:  # noqa: BLE001
                    _LOGGER.debug(
                        "Failed to poll power limit for %s",
                        mask_identifier(dev_id),
                    )

            result = self._device_record()

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
                # Do not re-enable polling that WS health suspended meanwhile.
                if self.update_interval is not None:
                    self.update_interval = timedelta(seconds=self._base_interval)

            return result

    def resume_polling(self, base_interval: int) -> None:
        """Resume REST polling at ``base_interval`` or the rate-limit backoff."""

        self.update_interval = timedelta(seconds=max(base_interval, self._backoff))
