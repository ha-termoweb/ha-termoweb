"""Shared websocket helpers."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, MutableMapping
from dataclasses import dataclass
import logging
import time
import typing
from typing import TYPE_CHECKING, Any

import aiohttp
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_send

from custom_components.termoweb.backend.rest_client import RESTClient
from custom_components.termoweb.const import (
    BRAND_DUCAHEAT,
    BRAND_TERMOWEB,
    DOMAIN as TERMOWEB_DOMAIN,
    WS_NAMESPACE,
    signal_ws_status,
)
from custom_components.termoweb.inventory import (
    Inventory,
    normalize_node_addr,
    normalize_node_type,
)
from custom_components.termoweb.runtime import require_runtime

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .ducaheat_ws import DucaheatWSClient
    from .termoweb_ws import TermoWebWSClient

from .ws_health import WsHealthTracker

_LOGGER = logging.getLogger(__name__)
DOMAIN = TERMOWEB_DOMAIN

CANONICAL_SETTING_KEYS: tuple[str, ...] = (
    "mode",
    "stemp",
    "mtemp",
    "temp",
    "prog",
    "ptemp",
    "units",
    "state",
    "max_power",
    "batt_level",
    "charge_level",
    "boost",
    "charging",
    "current_charge_per",
    "target_charge_per",
    "boost_active",
    "boost_remaining",
    "boost_time",
    "boost_temp",
    "boost_end_day",
    "boost_end_min",
    "boost_end_datetime",
    "boost_minutes_delta",
)


class ConnectionRateLimiter:
    """Throttle websocket connection attempts within a fixed window."""

    def __init__(
        self,
        *,
        min_interval: float = 1.0,
        max_attempts: int = 3,
        window_seconds: float = 10.0,
        clock: Callable[[], float] | None = None,
        sleeper: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        """Initialise the limiter with optional clock and sleep hooks."""

        self._min_interval = float(min_interval)
        self._max_attempts = max(1, int(max_attempts))
        self._window_seconds = max(1.0, float(window_seconds))
        self._clock = clock or time.monotonic
        self._sleep = sleeper or asyncio.sleep
        self._recent = deque[float]()
        self._last_attempt = 0.0
        self._lock = asyncio.Lock()

    async def wait_for_slot(self) -> float:
        """Sleep if necessary before allowing the next connection attempt."""

        async with self._lock:
            now = self._clock()
            while self._recent and now - self._recent[0] > self._window_seconds:
                self._recent.popleft()

            delay = 0.0
            if self._recent:
                delay = max(0.0, self._last_attempt + self._min_interval - now)
            if len(self._recent) >= self._max_attempts:
                window_wait = self._window_seconds - (now - self._recent[0])
                delay = max(delay, window_wait)

            target_time = now + delay
            self._recent.append(target_time)
            self._last_attempt = target_time

        if delay > 0:
            await self._sleep(delay)
        return delay


def clone_payload_value(value: Any) -> Any:
    """Return a shallow copy of mapping or sequence payload values."""

    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def build_settings_delta(section: str, payload: Any) -> dict[str, Any]:
    """Extract canonical settings keys from a websocket section payload."""

    if section in {"samples", "status", "capabilities"}:
        return {}
    if section == "prog":
        return {"prog": clone_payload_value(payload)} if payload is not None else {}
    if section == "prog_temps":
        return {"ptemp": clone_payload_value(payload)} if payload is not None else {}
    if not isinstance(payload, Mapping):
        return {}
    return {
        key: clone_payload_value(payload[key])
        for key in CANONICAL_SETTING_KEYS
        if key in payload
    }


def resolve_ws_update_section(section: str | None) -> tuple[str | None, str | None]:
    """Map a websocket path segment onto the node section bucket."""

    if not section:
        return None, None

    lowered = section.lower()
    if lowered in {"status", "samples", "settings", "advanced"}:
        return lowered, None
    if lowered == "advanced_setup":
        return "advanced", "advanced_setup"
    if lowered in {"setup", "prog", "prog_temps", "capabilities"}:
        return "settings", lowered
    return "settings", lowered


def forward_ws_sample_updates(
    hass: HomeAssistant,
    entry_id: str,
    dev_id: str,
    updates: Mapping[str, Mapping[str, typing.Any]],
    *,
    logger: logging.Logger | None = None,
    log_prefix: str = "WS",
) -> None:
    """Relay websocket heater sample updates to the energy coordinator."""

    try:
        runtime = require_runtime(hass, entry_id)
    except LookupError:
        return
    energy_coordinator = runtime.energy_coordinator
    handler = getattr(energy_coordinator, "handle_ws_samples", None)
    if not callable(handler):
        return

    inventory = runtime.inventory
    if not isinstance(inventory, Inventory):
        active_logger = logger or _LOGGER
        active_logger.error("%s: inventory unavailable", log_prefix)
        return

    alias_map = inventory.sample_alias_map(
        base_aliases={"htr": "htr", "acm": "acm", "pmo": "pmo"}
    )
    allowed_types = inventory.energy_sample_types

    normalized_updates: dict[str, dict[str, Any]] = {}
    lease_seconds: float | None = None
    for raw_type, section in updates.items():
        if not isinstance(section, Mapping):
            continue
        node_type = normalize_node_type(raw_type, use_default_when_falsey=True)
        if not node_type:
            continue
        canonical_type = alias_map.get(node_type, node_type)
        if allowed_types is not None:
            allowed = canonical_type in allowed_types
            if not allowed and canonical_type in {"heater", "heaters"}:
                allowed = "htr" in allowed_types
            if not allowed:
                continue
        elif canonical_type == "thm":
            continue
        samples_section: Mapping[str, typing.Any] | None = None
        lease_candidate: Any = None
        if "samples" in section and isinstance(section.get("samples"), Mapping):
            samples_section = section["samples"]
            lease_candidate = section.get("lease_seconds")
        else:
            samples_section = section
            lease_candidate = section.get("lease_seconds")
        if lease_candidate is not None:
            try:
                lease_value = float(lease_candidate)
            except (TypeError, ValueError):
                lease_value = None
            else:
                if lease_value > 0:
                    lease_seconds = (
                        max(lease_seconds or 0.0, lease_value)
                        if lease_seconds is not None
                        else lease_value
                    )
        if not isinstance(samples_section, Mapping):
            continue
        bucket = normalized_updates.setdefault(canonical_type, {})
        for raw_addr, payload in samples_section.items():
            if raw_addr == "lease_seconds":
                continue
            addr = normalize_node_addr(raw_addr, use_default_when_falsey=True)
            if not addr:
                continue
            bucket[addr] = payload

    normalized_updates = {
        node_type: dict(section)
        for node_type, section in normalized_updates.items()
        if section
    }
    if not normalized_updates:
        return

    active_logger = logger or _LOGGER
    try:
        handler(
            dev_id,
            normalized_updates,
            lease_seconds=lease_seconds,
        )
    except Exception:  # pragma: no cover - defensive logging  # noqa: BLE001
        active_logger.debug(
            "%s: forwarding heater samples failed",
            log_prefix,
            exc_info=True,
        )


def translate_path_update(
    payload: Any,
    *,
    resolve_section: Callable[[str | None], tuple[str | None, str | None]],
) -> dict[str, Any] | None:
    """Translate ``{"path": ..., "body": ...}`` websocket frames into nodes."""

    if not isinstance(payload, Mapping):
        return None
    if "nodes" in payload:
        return None
    path = payload.get("path")
    body = payload.get("body")
    if not isinstance(path, str) or body is None:
        return None

    path = path.split("?", 1)[0]
    segments = [segment for segment in path.split("/") if segment]
    if not segments:
        return None

    try:
        devs_idx = segments.index("devs")
    except ValueError:
        devs_idx = -1

    if devs_idx >= 0:
        relevant = segments[devs_idx + 1 :]
        node_type_idx = 1
        addr_idx = 2
        section_idx = 3
        if len(relevant) <= addr_idx:
            return None
    else:
        relevant = segments
        node_type_idx = 0
        addr_idx = 1
        section_idx = 2
        if len(relevant) <= addr_idx:
            return None

    node_type = normalize_node_type(relevant[node_type_idx])
    addr = normalize_node_addr(relevant[addr_idx])
    if not node_type or not addr:
        return None

    section = relevant[section_idx] if len(relevant) > section_idx else None
    remainder = relevant[section_idx + 1 :] if len(relevant) > section_idx + 1 else []

    target_section, nested_key = resolve_section(section)
    if target_section is None:
        return None

    payload_body: Any = body
    for segment in reversed(remainder):
        payload_body = {segment: payload_body}
    if nested_key:
        payload_body = {nested_key: payload_body}

    return {node_type: {target_section: {addr: payload_body}}}


DUCAHEAT_NAMESPACE = "/api/v2/socket_io"


@dataclass
class WSStats:
    """Track websocket frame and event stats."""

    frames_total: int = 0
    events_total: int = 0
    last_event_ts: float = 0.0
    last_paths: list[str] | None = None


class HandshakeError(RuntimeError):
    """Raised when a websocket handshake fails."""

    def __init__(
        self,
        status: int,
        url: str,
        detail: str,
        response_snippet: str | None = None,
    ) -> None:
        """Initialise a handshake failure with response metadata."""
        super().__init__(f"handshake failed: status={status}, detail={detail}")
        self.status = status
        self.url = url
        self.detail = detail
        self.response_snippet = (
            response_snippet if response_snippet is not None else detail
        )


class _WsLeaseMixin:
    """Provide reconnect backoff management for websocket clients."""

    def __init__(self) -> None:
        """Initialise lease tracking and backoff state."""
        self._payload_idle_window: float = 240.0
        self._subscription_refresh_lock = asyncio.Lock()
        self._subscription_refresh_failed = False
        self._backoff_idx = 0
        self._backoff = (5, 10, 30, 120, 300)

    def _reset_backoff(self) -> None:
        self._backoff_idx = 0

    def _next_backoff(self) -> float:
        idx = min(self._backoff_idx, len(self._backoff) - 1)
        self._backoff_idx = min(self._backoff_idx + 1, len(self._backoff) - 1)
        return self._backoff[idx]


class _WSStatusMixin:
    """Provide shared websocket status helpers."""

    hass: HomeAssistant
    entry_id: str
    dev_id: str

    def _status_should_reset_health(self, status: str) -> bool:
        """Return True when a status should clear healthy tracking."""

        return False

    def _ws_bucket_sizes(self) -> tuple[int, int]:
        """Return the current websocket state and tracker bucket sizes."""

        try:
            runtime = require_runtime(self.hass, self.entry_id)
        except LookupError:
            ws_size = 0
            trackers_size = 0
        else:
            ws_size = len(runtime.ws_state)
            trackers_size = len(runtime.ws_trackers)

        footprint = getattr(self, "_ws_bucket_baseline", None)
        snapshot = (ws_size, trackers_size)
        if footprint is None:
            setattr(self, "_ws_bucket_baseline", snapshot)
        else:
            grew = snapshot[0] > footprint[0] or snapshot[1] > footprint[1]
            if grew and _LOGGER.isEnabledFor(logging.DEBUG):
                _LOGGER.debug(
                    "WS: websocket bucket footprint grew from %s to %s for %s",  # pragma: no cover - debug only
                    footprint,
                    snapshot,
                    getattr(self, "dev_id", "unknown"),
                )
            if grew:
                setattr(self, "_ws_bucket_baseline", snapshot)
        return snapshot

    def _ws_state_bucket(self) -> dict[str, Any]:
        """Return the websocket state bucket for the current device."""

        ws_state = getattr(self, "_ws_state", None)
        if isinstance(ws_state, dict):
            return ws_state

        try:
            runtime = require_runtime(self.hass, self.entry_id)
        except LookupError:
            ws_state = {}
            setattr(self, "_ws_state", ws_state)
            return ws_state

        ws_bucket = runtime.ws_state
        if not isinstance(ws_bucket, dict):
            runtime.ws_state = {}
            ws_bucket = runtime.ws_state
        ws_state = ws_bucket.setdefault(self.dev_id, {})
        setattr(self, "_ws_state", ws_state)
        self._ws_bucket_sizes()
        return ws_state

    def _ws_health_tracker(self) -> WsHealthTracker:
        """Return the :class:`WsHealthTracker` for this websocket client."""

        cached = getattr(self, "_ws_tracker", None)
        if isinstance(cached, WsHealthTracker):
            if (
                getattr(self, "_pending_default_cadence_hint", False)
                and hasattr(self, "_apply_payload_window_hint")
                and not getattr(self, "_suppress_default_cadence_hint", False)
            ):
                self._pending_default_cadence_hint = False
                self._apply_payload_window_hint(
                    source="cadence",
                    lease_seconds=120,
                    candidates=[30, 75, "90"],
                )
            return cached

        try:
            runtime = require_runtime(self.hass, self.entry_id)
        except LookupError:
            tracker = WsHealthTracker(self.dev_id)
            setattr(self, "_ws_tracker", tracker)
            return tracker

        trackers = runtime.ws_trackers
        if not isinstance(trackers, dict):
            runtime.ws_trackers = {}
            trackers = runtime.ws_trackers
        tracker = trackers.get(self.dev_id)
        created = False
        if not isinstance(tracker, WsHealthTracker):
            tracker = WsHealthTracker(self.dev_id)
            trackers[self.dev_id] = tracker
            created = True
        legacy_status = getattr(self, "_status", None)
        if (
            isinstance(legacy_status, str)
            and legacy_status
            and tracker.status == "stopped"
        ):
            tracker.status = legacy_status
        legacy_since = getattr(self, "_healthy_since", None)
        if tracker.healthy_since is None and isinstance(legacy_since, (int, float)):
            tracker.healthy_since = float(legacy_since)
        legacy_payload = getattr(self, "_last_payload_at", None)
        if tracker.last_payload_at is None and isinstance(legacy_payload, (int, float)):
            tracker.last_payload_at = float(legacy_payload)
        legacy_heartbeat = getattr(self, "_last_heartbeat_at", None)
        if tracker.last_heartbeat_at is None and isinstance(
            legacy_heartbeat, (int, float)
        ):
            tracker.last_heartbeat_at = float(legacy_heartbeat)
        setattr(self, "_ws_tracker", tracker)
        self._ws_bucket_sizes()
        should_apply_hint = created and hasattr(self, "_apply_payload_window_hint")
        if should_apply_hint and getattr(self, "_suppress_default_cadence_hint", False):
            should_apply_hint = False
        if should_apply_hint:
            self._apply_payload_window_hint(
                source="cadence",
                lease_seconds=120,
                candidates=[30, 75, "90"],
            )
        elif (
            getattr(self, "_pending_default_cadence_hint", False)
            and hasattr(self, "_apply_payload_window_hint")
            and not getattr(self, "_suppress_default_cadence_hint", False)
        ):
            self._pending_default_cadence_hint = False
            self._apply_payload_window_hint(
                source="cadence",
                lease_seconds=120,
                candidates=[30, 75, "90"],
            )
        return tracker

    def _cleanup_ws_state(self) -> None:
        """Remove cached websocket state and tracker entries for this device."""

        try:
            runtime = require_runtime(self.hass, self.entry_id)
        except LookupError:
            return

        ws_bucket = runtime.ws_state
        if isinstance(ws_bucket, MutableMapping):
            ws_bucket.pop(self.dev_id, None)
        trackers = runtime.ws_trackers
        if isinstance(trackers, MutableMapping):
            trackers.pop(self.dev_id, None)

        setattr(self, "_ws_state", None)
        setattr(self, "_ws_tracker", None)
        setattr(self, "_ws_bucket_baseline", None)

    def _notify_ws_status(
        self,
        tracker: WsHealthTracker,
        *,
        reason: str,
        health_changed: bool = False,
        payload_changed: bool = False,
    ) -> None:
        """Dispatch websocket status updates with tracker metadata."""

        dispatcher = getattr(self, "_dispatcher_mock", None)
        if dispatcher is None:
            dispatcher = async_dispatcher_send

        payload: dict[str, Any] = {
            "dev_id": self.dev_id,
            "status": tracker.status,
            "reason": reason,
        }
        if health_changed:
            payload["health_changed"] = True
        if payload_changed:
            payload["payload_changed"] = True
        payload["payload_stale"] = tracker.payload_stale

        dispatcher(self.hass, signal_ws_status(self.entry_id), payload)

    def _sync_gateway_connection_state(self, *, now: float | None = None) -> None:
        """Update gateway connection state in the domain store when available."""

        coordinator = getattr(self, "_coordinator", None)
        updater = getattr(coordinator, "update_gateway_connection", None)
        if not callable(updater):
            return

        tracker = self._ws_health_tracker()
        ws_state = self._ws_state_bucket()
        last_event_at = ws_state.get("last_event_at")
        if not isinstance(last_event_at, (int, float)):
            last_event_at = None

        idle_restart_pending = ws_state.get("idle_restart_pending")
        if idle_restart_pending is not None:
            idle_restart_pending = bool(idle_restart_pending)

        now_ts = now if isinstance(now, (int, float)) else time.time()
        updater(
            status=tracker.status,
            connected=tracker.status in {"healthy", "connected"},
            last_event_at=last_event_at,
            healthy_since=tracker.healthy_since,
            healthy_minutes=tracker.healthy_minutes(now=now_ts),
            last_payload_at=tracker.last_payload_at,
            last_heartbeat_at=tracker.last_heartbeat_at,
            payload_stale=tracker.payload_stale,
            payload_stale_after=tracker.payload_stale_after,
            idle_restart_pending=idle_restart_pending,
        )

    def _mark_ws_payload(
        self,
        *,
        timestamp: float | None = None,
        stale_after: float | None = None,
        reason: str = "payload",
    ) -> None:
        """Update tracker payload timestamps and emit changes if required."""

        tracker = self._ws_health_tracker()
        changed = tracker.mark_payload(timestamp=timestamp, stale_after=stale_after)
        setattr(self, "_last_payload_at", tracker.last_payload_at)
        state = self._ws_state_bucket()
        state["last_payload_at"] = tracker.last_payload_at
        state["last_heartbeat_at"] = tracker.last_heartbeat_at
        state["payload_stale"] = tracker.payload_stale
        state["payload_stale_after"] = tracker.payload_stale_after
        if changed:
            self._notify_ws_status(
                tracker,
                reason=reason,
                payload_changed=True,
            )
        self._sync_gateway_connection_state(now=timestamp)

    def _mark_ws_heartbeat(
        self,
        *,
        timestamp: float | None = None,
        reason: str = "heartbeat",
    ) -> None:
        """Record a websocket heartbeat and emit staleness changes."""

        tracker = self._ws_health_tracker()
        changed = tracker.mark_heartbeat(timestamp=timestamp)
        state = self._ws_state_bucket()
        state["last_heartbeat_at"] = tracker.last_heartbeat_at
        state["payload_stale"] = tracker.payload_stale
        if changed:
            self._notify_ws_status(
                tracker,
                reason=reason,
                payload_changed=True,
            )
        self._sync_gateway_connection_state(now=timestamp)

    def _refresh_ws_payload_state(
        self,
        *,
        now: float | None = None,
        reason: str = "refresh",
    ) -> None:
        """Re-evaluate payload staleness and emit notifications if it changed."""

        tracker = self._ws_health_tracker()
        changed = tracker.refresh_payload_state(now=now)
        state = self._ws_state_bucket()
        state["payload_stale"] = tracker.payload_stale
        if changed:
            self._notify_ws_status(
                tracker,
                reason=reason,
                payload_changed=True,
            )
        self._sync_gateway_connection_state(now=now)

    def _update_status(self, status: str) -> None:
        """Publish websocket status updates to Home Assistant listeners."""

        tracker = self._ws_health_tracker()
        now = time.time()

        stats = getattr(self, "_stats", None)
        last_event_ts = getattr(stats, "last_event_ts", None) if stats else None
        last_event_at = getattr(self, "_last_event_at", None)

        healthy_since = tracker.healthy_since
        reset_health = False
        if status == "healthy" and healthy_since is None:
            candidate = last_event_at or last_event_ts or now
            healthy_since = candidate
        elif self._status_should_reset_health(status):
            healthy_since = None
            reset_health = True

        status_changed, health_changed = tracker.update_status(
            status,
            healthy_since=healthy_since,
            timestamp=now,
            reset_health=reset_health,
        )

        setattr(self, "_status", tracker.status)
        setattr(self, "_healthy_since", tracker.healthy_since)

        payload_changed = tracker.refresh_payload_state(now=now)

        state = self._ws_state_bucket()
        state["status"] = tracker.status
        state["last_event_at"] = last_event_ts or last_event_at or None
        state["healthy_since"] = tracker.healthy_since
        state["healthy_minutes"] = tracker.healthy_minutes(now=now)
        state["frames_total"] = getattr(stats, "frames_total", 0) if stats else 0
        state["events_total"] = getattr(stats, "events_total", 0) if stats else 0
        state["last_payload_at"] = tracker.last_payload_at
        state["last_heartbeat_at"] = tracker.last_heartbeat_at
        state["payload_stale"] = tracker.payload_stale

        if (
            status_changed
            or health_changed
            or payload_changed
            or status in {"healthy", "connected"}
        ):
            self._notify_ws_status(
                tracker,
                reason="status",
                health_changed=health_changed,
                payload_changed=payload_changed,
            )
        self._sync_gateway_connection_state(now=now)


class _WSCommon(_WSStatusMixin):
    """Shared helpers for websocket clients."""

    hass: HomeAssistant
    entry_id: str
    dev_id: str
    _coordinator: Any

    def __init__(self, *, inventory: Inventory | None = None) -> None:
        """Initialise shared websocket state."""

        self._inventory: Inventory | None = inventory
        self._connect_limiter = ConnectionRateLimiter()

    async def _throttle_connection_attempt(self) -> None:
        """Apply a defensive rate limit before dialing the backend."""

        limiter = getattr(self, "_connect_limiter", None)
        wait = getattr(limiter, "wait_for_slot", None)
        if callable(wait):
            await wait()

    def _ensure_type_bucket(
        self,
        nodes_by_type: Mapping[str, typing.Any] | MutableMapping[str, typing.Any],
        node_type: str,
        *,
        dev_map: MutableMapping[str, typing.Any] | None = None,
    ) -> Mapping[str, typing.Any]:
        """Return the node bucket for ``node_type`` without cloning metadata."""

        if not isinstance(nodes_by_type, Mapping):
            return None

        normalized_type = normalize_node_type(node_type)
        if not normalized_type:
            return None

        existing = nodes_by_type.get(normalized_type)
        if isinstance(existing, Mapping):
            bucket: Mapping[str, typing.Any] = existing
        else:
            mutable: dict[str, Any] = {}
            bucket = mutable
            if isinstance(nodes_by_type, MutableMapping):
                nodes_by_type[normalized_type] = mutable

        if not isinstance(dev_map, MutableMapping):
            return bucket

        if isinstance(self._inventory, Inventory):
            dev_map["inventory"] = self._inventory

        settings_section = dev_map.get("settings")
        if isinstance(settings_section, MutableMapping):
            settings_section.setdefault(normalized_type, {})
        elif settings_section is None or "settings" not in dev_map:
            dev_map["settings"] = {normalized_type: {}}

        return bucket


class WebSocketClient(_WsLeaseMixin, _WSStatusMixin):
    """Delegate to the correct backend websocket client."""

    def __init__(
        self,
        hass: HomeAssistant,
        *,
        entry_id: str,
        dev_id: str,
        api_client: RESTClient,
        coordinator: Any,
        session: aiohttp.ClientSession | None = None,
        protocol: str | None = None,
        namespace: str = WS_NAMESPACE,
        inventory: Inventory | None = None,
    ) -> None:
        """Initialise a websocket client wrapper for the active backend."""
        _WsLeaseMixin.__init__(self)
        self.hass = hass
        self.entry_id = entry_id
        self.dev_id = dev_id
        self._client = api_client
        self._coordinator = coordinator
        self._session = session or getattr(api_client, "_session", None)
        self._protocol_hint = protocol
        self._namespace = namespace or WS_NAMESPACE
        self._loop = getattr(hass, "loop", None) or asyncio.get_event_loop()
        self._delegate: DucaheatWSClient | TermoWebWSClient | None = None
        self._brand = (
            BRAND_DUCAHEAT
            if getattr(api_client, "_is_ducaheat", False)
            else BRAND_TERMOWEB
        )
        self._inventory = inventory

    def start(self) -> asyncio.Task:
        """Start the backend-specific websocket client."""
        if self._delegate is not None:
            return self._delegate.start()
        if self._brand == BRAND_DUCAHEAT:
            from .ducaheat_ws import DucaheatWSClient  # noqa: PLC0415

            self._delegate = DucaheatWSClient(
                self.hass,
                entry_id=self.entry_id,
                dev_id=self.dev_id,
                api_client=self._client,
                coordinator=self._coordinator,
                session=self._session,
                namespace=DUCAHEAT_NAMESPACE,
                inventory=self._inventory,
            )
        else:
            from .termoweb_ws import TermoWebWSClient  # noqa: PLC0415

            self._delegate = TermoWebWSClient(
                self.hass,
                entry_id=self.entry_id,
                dev_id=self.dev_id,
                api_client=self._client,
                coordinator=self._coordinator,
                session=self._session,
                namespace=self._namespace,
                inventory=self._inventory,
            )
        return self._delegate.start()

    async def stop(self) -> None:
        """Stop the backend-specific websocket client."""
        if self._delegate is not None:
            await self._delegate.stop()

    def is_running(self) -> bool:
        """Return True when the backend websocket client is running."""
        return bool(self._delegate and self._delegate.is_running())

    async def ws_url(self) -> str:
        """Return the active websocket URL when available."""
        if self._delegate and hasattr(self._delegate, "ws_url"):
            return await self._delegate.ws_url()
        return ""


__all__ = [
    "CANONICAL_SETTING_KEYS",
    "DUCAHEAT_NAMESPACE",
    "ConnectionRateLimiter",
    "HandshakeError",
    "WSStats",
    "WebSocketClient",
    "WsHealthTracker",
    "build_settings_delta",
    "clone_payload_value",
    "forward_ws_sample_updates",
    "resolve_ws_update_section",
    "translate_path_update",
]


def __getattr__(name: str) -> Any:
    """Lazily expose backend websocket client implementations."""

    if name == "DucaheatWSClient":
        from .ducaheat_ws import DucaheatWSClient as _DucaheatWSClient  # noqa: PLC0415

        return _DucaheatWSClient
    if name == "TermoWebWSClient":
        from .termoweb_ws import TermoWebWSClient as _TermoWebWSClient  # noqa: PLC0415

        return _TermoWebWSClient
    raise AttributeError(name)
