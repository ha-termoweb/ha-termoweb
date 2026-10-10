"""Shared websocket helpers."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import (
    Awaitable,
    Callable,
    Collection,
    Iterable,
    Mapping,
)
from dataclasses import dataclass
import logging
import time
import typing
from typing import Any

import aiohttp
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_send

from custom_components.termoweb.const import (
    ACCEPT_LANGUAGE,
    BRAND_TERMOWEB,
    DOMAIN,
    USER_AGENT,
    get_brand_requested_with,
    get_brand_user_agent,
    signal_ws_status,
)
from custom_components.termoweb.domain import (
    NodeId as DomainNodeId,
    NodeSettingsDelta,
    NodeType as DomainNodeType,
    canonicalize_settings_payload,
)
from custom_components.termoweb.inventory import (
    Inventory,
    normalize_node_addr,
    normalize_node_type,
)
from custom_components.termoweb.runtime import require_runtime

from .ws_health import WsHealthTracker

_LOGGER = logging.getLogger(__name__)

_LOGGED_NON_CANONICAL: set[frozenset[str]] = set()

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
    "lock",
    "priority",
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
    return value


def build_settings_delta(section: str, payload: Any) -> dict[str, Any]:
    """Extract canonical settings keys from a websocket section payload."""

    if section == "prog":
        return {"prog": clone_payload_value(payload)} if payload is not None else {}
    if section == "prog_temps":
        return {"ptemp": clone_payload_value(payload)} if payload is not None else {}
    extra_ws_keys = set(payload.keys()) - set(CANONICAL_SETTING_KEYS)
    if extra_ws_keys:
        frozen = frozenset(extra_ws_keys)
        if frozen not in _LOGGED_NON_CANONICAL:
            _LOGGED_NON_CANONICAL.add(frozen)
            _LOGGER.debug(
                "WS payload has non-canonical keys: %s",
                sorted(extra_ws_keys),
            )
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
        if canonical_type not in allowed_types:
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
    except Exception:  # one bad frame must not kill the read loop
        active_logger.exception("%s: forwarding heater samples failed", log_prefix)


def translate_path_update(
    payload: Any,
    *,
    resolve_section: Callable[
        [str | None], tuple[str | None, str | None]
    ] = resolve_ws_update_section,
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


_WS_CONNECT_TIMEOUT = 15.0
_WS_CLOSE_TIMEOUT = 10.0
_NODE_METADATA_KEYS = frozenset(
    {"type", "node_type", "addr", "address", "name", "title", "label"}
)
_NODE_TYPE_LEVEL_KEYS = ("lease_seconds", "cadence_seconds", "poll_seconds")
_BACKOFF_SEQ: tuple[float, ...] = (5, 10, 30, 120, 300)


@dataclass
class WSStats:
    """Track websocket frame and event stats."""

    frames_total: int = 0
    events_total: int = 0
    last_event_ts: float = 0.0


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

        runtime = require_runtime(self.hass, self.entry_id)
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

        ws_state = runtime.ws_state.setdefault(self.dev_id, {})
        setattr(self, "_ws_state", ws_state)
        self._ws_bucket_sizes()
        return ws_state

    def _ws_health_tracker(self) -> WsHealthTracker:
        """Return the :class:`WsHealthTracker` for this websocket client."""

        cached = getattr(self, "_ws_tracker", None)
        if isinstance(cached, WsHealthTracker):
            return cached

        try:
            runtime = require_runtime(self.hass, self.entry_id)
        except LookupError:
            tracker = WsHealthTracker(self.dev_id)
            setattr(self, "_ws_tracker", tracker)
            return tracker

        trackers = runtime.ws_trackers
        tracker = trackers.get(self.dev_id)
        if not isinstance(tracker, WsHealthTracker):
            tracker = WsHealthTracker(self.dev_id)
            trackers[self.dev_id] = tracker
        setattr(self, "_ws_tracker", tracker)
        self._ws_bucket_sizes()
        return tracker

    def _cleanup_ws_state(self) -> None:
        """Remove cached websocket state and tracker entries for this device."""

        try:
            runtime = require_runtime(self.hass, self.entry_id)
        except LookupError:
            return

        runtime.ws_state.pop(self.dev_id, None)
        runtime.ws_trackers.pop(self.dev_id, None)

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

        async_dispatcher_send(self.hass, signal_ws_status(self.entry_id), payload)

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

        if not (status_changed or health_changed or payload_changed):
            return
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
    _client: Any
    _loop: asyncio.AbstractEventLoop
    _session: aiohttp.ClientSession
    _task: asyncio.Task | None
    _runner: Callable[[], Awaitable[None]]
    _brand: str = BRAND_TERMOWEB

    def __init__(self, *, inventory: Inventory) -> None:
        """Initialise shared websocket state."""

        self._inventory: Inventory = inventory
        self._connect_limiter = ConnectionRateLimiter()
        self._payload_idle_window: float = 240.0
        self._subscription_refresh_lock = asyncio.Lock()
        self._backoff_idx = 0
        self._unknown_nodes_logged: set[tuple[str, str]] = set()

    def _prepare_start(self) -> None:
        """Reset per-run state before the runner task is created."""

    def start(self) -> asyncio.Task:
        """Start the websocket client background task."""

        if self._task and not self._task.done():
            return self._task
        self._prepare_start()
        self._task = self._loop.create_task(
            self._runner(), name=f"{DOMAIN}-ws-{self.dev_id}"
        )
        return self._task

    def _reset_backoff(self) -> None:
        """Restart the reconnect backoff sequence."""

        self._backoff_idx = 0

    def _next_backoff(self) -> float:
        """Return the next reconnect delay and advance the sequence."""

        idx = min(self._backoff_idx, len(_BACKOFF_SEQ) - 1)
        self._backoff_idx = idx + 1 if idx < len(_BACKOFF_SEQ) - 1 else idx
        return _BACKOFF_SEQ[idx]

    async def _throttle_connection_attempt(self) -> None:
        """Apply a defensive rate limit before dialing the backend."""

        await self._connect_limiter.wait_for_slot()

    def _session_received_payload(self, started_at: float) -> bool:
        """Return True when a payload arrived since ``started_at``."""

        last_payload = self._ws_health_tracker().last_payload_at
        return last_payload is not None and last_payload >= started_at

    def _brand_headers(self, *, origin: str | None = None) -> dict[str, str]:
        """Return baseline headers aligned with the client brand."""

        headers = {
            "User-Agent": get_brand_user_agent(self._brand) or USER_AGENT,
            "Accept-Language": ACCEPT_LANGUAGE,
        }
        requested_with = get_brand_requested_with(self._brand)
        if requested_with:
            headers["X-Requested-With"] = requested_with
        if origin:
            headers["Origin"] = origin
        return headers

    async def _get_token(self) -> str:
        """Reuse the REST client token for websocket authentication."""

        headers = await self._client.authed_headers()
        auth_header = (
            headers.get("Authorization") if isinstance(headers, dict) else None
        )
        token = (
            auth_header.partition(" ")[2].strip()
            if isinstance(auth_header, str)
            else ""
        )
        if not token:
            raise RuntimeError("authorization token missing")
        return token

    async def _open_websocket(
        self, url: str, headers: Mapping[str, str]
    ) -> aiohttp.ClientWebSocketResponse:
        """Open the websocket transport with the shared timeouts."""

        async with asyncio.timeout(_WS_CONNECT_TIMEOUT):
            return await self._session.ws_connect(
                url,
                timeout=aiohttp.ClientWSTimeout(ws_close=_WS_CLOSE_TIMEOUT),
                heartbeat=None,
                autoclose=False,
                headers=headers,
            )

    def _normalise_nodes(self, nodes: Mapping[str, typing.Any]) -> Any:
        """Normalise websocket node payloads via the REST client codec."""

        snapshot: Any = clone_payload_value(nodes)
        try:
            resolved = self._client.normalise_ws_nodes(snapshot)
        except Exception:
            _LOGGER.debug("WS: normalise_ws_nodes failed", exc_info=True)
            return snapshot
        if isinstance(resolved, Mapping) and not isinstance(resolved, dict):
            return dict(resolved)
        return resolved

    def _coerce_nodes_list(self, nodes: Any) -> dict[str, Any] | None:
        """Convert list-style node snapshots into ``{type: {section: {addr: v}}}``."""

        if nodes is None or isinstance(nodes, (Mapping, str, bytes, bytearray)):
            return None
        if not isinstance(nodes, Iterable):
            return None

        snapshot: dict[str, Any] = {}
        for node_type, addr, entry in self._inventory.iter_known_entries(nodes):
            type_bucket = snapshot.setdefault(node_type, {})
            for key in _NODE_TYPE_LEVEL_KEYS:
                if key in entry and key not in type_bucket:
                    type_bucket[key] = entry[key]
            for key, value in entry.items():
                if (
                    not isinstance(key, str)
                    or key in _NODE_METADATA_KEYS
                    or key in _NODE_TYPE_LEVEL_KEYS
                ):
                    continue
                section, nested_key = resolve_ws_update_section(key)
                if section is None:
                    continue
                section_bucket = type_bucket.setdefault(section, {})
                if nested_key:
                    existing = section_bucket.get(addr)
                    merged = dict(existing) if isinstance(existing, Mapping) else {}
                    merged[nested_key] = clone_payload_value(value)
                    section_bucket[addr] = merged
                else:
                    section_bucket[addr] = clone_payload_value(value)
        return snapshot or None

    def _translate_path_update(self, payload: Any) -> dict[str, Any] | None:
        """Translate ``{"path": ..., "body": ...}`` frames into nodes."""

        return translate_path_update(payload)

    def _nodes_to_deltas(
        self, nodes: Mapping[str, typing.Any]
    ) -> list[NodeSettingsDelta]:
        """Convert websocket node payloads into domain delta objects."""

        deltas: list[NodeSettingsDelta] = []
        for raw_type, sections in nodes.items():
            if not isinstance(raw_type, str) or not isinstance(sections, Mapping):
                continue
            node_type = DomainNodeType.coerce(raw_type)
            if node_type is None:
                continue

            per_addr: dict[str, dict[str, Any]] = {}
            for section, section_payload in sections.items():
                if not isinstance(section, str):
                    continue
                if section == "samples" or not isinstance(section_payload, Mapping):
                    continue
                for raw_addr, payload in section_payload.items():
                    addr = normalize_node_addr(raw_addr, use_default_when_falsey=True)
                    if not addr:
                        continue
                    bucket = per_addr.setdefault(addr, {})
                    settings_delta: Mapping[str, typing.Any] = {}
                    if section == "status" and isinstance(payload, Mapping):
                        settings_delta = canonicalize_settings_payload(
                            {"status": payload}
                        )
                    elif section == "capabilities":
                        continue
                    else:
                        settings_delta = build_settings_delta(section, payload)
                    if settings_delta:
                        bucket.update(settings_delta)

            for addr, payload in per_addr.items():
                node_id = DomainNodeId(node_type, addr)
                if not self._inventory.has_node(node_id.node_type.value, node_id.addr):
                    # Inventory is immutable, so a node unknown once stays unknown:
                    # log it a single time instead of on every frame.
                    key = (node_type.value, addr)
                    if key not in self._unknown_nodes_logged:
                        self._unknown_nodes_logged.add(key)
                        _LOGGER.debug(
                            "WS: ignoring updates for unknown node_type=%s addr=%s",
                            node_type.value,
                            addr,
                        )
                    continue
                deltas.append(NodeSettingsDelta(node_id=node_id, changes=payload))

        return deltas

    def _apply_deltas_to_store(
        self,
        deltas: Iterable[NodeSettingsDelta],
        *,
        replace: bool,
    ) -> None:
        """Apply deltas to the domain store via the coordinator."""

        coordinator = getattr(self, "_coordinator", None)
        handler = getattr(coordinator, "handle_ws_deltas", None)
        if not callable(handler):
            return
        try:
            handler(self.dev_id, tuple(deltas), replace=replace)
        except Exception:  # one bad frame must not kill the read loop
            _LOGGER.exception("WS: failed to apply websocket deltas")

    def _collect_sample_updates(
        self,
        nodes: Mapping[str, typing.Any],
        *,
        allowed_types: Collection[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Extract energy sample updates from a websocket node payload."""

        allowed: set[str] | None = None
        if allowed_types is not None:
            allowed = {
                normalized
                for candidate in allowed_types
                if (
                    normalized := normalize_node_type(
                        candidate, use_default_when_falsey=True
                    )
                )
            }

        updates: dict[str, dict[str, Any]] = {}
        for node_type, type_payload in nodes.items():
            if not isinstance(node_type, str) or not isinstance(type_payload, Mapping):
                continue
            canonical_type = normalize_node_type(
                node_type,
                use_default_when_falsey=True,
            )
            if not canonical_type:
                continue
            if allowed is not None:
                if canonical_type not in allowed:
                    continue
            elif canonical_type == "thm":
                continue
            samples = type_payload.get("samples")
            if not isinstance(samples, Mapping):
                continue
            bucket: dict[str, Any] = {}
            for addr, payload in samples.items():
                normalised_addr = normalize_node_addr(addr)
                if not normalised_addr:
                    continue
                bucket[normalised_addr] = payload
            lease_seconds = type_payload.get("lease_seconds")
            if bucket or lease_seconds is not None:
                updates[node_type] = {
                    "samples": bucket,
                    "lease_seconds": lease_seconds,
                }
        return updates

    def _forward_sample_updates(
        self, updates: Mapping[str, Mapping[str, typing.Any]]
    ) -> None:
        """Relay websocket heater sample updates to the energy coordinator."""

        forward_ws_sample_updates(
            self.hass,
            self.entry_id,
            self.dev_id,
            updates,
            logger=_LOGGER,
            log_prefix="WS",
        )


__all__ = [
    "CANONICAL_SETTING_KEYS",
    "ConnectionRateLimiter",
    "HandshakeError",
    "WSStats",
    "WsHealthTracker",
    "build_settings_delta",
    "clone_payload_value",
    "forward_ws_sample_updates",
    "resolve_ws_update_section",
    "translate_path_update",
]
