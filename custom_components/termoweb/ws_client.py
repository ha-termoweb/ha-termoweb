"""Unified websocket client for TermoWeb backends."""

from __future__ import annotations

import asyncio
import codecs
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
import json
import logging
import random
import time
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlencode, urlsplit, urlunsplit

import aiohttp
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_send
import socketio

from .api import RESTClient
from .const import API_BASE, DOMAIN, WS_NAMESPACE, signal_ws_data, signal_ws_status
from .nodes import (
    NODE_CLASS_BY_TYPE,
    addresses_by_node_type,
    build_node_inventory as _build_node_inventory,
    collect_heater_sample_addresses,
    ensure_node_inventory,
    heater_sample_subscription_targets,
    normalize_heater_addresses,
    normalize_node_addr,
)

_LOGGER = logging.getLogger(__name__)

build_node_inventory = _build_node_inventory  # re-exported for tests


_DEFAULT_SUBSCRIPTION_TTL = 300.0


@dataclass
class WSStats:
    """Track websocket activity statistics."""

    frames_total: int = 0
    events_total: int = 0
    last_event_ts: float = 0.0
    last_paths: list[str] | None = None


class HandshakeError(RuntimeError):
    """Capture context for failed websocket handshakes."""

    def __init__(self, status: int, url: str, body_snippet: str) -> None:
        """Initialise the error with the HTTP response details."""

        super().__init__(f"handshake failed (status={status})")
        self.status = status
        self.url = url
        self.body_snippet = body_snippet


class WebSocketClient:
    """Unified websocket client wrapping ``socketio.AsyncClient``."""

    def __init__(
        self,
        hass: HomeAssistant,
        *,
        entry_id: str,
        dev_id: str,
        api_client: RESTClient,
        coordinator: Any,
        session: aiohttp.ClientSession | None = None,
        handshake_fail_threshold: int = 5,  # legacy compatibility
        protocol: str | None = None,
    ) -> None:
        """Initialise the websocket client container."""
        self.hass = hass
        self.entry_id = entry_id
        self.dev_id = dev_id
        self._client = api_client
        self._coordinator = coordinator
        self._session = session or getattr(api_client, "_session", None)
        self._protocol_hint = protocol
        self._loop = getattr(hass, "loop", None) or asyncio.get_event_loop()
        self._task: asyncio.Task | None = None

        self._sio = socketio.AsyncClient(
            reconnection=False,
            logger=_LOGGER.getChild(f"socketio.{dev_id}"),
            engineio_logger=_LOGGER.getChild(f"engineio.{dev_id}"),
        )
        self._sio.start_background_task = self._wrap_background_task
        http_target: Any = self._session
        if http_target is None or not hasattr(http_target, "closed"):
            http_target = SimpleNamespace(closed=True)
        try:
            self._sio.http = http_target
        except AttributeError:
            setattr(self._sio, "http", http_target)
        if hasattr(self._sio, "eio"):
            self._sio.eio.start_background_task = self._wrap_background_task
            eio_http = getattr(self._sio.eio, "http", None)
            if eio_http is None or not hasattr(eio_http, "closed"):
                self._sio.eio.http = http_target
            else:
                self._sio.eio.http = eio_http

        self._sio.on("connect", handler=self._on_connect)
        self._sio.on("disconnect", handler=self._on_disconnect)
        self._sio.on("reconnect", handler=self._on_reconnect)
        self._sio.on("connect_error", handler=self._on_connect_error)
        self._sio.on("error", handler=self._on_error)
        self._sio.on("reconnect_failed", handler=self._on_reconnect_failed)
        self._sio.on(
            "disconnect", namespace=WS_NAMESPACE, handler=self._on_namespace_disconnect
        )
        self._sio.on(
            "dev_handshake", namespace=WS_NAMESPACE, handler=self._on_dev_handshake
        )
        self._sio.on("dev_data", namespace=WS_NAMESPACE, handler=self._on_dev_data)
        self._sio.on("update", namespace=WS_NAMESPACE, handler=self._on_update)

        self._closing = False
        self._status: str = "stopped"
        self._stop_event = asyncio.Event()
        self._disconnected = asyncio.Event()
        self._disconnected.set()
        self._backoff_seq = [5, 10, 30, 120, 300]
        self._backoff_idx = 0

        self._connected_since: float | None = None
        self._healthy_since: float | None = None
        self._last_event_at: float | None = None
        self._stats = WSStats()

        self._handshake_payload: dict[str, Any] | None = None
        self._nodes: dict[str, Any] = {}
        self._nodes_raw: dict[str, Any] = {}

        self._payload_idle_window: float = 3600.0
        self._idle_restart_task: asyncio.Task | None = None
        self._idle_restart_pending = False
        self._idle_monitor_task: asyncio.Task | None = None

        self._init_subscription_state()
        self._handshake_logged = False

        self._ws_state: dict[str, Any] | None = None
        self._ws_state_bucket()

    # ------------------------------------------------------------------
    # Public control
    # ------------------------------------------------------------------
    def _wrap_background_task(self, target: Any, *args: Any, **kwargs: Any) -> asyncio.Task:
        """Schedule socket.io background tasks on the HA loop."""
        coro = target(*args, **kwargs)
        if not asyncio.iscoroutine(coro):
            async def _runner() -> Any:
                return coro

            coro = _runner()
        return self._loop.create_task(coro)

    def start(self) -> asyncio.Task:
        """Start the websocket client background task."""
        if self._task and not self._task.done():
            return self._task
        _LOGGER.debug("WS %s: start requested", self.dev_id)
        self._closing = False
        self._stop_event = asyncio.Event()
        self._handshake_logged = False
        self._subscription_refresh_failed = False
        self._subscription_refresh_due = None
        self._task = self._loop.create_task(
            self._runner(), name=f"{DOMAIN}-ws-{self.dev_id}"
        )
        return self._task

    async def stop(self) -> None:
        """Cancel tasks and close websocket sessions."""
        _LOGGER.debug("WS %s: stop requested", self.dev_id)
        self._closing = True
        self._stop_event.set()
        if self._idle_restart_task:
            self._idle_restart_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._idle_restart_task
            self._idle_restart_task = None
        self._idle_restart_pending = False
        if self._idle_monitor_task:
            self._idle_monitor_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._idle_monitor_task
            self._idle_monitor_task = None
        await self._cancel_subscription_refresh()
        self._subscription_refresh_failed = False
        self._subscription_refresh_due = None
        await self._disconnect(reason="client stop")
        if self._task:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self._update_status("stopped")

    def is_running(self) -> bool:
        """Return True if the websocket client task is active."""
        return bool(self._task and not self._task.done())

    async def ws_url(self) -> str:
        """Return the websocket URL using the API client's token helper."""
        token = await self._get_token()
        base = self._api_base().rstrip("/")
        if not base.endswith("/api/v2"):
            base = f"{base}/api/v2"
        return f"{base}/socket_io?token={token}&dev_id={self.dev_id}"

    # ------------------------------------------------------------------
    # Core loop and protocol dispatch
    # ------------------------------------------------------------------
    async def _runner(self) -> None:
        """Manage connection attempts with backoff."""
        self._update_status("starting")
        try:
            while not self._closing:
                try:
                    await self._connect_once()
                    await self._wait_for_events()
                except asyncio.CancelledError:
                    raise
                except Exception as err:  # noqa: BLE001
                    _LOGGER.info(
                        "WS %s: connection error (%s: %s); will retry",
                        self.dev_id,
                        type(err).__name__,
                        err,
                    )
                    _LOGGER.debug(
                        "WS %s: connection error details", self.dev_id, exc_info=True
                    )
                finally:
                    await self._disconnect(reason="loop cleanup")
                    if not self._closing:
                        self._update_status("disconnected")
                if self._closing:
                    break
                delay = self._backoff_seq[
                    min(self._backoff_idx, len(self._backoff_seq) - 1)
                ]
                self._backoff_idx = min(self._backoff_idx + 1, len(self._backoff_seq) - 1)
                await asyncio.sleep(delay * random.uniform(0.8, 1.2))
        finally:
            self._update_status("stopped")

    async def _connect_once(self) -> None:
        """Open the Socket.IO connection."""
        if self._stop_event.is_set():
            return
        url, engineio_path = await self._build_engineio_target()
        _LOGGER.debug(
            "WS %s: connecting to %s (path=%s)", self.dev_id, url, engineio_path
        )
        self._disconnected.clear()
        self._backoff_idx = 0
        await self._sio.connect(
            url,
            transports=["websocket"],
            namespaces=[WS_NAMESPACE],
            socketio_path=engineio_path,
            wait=True,
            wait_timeout=15,
        )

    async def _wait_for_events(self) -> None:
        """Wait for the connection to close or stop to be requested."""
        stop_task = self._loop.create_task(self._stop_event.wait())
        disconnect_task = self._loop.create_task(self._disconnected.wait())
        try:
            done, pending = await asyncio.wait(
                [stop_task, disconnect_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            for task in done:
                with suppress(asyncio.CancelledError):
                    await task
        finally:
            stop_task.cancel()
            disconnect_task.cancel()
            with suppress(asyncio.CancelledError):
                await stop_task
            with suppress(asyncio.CancelledError):
                await disconnect_task

    async def _disconnect(self, *, reason: str) -> None:
        """Ensure the AsyncClient is disconnected."""
        if self._sio.connected:
            try:
                await self._sio.disconnect()
            except Exception:  # noqa: BLE001
                _LOGGER.debug(
                    "WS %s: disconnect due to %s failed", self.dev_id, reason, exc_info=True
                )
        self._disconnected.set()

    async def _build_engineio_target(self) -> tuple[str, str]:
        """Return the Engine.IO base URL and path."""
        token = await self._get_token()
        base = self._api_base().rstrip("/")
        parsed = urlsplit(base if base else API_BASE)
        scheme = parsed.scheme or "https"
        netloc = parsed.netloc or parsed.path
        if not netloc:
            raise RuntimeError("invalid API base")
        path = parsed.path.rstrip("/")
        if not path.endswith("/api/v2"):
            path = f"{path}/api/v2" if path else "/api/v2"
        socket_path = f"{path}/socket_io"
        engineio_path = socket_path.strip("/")
        query = urlencode({"token": token, "dev_id": self.dev_id})
        url = urlunsplit((scheme, netloc, socket_path, query, ""))
        return url, engineio_path

    # ------------------------------------------------------------------
    # Socket.IO event handlers
    # ------------------------------------------------------------------
    async def _on_connect(self) -> None:
        """Handle socket connection establishment."""
        _LOGGER.debug("WS %s: connected", self.dev_id)
        now = time.time()
        self._connected_since = now
        self._healthy_since = None
        self._last_event_at = now
        self._stats.frames_total = 0
        self._stats.events_total = 0
        self._handshake_logged = False
        self._subscription_refresh_failed = False
        self._update_status("connected")
        if self._idle_monitor_task is None or self._idle_monitor_task.done():
            self._idle_monitor_task = self._loop.create_task(self._idle_monitor())
        try:
            await self._sio.emit("join", namespace=WS_NAMESPACE)
            await self._sio.emit("dev_data", namespace=WS_NAMESPACE)
        except Exception:  # noqa: BLE001
            _LOGGER.debug("WS %s: namespace join failed", self.dev_id, exc_info=True)

    async def _on_disconnect(self) -> None:
        """Handle socket disconnection."""
        _LOGGER.debug("WS %s: disconnected", self.dev_id)
        if self._idle_monitor_task:
            self._idle_monitor_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._idle_monitor_task
            self._idle_monitor_task = None
        await self._cancel_subscription_refresh()
        self._subscription_refresh_failed = False
        self._subscription_refresh_due = None
        self._handshake_logged = False
        self._disconnected.set()

    async def _on_reconnect(self) -> None:
        """Handle socket reconnection attempts."""
        _LOGGER.debug("WS %s: reconnect event", self.dev_id)
        await self._subscribe_heater_samples()

    async def _on_connect_error(self, data: Any) -> None:
        """Log ``connect_error`` events with their payload."""

        if _LOGGER.isEnabledFor(logging.DEBUG):
            _LOGGER.debug("WS %s: connect_error payload: %s", self.dev_id, data)

    async def _on_error(self, data: Any) -> None:
        """Log socket.io ``error`` events with full details."""

        if _LOGGER.isEnabledFor(logging.DEBUG):
            _LOGGER.debug("WS %s: error event payload: %s", self.dev_id, data)

    async def _on_reconnect_failed(self, data: Any | None = None) -> None:
        """Log ``reconnect_failed`` events with the reported context."""

        if _LOGGER.isEnabledFor(logging.DEBUG):
            _LOGGER.debug("WS %s: reconnect_failed details: %s", self.dev_id, data)

    async def _on_namespace_disconnect(self, reason: Any | None = None) -> None:
        """Log namespace-level disconnect callbacks with their reason."""

        if _LOGGER.isEnabledFor(logging.DEBUG):
            _LOGGER.debug(
                "WS %s: namespace disconnect (%s): %s",
                self.dev_id,
                WS_NAMESPACE,
                reason,
            )

    async def _on_dev_handshake(self, data: Any) -> None:
        """Handle the ``dev_handshake`` payload."""
        self._stats.frames_total += 1
        if not self._handshake_logged and _LOGGER.isEnabledFor(logging.DEBUG):
            _LOGGER.debug("WS %s: dev_handshake payload: %s", self.dev_id, data)
            self._handshake_logged = True
        self._handle_handshake(data)

    async def _on_dev_data(self, data: Any) -> None:
        """Handle the ``dev_data`` payload."""
        self._stats.frames_total += 1
        self._handle_dev_data(data)
        await self._subscribe_heater_samples()

    async def _on_update(self, data: Any) -> None:
        """Handle the ``update`` payload."""
        self._stats.frames_total += 1
        self._handle_update(data)

    async def _idle_monitor(self) -> None:
        """Monitor idle websocket periods and trigger restarts."""
        while not self._closing:
            await asyncio.sleep(60)
            if not self._sio.connected:
                if self._disconnected.is_set():
                    break
                continue
            last_event = self._last_event_at or self._stats.last_event_ts
            if not last_event:
                continue
            idle_for = time.time() - last_event
            if idle_for >= self._payload_idle_window:
                _LOGGER.info(
                    "WS %s: idle for %.0fs; refreshing websocket lease",
                    self.dev_id,
                    idle_for,
                )
                try:
                    await self._refresh_subscription(reason="idle monitor")
                except asyncio.CancelledError:  # pragma: no cover - task lifecycle
                    raise
                except Exception:  # noqa: BLE001
                    self._schedule_idle_restart(
                        idle_for=idle_for, source="idle monitor refresh failed"
                    )
                    break
                continue
            if self._subscription_refresh_failed:
                # Retry quickly if the last scheduled renewal failed.
                _LOGGER.info(
                    "WS %s: retrying websocket lease after failure; idle for %.0fs",
                    self.dev_id,
                    idle_for,
                )
                try:
                    await self._refresh_subscription(reason="idle monitor retry")
                except asyncio.CancelledError:  # pragma: no cover - task lifecycle
                    raise
                except Exception:  # noqa: BLE001
                    self._schedule_idle_restart(
                        idle_for=idle_for, source="idle monitor retry failed"
                    )
                    break

    def _restart_subscription_refresh(self) -> None:
        """Restart the background lease refresh task."""

        task = self._subscription_refresh_task
        if task and not task.done():
            task.cancel()
        if self._closing or self._subscription_ttl <= 0:
            self._subscription_refresh_task = None
            return
        self._subscription_refresh_task = self._loop.create_task(
            self._subscription_refresh_loop(),
            name=f"{DOMAIN}-ws-refresh-{self.dev_id}",
        )

    async def _cancel_subscription_refresh(self) -> None:
        """Cancel the subscription refresh task if it is running."""

        task = self._subscription_refresh_task
        if task and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        self._subscription_refresh_task = None

    async def _subscription_refresh_loop(self) -> None:
        """Renew the server-side subscription before the TTL expires."""

        try:
            while not self._closing:
                ttl = max(self._subscription_ttl, 60.0)
                lead = min(max(ttl * 0.2, 30.0), ttl / 2)
                wait_for = max(ttl - lead, ttl * 0.5)
                wait_for *= random.uniform(0.85, 0.95)
                wait_for = max(wait_for, 30.0)
                self._subscription_refresh_due = time.time() + wait_for
                await asyncio.sleep(wait_for)
                if self._closing:
                    break
                if not self._sio.connected:
                    continue
                try:
                    await self._refresh_subscription(reason="periodic renewal")
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001
                    self._subscription_refresh_failed = True
                    _LOGGER.warning(
                        "WS %s: subscription refresh failed; monitoring for idle",
                        self.dev_id,
                        exc_info=True,
                    )
                    await asyncio.sleep(min(60.0, ttl / 2))
        finally:
            self._subscription_refresh_due = None

    async def _refresh_subscription(self, *, reason: str) -> None:
        """Re-request device data to renew the websocket lease."""

        async with self._subscription_refresh_lock:
            if not self._sio.connected:
                raise RuntimeError("websocket not connected")
            now = time.time()
            self._subscription_refresh_last_attempt = now
            if _LOGGER.isEnabledFor(logging.INFO):
                _LOGGER.info(
                    "WS %s: refreshing websocket lease (%s)",
                    self.dev_id,
                    reason,
                )
            await self._sio.emit("dev_data", namespace=WS_NAMESPACE)
            await self._subscribe_heater_samples()
            self._subscription_refresh_failed = False
            self._subscription_refresh_last_success = time.time()
            self._subscription_refresh_due = (
                self._subscription_refresh_last_success + self._subscription_ttl
            )
            _LOGGER.info(
                "WS %s: websocket lease refreshed; next in ~%.0fs",
                self.dev_id,
                self._subscription_ttl,
            )

    def _init_subscription_state(self) -> None:
        """Initialise subscription lease tracking attributes."""

        self._subscription_ttl: float = _DEFAULT_SUBSCRIPTION_TTL
        self._subscription_refresh_task: asyncio.Task | None = None
        self._subscription_refresh_lock = asyncio.Lock()
        self._subscription_refresh_due: float | None = None
        self._subscription_refresh_last_attempt: float = 0.0
        self._subscription_refresh_last_success: float | None = None
        self._subscription_refresh_failed = False

    def _apply_subscription_ttl(
        self,
        *,
        ttl: float,
        source: str,
        context: str,
        missing_hint: str,
        now: float | None = None,
    ) -> None:
        """Store subscription TTL metadata and schedule refreshes."""

        ttl = max(float(ttl), 15.0)
        idle_window = max(ttl * 1.5, ttl + 90.0)
        self._subscription_ttl = ttl
        self._payload_idle_window = idle_window
        moment = now if now is not None else time.time()
        self._subscription_refresh_due = moment + ttl
        self._subscription_refresh_failed = False
        self._subscription_refresh_last_success = None
        self._restart_subscription_refresh()
        if source == "default":
            _LOGGER.info(
                "WS %s: %s; using %.0fs TTL",
                self.dev_id,
                missing_hint,
                ttl,
            )
        else:
            _LOGGER.info(
                "WS %s: %s lease TTL %.0fs (source=%s)",
                self.dev_id,
                context,
                ttl,
                source,
            )

    def _extract_subscription_ttl(
        self, payload: Mapping[str, Any]
    ) -> tuple[float, str] | None:
        """Extract the subscription TTL in seconds from the handshake payload."""

        candidates: list[tuple[float, str]] = []

        def _walk(node: Any, path: str) -> None:
            if isinstance(node, Mapping):
                for key, value in node.items():
                    key_str = str(key)
                    key_lower = key_str.lower()
                    next_path = f"{path}.{key_str}" if path else key_str
                    if any(
                        token in key_lower
                        for token in ("ttl", "timeout", "lease", "expire")
                    ):
                        seconds = self._coerce_seconds(value, key_lower)
                        if seconds is not None and 15.0 <= seconds <= 21600.0:
                            candidates.append((seconds, next_path))
                    if isinstance(value, (Mapping, list, tuple)):
                        _walk(value, next_path)
            elif isinstance(node, (list, tuple)):
                for idx, item in enumerate(node):
                    _walk(item, f"{path}[{idx}]")

        _walk(payload, "")
        if not candidates:
            return None
        # Prefer the smallest positive TTL to refresh conservatively.
        ttl, path = min(candidates, key=lambda item: item[0])
        return ttl, path

    @staticmethod
    def _coerce_seconds(value: Any, key: str) -> float | None:
        """Convert ``value`` to seconds based on the associated key."""

        raw: float
        if isinstance(value, (int, float)):
            raw = float(value)
        elif isinstance(value, str):
            try:
                raw = float(value.strip())
            except ValueError:
                return None
        else:
            return None
        if "ms" in key or "milli" in key or (raw > 86400 and "hour" not in key and "day" not in key):
            raw /= 1000.0
        return raw

    # ------------------------------------------------------------------
    # Payload handlers
    # ------------------------------------------------------------------
    def _handle_handshake(self, data: Any) -> None:
        """Process the initial handshake payload from the server."""
        if isinstance(data, dict):
            self._handshake_payload = deepcopy(data)
            if _LOGGER.isEnabledFor(logging.DEBUG):
                lease_scalars: list[str] = []
                lease_tokens = ("ttl", "timeout", "lease", "expire")

                def _format_scalar(value: Any) -> str | None:
                    if isinstance(value, bool):
                        return None
                    if isinstance(value, (int, float)):
                        return format(value, "g")
                    if isinstance(value, str):
                        candidate = value.strip()
                        if not candidate:
                            return None
                        try:
                            float(candidate)
                        except ValueError:
                            return None
                        return candidate
                    return None

                def _collect_scalars(node: Any, path: str) -> None:
                    if isinstance(node, Mapping):
                        for key, value in node.items():
                            key_str = str(key)
                            key_lower = key_str.lower()
                            next_path = f"{path}.{key_str}" if path else key_str
                            if any(token in key_lower for token in lease_tokens):
                                formatted = _format_scalar(value)
                                if formatted is not None:
                                    lease_scalars.append(f"{next_path}={formatted}")
                            if isinstance(value, (Mapping, list, tuple)):
                                _collect_scalars(value, next_path)
                    elif isinstance(node, (list, tuple)):
                        for idx, item in enumerate(node):
                            _collect_scalars(item, f"{path}[{idx}]")

                _collect_scalars(data, "")
                summary = ", ".join(lease_scalars) if lease_scalars else "none"
                _LOGGER.debug(
                    "WS %s: handshake lease hints: %s", self.dev_id, summary
                )
            ttl_info = self._extract_subscription_ttl(data)
            ttl: float
            source: str
            if ttl_info is None:
                ttl = _DEFAULT_SUBSCRIPTION_TTL
                source = "default"
            else:
                ttl, source = ttl_info
            self._apply_subscription_ttl(
                ttl=ttl,
                source=source,
                context="handshake",
                missing_hint="handshake did not expose lease info",
                now=time.time(),
            )
            self._update_status("connected")
        else:
            _LOGGER.debug("WS %s: invalid handshake payload", self.dev_id)

    def _handle_dev_data(self, data: Any) -> None:
        """Handle the first full snapshot of nodes from the websocket."""
        self._apply_nodes_payload(data, merge=False, event="dev_data")

    def _handle_update(self, data: Any) -> None:
        """Merge incremental node updates from the websocket feed."""
        self._apply_nodes_payload(data, merge=True, event="update")

    def _apply_nodes_payload(self, payload: Any, *, merge: bool, event: str) -> None:
        """Update cached nodes from the websocket payload and notify listeners."""
        nodes = self._extract_nodes(payload)
        if nodes is None:
            _LOGGER.debug("WS %s: %s without nodes", self.dev_id, event)
            return
        normaliser = getattr(self._client, "normalise_ws_nodes", None)
        if callable(normaliser):
            try:
                nodes = normaliser(nodes)  # type: ignore[arg-type]
            except Exception:  # noqa: BLE001  # pragma: no cover - defensive logging only
                _LOGGER.debug(
                    "WS %s: normalise_ws_nodes failed; using raw payload", self.dev_id
                )
        if _LOGGER.isEnabledFor(logging.DEBUG):
            changed = self._collect_update_addresses(nodes)
            if merge:
                if changed:
                    _LOGGER.debug(
                        "WS %s: update event for %s",
                        self.dev_id,
                        ", ".join(
                            f"{node_type}/{addr}" for node_type, addr in changed
                        ),
                    )
                else:
                    _LOGGER.debug(
                        "WS %s: update event without address changes", self.dev_id
                    )
            else:
                _LOGGER.debug(
                    "WS %s: dev_data snapshot contains %d node groups",
                    self.dev_id,
                    len(nodes),
                )
        sample_updates: dict[str, dict[str, Any]] = {}
        for node_type, type_payload in nodes.items():
            if not isinstance(node_type, str) or not isinstance(type_payload, Mapping):
                continue
            samples = type_payload.get("samples")
            if not isinstance(samples, Mapping):
                continue
            bucket: dict[str, Any] = {}
            for addr, sample_payload in samples.items():
                normalised_addr = normalize_node_addr(addr)
                if not normalised_addr:
                    continue
                bucket[normalised_addr] = sample_payload
            if bucket:
                sample_updates[node_type] = bucket

        if merge and self._nodes_raw:
            self._merge_nodes(self._nodes_raw, nodes)
        else:
            self._nodes_raw = deepcopy(nodes)
        self._nodes = self._build_nodes_snapshot(self._nodes_raw)
        self._dispatch_nodes(self._nodes)
        if sample_updates:
            self._forward_sample_updates(sample_updates)
        self._mark_event(paths=None, count_event=True)

    def _forward_sample_updates(
        self, updates: Mapping[str, Mapping[str, Any]]
    ) -> None:
        """Relay websocket heater sample updates to the energy coordinator."""

        record = self.hass.data.get(DOMAIN, {}).get(self.entry_id)
        if not isinstance(record, Mapping):
            return
        energy_coordinator = record.get("energy_coordinator")
        handler = getattr(energy_coordinator, "handle_ws_samples", None)
        if not callable(handler):
            return
        try:
            handler(
                self.dev_id,
                {node_type: dict(section) for node_type, section in updates.items()},
                lease_seconds=self._subscription_ttl,
            )
        except Exception:  # noqa: BLE001  # pragma: no cover - defensive logging
            _LOGGER.debug(
                "WS %s: forwarding heater samples failed", self.dev_id, exc_info=True
            )

    def _extract_nodes(self, data: Any) -> dict[str, Any] | None:
        """Extract the nodes dictionary from a websocket payload."""
        if not isinstance(data, dict):
            return None
        nodes = data.get("nodes")
        if isinstance(nodes, dict):
            return nodes
        return None

    @staticmethod
    def _collect_update_addresses(nodes: Mapping[str, Any]) -> list[tuple[str, str]]:
        """Return a sorted list of ``(node_type, addr)`` pairs in ``nodes``."""

        found: set[tuple[str, str]] = set()
        for node_type, payload in nodes.items():
            if not isinstance(node_type, str) or not isinstance(payload, Mapping):
                continue
            for section in payload.values():
                if not isinstance(section, Mapping):
                    continue
                for addr, value in section.items():
                    if isinstance(addr, str) and value is not None:
                        found.add((node_type, addr))
        return sorted(found)

    def _dispatch_nodes(self, payload: dict[str, Any]) -> dict[str, list[str]]:
        """Publish node updates and return the address map by node type."""

        if not isinstance(payload, dict):  # pragma: no cover - defensive
            return {}

        is_snapshot = isinstance(payload.get("nodes_by_type"), dict)
        raw_nodes: Any
        snapshot: dict[str, Any]

        if is_snapshot:
            snapshot = payload
            raw_nodes = snapshot.get("nodes")
        else:
            raw_nodes = payload
            snapshot = {"nodes": deepcopy(raw_nodes), "nodes_by_type": {}}

        record = self.hass.data.get(DOMAIN, {}).get(self.entry_id)
        record_map: Mapping[str, Any]
        if isinstance(record, Mapping):
            record_map = record
        else:
            record_map = {}  # pragma: no cover - defensive default

        inventory = ensure_node_inventory(record_map, nodes=raw_nodes)

        addr_map, unknown_types = addresses_by_node_type(
            inventory, known_types=NODE_CLASS_BY_TYPE
        )
        if unknown_types:  # pragma: no cover - diagnostic branch
            _LOGGER.debug(
                "WS %s: unknown node types in inventory: %s",
                self.dev_id,
                ", ".join(sorted(unknown_types)),
            )

        if not is_snapshot:  # pragma: no cover - legacy branch
            nodes_by_type = {
                node_type: {"addrs": list(addrs)} for node_type, addrs in addr_map.items()
            }
            snapshot["nodes_by_type"] = nodes_by_type
            if "htr" in nodes_by_type:
                snapshot.setdefault("htr", nodes_by_type["htr"])

        if raw_nodes is None:  # pragma: no cover - defensive default
            raw_nodes = {}

        if hasattr(self._coordinator, "update_nodes"):
            self._coordinator.update_nodes(raw_nodes, inventory)

        if isinstance(record, dict):
            record["nodes"] = raw_nodes
            record["node_inventory"] = inventory

        self._apply_heater_addresses(addr_map, inventory=None)

        payload_copy = {
            "dev_id": self.dev_id,
            "node_type": None,
            "nodes": deepcopy(snapshot.get("nodes")),
            "nodes_by_type": deepcopy(snapshot.get("nodes_by_type", {})),
        }
        payload_copy.setdefault(
            "addr_map", {node_type: list(addrs) for node_type, addrs in addr_map.items()}
        )
        if unknown_types:
            payload_copy.setdefault("unknown_types", sorted(unknown_types))

        def _send() -> None:
            """Fire the dispatcher signal with the latest node payload."""

            async_dispatcher_send(
                self.hass,
                signal_ws_data(self.entry_id),
                payload_copy,
            )

        loop = getattr(self.hass, "loop", None)
        call_soon = getattr(loop, "call_soon_threadsafe", None)
        if callable(call_soon):
            call_soon(_send)
        else:  # pragma: no cover - legacy hass loop stub
            _send()

        return {node_type: list(addrs) for node_type, addrs in addr_map.items()}

    def _ensure_type_bucket(
        self,
        dev_map: dict[str, Any],
        nodes_by_type: dict[str, Any],
        node_type: str,
    ) -> dict[str, Any]:
        """Return the node bucket for ``node_type`` with default sections."""
        bucket = nodes_by_type.get(node_type)
        if bucket is None:
            bucket = {
                "addrs": [],
                "settings": {},
                "advanced": {},
                "samples": {},
            }
            nodes_by_type[node_type] = bucket
        else:
            bucket.setdefault("addrs", [])
            bucket.setdefault("settings", {})
            bucket.setdefault("advanced", {})
            bucket.setdefault("samples", {})
        if node_type == "htr":
            dev_map["htr"] = bucket
        return bucket

    def _apply_heater_addresses(
        self,
        addr_map: Mapping[Any, Iterable[Any]] | Iterable[Any] | None,
        *,
        inventory: list[Any] | None = None,
    ) -> dict[str, list[str]]:
        """Update entry and coordinator state with heater address data."""

        normalized_map, _compat_aliases = normalize_heater_addresses(addr_map)
        record = self.hass.data.get(DOMAIN, {}).get(self.entry_id)
        if isinstance(record, dict):
            if inventory is not None:
                record["node_inventory"] = inventory
            energy_coordinator = record.get("energy_coordinator")
            if hasattr(energy_coordinator, "update_addresses"):
                energy_coordinator.update_addresses(normalized_map)

        coordinator_data = getattr(self._coordinator, "data", None)
        if isinstance(coordinator_data, dict):
            dev_map = coordinator_data.get(self.dev_id)
            if isinstance(dev_map, dict):
                nodes_by_type: dict[str, Any] = dev_map.setdefault("nodes_by_type", {})
                for node_type, addrs in normalized_map.items():
                    if not addrs and node_type != "htr":
                        continue
                    bucket = self._ensure_type_bucket(
                        dev_map, nodes_by_type, node_type
                    )
                    if addrs:
                        bucket["addrs"] = list(addrs)
                updated = dict(coordinator_data)
                updated[self.dev_id] = dev_map
                self._coordinator.data = updated  # type: ignore[attr-defined]

        return normalized_map

    def _heater_sample_subscription_targets(self) -> list[tuple[str, str]]:
        """Return ordered ``(node_type, addr)`` heater sample subscriptions."""

        record = self.hass.data.get(DOMAIN, {}).get(self.entry_id)
        inventory, normalized_map, _ = collect_heater_sample_addresses(
            record,
            coordinator=self._coordinator,
        )
        normalized_map = self._apply_heater_addresses(
            normalized_map,
            inventory=inventory or None,
        )
        return heater_sample_subscription_targets(normalized_map)

    async def _subscribe_heater_samples(self) -> None:
        """Subscribe to heater and accumulator sample updates."""

        try:
            for node_type, addr in self._heater_sample_subscription_targets():
                await self._sio.emit(
                    "subscribe",
                    f"/{node_type}/{addr}/samples",
                    namespace=WS_NAMESPACE,
                )
        except asyncio.CancelledError:  # pragma: no cover - task lifecycle
            raise
        except Exception:  # noqa: BLE001 - defensive logging
            _LOGGER.debug(
                "WS %s: sample subscription setup failed", self.dev_id, exc_info=True
            )

    @staticmethod
    def _build_nodes_snapshot(nodes: dict[str, Any]) -> dict[str, Any]:
        """Normalise the nodes payload for consumers."""
        nodes_copy = deepcopy(nodes)
        nodes_by_type: dict[str, Any] = {
            node_type: payload
            for node_type, payload in nodes_copy.items()
            if isinstance(payload, dict)
        }
        snapshot: dict[str, Any] = {
            "nodes": nodes_copy,
            "nodes_by_type": nodes_by_type,
        }
        snapshot.update(nodes_by_type)
        if "htr" in nodes_by_type:
            snapshot.setdefault("htr", nodes_by_type["htr"])
        return snapshot

    @staticmethod
    def _merge_nodes(target: dict[str, Any], source: dict[str, Any]) -> None:
        """Deep-merge incremental node updates into the stored snapshot."""
        for key, value in source.items():
            if isinstance(value, dict):
                existing = target.get(key)
                if isinstance(existing, dict):
                    WebSocketClient._merge_nodes(existing, value)
                else:
                    target[key] = deepcopy(value)
            else:
                target[key] = value

    # ------------------------------------------------------------------
    # Helpers shared across implementations
    # ------------------------------------------------------------------
    def _ws_state_bucket(self) -> dict[str, Any]:
        """Return the websocket state bucket for this device."""
        if self._ws_state is None:
            if not hasattr(self.hass, "data") or self.hass.data is None:  # type: ignore[attr-defined]
                setattr(self.hass, "data", {})  # type: ignore[attr-defined]
            domain_bucket = self.hass.data.setdefault(DOMAIN, {})  # type: ignore[attr-defined]
            entry_bucket = domain_bucket.setdefault(self.entry_id, {})
            ws_state = entry_bucket.setdefault("ws_state", {})
            self._ws_state = ws_state.setdefault(self.dev_id, {})
        return self._ws_state

    def _update_status(self, status: str) -> None:
        """Publish the websocket status to Home Assistant listeners."""
        if status == self._status and status not in {"healthy", "connected"}:
            return
        self._status = status
        now = time.time()
        state = self._ws_state_bucket()
        last_event = self._stats.last_event_ts or self._last_event_at
        state["status"] = status
        state["last_event_at"] = last_event or None
        state["healthy_since"] = self._healthy_since
        state["healthy_minutes"] = (
            int((now - self._healthy_since) / 60) if self._healthy_since else 0
        )
        state["frames_total"] = self._stats.frames_total
        state["events_total"] = self._stats.events_total
        async_dispatcher_send(
            self.hass,
            signal_ws_status(self.entry_id),
            {"dev_id": self.dev_id, "status": status},
        )

    def _mark_event(
        self, *, paths: list[str] | None, count_event: bool = False
    ) -> None:
        """Record receipt of a websocket event batch for health tracking."""
        now = time.time()
        self._cancel_idle_restart()
        self._stats.last_event_ts = now
        self._last_event_at = now
        if paths:
            self._stats.events_total += 1
            if _LOGGER.isEnabledFor(logging.DEBUG):
                uniq: list[str] = []
                for path in paths:
                    if path not in uniq:
                        uniq.append(path)
                    if len(uniq) >= 5:
                        break
                self._stats.last_paths = uniq
        elif count_event:
            self._stats.events_total += 1
        state: dict[str, Any] = self._ws_state_bucket()
        state["last_event_at"] = now
        state["frames_total"] = self._stats.frames_total
        state["events_total"] = self._stats.events_total
        if self._healthy_since is None:
            self._healthy_since = now
            self._update_status("healthy")

    def _schedule_idle_restart(self, *, idle_for: float, source: str) -> None:
        """Log idle payload detection and close the websocket for restart."""

        if self._closing or self._idle_restart_pending:
            return
        self._idle_restart_pending = True
        _LOGGER.warning(
            "WS %s: no payloads for %.0f s (%s heartbeat); restarting",
            self.dev_id,
            idle_for,
            source,
        )

        async def _restart() -> None:
            try:
                await self._disconnect(reason="idle restart")
            finally:
                self._idle_restart_pending = False
                self._idle_restart_task = None

        self._idle_restart_task = self._loop.create_task(_restart())

    def _cancel_idle_restart(self) -> None:
        """Cancel any scheduled idle restart due to new payload activity."""

        task = self._idle_restart_task
        if task and not task.done():
            task.cancel()
        self._idle_restart_task = None
        self._idle_restart_pending = False

    async def _get_token(self) -> str:
        """Reuse the REST client token for websocket authentication."""
        headers = await self._client._authed_headers()  # noqa: SLF001
        auth_header = (
            headers.get("Authorization") if isinstance(headers, dict) else None
        )
        if not auth_header:
            raise RuntimeError("authorization token missing")
        return auth_header.split(" ", 1)[1]

    async def _force_refresh_token(self) -> None:
        """Force the REST client to fetch a fresh access token."""
        with suppress(AttributeError, RuntimeError):
            self._client._access_token = None  # type: ignore[attr-defined]  # noqa: SLF001
        await self._client._ensure_token()  # noqa: SLF001

    def _api_base(self) -> str:
        """Return the base REST API URL used for websocket routes."""
        base = getattr(self._client, "api_base", None)
        if isinstance(base, str) and base:
            return base.rstrip("/")
        return API_BASE


# ----------------------------------------------------------------------
# Legacy Socket.IO 0.9 client
# ----------------------------------------------------------------------


class TermoWebWSClient(WebSocketClient):  # pragma: no cover - legacy network client
    """Legacy Socket.IO 0.9 websocket client for TermoWeb."""

    def __init__(
        self,
        hass: HomeAssistant,
        *,
        entry_id: str,
        dev_id: str,
        api_client: RESTClient,
        coordinator: Any,
        session: aiohttp.ClientSession | None = None,
        handshake_fail_threshold: int = 5,
        protocol: str | None = None,
    ) -> None:
        """Initialise the legacy websocket client container."""

        self.hass = hass
        self.entry_id = entry_id
        self.dev_id = dev_id
        self._client = api_client
        self._coordinator = coordinator
        self._session = session or getattr(api_client, "_session", None)
        if self._session is None:
            raise RuntimeError("aiohttp session required for websocket client")
        self._protocol_hint = protocol
        self._loop = getattr(hass, "loop", None) or asyncio.get_event_loop()
        self._task: asyncio.Task | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._disconnected = asyncio.Event()
        self._disconnected.set()

        self._closing = False
        self._connected_since: float | None = None
        self._healthy_since: float | None = None
        self._hb_send_interval: float = 27.0
        self._hb_task: asyncio.Task | None = None

        self._backoff_seq = [5, 10, 30, 120, 300]
        self._backoff_idx = 0
        self._hs_fail_threshold = handshake_fail_threshold
        self._hs_fail_count: int = 0
        self._hs_fail_start: float = 0.0

        self._stats = WSStats()
        self._status: str = "stopped"
        self._stop_event = asyncio.Event()
        self._handshake_logged = False
        self._last_event_at: float | None = None

        self._handshake_payload: dict[str, Any] | None = None
        self._nodes: dict[str, Any] = {}
        self._nodes_raw: dict[str, Any] = {}

        self._payload_idle_window: float = 3600.0
        self._idle_restart_task: asyncio.Task | None = None
        self._idle_restart_pending = False
        self._idle_monitor_task: asyncio.Task | None = None

        self._init_subscription_state()
        self._legacy_subscription_configured = False

        self._ws_state: dict[str, Any] | None = None
        self._ws_state_bucket()

    # ------------------------------------------------------------------
    # Public control
    # ------------------------------------------------------------------
    async def stop(self) -> None:
        """Cancel tasks, close websocket sessions and reset legacy state."""

        self._closing = True
        if self._hb_task:
            self._hb_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._hb_task
            self._hb_task = None
        self._legacy_subscription_configured = False
        await super().stop()

    # ------------------------------------------------------------------
    # Core loop and protocol dispatch
    # ------------------------------------------------------------------
    async def _runner(self) -> None:
        """Manage reconnection attempts and lifecycle."""

        self._update_status("starting")
        try:
            await self._run_socketio_09()
        finally:
            self._update_status("stopped")

    async def _run_socketio_09(self) -> None:
        """Manage reconnection loops for the legacy Socket.IO protocol."""

        while not self._closing:
            should_retry = True
            try:
                self._legacy_subscription_configured = False
                _LOGGER.debug("WS %s: initiating Socket.IO 0.9 handshake", self.dev_id)
                sid, hb_timeout = await self._handshake()
                _LOGGER.debug(
                    "WS %s: handshake succeeded sid=%s hb_timeout=%s",
                    self.dev_id,
                    sid,
                    hb_timeout,
                )
                self._hs_fail_count = 0
                self._hs_fail_start = 0.0
                self._hb_send_interval = max(5.0, min(30.0, hb_timeout * 0.45))
                await self._connect_ws(sid)
                _LOGGER.debug("WS %s: websocket connection established", self.dev_id)
                await self._join_namespace()
                await self._send_snapshot_request()
                await self._subscribe_htr_samples()
                self._connected_since = time.time()
                self._healthy_since = None
                self._update_status("connected")
                self._hb_task = self._loop.create_task(self._heartbeat_loop())
                await self._read_loop()
            except asyncio.CancelledError:
                should_retry = False
            except HandshakeError as err:
                self._hs_fail_count += 1
                if self._hs_fail_count == 1:
                    self._hs_fail_start = time.time()
                _LOGGER.info(
                    "WS %s: connection error (%s: %s); will retry",
                    self.dev_id,
                    type(err).__name__,
                    err,
                )
                if self._hs_fail_count >= self._hs_fail_threshold:
                    elapsed = time.time() - self._hs_fail_start
                    _LOGGER.warning(
                        "WS %s: handshake failed %d times over %.1f s",
                        self.dev_id,
                        self._hs_fail_count,
                        elapsed,
                    )
                    self._hs_fail_count = 0
                    self._hs_fail_start = 0.0
                _LOGGER.debug(
                    "WS %s: handshake error url=%s body=%r",
                    self.dev_id,
                    err.url,
                    err.body_snippet,
                )
            except Exception as err:  # noqa: BLE001
                _LOGGER.info(
                    "WS %s: connection error (%s: %s); will retry",
                    self.dev_id,
                    type(err).__name__,
                    err,
                )
                _LOGGER.debug(
                    "WS %s: connection error details", self.dev_id, exc_info=True
                )
            finally:
                if self._hb_task:
                    self._hb_task.cancel()
                    self._hb_task = None
                await self._cancel_subscription_refresh()
                self._subscription_refresh_failed = False
                self._subscription_refresh_due = None
                if self._ws:
                    with suppress(aiohttp.ClientError, RuntimeError):
                        await self._ws.close()
                    self._ws = None
                self._update_status("disconnected")
            if self._closing or not should_retry:
                break
            delay = self._backoff_seq[
                min(self._backoff_idx, len(self._backoff_seq) - 1)
            ]
            self._backoff_idx = min(self._backoff_idx + 1, len(self._backoff_seq) - 1)
            jitter = random.uniform(0.8, 1.2)
            await asyncio.sleep(delay * jitter)

    async def _handshake(self) -> tuple[str, int]:
        """Perform the legacy GET /socket.io/1/ handshake."""

        token = await self._get_token()
        t_ms = int(time.time() * 1000)
        base = self._api_base()
        url = f"{base}/socket.io/1/?token={token}&dev_id={self.dev_id}&t={t_ms}"
        try:
            async with asyncio.timeout(15):
                async with self._session.get(
                    url, timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    body = await resp.text()
                    if resp.status == 401:
                        _LOGGER.info(
                            "WS %s: handshake 401; refreshing token", self.dev_id
                        )
                        await self._force_refresh_token()
                        raise HandshakeError(resp.status, url, body)
                    if resp.status != 200:
                        raise HandshakeError(resp.status, url, body)
        except TimeoutError as err:
            raise HandshakeError(599, url, "timeout") from err
        except aiohttp.ClientError as err:
            raise HandshakeError(598, url, str(err)) from err
        parts = body.strip().split(":")
        if len(parts) < 3:
            raise HandshakeError(590, url, body[:120])
        sid = parts[0]
        try:
            hb_timeout = int(parts[1])
        except ValueError as err:
            raise HandshakeError(591, url, body[:120]) from err
        return sid, hb_timeout

    async def _connect_ws(self, sid: str) -> None:
        """Open the websocket connection using the handshake session id."""

        token = await self._get_token()
        ws_url = (
            f"{self._socket_base()}/socket.io/1/websocket/{sid}"
            f"?token={token}&dev_id={self.dev_id}"
        )
        _LOGGER.info("WS %s: connecting to %s", self.dev_id, ws_url)
        self._ws = await self._session.ws_connect(
            ws_url,
            timeout=aiohttp.ClientTimeout(total=15),
            heartbeat=None,
            autoclose=False,
        )

    async def _join_namespace(self) -> None:
        """Join the API namespace after the websocket connects."""

        await self._send_text(f"1::{WS_NAMESPACE}")

    async def _send_snapshot_request(self) -> None:
        """Request the initial device snapshot."""

        payload = {"name": "dev_data", "args": []}
        await self._send_text(
            f"5::{WS_NAMESPACE}:{json.dumps(payload, separators=(',', ':'))}"
        )

    async def _subscribe_htr_samples(self) -> None:
        """Subscribe to heater sample updates."""

        targets = self._heater_sample_subscription_targets()
        if not targets:
            return
        for node_type, addr in targets:
            payload = {
                "name": "subscribe",
                "args": [f"/{node_type}/{addr}/samples"],
            }
            await self._send_text(
                f"5::{WS_NAMESPACE}:{json.dumps(payload, separators=(',', ':'))}"
            )

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeat frames to keep the connection alive."""

        try:
            await self._run_heartbeat(
                self._hb_send_interval, partial(self._send_text, "2::")
            )
        except asyncio.CancelledError:
            return
        except (aiohttp.ClientError, RuntimeError):  # pragma: no cover - best effort
            return

    async def _read_loop(self) -> None:
        """Consume websocket frames and route events for the legacy protocol."""

        ws = self._ws
        if ws is None:
            return
        async for data in self._ws_payload_stream(ws, context="websocket"):
            if data.startswith("2::"):
                self._record_heartbeat(source="socketio09")
                continue
            if data.startswith(f"1::{WS_NAMESPACE}"):
                continue
            if data.startswith(f"5::{WS_NAMESPACE}:"):
                try:
                    payload = json.loads(data.split(f"5::{WS_NAMESPACE}:", 1)[1])
                except Exception:  # noqa: BLE001
                    continue
                self._handle_event(payload)
                continue
            if data.startswith("0::"):
                raise RuntimeError("server disconnect")

    def _handle_event(self, evt: dict[str, Any]) -> None:
        """Process a Socket.IO 0.9 event payload."""

        if not isinstance(evt, dict):
            return
        name = evt.get("name")
        args = evt.get("args")
        if name != "data" or not isinstance(args, list) or not args:
            return
        batch = args[0] if isinstance(args[0], list) else None
        if not isinstance(batch, list):
            return
        self._maybe_configure_legacy_subscription(batch)
        paths: list[str] = []
        updated_nodes = False
        updated_addrs: list[tuple[str, str]] = []
        sample_addrs: list[tuple[str, str]] = []

        def _extract_type_addr(path: str) -> tuple[str | None, str | None]:
            """Extract the node type and address from a websocket path."""

            if not path:
                return None, None
            parts = [segment for segment in path.split("/") if segment]
            for idx in range(len(parts) - 2):
                node_type = parts[idx]
                addr = parts[idx + 1]
                leaf = parts[idx + 2]
                if leaf in {"settings", "samples", "advanced_setup"}:
                    return node_type, addr
            return None, None

        for item in batch:
            if not isinstance(item, Mapping):
                continue
            path = item.get("path")
            body = item.get("body")
            if not isinstance(path, str):
                continue
            paths.append(path)
            dev_map: dict[str, Any] = (self._coordinator.data or {}).get(
                self.dev_id
            ) or {}
            if not dev_map:
                htr_bucket: dict[str, Any] = {
                    "addrs": [],
                    "settings": {},
                    "advanced": {},
                    "samples": {},
                }
                dev_map = {
                    "dev_id": self.dev_id,
                    "name": f"Device {self.dev_id}",
                    "raw": {},
                    "connected": True,
                    "nodes": None,
                    "nodes_by_type": {"htr": htr_bucket},
                    "htr": htr_bucket,
                }
                cur = dict(self._coordinator.data or {})
                cur[self.dev_id] = dev_map
                self._coordinator.data = cur  # type: ignore[attr-defined]
            nodes_by_type: dict[str, Any] = dev_map.setdefault("nodes_by_type", {})
            if path.endswith("/mgr/nodes"):
                if isinstance(body, dict):
                    dev_map["nodes"] = body
                    self._nodes_raw = deepcopy(body)
                    self._nodes = self._build_nodes_snapshot(self._nodes_raw)
                    type_to_addrs = self._dispatch_nodes(body)
                    for node_type, addrs in type_to_addrs.items():
                        bucket = self._ensure_type_bucket(
                            dev_map, nodes_by_type, node_type
                        )
                        bucket["addrs"] = list(addrs)
                    updated_nodes = True
                continue
            node_type, addr = _extract_type_addr(path)
            if node_type and addr and node_type != "mgr":
                section = self._legacy_section_for_path(path)
                if section:
                    if section in {"settings", "advanced"} and not isinstance(
                        body, Mapping
                    ):
                        continue
                    if self._update_legacy_section(
                        node_type=node_type,
                        addr=addr,
                        section=section,
                        body=body,
                        dev_map=dev_map,
                        nodes_by_type=nodes_by_type,
                    ):
                        if section == "samples":
                            sample_addrs.append((node_type, addr))
                        elif section == "settings":
                            updated_addrs.append((node_type, addr))
                        updated_nodes = True
                    continue
            raw = dev_map.setdefault("raw", {})
            key = path.strip("/").replace("/", "_")
            raw[key] = body

        if updated_nodes:
            self._nodes = self._build_nodes_snapshot(self._nodes_raw)
        self._mark_event(paths=paths)
        payload_base = {
            "dev_id": self.dev_id,
            "ts": self._stats.last_event_ts,
            "node_type": None,
        }
        if updated_nodes:
            async_dispatcher_send(
                self.hass,
                signal_ws_data(self.entry_id),
                {**payload_base, "addr": None, "kind": "nodes"},
            )
        for node_type, addr in set(updated_addrs):
            async_dispatcher_send(
                self.hass,
                signal_ws_data(self.entry_id),
                {
                    **payload_base,
                    "addr": addr,
                    "kind": f"{node_type}_settings",
                    "node_type": node_type,
                },
            )
        for node_type, addr in set(sample_addrs):
            async_dispatcher_send(
                self.hass,
                signal_ws_data(self.entry_id),
                {
                    **payload_base,
                    "addr": addr,
                    "kind": f"{node_type}_samples",
                    "node_type": node_type,
                },
            )
        self._log_legacy_update(
            updated_nodes=updated_nodes,
            updated_addrs=updated_addrs,
            sample_addrs=sample_addrs,
        )

    def _update_legacy_section(
        self,
        *,
        node_type: str,
        addr: str,
        section: str,
        body: Any,
        dev_map: dict[str, Any],
        nodes_by_type: dict[str, Any],
    ) -> bool:
        """Store legacy section updates and mirror them in raw state."""

        bucket = self._ensure_type_bucket(dev_map, nodes_by_type, node_type)
        section_map: dict[str, Any] = bucket.setdefault(section, {})
        if not isinstance(section_map, dict):
            return False
        value: Any = dict(body) if isinstance(body, Mapping) else body
        section_map[addr] = value
        raw_bucket = self._nodes_raw.setdefault(node_type, {})
        raw_section = raw_bucket.setdefault(section, {})
        if isinstance(raw_section, dict):
            raw_section[addr] = deepcopy(body)
        return True

    @staticmethod
    def _legacy_section_for_path(path: str) -> str | None:
        """Return the legacy section identifier for ``path`` if supported."""

        if path.endswith("/settings"):
            return "settings"
        if path.endswith("/advanced_setup"):
            return "advanced"
        if path.endswith("/samples"):
            return "samples"
        return None

    def _log_legacy_update(
        self,
        *,
        updated_nodes: bool,
        updated_addrs: Iterable[tuple[str, str]],
        sample_addrs: Iterable[tuple[str, str]],
    ) -> None:
        """Emit debug logging for the legacy websocket update batch."""

        if not _LOGGER.isEnabledFor(logging.DEBUG):
            return
        addr_pairs = {f"{node_type}/{addr}" for node_type, addr in updated_addrs}
        addr_pairs.update(f"{node_type}/{addr}" for node_type, addr in sample_addrs)
        if addr_pairs:
            _LOGGER.debug(
                "WS %s: legacy update for %s", self.dev_id, ", ".join(sorted(addr_pairs))
            )
        elif updated_nodes:
            _LOGGER.debug("WS %s: legacy nodes refresh", self.dev_id)

    def _maybe_configure_legacy_subscription(self, batch: Iterable[Any]) -> None:
        """Inspect event payloads for TTL metadata and schedule refreshes."""

        if self._legacy_subscription_configured and self._subscription_refresh_task:
            return
        ttl_info: tuple[float, str] | None = None
        for item in batch:
            if not isinstance(item, Mapping):
                continue
            body = item.get("body")
            if isinstance(body, Mapping):
                ttl_info = self._extract_subscription_ttl(body)
                if ttl_info:
                    break
        if ttl_info:
            ttl, source = ttl_info
            self._apply_subscription_ttl(
                ttl=ttl,
                source=source,
                context="legacy session",
                missing_hint="legacy session did not expose lease info",
                now=time.time(),
            )
            self._legacy_subscription_configured = True
            return
        if self._subscription_refresh_task is None and self._subscription_refresh_due is None:
            self._apply_subscription_ttl(
                ttl=_DEFAULT_SUBSCRIPTION_TTL,
                source="default",
                context="legacy session",
                missing_hint="legacy session did not expose lease info",
                now=time.time(),
            )

    async def _subscription_refresh_loop(self) -> None:
        """Renew the legacy websocket lease before the TTL expires."""

        try:
            while not self._closing:
                ttl = max(self._subscription_ttl, 60.0)
                lead = min(max(ttl * 0.2, 30.0), ttl / 2)
                wait_for = max(ttl - lead, ttl * 0.5)
                wait_for *= random.uniform(0.85, 0.95)
                wait_for = max(wait_for, 30.0)
                self._subscription_refresh_due = time.time() + wait_for
                await asyncio.sleep(wait_for)
                if self._closing:
                    break
                ws = self._ws
                if ws is None or getattr(ws, "closed", False):
                    continue
                try:
                    await self._refresh_subscription(reason="periodic renewal")
                except asyncio.CancelledError:  # pragma: no cover - task lifecycle
                    raise
                except Exception:  # noqa: BLE001
                    self._subscription_refresh_failed = True
                    _LOGGER.warning(
                        "WS %s: subscription refresh failed; monitoring for idle",
                        self.dev_id,
                        exc_info=True,
                    )
                    await asyncio.sleep(min(60.0, ttl / 2))
        finally:
            self._subscription_refresh_due = None

    async def _refresh_subscription(self, *, reason: str) -> None:
        """Renew the legacy websocket lease by replaying subscription calls."""

        async with self._subscription_refresh_lock:
            ws = self._ws
            if ws is None or getattr(ws, "closed", False):
                raise RuntimeError("websocket not connected")
            now = time.time()
            self._subscription_refresh_last_attempt = now
            _LOGGER.info(
                "WS %s: refreshing websocket lease (%s)",
                self.dev_id,
                reason,
            )
            try:
                await self._send_snapshot_request()
                await self._subscribe_htr_samples()
            except asyncio.CancelledError:  # pragma: no cover - task lifecycle
                raise
            except Exception as err:
                self._subscription_refresh_failed = True
                _LOGGER.warning(
                    "WS %s: legacy lease refresh failed (%s: %s)",
                    self.dev_id,
                    type(err).__name__,
                    err,
                    exc_info=True,
                )
                self._schedule_idle_restart(
                    idle_for=self._payload_idle_window,
                    source="lease refresh failure",
                )
                raise
            self._subscription_refresh_failed = False
            self._subscription_refresh_last_success = time.time()
            self._subscription_refresh_due = (
                self._subscription_refresh_last_success + self._subscription_ttl
            )
            _LOGGER.info(
                "WS %s: websocket lease refreshed; next in ~%.0fs",
                self.dev_id,
                self._subscription_ttl,
            )

    async def _send_text(self, data: str) -> None:
        """Send a websocket text frame."""

        if not self._ws:
            raise RuntimeError("websocket not connected")
        await self._ws.send_str(data)

    async def _disconnect(self, *, reason: str) -> None:
        """Close the websocket connection if active."""

        if self._ws:
            with suppress(aiohttp.ClientError, RuntimeError):
                await self._ws.close(
                    code=aiohttp.WSCloseCode.GOING_AWAY,
                    message=reason.encode(),
                )
            self._ws = None

    async def _ws_payload_stream(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        *,
        context: str,
    ) -> AsyncIterator[str]:
        """Yield websocket payload strings."""

        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                self._stats.frames_total += 1
                yield msg.data
            elif msg.type == aiohttp.WSMsgType.BINARY:
                try:
                    decoded = codecs.decode(msg.data, "utf-8")
                except Exception:  # noqa: BLE001
                    continue
                self._stats.frames_total += 1
                yield decoded
            elif msg.type == aiohttp.WSMsgType.ERROR:
                raise RuntimeError(f"{context} error: {ws.exception()}")
            elif msg.type in {aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED}:
                raise RuntimeError(f"{context} closed")

    def _record_heartbeat(self, *, source: str) -> None:
        """Record receipt of a heartbeat frame."""

        now = time.time()
        self._stats.last_event_ts = now
        self._last_event_at = now
        self._cancel_idle_restart()

    async def _run_heartbeat(
        self, interval: float, send: Callable[[], Awaitable[Any]]
    ) -> None:
        """Send periodic heartbeats until cancelled."""

        while not self._closing:
            await asyncio.sleep(interval)
            try:
                await send()
            except Exception:  # noqa: BLE001
                return

    def _socket_base(self) -> str:
        """Return the base URL for websocket connections."""

        base = self._api_base().rstrip("/")
        parsed = urlsplit(base if base else API_BASE)
        scheme = parsed.scheme or "https"
        netloc = parsed.netloc or parsed.path
        path = parsed.path.rstrip("/")
        return urlunsplit((scheme, netloc, path or "", "", ""))

class DucaheatWSClient(WebSocketClient):
    """Verbose websocket client variant with payload debug logging."""

    def _summarise_addresses(self, data: Any) -> str:
        """Return a condensed summary of node addresses in ``data``."""

        nodes = self._extract_nodes(data)
        if not nodes:
            return "no node addresses"
        addresses = self._collect_update_addresses(nodes)
        if not addresses:
            return "no node addresses"
        return ", ".join(f"{node_type}/{addr}" for node_type, addr in addresses)

    async def _on_connect(self) -> None:
        """Log connect events before delegating to the base implementation."""
        if _LOGGER.isEnabledFor(logging.DEBUG):
            _LOGGER.debug("WS %s (ducaheat): connected", self.dev_id)
        await super()._on_connect()

    async def _on_disconnect(self) -> None:
        """Log disconnect events before delegating."""
        if _LOGGER.isEnabledFor(logging.DEBUG):
            _LOGGER.debug("WS %s (ducaheat): disconnected", self.dev_id)
        await super()._on_disconnect()

    async def _on_dev_handshake(self, data: Any) -> None:
        """Log handshake payloads prior to standard handling."""
        if _LOGGER.isEnabledFor(logging.DEBUG):
            _LOGGER.debug("WS %s (ducaheat): handshake payload: %s", self.dev_id, data)
        await super()._on_dev_handshake(data)

    async def _on_dev_data(self, data: Any) -> None:
        """Log initial dev_data snapshots with condensed address details."""
        if _LOGGER.isEnabledFor(logging.DEBUG):
            summary = self._summarise_addresses(data)
            _LOGGER.debug(
                "WS %s (ducaheat): dev_data addresses: %s", self.dev_id, summary
            )
        await super()._on_dev_data(data)

    async def _on_update(self, data: Any) -> None:
        """Log incremental update payloads with condensed address details."""
        if _LOGGER.isEnabledFor(logging.DEBUG):
            summary = self._summarise_addresses(data)
            _LOGGER.debug(
                "WS %s (ducaheat): update addresses: %s", self.dev_id, summary
            )
        await super()._on_update(data)


__all__ = [
    "DucaheatWSClient",
    "TermoWebWSClient",
    "WSStats",
    "WebSocketClient",
]
