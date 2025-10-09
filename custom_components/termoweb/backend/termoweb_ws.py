"""Unified websocket client for TermoWeb backends."""

from __future__ import annotations

import asyncio
import codecs
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Mapping,
    MutableMapping,
)
from contextlib import suppress
from copy import deepcopy
from functools import partial, wraps
import json
import logging
import random
import time
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
import weakref

import aiohttp
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_send
import socketio

from custom_components.termoweb.api import RESTClient
from custom_components.termoweb.backend.sanitize import (
    mask_identifier,
    redact_token_fragment,
)
from custom_components.termoweb.const import (
    ACCEPT_LANGUAGE,
    API_BASE,
    BRAND_DUCAHEAT,
    BRAND_TERMOWEB,
    DOMAIN,
    USER_AGENT,
    WS_NAMESPACE,
    get_brand_requested_with,
    get_brand_user_agent,
    signal_ws_data,
)
from custom_components.termoweb.installation import (
    InstallationSnapshot,
    ensure_snapshot,
)
from custom_components.termoweb.inventory import (
    HEATER_NODE_TYPES,
    Inventory,
    addresses_by_node_type as _addresses_by_node_type,
    build_node_inventory as _build_node_inventory,
    normalize_node_addr,
    normalize_node_type,
)
from custom_components.termoweb.nodes import ensure_node_inventory

from .ws_client import (
    WSStats,
    _prepare_nodes_dispatch,
    _WSStatusMixin,
    forward_ws_sample_updates,
    resolve_ws_update_section,
)

_LOGGER = logging.getLogger(__name__)

build_node_inventory = _build_node_inventory  # re-exported for tests
addresses_by_node_type = _addresses_by_node_type  # legacy import hook for tests


_SENSITIVE_PLACEHOLDERS: Mapping[str, tuple[str, Callable[[str | None], str]]] = {
    "token": ("{token}", redact_token_fragment),
    "dev_id": ("{dev_id}", mask_identifier),
    "sid": ("{sid}", mask_identifier),
}


class HandshakeError(RuntimeError):
    """Capture context for failed websocket handshakes."""

    def __init__(self, status: int, url: str, body_snippet: str) -> None:
        """Initialise the error with the HTTP response details."""

        super().__init__(f"handshake failed (status={status})")
        self.status = status
        self.url = url
        self.body_snippet = body_snippet


class WebSocketClient(_WSStatusMixin):
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
        namespace: str = WS_NAMESPACE,
    ) -> None:
        """Initialise the websocket client container."""
        self.hass = hass
        self.entry_id = entry_id
        self.dev_id = dev_id
        self._client = api_client
        self._coordinator = coordinator
        self._dispatcher = async_dispatcher_send
        self._session = session or getattr(api_client, "_session", None)
        self._protocol_hint = protocol
        self._loop = getattr(hass, "loop", None) or asyncio.get_event_loop()
        self._task: asyncio.Task | None = None
        self._namespace = namespace or WS_NAMESPACE

        is_ducaheat = getattr(api_client, "_is_ducaheat", False)
        brand = BRAND_DUCAHEAT if is_ducaheat else BRAND_TERMOWEB
        self._user_agent = (
            getattr(api_client, "user_agent", None)
            or get_brand_user_agent(brand)
        )
        requested_with = getattr(api_client, "requested_with", None)
        if requested_with is None:
            requested_with = get_brand_requested_with(brand)
        self._requested_with = requested_with

        http_session = (
            self._session
            if isinstance(self._session, aiohttp.ClientSession)
            else None
        )
        self._sio = socketio.AsyncClient(
            reconnection=False,
            logger=_LOGGER.getChild(f"socketio.{dev_id}"),
            engineio_logger=_LOGGER.getChild(f"engineio.{dev_id}"),
            http_session=http_session,
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
            "connect", namespace=self._namespace, handler=self._on_namespace_connect
        )
        self._sio.on(
            "disconnect",
            namespace=self._namespace,
            handler=self._on_namespace_disconnect,
        )
        self._sio.on(
            "dev_handshake", namespace=self._namespace, handler=self._on_dev_handshake
        )
        self._sio.on("dev_data", namespace=self._namespace, handler=self._on_dev_data)
        self._sio.on("update", namespace=self._namespace, handler=self._on_update)

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
        self._last_payload_at: float | None = None
        self._stats = WSStats()

        self._handshake_payload: dict[str, Any] | None = None
        self._nodes: dict[str, Any] = {}
        self._nodes_raw: dict[str, Any] = {}

        self._payload_idle_window: float = 240.0
        self._idle_restart_task: asyncio.Task | None = None
        self._idle_restart_pending = False
        self._idle_monitor_task: asyncio.Task | None = None

        self._subscription_refresh_lock = asyncio.Lock()
        self._subscription_refresh_failed = False
        self._subscription_refresh_last_attempt: float = 0.0
        self._subscription_refresh_last_success: float | None = None
        self._handshake_logged = False
        self._debug_catch_all_registered = False

        self._ws_state: dict[str, Any] | None = None
        state = self._ws_state_bucket()
        state.setdefault("last_payload_at", None)
        state.setdefault("idle_restart_pending", False)

    def _brand_headers(self, *, origin: str | None = None) -> dict[str, str]:
        """Return baseline headers aligned with the REST client brand."""

        headers = {
            "User-Agent": self._user_agent or USER_AGENT,
            "Accept-Language": ACCEPT_LANGUAGE,
        }
        if self._requested_with:
            headers["X-Requested-With"] = self._requested_with
        if origin:
            headers["Origin"] = origin
        return headers

    def _sanitise_headers(self, headers: Mapping[str, Any]) -> dict[str, Any]:
        """Redact sensitive header values for logging."""

        sanitised: dict[str, Any] = {}
        for key, value in headers.items():
            text: str
            if isinstance(value, bytes):
                text = value.decode(errors="ignore")
            else:
                text = str(value)
            if key.lower() == "authorization":
                prefix, _, token = text.partition(" ")
                if token:
                    text = f"{prefix} {redact_token_fragment(token)}".strip()
                else:
                    text = redact_token_fragment(text)
            elif key.lower() in {"cookie", "set-cookie"}:
                text = redact_token_fragment(text)
            sanitised[key] = text
        return sanitised

    def _sanitise_placeholder(self, key: str, value: str | None) -> str | None:
        """Return a placeholder for known sensitive values."""

        entry = _SENSITIVE_PLACEHOLDERS.get(key)
        if entry is None:
            return None
        placeholder, sanitizer = entry
        text = None if value is None else str(value)
        sanitizer(text)
        return placeholder

    def _sanitise_params(self, params: Mapping[str, str]) -> dict[str, str]:
        """Redact sensitive query parameter values for logging."""

        sanitised: dict[str, str] = {}
        for key, value in params.items():
            placeholder = self._sanitise_placeholder(key, value)
            if placeholder is None:
                sanitised[key] = value
            else:
                sanitised[key] = placeholder
        return sanitised

    def _sanitise_url(self, url: str) -> str:
        """Return a redacted representation of the websocket URL."""

        try:
            parsed = urlsplit(url)
        except ValueError:
            return url
        query_items = parse_qsl(parsed.query, keep_blank_values=True)
        sanitised_pairs = []
        for key, value in query_items:
            placeholder = self._sanitise_placeholder(key, value)
            sanitised_pairs.append((key, value if placeholder is None else placeholder))
        sanitised_query = urlencode(sanitised_pairs, doseq=True)
        sanitised_path = parsed.path
        if sanitised_path:
            segments = sanitised_path.split("/")
            if len(segments) >= 2 and segments[-2] == "websocket":
                placeholder = self._sanitise_placeholder("sid", segments[-1] or None)
                if placeholder is not None:
                    segments[-1] = placeholder
                    sanitised_path = "/".join(segments)
        return urlunsplit(
            (parsed.scheme, parsed.netloc, sanitised_path, sanitised_query, parsed.fragment)
        )


    # ------------------------------------------------------------------
    # Public control
    # ------------------------------------------------------------------
    def _wrap_background_task(
        self, target: Any, *args: Any, **kwargs: Any
    ) -> asyncio.Task:
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
        _LOGGER.debug("WS: start requested")
        self._closing = False
        self._stop_event = asyncio.Event()
        self._handshake_logged = False
        self._subscription_refresh_failed = False
        self._task = self._loop.create_task(
            self._runner(), name=f"{DOMAIN}-ws-{self.dev_id}"
        )
        return self._task

    async def stop(self) -> None:
        """Cancel tasks and close websocket sessions."""
        _LOGGER.debug("WS: stop requested")
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
        self._subscription_refresh_failed = False
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

        url, _ = await self._build_engineio_target()
        return url

    async def debug_probe(self) -> None:
        """Emit a dev_data probe for debugging purposes."""

        if not _LOGGER.isEnabledFor(logging.DEBUG):
            return
        try:
            await self._sio.emit("dev_data", namespace=self._namespace)
            _LOGGER.debug("WS: debug probe dev_data emitted")
        except Exception:
            _LOGGER.debug("WS: debug probe dev_data emit failed", exc_info=True)

    # ------------------------------------------------------------------
    # Core loop and protocol dispatch
    # ------------------------------------------------------------------
    async def _runner(self) -> None:
        """Manage connection attempts with backoff."""
        self._update_status("starting")
        try:
            while not self._closing:
                error: Exception | None = None
                try:
                    await self._connect_once()
                    await self._wait_for_events()
                except asyncio.CancelledError:
                    raise
                except Exception as err:
                    error = err
                    _LOGGER.info(
                        "WS: connection error (%s: %s); will retry",
                        type(err).__name__,
                        err,
                    )
                    _LOGGER.debug("WS: connection error details", exc_info=True)
                finally:
                    await self._disconnect(reason="loop cleanup")
                    if not self._closing:
                        self._update_status("disconnected")
                        await self._handle_connection_lost(error)
                if self._closing:
                    break
                delay = self._backoff_seq[
                    min(self._backoff_idx, len(self._backoff_seq) - 1)
                ]
                self._backoff_idx = min(
                    self._backoff_idx + 1, len(self._backoff_seq) - 1
                )
                await asyncio.sleep(delay * random.uniform(0.8, 1.2))
        finally:
            self._update_status("stopped")

    async def _handle_connection_lost(self, error: Exception | None) -> None:
        """Record connection loss metadata before the next restart."""

        state = self._ws_state_bucket()
        restart_count = int(state.get("restart_count") or 0) + 1
        state["restart_count"] = restart_count
        state["last_disconnect_at"] = time.time()
        state["last_disconnect_error"] = (
            f"{type(error).__name__}: {error}" if error else None
        )

    async def _connect_once(self) -> None:
        """Open the Socket.IO connection."""
        if self._stop_event.is_set():
            return
        url, engineio_path = await self._build_engineio_target()
        _LOGGER.debug("WS: connecting to %s (path=%s)", url, engineio_path)
        self._disconnected.clear()
        self._backoff_idx = 0
        await self._sio.connect(
            url,
            transports=["websocket"],
            namespaces=[self._namespace],
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
                [stop_task, disconnect_task], return_when=asyncio.FIRST_COMPLETED
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
            except Exception:
                _LOGGER.debug("WS: disconnect due to %s failed", reason, exc_info=True)
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
        query = urlencode({"token": token, "dev_id": self.dev_id})
        url = urlunsplit((scheme, netloc, "/socket.io", query, ""))
        return url, "socket.io"
        query = urlencode({"token": token, "dev_id": self.dev_id})
        url = urlunsplit((scheme, netloc, "/socket.io", query, ""))
        return url, "socket.io"

    # ------------------------------------------------------------------
    # Socket.IO event handlers
    # ------------------------------------------------------------------
    async def _on_connect(self) -> None:
        """Handle socket connection establishment."""
        _LOGGER.debug("WS: connected")
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
        self._register_debug_catch_all()

    async def _on_namespace_connect(self) -> None:
        """Join the namespace and request the initial snapshot."""

        try:
            if self._namespace != "/":
                await self._sio.emit("join", namespace=self._namespace)
            await self._sio.emit("dev_data", namespace=self._namespace)
        except Exception:
            _LOGGER.debug("WS: namespace join failed", exc_info=True)

    def _register_debug_catch_all(self) -> None:
        """Register a catch-all listener to trace websocket traffic when debugging."""

        if self._debug_catch_all_registered:
            return
        if not _LOGGER.isEnabledFor(logging.DEBUG):
            return

        async def _log_catch_all(event: str, *args: Any, **kwargs: Any) -> None:
            """Emit DEBUG logs for all websocket events received."""

            if not _LOGGER.isEnabledFor(logging.DEBUG):
                return
            _LOGGER.debug(
                "WS: catch-all (%s) event=%s args=%s kwargs=%s",
                self._namespace,
                event,
                args,
                kwargs,
            )

        self._sio.on("*", handler=_log_catch_all, namespace=self._namespace)
        self._debug_catch_all_registered = True

    async def _on_disconnect(self) -> None:
        """Handle socket disconnection."""
        _LOGGER.debug("WS: disconnected")
        if self._idle_monitor_task:
            self._idle_monitor_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._idle_monitor_task
            self._idle_monitor_task = None
        self._subscription_refresh_failed = False
        self._handshake_logged = False
        self._disconnected.set()

    async def _on_reconnect(self) -> None:
        """Handle socket reconnection attempts."""
        _LOGGER.debug("WS: reconnect event")
        await self._subscribe_heater_samples()

    async def _on_connect_error(self, data: Any) -> None:
        """Log ``connect_error`` events with their payload."""

        if _LOGGER.isEnabledFor(logging.DEBUG):
            _LOGGER.debug("WS: connect_error payload: %s", data)

    async def _on_error(self, data: Any) -> None:
        """Log socket.io ``error`` events with full details."""

        if _LOGGER.isEnabledFor(logging.DEBUG):
            _LOGGER.debug("WS: error event payload: %s", data)

    async def _on_reconnect_failed(self, data: Any | None = None) -> None:
        """Log ``reconnect_failed`` events with the reported context."""

        if _LOGGER.isEnabledFor(logging.DEBUG):
            _LOGGER.debug("WS: reconnect_failed details: %s", data)

    async def _on_namespace_disconnect(self, reason: Any | None = None) -> None:
        """Log namespace-level disconnect callbacks with their reason."""

        if _LOGGER.isEnabledFor(logging.DEBUG):
            _LOGGER.debug("WS: namespace disconnect (%s): %s", self._namespace, reason)

    async def _on_dev_handshake(self, data: Any) -> None:
        """Handle the ``dev_handshake`` payload."""
        self._stats.frames_total += 1
        if not self._handshake_logged and _LOGGER.isEnabledFor(logging.DEBUG):
            _LOGGER.debug("WS: dev_handshake payload: %s", data)
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
                _LOGGER.info("WS: idle for %.0fs; refreshing websocket lease", idle_for)
                try:
                    await self._refresh_subscription(reason="idle monitor")
                except asyncio.CancelledError:  # pragma: no cover - task lifecycle
                    raise
                except Exception:
                    self._schedule_idle_restart(
                        idle_for=idle_for, source="idle monitor refresh failed"
                    )
                    break
                continue
            if self._subscription_refresh_failed:
                # Retry quickly if the last scheduled renewal failed.
                _LOGGER.info(
                    "WS: retrying websocket lease after failure; idle for %.0fs",
                    idle_for,
                )
                try:
                    await self._refresh_subscription(reason="idle monitor retry")
                except asyncio.CancelledError:  # pragma: no cover - task lifecycle
                    raise
                except Exception:
                    self._schedule_idle_restart(
                        idle_for=idle_for, source="idle monitor retry failed"
                    )
                    break

    async def _refresh_subscription(self, *, reason: str) -> None:
        """Re-request device data to keep the websocket session active."""

        async with self._subscription_refresh_lock:
            if not self._sio.connected:
                raise RuntimeError("websocket not connected")
            now = time.time()
            self._subscription_refresh_last_attempt = now
            if _LOGGER.isEnabledFor(logging.INFO):
                _LOGGER.info("WS: refreshing websocket lease (%s)", reason)
            await self._sio.emit("dev_data", namespace=self._namespace)
            await self._subscribe_heater_samples()
            self._subscription_refresh_failed = False
            self._subscription_refresh_last_success = time.time()
    # ------------------------------------------------------------------
    # Payload handlers
    # ------------------------------------------------------------------
    def _handle_handshake(self, data: Any) -> None:
        """Process the initial handshake payload from the server."""
        if isinstance(data, dict):
            self._handshake_payload = deepcopy(data)
            if _LOGGER.isEnabledFor(logging.DEBUG):
                _LOGGER.debug(
                    "WS: dev_handshake payload keys: %s", ", ".join(sorted(data.keys()))
                )
            self._update_status("connected")
        else:
            _LOGGER.debug("WS: invalid handshake payload")

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
            nodes = self._translate_path_update(payload)
        if nodes is None:
            _LOGGER.debug("WS: %s without nodes", event)
            return
        normaliser = getattr(self._client, "normalise_ws_nodes", None)
        if callable(normaliser):
            try:
                nodes = normaliser(nodes)  # type: ignore[arg-type]
            except Exception:  # pragma: no cover - defensive logging only
                _LOGGER.debug("WS: normalise_ws_nodes failed; using raw payload")
        if _LOGGER.isEnabledFor(logging.DEBUG):
            changed = self._collect_update_addresses(nodes)
            if merge:
                if changed:
                    _LOGGER.debug(
                        "WS: update event for %s",
                        ", ".join(f"{node_type}/{addr}" for node_type, addr in changed),
                    )
                else:
                    _LOGGER.debug("WS: update event without address changes")
            else:
                _LOGGER.debug(
                    "WS: dev_data snapshot contains %d node groups", len(nodes)
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

    def _translate_path_update(self, payload: Any) -> dict[str, Any] | None:
        """Translate Ducaheat ``{"path": ..., "body": ...}`` frames into nodes."""

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
        remainder = (
            relevant[section_idx + 1 :]
            if len(relevant) > section_idx + 1
            else []
        )

        target_section, nested_key = self._resolve_update_section(section)
        if target_section is None:
            return None

        payload_body: Any = body
        for segment in reversed(remainder):
            payload_body = {segment: payload_body}
        if nested_key:
            payload_body = {nested_key: payload_body}

        return {node_type: {target_section: {addr: payload_body}}}

    @staticmethod
    def _resolve_update_section(section: str | None) -> tuple[str | None, str | None]:
        """Map a websocket path segment onto the node bucket name."""

        return resolve_ws_update_section(section)

    def _forward_sample_updates(self, updates: Mapping[str, Mapping[str, Any]]) -> None:
        """Relay websocket heater sample updates to the energy coordinator."""

        forward_ws_sample_updates(
            self.hass,
            self.entry_id,
            self.dev_id,
            updates,
            logger=_LOGGER,
            log_prefix="WS",
        )

    def _extract_nodes(self, data: Any) -> dict[str, Any] | None:
        """Extract the nodes dictionary from a websocket payload."""
        if not isinstance(data, dict):
            return None
        nodes = data.get("nodes")
        if isinstance(nodes, dict):
            return nodes
        if isinstance(nodes, list):
            mapped = self._translate_nodes_list(nodes)
            data["nodes"] = mapped
            return mapped
        return None

    def _translate_nodes_list(
        self, nodes: Iterable[Any]
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Convert list-based node payloads into the nested mapping schema."""

        translated: dict[str, dict[str, dict[str, Any]]] = {}
        for entry in nodes:
            if not isinstance(entry, Mapping):
                continue
            node_type = normalize_node_type(entry.get("type"))
            addr = normalize_node_addr(entry.get("addr"))
            if not node_type or not addr:
                continue
            node_bucket = translated.setdefault(node_type, {})
            added = False
            for key, value in entry.items():
                if key in {"type", "addr"}:
                    continue
                if not isinstance(key, str):
                    continue
                section, nested_key = self._resolve_update_section(key)
                if section is None:
                    continue
                section_bucket = node_bucket.setdefault(section, {})
                if nested_key:
                    existing = section_bucket.get(addr)
                    if isinstance(existing, Mapping):
                        merged = dict(existing)
                    else:
                        merged = {}
                    merged[nested_key] = deepcopy(value)
                    section_bucket[addr] = merged
                else:
                    section_bucket[addr] = deepcopy(value)
                added = True
            if not added and not node_bucket:
                translated.pop(node_type, None)
        return translated
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
            snapshot = {"nodes": None, "nodes_by_type": {}}

        context = _prepare_nodes_dispatch(
            self.hass,
            entry_id=self.entry_id,
            coordinator=self._coordinator,
            raw_nodes=raw_nodes,
        )

        inventory = context.inventory
        addr_map = context.addr_map
        unknown_types = context.unknown_types
        record = context.record
        snapshot_obj = context.snapshot
        raw_nodes_payload = inventory.payload if inventory.payload is not None else {}
        if unknown_types:  # pragma: no cover - diagnostic branch
            _LOGGER.debug(
                "WS: unknown node types in inventory: %s",
                ", ".join(sorted(unknown_types)),
            )

        if not is_snapshot:  # pragma: no cover - legacy branch
            nodes_by_type = {
                node_type: {"addrs": list(addrs)}
                for node_type, addrs in addr_map.items()
            }
            snapshot["nodes_by_type"] = nodes_by_type
            if "htr" in nodes_by_type:
                snapshot.setdefault("htr", nodes_by_type["htr"])
            snapshot["nodes"] = deepcopy(raw_nodes_payload)

        if isinstance(record, MutableMapping) and snapshot_obj is None:
            record["nodes"] = raw_nodes_payload
            record["node_inventory"] = list(inventory.nodes)

        self._apply_heater_addresses(
            addr_map,
            inventory=inventory,
            snapshot=snapshot_obj,
        )

        payload_copy = {
            "dev_id": self.dev_id,
            "node_type": None,
            "nodes": deepcopy(snapshot.get("nodes", raw_nodes_payload)),
            "nodes_by_type": deepcopy(snapshot.get("nodes_by_type", {})),
        }
        payload_copy.setdefault(
            "addr_map",
            {node_type: list(addrs) for node_type, addrs in addr_map.items()},
        )
        if unknown_types:  # pragma: no cover - diagnostic payload
            payload_copy.setdefault("unknown_types", sorted(unknown_types))

        def _send() -> None:
            """Fire the dispatcher signal with the latest node payload."""

            async_dispatcher_send(
                self.hass, signal_ws_data(self.entry_id), payload_copy
            )

        loop = getattr(self.hass, "loop", None)
        call_soon = getattr(loop, "call_soon_threadsafe", None)
        if callable(call_soon):
            call_soon(_send)
        else:  # pragma: no cover - legacy hass loop stub
            _send()

        return {node_type: list(addrs) for node_type, addrs in addr_map.items()}

    def _ensure_type_bucket(
        self, dev_map: dict[str, Any], nodes_by_type: dict[str, Any], node_type: str
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
        normalized_map: Mapping[Any, Iterable[Any]] | None,
        *,
        inventory: Inventory | None = None,
        snapshot: InstallationSnapshot | None = None,
    ) -> dict[str, list[str]]:
        """Update entry and coordinator state with heater address data."""

        cleaned_map: dict[str, list[str]] = {}
        if isinstance(normalized_map, Mapping):
            for raw_type, addrs in normalized_map.items():
                node_type = normalize_node_type(raw_type)
                if not node_type:
                    continue
                if node_type not in HEATER_NODE_TYPES:
                    continue
                if isinstance(addrs, (str, bytes)):
                    addr_iterable: Iterable[Any] = [addrs]
                else:
                    addr_iterable = addrs or []
                addresses: list[str] = []
                for candidate in addr_iterable:
                    addr = normalize_node_addr(candidate)
                    if not addr or addr in addresses:
                        continue
                    addresses.append(addr)
                if addresses:
                    cleaned_map[node_type] = addresses
                else:
                    cleaned_map.setdefault(node_type, [])

        cleaned_map.setdefault("htr", [])

        record_container = self.hass.data.get(DOMAIN, {})
        record = record_container.get(self.entry_id) if isinstance(record_container, dict) else None
        snapshot_obj = (
            snapshot
            if isinstance(snapshot, InstallationSnapshot)
            else ensure_snapshot(record)
        )

        inventory_nodes: list[Any] | None
        if isinstance(inventory, Inventory):
            inventory_nodes = list(inventory.nodes)
        elif inventory is not None:
            inventory_nodes = list(inventory)
            inventory = Inventory(self.dev_id, snapshot_obj.raw_nodes if snapshot_obj else {}, inventory_nodes)
        else:
            inventory_nodes = None

        if isinstance(snapshot_obj, InstallationSnapshot) and inventory is not None:
            snapshot_obj.update_nodes(snapshot_obj.raw_nodes, inventory=inventory)
            if isinstance(record, dict):
                record["node_inventory"] = list(snapshot_obj.inventory)
        elif isinstance(record, dict) and inventory_nodes is not None and snapshot_obj is None:
            record["node_inventory"] = list(inventory_nodes)

        energy_coordinator = (
            record.get("energy_coordinator") if isinstance(record, Mapping) else None
        )
        if hasattr(energy_coordinator, "update_addresses"):
            energy_coordinator.update_addresses(cleaned_map)

        coordinator_data = getattr(self._coordinator, "data", None)
        if isinstance(coordinator_data, dict):
            dev_map = coordinator_data.get(self.dev_id)
            if isinstance(dev_map, dict):
                nodes_by_type: dict[str, Any] = dev_map.setdefault("nodes_by_type", {})
                for node_type, addrs in cleaned_map.items():
                    if not addrs and node_type != "htr":
                        continue
                    bucket = self._ensure_type_bucket(dev_map, nodes_by_type, node_type)
                    if addrs:
                        bucket["addrs"] = list(addrs)
                updated = dict(coordinator_data)
                updated[self.dev_id] = dev_map
                self._coordinator.data = updated  # type: ignore[attr-defined]

        return cleaned_map

    def _heater_sample_subscription_targets(self) -> list[tuple[str, str]]:
        """Return ordered ``(node_type, addr)`` heater sample subscriptions."""

        record_container = self.hass.data.get(DOMAIN, {})
        record = record_container.get(self.entry_id) if isinstance(record_container, dict) else None
        snapshot_obj = ensure_snapshot(record)

        inventory_nodes: list[Any]
        normalized_map: dict[str, list[str]]
        inventory_container: Inventory | None
        if isinstance(snapshot_obj, InstallationSnapshot):
            inventory_nodes = list(snapshot_obj.inventory)
            normalized_map, _ = snapshot_obj.heater_sample_address_map
            inventory_container = Inventory(
                snapshot_obj.dev_id,
                snapshot_obj.raw_nodes,
                inventory_nodes,
            )
        else:
            nodes_payload: Any | None = None
            inventory_nodes = []
            if isinstance(record, MutableMapping):
                cached_inventory = record.get("node_inventory")
                if isinstance(cached_inventory, list):
                    inventory_nodes = list(cached_inventory)
                nodes_payload = record.get("nodes")
                if not inventory_nodes:
                    inventory_nodes = ensure_node_inventory(record, nodes=nodes_payload)
            elif isinstance(record, Mapping):
                nodes_payload = record.get("nodes")
                inventory_nodes = ensure_node_inventory(dict(record), nodes=nodes_payload)
            else:
                nodes_payload = None
                inventory_nodes = []

            container = Inventory(
                self.dev_id,
                nodes_payload if nodes_payload is not None else {},
                inventory_nodes,
            )
            normalized_map, _ = container.heater_sample_address_map
            inventory_container = container

        if not normalized_map.get("htr"):
            fallback: Iterable[Any] | None = None
            if hasattr(self._coordinator, "_addrs"):
                try:
                    fallback = self._coordinator._addrs()  # type: ignore[attr-defined]  # noqa: SLF001
                except Exception:  # pragma: no cover - defensive  # noqa: BLE001
                    fallback = None
            if fallback:
                normalised = list(normalized_map.get("htr", []))
                seen = set(normalised)
                for candidate in fallback:
                    addr = normalize_node_addr(candidate)
                    if not addr or addr in seen:
                        continue
                    seen.add(addr)
                    normalised.append(addr)
                if normalised:
                    normalized_map = dict(normalized_map)
                    normalized_map["htr"] = normalised

        normalized_map = self._apply_heater_addresses(
            normalized_map,
            inventory=inventory_container,
            snapshot=snapshot_obj,
        )

        other_types = sorted(node_type for node_type in normalized_map if node_type != "htr")
        order = ["htr", *other_types]
        return [
            (node_type, addr)
            for node_type in order
            for addr in normalized_map.get(node_type, []) or []
        ]

    async def _subscribe_heater_samples(self) -> None:
        """Subscribe to heater and accumulator sample updates."""

        try:
            for node_type, addr in self._heater_sample_subscription_targets():
                await self._sio.emit(
                    "subscribe",
                    f"/{node_type}/{addr}/samples",
                    namespace=self._namespace,
                )
        except asyncio.CancelledError:  # pragma: no cover - task lifecycle
            raise
        except Exception:
            _LOGGER.debug("WS: sample subscription setup failed", exc_info=True)

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

    def _mark_event(
        self, *, paths: list[str] | None, count_event: bool = False
    ) -> None:
        """Record receipt of a websocket event batch for health tracking."""
        now = time.time()
        self._cancel_idle_restart()
        self._stats.last_event_ts = now
        self._last_event_at = now
        self._last_payload_at = now
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
        state["last_payload_at"] = self._last_payload_at
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
        self._ws_state_bucket()["idle_restart_pending"] = True
        _LOGGER.warning(
            "WS: no payloads for %.0f s (%s heartbeat); restarting", idle_for, source
        )

        async def _restart() -> None:
            try:
                await self._disconnect(reason="idle restart")
            finally:
                self._idle_restart_pending = False
                self._idle_restart_task = None
                self._ws_state_bucket()["idle_restart_pending"] = False

        self._idle_restart_task = self._loop.create_task(_restart())

    def _cancel_idle_restart(self) -> None:
        """Cancel any scheduled idle restart due to new payload activity."""

        task = self._idle_restart_task
        if task and not task.done():
            task.cancel()
        self._idle_restart_task = None
        self._idle_restart_pending = False
        self._ws_state_bucket()["idle_restart_pending"] = False

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
        self._dispatcher = async_dispatcher_send
        self._session = session or getattr(api_client, "_session", None)
        if self._session is None:
            raise RuntimeError("aiohttp session required for websocket client")
        self._protocol_hint = protocol
        self._loop = getattr(hass, "loop", None) or asyncio.get_event_loop()
        self._task: asyncio.Task | None = None
        self._namespace = WS_NAMESPACE
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._disconnected = asyncio.Event()
        self._disconnected.set()

        self._user_agent = get_brand_user_agent(BRAND_TERMOWEB)
        self._requested_with = get_brand_requested_with(BRAND_TERMOWEB)

        self._closing = False
        self._connected_since: float | None = None
        self._healthy_since: float | None = None
        self._hb_send_interval: float = 27.0
        self._hb_task: asyncio.Task | None = None
        self._rtc_keepalive_task: asyncio.Task | None = None
        self._rtc_keepalive_interval: float = 30.0

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
        self._last_payload_at: float | None = None
        self._last_heartbeat_at: float | None = None

        self._handshake_payload: dict[str, Any] | None = None
        self._nodes: dict[str, Any] = {}
        self._nodes_raw: dict[str, Any] = {}

        self._payload_idle_window: float = 240.0
        self._idle_restart_task: asyncio.Task | None = None
        self._idle_restart_pending = False
        self._idle_monitor_task: asyncio.Task | None = None

        self._subscription_refresh_lock = asyncio.Lock()
        self._subscription_refresh_failed = False
        self._subscription_refresh_last_attempt: float = 0.0
        self._subscription_refresh_last_success: float | None = None

        self._ws_state: dict[str, Any] | None = None
        state = self._ws_state_bucket()
        state.setdefault("last_payload_at", None)
        state.setdefault("idle_restart_pending", False)

        self._write_hook_installed = False
        self._write_hook_original: Callable[..., Awaitable[Any]] | None = None
        self._install_write_hook()

    # ------------------------------------------------------------------
    # Public control
    # ------------------------------------------------------------------
    def _install_write_hook(self) -> None:
        """Wrap REST writes so we can observe successful node updates."""

        if self._write_hook_installed:
            return
        client = self._client
        original = getattr(client, "set_node_settings", None)
        if not callable(original):
            self._write_hook_installed = True
            return

        watchers: dict[str, weakref.WeakSet[TermoWebWSClient]]
        watchers = getattr(client, "_tw_ws_write_watchers", None)
        if watchers is None:
            watchers = {}
            setattr(client, "_tw_ws_write_watchers", watchers)

        dev_watchers = watchers.get(self.dev_id)
        if dev_watchers is None:
            dev_watchers = weakref.WeakSet()
            watchers[self.dev_id] = dev_watchers
        dev_watchers.add(self)

        if not hasattr(client, "_tw_ws_write_wrapper"):

            @wraps(original)
            async def _wrapped_set_node_settings(*args: Any, **kwargs: Any) -> Any:
                result = await original(*args, **kwargs)
                dev_id_arg = kwargs.get("dev_id") if "dev_id" in kwargs else None
                if dev_id_arg is None and args:
                    dev_id_arg = args[0]
                if isinstance(dev_id_arg, str):
                    hooks = getattr(client, "_tw_ws_write_watchers", {})
                    watchers_for_dev = hooks.get(dev_id_arg)
                    if watchers_for_dev:
                        for ws_client in list(watchers_for_dev):
                            maybe = getattr(
                                ws_client, "maybe_restart_after_write", None
                            )
                            if maybe is None or not callable(maybe):
                                continue
                            try:
                                await maybe()
                            except (
                                asyncio.CancelledError
                            ):  # pragma: no cover - passthrough
                                raise
                            except Exception:
                                _LOGGER.debug(
                                    "WS: maybe_restart_after_write failed: %s",
                                    ws_client.dev_id,
                                    exc_info=True,
                                )
                        if not watchers_for_dev:
                            hooks.pop(dev_id_arg, None)
                return result

            setattr(client, "set_node_settings", _wrapped_set_node_settings)
            setattr(client, "_tw_ws_write_wrapper", _wrapped_set_node_settings)
            setattr(client, "_tw_ws_write_original", original)

        self._write_hook_original = getattr(client, "_tw_ws_write_original", original)
        self._write_hook_installed = True

    async def maybe_restart_after_write(self) -> None:
        """Restart the websocket if writes follow long periods of inactivity."""

        last_payload = self._last_payload_at
        if last_payload is None:
            last_payload = self._stats.last_event_ts or self._last_event_at
        if not last_payload:
            return
        idle_for = time.time() - last_payload
        if idle_for < self._payload_idle_window:
            return
        if _LOGGER.isEnabledFor(logging.INFO):
            _LOGGER.info(
                "WS: write acknowledged after %.0f s without payloads; restarting",
                idle_for,
            )
        self._schedule_idle_restart(idle_for=idle_for, source="write notification")

    async def stop(self) -> None:
        """Cancel tasks, close websocket sessions and reset legacy state."""

        self._closing = True
        if self._hb_task:
            self._hb_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._hb_task
            self._hb_task = None
        if self._rtc_keepalive_task:
            self._rtc_keepalive_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._rtc_keepalive_task
            self._rtc_keepalive_task = None
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
                _LOGGER.debug("WS: initiating Socket.IO 0.9 handshake")
                sid, hb_timeout = await self._handshake()
                _LOGGER.debug(
                    "WS: handshake succeeded sid=%s hb_timeout=%s", sid, hb_timeout
                )
                self._hs_fail_count = 0
                self._hs_fail_start = 0.0
                self._hb_send_interval = max(5.0, min(30.0, hb_timeout * 0.45))
                await self._connect_ws(sid)
                _LOGGER.debug("WS: websocket connection established")
                await self._join_namespace()
                await self._send_snapshot_request()
                await self._subscribe_session_metadata()
                await self._subscribe_htr_samples()
                self._connected_since = time.time()
                self._healthy_since = None
                self._update_status("connected")
                if self._idle_monitor_task is None or self._idle_monitor_task.done():
                    _LOGGER.debug("WS: starting legacy idle monitor")
                    self._idle_monitor_task = self._loop.create_task(
                        self._idle_monitor()
                    )
                self._hb_task = self._loop.create_task(self._heartbeat_loop())
                if self._rtc_keepalive_task and not self._rtc_keepalive_task.done():
                    self._rtc_keepalive_task.cancel()
                self._rtc_keepalive_task = self._loop.create_task(
                    self._rtc_keepalive_loop()
                )
                await self._read_loop()
            except asyncio.CancelledError:
                should_retry = False
            except HandshakeError as err:
                self._hs_fail_count += 1
                if self._hs_fail_count == 1:
                    self._hs_fail_start = time.time()
                _LOGGER.info(
                    "WS: connection error (%s: %s); will retry", type(err).__name__, err
                )
                if self._hs_fail_count >= self._hs_fail_threshold:
                    elapsed = time.time() - self._hs_fail_start
                    _LOGGER.warning(
                        "WS: handshake failed %d times over %.1f s",
                        self._hs_fail_count,
                        elapsed,
                    )
                    self._hs_fail_count = 0
                    self._hs_fail_start = 0.0
                _LOGGER.debug(
                    "WS: handshake error url=%s body=%r",
                    self._sanitise_url(err.url),
                    err.body_snippet,
                )
            except Exception as err:
                _LOGGER.info(
                    "WS: connection error (%s: %s); will retry", type(err).__name__, err
                )
                _LOGGER.debug("WS: connection error details", exc_info=True)
            finally:
                if self._hb_task:
                    self._hb_task.cancel()
                    self._hb_task = None
                if self._rtc_keepalive_task:
                    self._rtc_keepalive_task.cancel()
                    self._rtc_keepalive_task = None
                if self._idle_monitor_task:
                    _LOGGER.debug("WS: stopping legacy idle monitor")
                    self._idle_monitor_task.cancel()
                    if hasattr(self._idle_monitor_task, "__await__"):
                        with suppress(asyncio.CancelledError):
                            await self._idle_monitor_task
                    self._idle_monitor_task = None
                self._subscription_refresh_failed = False
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
                headers = self._brand_headers(origin="https://localhost")
                async with self._session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=15),
                    headers=headers,
                ) as resp:
                    body = await resp.text()
                    if resp.status == 401:
                        _LOGGER.info("WS: handshake 401; refreshing token")
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
        _LOGGER.info("WS: connecting to %s", self._sanitise_url(ws_url))
        headers = self._brand_headers(origin="https://localhost")
        self._ws = await self._session.ws_connect(
            ws_url,
            timeout=aiohttp.ClientTimeout(total=15),
            heartbeat=None,
            autoclose=False,
            headers=headers,
        )

    async def _join_namespace(self) -> None:
        """Join the API namespace after the websocket connects."""

        await self._send_text(f"1::{self._namespace}")

    async def _send_snapshot_request(self) -> None:
        """Request the initial device snapshot."""

        payload = {"name": "dev_data", "args": []}
        await self._send_text(
            f"5::{self._namespace}:{json.dumps(payload, separators=(',', ':'))}"
        )

    async def _subscribe_session_metadata(self) -> None:
        """Subscribe to legacy session metadata updates."""

        payload = {"name": "subscribe", "args": ["/mgr/session"]}
        await self._send_text(
            f"5::{self._namespace}:{json.dumps(payload, separators=(',', ':'))}"
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
                f"5::{self._namespace}:{json.dumps(payload, separators=(',', ':'))}"
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

    async def _rtc_keepalive_loop(self) -> None:
        """Poll the REST API to keep the device session alive."""

        interval = self._rtc_keepalive_interval
        try:
            while not self._closing:
                try:
                    await self._client.get_rtc_time(self.dev_id)
                except asyncio.CancelledError:
                    raise
                except Exception as err:
                    if _LOGGER.isEnabledFor(logging.DEBUG):
                        _LOGGER.debug(
                            "WS: RTC keep-alive failed (%s: %s)",
                            type(err).__name__,
                            err,
                        )
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - defensive best effort
            if _LOGGER.isEnabledFor(logging.DEBUG):
                _LOGGER.debug(
                    "WS: RTC keep-alive loop stopped unexpectedly", exc_info=True
                )

    async def _read_loop(self) -> None:
        """Consume websocket frames and route events for the legacy protocol."""

        ws = self._ws
        if ws is None:
            return
        async for data in self._ws_payload_stream(ws, context="websocket"):
            if data.startswith("2::"):
                self._record_heartbeat(source="socketio09")
                continue
            if data.startswith(f"1::{self._namespace}"):
                continue
            if data.startswith(f"5::{self._namespace}:"):
                try:
                    payload = json.loads(data.split(f"5::{self._namespace}:", 1)[1])
                except Exception:
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

    def _mark_event(
        self, *, paths: list[str] | None, count_event: bool = False
    ) -> None:
        """Record payload batches and update the payload timestamp."""

        super()._mark_event(paths=paths, count_event=count_event)
        self._last_payload_at = self._stats.last_event_ts or self._last_event_at
        state = self._ws_state_bucket()
        state["last_payload_at"] = self._last_payload_at

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
        if (
            section == "settings"
            and normalize_node_type(node_type) == "acm"
            and isinstance(value, MutableMapping)
        ):
            coordinator = getattr(self, "_coordinator", None)
            apply_helper = getattr(coordinator, "_apply_accumulator_boost_metadata", None)
            if callable(apply_helper):
                now = None
                estimate = getattr(coordinator, "_device_now_estimate", None)
                if callable(estimate):
                    now = estimate()
                try:
                    apply_helper(value, now=now)
                except Exception as err:  # pragma: no cover - defensive
                    _LOGGER.debug(
                        "WS boost metadata derivation failed for %s/%s: %s",
                        node_type,
                        addr,
                        err,
                        exc_info=err,
                    )
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
            _LOGGER.debug("WS: legacy update for %s", ", ".join(sorted(addr_pairs)))
        elif updated_nodes:
            _LOGGER.debug("WS: legacy nodes refresh")

    async def _refresh_subscription(self, *, reason: str) -> None:
        """Replay subscription calls to keep the legacy websocket active."""

        async with self._subscription_refresh_lock:
            ws = self._ws
            if ws is None or getattr(ws, "closed", False):
                raise RuntimeError("websocket not connected")
            now = time.time()
            self._subscription_refresh_last_attempt = now
            _LOGGER.info("WS: refreshing websocket lease (%s)", reason)
            try:
                await self._send_snapshot_request()
                await self._subscribe_session_metadata()
                await self._subscribe_htr_samples()
            except asyncio.CancelledError:  # pragma: no cover - task lifecycle
                raise
            except Exception as err:
                self._subscription_refresh_failed = True
                _LOGGER.warning(
                    "WS: legacy lease refresh failed (%s: %s)",
                    type(err).__name__,
                    err,
                    exc_info=True,
                )
                self._schedule_idle_restart(
                    idle_for=self._payload_idle_window, source="lease refresh failure"
                )
                raise
            self._subscription_refresh_failed = False
            self._subscription_refresh_last_success = time.time()

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
                    code=aiohttp.WSCloseCode.GOING_AWAY, message=reason.encode()
                )
            self._ws = None

    async def _ws_payload_stream(
        self, ws: aiohttp.ClientWebSocketResponse, *, context: str
    ) -> AsyncIterator[str]:
        """Yield websocket payload strings."""

        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                self._stats.frames_total += 1
                yield msg.data
            elif msg.type == aiohttp.WSMsgType.BINARY:
                try:
                    decoded = codecs.decode(msg.data, "utf-8")
                except Exception:
                    continue
                self._stats.frames_total += 1
                yield decoded
            elif msg.type == aiohttp.WSMsgType.ERROR:
                raise RuntimeError(f"{context} error: {ws.exception()}")
            elif msg.type in {aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED}:
                raise RuntimeError(f"{context} closed")

    def _record_heartbeat(self, *, source: str) -> None:
        """Record receipt of a heartbeat frame."""

        _ = source
        now = time.time()
        self._stats.last_event_ts = now
        self._last_event_at = now
        self._last_heartbeat_at = now

    async def _idle_monitor(self) -> None:
        """Monitor payload idleness and restart stale websocket sessions."""

        while not self._closing:
            await asyncio.sleep(60)
            ws = self._ws
            if ws is None or getattr(ws, "closed", True):
                if self._disconnected.is_set():
                    break
                continue
            last_payload = self._last_payload_at
            now = time.time()
            idle_for: float | None = None
            if last_payload:
                idle_for = now - last_payload
                if idle_for >= self._payload_idle_window:
                    _LOGGER.info(
                        "WS: no payloads for %.0f s; scheduling websocket restart",
                        idle_for,
                    )
                    self._schedule_idle_restart(
                        idle_for=idle_for, source="idle monitor payload timeout"
                    )
                    break
            if self._subscription_refresh_failed:
                try:
                    await self._refresh_subscription(reason="idle monitor retry")
                except asyncio.CancelledError:  # pragma: no cover - task lifecycle
                    raise
                except Exception:
                    fallback_idle = (
                        idle_for if idle_for is not None else self._payload_idle_window
                    )
                    self._schedule_idle_restart(
                        idle_for=fallback_idle, source="idle monitor retry failed"
                    )
                    break

    async def _run_heartbeat(
        self, interval: float, send: Callable[[], Awaitable[Any]]
    ) -> None:
        """Send periodic heartbeats until cancelled."""

        while not self._closing:
            await asyncio.sleep(interval)
            try:
                await send()
            except Exception:
                return

    def _socket_base(self) -> str:
        """Return the base URL for websocket connections."""

        base = self._api_base().rstrip("/")
        parsed = urlsplit(base if base else API_BASE)
        scheme = parsed.scheme or "https"
        netloc = parsed.netloc or parsed.path
        path = parsed.path.rstrip("/")
        return urlunsplit((scheme, netloc, path or "", "", ""))


__all__ = [
    "HandshakeError",
    "TermoWebWSClient",
    "WSStats",
    "WebSocketClient",
    "build_node_inventory",
]
