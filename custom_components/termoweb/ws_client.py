"""Unified websocket client for TermoWeb backends."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass
import logging
import random
import time
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
    ensure_node_inventory,
    normalize_heater_addresses,
)

_LOGGER = logging.getLogger(__name__)

build_node_inventory = _build_node_inventory  # re-exported for tests


@dataclass
class WSStats:
    """Track websocket activity statistics."""

    frames_total: int = 0
    events_total: int = 0
    last_event_ts: float = 0.0
    last_paths: list[str] | None = None


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
        if hasattr(self._sio, "eio"):
            self._sio.eio.start_background_task = self._wrap_background_task
            if self._session is not None:
                self._sio.eio.http = self._session

        self._sio.on("connect", handler=self._on_connect)
        self._sio.on("disconnect", handler=self._on_disconnect)
        self._sio.on("reconnect", handler=self._on_reconnect)
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
        engineio_path = f"{path.strip('/')}/socket_io"
        query = urlencode({"token": token, "dev_id": self.dev_id})
        url = urlunsplit((scheme, netloc, "", query, ""))
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
        self._update_status("connected")
        if self._idle_monitor_task is None or self._idle_monitor_task.done():
            self._idle_monitor_task = self._loop.create_task(self._idle_monitor())
        try:
            await self._sio.emit("join", namespace=WS_NAMESPACE)
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
        self._disconnected.set()

    async def _on_reconnect(self) -> None:
        """Handle socket reconnection attempts."""
        _LOGGER.debug("WS %s: reconnect event", self.dev_id)

    async def _on_dev_handshake(self, data: Any) -> None:
        """Handle the ``dev_handshake`` payload."""
        self._stats.frames_total += 1
        self._handle_handshake(data)

    async def _on_dev_data(self, data: Any) -> None:
        """Handle the ``dev_data`` payload."""
        self._stats.frames_total += 1
        self._handle_dev_data(data)

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
                self._schedule_idle_restart(
                    idle_for=idle_for, source="idle monitor"
                )
                break

    # ------------------------------------------------------------------
    # Payload handlers
    # ------------------------------------------------------------------
    def _handle_handshake(self, data: Any) -> None:
        """Process the initial handshake payload from the server."""
        if isinstance(data, dict):
            self._handshake_payload = deepcopy(data)
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
        if merge and self._nodes_raw:
            self._merge_nodes(self._nodes_raw, nodes)
        else:
            self._nodes_raw = deepcopy(nodes)
        self._nodes = self._build_nodes_snapshot(self._nodes_raw)
        self._dispatch_nodes(self._nodes)
        self._mark_event(paths=None, count_event=True)

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
# Backwards compatibility aliases
# ----------------------------------------------------------------------
WebSocket09Client = WebSocketClient


class DucaheatWSClient(WebSocketClient):
    """Verbose websocket client variant with payload debug logging."""

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
        """Log initial dev_data snapshots with raw payload details."""
        if _LOGGER.isEnabledFor(logging.DEBUG):
            _LOGGER.debug("WS %s (ducaheat): dev_data payload: %s", self.dev_id, data)
        await super()._on_dev_data(data)

    async def _on_update(self, data: Any) -> None:
        """Log incremental update payloads before applying changes."""
        if _LOGGER.isEnabledFor(logging.DEBUG):
            _LOGGER.debug("WS %s (ducaheat): update payload: %s", self.dev_id, data)
        await super()._on_update(data)


__all__ = [
    "DucaheatWSClient",
    "WSStats",
    "WebSocket09Client",
    "WebSocketClient",
]
