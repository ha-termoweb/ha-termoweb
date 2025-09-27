"""Unified websocket client for TermoWeb backends."""

import asyncio
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass
import json
import logging
import random
import time
from typing import Any

import aiohttp
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_send

from .api import RESTClient
from .const import API_BASE, DOMAIN, WS_NAMESPACE, signal_ws_data, signal_ws_status
from .nodes import NODE_CLASS_BY_TYPE, build_node_inventory
from .utils import HEATER_NODE_TYPES, addresses_by_node_type, ensure_node_inventory

_LOGGER = logging.getLogger(__name__)

HandshakeResult = tuple[str, int]


@dataclass
class WSStats:
    """Track websocket activity statistics."""

    frames_total: int = 0
    events_total: int = 0
    last_event_ts: float = 0.0
    last_paths: list[str] | None = None


@dataclass
class EngineIOHandshake:
    """Details returned by the Engine.IO handshake."""

    sid: str
    ping_interval: float
    ping_timeout: float


class HandshakeError(RuntimeError):
    """Capture context for failed websocket handshakes."""

    def __init__(self, status: int, url: str, body_snippet: str) -> None:
        """Initialise the error with the HTTP response details."""
        super().__init__(f"handshake failed (status={status})")
        self.status = status
        self.url = url
        self.body_snippet = body_snippet


class TermoWebSocketClient:
    """Unified websocket client supporting legacy and Engine.IO endpoints."""

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
        """Initialise the websocket client container."""
        self.hass = hass
        self.entry_id = entry_id
        self.dev_id = dev_id
        self._client = api_client
        self._coordinator = coordinator
        self._session = session or getattr(api_client, "_session", None)
        self._task: asyncio.Task | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._engineio_ws: aiohttp.ClientWebSocketResponse | None = None

        self._closing = False
        self._connected_since: float | None = None
        self._healthy_since: float | None = None
        self._hb_send_interval: float = 27.0
        self._hb_task: asyncio.Task | None = None
        self._ping_task: asyncio.Task | None = None

        self._backoff_seq = [5, 10, 30, 120, 300]
        self._backoff_idx = 0
        self._hs_fail_count: int = 0
        self._hs_fail_start: float = 0.0
        self._hs_fail_threshold: int = handshake_fail_threshold

        self._stats = WSStats()
        self._protocol_hint = protocol
        self._protocol: str | None = None
        self._status: str = "stopped"
        self._stop_event = asyncio.Event()
        self._last_event_at: float | None = None
        self._engineio_sid: str | None = None
        self._engineio_ping_interval: float = 25.0
        self._engineio_ping_timeout: float = 60.0
        self._engineio_last_pong: float | None = None

        self._handshake: dict[str, Any] | None = None
        self._nodes: dict[str, Any] = {}
        self._nodes_raw: dict[str, Any] = {}

        if not hasattr(self.hass, "data") or self.hass.data is None:  # type: ignore[attr-defined]
            setattr(self.hass, "data", {})  # type: ignore[attr-defined]
        domain_bucket = self.hass.data.setdefault(DOMAIN, {})  # type: ignore[attr-defined]
        entry_bucket = domain_bucket.setdefault(self.entry_id, {})
        entry_bucket.setdefault("ws_state", {})

    # ------------------------------------------------------------------
    # Public control
    # ------------------------------------------------------------------
    def start(self) -> asyncio.Task:
        """Start the websocket client background task."""
        if self._task and not self._task.done():
            return self._task
        self._closing = False
        self._stop_event = asyncio.Event()
        self._task = self.hass.loop.create_task(
            self._runner(), name=f"{DOMAIN}-ws-{self.dev_id}"
        )
        return self._task

    async def stop(self) -> None:
        """Cancel tasks and close websocket sessions."""
        self._closing = True
        self._stop_event.set()
        if self._hb_task:
            self._hb_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._hb_task
            self._hb_task = None
        if self._ping_task:
            self._ping_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._ping_task
            self._ping_task = None
        if self._ws:
            with suppress(aiohttp.ClientError, RuntimeError):
                await self._ws.close(
                    code=aiohttp.WSCloseCode.GOING_AWAY, message=b"client stop"
                )
            self._ws = None
        if self._engineio_ws:
            with suppress(aiohttp.ClientError, RuntimeError):
                await self._engineio_ws.close(
                    code=aiohttp.WSCloseCode.GOING_AWAY, message=b"client stop"
                )
            self._engineio_ws = None
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
        base = getattr(self._client, "api_base", None)
        api_base = base.rstrip("/") if isinstance(base, str) and base else API_BASE
        return f"{api_base}/api/v2/socket_io?token={token}"

    # ------------------------------------------------------------------
    # Core loop and protocol dispatch
    # ------------------------------------------------------------------
    async def _runner(self) -> None:
        """Dispatch the appropriate websocket implementation."""
        self._update_status("starting")
        try:
            self._protocol = self._detect_protocol()
            if self._protocol == "engineio2":
                await self._run_engineio_v2()
            else:
                await self._run_socketio_09()
        finally:
            self._update_status("stopped")

    def _detect_protocol(self) -> str:
        """Return the websocket protocol to use for this client."""
        if self._protocol_hint:
            return self._protocol_hint
        base = getattr(self._client, "api_base", "") or ""
        base_lower = base.lower()
        if "tevolve" in base_lower or "/api/v2" in base_lower:
            return "engineio2"
        return "socketio09"

    # ------------------------------------------------------------------
    # Legacy Socket.IO 0.9 implementation
    # ------------------------------------------------------------------
    async def _run_socketio_09(self) -> None:
        """Manage reconnection loops and websocket lifecycle for Socket.IO 0.9."""
        while not self._closing:
            should_retry = True
            try:
                sid, hb_timeout = await self._handshake()
                self._hs_fail_count = 0
                self._hs_fail_start = 0.0
                self._hb_send_interval = max(5.0, min(30.0, hb_timeout * 0.45))
                await self._connect_ws_legacy(sid)
                await self._join_namespace()
                await self._send_snapshot_request()
                await self._subscribe_htr_samples()
                self._connected_since = time.time()
                self._healthy_since = None
                self._update_status("connected")
                self._hb_task = self.hass.loop.create_task(self._heartbeat_loop())
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
            self._backoff_idx = min(
                self._backoff_idx + 1, len(self._backoff_seq) - 1
            )
            jitter = random.uniform(0.8, 1.2)
            await asyncio.sleep(delay * jitter)

    async def _handshake(self) -> HandshakeResult:
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
                        token = await self._get_token()
                        base = self._api_base()
                        url = (
                            f"{base}/socket.io/1/?token={token}&dev_id={self.dev_id}"
                            f"&t={int(time.time() * 1000)}"
                        )
                        async with self._session.get(
                            url, timeout=aiohttp.ClientTimeout(total=15)
                        ) as resp2:
                            body = await resp2.text()
                            if resp2.status >= 400:
                                raise HandshakeError(resp2.status, url, body[:100])
                            sid, hb = self._parse_handshake_body(body)
                            self._backoff_idx = 0
                            return sid, hb
                    if resp.status >= 400:
                        raise HandshakeError(resp.status, url, body[:100])
                    sid, hb = self._parse_handshake_body(body)
                    self._backoff_idx = 0
                    return sid, hb
        except (TimeoutError, aiohttp.ClientError) as err:
            raise HandshakeError(-1, url, str(err)) from err

    async def _connect_ws_legacy(self, sid: str) -> None:
        """Establish the websocket connection using the handshake session id."""
        token = await self._get_token()
        base = self._api_base()
        ws_base = base.replace("https://", "wss://", 1)
        ws_url = (
            f"{ws_base}/socket.io/1/websocket/{sid}?token={token}&dev_id={self.dev_id}"
        )
        self._ws = await self._session.ws_connect(
            ws_url,
            heartbeat=None,
            timeout=15,
            autoclose=True,
            autoping=False,
            compress=0,
            protocols=("websocket",),
        )

    async def _join_namespace(self) -> None:
        """Enter the API namespace required for TermoWeb events."""
        await self._send_text(f"1::{WS_NAMESPACE}")

    async def _send_snapshot_request(self) -> None:
        """Request the initial device snapshot after connecting."""
        payload = {"name": "dev_data", "args": []}
        await self._send_text(
            f"5::{WS_NAMESPACE}:{json.dumps(payload, separators=(',', ':'))}"
        )

    async def _subscribe_htr_samples(self) -> None:  # noqa: C901
        """Request push updates for heater and accumulator energy samples."""
        inventory: list[Any] = []
        record: dict[str, Any] | None = None
        coordinator_inventory = getattr(self._coordinator, "_node_inventory", None)
        if isinstance(coordinator_inventory, list) and coordinator_inventory:
            inventory = list(coordinator_inventory)
        else:
            record = self.hass.data.get(DOMAIN, {}).get(self.entry_id)
            coordinator_nodes = getattr(self._coordinator, "_nodes", None)
            if isinstance(record, dict):
                inventory = ensure_node_inventory(record, nodes=coordinator_nodes)
            elif coordinator_nodes is not None:
                try:
                    inventory = build_node_inventory(coordinator_nodes)
                except Exception:  # pragma: no cover - defensive  # noqa: BLE001
                    inventory = []
        if record is None:
            record = self.hass.data.get(DOMAIN, {}).get(self.entry_id)
        addr_map: dict[str, list[str]] = {}
        if inventory:
            type_to_addrs, _ = addresses_by_node_type(
                inventory, known_types=NODE_CLASS_BY_TYPE
            )
            for node_type, addrs in type_to_addrs.items():
                if node_type in HEATER_NODE_TYPES and addrs:
                    addr_map[node_type] = list(addrs)
        if not addr_map and hasattr(self._coordinator, "_addrs"):
            try:
                fallback = list(self._coordinator._addrs())  # noqa: SLF001
            except Exception:  # pragma: no cover - defensive  # noqa: BLE001
                fallback = []
            if fallback:
                addr_map["htr"] = [str(addr).strip() for addr in fallback if str(addr).strip()]
        normalized_map: dict[str, list[str]] = {}
        for node_type in HEATER_NODE_TYPES:
            addrs = addr_map.get(node_type)
            if addrs:
                normalized_map[node_type] = list(addrs)
        normalized_map.setdefault("htr", list(addr_map.get("htr") or []))
        if not any(normalized_map.values()):
            return
        order = ["htr"] + [t for t in sorted(HEATER_NODE_TYPES) if t != "htr"]
        for node_type in order:
            addrs = normalized_map.get(node_type) or []
            for addr in addrs:
                payload = {"name": "subscribe", "args": [f"/{node_type}/{addr}/samples"]}
                await self._send_text(
                    f"5::{WS_NAMESPACE}:{json.dumps(payload, separators=(',', ':'))}"
                )
        if isinstance(record, dict):
            if inventory:
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
                    if addrs:
                        bucket["addrs"] = list(addrs)
                if "htr" in nodes_by_type:
                    dev_map["htr"] = nodes_by_type["htr"]
                updated = dict(coordinator_data)
                updated[self.dev_id] = dev_map
                self._coordinator.data = updated  # type: ignore[attr-defined]

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeat frames to keep the connection alive."""
        try:
            while True:
                await asyncio.sleep(self._hb_send_interval)
                await self._send_text("2::")
        except asyncio.CancelledError:
            raise
        except (aiohttp.ClientError, RuntimeError):
            return

    async def _read_loop(self) -> None:
        """Consume websocket frames and route events for the legacy protocol."""
        ws = self._ws
        if ws is None:
            return
        while True:
            msg = await ws.receive()
            if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE):
                exc = ws.exception()
                if exc is not None:
                    raise exc
                raise RuntimeError(
                    f"websocket closed: code={ws.close_code} reason={msg.extra}"
                )
            if msg.type == aiohttp.WSMsgType.ERROR:
                exc = ws.exception()
                if exc is not None:
                    raise exc
                raise RuntimeError("websocket error")
            if msg.type not in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY):
                continue
            data = (
                msg.data if isinstance(msg.data, str) else msg.data.decode("utf-8", "ignore")
            )
            self._stats.frames_total += 1
            if data.startswith("2::"):
                self._mark_event(paths=None)
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

    def _handle_event(self, evt: dict[str, Any]) -> None:  # noqa: C901
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

        def _ensure_type_bucket(node_type: str) -> dict[str, Any]:
            """Return the node bucket for ``node_type`` creating defaults."""
            nodes_by_type: dict[str, Any] = dev_map.setdefault("nodes_by_type", {})
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

        def _extract_type_addr(path: str) -> tuple[str | None, str | None]:
            """Extract the node type and address from a websocket path."""
            if not path:
                return None, None
            parts = [p for p in path.split("/") if p]
            for idx in range(len(parts) - 2):
                node_type = parts[idx]
                addr = parts[idx + 1]
                leaf = parts[idx + 2]
                if leaf in {"settings", "samples", "advanced_setup"}:
                    return node_type, addr
            return None, None

        for item in batch:
            if not isinstance(item, dict):
                continue
            path = item.get("path")
            body = item.get("body")
            if not isinstance(path, str):
                continue
            paths.append(path)
            dev_map: dict[str, Any] = (self._coordinator.data or {}).get(self.dev_id) or {}
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
            if path.endswith("/mgr/nodes"):
                if isinstance(body, dict):
                    dev_map["nodes"] = body
                    inventory: list[Any] = []
                    try:
                        inventory = build_node_inventory(body)
                    except Exception as err:  # pragma: no cover - defensive  # noqa: BLE001
                        _LOGGER.debug(
                            "WS %s: failed to build node inventory: %s",
                            self.dev_id,
                            err,
                            exc_info=err,
                        )
                    type_to_addrs, unknown_types = addresses_by_node_type(
                        inventory, known_types=NODE_CLASS_BY_TYPE
                    )
                    if unknown_types:
                        _LOGGER.debug(
                            "WS %s: unknown node types in inventory: %s",
                            self.dev_id,
                            ", ".join(sorted(unknown_types)),
                        )
                    for node_type, addrs in type_to_addrs.items():
                        bucket = _ensure_type_bucket(node_type)
                        bucket["addrs"] = list(addrs)
                    if "htr" not in dev_map and "htr" in type_to_addrs:
                        dev_map["htr"] = _ensure_type_bucket("htr")
                    if hasattr(self._coordinator, "update_nodes"):
                        self._coordinator.update_nodes(body, inventory)
                    record = self.hass.data.get(DOMAIN, {}).get(self.entry_id)
                    if isinstance(record, dict):
                        record["nodes"] = body
                        record["node_inventory"] = inventory
                        energy_coordinator = record.get("energy_coordinator")
                        if hasattr(energy_coordinator, "update_addresses"):
                            energy_coordinator.update_addresses(type_to_addrs)
                    updated_nodes = True
            else:
                node_type, addr = _extract_type_addr(path)
                if (
                    node_type
                    and addr
                    and path.endswith("/settings")
                    and node_type != "mgr"
                ):
                    bucket = _ensure_type_bucket(node_type)
                    settings_map: dict[str, Any] = bucket.setdefault("settings", {})
                    if isinstance(body, dict):
                        settings_map[addr] = body
                        updated_addrs.append((node_type, addr))
                    continue
                if (
                    node_type
                    and addr
                    and path.endswith("/advanced_setup")
                    and node_type != "mgr"
                ):
                    bucket = _ensure_type_bucket(node_type)
                    adv_map: dict[str, Any] = bucket.setdefault("advanced", {})
                    if isinstance(body, dict):
                        adv_map[addr] = body
                    continue
                if (
                    node_type
                    and addr
                    and path.endswith("/samples")
                    and node_type != "mgr"
                ):
                    bucket = _ensure_type_bucket(node_type)
                    samples_map: dict[str, Any] = bucket.setdefault("samples", {})
                    samples_map[addr] = body
                    sample_addrs.append((node_type, addr))
                    continue
                raw = dev_map.setdefault("raw", {})
                key = path.strip("/").replace("/", "_")
                raw[key] = body
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

    # ------------------------------------------------------------------
    # Engine.IO / Socket.IO v2 implementation
    # ------------------------------------------------------------------
    async def _run_engineio_v2(self) -> None:
        """Manage the Engine.IO websocket lifecycle."""
        self._update_status("connecting")
        if self._session is None:
            await self._stop_event.wait()
            return
        backoff_idx = 0
        while not self._closing:
            should_retry = True
            try:
                handshake = await self._engineio_handshake()
                self._engineio_sid = handshake.sid
                self._engineio_ping_interval = max(5.0, handshake.ping_interval)
                self._engineio_ping_timeout = max(5.0, handshake.ping_timeout)
                await self._engineio_connect(handshake.sid)
                self._connected_since = time.time()
                self._healthy_since = None
                self._update_status("connected")
                self._ping_task = self.hass.loop.create_task(self._engineio_ping_loop())
                await self._engineio_read_loop()
            except asyncio.CancelledError:
                should_retry = False
            except HandshakeError as err:
                _LOGGER.info(
                    "WS %s: connection error (%s: %s); will retry",
                    self.dev_id,
                    type(err).__name__,
                    err,
                )
                _LOGGER.debug(
                    "WS %s: handshake error url=%s body=%r",
                    self.dev_id,
                    err.url,
                    err.body_snippet,
                )
            except Exception as err:  # noqa: BLE001
                _LOGGER.info(
                    "WS %s: engine.io error (%s: %s); will retry",
                    self.dev_id,
                    type(err).__name__,
                    err,
                )
                _LOGGER.debug(
                    "WS %s: engine.io error details", self.dev_id, exc_info=True
                )
            finally:
                if self._ping_task:
                    self._ping_task.cancel()
                    self._ping_task = None
                if self._engineio_ws:
                    with suppress(aiohttp.ClientError, RuntimeError):
                        await self._engineio_ws.close()
                    self._engineio_ws = None
                self._update_status("disconnected")
            if self._closing or not should_retry:
                break
            delay = self._backoff_seq[min(backoff_idx, len(self._backoff_seq) - 1)]
            backoff_idx = min(backoff_idx + 1, len(self._backoff_seq) - 1)
            await asyncio.sleep(delay * random.uniform(0.8, 1.2))

    async def _engineio_handshake(self) -> EngineIOHandshake:
        """Perform the Engine.IO polling handshake to obtain session details."""
        token = await self._get_token()
        base = self._api_base()
        t_ms = int(time.time() * 1000)
        url = (
            f"{base}/socket.io/?EIO=3&transport=polling&token={token}&dev_id={self.dev_id}"
            f"&t={t_ms}"
        )
        try:
            async with asyncio.timeout(15):
                async with self._session.get(
                    url, timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    body = await resp.text()
                    if resp.status >= 400:
                        raise HandshakeError(resp.status, url, body[:100])
        except (TimeoutError, aiohttp.ClientError) as err:
            raise HandshakeError(-1, url, str(err)) from err
        return self._parse_engineio_handshake(body)

    @staticmethod
    def _parse_engineio_handshake(body: str) -> EngineIOHandshake:
        """Parse the Engine.IO handshake payload."""
        idx = body.find("{")
        if idx == -1:
            raise RuntimeError("handshake malformed")
        try:
            payload = json.loads(body[idx:])
        except (json.JSONDecodeError, TypeError) as err:
            raise RuntimeError("handshake malformed") from err
        sid = payload.get("sid")
        if not isinstance(sid, str) or not sid:
            raise RuntimeError("handshake missing sid")
        ping_interval = payload.get("pingInterval", 25000)
        ping_timeout = payload.get("pingTimeout", 60000)
        try:
            interval_s = float(ping_interval) / 1000.0
        except (TypeError, ValueError):
            interval_s = 25.0
        try:
            timeout_s = float(ping_timeout) / 1000.0
        except (TypeError, ValueError):
            timeout_s = 60.0
        return EngineIOHandshake(sid=sid, ping_interval=interval_s, ping_timeout=timeout_s)

    async def _engineio_connect(self, sid: str) -> None:
        """Upgrade the Engine.IO session to a websocket transport."""
        token = await self._get_token()
        base = self._api_base().replace("https://", "wss://", 1)
        url = (
            f"{base}/socket.io/?EIO=3&transport=websocket&sid={sid}&token={token}"
            f"&dev_id={self.dev_id}"
        )
        self._engineio_ws = await self._session.ws_connect(
            url,
            heartbeat=None,
            timeout=15,
            autoclose=True,
            autoping=False,
            compress=0,
            protocols=("websocket",),
        )
        await self._engineio_send("40")

    async def _engineio_ping_loop(self) -> None:
        """Send Engine.IO ping packets at the negotiated interval."""
        try:
            while True:
                await asyncio.sleep(self._engineio_ping_interval)
                await self._engineio_send("2")
        except asyncio.CancelledError:
            raise
        except (aiohttp.ClientError, OSError, RuntimeError):
            return

    async def _engineio_send(self, data: str) -> None:
        """Send an Engine.IO payload if the websocket is open."""
        if not self._engineio_ws:
            return
        await self._engineio_ws.send_str(data)

    async def _engineio_read_loop(self) -> None:
        """Consume Engine.IO websocket frames."""
        ws = self._engineio_ws
        if ws is None:
            await self._stop_event.wait()
            return
        while True:
            msg = await ws.receive()
            if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE):
                exc = ws.exception()
                if exc is not None:
                    raise exc
                raise RuntimeError(
                    f"engine.io websocket closed: code={ws.close_code} reason={msg.extra}"
                )
            if msg.type == aiohttp.WSMsgType.ERROR:
                exc = ws.exception()
                if exc is not None:
                    raise exc
                raise RuntimeError("engine.io websocket error")
            if msg.type not in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY):
                continue
            data = (
                msg.data if isinstance(msg.data, str) else msg.data.decode("utf-8", "ignore")
            )
            self._stats.frames_total += 1
            if data == "3":
                self._engineio_last_pong = time.time()
                self._mark_event(paths=None, count_event=True)
                continue
            if data == "2":
                await self._engineio_send("3")
                continue
            if data.startswith("42"):
                payload = data[2:]
                self._on_frame(payload)
                continue
            if data.startswith("41"):
                raise RuntimeError("engine.io server disconnect")

    def _on_frame(self, payload: str) -> None:
        """Process an incoming Socket.IO v2 frame payload."""
        try:
            message = json.loads(payload)
        except json.JSONDecodeError:
            _LOGGER.debug("WS %s: ignoring non-JSON frame", self.dev_id)
            return
        if not isinstance(message, dict):
            _LOGGER.debug("WS %s: ignoring non-dict frame", self.dev_id)
            return
        event = message.get("event")
        data = message.get("data")
        if event == "dev_handshake":
            self._handle_handshake(data)
        elif event == "dev_data":
            self._handle_dev_data(data)
        elif event == "update":
            self._handle_update(data)
        else:
            _LOGGER.debug("WS %s: unhandled event %s", self.dev_id, event)

    def _handle_handshake(self, data: Any) -> None:
        """Process the initial handshake payload from the server."""
        if isinstance(data, dict):
            self._handshake = deepcopy(data)
            self._update_status("connected")
        else:
            _LOGGER.debug("WS %s: invalid handshake payload", self.dev_id)

    def _handle_dev_data(self, data: Any) -> None:
        """Handle the first full snapshot of nodes from the websocket."""
        nodes = self._extract_nodes(data)
        if nodes is None:
            _LOGGER.debug("WS %s: dev_data without nodes", self.dev_id)
            return
        self._nodes_raw = deepcopy(nodes)
        self._nodes = self._build_nodes_snapshot(self._nodes_raw)
        self._dispatch_nodes(self._nodes)
        self._mark_event(paths=None, count_event=True)

    def _handle_update(self, data: Any) -> None:
        """Merge incremental node updates from the websocket feed."""
        nodes = self._extract_nodes(data)
        if nodes is None:
            _LOGGER.debug("WS %s: update without nodes", self.dev_id)
            return
        if not self._nodes_raw:
            self._nodes_raw = deepcopy(nodes)
        else:
            self._merge_nodes(self._nodes_raw, nodes)
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

    def _dispatch_nodes(self, snapshot: dict[str, Any]) -> None:
        """Publish the node snapshot to the coordinator and listeners."""
        record = self.hass.data.get(DOMAIN, {}).get(self.entry_id)
        inventory: list[Any] = []
        try:
            inventory = build_node_inventory(snapshot.get("nodes"))
        except Exception as err:  # pragma: no cover - defensive  # noqa: BLE001
            _LOGGER.debug(
                "WS %s: failed to build node inventory: %s",
                self.dev_id,
                err,
                exc_info=err,
            )
        if hasattr(self._coordinator, "update_nodes"):
            self._coordinator.update_nodes(snapshot.get("nodes"), inventory)
        if isinstance(record, dict):
            record["nodes"] = snapshot.get("nodes")
            record["node_inventory"] = inventory
            addr_map, unknown_types = addresses_by_node_type(
                inventory, known_types=NODE_CLASS_BY_TYPE
            )
            if unknown_types:
                _LOGGER.debug(
                    "WS %s: unknown node types in inventory: %s",
                    self.dev_id,
                    ", ".join(sorted(unknown_types)),
                )
            energy_coordinator = record.get("energy_coordinator")
            if hasattr(energy_coordinator, "update_addresses"):
                energy_coordinator.update_addresses(addr_map)
        payload = {
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
                payload,
            )

        self.hass.loop.call_soon_threadsafe(_send)

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
                    TermoWebSocketClient._merge_nodes(existing, value)
                else:
                    target[key] = deepcopy(value)
            else:
                target[key] = value

    # ------------------------------------------------------------------
    # Helpers shared across implementations
    # ------------------------------------------------------------------
    def _parse_handshake_body(self, body: str) -> HandshakeResult:
        """Parse the Socket.IO handshake response into (sid, timeout)."""
        parts = (body or "").strip().split(":")
        if len(parts) < 2:
            raise RuntimeError("handshake malformed")
        sid = parts[0]
        try:
            hb = int(parts[1])
        except (TypeError, ValueError):
            hb = 60
        return sid, hb

    async def _send_text(self, data: str) -> None:
        """Send a raw Socket.IO text frame if the websocket is open."""
        if not self._ws:
            return
        await self._ws.send_str(data)

    async def _get_token(self) -> str:
        """Reuse the REST client token for websocket authentication."""
        headers = await self._client._authed_headers()  # noqa: SLF001
        auth_header = headers.get("Authorization") if isinstance(headers, dict) else None
        if not auth_header:
            raise RuntimeError("authorization token missing")
        return auth_header.split(" ", 1)[1]

    async def _force_refresh_token(self) -> None:
        """Force the REST client to fetch a fresh access token."""
        with suppress(AttributeError):
            self._client._access_token = None  # type: ignore[attr-defined]  # noqa: SLF001
        await self._client._ensure_token()  # noqa: SLF001

    def _api_base(self) -> str:
        """Return the base REST API URL used for websocket routes."""
        base = getattr(self._client, "api_base", None)
        if isinstance(base, str) and base:
            return base.rstrip("/")
        return API_BASE

    def _update_status(self, status: str) -> None:
        """Publish the websocket status to Home Assistant listeners."""
        if status == self._status and status not in {"healthy", "connected"}:
            return
        self._status = status
        now = time.time()
        state_bucket = self.hass.data[DOMAIN][self.entry_id].setdefault("ws_state", {})
        state = state_bucket.setdefault(self.dev_id, {})
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
        state_bucket: dict[str, Any] = self.hass.data.setdefault(DOMAIN, {}).setdefault(
            self.entry_id, {}
        ).setdefault("ws_state", {})
        state: dict[str, Any] = state_bucket.setdefault(self.dev_id, {})
        state["last_event_at"] = now
        state["frames_total"] = self._stats.frames_total
        state["events_total"] = self._stats.events_total
        if self._protocol == "engineio2":
            if self._healthy_since is None:
                self._healthy_since = now
            self._update_status("healthy")
        elif (
            self._connected_since
            and not self._healthy_since
            and (now - self._connected_since) >= 300
        ):
            self._healthy_since = now


# ----------------------------------------------------------------------
# Backwards compatibility aliases
# ----------------------------------------------------------------------
WebSocket09Client = TermoWebSocketClient
DucaheatWSClient = TermoWebSocketClient

__all__ = [
    "DucaheatWSClient",
    "EngineIOHandshake",
    "HandshakeError",
    "TermoWebSocketClient",
    "WSStats",
    "WebSocket09Client",
]

