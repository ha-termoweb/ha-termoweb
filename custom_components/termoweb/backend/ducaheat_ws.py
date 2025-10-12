"""Ducaheat specific websocket client."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Iterable, Mapping, MutableMapping
from copy import deepcopy
import gzip
import json
import logging
import math
import random
import string
import time
from typing import Any
from urllib.parse import urlencode, urlsplit, urlunsplit

import aiohttp
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_send

from custom_components.termoweb.api import RESTClient
from custom_components.termoweb.backend.ws_client import (
    DUCAHEAT_NAMESPACE,
    HandshakeError,
    WSStats,
    _prepare_nodes_dispatch,
    _WSCommon,
    _WsLeaseMixin,
    forward_ws_sample_updates,
    resolve_ws_update_section,
    translate_path_update,
)
from custom_components.termoweb.backend.ws_health import WsHealthTracker
from custom_components.termoweb.const import (
    ACCEPT_LANGUAGE,
    API_BASE,
    BRAND_DUCAHEAT,
    DOMAIN,
    USER_AGENT,
    get_brand_api_base,
    get_brand_requested_with,
    get_brand_user_agent,
    signal_ws_data,
)
from custom_components.termoweb.inventory import (
    Inventory,
    normalize_node_addr,
    normalize_node_type,
    resolve_record_inventory,
)

_LOGGER = logging.getLogger(__name__)

_PAYLOAD_WINDOW_DEFAULT = 240.0
_PAYLOAD_WINDOW_MIN = 30.0
_PAYLOAD_WINDOW_MAX = 900.0
_PAYLOAD_WINDOW_MARGIN_RATIO = 0.25
_PAYLOAD_WINDOW_MARGIN_FLOOR = 15.0
_CADENCE_KEYS = ("lease_seconds", "cadence_seconds", "poll_seconds")
_NODE_METADATA_KEYS = {"type", "node_type", "addr", "address", "name", "title", "label"}
_NODE_TYPE_LEVEL_KEYS = {"lease_seconds", "cadence_seconds", "poll_seconds"}


def _rand_t() -> str:
    """Return a pseudo-random token string for polling requests."""

    return "P" + "".join(
        random.choice(string.ascii_letters + string.digits) for _ in range(7)
    )


def _encode_polling_packet(pkt: str) -> bytes:
    """Encode Engine.IO polling payloads."""

    data = pkt.encode("utf-8")
    return f"{len(data)}:".encode("ascii") + data


def _decode_polling_packets(body: bytes) -> list[str]:
    """Decode Engine.IO v3 binary polling frames."""

    buf = body
    if len(buf) >= 2 and buf[:2] == b"\x1f\x8b":
        try:
            buf = gzip.decompress(buf)
        except Exception:
            pass
    out: list[str] = []
    i, n = 0, len(buf)
    while i < n:
        if i + 4 > n:
            break
        _typ = buf[i]
        i += 1
        length = 0
        had_digit = False
        while i < n and buf[i] != 0xFF:
            d = buf[i]
            i += 1
            if d > 9:
                return out
            had_digit = True
            length = length * 10 + d
        if not had_digit or i >= n or buf[i] != 0xFF:
            return out
        i += 1
        end = i + length
        if end > n:
            return out
        payload = buf[i:end]
        i = end
        try:
            pkt = payload.decode("utf-8", errors="ignore")
        except Exception:
            pkt = ""
        if pkt:
            out.append(pkt)
    return out


def _brand_headers(user_agent: str, requested_with: str) -> dict[str, str]:
    """Construct brand-specific headers for polling."""

    return {
        "User-Agent": user_agent or USER_AGENT,
        "Accept-Language": ACCEPT_LANGUAGE,
        "X-Requested-With": requested_with,
        "Origin": "https://localhost",
        "Referer": "https://localhost/",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Connection": "keep-alive",
    }


class DucaheatWSClient(_WsLeaseMixin, _WSCommon):
    """Engine.IO v3 websocket client for the Ducaheat backend."""

    def __init__(
        self,
        hass: HomeAssistant,
        *,
        entry_id: str,
        dev_id: str,
        api_client: RESTClient,
        coordinator: Any,
        session: aiohttp.ClientSession | None = None,
        namespace: str | None = None,
        inventory: Inventory | None = None,
    ) -> None:
        _WsLeaseMixin.__init__(self)
        _WSCommon.__init__(self, inventory=inventory)
        self.hass = hass
        self.entry_id = entry_id
        self.dev_id = dev_id
        self._client = api_client
        self._coordinator = coordinator
        self._dispatcher = async_dispatcher_send
        self._session = session or getattr(api_client, "_session", None)
        if self._session is None:
            raise RuntimeError("aiohttp session required")
        self._namespace = namespace or DUCAHEAT_NAMESPACE
        self._loop = getattr(hass, "loop", None) or asyncio.get_event_loop()

        self._brand = BRAND_DUCAHEAT
        self._ua = get_brand_user_agent(self._brand)
        self._xrw = get_brand_requested_with(self._brand)

        self._task: asyncio.Task | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._stats = WSStats()
        self._nodes_raw: dict[str, Any] | None = None
        self._subscription_paths: set[str] = set()
        self._pending_dev_data = False
        self._keepalive_task: asyncio.Task | None = None
        self._ping_interval: float | None = None
        self._ping_timeout: float | None = None
        self._status: str = "stopped"
        self._healthy_since: float | None = None
        self._last_event_at: float | None = None
        self._default_payload_window = _PAYLOAD_WINDOW_DEFAULT
        self._payload_stale_after = self._default_payload_window
        self._payload_window_hint: float | None = None
        self._payload_window_source: str = "default"
        self._suppress_default_cadence_hint = True
        try:
            self._reset_payload_window(source="default")
        finally:
            self._suppress_default_cadence_hint = False
        self._pending_default_cadence_hint = True

    @property
    def _ws_health(self) -> WsHealthTracker:
        """Return the shared websocket health tracker for this client."""

        return self._ws_health_tracker()

    def _status_should_reset_health(self, status: str) -> bool:
        """Return True when a status transition should reset health."""

        return status != "healthy"

    def start(self) -> asyncio.Task:
        if self._task and not self._task.done():
            return self._task
        self._task = self._loop.create_task(
            self._runner(), name=f"termoweb-ws-{self.dev_id}"
        )
        return self._task

    async def stop(self) -> None:
        await self._disconnect("stop")
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def is_running(self) -> bool:
        return bool(self._task and not self._task.done())

    def _base_host(self) -> str:
        base = get_brand_api_base(self._brand).rstrip("/")
        parsed = urlsplit(base if base else API_BASE)
        return urlunsplit(
            (parsed.scheme or "https", parsed.netloc or parsed.path, "", "", "")
        )

    def _path(self) -> str:
        return "/socket.io/"

    def _build_handshake_url(self, params: Mapping[str, Any]) -> str:
        """Return a handshake URL with the provided query parameters."""

        parsed = urlsplit(self._base_host())
        query = urlencode(list(params.items()))
        return urlunsplit(parsed._replace(path=self._path(), query=query))

    async def ws_url(self) -> str:
        token = await self._get_token()
        params = {
            "token": token,
            "dev_id": self.dev_id,
            "EIO": "3",
            "transport": "polling",
            "t": _rand_t(),
        }
        return self._build_handshake_url(params)

    async def _runner(self) -> None:
        self._update_status("starting")
        try:
            while True:
                try:
                    await self._connect_once()
                    await self._read_loop_ws()
                except asyncio.CancelledError:
                    break
                except Exception as exc:  # pragma: no cover - defensive
                    _LOGGER.debug("WS (ducaheat): error %s", exc, exc_info=True)
                    await asyncio.sleep(self._next_backoff())
                finally:
                    await self._disconnect("loop")
                    self._update_status("disconnected")
        finally:
            self._update_status("stopped")

    async def _connect_once(self) -> None:
        token = await self._get_token()
        headers = _brand_headers(self._ua, self._xrw)

        open_params = {
            "token": token,
            "dev_id": self.dev_id,
            "EIO": "3",
            "transport": "polling",
            "t": _rand_t(),
        }
        open_url = self._build_handshake_url(open_params)
        # _LOGGER.debug("WS (ducaheat): OPEN GET %s", open_url.replace(token, "…"))
        async with asyncio.timeout(15):
            async with self._session.get(open_url, headers=headers) as resp:
                body = await resp.read()
                # _LOGGER.debug(
                #    "WS (ducaheat): OPEN GET -> %s bytes (status=%s)",
                #    len(body),
                #    resp.status,
                # )
                if resp.status != 200:
                    raise HandshakeError(resp.status, open_url, "open GET")
        packets = _decode_polling_packets(body)
        open_pkt = next((pkt for pkt in packets if pkt and pkt[0] == "0"), None)
        if not open_pkt:
            _LOGGER.debug("WS (ducaheat): open raw first 120 bytes: %r", body[:120])
            raise HandshakeError(590, open_url, "missing OPEN (0)")
        info = json.loads(open_pkt[1:])
        sid = info.get("sid")
        ping_interval = info.get("pingInterval")
        self._ping_interval = (
            float(ping_interval) / 1000
            if isinstance(ping_interval, (int, float))
            else None
        )
        ping_timeout = info.get("pingTimeout")
        self._ping_timeout = (
            float(ping_timeout) / 1000
            if isinstance(ping_timeout, (int, float))
            else None
        )
        _LOGGER.info(
            "WS (ducaheat): OPEN decoded sid=%s pingInterval=%s pingTimeout=%s",
            sid,
            info.get("pingInterval"),
            info.get("pingTimeout"),
        )
        if not isinstance(sid, str) or not sid:
            raise HandshakeError(592, open_url, "missing sid")

        post_params = {
            "token": token,
            "dev_id": self.dev_id,
            "EIO": "3",
            "transport": "polling",
            "t": _rand_t(),
            "sid": sid,
        }
        post_url = self._build_handshake_url(post_params)
        payload = _encode_polling_packet(f"40{self._namespace}")
        # _LOGGER.debug("WS (ducaheat): POST 40/ns %s", post_url.replace(token, "…"))
        async with asyncio.timeout(15):
            async with self._session.post(
                post_url,
                headers={**headers, "Content-Type": "text/plain;charset=UTF-8"},
                data=payload,
            ) as resp:
                await resp.read()
                #                _LOGGER.debug(
                #                    "WS (ducaheat): POST 40/ns -> status=%s len=%s",
                #                    resp.status,
                #                    len(drain),
                #                )
                if resp.status != 200:
                    raise HandshakeError(resp.status, post_url, "POST 40/ns")

        drain_params = dict(post_params)
        drain_params["t"] = _rand_t()
        drain_url = self._build_handshake_url(drain_params)
        #        _LOGGER.debug("WS (ducaheat): DRAIN GET %s", drain_url.replace(token, "…"))
        async with asyncio.timeout(15):
            async with self._session.get(drain_url, headers=headers) as resp:
                body = await resp.read()
                #                _LOGGER.debug(
                #                    "WS (ducaheat): DRAIN GET -> status=%s len=%s",
                #                    resp.status,
                #                    len(body),
                #                )
                if resp.status != 200:
                    raise HandshakeError(resp.status, drain_url, "drain GET")

        ws_params = {
            "token": token,
            "dev_id": self.dev_id,
            "EIO": "3",
            "transport": "websocket",
            "sid": sid,
        }
        ws_url = self._build_handshake_url(ws_params)
        ws_headers = {
            k: v
            for k, v in headers.items()
            if k.lower() not in ("connection", "accept-encoding")
        }
        #        _LOGGER.debug("WS (ducaheat): upgrading WS %s", ws_url.replace(token, "…"))
        self._ws = await self._session.ws_connect(
            ws_url,
            headers=ws_headers,
            heartbeat=None,
            autoclose=False,
            timeout=aiohttp.ClientTimeout(total=15),
        )
        #        _LOGGER.info("WS (ducaheat): upgrade OK")

        await self._ws.send_str("2probe")
        #        _LOGGER.debug("WS (ducaheat): -> 2probe")
        probe = await self._ws.receive_str()
        #        _LOGGER.debug("WS (ducaheat): <- %r", probe)
        if probe != "3probe":
            _LOGGER.debug("WS (ducaheat): unexpected probe ack: %r", probe)
        await self._ws.send_str("5")
        #        _LOGGER.debug("WS (ducaheat): -> 5 (upgrade)")

        await self._ws.send_str(f"40{self._namespace}")
        #        _LOGGER.debug("WS (ducaheat): -> 40%s", self._namespace)
        self._pending_dev_data = True
        #        _LOGGER.debug("WS (ducaheat): dev_data pending until namespace ack")
        self._healthy_since = None
        self._last_event_at = None
        self._stats.frames_total = 0
        self._stats.events_total = 0
        self._stats.last_event_ts = 0.0
        self._update_status("connected")
        self._start_keepalive()

    def _record_frame(self, *, timestamp: float | None = None) -> None:
        """Update cached websocket frame statistics and timestamps."""

        now = timestamp or time.time()
        self._stats.last_event_ts = now
        self._last_event_at = now
        self._mark_ws_heartbeat(timestamp=now)
        state = self._ws_state_bucket()
        state["last_event_at"] = now
        state["frames_total"] = self._stats.frames_total
        state["events_total"] = self._stats.events_total

    async def _read_loop_ws(self) -> None:
        ws = self._ws
        if ws is None:
            return
        try:
            while True:
                tracker = self._ws_health
                if tracker.payload_stale and tracker.status == "healthy":
                    self._update_status("connected")
                deadline = tracker.stale_deadline()
                timeout: float | None = None
                if isinstance(deadline, (int, float)):
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        now = time.time()
                        self._refresh_ws_payload_state(now=now, reason="payload_timeout")
                        if tracker.payload_stale and tracker.status == "healthy":
                            self._update_status("connected")
                        remaining = 1.0
                    timeout = max(1.0, remaining)
                try:
                    if timeout is None:
                        msg = await ws.receive()
                    else:
                        msg = await asyncio.wait_for(ws.receive(), timeout=timeout)
                except asyncio.TimeoutError:
                    now = time.time()
                    self._refresh_ws_payload_state(now=now, reason="payload_timeout")
                    tracker = self._ws_health
                    if tracker.payload_stale and tracker.status == "healthy":
                        self._update_status("connected")
                    continue

                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = msg.data
                    self._stats.frames_total += 1
                    now = time.time()
                    self._record_frame(timestamp=now)
                    while data:
                        if data == "2":
                            await ws.send_str("3")
                            break
                        if data == "3":
                            self._update_status("healthy")
                            break
                        if not data.startswith("4"):
                            break
                        if data.startswith("40"):
                            ns_payload = data[2:]
                            rest_payload = ""
                            if ns_payload:
                                namespace, sep, remainder = ns_payload.partition(",")
                                if namespace and namespace != self._namespace:
                                    _LOGGER.debug(
                                        "WS (ducaheat): <- SIO 40 for unexpected namespace %s",
                                        namespace,
                                    )
                                    if sep and remainder:
                                        data = (
                                            remainder
                                            if remainder.startswith("4")
                                            else "4" + remainder
                                        )
                                        continue
                                    break
                                if sep and remainder:
                                    rest_payload = remainder
                            #                            _LOGGER.debug("WS (ducaheat): <- SIO 40 (namespace ack)")
                            self._update_status("healthy")
                            if self._pending_dev_data:
                                self._pending_dev_data = False
                                try:
                                    await self._emit_sio("dev_data")
                                    await self._replay_subscription_paths()
                                except Exception:  # noqa: BLE001  # pragma: no cover - defensive
                                    _LOGGER.debug(
                                        "WS (ducaheat): failed to emit dev_data or replay subscriptions",
                                        exc_info=True,
                                    )
                            if rest_payload:
                                data = (
                                    rest_payload
                                    if rest_payload.startswith("4")
                                    else "4" + rest_payload
                                )
                                continue
                            break
                        payload = data[1:]
                        if payload == "2":
                            await ws.send_str("3")
                            break
                        if payload == "3":
                            self._update_status("healthy")
                            break
                        if payload.startswith("2/"):
                            ns_payload = payload[1:]
                            ns, sep, body = ns_payload.partition(",")
                            if not sep or body in {"", "[]", '["ping"]'}:
                                await ws.send_str("3" + ns)
                                break

                        content: str | None = None
                        if data.startswith("42"):
                            content = data[2:]
                        elif payload.startswith("42"):
                            content = payload[2:]
                        if content is None:
                            break
                        if content.startswith("/"):
                            _ns, sep, content = content.partition(",")
                            if sep != ",":
                                break
                        try:
                            arr = json.loads(content)
                        except Exception:
                            break
                        if not isinstance(arr, list) or not arr:
                            break
                        evt, *args = arr
                        self._stats.events_total += 1
                        self._record_frame(timestamp=now)
                        # _LOGGER.debug("WS (ducaheat): <- SIO 42 event=%s args_len=%d", evt, len(args))

                        if evt == "message" and args and args[0] == "ping":
                            await self._emit_sio("message", "pong")
                            break

                        if evt == "dev_data" and args:
                            payload_map = self._extract_dev_data_payload(args)
                            nodes_map: Mapping[str, Any] | None = None
                            if isinstance(payload_map, Mapping):
                                nodes_candidate = payload_map.get("nodes")
                                if isinstance(nodes_candidate, Mapping):
                                    nodes_map = nodes_candidate
                                else:
                                    nodes_map = self._coerce_dev_data_nodes(nodes_candidate)
                            if isinstance(nodes_map, Mapping):
                                self._log_nodes_summary(nodes_map)
                                normalised = self._normalise_nodes_payload(nodes_map)
                                dispatch_payload: Mapping[str, Any] | None
                                if isinstance(normalised, Mapping):
                                    self._nodes_raw = deepcopy(normalised)
                                    dispatch_payload = self._nodes_raw
                                else:
                                    self._nodes_raw = None
                                    dispatch_payload = nodes_map
                                if isinstance(dispatch_payload, Mapping):
                                    self._dispatch_nodes(dispatch_payload)
                                subs = await self._subscribe_feeds(nodes_map)
                                _LOGGER.info("WS (ducaheat): subscribed %d feeds", subs)
                                self._update_status("healthy")
                            break

                        if evt == "update" and args:
                            payload_body = args[0]
                            self._log_update_brief(payload_body)
                            translated = self._translate_path_update(payload_body)
                            if translated:
                                normalised_update = self._normalise_nodes_payload(
                                    translated
                                )
                                if (
                                    isinstance(normalised_update, dict)
                                    and normalised_update
                                ):
                                    sample_updates = self._collect_sample_updates(
                                        normalised_update
                                    )
                                    if self._nodes_raw is None:
                                        self._nodes_raw = deepcopy(normalised_update)
                                    else:
                                        self._merge_nodes(
                                            self._nodes_raw, normalised_update
                                        )
                                    dispatch_payload: Mapping[str, Any] | None
                                    if isinstance(self._nodes_raw, Mapping):
                                        dispatch_payload = self._nodes_raw
                                    else:
                                        dispatch_payload = normalised_update
                                    if isinstance(dispatch_payload, Mapping):
                                        self._dispatch_nodes(dispatch_payload)
                                    if sample_updates:
                                        self._forward_sample_updates(sample_updates)
                            self._update_status("healthy")
                            break
                        break
                        break
                    continue
                if msg.type == aiohttp.WSMsgType.ERROR:
                    raise RuntimeError(f"websocket error: {ws.exception()}")
                if msg.type in {aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED}:
                    raise RuntimeError("websocket closed")
        finally:
            _LOGGER.debug("WS (ducaheat): read loop ended (ws closed or error)")

    def _start_keepalive(self) -> None:
        """Launch the keepalive loop when the websocket is connected."""

        if self._keepalive_task and not self._keepalive_task.done():
            return
        if self._ping_interval is None or not self._ws or self._ws.closed:
            return
        self._keepalive_task = self._loop.create_task(
            self._keepalive_loop(),
            name=f"termoweb-ws-keepalive-{self.dev_id}",
        )

    async def _keepalive_loop(self) -> None:
        """Send Engine.IO ping frames to keep the websocket alive."""

        task = asyncio.current_task()
        try:
            while True:
                interval = self._ping_interval
                ws = self._ws
                if interval is None or ws is None or ws.closed:
                    break
                delay = max(0.1, interval * 0.9)
                await asyncio.sleep(delay)
                if ws is not self._ws or ws.closed:
                    continue
                try:
                    await ws.send_str("2")
                    # _LOGGER.debug("WS (ducaheat): -> 2 (keepalive ping)")
                except Exception:  # noqa: BLE001 - defensive logging
                    _LOGGER.debug("WS (ducaheat): keepalive ping failed", exc_info=True)
                    break
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - defensive
            _LOGGER.debug("WS (ducaheat): keepalive loop error", exc_info=True)
        finally:
            if self._keepalive_task is task:
                self._keepalive_task = None
            _LOGGER.debug("WS (ducaheat): keepalive loop stopped")

    def _extract_dev_data_payload(
        self, args: Iterable[Any]
    ) -> Mapping[str, Any] | None:
        """Return the mapping payload embedded in a dev_data Socket.IO event."""

        queue: deque[Any] = deque(args)
        seen: set[int] = set()
        while queue:
            item = queue.popleft()
            marker = id(item)
            if marker in seen:
                continue
            seen.add(marker)
            if isinstance(item, Mapping):
                nodes = item.get("nodes") if "nodes" in item else None
                if isinstance(nodes, Mapping):
                    return item
                coerced = self._coerce_dev_data_nodes(nodes)
                if coerced is not None:
                    combined = dict(item)
                    combined["nodes"] = coerced
                    return combined
                for key in ("data", "payload", "body"):
                    nested = item.get(key)
                    if nested is not None:
                        queue.append(nested)
                continue
            if isinstance(item, str):
                try:
                    decoded = json.loads(item)
                except Exception:  # pragma: no cover - defensive
                    continue
                queue.append(decoded)
                continue
            if isinstance(item, (list, tuple)):
                queue.extend(item)
        return None

    def _coerce_dev_data_nodes(self, nodes: Any) -> dict[str, Any] | None:
        """Convert list-style websocket snapshots into mapping payloads."""

        if nodes is None or isinstance(nodes, Mapping):
            return None
        if isinstance(nodes, (str, bytes, bytearray)):
            return None
        if not isinstance(nodes, Iterable):
            return None

        snapshot: dict[str, Any] = {}
        for entry in nodes:
            if not isinstance(entry, Mapping):
                continue
            node_type = normalize_node_type(
                entry.get("type") or entry.get("node_type"),
                use_default_when_falsey=True,
            )
            if not node_type:
                continue
            type_bucket = snapshot.setdefault(node_type, {})
            for key in _NODE_TYPE_LEVEL_KEYS:
                if key in entry and key not in type_bucket:
                    type_bucket[key] = entry[key]
            addr = normalize_node_addr(
                entry.get("addr") or entry.get("address"),
                use_default_when_falsey=True,
            )
            if not addr:
                continue
            for key, value in entry.items():
                if key in _NODE_METADATA_KEYS or key in _NODE_TYPE_LEVEL_KEYS:
                    continue
                existing = type_bucket.get(key)
                if isinstance(existing, MutableMapping):
                    existing[addr] = value
                else:
                    type_bucket[key] = {addr: value}

        return snapshot or None

    def _log_nodes_summary(self, nodes: Mapping[str, Any]) -> None:
        if not _LOGGER.isEnabledFor(logging.INFO):
            return
        kinds = []
        for key, value in nodes.items():
            if isinstance(value, Mapping):
                addrs = set()
                for section in ("settings", "samples", "status", "advanced"):
                    sec = value.get(section)
                    if isinstance(sec, Mapping):
                        addrs.update(
                            addr for addr in sec.keys() if isinstance(addr, str)
                        )
                kinds.append(f"{key}={len(addrs) if addrs else 0}")
        _LOGGER.info(
            "WS (ducaheat): dev_data nodes: %s",
            " ".join(kinds) if kinds else "(no nodes)",
        )

    def _log_update_brief(self, body: Any) -> None:
        if not _LOGGER.isEnabledFor(logging.DEBUG):
            return
        path = None
        keys = None
        if isinstance(body, Mapping):
            path = body.get("path")
            payload = body.get("body")
            if isinstance(payload, Mapping):
                keys = ",".join(list(payload.keys())[:6])
        _LOGGER.debug("WS (ducaheat): update path=%s keys=%s", path, keys)

    async def _emit_sio(self, event: str, *args: Any) -> None:
        if not self._ws or self._ws.closed:
            raise RuntimeError("websocket not connected")
        arr = [event, *args]
        payload = json.dumps(arr, separators=(",", ":"), default=str)
        frame = f"42{self._namespace}," + payload
        await self._ws.send_str(frame)
        if _LOGGER.isEnabledFor(logging.DEBUG):
            summary = ""
            if event in {"subscribe", "unsubscribe"} and args:
                summary = f" path={args[0]}"
            elif args:
                summary = f" args={args}"
            _LOGGER.debug("WS (ducaheat): -> 42 %s%s", event, summary)

    def _normalise_nodes_payload(self, nodes: Mapping[str, Any]) -> Any:
        """Normalise websocket node payloads via the REST client helper."""

        normaliser = getattr(self._client, "normalise_ws_nodes", None)
        snapshot: Any = deepcopy(nodes) if isinstance(nodes, Mapping) else nodes
        if isinstance(snapshot, Mapping) and not isinstance(snapshot, dict):
            snapshot = dict(snapshot)
        if callable(normaliser):
            try:
                resolved = normaliser(snapshot)  # type: ignore[arg-type]
                if isinstance(resolved, Mapping) and not isinstance(resolved, dict):
                    return dict(resolved)
                return resolved
            except Exception:  # noqa: BLE001  # pragma: no cover - defensive logging
                _LOGGER.debug("WS (ducaheat): normalise_ws_nodes failed", exc_info=True)
                return snapshot
        return snapshot

    def _update_legacy_section(
        self,
        *,
        node_type: str,
        addr: str,
        section: str,
        body: Any,
        dev_map: MutableMapping[str, Any],
    ) -> bool:
        """Store legacy section updates with defensive type checks."""

        if not isinstance(dev_map, MutableMapping):
            return False

        if section == "settings":
            canonical_type = (
                normalize_node_type(node_type, use_default_when_falsey=True)
                or node_type
            )
            section_root = dev_map.get("settings")
            if section_root is not None and not isinstance(section_root, MutableMapping):
                return False
            if canonical_type and isinstance(section_root, MutableMapping):
                existing_section = section_root.get(canonical_type)
                if existing_section is not None and not isinstance(
                    existing_section, MutableMapping
                ):
                    return False

        from .termoweb_ws import TermoWebWSClient

        return TermoWebWSClient._update_legacy_section(
            self,
            node_type=node_type,
            addr=addr,
            section=section,
            body=body,
            dev_map=dev_map,
        )

    def _normalise_cadence_value(self, value: Any) -> float | None:
        """Return a positive cadence hint in seconds."""

        if value is None:
            return None
        try:
            candidate = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(candidate) or candidate <= 0:
            return None
        return candidate

    def _extract_cadence_candidates(self, payload: Mapping[str, Any]) -> list[float]:
        """Return cadence hints discovered in a mapping payload."""

        values: list[float] = []
        stack: list[Mapping[str, Any]] = [payload]
        seen: set[int] = set()
        while stack:
            current = stack.pop()
            if not isinstance(current, Mapping):
                continue
            obj_id = id(current)
            if obj_id in seen:
                continue
            seen.add(obj_id)
            for key in _CADENCE_KEYS:
                if key not in current:
                    continue
                candidate = self._normalise_cadence_value(current.get(key))
                if candidate is not None:
                    values.append(candidate)
            for nested in current.values():
                if isinstance(nested, Mapping):
                    stack.append(nested)
        return values

    def _update_payload_window_from_mapping(
        self,
        payload: Mapping[str, Any],
        *,
        source: str,
    ) -> None:
        """Adjust the payload stale window based on cadence hints."""

        candidates = self._extract_cadence_candidates(payload)
        if candidates:
            self._apply_payload_window_hint(source=source, candidates=candidates)

    def _apply_payload_window_hint(
        self,
        *,
        source: str,
        lease_seconds: Any | None = None,
        candidates: Iterable[Any] | None = None,
    ) -> None:
        """Update the payload window using cadence metadata."""

        values: list[float] = []
        lease_value = self._normalise_cadence_value(lease_seconds)
        if lease_value is not None:
            values.append(lease_value)
        if candidates is not None:
            for candidate in candidates:
                normalised = self._normalise_cadence_value(candidate)
                if normalised is not None:
                    values.append(normalised)
        if not values:
            return

        hint = max(values)
        margin = max(
            hint * _PAYLOAD_WINDOW_MARGIN_RATIO,
            _PAYLOAD_WINDOW_MARGIN_FLOOR,
        )
        window = hint + margin
        window = max(_PAYLOAD_WINDOW_MIN, min(window, _PAYLOAD_WINDOW_MAX))
        if math.isclose(self._payload_stale_after, window, rel_tol=1e-3):
            return

        previous = self._payload_stale_after
        self._payload_stale_after = window
        self._payload_window_hint = hint
        self._payload_window_source = source

        previous_suppression = getattr(self, "_suppress_default_cadence_hint", False)
        self._suppress_default_cadence_hint = True
        try:
            tracker = self._ws_health_tracker()
        finally:
            self._suppress_default_cadence_hint = previous_suppression
        staleness_changed = tracker.set_payload_window(window)

        state = self._ws_state_bucket()
        state["payload_stale_after"] = tracker.payload_stale_after
        state["payload_window_hint"] = self._payload_window_hint
        state["payload_window_source"] = self._payload_window_source
        state["payload_stale"] = tracker.payload_stale

        _LOGGER.debug(
            "WS (ducaheat): payload stale window %.1f->%.1f s (hint=%.1f s, source=%s)",
            previous,
            window,
            hint,
            source,
        )

        if staleness_changed:
            self._notify_ws_status(
                tracker,
                reason="payload_window_update",
                payload_changed=True,
            )

    def _reset_payload_window(self, *, source: str) -> None:
        """Reset the payload stale window to the default."""

        previous = self._payload_stale_after
        self._payload_stale_after = self._default_payload_window
        self._payload_window_hint = None
        self._payload_window_source = source
        previous_suppression = getattr(self, "_suppress_default_cadence_hint", False)
        self._suppress_default_cadence_hint = True
        try:
            tracker = self._ws_health_tracker()
        finally:
            self._suppress_default_cadence_hint = previous_suppression
        staleness_changed = tracker.set_payload_window(self._payload_stale_after)

        self._pending_default_cadence_hint = True

        state = self._ws_state_bucket()
        state["payload_stale_after"] = tracker.payload_stale_after
        state["payload_window_hint"] = self._payload_window_hint
        state["payload_window_source"] = self._payload_window_source
        state["payload_stale"] = tracker.payload_stale

        if not math.isclose(previous, self._payload_stale_after, rel_tol=1e-3):
            _LOGGER.debug(
                "WS (ducaheat): payload stale window reset to %.1f s (source=%s)",
                self._payload_stale_after,
                source,
            )

        if staleness_changed:
            self._notify_ws_status(
                tracker,
                reason="payload_window_reset",
                payload_changed=True,
            )

    def _dispatch_nodes(self, payload: Mapping[str, Any]) -> None:
        """Publish inventory-derived node addresses for downstream consumers."""

        if not isinstance(payload, Mapping):  # pragma: no cover - defensive
            return

        raw_nodes: Mapping[str, Any] | None
        if "nodes" in payload and isinstance(payload.get("nodes"), Mapping):
            raw_nodes = payload.get("nodes")  # type: ignore[assignment]
        else:
            raw_nodes = payload

        cadence_source: Mapping[str, Any] | None = None
        if isinstance(raw_nodes, Mapping):
            cadence_source = raw_nodes

        context = _prepare_nodes_dispatch(
            self.hass,
            entry_id=self.entry_id,
            coordinator=self._coordinator,
            raw_nodes=raw_nodes,
            inventory=self._inventory,
        )

        inventory = context.inventory if isinstance(context.inventory, Inventory) else None
        if self._inventory is None and inventory is not None:
            self._inventory = inventory

        record = context.record
        if isinstance(record, MutableMapping) and inventory is not None:
            record["inventory"] = inventory

        if inventory is not None:
            try:
                addresses_by_type = inventory.addresses_by_type
            except Exception:  # pragma: no cover - defensive cache guard
                addresses_by_type = {}
        else:
            addresses_by_type = {}

        if not addresses_by_type:
            addresses_by_type = {
                node_type: list(addrs) for node_type, addrs in context.addr_map.items()
            }

        self._apply_heater_addresses(
            addresses_by_type,
            inventory=inventory,
            log_prefix="WS (ducaheat)",
            logger=_LOGGER,
        )

        payload_copy: dict[str, Any] = {
            "dev_id": self.dev_id,
            "node_type": None,
            "addr_map": {
                node_type: list(addrs) for node_type, addrs in context.addr_map.items()
            },
            "addresses_by_type": addresses_by_type,
        }

        unknown_types = context.unknown_types
        if unknown_types:
            payload_copy["unknown_types"] = sorted(unknown_types)

        cadence_payload = cadence_source or context.payload
        if isinstance(cadence_payload, Mapping):
            self._update_payload_window_from_mapping(
                cadence_payload,
                source="dispatch_nodes",
            )

        self._mark_ws_payload(
            timestamp=time.time(),
            stale_after=self._payload_stale_after,
        )
        self._dispatcher(self.hass, signal_ws_data(self.entry_id), payload_copy)

    @staticmethod
    def _merge_nodes(target: dict[str, Any], source: Mapping[str, Any]) -> None:
        """Deep merge ``source`` updates into ``target`` in place."""

        for key, value in source.items():
            if isinstance(value, Mapping):
                existing = target.get(key)
                if isinstance(existing, dict):
                    DucaheatWSClient._merge_nodes(existing, value)
                else:
                    target[key] = deepcopy(value)
            else:
                target[key] = deepcopy(value)

    def _collect_sample_updates(self, nodes: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        """Extract heater sample updates from a websocket payload."""

        updates: dict[str, dict[str, Any]] = {}
        for node_type, type_payload in nodes.items():
            if not isinstance(node_type, str) or not isinstance(type_payload, Mapping):
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
            sample_lease = samples.get("lease_seconds") if isinstance(samples, Mapping) else None
            if sample_lease is not None:
                self._apply_payload_window_hint(
                    source="sample_section",
                    lease_seconds=sample_lease,
                )
            lease_seconds = type_payload.get("lease_seconds")
            if bucket or lease_seconds is not None:
                updates[node_type] = {
                    "samples": bucket,
                    "lease_seconds": lease_seconds,
                }
            if lease_seconds is not None:
                self._apply_payload_window_hint(
                    source="sample_updates",
                    lease_seconds=lease_seconds,
                )
        return updates

    def _translate_path_update(self, payload: Any) -> dict[str, Any] | None:
        """Translate ``{"path": ..., "body": ...}`` websocket frames into nodes."""

        return translate_path_update(
            payload,
            resolve_section=self._resolve_update_section,
        )

    @staticmethod
    def _resolve_update_section(section: str | None) -> tuple[str | None, str | None]:
        """Map a websocket path segment to the node section bucket."""

        return resolve_ws_update_section(section)

    def _forward_sample_updates(self, updates: Mapping[str, Mapping[str, Any]]) -> None:
        """Forward websocket heater sample updates to the energy coordinator."""

        has_payload = False
        for node_payload in updates.values():
            if not isinstance(node_payload, Mapping):
                continue
            samples = node_payload.get("samples")
            if isinstance(samples, Mapping) and any(samples):
                has_payload = True
                break
            if node_payload.get("lease_seconds") is not None:
                has_payload = True
                break
        if updates:
            self._update_payload_window_from_mapping(
                updates,
                source="forward_samples",
            )
        if has_payload:
            self._mark_ws_payload(
                timestamp=time.time(),
                stale_after=self._payload_stale_after,
            )
        forward_ws_sample_updates(
            self.hass,
            self.entry_id,
            self.dev_id,
            updates,
            logger=_LOGGER,
            log_prefix="WS (ducaheat)",
        )

    async def _subscribe_feeds(self, nodes: Mapping[str, Any] | None) -> int:
        """Subscribe to heater status and sample feeds."""

        try:
            domain_bucket = self.hass.data.setdefault(DOMAIN, {})
            existing_record = domain_bucket.get(self.entry_id)

            record_mapping: MutableMapping[str, Any]
            if isinstance(existing_record, MutableMapping):
                record_mapping = existing_record
            else:
                record_mapping = {}
                if isinstance(existing_record, Mapping):
                    record_mapping.update(existing_record)
                domain_bucket[self.entry_id] = record_mapping

            inventory_container: Inventory | None = (
                self._inventory if isinstance(self._inventory, Inventory) else None
            )
            if inventory_container is None:
                cached_inventory = record_mapping.get("inventory")
                if isinstance(cached_inventory, Inventory):
                    inventory_container = cached_inventory

            coordinator_inventory = getattr(self._coordinator, "_inventory", None)
            coordinator_nodes: Iterable[Any] | None = None
            if isinstance(coordinator_inventory, Inventory):
                coordinator_nodes = coordinator_inventory.nodes

            should_resolve = inventory_container is None and (
                isinstance(nodes, Mapping)
                or not isinstance(coordinator_inventory, Inventory)
            )

            if should_resolve:
                resolution = resolve_record_inventory(
                    record_mapping,
                    dev_id=self.dev_id,
                    nodes_payload=nodes if isinstance(nodes, Mapping) else None,
                    node_list=coordinator_nodes,
                )
                inventory_container = resolution.inventory

            if inventory_container is None and isinstance(
                coordinator_inventory, Inventory
            ):
                inventory_container = coordinator_inventory

            if not isinstance(inventory_container, Inventory):
                _LOGGER.error(
                    "WS (ducaheat): missing inventory for device %s; skipping heater subscriptions",
                    self.dev_id,
                )
                return 0

            self._inventory = inventory_container
            record_mapping["inventory"] = inventory_container

            targets: list[tuple[str, str]] = list(
                inventory_container.heater_sample_targets
            )
            if not any(node_type == "htr" for node_type, _ in targets):
                fallback: Iterable[Any] | None = None
                if hasattr(self._coordinator, "_addrs"):
                    try:
                        fallback = self._coordinator._addrs()  # type: ignore[attr-defined]  # noqa: SLF001
                    except Exception:  # pragma: no cover - defensive  # noqa: BLE001
                        fallback = None
                if fallback:
                    seen_pairs = set(targets)
                    for candidate in fallback:
                        addr = normalize_node_addr(candidate)
                        if not addr or ("htr", addr) in seen_pairs:
                            continue
                        seen_pairs.add(("htr", addr))
                        targets.append(("htr", addr))

            paths: set[str] = set()
            for node_type, addr in targets:
                paths.add(f"/{node_type}/{addr}/status")
                paths.add(f"/{node_type}/{addr}/samples")

            if not paths:
                return 0

            for path in sorted(paths):
                await self._emit_sio("subscribe", path)

            self._subscription_paths = paths
            return len(paths)
        except Exception:  # pragma: no cover - defensive
            _LOGGER.debug("WS (ducaheat): subscribe failed", exc_info=True)
            return 0

    async def _replay_subscription_paths(self) -> None:
        """Replay cached subscription paths after a reconnect."""

        if not self._subscription_paths:
            return
        for path in sorted(self._subscription_paths):
            try:
                await self._emit_sio("subscribe", path)
            except Exception:  # noqa: BLE001  # pragma: no cover - defensive logging
                _LOGGER.debug(
                    "WS (ducaheat): replay subscribe failed for path %s",
                    path,
                    exc_info=True,
                )

    async def _disconnect(self, reason: str) -> None:
        task = self._keepalive_task
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            finally:
                self._keepalive_task = None
        if self._ws:
            try:
                await self._ws.close(
                    code=aiohttp.WSCloseCode.GOING_AWAY,
                    message=reason.encode(),
                )
            except Exception:  # pragma: no cover - defensive
                _LOGGER.debug("WS (ducaheat): close failed", exc_info=True)
            self._ws = None
        self._pending_dev_data = False
        self._ping_interval = None
        self._ping_timeout = None
        previous_suppression = getattr(self, "_suppress_default_cadence_hint", False)
        self._suppress_default_cadence_hint = True
        try:
            self._reset_payload_window(source="disconnect")
            tracker = self._ws_health
        finally:
            self._suppress_default_cadence_hint = previous_suppression
        tracker.last_payload_at = None
        tracker.last_heartbeat_at = None
        tracker.healthy_since = None
        state = self._ws_state_bucket()
        state["last_payload_at"] = tracker.last_payload_at
        state["last_heartbeat_at"] = tracker.last_heartbeat_at
        changed = tracker.refresh_payload_state(now=time.time())
        state["payload_stale"] = tracker.payload_stale
        if changed:
            self._notify_ws_status(
                tracker,
                reason=reason,
                payload_changed=True,
            )

    async def _get_token(self) -> str:
        headers = await self._client.authed_headers()
        auth = headers.get("Authorization") if isinstance(headers, dict) else None
        if not auth:
            raise RuntimeError("missing Authorization")
        return auth.split(" ", 1)[1]


__all__ = ["DucaheatWSClient"]
