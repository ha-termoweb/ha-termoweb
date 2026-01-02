"""Ducaheat specific websocket client."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Collection, Iterable, Mapping, MutableMapping
import contextlib
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
    build_settings_delta,
    clone_payload_value,
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

_LOGGER = logging.getLogger(__name__)

_PAYLOAD_WINDOW_DEFAULT = 240.0
_PAYLOAD_WINDOW_MIN = 30.0
_PAYLOAD_WINDOW_MAX = 900.0
_PAYLOAD_WINDOW_MARGIN_RATIO = 0.25
_PAYLOAD_WINDOW_MARGIN_FLOOR = 15.0
_CADENCE_KEYS = ("lease_seconds", "cadence_seconds", "poll_seconds")
_NODE_METADATA_KEYS = {"type", "node_type", "addr", "address", "name", "title", "label"}
_NODE_TYPE_LEVEL_KEYS = {"lease_seconds", "cadence_seconds", "poll_seconds"}
_SUBSCRIBE_BACKOFF_INITIAL = 5.0
_SUBSCRIBE_BACKOFF_MAX = 120.0
_PROBE_ACK_TIMEOUT = 5.0
_UPGRADE_DRAIN_TIMEOUT = 1.0
_PARSE_ERROR_WINDOW_S = 30.0
_PARSE_ERROR_THRESHOLD = 3


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
        with contextlib.suppress(Exception):
            buf = gzip.decompress(buf)
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
        except (UnicodeDecodeError, ValueError):
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
        """Initialise the Ducaheat websocket client."""
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
        self._subscription_paths: set[str] = set()
        self._pending_subscribe = True
        self._resubscribe_kick_task: asyncio.Task | None = None
        self._last_subscribe_attempt_ts = 0.0
        self._last_subscribe_log_ts = 0.0
        self._subscribe_backoff_s = _SUBSCRIBE_BACKOFF_INITIAL
        self._subscription_refresh_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()
        self._idle_monitor_task: asyncio.Task | None = None
        self._idle_recovery_lock = asyncio.Lock()
        self._last_idle_recovery_at = 0.0
        self._idle_recovery_attempts = 0
        self._idle_recovery_window_start = 0.0
        self._idle_timeout_flag = False
        self._pending_dev_data = False
        self._keepalive_task: asyncio.Task | None = None
        self._ping_interval: float | None = None
        self._ping_timeout: float | None = None
        self._status: str = "stopped"
        self._healthy_since: float | None = None
        self._last_event_at: float | None = None
        self._last_update_event_at: float | None = None
        self._parse_error_window: deque[float] = deque(maxlen=_PARSE_ERROR_THRESHOLD)
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
        self._bind_inventory_from_context()
        state = self._ws_state_bucket()
        state.setdefault("subscribe_attempts_total", 0)
        state.setdefault("subscribe_success_total", 0)
        state.setdefault("subscribe_fail_total", 0)
        state.setdefault("last_subscribe_success_at", None)
        state.setdefault("last_recovery_at", None)
        state.setdefault("recovery_attempts_total", 0)
        state.setdefault("last_update_event_at", None)
        state.setdefault("parse_errors_total", 0)

    @property
    def _ws_health(self) -> WsHealthTracker:
        """Return the shared websocket health tracker for this client."""

        return self._ws_health_tracker()

    async def _send_str(
        self,
        payload: str,
        *,
        context: str,
        ws: aiohttp.ClientWebSocketResponse | None = None,
    ) -> None:
        """Send a websocket frame with serialized access."""

        target = ws or self._ws
        if target is None:
            raise RuntimeError("websocket not connected")
        async with self._send_lock:
            if target is not self._ws or target.closed:
                raise RuntimeError("websocket not connected")
            await target.send_str(payload)

    def _status_should_reset_health(self, status: str) -> bool:
        """Return True when a status transition should reset health."""

        return status != "healthy"

    def _increment_state_counter(self, key: str, *, delta: int = 1) -> int:
        """Increment a numeric counter in the websocket state bucket."""

        state = self._ws_state_bucket()
        current = state.get(key, 0)
        try:
            value = int(current)
        except (TypeError, ValueError):
            value = 0
        value += delta
        state[key] = value
        return value

    def _record_update_event(self, *, timestamp: float | None = None) -> None:
        """Record a meaningful update event timestamp."""

        now = timestamp if isinstance(timestamp, (int, float)) else time.time()
        self._last_update_event_at = now
        state = self._ws_state_bucket()
        state["last_update_event_at"] = now

    def _record_parse_error(self, *, now: float, reason: str) -> bool:
        """Track JSON parse failures and return True when reconnection is advised."""

        state = self._ws_state_bucket()
        state["last_parse_error_at"] = now
        total = self._increment_state_counter("parse_errors_total")
        window = self._parse_error_window
        window.append(now)
        while window and now - window[0] > _PARSE_ERROR_WINDOW_S:
            window.popleft()
        should_disconnect = len(window) >= _PARSE_ERROR_THRESHOLD
        if should_disconnect:
            _LOGGER.warning(
                "WS (ducaheat): repeated parse errors (%d in %.0fs); reconnecting",
                total,
                _PARSE_ERROR_WINDOW_S,
            )
        elif _LOGGER.isEnabledFor(logging.DEBUG):
            _LOGGER.debug(
                "WS (ducaheat): parse error (%s) count=%d window=%d",
                reason,
                total,
                len(window),
            )
        return should_disconnect

    def _decode_socketio_event(
        self, payload: str, *, now: float
    ) -> tuple[str, list[Any]] | None:
        """Decode a Socket.IO ``42`` payload safely."""

        content = payload
        if content.startswith("/"):
            _namespace, sep, content = content.partition(",")
            if sep != ",":
                self._record_parse_error(now=now, reason="namespace")
                return None
        try:
            arr = json.loads(content)
        except json.JSONDecodeError:
            should_disconnect = self._record_parse_error(now=now, reason="json")
            if should_disconnect:
                self._loop.create_task(self._disconnect("parse_errors"))
            return None
        if not isinstance(arr, list) or not arr:
            should_disconnect = self._record_parse_error(now=now, reason="shape")
            if should_disconnect:
                self._loop.create_task(self._disconnect("parse_errors"))
            return None
        return arr[0], arr[1:]

    def _update_status_from_heartbeat(self, *, now: float) -> None:
        """Update status based on heartbeat activity and payload recency."""

        tracker = self._ws_health
        has_payload = tracker.last_payload_at is not None or (
            self._last_update_event_at is not None
        )
        if has_payload:
            self._update_status("healthy")
            return
        if self._status not in {"connected", "healthy"}:
            self._update_status("connected")

    def start(self) -> asyncio.Task:
        """Start the websocket runner task."""
        if self._task and not self._task.done():
            return self._task
        self._task = self._loop.create_task(
            self._runner(), name=f"termoweb-ws-{self.dev_id}"
        )
        return self._task

    async def stop(self) -> None:
        """Stop the websocket client and background tasks."""
        await self._disconnect("stop")
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self._cleanup_ws_state()

    def is_running(self) -> bool:
        """Return True when the websocket runner task is active."""
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
        """Return the websocket handshake URL."""
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
                    await self._throttle_connection_attempt()
                    await self._connect_once()
                    await self._read_loop_ws()
                except asyncio.CancelledError:
                    break
                except Exception as exc:  # noqa: BLE001  # pragma: no cover - defensive
                    _LOGGER.debug("WS (ducaheat): error %s", exc, exc_info=True)
                    await asyncio.sleep(self._next_backoff())
                finally:
                    await self._disconnect("loop")
                    self._update_status("disconnected")
        finally:
            self._update_status("stopped")

    async def _connect_once(self) -> None:
        """Perform a single websocket handshake and upgrade attempt."""
        token = await self._get_token()
        headers = _brand_headers(self._ua, self._xrw)
        self._idle_timeout_flag = False

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

        await self._send_str("2probe", context="probe", ws=self._ws)
        #        _LOGGER.debug("WS (ducaheat): -> 2probe")
        probe_deadline = time.monotonic() + _PROBE_ACK_TIMEOUT
        probe_ack = False
        while True:
            remaining = probe_deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                async with asyncio.timeout(remaining):
                    probe = await self._ws.receive_str()
            except TimeoutError:
                break
            #            _LOGGER.debug("WS (ducaheat): <- %r", probe)
            if probe == "3probe":
                probe_ack = True
                break
            if probe == "2":
                await self._send_str("3", context="probe-pong", ws=self._ws)
                continue
            if _LOGGER.isEnabledFor(logging.DEBUG):
                _LOGGER.debug(
                    "WS (ducaheat): unexpected probe frame: %r", probe
                )
        if not probe_ack:
            raise HandshakeError(408, ws_url, "probe ack timeout")
        await self._send_str("5", context="upgrade", ws=self._ws)
        #        _LOGGER.debug("WS (ducaheat): -> 5 (upgrade)")

        upgrade_deadline = time.monotonic() + _UPGRADE_DRAIN_TIMEOUT
        with contextlib.suppress(TimeoutError):
            while True:
                remaining = upgrade_deadline - time.monotonic()
                if remaining <= 0:
                    break
                async with asyncio.timeout(remaining):
                    frame = await self._ws.receive_str()
                if frame == "2":
                    await self._send_str("3", context="upgrade-pong", ws=self._ws)
                    continue
                if _LOGGER.isEnabledFor(logging.DEBUG):
                    _LOGGER.debug(
                        "WS (ducaheat): discarding frame after upgrade: %r", frame
                    )
                break

        await self._send_str(
            f"40{self._namespace}", context="namespace-open", ws=self._ws
        )
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

    def _record_frame(
        self, *, timestamp: float | None = None, mark_activity: bool = True
    ) -> None:
        """Update cached websocket frame statistics and timestamps."""

        now = timestamp or time.time()
        self._stats.last_event_ts = now
        if mark_activity:
            self._last_event_at = now
            self._reset_idle_recovery_state(now=now)
        self._mark_ws_heartbeat(timestamp=now)
        state = self._ws_state_bucket()
        state["last_event_at"] = self._last_event_at
        state["frames_total"] = self._stats.frames_total
        state["events_total"] = self._stats.events_total

    def _idle_threshold(self) -> float:
        """Return the idle detection threshold in seconds."""

        tracker = self._ws_health
        payload_window = (
            tracker.payload_stale_after or self._payload_stale_after or 60.0
        )
        try:
            base = float(payload_window)
        except (TypeError, ValueError):
            base = 60.0
        base = base if math.isfinite(base) and base > 0 else 60.0
        return max(30.0, base * 1.2)

    def _idle_monitor_interval(self) -> float:
        """Return the sleep interval between idle checks."""

        threshold = self._idle_threshold()
        return max(5.0, min(30.0, threshold / 4.0))

    def _reset_idle_recovery_state(self, *, now: float | None = None) -> None:
        """Clear idle recovery flags after activity is observed."""

        self._idle_timeout_flag = False
        self._last_idle_recovery_at = now if now is not None else 0.0
        self._idle_recovery_attempts = 0
        self._idle_recovery_window_start = 0.0

    def _start_idle_monitor(self) -> None:
        """Start the idle monitor when the websocket is ready."""

        if self._idle_monitor_task and not self._idle_monitor_task.done():
            return
        if self._ws is None or self._ws.closed:
            return
        self._idle_monitor_task = self._loop.create_task(
            self._idle_monitor(),
            name=f"termoweb-ws-idle-{self.dev_id}",
        )

    async def _stop_idle_monitor(self) -> None:
        """Stop the idle monitor task if it is running."""

        task = self._idle_monitor_task
        if task is None:
            return
        if task is asyncio.current_task():
            self._idle_monitor_task = None
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            if self._idle_monitor_task is task:
                self._idle_monitor_task = None

    async def _handle_idle_check(self, now: float) -> bool:
        """Perform a single idle check iteration."""

        ws = self._ws
        if ws is None or ws.closed:
            return True
        tracker = self._ws_health
        last_payload = tracker.last_payload_at
        last_update = self._last_update_event_at
        last_event = last_update or last_payload or tracker.last_heartbeat_at
        idle_for = now - last_event if isinstance(last_event, (int, float)) else None
        self._refresh_ws_payload_state(now=now, reason="idle_monitor")
        threshold = self._idle_threshold()
        update_idle_for: float | None
        if last_update is not None:
            update_idle_for = now - last_update
        elif last_payload is not None:
            update_idle_for = now - last_payload
        else:
            update_idle_for = threshold
        should_recover = tracker.payload_stale or self._idle_timeout_flag
        if not should_recover and update_idle_for is not None:
            should_recover = update_idle_for >= threshold
        if should_recover:
            await self._recover_from_idle(
                now=now,
                idle_for=idle_for,
                payload_stale=tracker.payload_stale,
            )
        ws_after = self._ws
        return ws_after is None or ws_after.closed

    async def _recover_from_idle(
        self,
        *,
        now: float,
        idle_for: float | None,
        payload_stale: bool,
    ) -> None:
        """Attempt resubscription or reconnect when the socket is idle."""

        if self._ws is None or self._ws.closed:
            return
        async with self._idle_recovery_lock:
            if self._ws is None or self._ws.closed:
                return
            if (
                not self._idle_recovery_window_start
                or now - self._idle_recovery_window_start > 300.0
            ):
                self._idle_recovery_window_start = now
                self._idle_recovery_attempts = 0
            self._idle_recovery_attempts += 1
            self._last_idle_recovery_at = now
            self._idle_timeout_flag = False
            self._increment_state_counter("recovery_attempts_total")
            state = self._ws_state_bucket()
            state["last_recovery_at"] = now
            summary = (
                f"idle_for={idle_for:.0f}s" if idle_for is not None else "idle_for=?"
            )
            _LOGGER.info(
                "WS (ducaheat): idle recovery (%s, stale=%s, subs=%d, pending=%s, attempts=%d)",
                summary,
                payload_stale,
                len(self._subscription_paths),
                self._pending_subscribe,
                self._idle_recovery_attempts,
            )
            try:
                await self._emit_sio("dev_data")
            except Exception:  # noqa: BLE001  # pragma: no cover - defensive
                _LOGGER.debug(
                    "WS (ducaheat): idle recovery dev_data probe failed",
                    exc_info=True,
                )
            try:
                await self._replay_subscription_paths()
            except Exception:  # noqa: BLE001  # pragma: no cover - defensive
                _LOGGER.debug(
                    "WS (ducaheat): idle recovery replay failed",
                    exc_info=True,
                )
            try:
                await self._maybe_subscribe(now)
            except Exception:  # noqa: BLE001  # pragma: no cover - defensive
                _LOGGER.debug(
                    "WS (ducaheat): idle recovery subscribe failed",
                    exc_info=True,
                )
            tracker = self._ws_health
            last_event = (
                self._last_update_event_at
                or tracker.last_payload_at
                or tracker.last_heartbeat_at
            )
            if last_event is None:
                last_event = tracker.last_heartbeat_at
            idle_after = now - last_event if isinstance(last_event, (int, float)) else 0
            window_age = now - self._idle_recovery_window_start
            should_disconnect = (
                self._idle_recovery_attempts >= 3
                and window_age <= 300.0
                and (
                    tracker.payload_stale
                    or idle_after >= self._idle_threshold()
                    or (idle_for is not None and idle_after >= idle_for)
                )
            )
            if should_disconnect:
                _LOGGER.warning(
                    "WS (ducaheat): idle recovery escalation after %d attempts; reconnecting",
                    self._idle_recovery_attempts,
                )
                await self._disconnect("idle_recovery_failed")

    async def _idle_monitor(self) -> None:
        """Monitor websocket idleness and trigger recovery steps."""

        task = asyncio.current_task()
        try:
            while True:
                await asyncio.sleep(self._idle_monitor_interval())
                now = time.time()
                should_exit = await self._handle_idle_check(now)
                if should_exit:
                    break
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001  # pragma: no cover - defensive
            _LOGGER.debug("WS (ducaheat): idle monitor error", exc_info=True)
        finally:
            if self._idle_monitor_task is task:
                self._idle_monitor_task = None
            _LOGGER.debug("WS (ducaheat): idle monitor stopped")

    async def _read_loop_ws(self) -> None:  # noqa: C901
        ws = self._ws
        if ws is None:
            return
        try:
            while True:
                tracker = self._ws_health
                await self._maybe_subscribe(time.time())
                if tracker.payload_stale and tracker.status == "healthy":
                    self._update_status("connected")
                deadline = tracker.stale_deadline()
                timeout: float | None = None
                if isinstance(deadline, (int, float)):
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        now = time.time()
                        self._refresh_ws_payload_state(
                            now=now, reason="payload_timeout"
                        )
                        if tracker.payload_stale and tracker.status == "healthy":
                            self._update_status("connected")
                        remaining = 1.0
                    timeout = max(1.0, remaining)
                try:
                    if timeout is None:
                        msg = await ws.receive()
                    else:
                        msg = await asyncio.wait_for(ws.receive(), timeout=timeout)
                except TimeoutError:
                    now = time.time()
                    self._refresh_ws_payload_state(now=now, reason="payload_timeout")
                    tracker = self._ws_health
                    if tracker.payload_stale and tracker.status == "healthy":
                        self._update_status("connected")
                    self._idle_timeout_flag = True
                    continue

                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = msg.data
                    self._stats.frames_total += 1
                    now = time.time()
                    frame_recorded = False
                    while data:
                        if data == "2":
                            if not frame_recorded:
                                self._record_frame(timestamp=now, mark_activity=False)
                                frame_recorded = True
                            await self._send_str("3", context="engineio-pong", ws=ws)
                            break
                        if data == "3":
                            if not frame_recorded:
                                self._record_frame(timestamp=now, mark_activity=False)
                                frame_recorded = True
                            self._update_status_from_heartbeat(now=now)
                            break
                        if not data.startswith("4"):
                            if not frame_recorded:
                                self._record_frame(timestamp=now, mark_activity=False)
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
                            if not frame_recorded:
                                self._record_frame(timestamp=now)
                                frame_recorded = True
                            self._update_status("healthy")
                            if self._pending_dev_data:
                                self._pending_dev_data = False
                                try:
                                    await self._emit_sio("dev_data")
                                    await self._replay_subscription_paths()
                                    if self._subscription_paths:
                                        self._pending_subscribe = False
                                    await self._maybe_subscribe(now)
                                    self._start_idle_monitor()
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
                            if not frame_recorded:
                                self._record_frame(timestamp=now, mark_activity=False)
                                frame_recorded = True
                            await self._send_str("3", context="engineio-pong", ws=ws)
                            break
                        if payload == "3":
                            if not frame_recorded:
                                self._record_frame(timestamp=now, mark_activity=False)
                                frame_recorded = True
                            self._update_status_from_heartbeat(now=now)
                            break
                        if not frame_recorded:
                            self._record_frame(timestamp=now)
                            frame_recorded = True
                        if payload.startswith("2/"):
                            ns_payload = payload[1:]
                            ns, sep, body = ns_payload.partition(",")
                            if not sep or body in {"", "[]", '["ping"]'}:
                                await self._send_str(
                                    "3" + ns,
                                    context="engineio-namespace-pong",
                                    ws=ws,
                                )
                                break

                        content: str | None = None
                        if data.startswith("42"):
                            content = data[2:]
                        elif payload.startswith("42"):
                            content = payload[2:]
                        if content is None:
                            break
                        decoded = self._decode_socketio_event(content, now=now)
                        if decoded is None:
                            break
                        evt, args = decoded
                        self._stats.events_total += 1
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
                                    nodes_map = self._coerce_dev_data_nodes(
                                        nodes_candidate
                                    )
                            if isinstance(nodes_map, Mapping):
                                self._log_nodes_summary(nodes_map)
                                normalised = self._normalise_nodes_payload(nodes_map)
                                inventory = (
                                    self._inventory
                                    if isinstance(self._inventory, Inventory)
                                    else self._bind_inventory_from_context()
                                )
                                dispatch_payload: Mapping[str, Any] | None
                                if isinstance(normalised, Mapping):
                                    dispatch_payload = normalised
                                    deltas = self._nodes_to_deltas(
                                        normalised,
                                        inventory=inventory,
                                    )
                                    if deltas:
                                        self._apply_deltas_to_store(
                                            deltas,
                                            replace=True,
                                        )
                                else:
                                    dispatch_payload = nodes_map
                                if isinstance(dispatch_payload, Mapping):
                                    self._dispatch_nodes(dispatch_payload)
                                self._record_update_event(timestamp=now)
                                subs = await self._maybe_subscribe(now)
                                if subs:
                                    _LOGGER.info(
                                        "WS (ducaheat): subscribed %d feeds", subs
                                    )
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
                                    inventory = (
                                        self._inventory
                                        if isinstance(self._inventory, Inventory)
                                        else self._bind_inventory_from_context()
                                    )
                                    deltas = self._nodes_to_deltas(
                                        normalised_update,
                                        inventory=inventory,
                                    )
                                    if deltas:
                                        self._apply_deltas_to_store(
                                            deltas,
                                            replace=False,
                                        )
                                    self._dispatch_nodes(normalised_update)
                                    if not isinstance(inventory, Inventory):
                                        inventory = (
                                            self._inventory
                                            if isinstance(self._inventory, Inventory)
                                            else None
                                        )
                                    allowed_types = (
                                        inventory.energy_sample_types
                                        if isinstance(inventory, Inventory)
                                        else None
                                    )
                                    sample_updates = self._collect_sample_updates(
                                        normalised_update,
                                        allowed_types=allowed_types,
                                    )
                                    if sample_updates:
                                        self._forward_sample_updates(sample_updates)
                            self._record_update_event(timestamp=now)
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
                    await self._send_str("2", context="keepalive-ping", ws=ws)
                    # _LOGGER.debug("WS (ducaheat): -> 2 (keepalive ping)")
                except Exception:  # noqa: BLE001 - defensive logging
                    _LOGGER.debug("WS (ducaheat): keepalive ping failed", exc_info=True)
                    break
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001  # pragma: no cover - defensive
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
                except json.JSONDecodeError:  # pragma: no cover - defensive
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

        inventory = self._inventory if isinstance(self._inventory, Inventory) else None
        if inventory is None:
            _LOGGER.error(
                "WS (ducaheat): cannot coerce nodes payload without inventory for %s",
                self.dev_id,
            )
            return None

        snapshot: dict[str, Any] = {}
        for node_type, addr, entry in inventory.iter_known_entries(nodes):
            type_bucket = snapshot.setdefault(node_type, {})
            for key in _NODE_TYPE_LEVEL_KEYS:
                if key in entry and key not in type_bucket:
                    type_bucket[key] = entry[key]
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
                        addrs.update(addr for addr in sec if isinstance(addr, str))
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
        await self._send_str(frame, context=f"sio-{event}", ws=self._ws)
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
        snapshot: Any = clone_payload_value(nodes)
        if isinstance(snapshot, Mapping) and not isinstance(snapshot, dict):
            snapshot = dict(snapshot)
        if callable(normaliser):
            resolved: Any
            try:
                resolved = normaliser(snapshot)  # type: ignore[arg-type]
            except Exception:  # noqa: BLE001  # pragma: no cover - defensive logging
                _LOGGER.debug("WS (ducaheat): normalise_ws_nodes failed", exc_info=True)
                return snapshot
            else:
                if isinstance(resolved, Mapping) and not isinstance(resolved, dict):
                    return dict(resolved)
                return resolved
        return snapshot

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
            stack.extend(
                nested for nested in current.values() if isinstance(nested, Mapping)
            )
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

        inventory = (
            context.inventory if isinstance(context.inventory, Inventory) else None
        )
        if isinstance(inventory, Inventory):
            self._inventory = inventory

        if not isinstance(inventory, Inventory):
            _LOGGER.error(
                "WS (ducaheat): missing inventory for node dispatch on %s", self.dev_id
            )
            return

        self._apply_heater_addresses(
            raw_nodes,
            inventory=inventory,
            log_prefix="WS (ducaheat)",
            logger=_LOGGER,
        )

        payload_copy: dict[str, Any] = {
            "dev_id": self.dev_id,
            "node_type": None,
            "inventory": inventory,
        }

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

    def _collect_sample_updates(
        self,
        nodes: Mapping[str, Any],
        *,
        allowed_types: Collection[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Extract heater sample updates from a websocket payload."""

        allowed: set[str] | None = None
        if allowed_types is not None:
            allowed = set()
            for candidate in allowed_types:
                normalized = normalize_node_type(
                    candidate,
                    use_default_when_falsey=True,
                )
                if normalized:
                    allowed.add(normalized)

        updates: dict[str, dict[str, Any]] = {}
        for node_type, type_payload in nodes.items():
            if not isinstance(node_type, str) or not isinstance(type_payload, Mapping):
                continue
            canonical_type = normalize_node_type(
                node_type,
                use_default_when_falsey=True,
            )
            if canonical_type is None:
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
            sample_lease = (
                samples.get("lease_seconds") if isinstance(samples, Mapping) else None
            )
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

    def _nodes_to_deltas(
        self,
        nodes: Mapping[str, Any],
        *,
        inventory: Inventory | None,
    ) -> list[NodeSettingsDelta]:
        """Convert websocket node payloads into domain delta objects."""

        resolved_inventory = inventory if isinstance(inventory, Inventory) else None
        if resolved_inventory is None:
            _LOGGER.warning(
                "WS (ducaheat): missing inventory for node delta translation on %s",
                self.dev_id,
            )
            return []

        deltas: list[NodeSettingsDelta] = []
        for raw_type, sections in nodes.items():
            if not isinstance(raw_type, str) or not isinstance(sections, Mapping):
                continue
            try:
                node_type = DomainNodeType(str(raw_type).lower())
            except ValueError:
                continue

            per_addr: dict[str, dict[str, Any]] = {}
            for section, section_payload in sections.items():
                if not isinstance(section, str):
                    continue
                if section == "samples" or not isinstance(section_payload, Mapping):
                    continue
                for raw_addr, payload in section_payload.items():
                    addr = normalize_node_addr(
                        raw_addr,
                        use_default_when_falsey=True,
                    )
                    if not addr:
                        continue
                    bucket = per_addr.setdefault(addr, {})
                    settings_delta: Mapping[str, Any] = {}
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
                try:
                    node_id = DomainNodeId(node_type, addr)
                except ValueError:
                    continue
                if not resolved_inventory.has_node(
                    node_id.node_type.value, node_id.addr
                ):
                    _LOGGER.warning(
                        "WS (ducaheat): ignoring update for unknown node_type=%s addr=%s on %s",
                        node_type.value,
                        addr,
                        self.dev_id,
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
        except Exception:  # noqa: BLE001
            _LOGGER.debug(
                "WS (ducaheat): failed to apply websocket deltas",
                exc_info=True,
            )

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

    async def _subscribe_feeds(self, *, now: float | None = None) -> int:
        """Subscribe to heater status and sample feeds."""

        attempt_ts = now if isinstance(now, (int, float)) else time.time()
        self._last_subscribe_attempt_ts = attempt_ts
        self._increment_state_counter("subscribe_attempts_total")
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

            try:
                inventory_container = Inventory.require_from_context(
                    inventory=self._inventory,
                    container=record_mapping,
                    hass=self.hass,
                    entry_id=self.entry_id,
                    coordinator=self._coordinator,
                )
            except LookupError:
                self._pending_subscribe = True
                self._increment_state_counter("subscribe_fail_total")
                should_log = (
                    attempt_ts - self._last_subscribe_log_ts
                    >= self._subscribe_backoff_s
                )
                if should_log:
                    self._last_subscribe_log_ts = attempt_ts
                    _LOGGER.info(
                        "WS (ducaheat): inventory not ready for device %s; will retry",
                        self.dev_id,
                    )
                return 0

            self._inventory = inventory_container

            energy_coordinator = record_mapping.get("energy_coordinator")
            if hasattr(energy_coordinator, "update_addresses"):
                try:
                    energy_coordinator.update_addresses(inventory_container)
                except Exception:  # noqa: BLE001  # pragma: no cover - defensive logging
                    _LOGGER.debug(
                        "WS (ducaheat): failed to update coordinator addresses during subscribe",
                        exc_info=True,
                    )

            paths: set[str] = set()
            for node_type, addr in inventory_container.heater_sample_targets:
                base_path = f"/{node_type}/{addr}"
                paths.add(f"{base_path}/status")
                paths.add(f"{base_path}/samples")

            if not paths:
                self._pending_subscribe = False
                self._subscription_paths = set()
                self._subscribe_backoff_s = _SUBSCRIBE_BACKOFF_INITIAL
                self._increment_state_counter("subscribe_success_total")
                state = self._ws_state_bucket()
                state["last_subscribe_success_at"] = attempt_ts
                return 0

            for path in sorted(paths):
                await self._emit_sio("subscribe", path)

            self._subscription_paths = paths
            self._pending_subscribe = False
            self._subscribe_backoff_s = _SUBSCRIBE_BACKOFF_INITIAL
            self._increment_state_counter("subscribe_success_total")
            state = self._ws_state_bucket()
            state["last_subscribe_success_at"] = attempt_ts
            return len(paths)
        except Exception:  # noqa: BLE001  # pragma: no cover - defensive
            _LOGGER.debug("WS (ducaheat): subscribe failed", exc_info=True)
            self._pending_subscribe = True
            self._increment_state_counter("subscribe_fail_total")
            return 0

    async def _maybe_subscribe(self, now: float) -> int:
        """Attempt subscription installation when prerequisites are met."""

        ws = self._ws
        if ws is None or ws.closed:
            return 0
        if self._pending_dev_data:
            return 0
        if self._status not in {"connected", "healthy"}:
            return 0
        if not self._pending_subscribe:
            return 0

        backoff = max(self._subscribe_backoff_s, _SUBSCRIBE_BACKOFF_INITIAL)
        if now - self._last_subscribe_attempt_ts < backoff:
            return 0

        async with self._subscription_refresh_lock:
            if now - self._last_subscribe_attempt_ts < backoff:
                return 0
            self._last_subscribe_attempt_ts = now
            result = await self._subscribe_feeds(now=now)

        if self._pending_subscribe and (result > 0 or self._subscription_paths):
            self._pending_subscribe = False

        if not self._pending_subscribe:
            self._subscribe_backoff_s = _SUBSCRIBE_BACKOFF_INITIAL
            return result
        if result > 0 or self._subscription_paths:
            self._subscribe_backoff_s = _SUBSCRIBE_BACKOFF_INITIAL
            return result

        self._subscribe_backoff_s = min(
            max(self._subscribe_backoff_s * 1.5, _SUBSCRIBE_BACKOFF_INITIAL),
            _SUBSCRIBE_BACKOFF_MAX,
        )
        return result

    def request_resubscribe(self, reason: str) -> None:
        """Flag the subscription set for reinstatement and trigger a prompt attempt."""

        self._pending_subscribe = True
        is_inventory_ready = reason == "inventory_ready"
        if is_inventory_ready:
            self._subscribe_backoff_s = _SUBSCRIBE_BACKOFF_INITIAL
            self._last_subscribe_attempt_ts = 0.0
            self._schedule_resubscribe_kick()
        else:
            self._subscribe_backoff_s = max(
                self._subscribe_backoff_s, _SUBSCRIBE_BACKOFF_INITIAL
            )
        if _LOGGER.isEnabledFor(logging.DEBUG):
            _LOGGER.debug(
                "WS (ducaheat): resubscribe requested (%s) for %s",
                reason,
                self.dev_id,
            )

    def _schedule_resubscribe_kick(self) -> None:
        """Schedule an immediate subscribe attempt when conditions allow."""

        ws = self._ws
        if ws is None or ws.closed:
            return
        if self._status not in {"connected", "healthy"}:
            return
        if self._pending_dev_data:
            return
        if self._resubscribe_kick_task and not self._resubscribe_kick_task.done():
            return
        self._resubscribe_kick_task = self._loop.create_task(
            self._run_resubscribe_kick(),
            name=f"termoweb-ws-{self.dev_id}-resubscribe-kick",
        )

    async def _run_resubscribe_kick(self) -> None:
        """Attempt a single subscription refresh after a resubscribe request."""

        try:
            await self._maybe_subscribe(now=time.time())
        except Exception:  # noqa: BLE001  # pragma: no cover - defensive logging
            _LOGGER.debug(
                "WS (ducaheat): resubscribe kick failed for %s",
                self.dev_id,
                exc_info=True,
            )
        finally:
            self._resubscribe_kick_task = None

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
        await self._stop_idle_monitor()
        self._reset_idle_recovery_state(now=None)
        self._last_update_event_at = None
        self._parse_error_window.clear()
        task = self._keepalive_task
        if task:
            task.cancel()
            try:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            finally:
                self._keepalive_task = None
        kick_task = self._resubscribe_kick_task
        if kick_task:
            kick_task.cancel()
            try:
                with contextlib.suppress(asyncio.CancelledError):
                    await kick_task
            finally:
                self._resubscribe_kick_task = None
        if self._ws:
            try:
                await self._ws.close(
                    code=aiohttp.WSCloseCode.GOING_AWAY,
                    message=reason.encode(),
                )
            except Exception:  # noqa: BLE001  # pragma: no cover - defensive
                _LOGGER.debug("WS (ducaheat): close failed", exc_info=True)
            self._ws = None
        self._pending_dev_data = False
        self._cleanup_ws_state()
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
