"""Socket.IO v2 websocket client for the Ducaheat backend."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import random
import time
from typing import Any
from urllib.parse import urlencode

import aiohttp
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_send

from .const import DOMAIN, WS_NAMESPACE, signal_ws_data, signal_ws_status
from .utils import extract_heater_addrs
from .ws_client_legacy import HandshakeError, WSStats

_LOGGER = logging.getLogger(__name__)


class TermoWebWSV2Client:
    """Socket.IO v2 client used by the Ducaheat backend."""

    def __init__(
        self,
        hass: HomeAssistant,
        *,
        entry_id: str,
        dev_id: str,
        api_client,
        coordinator,
        session: aiohttp.ClientSession | None = None,
        handshake_fail_threshold: int = 5,
    ) -> None:
        """Initialize the websocket client with integration helpers."""
        self.hass = hass
        self.entry_id = entry_id
        self.dev_id = dev_id
        self._client = api_client
        self._coordinator = coordinator
        self._session = session or api_client._session  # noqa: SLF001
        self._task: asyncio.Task | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None

        self._closing = False
        self._connected_since: float | None = None
        self._healthy_since: float | None = None
        self._hb_send_interval: float = 20.0
        self._ping_timeout: float = 60.0
        self._hb_task: asyncio.Task | None = None

        self._backoff_seq = [5, 10, 30, 120, 300]
        self._backoff_idx = 0

        self._stats = WSStats()
        self._hs_fail_count = 0
        self._hs_fail_start = 0.0
        self._hs_fail_threshold = handshake_fail_threshold

    # ----------------- Public control -----------------

    def start(self) -> asyncio.Task:
        """Start the websocket runner task if not already running."""
        if self._task and not self._task.done():
            return self._task
        self._closing = False
        self._task = self.hass.loop.create_task(
            self._runner(), name=f"{DOMAIN}-ws-v2-{self.dev_id}"
        )
        return self._task

    async def stop(self) -> None:
        """Stop the websocket client and cancel background tasks."""
        self._closing = True
        if self._hb_task:
            self._hb_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._hb_task
            self._hb_task = None
        if self._ws:
            with contextlib.suppress(TimeoutError, aiohttp.ClientError, RuntimeError):
                await self._ws.close(
                    code=aiohttp.WSCloseCode.GOING_AWAY, message=b"client stop"
                )
            self._ws = None
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self._update_status("stopped")

    def is_running(self) -> bool:
        """Return True if the websocket runner task is active."""
        return bool(self._task and not self._task.done())

    # ----------------- Core loop -----------------

    async def _runner(self) -> None:
        self._update_status("starting")
        while not self._closing:
            should_retry = True
            try:
                sid, ping_interval, ping_timeout = await self._handshake()
                self._hs_fail_count = 0
                self._hs_fail_start = 0.0
                self._hb_send_interval = max(5.0, min(30.0, ping_interval * 0.6))
                self._ping_timeout = ping_timeout
                await self._connect_ws(sid)
                await self._join_namespace()
                await self._request_snapshot()
                self._connected_since = time.time()
                self._healthy_since = None
                self._update_status("connected")

                self._hb_task = self.hass.loop.create_task(self._heartbeat_loop())
                await self._read_loop()

            except asyncio.CancelledError:
                should_retry = False
                break
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
            except (
                aiohttp.ClientError,
                TimeoutError,
                RuntimeError,
                ValueError,
                OSError,
            ) as err:
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
                    with contextlib.suppress(
                        TimeoutError, aiohttp.ClientError, RuntimeError
                    ):
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

        self._update_status("stopped")

    # ----------------- Protocol steps -----------------

    async def _handshake(self) -> tuple[str, float, float]:
        token = await self._get_token()
        base = self._api_base()
        params = {"EIO": "3", "transport": "polling"}
        url = f"{base}/api/v2/socket_io/?{urlencode(params)}"
        headers = {"Authorization": f"Bearer {token}"}

        try:
            async with asyncio.timeout(15):
                async with self._session.get(
                    url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    body = await resp.text()
                    if resp.status == 401:
                        _LOGGER.info(
                            "WS %s: handshake 401; refreshing token", self.dev_id
                        )
                        await self._force_refresh_token()
                        token = await self._get_token()
                        headers["Authorization"] = f"Bearer {token}"
                        async with self._session.get(
                            url,
                            headers=headers,
                            timeout=aiohttp.ClientTimeout(total=15),
                        ) as resp2:
                            body = await resp2.text()
                            if resp2.status >= 400:
                                raise HandshakeError(resp2.status, url, body[:160])
                            sid, ping_interval, ping_timeout = self._parse_handshake(
                                body
                            )
                            self._backoff_idx = 0
                            return sid, ping_interval, ping_timeout

                    if resp.status >= 400:
                        raise HandshakeError(resp.status, url, body[:160])

                    sid, ping_interval, ping_timeout = self._parse_handshake(body)
                    self._backoff_idx = 0
                    return sid, ping_interval, ping_timeout
        except (TimeoutError, aiohttp.ClientError) as err:
            raise HandshakeError(-1, url, str(err)) from err

    async def _connect_ws(self, sid: str) -> None:
        token = await self._get_token()
        base = self._api_base()
        ws_url = f"{base}/api/v2/socket_io/?EIO=3&transport=websocket&sid={sid}"
        if ws_url.startswith("https://"):
            ws_url = "wss://" + ws_url[len("https://") :]
        elif ws_url.startswith("http://"):
            ws_url = "ws://" + ws_url[len("http://") :]
        self._ws = await self._session.ws_connect(
            ws_url,
            heartbeat=None,
            timeout=15,
            autoclose=True,
            autoping=False,
            compress=0,
            headers={"Authorization": f"Bearer {token}"},
        )

    async def _join_namespace(self) -> None:
        await self._send_text(f"40{WS_NAMESPACE}")

    async def _request_snapshot(self) -> None:
        payload = json.dumps(["dev_data", {"dev_id": self.dev_id}], separators=(",", ":"))
        await self._send_text(f"42{WS_NAMESPACE},{payload}")

    # ----------------- Loops -----------------

    async def _heartbeat_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._hb_send_interval)
                await self._send_text("2")
        except asyncio.CancelledError:
            return
        except (aiohttp.ClientError, RuntimeError, ConnectionError):
            return

    async def _read_loop(self) -> None:
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
                msg.data
                if isinstance(msg.data, str)
                else msg.data.decode("utf-8", "ignore")
            )
            if not isinstance(data, str):
                continue
            self._stats.frames_total += 1

            if data == "3":
                self._mark_event(paths=None)
                continue
            if data == "2":
                await self._send_text("3")
                continue
            if data.startswith("0"):
                self._handle_open_frame(data)
                continue
            if data.startswith(f"40{WS_NAMESPACE}"):
                self._mark_event(paths=None)
                continue
            if data.startswith("42"):
                event = self._parse_event_frame(data)
                if event:
                    self._handle_event(*event)
                continue
            if data.startswith("41"):
                raise RuntimeError("server disconnect")

    # ----------------- Event handling -----------------

    def _handle_open_frame(self, payload: str) -> None:
        body = payload[1:]
        try:
            js = json.loads(body)
        except json.JSONDecodeError:
            return
        if isinstance(js, dict):
            ping_interval_ms = float(js.get("pingInterval", 25000))
            ping_timeout_ms = float(js.get("pingTimeout", 60000))
            self._hb_send_interval = max(5.0, min(30.0, ping_interval_ms / 1000.0 * 0.6))
            self._ping_timeout = ping_timeout_ms / 1000.0

    def _parse_event_frame(self, payload: str) -> tuple[str, list[Any]] | None:
        data = payload[2:]
        namespace = ""
        if data.startswith("/"):
            try:
                namespace, data = data.split(",", 1)
            except ValueError:
                namespace, data = data, ""
        if namespace and namespace != WS_NAMESPACE:
            return None
        if not data:
            return None
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, list) or not parsed:
            return None
        name = parsed[0]
        args = parsed[1:]
        if not isinstance(name, str):
            return None
        return name, args

    def _handle_event(self, name: str, args: list[Any]) -> None:
        batches = self._extract_updates(name, args)
        for dev_id, updates in batches:
            if not updates:
                continue
            paths, updated_nodes, setting_addrs, sample_addrs = self._apply_updates(
                dev_id, updates
            )
            self._mark_event(paths=paths)
            payload_base = {"dev_id": dev_id, "ts": self._stats.last_event_ts}
            if updated_nodes:
                async_dispatcher_send(
                    self.hass,
                    signal_ws_data(self.entry_id),
                    {**payload_base, "addr": None, "kind": "nodes"},
                )
            for addr in setting_addrs:
                async_dispatcher_send(
                    self.hass,
                    signal_ws_data(self.entry_id),
                    {**payload_base, "addr": addr, "kind": "htr_settings"},
                )
            for addr in sample_addrs:
                async_dispatcher_send(
                    self.hass,
                    signal_ws_data(self.entry_id),
                    {**payload_base, "addr": addr, "kind": "htr_samples"},
                )

    def _extract_updates(
        self, name: str, args: list[Any]
    ) -> list[tuple[str, list[dict[str, Any]]]]:
        result: list[tuple[str, list[dict[str, Any]]]] = []
        if name == "dev_handshake" and args:
            first = args[0]
            devices = first.get("devices") if isinstance(first, dict) else None
            if isinstance(devices, list):
                for dev in devices:
                    dev_id = self._coerce_dev_id(dev)
                    updates = self._coerce_updates(dev)
                    if dev_id and updates:
                        result.append((dev_id, updates))
            return result

        for item in args:
            if not isinstance(item, dict):
                continue
            dev_id = self._coerce_dev_id(item)
            updates = self._coerce_updates(item)
            if dev_id and updates:
                result.append((dev_id, updates))
        return result

    def _coerce_dev_id(self, item: dict[str, Any] | None) -> str | None:
        if not isinstance(item, dict):
            return None
        dev_id = item.get("dev_id") or item.get("id") or item.get("serial_id")
        if not dev_id:
            return self.dev_id
        return str(dev_id)

    def _coerce_updates(self, item: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not isinstance(item, dict):
            return []
        updates: list[dict[str, Any]] = []
        raw_updates = item.get("updates")
        if isinstance(raw_updates, list):
            updates.extend(
                entry for entry in raw_updates if isinstance(entry, dict)
            )
        nodes = item.get("nodes")
        if isinstance(nodes, dict):
            updates.append({"path": "/mgr/nodes", "body": nodes})
        htr = item.get("htr")
        if isinstance(htr, dict):
            settings = htr.get("settings")
            if isinstance(settings, dict):
                for addr, body in settings.items():
                    updates.append(
                        {
                            "path": f"/htr/{addr}/settings",
                            "body": body,
                        }
                    )
            advanced = htr.get("advanced")
            if isinstance(advanced, dict):
                for addr, body in advanced.items():
                    updates.append(
                        {
                            "path": f"/htr/{addr}/advanced_setup",
                            "body": body,
                        }
                    )
        return updates

    def _apply_updates(
        self, dev_id: str, updates: list[dict[str, Any]]
    ) -> tuple[list[str], bool, list[str], list[str]]:
        paths: list[str] = []
        updated_nodes = False
        updated_addrs: list[str] = []
        sample_addrs: list[str] = []

        for item in updates:
            if not isinstance(item, dict):
                continue
            path = item.get("path")
            body = item.get("body")
            if not isinstance(path, str):
                continue
            paths.append(path)

            dev_map: dict[str, Any] = (self._coordinator.data or {}).get(dev_id) or {}
            if not dev_map:
                dev_map = {
                    "dev_id": dev_id,
                    "name": f"Device {dev_id}",
                    "raw": {},
                    "connected": True,
                    "nodes": None,
                    "htr": {"addrs": [], "settings": {}},
                }
                cur = dict(self._coordinator.data or {})
                cur[dev_id] = dev_map
                self._coordinator.data = cur  # type: ignore[attr-defined]

            if path.endswith("/mgr/nodes"):
                if isinstance(body, dict):
                    dev_map["nodes"] = body
                    addrs = extract_heater_addrs(body)
                    dev_map.setdefault("htr", {}).setdefault("settings", {})
                    dev_map["htr"]["addrs"] = addrs
                    updated_nodes = True

            elif "/htr/" in path and path.endswith("/settings"):
                addr = path.split("/htr/")[1].split("/")[0]
                settings_map: dict[str, Any] = dev_map.setdefault("htr", {}).setdefault(
                    "settings", {}
                )
                if isinstance(body, dict):
                    settings_map[addr] = body
                    updated_addrs.append(addr)

            elif "/htr/" in path and path.endswith("/advanced_setup"):
                addr = path.split("/htr/")[1].split("/")[0]
                adv_map: dict[str, Any] = dev_map.setdefault("htr", {}).setdefault(
                    "advanced", {}
                )
                if isinstance(body, dict):
                    adv_map[addr] = body

            elif "/htr/" in path and path.endswith("/samples"):
                addr = path.split("/htr/")[1].split("/")[0]
                sample_addrs.append(addr)

            else:
                raw = dev_map.setdefault("raw", {})
                key = path.strip("/").replace("/", "_")
                raw[key] = body

        return paths, updated_nodes, list(dict.fromkeys(updated_addrs)), list(
            dict.fromkeys(sample_addrs)
        )

    # ----------------- Helpers -----------------

    def _parse_handshake(self, body: str) -> tuple[str, float, float]:
        payload = (body or "").strip()
        if payload and payload[0].isdigit():
            idx = payload.find(":")
            if idx != -1:
                payload = payload[idx + 1 :]
        payload = payload.removeprefix("0")
        try:
            js = json.loads(payload)
        except json.JSONDecodeError as err:  # pragma: no cover
            raise ValueError("handshake malformed") from err
        if not isinstance(js, dict):
            raise TypeError("handshake malformed")
        sid = js.get("sid")
        if not isinstance(sid, str):
            raise TypeError("handshake missing sid")
        ping_interval = float(js.get("pingInterval", 25000)) / 1000.0
        ping_timeout = float(js.get("pingTimeout", 60000)) / 1000.0
        return sid, ping_interval, ping_timeout

    async def _send_text(self, data: str) -> None:
        if not self._ws:
            return
        await self._ws.send_str(data)

    async def _get_token(self) -> str:
        headers = await self._client._authed_headers()  # noqa: SLF001
        return headers["Authorization"].split(" ", 1)[1]

    async def _force_refresh_token(self) -> None:
        with contextlib.suppress(AttributeError):
            self._client._access_token = None  # type: ignore[attr-defined] # noqa: SLF001
        await self._client._ensure_token()  # noqa: SLF001

    def _api_base(self) -> str:
        base = getattr(self._client, "api_base", None)
        if isinstance(base, str) and base:
            return base.rstrip("/")
        return "https://api-tevolve.termoweb.net"

    def _update_status(self, status: str) -> None:
        state_bucket = self.hass.data[DOMAIN][self.entry_id].setdefault("ws_state", {})
        state = state_bucket.setdefault(self.dev_id, {})
        now = time.time()
        state["status"] = status
        state["last_event_at"] = self._stats.last_event_ts or None
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

    def _mark_event(self, *, paths: list[str] | None) -> None:
        now = time.time()
        self._stats.last_event_ts = now
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

        domain_bucket: dict[str, Any] = self.hass.data.setdefault(DOMAIN, {})
        entry_bucket: dict[str, Any] = domain_bucket.setdefault(self.entry_id, {})
        state_bucket: dict[str, dict[str, Any]] = entry_bucket.setdefault(
            "ws_state", {}
        )
        state: dict[str, Any] = state_bucket.setdefault(self.dev_id, {})
        state["last_event_at"] = now
        state["frames_total"] = self._stats.frames_total
        state["events_total"] = self._stats.events_total

        if (
            self._connected_since
            and not self._healthy_since
            and (now - self._connected_since) >= 300
        ):
            self._healthy_since = now
            self._update_status("healthy")
