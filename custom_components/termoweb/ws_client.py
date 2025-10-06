# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass
import gzip
import json
import logging
import random
import string
import time
from time import monotonic as time_mod
from typing import Any, Iterable, Mapping
from urllib.parse import urlencode, urlsplit, urlunsplit

import aiohttp
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_send

from .api import RESTClient
from .const import (
    ACCEPT_LANGUAGE,
    API_BASE,
    BRAND_DUCAHEAT,
    BRAND_TERMOWEB,
    DOMAIN,
    USER_AGENT,
    WS_NAMESPACE,
    get_brand_api_base,
    get_brand_requested_with,
    get_brand_user_agent,
    signal_ws_data,
    signal_ws_status,
)
from .installation import InstallationSnapshot, ensure_snapshot
from .nodes import (
    NODE_CLASS_BY_TYPE,
    addresses_by_node_type,
    ensure_node_inventory,
    collect_heater_sample_addresses,
    heater_sample_subscription_targets,
    normalize_heater_addresses,
    normalize_node_addr,
    normalize_node_type,
)

_LOGGER = logging.getLogger(__name__)

DUCAHEAT_NAMESPACE = "/api/v2/socket_io"


@dataclass
class WSStats:
    frames_total: int = 0
    events_total: int = 0
    last_event_ts: float = 0.0


class HandshakeError(RuntimeError):
    def __init__(self, status: int, url: str, detail: str) -> None:
        super().__init__(f"handshake failed: status={status}, detail={detail}")
        self.status = status
        self.url = url


class _WsLeaseMixin:
    def __init__(self) -> None:
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


class WebSocketClient(_WsLeaseMixin):
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
    ) -> None:
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
        self._delegate: Any | None = None
        self._brand = BRAND_DUCAHEAT if getattr(api_client, "_is_ducaheat", False) else BRAND_TERMOWEB

    def start(self) -> asyncio.Task:
        if self._delegate is not None:
            return self._delegate.start()
        if self._brand == BRAND_DUCAHEAT:
            self._delegate = DucaheatWSClient(
                self.hass,
                entry_id=self.entry_id,
                dev_id=self.dev_id,
                api_client=self._client,
                coordinator=self._coordinator,
                session=self._session,
                namespace=DUCAHEAT_NAMESPACE,
            )
        else:
            self._delegate = TermoWebWSClient(
                self.hass,
                entry_id=self.entry_id,
                dev_id=self.dev_id,
                api_client=self._client,
                coordinator=self._coordinator,
                session=self._session,
                namespace=self._namespace,
            )
        return self._delegate.start()

    async def stop(self) -> None:
        if self._delegate is not None:
            await self._delegate.stop()

    def is_running(self) -> bool:
        return bool(self._delegate and self._delegate.is_running())

    async def ws_url(self) -> str:
        if self._delegate and hasattr(self._delegate, "ws_url"):
            return await self._delegate.ws_url()
        return ""


# ---------------------- shared helpers ----------------------

def _now_pair() -> tuple[float, float]:
    return time_mod(), time.time()


def _rand_t() -> str:
    alphabet = string.ascii_letters + string.digits
    return "P" + "".join(random.choice(alphabet) for _ in range(7))


def _encode_polling_packet(pkt: str) -> bytes:
    data = pkt.encode("utf-8")
    return f"{len(data)}:".encode("ascii") + data


def _decode_polling_packets(body: bytes) -> list[str]:
    """Engine.IO v3 *binary* polling decoder per vendor dump.
    Format (repeat): [0x00|0x01] <digit-bytes '0'..'9' until 0xFF> 0xFF <payload>
    length is the decimal value of the digit bytes sequence.
    """
    buf = body
    if len(buf) >= 2 and buf[:2] == b"\x1f\x8b":
        try:
            buf = gzip.decompress(buf)
        except Exception:
            pass
    out: list[str] = []
    i = 0
    n = len(buf)
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
    # For polling requests only (WS sets its own)
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


class _WSCommon:
    def _ws_state_bucket(self) -> dict[str, Any]:
        if not hasattr(self.hass, "data") or self.hass.data is None:  # type: ignore[attr-defined]
            setattr(self.hass, "data", {})  # type: ignore[attr-defined]
        domain_bucket = self.hass.data.setdefault(DOMAIN, {})  # type: ignore[attr-defined]
        entry_bucket = domain_bucket.setdefault(self.entry_id, {})
        ws_state = entry_bucket.setdefault("ws_state", {})
        return ws_state.setdefault(self.dev_id, {})

    def _update_status(self, status: str) -> None:
        async_dispatcher_send(
            self.hass, signal_ws_status(self.entry_id), {"dev_id": self.dev_id, "status": status}
        )

    def _dispatch_nodes(self, payload: dict[str, Any]) -> None:
        raw_nodes = payload.get("nodes") if "nodes" in payload else payload
        record = self.hass.data.get(DOMAIN, {}).get(self.entry_id)
        snapshot_obj = ensure_snapshot(record)
        if isinstance(snapshot_obj, InstallationSnapshot):
            snapshot_obj.update_nodes(raw_nodes)
            inventory = snapshot_obj.inventory
            if isinstance(record, dict):
                record["node_inventory"] = list(inventory)
        else:
            record_map: Mapping[str, Any] = record if isinstance(record, Mapping) else {}
            inventory = ensure_node_inventory(record_map, nodes=raw_nodes)
        addr_map, _ = addresses_by_node_type(inventory, known_types=NODE_CLASS_BY_TYPE)
        if hasattr(self._coordinator, "update_nodes"):
            self._coordinator.update_nodes(raw_nodes, inventory)
        payload_copy = {
            "dev_id": self.dev_id,
            "node_type": None,
            "nodes": deepcopy(raw_nodes),
            "addr_map": {t: list(a) for t, a in addr_map.items()},
        }
        async_dispatcher_send(self.hass, signal_ws_data(self.entry_id), payload_copy)


# ---------------------- TermoWeb legacy stub (not used for Ducaheat) ----------------------

class TermoWebWSClient(_WsLeaseMixin, _WSCommon):
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
    ) -> None:
        _WsLeaseMixin.__init__(self)
        self.hass = hass
        self.entry_id = entry_id
        self.dev_id = dev_id
        self._client = api_client
        self._coordinator = coordinator
        self._session = session or getattr(api_client, "_session", None)
        self._loop = getattr(hass, "loop", None) or asyncio.get_event_loop()
        self._namespace = namespace or WS_NAMESPACE
        self._task: asyncio.Task | None = None

    def start(self) -> asyncio.Task:
        if self._task and not self._task.done():
            return self._task
        self._task = self._loop.create_task(self._stub(), name=f"{DOMAIN}-ws-{self.dev_id}")
        return self._task

    async def _stub(self) -> None:
        _LOGGER.info("TermoWebWSClient: legacy path not modified in this file")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    def is_running(self) -> bool:
        return bool(self._task and not self._task.done())


# ---------------------- Ducaheat exact EIO3 client ----------------------

class DucaheatWSClient(_WsLeaseMixin, _WSCommon):
    """Vendor flow: /socket.io/ + EIO=3 polling -> POST '40/ns' -> drain GET -> WS upgrade."""

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
    ) -> None:
        _WsLeaseMixin.__init__(self)
        self.hass = hass
        self.entry_id = entry_id
        self.dev_id = dev_id
        self._client = api_client
        self._coordinator = coordinator
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

    def start(self) -> asyncio.Task:
        if self._task and not self._task.done():
            return self._task
        self._task = self._loop.create_task(self._runner(), name=f"{DOMAIN}-ws-{self.dev_id}")
        return self._task

    async def stop(self) -> None:
        await self._disconnect("stop")
        if self._task:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    def is_running(self) -> bool:
        return bool(self._task and not self._task.done())

    def _base_host(self) -> str:
        base = get_brand_api_base(self._brand).rstrip("/")
        parsed = urlsplit(base if base else API_BASE)
        return urlunsplit((parsed.scheme or "https", parsed.netloc or parsed.path, "", "", ""))

    def _path(self) -> str:
        return "/socket.io/"  # trailing slash per dump

    async def ws_url(self) -> str:
        token = await self._get_token()
        params = {
            "token": token,
            "dev_id": self.dev_id,
            "EIO": "3",
            "transport": "polling",
            "t": _rand_t(),
        }
        return urlunsplit(urlsplit(self._base_host())._replace(path=self._path(), query=urlencode(params)))

    async def _runner(self) -> None:
        self._update_status("starting")
        try:
            while True:
                try:
                    await self._connect_once()
                    await self._read_loop_ws()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    _LOGGER.debug("WS (ducaheat): error %s", e, exc_info=True)
                    await asyncio.sleep(self._next_backoff())
                finally:
                    await self._disconnect("loop")
                    self._update_status("disconnected")
        finally:
            self._update_status("stopped")

    async def _connect_once(self) -> None:
        token = await self._get_token()
        headers = _brand_headers(self._ua, self._xrw)

        # 1) Polling OPEN (GET)
        open_params = {
            "token": token,
            "dev_id": self.dev_id,
            "EIO": "3",
            "transport": "polling",
            "t": _rand_t(),
        }
        open_url = urlunsplit(urlsplit(self._base_host())._replace(path=self._path(), query=urlencode(open_params)))
        _LOGGER.debug("WS (ducaheat): OPEN GET %s", open_url.replace(token, "…"))
        async with asyncio.timeout(15):
            async with self._session.get(open_url, headers=headers) as resp:
                body = await resp.read()
                _LOGGER.debug("WS (ducaheat): OPEN GET -> %s bytes (status=%s)", len(body), resp.status)
                if resp.status != 200:
                    raise HandshakeError(resp.status, open_url, "open GET")
        packets = _decode_polling_packets(body)
        open_pkt = next((p for p in packets if p and p[0] == "0"), None)
        if not open_pkt:
            _LOGGER.debug("WS (ducaheat): open raw first 120 bytes: %r", body[:120])
            raise HandshakeError(590, open_url, "missing OPEN (0)")
        try:
            info = json.loads(open_pkt[1:])
        except Exception:
            raise HandshakeError(591, open_url, "invalid OPEN json")
        sid = info.get("sid")
        ping_interval = info.get("pingInterval")
        ping_timeout = info.get("pingTimeout")
        _LOGGER.info("WS (ducaheat): OPEN decoded sid=%s pingInterval=%s pingTimeout=%s", sid, ping_interval, ping_timeout)
        if not isinstance(sid, str) or not sid:
            raise HandshakeError(592, open_url, "missing sid")

        # 2) Polling POST SIO open "40/ns"
        post_params = {
            "token": token,
            "dev_id": self.dev_id,
            "EIO": "3",
            "transport": "polling",
            "t": _rand_t(),
            "sid": sid,
        }
        post_url = urlunsplit(urlsplit(self._base_host())._replace(path=self._path(), query=urlencode(post_params)))
        sio_open = f"40{self._namespace}"
        payload = _encode_polling_packet(sio_open)
        _LOGGER.debug("WS (ducaheat): POST 40/ns %s", post_url.replace(token, "…"))
        async with asyncio.timeout(15):
            async with self._session.post(
                post_url,
                headers={**headers, "Content-Type": "text/plain;charset=UTF-8"},
                data=payload,
            ) as resp:
                body = await resp.read()
                _LOGGER.debug("WS (ducaheat): POST 40/ns -> status=%s len=%s", resp.status, len(body))
                if resp.status != 200:
                    raise HandshakeError(resp.status, post_url, "POST 40/ns")

        # 2b) Polling GET to drain acks
        drain_params = dict(post_params)
        drain_params["t"] = _rand_t()
        drain_url = urlunsplit(urlsplit(self._base_host())._replace(path=self._path(), query=urlencode(drain_params)))
        _LOGGER.debug("WS (ducaheat): DRAIN GET %s", drain_url.replace(token, "…"))
        async with asyncio.timeout(15):
            async with self._session.get(drain_url, headers=headers) as resp:
                body = await resp.read()
                _LOGGER.debug("WS (ducaheat): DRAIN GET -> status=%s len=%s", resp.status, len(body))
                if resp.status != 200:
                    raise HandshakeError(resp.status, drain_url, "drain GET")

        # 3) WebSocket upgrade (no t param)
        ws_params = {
            "token": token,
            "dev_id": self.dev_id,
            "EIO": "3",
            "transport": "websocket",
            "sid": sid,
        }
        ws_url = urlunsplit(urlsplit(self._base_host())._replace(path=self._path(), query=urlencode(ws_params)))
        ws_headers = {k: v for k, v in headers.items() if k.lower() not in ("connection", "accept-encoding")}
        _LOGGER.debug("WS (ducaheat): upgrading WS %s", ws_url.replace(token, "…"))
        self._ws = await self._session.ws_connect(
            ws_url,
            headers=ws_headers,
            heartbeat=None,
            autoclose=False,
            timeout=aiohttp.ClientTimeout(total=15),
        )
        _LOGGER.info("WS (ducaheat): upgrade OK")

        # EIO3 probe/upgrade
        await self._ws.send_str("2probe")
        _LOGGER.debug("WS (ducaheat): sent 2probe")
        probe = await self._ws.receive_str()
        _LOGGER.debug("WS (ducaheat): recv %r", probe)
        if probe != "3probe":
            _LOGGER.debug("WS (ducaheat): unexpected probe ack: %r", probe)
        await self._ws.send_str("5")
        _LOGGER.debug("WS (ducaheat): sent 5 (upgrade)")

        await self._ws.send_str(f"40{self._namespace}")
        _LOGGER.debug("WS (ducaheat): namespace open sent %s", self._namespace)

        await self._emit_sio("dev_data")
        _LOGGER.debug("WS (ducaheat): dev_data emitted")

        subs = await self._subscribe_samples()
        _LOGGER.info("WS (ducaheat): subscribed %d sample targets", subs)

        self._update_status("connected")

    async def _read_loop_ws(self) -> None:
        ws = self._ws
        if ws is None:
            return
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = msg.data
                self._stats.frames_total += 1
                if data == "2":  # ping
                    _LOGGER.debug("WS (ducaheat): <- 2 (ping) -> 3 (pong)")
                    await ws.send_str("3")
                    continue
                if not data.startswith("4"):  # not Engine.IO message
                    continue
                payload = data[1:]
                if payload.startswith("40"):
                    _LOGGER.debug("WS (ducaheat): 40 (namespace open ack)")
                    self._update_status("healthy")
                    continue
                if payload.startswith("42"):
                    content = payload[2:]
                    if content.startswith("/"):
                        _ns, sep, content = content.partition(",")
                        if sep != ",":
                            continue
                    try:
                        arr = json.loads(content)
                    except Exception:
                        continue
                    if not isinstance(arr, list) or not arr:
                        continue
                    evt = arr[0]
                    args = arr[1:]
                    _LOGGER.debug("WS (ducaheat): 42 event=%s size=%s", evt, sum(len(json.dumps(a)) if not isinstance(a, str) else len(a) for a in args))
                    if evt == "message" and args and args[0] == "ping":
                        _LOGGER.debug("WS (ducaheat): app ping -> pong")
                        await self._emit_sio("message", "pong")
                        continue
                    if evt == "dev_data" and args and isinstance(args[0], dict):
                        nodes = args[0].get("nodes") if isinstance(args[0], dict) else None
                        if isinstance(nodes, dict):
                            self._log_nodes_summary(nodes)
                            self._dispatch_nodes({"nodes": nodes})
                            self._update_status("healthy")
                        continue
                    if evt == "update" and args:
                        self._log_update_brief(args[0])
                        self._update_status("healthy")
                        continue
                    continue
            elif msg.type == aiohttp.WSMsgType.ERROR:
                raise RuntimeError(f"websocket error: {ws.exception()}")
            elif msg.type in {aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED}:
                raise RuntimeError("websocket closed")

    def _log_nodes_summary(self, nodes: Mapping[str, Any]) -> None:
        if not _LOGGER.isEnabledFor(logging.INFO):
            return
        kinds = []
        for k, v in nodes.items():
            if isinstance(v, Mapping):
                # count addrs under settings/samples/etc.
                addrs = set()
                for section in ("settings", "samples", "status", "advanced"):
                    sec = v.get(section)
                    if isinstance(sec, Mapping):
                        addrs.update(a for a in sec.keys() if isinstance(a, str))
                kinds.append(f"{k}={len(addrs) if addrs else 0}")
        _LOGGER.info("WS (ducaheat): dev_data nodes: %s", " ".join(kinds) if kinds else "(no nodes)")

    def _log_update_brief(self, body: Any) -> None:
        if not _LOGGER.isEnabledFor(logging.DEBUG):
            return
        path = None
        keys = None
        if isinstance(body, Mapping):
            path = body.get("path")
            b = body.get("body")
            if isinstance(b, Mapping):
                keys = ",".join(list(b.keys())[:6])
        _LOGGER.debug("WS (ducaheat): update path=%s keys=%s", path, keys)

    def _translate_path_update(self, payload: Mapping[str, Any]) -> dict[str, Any] | None:
        path = payload.get("path")
        body = payload.get("body")
        if not isinstance(path, str):
            return None
        path = path.split("?", 1)[0]
        parts = [p for p in path.split("/") if p]
        if not parts:
            return None
        try:
            idx = parts.index("devs")
            parts = parts[idx + 1 :]
        except ValueError:
            pass
        if len(parts) < 3:
            return None
        node_type = normalize_node_type(parts[0])
        addr = normalize_node_addr(parts[1])
        if not node_type or not addr:
            return None
        section = parts[2] if len(parts) > 2 else None
        if section in {"samples", "status", "settings", "advanced"}:
            return {node_type: {section: {addr: body}}}
        return {node_type: {"settings": {addr: {section: body} if section else body}}}

    async def _emit_sio(self, event: str, *args: Any) -> None:
        if not self._ws or self._ws.closed:
            raise RuntimeError("websocket not connected")
        arr = [event, *args]
        payload = json.dumps(arr, separators=(",", ":"), default=str)
        pkt = f"42{DUCAHEAT_NAMESPACE}," + payload
        await self._ws.send_str(pkt)

    async def _subscribe_samples(self) -> int:
        """Subscribe to heater/accumulator sample updates; return count."""
        count = 0
        try:
            record = self.hass.data.get(DOMAIN, {}).get(self.entry_id)
            inventory, normalized_map, _ = collect_heater_sample_addresses(
                record, coordinator=self._coordinator
            )
            normalized_map, _ = normalize_heater_addresses(normalized_map)
            targets = list(heater_sample_subscription_targets(normalized_map))
            for node_type, addr in targets:
                await self._emit_sio("subscribe", f"/{node_type}/{addr}/samples")
            count = len(targets)
        except Exception:
            _LOGGER.debug("WS (ducaheat): subscribe failed", exc_info=True)
        return count

    async def _disconnect(self, reason: str) -> None:
        if self._ws:
            with suppress(aiohttp.ClientError, RuntimeError):
                await self._ws.close(code=aiohttp.WSCloseCode.GOING_AWAY, message=reason.encode())
            self._ws = None

    async def _get_token(self) -> str:
        headers = await self._client.authed_headers()
        auth = headers.get("Authorization") if isinstance(headers, dict) else None
        if not auth:
            raise RuntimeError("missing Authorization")
        return auth.split(" ", 1)[1]


__all__ = ["WebSocketClient", "DucaheatWSClient", "TermoWebWSClient"]

