# -*- coding: utf-8 -*-
"""Ducaheat specific websocket client."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import gzip
import json
import logging
import random
import string
from typing import Any, Mapping
from urllib.parse import urlencode, urlsplit, urlunsplit

import aiohttp
from homeassistant.core import HomeAssistant

from ..api import RESTClient
from ..const import (
    ACCEPT_LANGUAGE,
    API_BASE,
    BRAND_DUCAHEAT,
    DOMAIN,
    USER_AGENT,
    get_brand_api_base,
    get_brand_requested_with,
    get_brand_user_agent,
)
from ..nodes import (
    collect_heater_sample_addresses,
    heater_sample_subscription_targets,
    normalize_heater_addresses,
    normalize_node_addr,
    normalize_node_type,
)
from .ws_client import (
    DUCAHEAT_NAMESPACE,
    HandshakeError,
    WSStats,
    _WSCommon,
    _WsLeaseMixin,
)

_LOGGER = logging.getLogger(__name__)


def _rand_t() -> str:
    """Return a pseudo-random token string for polling requests."""

    return "P" + "".join(random.choice(string.ascii_letters + string.digits) for _ in range(7))


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
        self._latest_nodes: Mapping[str, Any] | None = None
        self._nodes_raw: dict[str, Any] | None = None
        self._subscription_paths: set[str] = set()
        self._pending_dev_data = False
        self._keepalive_task: asyncio.Task | None = None
        self._ping_interval: float | None = None
        self._ping_timeout: float | None = None

    def start(self) -> asyncio.Task:
        if self._task and not self._task.done():
            return self._task
        self._task = self._loop.create_task(self._runner(), name=f"termoweb-ws-{self.dev_id}")
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
        return urlunsplit((parsed.scheme or "https", parsed.netloc or parsed.path, "", "", ""))

    def _path(self) -> str:
        return "/socket.io/"

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
        open_url = urlunsplit(urlsplit(self._base_host())._replace(path=self._path(), query=urlencode(open_params)))
        _LOGGER.debug("WS (ducaheat): OPEN GET %s", open_url.replace(token, "…"))
        async with asyncio.timeout(15):
            async with self._session.get(open_url, headers=headers) as resp:
                body = await resp.read()
                _LOGGER.debug(
                    "WS (ducaheat): OPEN GET -> %s bytes (status=%s)",
                    len(body),
                    resp.status,
                )
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
            float(ping_interval) / 1000 if isinstance(ping_interval, (int, float)) else None
        )
        ping_timeout = info.get("pingTimeout")
        self._ping_timeout = (
            float(ping_timeout) / 1000 if isinstance(ping_timeout, (int, float)) else None
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
        post_url = urlunsplit(urlsplit(self._base_host())._replace(path=self._path(), query=urlencode(post_params)))
        payload = _encode_polling_packet(f"40{self._namespace}")
        _LOGGER.debug("WS (ducaheat): POST 40/ns %s", post_url.replace(token, "…"))
        async with asyncio.timeout(15):
            async with self._session.post(
                post_url,
                headers={**headers, "Content-Type": "text/plain;charset=UTF-8"},
                data=payload,
            ) as resp:
                drain = await resp.read()
                _LOGGER.debug(
                    "WS (ducaheat): POST 40/ns -> status=%s len=%s",
                    resp.status,
                    len(drain),
                )
                if resp.status != 200:
                    raise HandshakeError(resp.status, post_url, "POST 40/ns")

        drain_params = dict(post_params)
        drain_params["t"] = _rand_t()
        drain_url = urlunsplit(urlsplit(self._base_host())._replace(path=self._path(), query=urlencode(drain_params)))
        _LOGGER.debug("WS (ducaheat): DRAIN GET %s", drain_url.replace(token, "…"))
        async with asyncio.timeout(15):
            async with self._session.get(drain_url, headers=headers) as resp:
                body = await resp.read()
                _LOGGER.debug(
                    "WS (ducaheat): DRAIN GET -> status=%s len=%s",
                    resp.status,
                    len(body),
                )
                if resp.status != 200:
                    raise HandshakeError(resp.status, drain_url, "drain GET")

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

        await self._ws.send_str("2probe")
        _LOGGER.debug("WS (ducaheat): -> 2probe")
        probe = await self._ws.receive_str()
        _LOGGER.debug("WS (ducaheat): <- %r", probe)
        if probe != "3probe":
            _LOGGER.debug("WS (ducaheat): unexpected probe ack: %r", probe)
        await self._ws.send_str("5")
        _LOGGER.debug("WS (ducaheat): -> 5 (upgrade)")

        await self._ws.send_str(f"40{self._namespace}")
        _LOGGER.debug("WS (ducaheat): -> 40%s", self._namespace)
        self._pending_dev_data = True
        _LOGGER.debug("WS (ducaheat): dev_data pending until namespace ack")
        self._update_status("connected")
        self._start_keepalive()

    async def _read_loop_ws(self) -> None:
        ws = self._ws
        if ws is None:
            return
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = msg.data
                    self._stats.frames_total += 1
                    while data:
                        if data == "2":
                            _LOGGER.debug("WS (ducaheat): <- EIO 2 (ping) -> EIO 3 (pong)")
                            await ws.send_str("3")
                            break
                        if data == "3":
                            _LOGGER.debug("WS (ducaheat): <- 3 (pong)")
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
                                        data = remainder if remainder.startswith("4") else "4" + remainder
                                        continue
                                    break
                                if sep and remainder:
                                    rest_payload = remainder
                            _LOGGER.debug("WS (ducaheat): <- SIO 40 (namespace ack)")
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
                                data = rest_payload if rest_payload.startswith("4") else "4" + rest_payload
                                continue
                            break
                        payload = data[1:]
                        if payload == "2":
                            _LOGGER.debug("WS (ducaheat): <- SIO 2 (ping) -> SIO 3 (pong)")
                            await ws.send_str("3")
                            break
                        if payload == "3":
                            _LOGGER.debug("WS (ducaheat): <- SIO 3 (pong)")
                            self._update_status("healthy")
                            break
                        if payload.startswith("2/"):
                            ns_payload = payload[1:]
                            ns, sep, body = ns_payload.partition(",")
                            if not sep or body in {"", "[]", "[\"ping\"]"}:
                                _LOGGER.debug(
                                    "WS (ducaheat): <- SIO 2%s (ping) -> SIO 3%s (pong)",
                                    ns,
                                    ns,
                                )
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
                        _LOGGER.debug("WS (ducaheat): <- SIO 42 event=%s args_len=%d", evt, len(args))

                        if evt == "message" and args and args[0] == "ping":
                            _LOGGER.debug("WS (ducaheat): app ping -> pong")
                            await self._emit_sio("message", "pong")
                            break

                        if evt == "dev_data" and args and isinstance(args[0], dict):
                            nodes = args[0].get("nodes") if isinstance(args[0], dict) else None
                            if isinstance(nodes, dict):
                                self._latest_nodes = nodes
                                self._log_nodes_summary(nodes)
                                normalised = self._normalise_nodes_payload(nodes)
                                if isinstance(normalised, dict):
                                    self._nodes_raw = deepcopy(normalised)
                                    snapshot = self._build_nodes_snapshot(self._nodes_raw)
                                else:
                                    self._nodes_raw = None
                                    snapshot = {"nodes": nodes}
                                self._dispatch_nodes(snapshot)
                                subs = await self._subscribe_feeds(nodes)
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
                                normalised_update = self._normalise_nodes_payload(translated)
                                if isinstance(normalised_update, dict) and normalised_update:
                                    sample_updates = self._collect_sample_updates(normalised_update)
                                    if self._nodes_raw is None:
                                        self._nodes_raw = deepcopy(normalised_update)
                                    else:
                                        self._merge_nodes(self._nodes_raw, normalised_update)
                                    snapshot = self._build_nodes_snapshot(self._nodes_raw)
                                    self._dispatch_nodes(snapshot)
                                    if sample_updates:
                                        self._forward_sample_updates(sample_updates)
                            self._update_status("healthy")
                            break
                        break
                        break
                    continue
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    raise RuntimeError(f"websocket error: {ws.exception()}")
                elif msg.type in {aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED}:
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
                    _LOGGER.debug("WS (ducaheat): -> 2 (keepalive ping)")
                except Exception:  # noqa: BLE001 - defensive logging
                    _LOGGER.debug(
                        "WS (ducaheat): keepalive ping failed", exc_info=True
                    )
                    break
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - defensive
            _LOGGER.debug("WS (ducaheat): keepalive loop error", exc_info=True)
        finally:
            if self._keepalive_task is task:
                self._keepalive_task = None
            _LOGGER.debug("WS (ducaheat): keepalive loop stopped")

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
                        addrs.update(addr for addr in sec.keys() if isinstance(addr, str))
                kinds.append(f"{key}={len(addrs) if addrs else 0}")
        _LOGGER.info("WS (ducaheat): dev_data nodes: %s", " ".join(kinds) if kinds else "(no nodes)")

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

    @staticmethod
    def _build_nodes_snapshot(nodes: dict[str, Any]) -> dict[str, Any]:
        """Return a snapshot structure with ``nodes`` and ``nodes_by_type`` buckets."""

        nodes_copy = deepcopy(nodes)
        nodes_by_type = {
            node_type: payload
            for node_type, payload in nodes_copy.items()
            if isinstance(payload, dict)
        }
        snapshot: dict[str, Any] = {"nodes": nodes_copy, "nodes_by_type": nodes_by_type}
        snapshot.update(nodes_by_type)
        return snapshot

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

    @staticmethod
    def _collect_sample_updates(nodes: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
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
            if bucket:
                updates[node_type] = bucket
        return updates

    def _translate_path_update(self, payload: Any) -> dict[str, Any] | None:
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
        """Map a websocket path segment to the node section bucket."""

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

    def _forward_sample_updates(self, updates: Mapping[str, Mapping[str, Any]]) -> None:
        """Forward websocket heater sample updates to the energy coordinator."""

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
            )
        except Exception:  # noqa: BLE001  # pragma: no cover - defensive logging
            _LOGGER.debug("WS (ducaheat): forwarding heater samples failed", exc_info=True)

    async def _subscribe_feeds(self, nodes: Mapping[str, Any] | None) -> int:
        """Subscribe to heater status and sample feeds."""

        resolved_nodes: Mapping[str, Any] | None = nodes if isinstance(nodes, Mapping) else None
        if resolved_nodes is None and isinstance(self._latest_nodes, Mapping):
            resolved_nodes = self._latest_nodes

        paths: set[str] = set()
        try:
            domain_bucket = self.hass.data.setdefault(DOMAIN, {})
            entry_bucket = domain_bucket.setdefault(self.entry_id, {})
            if isinstance(entry_bucket, dict) and isinstance(resolved_nodes, Mapping):
                entry_bucket["nodes"] = resolved_nodes

            record = domain_bucket.get(self.entry_id)
            _, normalized_map, _ = collect_heater_sample_addresses(
                record,
                coordinator=self._coordinator,
            )
            normalized_map, _ = normalize_heater_addresses(normalized_map)
            targets = list(heater_sample_subscription_targets(normalized_map))
            for node_type, addr in targets:
                paths.add(f"/{node_type}/{addr}/status")
                paths.add(f"/{node_type}/{addr}/samples")
        except Exception:  # pragma: no cover - defensive
            _LOGGER.debug("WS (ducaheat): subscribe failed", exc_info=True)
            return 0

        if not paths:
            return 0

        for path in sorted(paths):
            await self._emit_sio("subscribe", path)

        self._subscription_paths = paths
        return len(paths)

    async def _replay_subscription_paths(self) -> None:
        """Replay cached subscription paths after a reconnect."""

        if not self._subscription_paths:
            return
        for path in sorted(self._subscription_paths):
            try:
                await self._emit_sio("subscribe", path)
            except Exception:  # noqa: BLE001  # pragma: no cover - defensive logging
                _LOGGER.debug(
                    "WS (ducaheat): replay subscribe failed for path %s", path, exc_info=True
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

    async def _get_token(self) -> str:
        headers = await self._client.authed_headers()
        auth = headers.get("Authorization") if isinstance(headers, dict) else None
        if not auth:
            raise RuntimeError("missing Authorization")
        return auth.split(" ", 1)[1]


__all__ = ["DucaheatWSClient"]
