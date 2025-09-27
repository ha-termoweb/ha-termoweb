from __future__ import annotations

import asyncio
from collections.abc import Iterable
import json
import logging
import random
import time
from typing import Any

import aiohttp
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_send

from .api import TermoWebClient
from .const import API_BASE, DOMAIN, WS_NAMESPACE, signal_ws_data, signal_ws_status
from .ws_shared import HandshakeError, TermoWebWSShared, apply_updates

_LOGGER = logging.getLogger(__name__)

HandshakeResult = tuple[str, int]  # (sid, heartbeat_timeout_s)


class TermoWebWSLegacyClient(TermoWebWSShared):
    """Minimal read-only Socket.IO 0.9 client for TermoWeb cloud.

    Behavior:
      - GET /socket.io/1/ handshake (with token/dev_id query)
      - WebSocket connect to /socket.io/1/websocket/<sid>?token=...&dev_id=...
      - Join namespace: "1::/api/v2/socket_io"
      - Send one snapshot request: 5::/api/v2/socket_io:{"name":"dev_data","args":[]}
      - Maintain heartbeat: reply/send "2::" periodically
      - Parse "5::/api/v2/socket_io:{...}" events; look for {"name":"data","args":[[{"path", "body"}, ...]]}
      - Normalize updates into coordinator.data and dispatch via HA dispatcher
      - Read-only: never sends control commands
    """

    def __init__(
        self,
        hass: HomeAssistant,
        *,
        entry_id: str,
        dev_id: str,
        api_client: TermoWebClient,
        coordinator,  # TermoWebCoordinator; typed as Any to avoid import cycle
        session: aiohttp.ClientSession | None = None,
        handshake_fail_threshold: int = 5,
    ) -> None:
        super().__init__(
            hass,
            entry_id=entry_id,
            dev_id=dev_id,
            task_name=f"{DOMAIN}-ws-{dev_id}",
        )
        self._client = api_client
        self._coordinator = coordinator
        self._session = session or api_client._session  # reuse HA session
        self._hb_send_interval: float = 27.0  # default; refined from handshake timeout

        self._backoff_seq = [5, 10, 30, 120, 300]  # seconds
        self._backoff_idx = 0

        self._hs_fail_count: int = 0
        self._hs_fail_start: float = 0.0
        self._hs_fail_threshold: int = handshake_fail_threshold

    # ----------------- Core loop -----------------

    async def _runner(self) -> None:
        self._update_status("starting")
        while not self._closing:
            should_retry = True
            try:
                sid, hb_timeout = await self._handshake()
                self._hs_fail_count = 0
                self._hs_fail_start = 0.0
                # keep client-side send interval well under server timeout
                self._hb_send_interval = max(5.0, min(30.0, hb_timeout * 0.45))
                await self._connect_ws(sid)
                await self._join_namespace()
                await self._send_snapshot_request()
                await self._subscribe_htr_samples()
                self._connected_since = time.time()
                self._healthy_since = None
                self._update_status("connected")

                # Start heartbeat sender
                self._hb_task = self.hass.loop.create_task(self._heartbeat_loop())

                # Read until disconnect
                await self._read_loop()

            except asyncio.CancelledError:
                should_retry = False
            except HandshakeError as e:
                self._hs_fail_count += 1
                if self._hs_fail_count == 1:
                    self._hs_fail_start = time.time()
                # Avoid noisy logs; just one INFO per failure with brief cause.
                _LOGGER.info(
                    "WS %s: connection error (%s: %s); will retry",
                    self.dev_id,
                    type(e).__name__,
                    e,
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
                    e.url,
                    e.body_snippet,
                )
            except Exception as e:
                # Avoid noisy logs; just one INFO per failure with brief cause.
                _LOGGER.info(
                    "WS %s: connection error (%s: %s); will retry",
                    self.dev_id,
                    type(e).__name__,
                    e,
                )
                _LOGGER.debug(
                    "WS %s: connection error details", self.dev_id, exc_info=True
                )
            finally:
                # Clean up this attempt
                if self._hb_task:
                    self._hb_task.cancel()
                    self._hb_task = None
                if self._ws:
                    try:
                        await self._ws.close()
                    except Exception:
                        pass
                    self._ws = None

                self._update_status("disconnected")

            if self._closing or not should_retry:
                break

            # Backoff with jitter
            delay = self._backoff_seq[
                min(self._backoff_idx, len(self._backoff_seq) - 1)
            ]
            self._backoff_idx = min(
                self._backoff_idx + 1, len(self._backoff_seq) - 1
            )
            jitter = random.uniform(0.8, 1.2)
            await asyncio.sleep(delay * jitter)

        # End loop
        self._update_status("stopped")

    # ----------------- Protocol steps -----------------

    async def _handshake(self) -> HandshakeResult:
        """GET /socket.io/1/?token=<Bearer>&dev_id=<dev_id>&t=<ms>
        Returns: <sid>:<hb>:<disc>:websocket,xhr-polling
        """
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
                        # Token expired; refresh once and retry
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
                            self._backoff_idx = 0  # success resets backoff
                            return sid, hb

                    if resp.status >= 400:
                        raise HandshakeError(resp.status, url, body[:100])

                    sid, hb = self._parse_handshake_body(body)
                    self._backoff_idx = 0  # success resets backoff
                    return sid, hb
        except (asyncio.TimeoutError, aiohttp.ClientError) as error:
            raise HandshakeError(-1, url, str(error)) from error

    async def _connect_ws(self, sid: str) -> None:
        token = await self._get_token()
        base = self._api_base()
        ws_base = base.replace("https://", "wss://", 1)
        ws_url = (
            f"{ws_base}/socket.io/1/websocket/{sid}?token={token}&dev_id={self.dev_id}"
        )
        self._ws = await self._session.ws_connect(
            ws_url,
            heartbeat=None,  # we implement our own '2::' heartbeats
            timeout=15,
            autoclose=True,
            autoping=False,
            compress=0,
            protocols=("websocket",),
        )

    async def _join_namespace(self) -> None:
        await self._send_text(f"1::{WS_NAMESPACE}")

    async def _send_snapshot_request(self) -> None:
        # 5::/api/v2/socket_io:{"name":"dev_data","args":[]}
        payload = {"name": "dev_data", "args": []}
        await self._send_text(
            f"5::{WS_NAMESPACE}:{json.dumps(payload, separators=(',', ':'))}"
        )

    async def _subscribe_htr_samples(self) -> None:
        """Request push updates for heater energy samples."""
        addrs = (
            self._coordinator._addrs() if hasattr(self._coordinator, "_addrs") else []
        )
        for addr in addrs:
            payload = {"name": "subscribe", "args": [f"/htr/{addr}/samples"]}
            await self._send_text(
                f"5::{WS_NAMESPACE}:{json.dumps(payload, separators=(',', ':'))}"
            )

    # ----------------- Loops -----------------

    async def _heartbeat_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._hb_send_interval)
                await self._send_text("2::")
        except asyncio.CancelledError:
            return
        except Exception:
            # Sending heartbeat failed; the read loop will notice soon.
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
            self._stats.frames_total += 1

            # Socket.IO 0.9 frames
            if data.startswith("2::"):
                # Heartbeat from server; track liveness
                self._mark_event(paths=None)
                continue
            if data.startswith(f"1::{WS_NAMESPACE}"):
                # Namespace ack; nothing to do
                continue
            if data.startswith(f"5::{WS_NAMESPACE}:"):
                try:
                    js = json.loads(data.split(f"5::{WS_NAMESPACE}:", 1)[1])
                except Exception:
                    continue
                self._handle_event(js)
                continue
            if data.startswith("0::"):
                # Disconnect
                raise RuntimeError("server disconnect")
            # ignore other frame types

    # ----------------- Event handling -----------------

    def _handle_event(self, evt: dict[str, Any]) -> None:
        """Expecting: {"name": "data", "args": [ [ {"path": "...", "body": {...}}, ... ] ]}"""
        if not isinstance(evt, dict):
            return
        name = evt.get("name")
        args = evt.get("args")
        if name != "data" or not isinstance(args, list) or not args:
            return
        batch = args[0] if isinstance(args[0], list) else None
        if not isinstance(batch, list):
            return

        (
            paths,
            updated_nodes,
            updated_addrs,
            sample_addrs,
        ) = apply_updates(self._coordinator, self.dev_id, batch)

        # Dispatch (one compact signal)
        self._mark_event(paths=paths)
        payload_base = {"dev_id": self.dev_id, "ts": self._stats.last_event_ts}
        if updated_nodes:
            async_dispatcher_send(
                self.hass,
                signal_ws_data(self.entry_id),
                {**payload_base, "addr": None, "kind": "nodes"},
            )
        for addr in updated_addrs:
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

    # ----------------- Helpers -----------------

    def _parse_handshake_body(self, body: str) -> HandshakeResult:
        # "<sid>:<heartbeat>:<disconnect>:<transports>"
        parts = (body or "").strip().split(":")
        if len(parts) < 2:
            raise RuntimeError("handshake malformed")
        sid = parts[0]
        try:
            hb = int(parts[1])
        except Exception:
            hb = 60
        return sid, hb

    async def _send_text(self, data: str) -> None:
        if not self._ws:
            return
        await self._ws.send_str(data)

    async def _get_token(self) -> str:
        # Borrow token from API client without logging it
        headers = await self._client._authed_headers()
        return headers["Authorization"].split(" ", 1)[1]

    async def _force_refresh_token(self) -> None:
        # Clear cached token and fetch a new one
        try:
            # noinspection PyUnresolvedReferences
            self._client._access_token = None  # type: ignore[attr-defined]
        except Exception:
            pass
        await self._client._ensure_token()

    def _api_base(self) -> str:
        base = getattr(self._client, "api_base", None)
        if isinstance(base, str) and base:
            return base.rstrip("/")
        return API_BASE

    def _ws_close_exceptions(self) -> Iterable[type[BaseException]]:  # type: ignore[override]
        return (Exception,)


__all__ = ["HandshakeError", "TermoWebWSLegacyClient", "signal_ws_status"]
