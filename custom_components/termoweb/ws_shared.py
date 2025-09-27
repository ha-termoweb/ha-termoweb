"""Shared helpers for TermoWeb websocket clients."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
import contextlib
from dataclasses import dataclass
import logging
import time
from typing import Any

import aiohttp
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_send

from .const import DOMAIN, signal_ws_status
from .utils import extract_heater_addrs

_LOGGER = logging.getLogger(__name__)


class HandshakeError(RuntimeError):
    """Raised when the websocket handshake fails."""

    def __init__(self, status: int, url: str, body_snippet: str) -> None:
        """Store response information for logging and retries."""

        super().__init__(f"handshake failed (status={status})")
        self.status = status
        self.url = url
        self.body_snippet = body_snippet


@dataclass
class WSStats:
    """Simple counter bucket for websocket activity."""

    frames_total: int = 0
    events_total: int = 0
    last_event_ts: float = 0.0
    last_paths: list[str] | None = None


class TermoWebWSShared:
    """Mixin providing lifecycle and status helpers for websocket clients."""

    def __init__(
        self,
        hass: HomeAssistant,
        *,
        entry_id: str,
        dev_id: str,
        task_name: str | None = None,
    ) -> None:
        """Initialize shared websocket bookkeeping fields."""

        self.hass = hass
        self.entry_id = entry_id
        self.dev_id = dev_id
        self._runner_task_name_value = task_name
        self._task: asyncio.Task | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._hb_task: asyncio.Task | None = None
        self._closing = False
        self._connected_since: float | None = None
        self._healthy_since: float | None = None
        self._stats = WSStats()

    # ----------------- Public control -----------------

    def start(self) -> asyncio.Task:
        """Start the websocket runner task if not already running."""

        if self._task and not self._task.done():
            return self._task
        self._closing = False
        name = self._runner_task_name()
        self._task = self.hass.loop.create_task(self._runner(), name=name)
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
            with contextlib.suppress(*self._ws_close_exceptions()):
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

    # ----------------- Helpers -----------------

    def _runner_task_name(self) -> str:
        return self._runner_task_name_value or f"{DOMAIN}-ws-{self.dev_id}"

    def _ws_close_exceptions(self) -> Iterable[type[BaseException]]:
        return (TimeoutError, aiohttp.ClientError, RuntimeError)

    def _update_status(self, status: str) -> None:
        """Update shared websocket state and dispatch notifications."""

        state_bucket = self._state_bucket()
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
        """Record websocket activity and update health state."""

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

        state_bucket = self._state_bucket()
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

    def _state_bucket(self) -> dict[str, dict[str, Any]]:
        """Return the websocket state bucket, creating it if necessary."""

        domain_bucket: dict[str, Any] = self.hass.data.setdefault(DOMAIN, {})
        entry_bucket: dict[str, Any] = domain_bucket.setdefault(self.entry_id, {})
        return entry_bucket.setdefault("ws_state", {})


def apply_updates(
    coordinator: Any,
    dev_id: str,
    updates: Iterable[dict[str, Any] | None],
) -> tuple[list[str], bool, list[str], list[str]]:
    """Normalize websocket updates into coordinator data and collect change hints."""

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

        current = getattr(coordinator, "data", None) or {}
        dev_map: dict[str, Any] = current.get(dev_id) or {}
        if not dev_map:
            dev_map = {
                "dev_id": dev_id,
                "name": f"Device {dev_id}",
                "raw": {},
                "connected": True,
                "nodes": None,
                "htr": {"addrs": [], "settings": {}},
            }
            cur = dict(current)
            cur[dev_id] = dev_map
            coordinator.data = cur  # type: ignore[attr-defined]
            current = cur

        if path.endswith("/mgr/nodes"):
            if isinstance(body, dict):
                dev_map["nodes"] = body
                addrs = extract_heater_addrs(body)
                htr_map: dict[str, Any] = dev_map.setdefault("htr", {})
                htr_map.setdefault("settings", {})
                htr_map["addrs"] = addrs
                updated_nodes = True

        elif "/htr/" in path and path.endswith("/settings"):
            addr = path.split("/htr/")[1].split("/")[0]
            htr_map = dev_map.setdefault("htr", {})
            settings_map: dict[str, Any] = htr_map.setdefault("settings", {})
            if isinstance(body, dict):
                settings_map[addr] = body
                updated_addrs.append(addr)

        elif "/htr/" in path and path.endswith("/advanced_setup"):
            addr = path.split("/htr/")[1].split("/")[0]
            htr_map = dev_map.setdefault("htr", {})
            adv_map: dict[str, Any] = htr_map.setdefault("advanced", {})
            if isinstance(body, dict):
                adv_map[addr] = body

        elif "/htr/" in path and path.endswith("/samples"):
            addr = path.split("/htr/")[1].split("/")[0]
            sample_addrs.append(addr)

        else:
            raw = dev_map.setdefault("raw", {})
            key = path.strip("/").replace("/", "_")
            raw[key] = body

    deduped_settings = list(dict.fromkeys(updated_addrs))
    deduped_samples = list(dict.fromkeys(sample_addrs))
    return paths, updated_nodes, deduped_settings, deduped_samples

