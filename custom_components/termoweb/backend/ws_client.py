# -*- coding: utf-8 -*-
"""Shared websocket helpers."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass
import gzip
import json
import logging
import random
import string
import time
from typing import TYPE_CHECKING, Any, Mapping
from urllib.parse import urlencode, urlsplit, urlunsplit

import aiohttp
import socketio
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_send

from ..api import RESTClient
from ..const import (
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
from ..installation import InstallationSnapshot, ensure_snapshot
from ..nodes import (
    NODE_CLASS_BY_TYPE,
    addresses_by_node_type,
    collect_heater_sample_addresses,
    ensure_node_inventory,
    heater_sample_subscription_targets,
    normalize_heater_addresses,
    normalize_node_addr,
    normalize_node_type,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .ducaheat_ws import DucaheatWSClient
    from .termoweb_ws import TermoWebWSClient

_LOGGER = logging.getLogger(__name__)

DUCAHEAT_NAMESPACE = "/api/v2/socket_io"


@dataclass
class WSStats:
    """Track websocket frame and event stats."""

    frames_total: int = 0
    events_total: int = 0
    last_event_ts: float = 0.0


class HandshakeError(RuntimeError):
    """Raised when a websocket handshake fails."""

    def __init__(self, status: int, url: str, detail: str) -> None:
        super().__init__(f"handshake failed: status={status}, detail={detail}")
        self.status = status
        self.url = url


class _WsLeaseMixin:
    """Provide reconnect backoff management for websocket clients."""

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


class _WSCommon:
    """Shared helpers for websocket clients."""

    hass: HomeAssistant
    entry_id: str
    dev_id: str
    _coordinator: Any

    def _ws_state_bucket(self) -> dict[str, Any]:
        domain_bucket = self.hass.data.setdefault(DOMAIN, {})
        entry_bucket = domain_bucket.setdefault(self.entry_id, {})
        ws_state = entry_bucket.setdefault("ws_state", {})
        return ws_state.setdefault(self.dev_id, {})

    def _update_status(self, status: str) -> None:
        async_dispatcher_send(
            self.hass,
            signal_ws_status(self.entry_id),
            {"dev_id": self.dev_id, "status": status},
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


class WebSocketClient(_WsLeaseMixin):
    """Delegate to the correct backend websocket client."""

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
        self._delegate: DucaheatWSClient | TermoWebWSClient | None = None
        self._brand = (
            BRAND_DUCAHEAT
            if getattr(api_client, "_is_ducaheat", False)
            else BRAND_TERMOWEB
        )

    def start(self) -> asyncio.Task:
        if self._delegate is not None:
            return self._delegate.start()
        if self._brand == BRAND_DUCAHEAT:
            from .ducaheat_ws import DucaheatWSClient

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
            from .termoweb_ws import TermoWebWSClient

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


__all__ = [
    "ACCEPT_LANGUAGE",
    "API_BASE",
    "BRAND_DUCAHEAT",
    "BRAND_TERMOWEB",
    "DOMAIN",
    "DUCAHEAT_NAMESPACE",
    "DucaheatWSClient",
    "HandshakeError",
    "HomeAssistant",
    "RESTClient",
    "USER_AGENT",
    "WS_NAMESPACE",
    "WSStats",
    "WebSocketClient",
    "_WSCommon",
    "_WsLeaseMixin",
    "addresses_by_node_type",
    "aiohttp",
    "async_dispatcher_send",
    "asyncio",
    "collect_heater_sample_addresses",
    "ensure_node_inventory",
    "get_brand_api_base",
    "get_brand_requested_with",
    "get_brand_user_agent",
    "gzip",
    "heater_sample_subscription_targets",
    "json",
    "logging",
    "normalize_heater_addresses",
    "normalize_node_addr",
    "normalize_node_type",
    "random",
    "signal_ws_data",
    "signal_ws_status",
    "socketio",
    "string",
    "time",
    "time_mod",
    "TermoWebWSClient",
    "urlencode",
    "urlsplit",
    "urlunsplit",
]

time_mod = time.monotonic


def __getattr__(name: str) -> Any:
    """Lazily expose backend websocket client implementations."""

    if name == "DucaheatWSClient":
        from .ducaheat_ws import DucaheatWSClient as _DucaheatWSClient

        return _DucaheatWSClient
    if name == "TermoWebWSClient":
        from .termoweb_ws import TermoWebWSClient as _TermoWebWSClient

        return _TermoWebWSClient
    raise AttributeError(name)
