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
from collections.abc import MutableMapping
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
from ..inventory import Inventory, NODE_CLASS_BY_TYPE, addresses_by_node_type
from ..nodes import ensure_node_inventory

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .ducaheat_ws import DucaheatWSClient
    from .termoweb_ws import TermoWebWSClient

_LOGGER = logging.getLogger(__name__)


def resolve_ws_update_section(section: str | None) -> tuple[str | None, str | None]:
    """Map a websocket path segment onto the node section bucket."""

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


def forward_ws_sample_updates(
    hass: HomeAssistant,
    entry_id: str,
    dev_id: str,
    updates: Mapping[str, Mapping[str, Any]],
    *,
    logger: logging.Logger | None = None,
    log_prefix: str = "WS",
) -> None:
    """Relay websocket heater sample updates to the energy coordinator."""

    record = hass.data.get(DOMAIN, {}).get(entry_id)
    if not isinstance(record, Mapping):
        return
    energy_coordinator = record.get("energy_coordinator")
    handler = getattr(energy_coordinator, "handle_ws_samples", None)
    if not callable(handler):
        return

    active_logger = logger or _LOGGER
    try:
        handler(
            dev_id,
            {node_type: dict(section) for node_type, section in updates.items()},
        )
    except Exception:  # pragma: no cover - defensive logging
        active_logger.debug(
            "%s: forwarding heater samples failed",
            log_prefix,
            exc_info=True,
        )

DUCAHEAT_NAMESPACE = "/api/v2/socket_io"


@dataclass
class WSStats:
    """Track websocket frame and event stats."""

    frames_total: int = 0
    events_total: int = 0
    last_event_ts: float = 0.0
    last_paths: list[str] | None = None


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


class _WSStatusMixin:
    """Provide shared websocket status helpers."""

    hass: HomeAssistant
    entry_id: str
    dev_id: str

    def _status_should_reset_health(self, status: str) -> bool:
        """Return True when a status should clear healthy tracking."""

        return False

    def _ws_state_bucket(self) -> dict[str, Any]:
        """Return the websocket state bucket for the current device."""

        ws_state = getattr(self, "_ws_state", None)
        if isinstance(ws_state, dict):
            return ws_state

        hass_data = getattr(self.hass, "data", None)
        if hass_data is None:
            hass_data = {}
            setattr(self.hass, "data", hass_data)  # type: ignore[attr-defined]

        domain_bucket = hass_data.setdefault(DOMAIN, {})
        entry_bucket = domain_bucket.setdefault(self.entry_id, {})
        ws_bucket = entry_bucket.setdefault("ws_state", {})
        ws_state = ws_bucket.setdefault(self.dev_id, {})
        setattr(self, "_ws_state", ws_state)
        return ws_state

    def _update_status(self, status: str) -> None:
        """Publish websocket status updates to Home Assistant listeners."""

        current = getattr(self, "_status", None)
        if status == current and status not in {"healthy", "connected"}:
            return

        now = time.time()
        if status == "healthy":
            healthy_since = getattr(self, "_healthy_since", None)
            if healthy_since is None:
                last_event = getattr(self, "_last_event_at", None)
                stats = getattr(self, "_stats", None)
                if last_event is None and stats is not None:
                    last_event = getattr(stats, "last_event_ts", None)
                setattr(self, "_healthy_since", last_event or now)
        elif self._status_should_reset_health(status):
            setattr(self, "_healthy_since", None)

        setattr(self, "_status", status)

        stats = getattr(self, "_stats", None)
        last_event_ts = getattr(stats, "last_event_ts", None) if stats else None
        last_event_at = getattr(self, "_last_event_at", None)
        healthy_since = getattr(self, "_healthy_since", None)

        state = self._ws_state_bucket()
        state["status"] = status
        state["last_event_at"] = last_event_ts or last_event_at or None
        state["healthy_since"] = healthy_since
        state["healthy_minutes"] = (
            int((now - healthy_since) / 60) if healthy_since else 0
        )
        state["frames_total"] = getattr(stats, "frames_total", 0) if stats else 0
        state["events_total"] = getattr(stats, "events_total", 0) if stats else 0

        dispatcher = getattr(self, "_dispatcher_mock", None)
        if dispatcher is None:
            dispatcher = getattr(self, "_dispatcher", None)
        if dispatcher is None:
            dispatcher = async_dispatcher_send

        dispatcher(
            self.hass,
            signal_ws_status(self.entry_id),
            {"dev_id": self.dev_id, "status": status},
        )


@dataclass
class NodeDispatchContext:
    """Container for shared node dispatch metadata."""

    payload: Any
    inventory: Inventory
    addr_map: dict[str, list[str]]
    unknown_types: set[str]
    record: MutableMapping[str, Any] | None
    snapshot: InstallationSnapshot | None


def _prepare_nodes_dispatch(
    hass: HomeAssistant,
    *,
    entry_id: str,
    coordinator: Any,
    raw_nodes: Any,
) -> NodeDispatchContext:
    """Normalise node payload data for downstream websocket dispatch."""

    record_container = hass.data.get(DOMAIN, {})
    record = record_container.get(entry_id) if isinstance(record_container, dict) else None
    record_mapping: MutableMapping[str, Any] | None = (
        record if isinstance(record, MutableMapping) else None
    )

    snapshot_obj = ensure_snapshot(record)

    if isinstance(snapshot_obj, InstallationSnapshot):
        dev_id = snapshot_obj.dev_id
    else:
        dev_id = str(getattr(coordinator, "_dev_id", ""))
        if not dev_id:
            dev_id = str(getattr(coordinator, "dev_id", ""))
        if not dev_id and isinstance(record_mapping, Mapping):
            dev_id = str(record_mapping.get("dev_id", ""))

    source_mapping: Mapping[str, Any]
    if isinstance(record_mapping, Mapping):
        source_mapping = record_mapping
    else:
        source_mapping = {}

    inventory_nodes = ensure_node_inventory(source_mapping, nodes=raw_nodes)
    inventory_container = Inventory(dev_id, raw_nodes, inventory_nodes)

    if isinstance(snapshot_obj, InstallationSnapshot):
        snapshot_obj.update_nodes(raw_nodes, inventory=inventory_container)
        if record_mapping is not None:
            record_mapping["node_inventory"] = list(snapshot_obj.inventory)
    elif record_mapping is not None:
        record_mapping["node_inventory"] = list(inventory_container.nodes)

    addr_map_raw, unknown_types = addresses_by_node_type(
        inventory_container.nodes, known_types=NODE_CLASS_BY_TYPE
    )
    addr_map = {node_type: list(addrs) for node_type, addrs in addr_map_raw.items()}

    if hasattr(coordinator, "update_nodes"):
        coordinator.update_nodes(raw_nodes, inventory_container)

    normalized_unknown: set[str] = {
        node_str
        for node_str in (
            str(node_type).strip()
            for node_type in unknown_types
            if node_type is not None
        )
        if node_str
    }

    return NodeDispatchContext(
        payload=raw_nodes,
        inventory=inventory_container,
        addr_map=addr_map,
        unknown_types=normalized_unknown,
        record=record_mapping,
        snapshot=snapshot_obj if isinstance(snapshot_obj, InstallationSnapshot) else None,
    )


class _WSCommon(_WSStatusMixin):
    """Shared helpers for websocket clients."""

    hass: HomeAssistant
    entry_id: str
    dev_id: str
    _coordinator: Any

    def _dispatch_nodes(self, payload: dict[str, Any]) -> None:
        raw_nodes = payload.get("nodes") if "nodes" in payload else payload
        context = _prepare_nodes_dispatch(
            self.hass,
            entry_id=self.entry_id,
            coordinator=self._coordinator,
            raw_nodes=raw_nodes,
        )
        nodes_payload = (
            context.inventory.payload if context.inventory is not None else context.payload
        )
        payload_copy = {
            "dev_id": self.dev_id,
            "node_type": None,
            "nodes": deepcopy(nodes_payload),
            "addr_map": {t: list(a) for t, a in context.addr_map.items()},
        }
        async_dispatcher_send(self.hass, signal_ws_data(self.entry_id), payload_copy)


class WebSocketClient(_WsLeaseMixin, _WSStatusMixin):
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
    "DUCAHEAT_NAMESPACE",
    "HandshakeError",
    "WSStats",
    "forward_ws_sample_updates",
    "resolve_ws_update_section",
    "WebSocketClient",
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
