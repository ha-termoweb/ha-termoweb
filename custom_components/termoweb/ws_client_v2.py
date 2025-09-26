"""Ducaheat WebSocket client skeleton used for testing infrastructure."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import logging
import time
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_send

from .api import RESTClient
from .const import API_BASE, DOMAIN, signal_ws_data, signal_ws_status
from .nodes import build_node_inventory
from .utils import HEATER_NODE_TYPES, addresses_by_type

_LOGGER = logging.getLogger(__name__)


class DucaheatWSClient:
    """Minimal Socket.IO v2 client skeleton for Ducaheat cloud."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry_id: str,
        dev_id: str,
        api_client: RESTClient,
        coordinator: Any,
    ) -> None:
        """Initialize the client."""
        self.hass = hass
        self.entry_id = entry_id
        self.dev_id = dev_id
        self._client = api_client
        self._coordinator = coordinator
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._closing = False

        self._status: str = "stopped"
        self._last_event_at: float | None = None
        self._healthy_since: float | None = None
        self._handshake: dict[str, Any] | None = None
        self._nodes: dict[str, Any] = {}

        domain_bucket = self.hass.data.setdefault(DOMAIN, {})
        entry_bucket = domain_bucket.setdefault(self.entry_id, {})
        entry_bucket.setdefault("ws_state", {})

    def start(self) -> asyncio.Task:
        """Start the background runner."""
        if self._task and not self._task.done():
            return self._task
        _LOGGER.info("WS %s: starting Ducaheat client", self.dev_id)
        self._closing = False
        self._stop_event = asyncio.Event()
        self._task = self.hass.loop.create_task(
            self._runner(), name=f"{DOMAIN}-ws-v2-{self.dev_id}"
        )
        return self._task

    async def stop(self) -> None:
        """Stop the runner."""
        if self._task is None:
            self._update_status("stopped")
            return
        _LOGGER.info("WS %s: stopping Ducaheat client", self.dev_id)
        self._closing = True
        self._stop_event.set()
        task = self._task
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None

    async def ws_url(self) -> str:
        """Return the websocket URL using the API client's token helper."""
        headers = await self._client._authed_headers()  # noqa: SLF001
        token: str | None = None
        auth_header = (
            headers.get("Authorization") if isinstance(headers, dict) else None
        )
        if isinstance(auth_header, str):
            parts = auth_header.split(" ", 1)
            token = parts[1] if len(parts) == 2 else parts[0]
        if not token:
            raise RuntimeError("missing access token for WS connection")
        base = getattr(self._client, "api_base", None)
        api_base = base.rstrip("/") if isinstance(base, str) and base else API_BASE
        return f"{api_base}/api/v2/socket_io?token={token}"

    async def _runner(self) -> None:
        """Background task waiting for stop events."""
        self._healthy_since = None
        self._last_event_at = None
        self._nodes = {}
        self._update_status("connecting")
        try:
            await self._stop_event.wait()
        except asyncio.CancelledError:
            pass
        finally:
            self._update_status("stopped")

    def _on_frame(self, payload: str) -> None:
        """Process an incoming Socket.IO frame."""
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
        if isinstance(data, dict):
            self._handshake = deepcopy(data)
            self._update_status("connected")
        else:
            _LOGGER.debug("WS %s: invalid handshake payload", self.dev_id)

    def _handle_dev_data(self, data: Any) -> None:
        nodes = self._extract_nodes(data)
        if nodes is None:
            _LOGGER.debug("WS %s: dev_data without nodes", self.dev_id)
            return
        self._nodes = deepcopy(nodes)
        self._dispatch_nodes(self._nodes)
        self._mark_event()

    def _handle_update(self, data: Any) -> None:
        nodes = self._extract_nodes(data)
        if nodes is None:
            _LOGGER.debug("WS %s: update without nodes", self.dev_id)
            return
        if not self._nodes:
            self._nodes = deepcopy(nodes)
        else:
            self._merge_nodes(self._nodes, nodes)
        self._dispatch_nodes(self._nodes)
        self._mark_event()

    def _extract_nodes(self, data: Any) -> dict[str, Any] | None:
        if not isinstance(data, dict):
            return None
        nodes = data.get("nodes")
        if isinstance(nodes, dict):
            return nodes
        return None

    def _dispatch_nodes(self, nodes: dict[str, Any]) -> None:
        record = self.hass.data.get(DOMAIN, {}).get(self.entry_id)
        brand = ""
        if isinstance(record, dict):
            brand = str(record.get("brand") or "").strip()

        inventory: list[Any] = []
        if brand:
            try:
                inventory = build_node_inventory(nodes, brand)
            except ValueError as err:  # pragma: no cover - defensive
                _LOGGER.debug(
                    "WS %s: failed to build node inventory: %s",
                    self.dev_id,
                    err,
                    exc_info=err,
                )

        if hasattr(self._coordinator, "update_nodes"):
            self._coordinator.update_nodes(nodes, inventory)

        if isinstance(record, dict):
            record["nodes"] = nodes
            record["node_inventory"] = inventory
            record["htr_addrs"] = addresses_by_type(inventory, HEATER_NODE_TYPES)

        payload = {"dev_id": self.dev_id, "nodes": deepcopy(nodes)}

        def _send() -> None:
            async_dispatcher_send(
                self.hass,
                signal_ws_data(self.entry_id),
                payload,
            )

        self.hass.loop.call_soon_threadsafe(_send)

    def _mark_event(self) -> None:
        now = time.time()
        self._last_event_at = now
        if self._healthy_since is None:
            self._healthy_since = now
        self._update_status("healthy")

    @staticmethod
    def _merge_nodes(target: dict[str, Any], source: dict[str, Any]) -> None:
        for key, value in source.items():
            if isinstance(value, dict):
                existing = target.get(key)
                if isinstance(existing, dict):
                    DucaheatWSClient._merge_nodes(existing, value)
                else:
                    target[key] = deepcopy(value)
            else:
                target[key] = value

    def _update_status(self, status: str) -> None:
        if status == self._status and status != "healthy":
            return
        self._status = status
        now = time.time()
        domain_bucket: dict[str, Any] = self.hass.data.setdefault(DOMAIN, {})
        entry_bucket: dict[str, Any] = domain_bucket.setdefault(self.entry_id, {})
        ws_bucket: dict[str, dict[str, Any]] = entry_bucket.setdefault("ws_state", {})
        dev_bucket: dict[str, Any] = ws_bucket.setdefault(self.dev_id, {})
        dev_bucket["status"] = status
        dev_bucket["last_event_at"] = self._last_event_at
        dev_bucket["healthy_since"] = self._healthy_since
        dev_bucket["healthy_minutes"] = (
            int((now - self._healthy_since) / 60) if self._healthy_since else 0
        )

        status_payload = {"dev_id": self.dev_id, "status": status}

        def _send() -> None:
            async_dispatcher_send(
                self.hass,
                signal_ws_status(self.entry_id),
                status_payload,
            )

        self.hass.loop.call_soon_threadsafe(_send)
