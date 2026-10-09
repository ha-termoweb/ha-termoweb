"""Push client for the radio backend: answers heater traffic and keeps state fresh.

``RadioListener`` plays the part the cloud websocket clients play. It owns the
station duties a real TermoWeb gateway performs (clock sync on registration,
power grants, report confirmations, keepalive clock syncs) and turns heater
traffic and periodic status reads into ``NodeSettingsDelta`` updates.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine, Mapping
import contextlib
import logging
import time
from typing import Any

from homeassistant.core import HomeAssistant

from custom_components.termoweb.codecs.radio_codec import (
    prog_from_program,
    settings_from_power_request,
    settings_from_status,
)
from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.domain.ids import NodeId, NodeType
from custom_components.termoweb.domain.state import NodeSettingsDelta
from custom_components.termoweb.inventory import Inventory

from .radio import protocol
from .radio.link import RadioLinkError, ReceivedFrame
from .radio_client import RadioClient, RadioCommandError, radio_addr
from .ws_client import _WSStatusMixin

_LOGGER = logging.getLogger(__name__)

REFRESH_INTERVAL_S = 120.0  # keepalive clock sync + status read; heaters need <=150 s
PAYLOAD_STALE_AFTER_S = 3 * REFRESH_INTERVAL_S
RECONNECT_BACKOFF_S = (5.0, 10.0, 30.0, 120.0, 300.0)

Sleep = Callable[[float], Awaitable[Any]]


class RadioListener(_WSStatusMixin):
    """Station duties and push updates for heaters behind the radio gateway."""

    def __init__(
        self,
        hass: HomeAssistant,
        *,
        entry_id: str,
        dev_id: str,
        client: RadioClient,
        coordinator: Any,
        inventory: Inventory | None,
        refresh_interval: float = REFRESH_INTERVAL_S,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        """Store collaborators; nothing runs until start()."""

        if not isinstance(inventory, Inventory):
            raise TypeError("RadioListener requires the immutable Inventory")
        self.hass = hass
        self.entry_id = entry_id
        self.dev_id = dev_id
        self._client = client
        self._coordinator = coordinator
        self._inventory = inventory
        self._refresh_interval = refresh_interval
        self._sleep = sleep
        self._task: asyncio.Task[None] | None = None
        self._jobs: set[asyncio.Task[Any]] = set()
        self._remove_frame_listener: Callable[[], None] | None = None
        self._remove_disconnect_listener: Callable[[], None] | None = None
        self._backoff_idx = 0

    # --- lifecycle -----------------------------------------------------------

    def start(self) -> asyncio.Task[None]:
        """Start the background task (idempotent) and return it."""

        if self._task is not None and not self._task.done():
            return self._task
        _LOGGER.info("Radio listener starting for %s", self.dev_id)
        self._task = asyncio.get_running_loop().create_task(
            self._run(), name=f"{DOMAIN}-radio-{self.dev_id}"
        )
        return self._task

    async def stop(self) -> None:
        """Cancel the task and pending replies, then close the gateway link."""

        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        for job in list(self._jobs):
            job.cancel()
        if self._jobs:
            await asyncio.gather(*self._jobs, return_exceptions=True)
        self._detach()
        await self._client.async_close()
        self._update_status("stopped")
        _LOGGER.info("Radio listener stopped for %s", self.dev_id)

    def is_running(self) -> bool:
        """Return True while the background task runs."""

        return self._task is not None and not self._task.done()

    def _status_should_reset_health(self, status: str) -> bool:
        """Clear healthy tracking whenever the gateway is not connected."""

        return status in {"disconnected", "stopped"}

    # --- main loop -----------------------------------------------------------

    async def _run(self) -> None:
        """Connect, serve the gateway until it drops, back off and reconnect."""

        while True:
            try:
                link = await self._client.async_connect()
            except RadioLinkError as err:
                _LOGGER.info("Radio gateway connect failed: %s", err)
            else:
                self._backoff_idx = 0
                self._remove_frame_listener = link.add_listener(self._on_frame)
                self._remove_disconnect_listener = self._client.add_disconnect_listener(
                    self._on_disconnect
                )
                self._update_status("connected")
                try:
                    while self._client.connected:
                        await self._refresh_all()
                        await self._sleep(self._refresh_interval)
                finally:
                    self._detach()
            self._update_status("disconnected")
            await self._sleep(self._next_backoff())

    def _next_backoff(self) -> float:
        """Return the next reconnect delay."""

        delay = RECONNECT_BACKOFF_S[
            min(self._backoff_idx, len(RECONNECT_BACKOFF_S) - 1)
        ]
        self._backoff_idx += 1
        return delay

    def _detach(self) -> None:
        """Remove the frame and disconnect listeners from the link."""

        for remover in (self._remove_frame_listener, self._remove_disconnect_listener):
            if remover is not None:
                remover()
        self._remove_frame_listener = None
        self._remove_disconnect_listener = None

    def _on_disconnect(self) -> None:
        """Report the dropped gateway connection at once."""

        self._update_status("disconnected")

    def _heaters(self) -> list[tuple[str, str]]:
        """Return ``(node_type, addr)`` for every heater-like inventory node."""

        forward, _reverse = self._inventory.heater_address_map
        return [
            (node_type, addr) for node_type, addrs in forward.items() for addr in addrs
        ]

    def _node_type_for(self, addr: str) -> str | None:
        """Return the inventory node type for heater address ``addr``."""

        _forward, reverse = self._inventory.heater_address_map
        types = reverse.get(addr)
        if not types:
            return None
        return "htr" if "htr" in types else min(types)

    async def _refresh_all(self) -> None:
        """Send each heater a keepalive clock sync and read its status."""

        for node_type, addr in self._heaters():
            try:
                radio_id = radio_addr(addr)
            except ValueError as err:
                _LOGGER.error("Skipping heater with a bad radio address: %s", err)
                continue
            try:
                await self._client.async_sync_clock(radio_id, registering=False)
            except (RadioLinkError, RadioCommandError) as err:
                _LOGGER.debug("Keepalive clock sync to heater %s failed: %s", addr, err)
            else:
                self._mark_ws_heartbeat(reason="keepalive")
            settings = await self._client.get_node_settings(
                self.dev_id, (node_type, addr)
            )
            if settings:
                self._push(node_type, addr, settings)

    # --- unsolicited heater frames ------------------------------------------

    def _on_frame(self, received: ReceivedFrame) -> None:
        """Dispatch a frame a heater sent to this station."""

        frame = received.frame
        link = self._client.link
        if (
            link is None
            or not frame.ok
            or frame.is_ack
            or frame.dst != link.station_id
            or frame.src is None
        ):
            return
        addr = str(frame.src)
        node_type = self._node_type_for(addr)
        if node_type is None:
            _LOGGER.debug("Ignoring radio frame from unknown node %s", addr)
            return
        kind = protocol.classify_unsolicited(frame)
        payload = frame.payload
        if kind is protocol.Unsolicited.REGISTRATION:
            _LOGGER.info("Heater %s registered; sending clock sync", addr)
            self._spawn(self._client.async_sync_clock(frame.src, registering=True))
        elif kind is protocol.Unsolicited.POWER_REQUEST:
            self._spawn(self._client.async_send(frame.src, protocol.power_verdict()))
            request = protocol.decode_power_request(payload)
            if request is not None:
                self._client.note_max_power(frame.src, request.measured_power_w)
                self._push(node_type, addr, settings_from_power_request(request))
        elif kind is protocol.Unsolicited.REPORT:
            self._spawn(self._client.async_send(frame.src, protocol.confirm_report()))
            record = protocol.decode_status(payload)
            if record is not None:
                self._push(node_type, addr, settings_from_status(record))
        elif kind is protocol.Unsolicited.PROGRAM_REPORT:
            self._spawn(self._client.async_send(frame.src, protocol.confirm_report()))
            program = protocol.decode_program(payload)
            prog = None if program is None else prog_from_program(program)
            if prog is not None:
                self._push(node_type, addr, {"prog": prog})

    def _spawn(self, coro: Coroutine[Any, Any, Any]) -> None:
        """Run a station reply in the background, logging its failure."""

        task = asyncio.get_running_loop().create_task(self._reply(coro))
        self._jobs.add(task)
        task.add_done_callback(self._jobs.discard)

    @staticmethod
    async def _reply(coro: Coroutine[Any, Any, Any]) -> None:
        """Await a station reply; a heater that misses it simply repeats."""

        try:
            await coro
        except (RadioLinkError, RadioCommandError) as err:
            _LOGGER.debug("Radio station reply failed: %s", err)

    def _push(self, node_type: str, addr: str, settings: Mapping[str, Any]) -> None:
        """Send one settings delta to the coordinator and mark the payload fresh."""

        delta = NodeSettingsDelta(NodeId(NodeType(node_type), addr), dict(settings))
        self._coordinator.handle_ws_deltas(self.dev_id, [delta])
        self._mark_ws_payload(timestamp=time.time(), stale_after=PAYLOAD_STALE_AFTER_S)
        if self._ws_health_tracker().status != "healthy":
            self._update_status("healthy")


__all__ = ["RadioListener"]
