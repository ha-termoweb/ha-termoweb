"""Listen-only radio entries: hear every frame, transmit nothing.

A listen-only ("monitor") entry shares the air with a real TermoWeb gateway,
so it must never transmit. Its client's link refuses every transmit command,
and its push client has none of the station duties (no clock syncs, power
grants, report confirmations or polling). It only counts the frames it hears
for the "frames heard" sensor and, on firmware that switches dialects at
runtime (ESP32), alternates the receive dialect so both dialects are heard.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
import logging
import time
from typing import Any

from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.util import dt as dt_util

from custom_components.termoweb.backend.base import (
    Backend,
    BackendCapabilities,
    WsClientProto,
)
from custom_components.termoweb.const import signal_radio_frames
from custom_components.termoweb.inventory import Inventory

from .radio.discovery import DISCOVERY_DIALECTS, LISTEN_WINDOW_S
from .radio.link import RadioLinkError, ReceivedFrame
from .radio_client import RadioClient
from .radio_ws import PAYLOAD_STALE_AFTER_S, RadioListener

_LOGGER = logging.getLogger(__name__)

MONITOR_WINDOW_S = LISTEN_WINDOW_S  # per-dialect window, as discovery listens


class RadioMonitor(RadioListener):
    """Push client of a listen-only entry: counts frames, never transmits."""

    def __init__(
        self,
        hass: Any,
        *,
        entry_id: str,
        dev_id: str,
        client: RadioClient,
        coordinator: Any,
        inventory: Inventory | None,
        window_s: float = MONITOR_WINDOW_S,
        **kwargs: Any,
    ) -> None:
        """Store collaborators; the reconnect loop runs one dialect window per pass."""

        super().__init__(
            hass,
            entry_id=entry_id,
            dev_id=dev_id,
            client=client,
            coordinator=coordinator,
            inventory=inventory,
            refresh_interval=window_s,
            **kwargs,
        )
        self.frames = 0
        self.last_frame_at: datetime | None = None
        self._window = 0

    async def _refresh_all(self) -> None:
        """Switch to the next dialect each window; dialect-A-only firmware stays put."""

        link = self._client.link
        info = None if link is None else link.gateway_info
        if link is None or info is None or info.dialect is None:
            return
        dialect = DISCOVERY_DIALECTS[self._window % len(DISCOVERY_DIALECTS)]
        self._window += 1
        try:
            await link.set_dialect(dialect)
        except RadioLinkError as err:
            _LOGGER.debug("Radio monitor could not switch dialect: %s", err)

    def _on_frame(self, received: ReceivedFrame) -> None:
        """Count a heard frame and publish the count; nothing is answered."""

        self.frames += 1
        self.last_frame_at = dt_util.utcnow()
        self._mark_ws_payload(timestamp=time.time(), stale_after=PAYLOAD_STALE_AFTER_S)
        if self._ws_health_tracker().status != "healthy":
            self._update_status("healthy")
        async_dispatcher_send(
            self.hass,
            signal_radio_frames(self.entry_id),
            {"frames": self.frames, "last_frame_at": self.last_frame_at.isoformat()},
        )


class RadioMonitorBackend(Backend):
    """Backend of a listen-only radio entry: no heaters, no station duties."""

    capabilities = BackendCapabilities(frame_monitor=True)

    def create_ws_client(
        self,
        hass: Any,
        entry_id: str,
        dev_id: str,
        coordinator: Any,
        *,
        inventory: Inventory | None = None,
    ) -> WsClientProto:
        """Return the frame-counting monitor in the websocket client's role."""

        client = self.client
        if not isinstance(client, RadioClient) or not client.listen_only:
            raise TypeError("RadioMonitorBackend requires a listen-only RadioClient")
        return RadioMonitor(
            hass,
            entry_id=entry_id,
            dev_id=dev_id,
            client=client,
            coordinator=coordinator,
            inventory=inventory,
        )

    async def fetch_hourly_samples(
        self,
        dev_id: str,
        nodes: Iterable[tuple[str, str]],
        start_local: datetime,
        end_local: datetime,
    ) -> dict[tuple[str, str], list[dict[str, Any]]]:
        """Return no samples: a listen-only entry has no heaters."""

        return {}


__all__ = ["MONITOR_WINDOW_S", "RadioMonitor", "RadioMonitorBackend"]
