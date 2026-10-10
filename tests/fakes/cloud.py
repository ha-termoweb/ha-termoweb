"""The TermoWeb cloud at the backend boundary: canned account data and fakes."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

VERSION = json.loads(
    (Path(__file__).parents[2] / "custom_components/termoweb/manifest.json").read_text()
)["version"]

# Synthetic identifiers only.
DEV_ID = "0123456789abcdef"
USERNAME = "user@example.com"
PASSWORD = "secret"

DEVICES = [{"dev_id": DEV_ID, "name": "Home", "serial_id": "SN-TEST"}]
NODES = {"nodes": [{"type": "htr", "addr": 1, "name": "Living room"}]}
HTR_SETTINGS = {
    "mode": "auto",
    "state": "off",
    "stemp": "21.0",
    "mtemp": "20.5",
    "units": "C",
    "prog": [0] * 168,
    "ptemp": ["7.0", "17.0", "21.0"],
}


class FakeWSClient:
    """Websocket client stand-in: a task that idles until stopped."""

    def __init__(self, hass: Any) -> None:
        """Remember hass so the idle task is tracked by it."""
        self.hass = hass
        self.task: asyncio.Task[None] | None = None
        self.stop_calls = 0

    def start(self) -> asyncio.Task[None]:
        """Start the idle task, mirroring the real client's contract."""
        self.task = self.hass.async_create_background_task(
            asyncio.Event().wait(), "termoweb-fake-ws"
        )
        return self.task

    async def stop(self) -> None:
        """Record the stop request and cancel the idle task."""
        self.stop_calls += 1
        if self.task is not None:
            self.task.cancel()


class FakeCloud:
    """Handles to the patched backend-boundary methods used by the tests."""

    def __init__(self) -> None:
        """Create async mocks with a healthy single-heater account."""
        self.list_devices = AsyncMock(return_value=DEVICES)
        self.get_nodes = AsyncMock(return_value=NODES)
        self.get_geo_data = AsyncMock(return_value=None)
        self.get_node_settings = AsyncMock(return_value=HTR_SETTINGS)
        self.get_node_samples = AsyncMock(return_value=[])
        self.get_rtc_time = AsyncMock(return_value={})
        self.get_power_limit = AsyncMock(return_value=None)
        self.ws_clients: list[FakeWSClient] = []

    def create_ws_client(self, hass: Any, *args: Any, **kwargs: Any) -> FakeWSClient:
        """Return a fake websocket client and keep it for assertions."""
        client = FakeWSClient(hass)
        self.ws_clients.append(client)
        return client
