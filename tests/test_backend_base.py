"""Tests for the abstract backend base helpers."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from types import SimpleNamespace
from typing import Any

import pytest

from custom_components.termoweb.backend.base import Backend


class DummyWsClient:
    """Simple websocket client stub that tracks lifecycle calls."""

    def __init__(self, hass: SimpleNamespace, *, entry_id: str, dev_id: str) -> None:
        self.hass = hass
        self.entry_id = entry_id
        self.dev_id = dev_id
        self._loop = getattr(hass, "loop", asyncio.get_event_loop())
        self._task: asyncio.Task | None = None
        self.started = False
        self.stopped = False

    async def _runner(self) -> None:
        self.started = True
        await asyncio.sleep(0)

    def start(self) -> asyncio.Task:
        if self._task is None:
            self._task = self._loop.create_task(self._runner())
        return self._task

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task
        self._task = None
        self.stopped = True


class ExampleBackend(Backend):
    """Concrete backend used to exercise the abstract base class."""

    def __init__(self, *, brand: str, client: Any) -> None:
        super().__init__(brand=brand, client=client)
        self.calls: list[dict[str, Any]] = []

    def create_ws_client(
        self,
        hass: SimpleNamespace,
        entry_id: str,
        dev_id: str,
        coordinator: Any,
    ) -> DummyWsClient:
        self.calls.append(
            {
                "hass": hass,
                "entry_id": entry_id,
                "dev_id": dev_id,
                "coordinator": coordinator,
            }
        )
        return DummyWsClient(hass, entry_id=entry_id, dev_id=dev_id)


@pytest.mark.asyncio
async def test_backend_properties_and_ws_creation() -> None:
    """The backend stores metadata and returns the websocket stub."""

    hass = SimpleNamespace(loop=asyncio.get_running_loop())
    client = object()
    coordinator = object()
    backend = ExampleBackend(brand="termoweb", client=client)

    assert backend.brand == "termoweb"
    assert backend.client is client

    ws_client = backend.create_ws_client(
        hass,
        entry_id="entry-1",
        dev_id="dev-1",
        coordinator=coordinator,
    )
    assert isinstance(ws_client, DummyWsClient)
    assert backend.calls == [
        {
            "hass": hass,
            "entry_id": "entry-1",
            "dev_id": "dev-1",
            "coordinator": coordinator,
        }
    ]

    task = ws_client.start()
    assert isinstance(task, asyncio.Task)
    assert not task.done()

    await ws_client.stop()
    assert ws_client.stopped is True
