from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from custom_components.termoweb.backend import create_backend
from custom_components.termoweb.backend.termoweb import TermoWebBackend
from custom_components.termoweb.ws_client_legacy import TermoWebWSLegacyClient


class DummyHttpClient:
    def __init__(self) -> None:
        self._session = SimpleNamespace()  # needed by WS client

    async def list_devices(self) -> list[dict[str, Any]]:
        return []

    async def get_nodes(self, dev_id: str) -> dict[str, Any]:
        return {"dev_id": dev_id}

    async def get_htr_settings(self, dev_id: str, addr: str | int) -> dict[str, Any]:
        return {"dev_id": dev_id, "addr": addr}

    async def set_htr_settings(
        self,
        dev_id: str,
        addr: str | int,
        *,
        mode: str | None = None,
        stemp: float | None = None,
        prog: list[int] | None = None,
        ptemp: list[float] | None = None,
        units: str = "C",
    ) -> dict[str, Any]:
        return {
            "dev_id": dev_id,
            "addr": addr,
            "mode": mode,
            "stemp": stemp,
            "prog": prog,
            "ptemp": ptemp,
            "units": units,
        }

    async def get_htr_samples(
        self,
        dev_id: str,
        addr: str | int,
        start: float,
        stop: float,
    ) -> list[dict[str, str | int]]:
        return [{"t": int(start), "counter": "1"}, {"t": int(stop), "counter": "2"}]


def test_create_backend_returns_termoweb_backend() -> None:
    client = DummyHttpClient()
    backend = create_backend(brand="termoweb", client=client)
    assert isinstance(backend, TermoWebBackend)
    assert backend.brand == "termoweb"
    assert backend.client is client


def test_termoweb_backend_creates_ws_client() -> None:
    client = DummyHttpClient()
    backend = TermoWebBackend(brand="termoweb", client=client)
    coordinator = object()
    loop = asyncio.new_event_loop()
    try:
        fake_hass = SimpleNamespace(loop=loop)
        ws_client = backend.create_ws_client(
            fake_hass,
            entry_id="entry123",
            dev_id="device456",
            coordinator=coordinator,
        )
    finally:
        loop.close()

    assert isinstance(ws_client, TermoWebWSLegacyClient)
    assert ws_client.dev_id == "device456"
    assert ws_client.entry_id == "entry123"
    assert ws_client._coordinator is coordinator


def test_termoweb_backend_fallback_ws_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    import custom_components.termoweb as init_module

    monkeypatch.setattr(init_module, "TermoWebWSLegacyClient", None)
    client = DummyHttpClient()
    backend = TermoWebBackend(brand="termoweb", client=client)
    loop = asyncio.new_event_loop()
    try:
        fake_hass = SimpleNamespace(loop=loop)
        ws_client = backend.create_ws_client(
            fake_hass,
            entry_id="entry456",
            dev_id="dev789",
            coordinator=object(),
        )
    finally:
        loop.close()

    assert isinstance(ws_client, TermoWebWSLegacyClient)
