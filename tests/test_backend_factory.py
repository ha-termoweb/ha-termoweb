from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace
from typing import Any

import pytest

from custom_components.termoweb.backend import create_backend
from custom_components.termoweb.backend import termoweb as termoweb_backend
from custom_components.termoweb.backend.ducaheat import DucaheatBackend
from custom_components.termoweb.const import BRAND_DUCAHEAT


class DummyHttpClient:
    """Minimal HTTP client stub exposing a session attribute."""

    def __init__(self) -> None:
        self._session = SimpleNamespace()

    async def list_devices(self) -> list[dict[str, Any]]:
        return []

    async def get_nodes(self, dev_id: str) -> dict[str, Any]:
        return {"dev_id": dev_id}

    async def get_node_settings(
        self, dev_id: str, node: tuple[str, str | int]
    ) -> dict[str, Any]:
        node_type, addr = node
        return {"dev_id": dev_id, "node_type": node_type, "addr": addr}

    async def set_node_settings(
        self,
        dev_id: str,
        node: tuple[str, str | int],
        *,
        mode: str | None = None,
        stemp: float | None = None,
        prog: list[int] | None = None,
        ptemp: list[float] | None = None,
        units: str = "C",
        boost_time: int | None = None,
        cancel_boost: bool = False,
    ) -> dict[str, Any]:
        node_type, addr = node
        return {
            "dev_id": dev_id,
            "node_type": node_type,
            "addr": addr,
            "mode": mode,
            "stemp": stemp,
            "prog": prog,
            "ptemp": ptemp,
            "units": units,
            "boost_time": boost_time,
            "cancel_boost": cancel_boost,
        }

    async def get_node_samples(
        self,
        dev_id: str,
        node: tuple[str, str | int],
        start: float,
        stop: float,
    ) -> list[dict[str, str | int]]:
        return [
            {"t": int(start), "counter": "1"},
            {"t": int(stop), "counter": "2"},
        ]


def test_create_backend_returns_termoweb_backend() -> None:
    client = DummyHttpClient()
    backend = create_backend(brand="termoweb", client=client)
    assert isinstance(backend, termoweb_backend.TermoWebBackend)
    assert backend.brand == "termoweb"
    assert backend.client is client


def test_create_backend_returns_ducaheat_backend() -> None:
    client = DummyHttpClient()
    backend = create_backend(brand=BRAND_DUCAHEAT, client=client)
    assert isinstance(backend, DucaheatBackend)
    assert backend.brand == BRAND_DUCAHEAT
    assert backend.client is client


def test_termoweb_backend_creates_ws_client() -> None:
    client = DummyHttpClient()
    backend = termoweb_backend.TermoWebBackend(brand="termoweb", client=client)
    coordinator = object()
    loop = asyncio.new_event_loop()
    try:
        fake_hass = SimpleNamespace(loop=loop)
        inventory = object()
        ws_client = backend.create_ws_client(
            fake_hass,
            entry_id="entry123",
            dev_id="device456",
            coordinator=coordinator,
            inventory=inventory,
        )
    finally:
        loop.close()

    assert isinstance(ws_client, termoweb_backend.TermoWebWSClient)
    assert ws_client.dev_id == "device456"
    assert ws_client.entry_id == "entry123"
    assert ws_client._coordinator is coordinator
    assert ws_client._protocol_hint is None
    assert getattr(ws_client, "_inventory", None) is inventory


def test_termoweb_backend_sets_protocol_for_websocket_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TermoWebBackend should pass the socket.io protocol hint to WebSocketClient subclasses."""

    client = DummyHttpClient()
    backend = termoweb_backend.TermoWebBackend(brand="termoweb", client=client)

    class FakeWS(termoweb_backend.WebSocketClient):
        def __init__(self, hass: Any, **kwargs: Any) -> None:
            kwargs.setdefault("session", SimpleNamespace(closed=True))
            super().__init__(hass, **kwargs)

    monkeypatch.setattr(termoweb_backend, "TermoWebWSClient", FakeWS)

    loop = asyncio.new_event_loop()
    try:
        hass = SimpleNamespace(loop=loop)
        ws_client = backend.create_ws_client(
            hass,
            entry_id="entry",
            dev_id="dev",
            coordinator=object(),
        )
    finally:
        loop.close()

    assert isinstance(ws_client, FakeWS)
    assert ws_client._protocol_hint == "socketio09"


def test_termoweb_backend_resolves_direct_import() -> None:
    backend = termoweb_backend.TermoWebBackend(
        brand="termoweb", client=DummyHttpClient()
    )
    resolved = backend._resolve_ws_client_cls()
    assert resolved is termoweb_backend.TermoWebWSClient


def test_termoweb_backend_import_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    module = termoweb_backend
    ws_module = importlib.import_module(
        "custom_components.termoweb.backend.termoweb_ws"
    )

    with monkeypatch.context() as patch:
        patch.delattr(ws_module, "TermoWebWSClient")
        importlib.reload(module)
        backend = module.TermoWebBackend(brand="termoweb", client=DummyHttpClient())
        resolved = backend._resolve_ws_client_cls()
        assert resolved is module.WebSocketClient

    importlib.reload(module)
    backend = module.TermoWebBackend(brand="termoweb", client=DummyHttpClient())
    resolved = backend._resolve_ws_client_cls()
    assert resolved is module.TermoWebWSClient


def test_termoweb_backend_resolves_non_type(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = termoweb_backend.TermoWebBackend(
        brand="termoweb", client=DummyHttpClient()
    )
    monkeypatch.setattr(termoweb_backend, "TermoWebWSClient", object())
    resolved = backend._resolve_ws_client_cls()
    assert resolved is termoweb_backend.WebSocketClient
