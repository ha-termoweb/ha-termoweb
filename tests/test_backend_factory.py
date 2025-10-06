from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from types import ModuleType

import pytest

from custom_components.termoweb.backend import create_backend
from custom_components.termoweb.backend import termoweb as termoweb_backend
from custom_components.termoweb.backend.ducaheat import DucaheatBackend
from custom_components.termoweb.backend.termoweb import TermoWebBackend
from custom_components.termoweb.const import BRAND_DUCAHEAT
from custom_components.termoweb.backend.termoweb_ws import TermoWebWSClient
from custom_components.termoweb.backend.ws_client import WebSocketClient as LegacyWSClient


class DummyHttpClient:
    def __init__(self) -> None:
        self._session = SimpleNamespace()  # needed by WS client

    async def list_devices(self) -> list[dict[str, Any]]:
        return []

    async def get_nodes(self, dev_id: str) -> dict[str, Any]:
        return {"dev_id": dev_id}

    async def get_node_settings(
        self, dev_id: str, node: tuple[str, str | int]
    ) -> dict[str, Any]:
        node_type, addr = node
        return {"dev_id": dev_id, "node_type": node_type, "addr": addr}

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
    assert isinstance(backend, TermoWebBackend)
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

    assert isinstance(ws_client, TermoWebWSClient)
    assert ws_client.dev_id == "device456"
    assert ws_client.entry_id == "entry123"
    assert ws_client._coordinator is coordinator
    assert ws_client._protocol_hint is None


def test_termoweb_backend_sets_protocol_for_websocket_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """TermoWebBackend should pass the socket.io protocol hint to WebSocketClient subclasses."""

    from custom_components.termoweb.backend.ws_client import WebSocketClient

    client = DummyHttpClient()
    backend = TermoWebBackend(brand="termoweb", client=client)

    class FakeWS(WebSocketClient):
        def __init__(self, hass: Any, **kwargs: Any) -> None:
            kwargs.setdefault("session", SimpleNamespace(closed=True))
            super().__init__(hass, **kwargs)

    monkeypatch.setitem(
        termoweb_backend.sys.modules,
        "custom_components.termoweb",
        SimpleNamespace(TermoWebWSClient=FakeWS),
    )
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


def test_termoweb_backend_fallback_ws_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    import custom_components.termoweb as init_module

    monkeypatch.setattr(init_module, "TermoWebWSClient", None)
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

    assert isinstance(ws_client, TermoWebWSClient)
    assert ws_client._protocol_hint is None


def test_termoweb_backend_resolve_ws_client_cls_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = TermoWebBackend(brand="termoweb", client=DummyHttpClient())
    monkeypatch.setitem(
        termoweb_backend.sys.modules,
        "custom_components.termoweb",
        SimpleNamespace(TermoWebWSClient=None),
    )
    monkeypatch.setitem(
        termoweb_backend.sys.modules,
        "custom_components.termoweb.__init__",
        SimpleNamespace(TermoWebWSClient=None),
    )

    resolved = backend._resolve_ws_client_cls()
    assert resolved is LegacyWSClient


def test_termoweb_backend_resolve_ws_no_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = TermoWebBackend(brand="termoweb", client=DummyHttpClient())
    monkeypatch.setitem(termoweb_backend.sys.modules, "custom_components.termoweb", None)
    monkeypatch.setitem(
        termoweb_backend.sys.modules,
        "custom_components.termoweb.__init__",
        None,
    )
    resolved = backend._resolve_ws_client_cls()
    assert resolved is LegacyWSClient


def test_termoweb_backend_resolve_ws_no_module_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = TermoWebBackend(brand="termoweb", client=DummyHttpClient())
    marker = object()
    monkeypatch.setitem(
        termoweb_backend.sys.modules,
        "custom_components.termoweb",
        marker,
    )
    monkeypatch.setitem(
        termoweb_backend.sys.modules,
        "custom_components.termoweb.__init__",
        marker,
    )
    resolved = backend._resolve_ws_client_cls()
    assert resolved is LegacyWSClient


def test_termoweb_backend_resolve_ws_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = TermoWebBackend(brand="termoweb", client=DummyHttpClient())
    module = ModuleType("custom_components.termoweb")
    module.TermoWebWSClient = None  # type: ignore[attr-defined]
    init_module = ModuleType("custom_components.termoweb.__init__")
    init_module.TermoWebWSClient = None  # type: ignore[attr-defined]
    monkeypatch.setitem(termoweb_backend.sys.modules, module.__name__, module)
    monkeypatch.setitem(termoweb_backend.sys.modules, init_module.__name__, init_module)
    def _raise(_: str) -> None:
        raise ImportError

    monkeypatch.setattr(termoweb_backend, "import_module", _raise)
    resolved = backend._resolve_ws_client_cls()
    assert resolved is LegacyWSClient


def test_termoweb_backend_resolve_ws_missing_attr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = TermoWebBackend(brand="termoweb", client=DummyHttpClient())
    module = ModuleType("custom_components.termoweb")
    module.TermoWebWSClient = None  # type: ignore[attr-defined]
    init_module = ModuleType("custom_components.termoweb.__init__")
    init_module.TermoWebWSClient = None  # type: ignore[attr-defined]
    monkeypatch.setitem(termoweb_backend.sys.modules, module.__name__, module)
    monkeypatch.setitem(termoweb_backend.sys.modules, init_module.__name__, init_module)

    ws_module = ModuleType("custom_components.termoweb.backend.termoweb_ws")
    monkeypatch.setattr(termoweb_backend, "import_module", lambda name: ws_module)
    resolved = backend._resolve_ws_client_cls()
    assert resolved is LegacyWSClient


def test_termoweb_backend_legacy_ws_class(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = TermoWebBackend(brand="termoweb", client=DummyHttpClient())

    class LegacyWS:
        def __init__(
            self,
            hass,
            *,
            entry_id: str,
            dev_id: str,
            api_client: Any,
            coordinator: Any,
            **kwargs: Any,
        ) -> None:
            self.hass = hass
            self.entry_id = entry_id
            self.dev_id = dev_id
            self.api_client = api_client
            self.coordinator = coordinator
            self.kwargs = kwargs

    monkeypatch.setitem(
        termoweb_backend.sys.modules,
        "custom_components.termoweb",
        SimpleNamespace(TermoWebWSClient=LegacyWS),
    )

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

    assert isinstance(ws_client, LegacyWS)
    assert "protocol" not in ws_client.kwargs
