from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from conftest import _install_stubs

_install_stubs()

from custom_components.termoweb.backend import (  # noqa: E402
    Backend,
    DucaheatBackend,
    TermoWebBackend,
    create_backend,
)
from custom_components.termoweb.const import BRAND_DUCAHEAT, WS_NAMESPACE  # noqa: E402
from custom_components.termoweb.backend.ducaheat_ws import DucaheatWSClient  # noqa: E402
from custom_components.termoweb.backend.termoweb_ws import TermoWebWSClient  # noqa: E402


class DummyHttpClient:
    """Minimal HTTP client stub exposing a session attribute."""

    def __init__(self) -> None:
        self._session = SimpleNamespace()


def _make_hass(loop: asyncio.AbstractEventLoop) -> SimpleNamespace:
    """Return a fake Home Assistant object with the required loop."""

    return SimpleNamespace(loop=loop)


def test_backend_factory_returns_expected_clients() -> None:
    """Backends created via the factory expose the correct websocket clients."""

    client = DummyHttpClient()

    termoweb_backend = create_backend(brand="termoweb", client=client)
    assert isinstance(termoweb_backend, TermoWebBackend)

    loop = asyncio.new_event_loop()
    try:
        hass = _make_hass(loop)
        ws_client = termoweb_backend.create_ws_client(
            hass,
            entry_id="entry",
            dev_id="dev",
            coordinator=object(),
        )
        assert isinstance(ws_client, TermoWebWSClient)
        assert ws_client._protocol_hint is None
        loop.run_until_complete(ws_client.stop())
    finally:
        loop.close()

    ducaheat_backend = create_backend(brand=BRAND_DUCAHEAT, client=client)
    assert isinstance(ducaheat_backend, DucaheatBackend)

    loop = asyncio.new_event_loop()
    try:
        hass = _make_hass(loop)
        ws_client = ducaheat_backend.create_ws_client(
            hass,
            entry_id="entry",
            dev_id="dev",
            coordinator=object(),
        )
        assert isinstance(ws_client, DucaheatWSClient)
        assert ws_client._namespace == WS_NAMESPACE
        loop.run_until_complete(ws_client.stop())
    finally:
        loop.close()

    default_backend = create_backend(brand="unknown", client=client)
    assert isinstance(default_backend, TermoWebBackend)


def test_backend_requires_create_override() -> None:
    """Attempting to instantiate a backend without a websocket factory fails."""

    class InvalidBackend(Backend):
        pass

    client = DummyHttpClient()
    with pytest.raises(TypeError):
        InvalidBackend(brand="termoweb", client=client)
