from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

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
from homeassistant.util import dt as dt_util


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
        inventory = object()
        ws_client = termoweb_backend.create_ws_client(
            hass,
            entry_id="entry",
            dev_id="dev",
            coordinator=object(),
            inventory=inventory,
        )
        assert isinstance(ws_client, TermoWebWSClient)
        assert ws_client._protocol_hint is None
        assert getattr(ws_client, "_inventory", None) is inventory
        loop.run_until_complete(ws_client.stop())
    finally:
        loop.close()

    ducaheat_backend = create_backend(brand=BRAND_DUCAHEAT, client=client)
    assert isinstance(ducaheat_backend, DucaheatBackend)

    loop = asyncio.new_event_loop()
    try:
        hass = _make_hass(loop)
        inventory = object()
        ws_client = ducaheat_backend.create_ws_client(
            hass,
            entry_id="entry",
            dev_id="dev",
            coordinator=object(),
            inventory=inventory,
        )
        assert isinstance(ws_client, DucaheatWSClient)
        assert ws_client._namespace == WS_NAMESPACE
        assert getattr(ws_client, "_inventory", None) is inventory
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


def test_backend_module_exports_expected_classes() -> None:
    """The backend module exposes the concrete backend implementations."""

    backend_module = __import__(
        Backend.__module__.rsplit(".", 1)[0],
        fromlist=["DucaheatBackend", "TermoWebBackend"],
    )

    assert getattr(backend_module, "DucaheatBackend") is DucaheatBackend
    assert getattr(backend_module, "TermoWebBackend") is TermoWebBackend

    with pytest.raises(AttributeError):
        getattr(backend_module, "MissingThing")


@pytest.mark.asyncio
async def test_termoweb_backend_fetch_hourly_samples_normalises() -> None:
    """TermoWeb hourly fetch returns UTC timestamps and energy in Wh."""

    client = SimpleNamespace()
    client.get_node_samples = AsyncMock(
        return_value=[{"t": 1_700_000_000, "counter": 1_800.0, "power": 600.0}]
    )
    backend = TermoWebBackend(brand="termoweb", client=client)
    tz = dt_util.get_time_zone("Europe/Paris")
    start_local = datetime(2023, 3, 27, 9, 0, tzinfo=tz)
    end_local = start_local + timedelta(hours=1)

    result = await backend.fetch_hourly_samples(
        "dev",
        [("htr", "A")],
        start_local,
        end_local,
    )

    call = client.get_node_samples.await_args
    assert call.args[0] == "dev"
    assert call.args[1] == ("htr", "A")
    assert call.args[2] == pytest.approx(
        start_local.astimezone(timezone.utc).timestamp()
    )
    assert call.args[3] == pytest.approx(end_local.astimezone(timezone.utc).timestamp())

    bucket = result[("htr", "A")]
    assert len(bucket) == 1
    sample = bucket[0]
    assert sample["energy_wh"] == pytest.approx(1_800.0)
    assert sample["power_w"] == pytest.approx(600.0)
    assert sample["ts"].tzinfo is timezone.utc


@pytest.mark.asyncio
async def test_ducaheat_backend_fetch_hourly_samples_normalises() -> None:
    """Ducaheat hourly fetch delegates to the shared normalisation helper."""

    client = SimpleNamespace()
    client.get_node_samples = AsyncMock(
        return_value=[{"t": 1_700_000_000, "counter": 7_200_000.0}]
    )
    backend = DucaheatBackend(brand=BRAND_DUCAHEAT, client=client)
    tz = dt_util.get_time_zone("Europe/Paris")
    start_local = datetime(2023, 3, 27, 9, 0, tzinfo=tz)
    end_local = start_local + timedelta(hours=1)

    result = await backend.fetch_hourly_samples(
        "dev",
        [("pmo", "M")],
        start_local,
        end_local,
    )

    call = client.get_node_samples.await_args
    assert call.args[1] == ("pmo", "M")
    bucket = result[("pmo", "M")]
    assert bucket[0]["energy_wh"] == pytest.approx(2_000.0)
    assert bucket[0]["ts"].tzinfo is timezone.utc
