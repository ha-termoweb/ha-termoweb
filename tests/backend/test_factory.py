"""Tests for backend selection and capabilities (backend/factory.py)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import inspect
import re
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from homeassistant.util import dt as dt_util
import pytest

from custom_components import termoweb
from custom_components.termoweb import config_flow, const, diagnostics, utils
from custom_components.termoweb.backend import (
    Backend,  # noqa: E402
    create_backend,
    termoweb as termoweb_backend,
)
from custom_components.termoweb.backend.ducaheat import (
    DucaheatBackend,
    DucaheatRESTClient,
)
from custom_components.termoweb.backend.factory import create_rest_client
from custom_components.termoweb.backend.rest_client import RESTClient
from custom_components.termoweb.backend.termoweb import TermoWebBackend  # noqa: E402
from custom_components.termoweb.const import (
    BRAND_DUCAHEAT,
    BRAND_TEVOLVE,
    get_brand_api_base,
    get_brand_basic_auth,
)
from custom_components.termoweb.identifiers import build_cloud_unique_id
from tests.fakes.runtime import build_entry_runtime


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


def test_create_backend_returns_tevolve_backend() -> None:
    client = DummyHttpClient()
    backend = create_backend(brand=BRAND_TEVOLVE, client=client)
    assert isinstance(backend, DucaheatBackend)
    assert backend.brand == BRAND_TEVOLVE
    assert backend.client is client


async def test_termoweb_backend_creates_ws_client(hass) -> None:
    client = DummyHttpClient()
    backend = termoweb_backend.TermoWebBackend(brand="termoweb", client=client)
    coordinator = object()
    build_entry_runtime(hass=hass, entry_id="entry123", dev_id="device456")
    inventory = object()
    ws_client = backend.create_ws_client(
        hass,
        entry_id="entry123",
        dev_id="device456",
        coordinator=coordinator,
        inventory=inventory,
    )

    assert isinstance(ws_client, termoweb_backend.TermoWebWSClient)
    assert ws_client.dev_id == "device456"
    assert ws_client.entry_id == "entry123"
    assert ws_client._coordinator is coordinator
    assert getattr(ws_client, "_inventory", None) is inventory


def test_backend_capabilities_follow_the_backend_class() -> None:
    """Optional features are declared per backend class, looked up by brand."""

    from custom_components.termoweb.backend import (
        BackendCapabilities,
        backend_capabilities,
    )
    from custom_components.termoweb.backend.base import Backend
    from custom_components.termoweb.const import BRAND_TERMOWEB

    assert backend_capabilities(BRAND_TERMOWEB) == BackendCapabilities(
        power_limit=True,
        priority=True,
        energy_history=True,
        energy=True,
        geo_data=True,
    )
    for brand in (BRAND_DUCAHEAT, BRAND_TEVOLVE):
        assert backend_capabilities(brand) == BackendCapabilities(
            lock=True,
            priority=True,
            energy_history=True,
            energy=True,
            geo_data=True,
            account_scope="ducaheat",
        )
    radio = backend_capabilities("radio")
    assert radio.energy and not radio.geo_data  # estimated energy, no location
    assert radio.local_radio and radio.options_flow and radio.site_device
    assert not radio.web_portal
    assert Backend.capabilities == BackendCapabilities()
    backend = create_backend(brand=BRAND_DUCAHEAT, client=DummyHttpClient())
    assert backend.capabilities is backend_capabilities(BRAND_DUCAHEAT)


@pytest.mark.parametrize(
    ("brand", "client_cls"),
    [
        ("termoweb", RESTClient),
        (BRAND_DUCAHEAT, DucaheatRESTClient),
        (BRAND_TEVOLVE, DucaheatRESTClient),
    ],
)
async def test_create_rest_client_selects_the_brand_backend(
    hass, brand: str, client_cls: type[RESTClient]
) -> None:
    """Each brand gets its client class, API base and basic-auth credentials."""
    client = create_rest_client(hass, "user", "pw", brand)

    assert type(client) is client_cls
    assert client.api_base == get_brand_api_base(brand)
    assert client._basic_auth_b64 == get_brand_basic_auth(brand)  # noqa: SLF001


def test_backend_requires_create_override() -> None:
    """Attempting to instantiate a backend without a websocket factory fails."""

    class InvalidBackend(Backend):
        pass

    client = DummyHttpClient()
    with pytest.raises(TypeError):
        InvalidBackend(brand="termoweb", client=client)


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


# Brand names or vendor client types that only backend/ may branch on.
_VENDOR_BRANCH = re.compile(
    r"BRAND_RADIO|RADIO_BRANDS|brand(\)|\])? [!=]=|isinstance\([^)]*RadioClient"
)


@pytest.mark.parametrize(
    "source",
    [
        termoweb,
        utils,
        diagnostics,
        config_flow.TermoWebConfigFlow.async_supports_options_flow,
        config_flow.TermoWebConfigFlow.async_step_reconfigure,
        config_flow._radio_in_use,  # noqa: SLF001
        const.get_brand_label,
        build_cloud_unique_id,
    ],
    ids=lambda source: (
        getattr(source, "__qualname__", None) or getattr(source, "__name__", "?")
    ),
)
def test_shared_code_does_not_branch_on_brand(source: ModuleType) -> None:
    """Radio and brand differences come from BackendCapabilities or the factory."""
    assert not _VENDOR_BRANCH.findall(inspect.getsource(source))


@pytest.mark.parametrize(
    ("brand", "unique_id"),
    [
        (const.BRAND_TERMOWEB, "user@example.com"),
        (const.BRAND_DUCAHEAT, "ducaheat:user@example.com"),
        (const.BRAND_TEVOLVE, "ducaheat:user@example.com"),
    ],
)
def test_cloud_unique_id_is_scoped_by_backend(brand: str, unique_id: str) -> None:
    """Brands that share a backend share an account namespace."""
    assert build_cloud_unique_id(brand, " User@Example.COM ") == unique_id
