from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from custom_components.termoweb.backend import (
    create_backend,
    termoweb as termoweb_backend,
)
from custom_components.termoweb.backend.ducaheat import DucaheatBackend
from custom_components.termoweb.const import BRAND_DUCAHEAT, BRAND_TEVOLVE
from tests_ha.fakes.runtime import build_entry_runtime


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
