"""Backend factory."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers import aiohttp_client

from custom_components.termoweb.const import (
    BRAND_RADIO,
    get_brand_api_base,
    get_brand_basic_auth,
    uses_ducaheat_backend,
)

from .base import Backend, BackendCapabilities, HttpClientProto
from .rest_client import RESTClient

if TYPE_CHECKING:
    from .radio_client import RadioClient
    from .radio_power import PowerManager


def _backend_class(brand: str) -> type[Backend]:
    """Return the backend class serving the given brand."""

    if brand == BRAND_RADIO:
        from .radio_backend import RadioBackend  # noqa: PLC0415

        return RadioBackend
    if uses_ducaheat_backend(brand):
        from . import DucaheatBackend  # noqa: PLC0415

        return DucaheatBackend

    from . import TermoWebBackend  # noqa: PLC0415

    return TermoWebBackend


def create_backend(*, brand: str, client: HttpClientProto) -> Backend:
    """Create a backend for the given brand."""

    return _backend_class(brand)(brand=brand, client=client)


def backend_capabilities(brand: str) -> BackendCapabilities:
    """Return the optional features the brand's backend supports."""

    return _backend_class(brand).capabilities


def create_radio_client(
    host: str,
    port: int,
    dialect: str,
    nodes: Iterable[Mapping[str, Any]],
    network_id: bytes | None,
    *,
    power: PowerManager | None = None,
) -> RadioClient:
    """Return a radio client for the gateway at ``host:port`` and its stored nodes."""

    from .radio_client import RadioClient  # noqa: PLC0415

    return RadioClient(host, port, dialect, nodes, network_id=network_id, power=power)


def create_rest_client(
    hass: HomeAssistant, username: str, password: str, brand: str
) -> RESTClient:
    """Return a REST client configured for the selected brand."""

    session = aiohttp_client.async_get_clientsession(hass)
    api_base = get_brand_api_base(brand)
    basic_auth = get_brand_basic_auth(brand)
    if uses_ducaheat_backend(brand):
        from .ducaheat import DucaheatRESTClient  # noqa: PLC0415

        client_cls = DucaheatRESTClient
    else:
        client_cls = RESTClient
    return client_cls(
        session,
        username,
        password,
        api_base=api_base,
        basic_auth_b64=basic_auth,
    )
