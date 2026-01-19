"""Backend factory."""

from __future__ import annotations

from homeassistant.core import HomeAssistant
from homeassistant.helpers import aiohttp_client

from custom_components.termoweb.const import (
    get_brand_api_base,
    get_brand_basic_auth,
    uses_ducaheat_backend,
)

from .base import Backend, HttpClientProto
from .rest_client import RESTClient


def create_backend(*, brand: str, client: HttpClientProto) -> Backend:
    """Create a backend for the given brand."""

    if uses_ducaheat_backend(brand):
        from . import DucaheatBackend  # noqa: PLC0415

        return DucaheatBackend(brand=brand, client=client)

    from . import TermoWebBackend  # noqa: PLC0415

    return TermoWebBackend(brand=brand, client=client)


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
