"""Helpers for creating REST clients and shared API calls."""

from __future__ import annotations

import logging
from typing import Any

from aiohttp import ClientError
from homeassistant.core import HomeAssistant
from homeassistant.helpers import aiohttp_client

from .api import BackendAuthError, BackendRateLimitError, RESTClient
from .backend import DucaheatRESTClient
from .const import (
    BRAND_DUCAHEAT,
    DEFAULT_BRAND,
    get_brand_api_base,
    get_brand_basic_auth,
)

_LOGGER = logging.getLogger(__name__)


def create_rest_client(
    hass: HomeAssistant, username: str, password: str, brand: str | None
) -> RESTClient:
    """Return a REST client configured for the requested brand."""

    normalized = brand or DEFAULT_BRAND
    session = aiohttp_client.async_get_clientsession(hass)
    api_base = get_brand_api_base(normalized)
    basic_auth = get_brand_basic_auth(normalized)
    client_cls = DucaheatRESTClient if normalized == BRAND_DUCAHEAT else RESTClient
    return client_cls(
        session,
        username,
        password,
        api_base=api_base,
        basic_auth_b64=basic_auth,
    )


async def async_list_devices_with_logging(client: RESTClient) -> Any:
    """Call ``list_devices`` logging consistent diagnostic information."""

    try:
        return await client.list_devices()
    except BackendAuthError as err:
        _LOGGER.info("list_devices auth error: %s", err)
        raise
    except (TimeoutError, ClientError, BackendRateLimitError) as err:
        _LOGGER.info("list_devices connection error: %s", err)
        raise
