"""Backend factory."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import functools
from typing import TYPE_CHECKING, Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers import aiohttp_client

from custom_components.termoweb.const import (
    BRAND_RADIO,
    BRAND_RADIO_MONITOR,
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
    if brand == BRAND_RADIO_MONITOR:
        from .radio_monitor import RadioMonitorBackend  # noqa: PLC0415

        return RadioMonitorBackend
    if uses_ducaheat_backend(brand):
        from .ducaheat import DucaheatBackend  # noqa: PLC0415

        return DucaheatBackend

    from .termoweb import TermoWebBackend  # noqa: PLC0415

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
    serial_url: str | None = None,
    device_id: str | None = None,
    listen_only: bool = False,
) -> RadioClient:
    """Return a radio client for the gateway at ``host:port`` (or a USB stick)."""

    from .radio_client import LISTEN_ONLY_STATION_ID, NANOCUL_MODEL, RadioClient  # noqa: PLC0415

    listen: dict[str, Any] = (
        {"listen_only": True, "station_id": LISTEN_ONLY_STATION_ID}
        if listen_only
        else {}
    )
    if serial_url is None:
        return RadioClient(
            host, port, dialect, nodes, network_id=network_id, power=power, **listen
        )
    from .radio.link import RadioLink  # noqa: PLC0415
    from .radio.serial_link import serial_opener  # noqa: PLC0415

    return RadioClient(
        serial_url,
        port,
        dialect,
        nodes,
        network_id=network_id,
        power=power,
        link_factory=functools.partial(
            RadioLink, open_connection=serial_opener(serial_url)
        ),
        device_id=device_id,
        model=NANOCUL_MODEL,
        **listen,
    )


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
