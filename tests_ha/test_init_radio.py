"""Radio entry setup failure releases the gateway on real Home Assistant."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any
from unittest.mock import patch

from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.termoweb.backend import factory
from custom_components.termoweb.backend.radio import GatewayInfo, RadioLinkError
from custom_components.termoweb.backend.radio_client import RadioClient
from custom_components.termoweb.const import BRAND_RADIO_MONITOR, CONF_BRAND, DOMAIN


class FakeLink:
    """Gateway connection that reports no MAC, so setup fails after connecting."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Start disconnected."""
        self.connected = False
        self.gateway_info: GatewayInfo | None = None
        self.connects = 0
        self.closes = 0

    async def connect(self) -> GatewayInfo:
        """Open the (fake) connection; the banner has no MAC address."""
        self.connects += 1
        self.connected = True
        self.gateway_info = GatewayInfo("3.5", "869.525", "2DE5", False, 1, None, "")
        return self.gateway_info

    async def close(self) -> None:
        """Close the (fake) connection."""
        self.closes += 1
        self.connected = False


@pytest.fixture
def radio_clients() -> Generator[list[tuple[RadioClient, list[FakeLink]]]]:
    """Build real RadioClients over fake links; keep them for assertions."""
    made: list[tuple[RadioClient, list[FakeLink]]] = []
    real_create = factory.create_radio_client

    def _create(*args: Any, **kwargs: Any) -> RadioClient:
        client = real_create(*args, **kwargs)
        links: list[FakeLink] = []

        def _factory(*a: Any, **kw: Any) -> FakeLink:
            links.append(FakeLink(*a, **kw))
            return links[-1]

        client._link_factory = _factory  # noqa: SLF001 - the transport boundary
        made.append((client, links))
        return client

    with patch.object(factory, "create_radio_client", _create):
        yield made


async def test_failed_radio_setup_closes_link_for_good(
    hass: HomeAssistant, radio_clients: list[tuple[RadioClient, list[FakeLink]]]
) -> None:
    """B4/B5: a failed setup closes its connection and the client stays closed."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        minor_version=2,
        unique_id=f"{BRAND_RADIO_MONITOR}:0a0b0c0d0e0f",
        data={CONF_BRAND: BRAND_RADIO_MONITOR, "host": "10.0.0.5", "port": 2323},
    )
    entry.add_to_hass(hass)

    assert not await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    assert entry.state is ConfigEntryState.SETUP_RETRY
    [(client, [link])] = radio_clients
    assert (link.connects, link.closes) == (1, 1)
    assert not client.connected
    with pytest.raises(RadioLinkError, match="closed"):
        await client.async_connect()
    assert link.connects == 1
