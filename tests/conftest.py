"""Real Home Assistant test harness (pytest-homeassistant-custom-component).

Every test runs against the real ``homeassistant`` and ``aiohttp`` packages.
The fakes live at the backend boundary only (``tests/fakes``): the public
``RESTClient`` methods, an ``aiohttp`` session double and the websocket client
factory. Home Assistant itself is never patched. Pure-module tests (codecs,
domain, backend clients, radio protocol) simply do not request ``hass``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Generator, Iterable, Mapping
from typing import Any
from unittest.mock import patch

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.termoweb.backend import rest_client as rest_client_module
from custom_components.termoweb.backend.rest_client import RESTClient
from custom_components.termoweb.backend.termoweb import TermoWebBackend
from custom_components.termoweb.const import BRAND_TERMOWEB, CONF_BRAND, DOMAIN
from custom_components.termoweb.inventory import Inventory, build_node_inventory
from tests.fakes.cloud import PASSWORD, USERNAME, FakeCloud


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(request: pytest.FixtureRequest) -> None:
    """Let Home Assistant load ``custom_components/termoweb`` in tests using hass.

    Pure-module tests never request ``hass``, so they skip building one.
    """
    if "hass" in request.fixturenames:
        request.getfixturevalue("enable_custom_integrations")


@pytest.fixture(autouse=True)
async def enable_event_loop_debug(request: pytest.FixtureRequest) -> None:
    """Run the loop in debug mode for tests using hass (overrides the plugin's).

    Debug mode catches Home Assistant thread-safety mistakes but makes the
    fake-clock radio tests about 20x slower, and pure-module tests have no
    hass to misuse.
    """
    if "hass" in request.fixturenames:
        asyncio.get_running_loop().set_debug(True)


@pytest.fixture(autouse=True)
def _no_rest_spacing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable the 0.5 s REST spacing; limiter tests inject their own clock."""
    monkeypatch.setattr(rest_client_module, "REST_MIN_INTERVAL_S", 0.0)


@pytest.fixture
def inventory_builder() -> Callable[..., Inventory]:
    """Return a helper that builds an ``Inventory`` from a nodes payload or list."""

    def _factory(
        dev_id: str,
        payload: Mapping[str, Any] | None = None,
        nodes: Iterable[Any] | None = None,
    ) -> Inventory:
        node_list = list(nodes or [])
        if not node_list and payload is not None:
            node_list = list(build_node_inventory(payload))
        return Inventory(dev_id, node_list)

    return _factory


@pytest.fixture
def cloud(hass: Any) -> Generator[FakeCloud]:
    """Patch the REST client and websocket factory; unpatch only after hass stops."""
    fake = FakeCloud()
    methods = (
        "list_devices",
        "get_nodes",
        "get_geo_data",
        "get_node_settings",
        "get_node_samples",
        "get_rtc_time",
        "get_power_limit",
    )
    patches = [patch.object(RESTClient, name, getattr(fake, name)) for name in methods]
    patches.append(
        patch.object(
            TermoWebBackend,
            "create_ws_client",
            lambda _self, hass, *a, **kw: fake.create_ws_client(hass, *a, **kw),
        )
    )
    for p in patches:
        p.start()
    yield fake
    for p in reversed(patches):
        p.stop()


@pytest.fixture
def config_entry() -> MockConfigEntry:
    """Return a TermoWeb cloud config entry (not yet added to hass)."""
    return MockConfigEntry(
        domain=DOMAIN,
        title=f"TermoWeb ({USERNAME})",
        unique_id=USERNAME,
        data={"username": USERNAME, "password": PASSWORD, CONF_BRAND: BRAND_TERMOWEB},
    )
