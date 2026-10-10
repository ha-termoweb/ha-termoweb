"""Real Home Assistant test harness (pytest-homeassistant-custom-component).

These tests run against the real ``homeassistant`` and ``aiohttp`` packages, so
they cannot share a pytest session with ``tests/`` (whose conftest replaces
``homeassistant`` in ``sys.modules`` with stubs). Run them in their own
invocation::

    pytest tests_ha -p homeassistant -o asyncio_mode=auto

The fakes live at the backend boundary only (``tests_ha/fakes``): the public
``RESTClient`` methods, an ``aiohttp`` session double and the websocket client
factory. Home Assistant itself is never patched. Pure-module tests (codecs,
domain, backend clients, radio protocol) simply do not request ``hass``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Generator, Iterable, Mapping
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.termoweb.backend import rest_client as rest_client_module
from custom_components.termoweb.backend.rest_client import RESTClient
from custom_components.termoweb.backend.termoweb import TermoWebBackend
from custom_components.termoweb.const import BRAND_TERMOWEB, CONF_BRAND, DOMAIN
from custom_components.termoweb.inventory import Inventory, build_node_inventory

VERSION = json.loads(
    (Path(__file__).parents[1] / "custom_components/termoweb/manifest.json").read_text()
)["version"]

# Synthetic identifiers only.
DEV_ID = "0123456789abcdef"
USERNAME = "user@example.com"
PASSWORD = "secret"

DEVICES = [{"dev_id": DEV_ID, "name": "Home", "serial_id": "SN-TEST"}]
NODES = {"nodes": [{"type": "htr", "addr": 1, "name": "Living room"}]}
HTR_SETTINGS = {
    "mode": "auto",
    "state": "off",
    "stemp": "21.0",
    "mtemp": "20.5",
    "units": "C",
    "prog": [0] * 168,
    "ptemp": ["7.0", "17.0", "21.0"],
}


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


class FakeWSClient:
    """Websocket client stand-in: a task that idles until stopped."""

    def __init__(self, hass: Any) -> None:
        """Remember hass so the idle task is tracked by it."""
        self.hass = hass
        self.task: asyncio.Task[None] | None = None
        self.stop_calls = 0

    def start(self) -> asyncio.Task[None]:
        """Start the idle task, mirroring the real client's contract."""
        self.task = self.hass.async_create_background_task(
            asyncio.Event().wait(), "termoweb-fake-ws"
        )
        return self.task

    async def stop(self) -> None:
        """Record the stop request and cancel the idle task."""
        self.stop_calls += 1
        if self.task is not None:
            self.task.cancel()


class FakeCloud:
    """Handles to the patched backend-boundary methods used by the tests."""

    def __init__(self) -> None:
        """Create async mocks with a healthy single-heater account."""
        self.list_devices = AsyncMock(return_value=DEVICES)
        self.get_nodes = AsyncMock(return_value=NODES)
        self.get_geo_data = AsyncMock(return_value=None)
        self.get_node_settings = AsyncMock(return_value=HTR_SETTINGS)
        self.get_node_samples = AsyncMock(return_value=[])
        self.get_rtc_time = AsyncMock(return_value={})
        self.get_power_limit = AsyncMock(return_value=None)
        self.ws_clients: list[FakeWSClient] = []

    def create_ws_client(self, hass: Any, *args: Any, **kwargs: Any) -> FakeWSClient:
        """Return a fake websocket client and keep it for assertions."""
        client = FakeWSClient(hass)
        self.ws_clients.append(client)
        return client


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
