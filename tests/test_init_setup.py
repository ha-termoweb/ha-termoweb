# ruff: noqa: D100,D101,D102,D103,D104,D105,D106,D107,INP001,E402
from __future__ import annotations

import asyncio
import importlib
import sys
from datetime import timedelta
from typing import Any, Callable, Coroutine

import pytest

from homeassistant import const as const_mod
from homeassistant import helpers as helpers_mod
from homeassistant import loader as loader_mod
from homeassistant.config_entries import ConfigEntry
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from homeassistant.helpers import aiohttp_client as aiohttp_client_mod
from homeassistant.helpers import dispatcher as dispatcher_mod
from homeassistant.helpers import entity_registry as entity_registry_mod

helpers_mod.aiohttp_client = aiohttp_client_mod
helpers_mod.entity_registry = entity_registry_mod
helpers_mod.dispatcher = dispatcher_mod
const_mod.EVENT_HOMEASSISTANT_STARTED = getattr(
    const_mod, "EVENT_HOMEASSISTANT_STARTED", "homeassistant_started"
)


class _StubClientSession:  # pragma: no cover - placeholder session
    pass


_STUB_SESSION = _StubClientSession()


def async_get_clientsession(hass: Any) -> _StubClientSession:  # pragma: no cover - stub
    if hasattr(hass, "client_session_calls"):
        hass.client_session_calls += 1
    return _STUB_SESSION


aiohttp_client_mod.async_get_clientsession = async_get_clientsession


def async_get_entity_registry(hass: Any) -> Any:  # pragma: no cover - stub
    return None


def async_dispatcher_connect(
    hass: Any, signal: str, callback: Callable[[dict[str, Any]], None]
) -> Callable[[], None]:  # pragma: no cover - stub
    if hasattr(hass, "dispatcher_connections"):
        hass.dispatcher_connections.append((signal, callback))
    return lambda: None


entity_registry_mod.async_get = async_get_entity_registry
dispatcher_mod.async_dispatcher_connect = async_dispatcher_connect


class _StubIntegration:  # pragma: no cover - minimal stub
    def __init__(self, domain: str) -> None:
        self.domain = domain
        self.version = "test-version"


async def async_get_integration(
    hass: Any, domain: str
) -> _StubIntegration:  # pragma: no cover
    if hasattr(hass, "integration_requests"):
        hass.integration_requests.append(domain)
    return _StubIntegration(domain)


loader_mod.async_get_integration = async_get_integration


class StubServices:
    def __init__(self) -> None:
        self._services: dict[tuple[str, str], Any] = {}

    def has_service(self, domain: str, service: str) -> bool:
        return (domain, service) in self._services

    def async_register(self, domain: str, service: str, handler: Any) -> None:
        self._services[(domain, service)] = handler


class StubConfigEntriesManager:
    def __init__(self) -> None:
        self.forwarded: list[tuple[ConfigEntry, tuple[str, ...]]] = []

    async def async_forward_entry_setups(
        self, entry: ConfigEntry, platforms: list[str] | tuple[str, ...]
    ) -> None:
        self.forwarded.append((entry, tuple(platforms)))

    def async_update_entry(
        self, entry: ConfigEntry, *, options: dict[str, Any] | None = None
    ) -> None:
        if options is not None:
            entry.options = dict(options)

    def async_get_entry(self, entry_id: str) -> ConfigEntry | None:
        return None


class StubHass:
    def __init__(self) -> None:
        self.data: dict[str, Any] = {}
        self.services = StubServices()
        self.config_entries = StubConfigEntriesManager()
        self.dispatcher_connections: list[tuple[str, Callable[[dict[str, Any]], None]]] = []
        self.tasks: list[asyncio.Task[Any]] = []
        self.integration_requests: list[str] = []
        self.client_session_calls = 0
        self.is_running = True

    def async_create_task(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro)
        self.tasks.append(task)
        return task


class FakeTask:
    def __init__(self) -> None:
        self._done = False

    def done(self) -> bool:
        return self._done


class FakeWSClient:
    def __init__(
        self,
        hass: StubHass,
        *,
        entry_id: str,
        dev_id: str,
        api_client: Any,
        coordinator: Any,
    ) -> None:
        self.hass = hass
        self.entry_id = entry_id
        self.dev_id = dev_id
        self.api_client = api_client
        self.coordinator = coordinator
        self.start_calls = 0

    def start(self) -> FakeTask:
        self.start_calls += 1
        return FakeTask()


class FakeCoordinator:
    def __init__(
        self,
        hass: StubHass,
        client: Any,
        base_interval: int,
        dev_id: str,
        dev: dict[str, Any],
        nodes: dict[str, Any],
    ) -> None:
        self.hass = hass
        self.client = client
        self.base_interval = base_interval
        self.dev_id = dev_id
        self.dev = dev
        self.nodes = nodes
        self.update_interval = timedelta(seconds=base_interval)
        self.data: dict[str, Any] = {dev_id: dev}
        self.listeners: list[Callable[[], None]] = []
        self.refresh_calls = 0

    async def async_config_entry_first_refresh(self) -> None:
        self.refresh_calls += 1

    def async_add_listener(self, listener: Callable[[], None]) -> None:
        self.listeners.append(listener)


class BaseFakeClient:
    def __init__(self, session: Any, username: str, password: str) -> None:
        self.session = session
        self.username = username
        self.password = password
        self.get_nodes_calls: list[str] = []

    async def get_nodes(self, dev_id: str) -> dict[str, Any]:
        self.get_nodes_calls.append(dev_id)
        return {}


@pytest.fixture
def termoweb_init(monkeypatch: pytest.MonkeyPatch) -> Any:
    for name in list(sys.modules):
        if name.startswith("custom_components.termoweb"):
            sys.modules.pop(name)

    module = importlib.import_module("custom_components.termoweb.__init__")
    module = importlib.reload(module)
    monkeypatch.setattr(module, "TermoWebCoordinator", FakeCoordinator)
    monkeypatch.setattr(module, "TermoWebWSLegacyClient", FakeWSClient)
    monkeypatch.setattr(module, "extract_heater_addrs", lambda _nodes: [])
    return module


@pytest.fixture
def stub_hass() -> StubHass:
    return StubHass()


def test_async_setup_entry_auth_error(
    termoweb_init: Any, stub_hass: StubHass, monkeypatch: pytest.MonkeyPatch
) -> None:
    class AuthErrorClient(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            raise termoweb_init.TermoWebAuthError("bad credentials")

    monkeypatch.setattr(termoweb_init, "TermoWebClient", AuthErrorClient)
    entry = ConfigEntry("auth", data={"username": "user", "password": "pw"})

    with pytest.raises(ConfigEntryAuthFailed):
        asyncio.run(termoweb_init.async_setup_entry(stub_hass, entry))


@pytest.mark.parametrize("error_case", ["timeout", "rate_limit"], ids=["timeout", "rate_limit"])
def test_async_setup_entry_connection_errors(
    termoweb_init: Any,
    stub_hass: StubHass,
    monkeypatch: pytest.MonkeyPatch,
    error_case: str,
) -> None:
    class ConnectionErrorClient(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            if error_case == "timeout":
                raise TimeoutError("timeout")
            raise termoweb_init.TermoWebRateLimitError("rate limited")

    monkeypatch.setattr(termoweb_init, "TermoWebClient", ConnectionErrorClient)
    entry = ConfigEntry("conn", data={"username": "user", "password": "pw"})

    with pytest.raises(ConfigEntryNotReady):
        asyncio.run(termoweb_init.async_setup_entry(stub_hass, entry))


def test_async_setup_entry_no_devices(
    termoweb_init: Any, stub_hass: StubHass, monkeypatch: pytest.MonkeyPatch
) -> None:
    class EmptyClient(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            return []

    monkeypatch.setattr(termoweb_init, "TermoWebClient", EmptyClient)
    entry = ConfigEntry("empty", data={"username": "user", "password": "pw"})

    with pytest.raises(ConfigEntryNotReady):
        asyncio.run(termoweb_init.async_setup_entry(stub_hass, entry))


def test_async_setup_entry_happy_path(
    termoweb_init: Any, stub_hass: StubHass, monkeypatch: pytest.MonkeyPatch
) -> None:
    class HappyClient(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            return [{"dev_id": "dev-1"}]

    monkeypatch.setattr(termoweb_init, "TermoWebClient", HappyClient)
    entry = ConfigEntry("happy", data={"username": "user", "password": "pw"})

    async def _run_setup() -> bool:
        result = await termoweb_init.async_setup_entry(stub_hass, entry)
        if stub_hass.tasks:
            await asyncio.gather(*stub_hass.tasks, return_exceptions=True)
        return result

    result = asyncio.run(_run_setup())
    assert result is True

    assert termoweb_init.DOMAIN in stub_hass.data
    assert stub_hass.config_entries.forwarded == [
        (entry, tuple(termoweb_init.PLATFORMS))
    ]
    assert stub_hass.services.has_service(
        termoweb_init.DOMAIN, "import_energy_history"
    )
