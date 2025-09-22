# ruff: noqa: D100,D101,D102,D103,D104,D105,D106,D107,INP001,E402
from __future__ import annotations

import asyncio
import importlib
import sys
from datetime import timedelta
from types import SimpleNamespace
from typing import Any, Callable, Coroutine

import pytest
from unittest.mock import AsyncMock

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


def async_dispatcher_connect(
    hass: Any, signal: str, callback: Callable[[dict[str, Any]], None]
) -> Callable[[], None]:  # pragma: no cover - stub
    if hasattr(hass, "dispatcher_connections"):
        hass.dispatcher_connections.append((signal, callback))
    return lambda: None


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

    def get(self, domain: str, service: str) -> Any:
        return self._services[(domain, service)]


class StubBus:
    def __init__(self) -> None:
        self.listeners: list[tuple[str, Callable[[Any], Any]]] = []

    def async_listen_once(self, event: str, callback: Callable[[Any], Any]) -> None:
        self.listeners.append((event, callback))


class StubConfigEntriesManager:
    def __init__(self) -> None:
        self.forwarded: list[tuple[ConfigEntry, tuple[str, ...]]] = []
        self.unloaded: list[tuple[ConfigEntry, tuple[str, ...]]] = []
        self._entries: dict[str, ConfigEntry] = {}
        self.updated: list[tuple[ConfigEntry, dict[str, Any] | None]] = []

    def add(self, entry: ConfigEntry) -> None:
        self._entries[entry.entry_id] = entry

    async def async_forward_entry_setups(
        self, entry: ConfigEntry, platforms: list[str] | tuple[str, ...]
    ) -> None:
        self.forwarded.append((entry, tuple(platforms)))

    async def async_unload_platforms(
        self, entry: ConfigEntry, platforms: list[str] | tuple[str, ...]
    ) -> bool:
        self.unloaded.append((entry, tuple(platforms)))
        return True

    def async_update_entry(
        self, entry: ConfigEntry, *, options: dict[str, Any] | None = None
    ) -> None:
        if options is not None:
            entry.options = dict(options)
        self.updated.append((entry, options))
        self._entries[entry.entry_id] = entry

    def async_get_entry(self, entry_id: str) -> ConfigEntry | None:
        return self._entries.get(entry_id)


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
        self.bus = StubBus()

    def async_create_task(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro)
        self.tasks.append(task)
        return task


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
        self.start_calls: list[asyncio.Task[Any]] = []
        self.stop_calls = 0

    def start(self) -> asyncio.Task[Any]:
        async def _runner() -> None:
            await asyncio.sleep(0)

        task = asyncio.create_task(_runner())
        self.start_calls.append(task)
        return task

    async def stop(self) -> None:
        self.stop_calls += 1


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

    async def list_devices(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def get_nodes(self, dev_id: str) -> dict[str, Any]:
        self.get_nodes_calls.append(dev_id)
        return {}


def _extract_addrs(nodes: dict[str, Any]) -> list[str]:
    addrs: list[str] = []
    node_list = nodes.get("nodes") if isinstance(nodes, dict) else None
    if isinstance(node_list, list):
        for node in node_list:
            if isinstance(node, dict) and "addr" in node:
                addrs.append(str(node["addr"]))
    return addrs


async def _drain_tasks(hass: StubHass) -> None:
    if hass.tasks:
        await asyncio.gather(*hass.tasks, return_exceptions=True)
        hass.tasks.clear()


@pytest.fixture
def termoweb_init(monkeypatch: pytest.MonkeyPatch) -> Any:
    for name in list(sys.modules):
        if name.startswith("custom_components.termoweb"):
            sys.modules.pop(name)

    module = importlib.import_module("custom_components.termoweb.__init__")
    module = importlib.reload(module)
    monkeypatch.setattr(module, "TermoWebCoordinator", FakeCoordinator)
    monkeypatch.setattr(module, "TermoWebWSLegacyClient", FakeWSClient)
    monkeypatch.setattr(module, "extract_heater_addrs", _extract_addrs)
    return module


@pytest.fixture
def stub_hass() -> StubHass:
    hass = StubHass()
    hass.data = {}
    return hass


class StubEntityEntry:
    def __init__(
        self,
        entity_id: str,
        *,
        unique_id: str,
        platform: str,
        config_entry_id: str,
    ) -> None:
        self.entity_id = entity_id
        self.unique_id = unique_id
        self.platform = platform
        self.config_entry_id = config_entry_id


class StubEntityRegistry:
    def __init__(self) -> None:
        self._entities: dict[str, StubEntityEntry] = {}

    def add(
        self,
        entity_id: str,
        *,
        unique_id: str,
        platform: str,
        config_entry_id: str,
    ) -> StubEntityEntry:
        entry = StubEntityEntry(
            entity_id,
            unique_id=unique_id,
            platform=platform,
            config_entry_id=config_entry_id,
        )
        self._entities[entity_id] = entry
        return entry

    def async_get(self, entity_id: str) -> StubEntityEntry | None:
        return self._entities.get(entity_id)


def test_async_setup_entry_happy_path(
    termoweb_init: Any, stub_hass: StubHass, monkeypatch: pytest.MonkeyPatch
) -> None:
    class HappyClient(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            return [{"dev_id": "dev-1"}]

        async def get_nodes(self, dev_id: str) -> dict[str, Any]:
            assert dev_id == "dev-1"
            return {"nodes": [{"addr": "A"}, {"addr": "B"}]}

    monkeypatch.setattr(termoweb_init, "TermoWebClient", HappyClient)
    import_mock = AsyncMock()
    monkeypatch.setattr(termoweb_init, "_async_import_energy_history", import_mock)

    entry = ConfigEntry("happy", data={"username": "user", "password": "pw"})
    stub_hass.config_entries.add(entry)

    async def _run() -> bool:
        result = await termoweb_init.async_setup_entry(stub_hass, entry)
        await _drain_tasks(stub_hass)
        return result

    assert asyncio.run(_run()) is True

    record = stub_hass.data[termoweb_init.DOMAIN][entry.entry_id]
    assert isinstance(record["client"], HappyClient)
    assert isinstance(record["coordinator"], FakeCoordinator)
    assert record["coordinator"].refresh_calls == 1
    assert record["htr_addrs"] == ["A", "B"]
    assert stub_hass.client_session_calls == 1
    assert stub_hass.config_entries.forwarded == [
        (entry, tuple(termoweb_init.PLATFORMS))
    ]
    assert stub_hass.services.has_service(
        termoweb_init.DOMAIN, "import_energy_history"
    )
    import_mock.assert_awaited_once_with(stub_hass, entry)


def test_async_setup_entry_auth_error(
    termoweb_init: Any, stub_hass: StubHass, monkeypatch: pytest.MonkeyPatch
) -> None:
    class AuthClient(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            raise termoweb_init.TermoWebAuthError("bad credentials")

    monkeypatch.setattr(termoweb_init, "TermoWebClient", AuthClient)
    entry = ConfigEntry("auth", data={"username": "user", "password": "pw"})
    stub_hass.config_entries.add(entry)

    async def _run() -> None:
        await termoweb_init.async_setup_entry(stub_hass, entry)

    with pytest.raises(ConfigEntryAuthFailed):
        asyncio.run(_run())


@pytest.mark.parametrize(
    "error_case",
    ["timeout", "rate_limit"],
    ids=["timeout", "rate_limit"],
)
def test_async_setup_entry_transient_errors(
    termoweb_init: Any,
    stub_hass: StubHass,
    monkeypatch: pytest.MonkeyPatch,
    error_case: str,
) -> None:
    class ErrorClient(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            if error_case == "timeout":
                raise TimeoutError("timeout")
            raise termoweb_init.TermoWebRateLimitError("rate limit")

    monkeypatch.setattr(termoweb_init, "TermoWebClient", ErrorClient)
    entry = ConfigEntry("transient", data={"username": "user", "password": "pw"})
    stub_hass.config_entries.add(entry)

    async def _run() -> None:
        await termoweb_init.async_setup_entry(stub_hass, entry)

    with pytest.raises(ConfigEntryNotReady):
        asyncio.run(_run())


def test_async_setup_entry_no_devices(
    termoweb_init: Any, stub_hass: StubHass, monkeypatch: pytest.MonkeyPatch
) -> None:
    class EmptyClient(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            return []

    monkeypatch.setattr(termoweb_init, "TermoWebClient", EmptyClient)
    entry = ConfigEntry("empty", data={"username": "user", "password": "pw"})
    stub_hass.config_entries.add(entry)

    async def _run() -> None:
        await termoweb_init.async_setup_entry(stub_hass, entry)

    with pytest.raises(ConfigEntryNotReady):
        asyncio.run(_run())


def test_async_setup_entry_defers_until_started(
    termoweb_init: Any, stub_hass: StubHass, monkeypatch: pytest.MonkeyPatch
) -> None:
    class HappyClient(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            return [{"dev_id": "dev-1"}]

        async def get_nodes(self, dev_id: str) -> dict[str, Any]:
            return {"nodes": [{"addr": "A"}]}

    monkeypatch.setattr(termoweb_init, "TermoWebClient", HappyClient)
    import_mock = AsyncMock()
    monkeypatch.setattr(termoweb_init, "_async_import_energy_history", import_mock)

    entry = ConfigEntry("startup", data={"username": "user", "password": "pw"})
    stub_hass.config_entries.add(entry)
    stub_hass.is_running = False

    async def _run() -> None:
        await termoweb_init.async_setup_entry(stub_hass, entry)
        await _drain_tasks(stub_hass)
        assert not import_mock.await_count
        assert stub_hass.bus.listeners
        event, callback = stub_hass.bus.listeners[0]
        assert event == termoweb_init.EVENT_HOMEASSISTANT_STARTED
        await callback(None)
        await _drain_tasks(stub_hass)

    asyncio.run(_run())
    import_mock.assert_awaited_once_with(stub_hass, entry)


def test_import_energy_history_service_invocation(
    termoweb_init: Any, stub_hass: StubHass, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = StubEntityRegistry()
    monkeypatch.setattr(
        termoweb_init.er, "async_get", lambda hass: registry, raising=False
    )
    monkeypatch.setattr(
        entity_registry_mod, "async_get", lambda hass: registry, raising=False
    )

    class HappyClient(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            return [{"dev_id": "dev-1"}]

        async def get_nodes(self, dev_id: str) -> dict[str, Any]:
            return {"nodes": [{"addr": "A"}, {"addr": "B"}]}

    monkeypatch.setattr(termoweb_init, "TermoWebClient", HappyClient)
    import_mock = AsyncMock()
    monkeypatch.setattr(termoweb_init, "_async_import_energy_history", import_mock)

    entry = ConfigEntry("service", data={"username": "user", "password": "pw"})
    stub_hass.config_entries.add(entry)

    async def _run() -> None:
        assert await termoweb_init.async_setup_entry(stub_hass, entry)
        await _drain_tasks(stub_hass)
        import_mock.reset_mock()

        service = stub_hass.services.get(
            termoweb_init.DOMAIN, "import_energy_history"
        )

        registry.add(
            "sensor.dev_a_energy",
            unique_id=f"{termoweb_init.DOMAIN}:dev-1:htr:A:energy",
            platform=termoweb_init.DOMAIN,
            config_entry_id=entry.entry_id,
        )
        registry.add(
            "sensor.dev_b_energy",
            unique_id=f"{termoweb_init.DOMAIN}:dev-1:htr:B:energy",
            platform=termoweb_init.DOMAIN,
            config_entry_id=entry.entry_id,
        )

        call = SimpleNamespace(
            data={
                "entity_id": ["sensor.dev_a_energy", "sensor.dev_b_energy"],
                "reset_progress": True,
                "max_history_retrieval": 10,
            }
        )
        await service(call)
        assert import_mock.await_count == 1
        args, kwargs = import_mock.await_args
        assert args[0] is stub_hass
        assert args[1] is entry
        assert set(args[2]) == {"A", "B"}
        assert kwargs == {"reset_progress": True, "max_days": 10}

        import_mock.reset_mock()
        call_all = SimpleNamespace(data={"max_history_retrieval": 3})
        await service(call_all)
        assert import_mock.await_count == 1
        args, kwargs = import_mock.await_args
        assert args[0] is stub_hass
        assert args[1] is entry
        assert args[2] is None
        assert kwargs == {"reset_progress": False, "max_days": 3}

    asyncio.run(_run())


def test_async_unload_entry_cleans_up(
    termoweb_init: Any, stub_hass: StubHass
) -> None:
    entry = ConfigEntry("unload", data={})
    stub_hass.config_entries.add(entry)

    async def _run() -> tuple[bool, list[bool], int, bool]:
        cancel_events: list[bool] = []
        unsubscribed: list[bool] = []

        async def _ws_runner() -> None:
            wait = asyncio.Event()
            try:
                await wait.wait()
            except asyncio.CancelledError:
                cancel_events.append(True)
                raise

        ws_task = asyncio.create_task(_ws_runner())
        await asyncio.sleep(0)

        class DummyClient:
            def __init__(self) -> None:
                self.stop_calls = 0

            async def stop(self) -> None:
                self.stop_calls += 1

        client = DummyClient()

        record = {
            "ws_tasks": {"dev": ws_task},
            "ws_clients": {"dev": client},
            "unsub_ws_status": lambda: unsubscribed.append(True),
            "recalc_poll": lambda: None,
        }
        stub_hass.data.setdefault(termoweb_init.DOMAIN, {})[entry.entry_id] = record

        result = await termoweb_init.async_unload_entry(stub_hass, entry)
        return result, cancel_events, client.stop_calls, ws_task.cancelled()

    result, cancel_events, stop_calls, task_cancelled = asyncio.run(_run())
    assert result is True
    assert cancel_events == [True]
    assert stop_calls == 1
    assert task_cancelled is True
    assert stub_hass.config_entries.unloaded == [
        (entry, tuple(termoweb_init.PLATFORMS))
    ]
    assert entry.entry_id not in stub_hass.data.get(termoweb_init.DOMAIN, {})


def test_async_update_entry_options_recalculates_poll(
    termoweb_init: Any, stub_hass: StubHass
) -> None:
    entry = ConfigEntry("options", data={})
    stub_hass.config_entries.add(entry)
    recalc_calls: list[bool] = []
    stub_hass.data.setdefault(termoweb_init.DOMAIN, {})[entry.entry_id] = {
        "ws_tasks": {},
        "ws_clients": {},
        "recalc_poll": lambda: recalc_calls.append(True),
    }

    asyncio.run(termoweb_init.async_update_entry_options(stub_hass, entry))
    assert recalc_calls == [True]
