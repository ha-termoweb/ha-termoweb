# ruff: noqa: D100,D101,D102,D103,D104,D105,D106,D107,INP001,E402
from __future__ import annotations

import asyncio
import types
import importlib
import logging
import sys
from datetime import timedelta
from types import SimpleNamespace
from typing import Any, Callable, Coroutine, Mapping

import pytest
from unittest.mock import AsyncMock

from conftest import FakeCoordinator, _install_stubs

_install_stubs()

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EVENT_HOMEASSISTANT_STARTED
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from homeassistant.helpers import entity_registry as entity_registry_mod

from custom_components.termoweb.identifiers import build_heater_energy_unique_id
from custom_components.termoweb.inventory import build_heater_address_map


class FakeWSClient:
    def __init__(
        self,
        hass: HomeAssistant,
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


class BaseFakeClient:
    def __init__(
        self,
        session: Any,
        username: str,
        password: str,
        **kwargs: Any,
    ) -> None:
        self.session = session
        self.username = username
        self.password = password
        self.api_base = kwargs.get("api_base")
        self.basic_auth_b64 = kwargs.get("basic_auth_b64")
        self.get_nodes_calls: list[str] = []

    async def list_devices(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def get_nodes(self, dev_id: str) -> dict[str, Any]:
        self.get_nodes_calls.append(dev_id)
        return {}


class DiagnosticsConfigEntry(ConfigEntry):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.supports_diagnostics: Any = None
        self.update_calls: list[dict[str, Any]] = []

    async def async_update(self, data: Mapping[str, Any]) -> None:
        self.update_calls.append(dict(data))
        self.data = dict(data)


class DiagnosticsNoUpdateEntry(ConfigEntry):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.supports_diagnostics: Any = None


def test_create_rest_client_selects_brand(
    termoweb_init: Any,
    stub_hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DefaultClient(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            return []

    class DucaClient(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            return []

    monkeypatch.setattr(termoweb_init, "RESTClient", DefaultClient)
    monkeypatch.setattr(termoweb_init, "DucaheatRESTClient", DucaClient)

    default_client = termoweb_init.create_rest_client(
        stub_hass,
        "user",
        "pw",
        termoweb_init.DEFAULT_BRAND,
    )
    duca_client = termoweb_init.create_rest_client(
        stub_hass,
        "user2",
        "pw2",
        termoweb_init.BRAND_DUCAHEAT,
    )

    assert isinstance(default_client, DefaultClient)
    assert isinstance(duca_client, DucaClient)
    assert default_client.api_base == termoweb_init.get_brand_api_base(
        termoweb_init.DEFAULT_BRAND
    )
    assert duca_client.basic_auth_b64 == termoweb_init.get_brand_basic_auth(
        termoweb_init.BRAND_DUCAHEAT
    )
    assert stub_hass.client_session_calls == 2


def test_diagnostics_module_registers_via_home_assistant_helper(
    termoweb_init: Any,
    stub_hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostics_calls: list[tuple[str, types.ModuleType]] = []

    diagnostics_module = types.ModuleType("diagnostics_component_stub")
    diagnostics_module.DOMAIN = "diagnostics"
    diagnostics_module._DIAGNOSTICS_DATA = "diag_key"

    diagnostics_module.async_redact_data = lambda data, fields: data

    stub_hass.data[diagnostics_module._DIAGNOSTICS_DATA] = SimpleNamespace(platforms={})

    def _register(hass: HomeAssistant, domain: str, platform: types.ModuleType) -> None:
        diagnostics_calls.append((domain, platform))
        hass.data[diagnostics_module._DIAGNOSTICS_DATA].platforms[domain] = platform

    diagnostics_module._register_diagnostics_platform = _register

    monkeypatch.setitem(
        sys.modules, "homeassistant.components.diagnostics", diagnostics_module
    )

    diagnostics_platform = importlib.import_module(
        "custom_components.termoweb.diagnostics"
    )

    diagnostics_module._register_diagnostics_platform(
        stub_hass, termoweb_init.DOMAIN, diagnostics_platform
    )

    assert diagnostics_calls == [(termoweb_init.DOMAIN, diagnostics_platform)]
    assert (
        stub_hass.data[diagnostics_module._DIAGNOSTICS_DATA].platforms[
            termoweb_init.DOMAIN
        ]
        is diagnostics_platform
    )
    assert hasattr(diagnostics_platform, "async_get_config_entry_diagnostics")


async def _drain_tasks(hass: HomeAssistant) -> None:
    if hass.tasks:
        await asyncio.gather(*hass.tasks, return_exceptions=True)
        hass.tasks.clear()


@pytest.fixture
def termoweb_init(monkeypatch: pytest.MonkeyPatch) -> Any:
    for name in list(sys.modules):
        if name.startswith("custom_components.termoweb"):
            sys.modules.pop(name)

    FakeCoordinator.instances.clear()
    module = importlib.import_module("custom_components.termoweb.__init__")
    module = importlib.reload(module)
    monkeypatch.setattr(module, "StateCoordinator", FakeCoordinator)
    ws_module = importlib.import_module(
        "custom_components.termoweb.backend.termoweb_ws"
    )
    ws_client_module = importlib.import_module(
        "custom_components.termoweb.backend.ws_client"
    )
    monkeypatch.setattr(ws_module, "TermoWebWSClient", FakeWSClient)
    monkeypatch.setattr(
        ws_client_module, "TermoWebWSClient", FakeWSClient, raising=False
    )
    backend_module = importlib.import_module(
        "custom_components.termoweb.backend.termoweb"
    )
    monkeypatch.setattr(backend_module, "TermoWebWSClient", FakeWSClient, raising=False)
    module._test_helpers = SimpleNamespace(
        fake_coordinator=FakeCoordinator,
        get_record=lambda hass, entry: hass.data[module.DOMAIN][entry.entry_id],
        get_ws_tasks=lambda hass, entry: hass.data[module.DOMAIN][entry.entry_id][
            "ws_tasks"
        ],
        get_ws_state=lambda hass, entry: hass.data[module.DOMAIN][entry.entry_id][
            "ws_state"
        ],
        get_recalc=lambda hass, entry: hass.data[module.DOMAIN][entry.entry_id][
            "recalc_poll"
        ],
    )
    return module


@pytest.fixture
def stub_hass() -> HomeAssistant:
    hass = HomeAssistant()
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


class _StubServices:
    def __init__(self) -> None:
        self._handlers: dict[tuple[str, str], Callable[[ServiceCall], Any]] = {}

    def has_service(self, domain: str, service: str) -> bool:
        return (domain, service) in self._handlers

    def async_register(
        self,
        domain: str,
        service: str,
        handler: Callable[[ServiceCall], Any],
    ) -> None:
        self._handlers[(domain, service)] = handler


@pytest.mark.asyncio
async def test_ws_debug_probe_service_handles_client_matrix(
    termoweb_init: Any,
) -> None:
    services = _StubServices()
    hass = SimpleNamespace(
        services=services,
        data={},
        loop=asyncio.get_running_loop(),
    )

    await termoweb_init.async_register_ws_debug_probe_service(hass)
    handler = services._handlers[(termoweb_init.DOMAIN, "ws_debug_probe")]

    hass.data[termoweb_init.DOMAIN] = ["invalid"]
    await handler(ServiceCall({}))

    hass.data[termoweb_init.DOMAIN] = {
        "other": {"debug": True, "ws_clients": {}},
    }
    await handler(ServiceCall({"entry_id": "missing"}))

    class TypeErrorClient:
        def debug_probe(self, _: str) -> None:
            return None

    class NonAwaitClient:
        def debug_probe(self) -> str:
            return "not-awaitable"

    class AsyncProbeClient:
        def __init__(self, *, should_fail: bool = False) -> None:
            self.should_fail = should_fail
            self.calls = 0

        def debug_probe(self) -> Coroutine[Any, Any, int]:
            self.calls += 1

            async def _runner() -> int:
                if self.should_fail:
                    raise RuntimeError("boom")
                return self.calls

            return _runner()

    class CancelledProbeClient:
        def __init__(self) -> None:
            self.calls = 0

        def debug_probe(self) -> Coroutine[Any, Any, None]:
            self.calls += 1

            async def _runner() -> None:
                raise asyncio.CancelledError

            return _runner()

    clients = {
        "missing": None,
        "no_probe": object(),
        "type_error": TypeErrorClient(),
        "nonawait": NonAwaitClient(),
        "ok": AsyncProbeClient(),
        "error": AsyncProbeClient(should_fail=True),
        "cancel": CancelledProbeClient(),
    }

    hass.data[termoweb_init.DOMAIN] = {
        "entry1": {"debug": False, "ws_clients": {}},
        "entry2": {"debug": True, "ws_clients": clients},
        "entry3": {"debug": True, "ws_clients": []},
    }

    await handler(ServiceCall({"entry_id": "entry2", "dev_id": "missing"}))

    with pytest.raises(asyncio.CancelledError):
        await handler(ServiceCall({}))

    clients["cancel"] = AsyncProbeClient()

    await handler(ServiceCall({}))

    assert clients["ok"].calls == 2
    assert clients["error"].calls == 2


def test_async_setup_entry_happy_path(
    termoweb_init: Any, stub_hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    class HappyClient(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            return [{"dev_id": "dev-1"}]

        async def get_nodes(self, dev_id: str) -> dict[str, Any]:
            assert dev_id == "dev-1"
            return {
                "nodes": [
                    {"addr": "A", "type": "htr"},
                    {"addr": "B", "type": "acm"},
                ]
            }

    monkeypatch.setattr(termoweb_init, "RESTClient", HappyClient)
    create_calls: list[tuple[Any, str, str, str]] = []
    orig_create = termoweb_init.create_rest_client

    def fake_create(
        hass_in: HomeAssistant, username: str, password: str, brand: str
    ) -> Any:
        create_calls.append((hass_in, username, password, brand))
        return orig_create(hass_in, username, password, brand)

    monkeypatch.setattr(termoweb_init, "create_rest_client", fake_create)

    list_calls: list[Any] = []
    orig_list_devices = termoweb_init.async_list_devices

    async def fake_async_list(client: Any) -> list[dict[str, Any]]:
        list_calls.append(client)
        return await orig_list_devices(client)

    monkeypatch.setattr(termoweb_init, "async_list_devices", fake_async_list)
    import_mock = AsyncMock()
    monkeypatch.setattr(termoweb_init, "_async_import_energy_history", import_mock)

    entry = ConfigEntry("happy", data={"username": "user", "password": "pw"})
    stub_hass.config_entries.add(entry)

    async def _run() -> bool:
        result = await termoweb_init.async_setup_entry(stub_hass, entry)
        await _drain_tasks(stub_hass)
        return result

    assert asyncio.run(_run()) is True
    assert create_calls == [(stub_hass, "user", "pw", termoweb_init.DEFAULT_BRAND)]
    assert len(list_calls) == 1
    assert isinstance(list_calls[0], HappyClient)

    record = stub_hass.data[termoweb_init.DOMAIN][entry.entry_id]
    assert isinstance(record["client"], HappyClient)
    assert isinstance(record["coordinator"], FakeCoordinator)
    assert record["coordinator"].refresh_calls == 1
    by_type, _ = build_heater_address_map(record["node_inventory"])
    assert by_type == {"htr": ["A"], "acm": ["B"]}
    assert [node.addr for node in record["node_inventory"]] == ["A", "B"]
    assert [node.type for node in record["node_inventory"]] == ["htr", "acm"]
    assert stub_hass.client_session_calls == 1
    assert stub_hass.config_entries.forwarded == [
        (entry, tuple(termoweb_init.PLATFORMS))
    ]
    assert stub_hass.services.has_service(termoweb_init.DOMAIN, "import_energy_history")
    assert import_mock.await_count == 0


def test_async_setup_entry_sets_supports_diagnostics(
    termoweb_init: Any, stub_hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    class DiagnosticsClient(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            return [{"dev_id": "dev-1"}]

    monkeypatch.setattr(termoweb_init, "RESTClient", DiagnosticsClient)

    sentinel = SimpleNamespace(YES=object())
    monkeypatch.setattr(termoweb_init, "SupportsDiagnostics", sentinel)

    entry = DiagnosticsConfigEntry("diag", data={"username": "user", "password": "pw"})
    stub_hass.config_entries.add(entry)

    assert entry.supports_diagnostics is None

    async def _run() -> bool:
        result = await termoweb_init.async_setup_entry(stub_hass, entry)
        await _drain_tasks(stub_hass)
        return result

    assert asyncio.run(_run()) is True
    assert entry.supports_diagnostics is sentinel.YES


def test_async_setup_entry_backfills_diagnostics_marker(
    termoweb_init: Any, stub_hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    class DiagnosticsClient(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            return [{"dev_id": "dev-1"}]

    monkeypatch.setattr(termoweb_init, "RESTClient", DiagnosticsClient)

    sentinel = SimpleNamespace(YES=object())
    monkeypatch.setattr(termoweb_init, "SupportsDiagnostics", sentinel)

    entry = DiagnosticsConfigEntry("legacy", data={"username": "user", "password": "pw"})
    stub_hass.config_entries.add(entry)

    assert "supports_diagnostics" not in entry.data

    async def _run() -> bool:
        result = await termoweb_init.async_setup_entry(stub_hass, entry)
        await _drain_tasks(stub_hass)
        return result

    assert asyncio.run(_run()) is True
    assert entry.supports_diagnostics is sentinel.YES
    assert entry.update_calls[-1]["supports_diagnostics"] is True
    assert entry.data["supports_diagnostics"] is True


def test_async_setup_entry_updates_via_hass_config_entries(
    termoweb_init: Any, stub_hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    class DiagnosticsClient(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            return [{"dev_id": "dev-1"}]

    monkeypatch.setattr(termoweb_init, "RESTClient", DiagnosticsClient)

    sentinel = SimpleNamespace(YES=object())
    monkeypatch.setattr(termoweb_init, "SupportsDiagnostics", sentinel)

    entry = DiagnosticsNoUpdateEntry("fallback", data={"username": "user", "password": "pw"})
    stub_hass.config_entries.add(entry)

    assert stub_hass.config_entries.updated_entries == []

    async def _run() -> bool:
        result = await termoweb_init.async_setup_entry(stub_hass, entry)
        await _drain_tasks(stub_hass)
        return result

    assert asyncio.run(_run()) is True
    assert entry.supports_diagnostics is sentinel.YES
    assert entry.data["supports_diagnostics"] is True
    assert stub_hass.config_entries.updated_entries[-1][1] == entry.data


def test_async_setup_entry_resets_diagnostics_cache(
    termoweb_init: Any,
    stub_hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class DiagnosticsClient(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            return [{"dev_id": "dev-1"}]

        async def get_nodes(self, dev_id: str) -> dict[str, Any]:
            await super().get_nodes(dev_id)
            return {"nodes": []}

    monkeypatch.setattr(termoweb_init, "RESTClient", DiagnosticsClient)

    loader_module = types.ModuleType("homeassistant.loader")
    missing_cache: dict[str, Any] = {
        "termoweb.diagnostics": object(),
        "other": object(),
    }
    loader_module.DATA_MISSING_PLATFORMS = missing_cache

    async def _async_get_integration(_hass: HomeAssistant, _domain: str) -> Any:
        return SimpleNamespace(version="1.0.0")

    loader_module.async_get_integration = _async_get_integration

    diagnostics_calls: list[tuple[HomeAssistant, str, types.ModuleType]] = []
    diagnostics_module = types.ModuleType("homeassistant.components.diagnostics")
    diagnostics_module.async_redact_data = lambda data, fields: data

    class _RegisterStub:
        def __init__(self) -> None:
            self.should_fail = False

        def __call__(
            self, hass: HomeAssistant, domain: str, module: types.ModuleType
        ) -> None:
            if self.should_fail:
                raise RuntimeError("boom")
            diagnostics_calls.append((hass, domain, module))

    register_stub = _RegisterStub()
    diagnostics_module._register_diagnostics_platform = register_stub
    monkeypatch.setitem(
        sys.modules, "homeassistant.components.diagnostics", diagnostics_module
    )

    import_mock = AsyncMock()
    monkeypatch.setattr(termoweb_init, "_async_import_energy_history", import_mock)

    entry = ConfigEntry("diag-cache", data={"username": "user", "password": "pw"})
    stub_hass.config_entries.add(entry)

    async def _run() -> bool:
        result = await termoweb_init.async_setup_entry(stub_hass, entry)
        await _drain_tasks(stub_hass)
        return result

    register_stub.should_fail = True
    monkeypatch.setattr(termoweb_init, "ha_loader", None)
    with caplog.at_level(logging.DEBUG, logger=termoweb_init.__name__):
        termoweb_init._register_diagnostics_platform(stub_hass)
        assert "Diagnostics cache unavailable" in caplog.text
        assert "Diagnostics registration raised exception" in caplog.text
    caplog.clear()

    register_stub.should_fail = False
    monkeypatch.setattr(termoweb_init, "ha_loader", loader_module)
    monkeypatch.setitem(sys.modules, "homeassistant.loader", loader_module)

    with caplog.at_level(logging.DEBUG, logger=termoweb_init.__name__):
        assert asyncio.run(_run()) is True
        assert "Attempting to remove termoweb.diagnostics" in caplog.text
        assert "Diagnostics platform registered successfully" in caplog.text
    caplog.clear()

    assert "termoweb.diagnostics" not in loader_module.DATA_MISSING_PLATFORMS
    assert diagnostics_calls and len(diagnostics_calls) == 1
    hass_call, domain_call, module_call = diagnostics_calls[0]
    assert hass_call is stub_hass
    assert domain_call == termoweb_init.DOMAIN
    assert module_call.__name__ == "custom_components.termoweb.diagnostics"

    register_stub.should_fail = True
    with caplog.at_level(logging.DEBUG, logger=termoweb_init.__name__):
        termoweb_init._register_diagnostics_platform(stub_hass)
        assert "No cache entry stored for termoweb.diagnostics" in caplog.text
        assert "Diagnostics registration raised exception" in caplog.text


@pytest.mark.asyncio
async def test_async_setup_entry_waits_for_delayed_diagnostics(
    termoweb_init: Any,
    stub_hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class DiagnosticsClient(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            return [{"dev_id": "dev-1"}]

        async def get_nodes(self, dev_id: str) -> dict[str, Any]:
            await super().get_nodes(dev_id)
            return {"nodes": []}

    monkeypatch.setattr(termoweb_init, "RESTClient", DiagnosticsClient)

    callbacks: list[Callable[[HomeAssistant, Any], Coroutine[Any, Any, None]]] = []

    def fake_async_when_setup(
        hass: HomeAssistant,
        component: str,
        callback: Callable[[HomeAssistant, Any], Coroutine[Any, Any, None]],
    ) -> Callable[[], None]:
        assert component == "diagnostics"
        callbacks.append(callback)

        def _unsub() -> None:
            if callback in callbacks:
                callbacks.remove(callback)

        return _unsub

    monkeypatch.setattr(termoweb_init, "async_when_setup", fake_async_when_setup)

    attempts: list[bool] = []

    def fake_register(_: HomeAssistant, __: types.ModuleType) -> bool:
        attempts.append(True)
        return len(attempts) >= 2

    monkeypatch.setattr(
        termoweb_init, "_register_diagnostics_platform", fake_register
    )

    diagnostics_module = types.ModuleType("custom_components.termoweb.diagnostics")
    monkeypatch.setitem(
        sys.modules, "custom_components.termoweb.diagnostics", diagnostics_module
    )

    monkeypatch.setattr(termoweb_init, "DIAGNOSTICS_RETRY_DELAY", 0)

    import_mock = AsyncMock()
    monkeypatch.setattr(termoweb_init, "_async_import_energy_history", import_mock)

    entry = ConfigEntry("diag-wait", data={"username": "user", "password": "pw"})
    stub_hass.config.components = set()
    stub_hass.config_entries.add(entry)

    with caplog.at_level(logging.DEBUG, logger=termoweb_init.__name__):
        assert await termoweb_init.async_setup_entry(stub_hass, entry)
        await asyncio.sleep(0)
        assert not attempts
        assert callbacks

        stub_hass.config.components.add("diagnostics")
        await callbacks[0](stub_hass, None)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        record = termoweb_init._test_helpers.get_record(stub_hass, entry)
        diag_task = record.get("diagnostics_task")
        assert diag_task is not None
        await diag_task

    assert len(attempts) == 2
    assert "Diagnostics component not loaded; waiting for setup" in caplog.text
    assert "Diagnostics registration attempt 1 starting" in caplog.text
    assert "Diagnostics registration attempt 2 finished with success=True" in caplog.text


def test_diagnostics_component_loaded_checks_sources(
    termoweb_init: Any, stub_hass: HomeAssistant
) -> None:
    stub_hass.config.components = {"diagnostics"}
    assert termoweb_init._diagnostics_component_loaded(stub_hass) is True

    stub_hass.config.components = ["other", "diagnostics"]
    assert termoweb_init._diagnostics_component_loaded(stub_hass) is True

    stub_hass.config.components = []
    stub_hass.data["components"] = {"diagnostics"}
    assert termoweb_init._diagnostics_component_loaded(stub_hass) is True

    stub_hass.data["components"] = {"diagnostics": object()}
    assert termoweb_init._diagnostics_component_loaded(stub_hass) is True

    stub_hass.data["components"] = {}
    assert termoweb_init._diagnostics_component_loaded(stub_hass) is False

    stub_hass.data.pop("components", None)
    assert termoweb_init._diagnostics_component_loaded(stub_hass) is False


@pytest.mark.asyncio
async def test_async_register_diagnostics_when_ready_component_loaded(
    termoweb_init: Any, stub_hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    diagnostics_module = types.ModuleType("custom_components.termoweb.diagnostics")
    monkeypatch.setitem(
        sys.modules, "custom_components.termoweb.diagnostics", diagnostics_module
    )

    calls: list[types.ModuleType] = []

    def fake_register(_: HomeAssistant, module: types.ModuleType) -> bool:
        calls.append(module)
        return True

    monkeypatch.setattr(termoweb_init, "_register_diagnostics_platform", fake_register)

    stub_hass.config.components = {"diagnostics"}

    with caplog.at_level(logging.DEBUG, logger=termoweb_init.__name__):
        await termoweb_init._async_register_diagnostics_when_ready(stub_hass)

    assert calls and calls[0] is diagnostics_module
    assert "Diagnostics component already loaded" in caplog.text


@pytest.mark.asyncio
async def test_async_register_diagnostics_when_ready_without_when_setup(
    termoweb_init: Any, stub_hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    diagnostics_module = types.ModuleType("custom_components.termoweb.diagnostics")
    monkeypatch.setitem(
        sys.modules, "custom_components.termoweb.diagnostics", diagnostics_module
    )

    calls: list[types.ModuleType] = []

    def fake_register(_: HomeAssistant, module: types.ModuleType) -> bool:
        calls.append(module)
        return True

    monkeypatch.setattr(termoweb_init, "_register_diagnostics_platform", fake_register)

    stub_hass.config.components = set()
    monkeypatch.setattr(termoweb_init, "async_when_setup", None)

    with caplog.at_level(logging.DEBUG, logger=termoweb_init.__name__):
        await termoweb_init._async_register_diagnostics_when_ready(stub_hass)

    assert calls and calls[0] is diagnostics_module
    assert "async_when_setup unavailable" in caplog.text


@pytest.mark.asyncio
async def test_async_register_diagnostics_when_ready_retries_import(
    termoweb_init: Any, stub_hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    diagnostics_module = types.ModuleType("custom_components.termoweb.diagnostics")

    import_calls = 0

    original_import = termoweb_init.import_module

    def fake_import(name: str) -> types.ModuleType:
        nonlocal import_calls
        if name == "custom_components.termoweb.diagnostics":
            import_calls += 1
            if import_calls == 1:
                raise ImportError("boom")
            return diagnostics_module
        return original_import(name)

    monkeypatch.setattr(termoweb_init, "import_module", fake_import)

    monkeypatch.setattr(termoweb_init, "DIAGNOSTICS_RETRY_DELAY", 0)

    def fake_register(_: HomeAssistant, module: types.ModuleType) -> bool:
        return True

    monkeypatch.setattr(termoweb_init, "_register_diagnostics_platform", fake_register)

    stub_hass.config.components = {"diagnostics"}

    with caplog.at_level(logging.DEBUG, logger=termoweb_init.__name__):
        await termoweb_init._async_register_diagnostics_when_ready(stub_hass)

    assert import_calls >= 2
    assert "Diagnostics module import failed" in caplog.text


@pytest.mark.asyncio
async def test_async_register_diagnostics_when_ready_handles_cancelled(
    termoweb_init: Any, stub_hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    diagnostics_module = types.ModuleType("custom_components.termoweb.diagnostics")
    monkeypatch.setitem(
        sys.modules, "custom_components.termoweb.diagnostics", diagnostics_module
    )

    def fake_register(_: HomeAssistant, __: types.ModuleType) -> bool:
        raise asyncio.CancelledError

    monkeypatch.setattr(termoweb_init, "_register_diagnostics_platform", fake_register)

    stub_hass.config.components = {"diagnostics"}

    with caplog.at_level(logging.DEBUG, logger=termoweb_init.__name__):
        with pytest.raises(asyncio.CancelledError):
            await termoweb_init._async_register_diagnostics_when_ready(stub_hass)

    assert "Diagnostics registration attempts cancelled" in caplog.text


def test_build_heater_address_map_filters_invalid_nodes(termoweb_init: Any) -> None:
    inventory = [
        SimpleNamespace(type="htr", addr="A"),
        SimpleNamespace(type="acm", addr=" "),
        SimpleNamespace(type="unknown", addr="B"),
        SimpleNamespace(type="pmo", addr=""),
    ]

    by_type, reverse = build_heater_address_map(inventory)

    assert by_type == {"htr": ["A"]}
    assert reverse == {"A": {"htr"}}


def test_async_setup_entry_auth_error(
    termoweb_init: Any, stub_hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    class AuthClient(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            raise termoweb_init.BackendAuthError("bad credentials")

    monkeypatch.setattr(termoweb_init, "RESTClient", AuthClient)
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
    stub_hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    error_case: str,
) -> None:
    class ErrorClient(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            if error_case == "timeout":
                raise TimeoutError("timeout")
            raise termoweb_init.BackendRateLimitError("rate limit")

    monkeypatch.setattr(termoweb_init, "RESTClient", ErrorClient)
    entry = ConfigEntry("transient", data={"username": "user", "password": "pw"})
    stub_hass.config_entries.add(entry)

    async def _run() -> None:
        await termoweb_init.async_setup_entry(stub_hass, entry)

    with pytest.raises(ConfigEntryNotReady):
        asyncio.run(_run())


def test_async_setup_entry_no_devices(
    termoweb_init: Any, stub_hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    class EmptyClient(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            return []

    monkeypatch.setattr(termoweb_init, "RESTClient", EmptyClient)
    entry = ConfigEntry("empty", data={"username": "user", "password": "pw"})
    stub_hass.config_entries.add(entry)

    async def _run() -> None:
        await termoweb_init.async_setup_entry(stub_hass, entry)

    with pytest.raises(ConfigEntryNotReady):
        asyncio.run(_run())


def test_async_setup_entry_skips_devices_without_identifier(
    termoweb_init: Any,
    stub_hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class PartialClient(BaseFakeClient):
        instances: list["PartialClient"] = []

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            type(self).instances.append(self)

        async def list_devices(self) -> list[dict[str, Any]]:
            return [
                "invalid",
                {"name": "No identifier"},
                {"id": " dev-2 ", "name": "Valid"},
            ]

        async def get_nodes(self, dev_id: str) -> dict[str, Any]:
            await super().get_nodes(dev_id)
            return {"nodes": []}

    monkeypatch.setattr(termoweb_init, "RESTClient", PartialClient)
    import_mock = AsyncMock()
    monkeypatch.setattr(termoweb_init, "_async_import_energy_history", import_mock)

    entry = ConfigEntry("partial", data={"username": "user", "password": "pw"})
    stub_hass.config_entries.add(entry)

    async def _run() -> None:
        await termoweb_init.async_setup_entry(stub_hass, entry)

    with caplog.at_level(logging.DEBUG):
        asyncio.run(_run())

    assert any(
        "Skipping device entry without identifier" in message
        for message in caplog.messages
    )
    assert FakeCoordinator.instances
    record = FakeCoordinator.instances[0]
    assert record.dev_id == "dev-2"
    assert PartialClient.instances[0].get_nodes_calls == ["dev-2"]


def test_async_setup_entry_rejects_all_devices_without_identifier(
    termoweb_init: Any,
    stub_hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class InvalidClient(BaseFakeClient):
        instances: list["InvalidClient"] = []

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            type(self).instances.append(self)

        async def list_devices(self) -> list[dict[str, Any]]:
            return [{"name": "Missing"}, {}]

    monkeypatch.setattr(termoweb_init, "RESTClient", InvalidClient)
    entry = ConfigEntry("invalid", data={"username": "user", "password": "pw"})
    stub_hass.config_entries.add(entry)

    async def _run() -> None:
        await termoweb_init.async_setup_entry(stub_hass, entry)

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ConfigEntryNotReady):
            asyncio.run(_run())

    assert any(
        "Skipping device entry without identifier" in message
        for message in caplog.messages
    )
    assert not FakeCoordinator.instances
    assert InvalidClient.instances[0].get_nodes_calls == []


def test_async_setup_entry_supports_mapping_devices_payload(
    termoweb_init: Any,
    stub_hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MappingClient(BaseFakeClient):
        instances: list["MappingClient"] = []

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            type(self).instances.append(self)

        async def list_devices(self) -> dict[str, Any]:
            return {"serial_id": " mapping-dev "}

        async def get_nodes(self, dev_id: str) -> dict[str, Any]:
            await super().get_nodes(dev_id)
            return {}

    monkeypatch.setattr(termoweb_init, "RESTClient", MappingClient)
    import_mock = AsyncMock()
    monkeypatch.setattr(termoweb_init, "_async_import_energy_history", import_mock)

    entry = ConfigEntry("mapping", data={"username": "user", "password": "pw"})
    stub_hass.config_entries.add(entry)

    async def _run() -> None:
        await termoweb_init.async_setup_entry(stub_hass, entry)

    asyncio.run(_run())

    assert FakeCoordinator.instances
    record = FakeCoordinator.instances[0]
    assert record.dev_id == "mapping-dev"
    assert MappingClient.instances[0].get_nodes_calls == ["mapping-dev"]


def test_async_setup_entry_logs_unexpected_devices_payload(
    termoweb_init: Any,
    stub_hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class WeirdClient(BaseFakeClient):
        instances: list["WeirdClient"] = []

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            type(self).instances.append(self)

        async def list_devices(self) -> Any:
            return "unexpected"

    monkeypatch.setattr(termoweb_init, "RESTClient", WeirdClient)
    entry = ConfigEntry("weird", data={"username": "user", "password": "pw"})
    stub_hass.config_entries.add(entry)

    async def _run() -> None:
        await termoweb_init.async_setup_entry(stub_hass, entry)

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ConfigEntryNotReady):
            asyncio.run(_run())

    assert any(
        "Unexpected list_devices payload" in message for message in caplog.messages
    )
    assert not FakeCoordinator.instances
    assert WeirdClient.instances[0].get_nodes_calls == []


def test_async_setup_entry_defers_until_started(
    termoweb_init: Any, stub_hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    class HappyClient(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            return [{"dev_id": "dev-1"}]

        async def get_nodes(self, dev_id: str) -> dict[str, Any]:
            return {"nodes": [{"addr": "A", "type": "htr"}]}

    monkeypatch.setattr(termoweb_init, "RESTClient", HappyClient)
    import_mock = AsyncMock()
    monkeypatch.setattr(termoweb_init, "_async_import_energy_history", import_mock)

    entry = ConfigEntry("startup", data={"username": "user", "password": "pw"})
    stub_hass.config_entries.add(entry)
    stub_hass.is_running = False

    async def _run() -> None:
        await termoweb_init.async_setup_entry(stub_hass, entry)
        await _drain_tasks(stub_hass)
        assert import_mock.await_count == 0
        assert all(
            event != EVENT_HOMEASSISTANT_STARTED
            for event, _ in stub_hass.bus.listeners
        )

    asyncio.run(_run())


def test_import_energy_history_service_invocation(
    termoweb_init: Any, stub_hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = StubEntityRegistry()
    energy_module = importlib.import_module("custom_components.termoweb.energy")
    monkeypatch.setattr(
        energy_module.er, "async_get", lambda hass: registry, raising=False
    )
    monkeypatch.setattr(
        entity_registry_mod, "async_get", lambda hass: registry, raising=False
    )

    class HappyClient(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            return [{"dev_id": "dev-1"}]

        async def get_nodes(self, dev_id: str) -> dict[str, Any]:
            return {
                "nodes": [
                    {"addr": "A", "type": "htr"},
                    {"addr": "B", "type": "acm"},
                ]
            }

    monkeypatch.setattr(termoweb_init, "RESTClient", HappyClient)
    import_mock = AsyncMock()
    monkeypatch.setattr(termoweb_init, "_async_import_energy_history", import_mock)

    entry = ConfigEntry("service", data={"username": "user", "password": "pw"})
    stub_hass.config_entries.add(entry)

    async def _run() -> None:
        assert await termoweb_init.async_setup_entry(stub_hass, entry)
        await _drain_tasks(stub_hass)
        import_mock.reset_mock()

        service = stub_hass.services.get(termoweb_init.DOMAIN, "import_energy_history")

        registry.add(
            "sensor.dev_a_energy",
            unique_id=build_heater_energy_unique_id("dev-1", "htr", "A"),
            platform=termoweb_init.DOMAIN,
            config_entry_id=entry.entry_id,
        )
        registry.add(
            "sensor.dev_b_energy",
            unique_id=build_heater_energy_unique_id("dev-1", "acm", "B"),
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
        assert args[2] == {"htr": ["A"], "acm": ["B"]}
        assert kwargs == {"reset_progress": True, "max_days": 10}

        import_mock.reset_mock()
        call_all = SimpleNamespace(data={"max_history_retrieval": 3})
        await service(call_all)
        assert import_mock.await_count == 1
        args, kwargs = import_mock.await_args
        assert args[0] is stub_hass
        assert args[1] is entry
        assert args[2] == {"htr": ["A"], "acm": ["B"]}
        assert kwargs == {"reset_progress": False, "max_days": 3}

    asyncio.run(_run())


def test_recalc_poll_interval_transitions(
    termoweb_init: Any, stub_hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    class PollClient(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            return [{"dev_id": "dev-1"}]

    monkeypatch.setattr(termoweb_init, "RESTClient", PollClient)
    entry = ConfigEntry("poll", data={"username": "user", "password": "pw"})
    stub_hass.config_entries.add(entry)

    async def _run() -> None:
        assert await termoweb_init.async_setup_entry(stub_hass, entry)
        await _drain_tasks(stub_hass)

        record = termoweb_init._test_helpers.get_record(stub_hass, entry)
        coordinator: FakeCoordinator = record["coordinator"]

        if record["ws_tasks"]:
            await asyncio.gather(*record["ws_tasks"].values(), return_exceptions=True)
        record["ws_tasks"].clear()
        record["ws_state"].clear()

        base_interval = record["base_poll_interval"]

        # (a) No running tasks with stretched=True restores base interval
        record["stretched"] = True
        coordinator.update_interval = timedelta(seconds=999)
        record["recalc_poll"]()
        assert record["stretched"] is False
        assert coordinator.update_interval == timedelta(seconds=base_interval)

        # (b) All healthy tasks stretch polling interval
        healthy_event = asyncio.Event()
        healthy_task = asyncio.create_task(healthy_event.wait())
        record["ws_tasks"]["dev-healthy"] = healthy_task
        record["ws_state"]["dev-healthy"] = {"status": "healthy"}
        record["stretched"] = False
        coordinator.update_interval = timedelta(seconds=base_interval)
        record["recalc_poll"]()
        assert record["stretched"] is True
        assert coordinator.update_interval == timedelta(
            seconds=termoweb_init.STRETCHED_POLL_INTERVAL
        )
        healthy_event.set()
        await healthy_task

        # (c) Unhealthy status reverts stretched polling
        record["ws_tasks"].clear()
        record["ws_state"].clear()
        unhealthy_event = asyncio.Event()
        unhealthy_task = asyncio.create_task(unhealthy_event.wait())
        record["ws_tasks"]["dev-bad"] = unhealthy_task
        record["ws_state"]["dev-bad"] = {"status": "degraded"}
        record["stretched"] = True
        coordinator.update_interval = timedelta(
            seconds=termoweb_init.STRETCHED_POLL_INTERVAL
        )
        record["recalc_poll"]()
        assert record["stretched"] is False
        assert coordinator.update_interval == timedelta(seconds=base_interval)
        unhealthy_event.set()
        await unhealthy_task

    asyncio.run(_run())


def test_ws_status_dispatcher_filters_entry(
    termoweb_init: Any, stub_hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    class DispatchClient(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            return [{"dev_id": "dev-1"}]

    monkeypatch.setattr(termoweb_init, "RESTClient", DispatchClient)
    entry1 = ConfigEntry("dispatch1", data={"username": "user", "password": "pw"})
    entry2 = ConfigEntry("dispatch2", data={"username": "user", "password": "pw"})
    stub_hass.config_entries.add(entry1)
    stub_hass.config_entries.add(entry2)

    async def _run() -> None:
        assert await termoweb_init.async_setup_entry(stub_hass, entry1)
        await _drain_tasks(stub_hass)
        assert await termoweb_init.async_setup_entry(stub_hass, entry2)
        await _drain_tasks(stub_hass)

        record1 = termoweb_init._test_helpers.get_record(stub_hass, entry1)
        coordinator1: FakeCoordinator = record1["coordinator"]
        base_interval = record1["base_poll_interval"]

        callbacks = {
            signal: callback for signal, callback in stub_hass.dispatcher_connections
        }
        cb1 = callbacks[termoweb_init.signal_ws_status(entry1.entry_id)]
        cb2 = callbacks[termoweb_init.signal_ws_status(entry2.entry_id)]

        if record1["ws_tasks"]:
            await asyncio.gather(*record1["ws_tasks"].values(), return_exceptions=True)
        record1["ws_tasks"].clear()
        record1["ws_state"].clear()

        # Matching payload triggers recalc for entry1
        healthy_event = asyncio.Event()
        healthy_task = asyncio.create_task(healthy_event.wait())
        record1["ws_tasks"]["dev-1"] = healthy_task
        record1["ws_state"]["dev-1"] = {"status": "healthy"}
        record1["stretched"] = False
        coordinator1.update_interval = timedelta(seconds=base_interval)
        cb1({"entry_id": entry1.entry_id})
        assert record1["stretched"] is True
        assert coordinator1.update_interval == timedelta(
            seconds=termoweb_init.STRETCHED_POLL_INTERVAL
        )
        healthy_event.set()
        await healthy_task

        # Mismatching payload (other entry callback) does not affect entry1
        record1["ws_tasks"].clear()
        record1["ws_state"].clear()
        other_event = asyncio.Event()
        other_task = asyncio.create_task(other_event.wait())
        record1["ws_tasks"]["dev-1"] = other_task
        record1["ws_state"]["dev-1"] = {"status": "healthy"}
        record1["stretched"] = False
        coordinator1.update_interval = timedelta(seconds=base_interval)
        cb2({"entry_id": entry1.entry_id})
        assert record1["stretched"] is False
        assert coordinator1.update_interval == timedelta(seconds=base_interval)
        other_event.set()
        await other_task

    asyncio.run(_run())


def test_coordinator_listener_starts_new_ws(
    termoweb_init: Any, stub_hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    start_events: list[asyncio.Event] = []

    class SlowWSClient(FakeWSClient):
        def start(self) -> asyncio.Task[Any]:
            event = asyncio.Event()
            start_events.append(event)
            task = asyncio.create_task(event.wait())
            self.start_calls.append(task)
            return task

    class ListenerClient(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            return [{"dev_id": "dev-1"}]

    monkeypatch.setattr(termoweb_init, "RESTClient", ListenerClient)
    ws_module = importlib.import_module(
        "custom_components.termoweb.backend.termoweb_ws"
    )
    ws_client_module = importlib.import_module(
        "custom_components.termoweb.backend.ws_client"
    )
    monkeypatch.setattr(ws_module, "TermoWebWSClient", SlowWSClient)
    monkeypatch.setattr(
        ws_client_module, "TermoWebWSClient", SlowWSClient, raising=False
    )
    backend_module = importlib.import_module(
        "custom_components.termoweb.backend.termoweb"
    )
    monkeypatch.setattr(backend_module, "TermoWebWSClient", SlowWSClient, raising=False)
    entry = ConfigEntry("listener", data={"username": "user", "password": "pw"})
    stub_hass.config_entries.add(entry)

    async def _run() -> None:
        assert await termoweb_init.async_setup_entry(stub_hass, entry)
        await _drain_tasks(stub_hass)

        record = termoweb_init._test_helpers.get_record(stub_hass, entry)
        coordinator: FakeCoordinator = record["coordinator"]

        existing_task = record["ws_tasks"].get("dev-1")
        assert isinstance(existing_task, asyncio.Task)
        assert not existing_task.done()

        stub_hass.tasks.clear()
        coordinator.data = dict(coordinator.data)
        coordinator.data["dev-2"] = {"dev_id": "dev-2"}
        assert coordinator.listeners
        listener = coordinator.listeners[0]

        listener()
        assert len(stub_hass.tasks) == 1
        await _drain_tasks(stub_hass)
        assert set(record["ws_tasks"]) == {"dev-1", "dev-2"}
        assert record["ws_tasks"]["dev-1"] is existing_task
        assert isinstance(record["ws_tasks"]["dev-2"], asyncio.Task)
        assert not record["ws_tasks"]["dev-2"].done()

        listener()
        assert not stub_hass.tasks

        for event in start_events:
            event.set()
        await asyncio.gather(*record["ws_tasks"].values(), return_exceptions=True)

    asyncio.run(_run())


def test_import_energy_history_service_error_logging(
    termoweb_init: Any, stub_hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = StubEntityRegistry()
    energy_module = importlib.import_module("custom_components.termoweb.energy")
    monkeypatch.setattr(
        energy_module.er, "async_get", lambda hass: registry, raising=False
    )
    monkeypatch.setattr(
        entity_registry_mod, "async_get", lambda hass: registry, raising=False
    )

    class ServiceClient(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            return [{"dev_id": "dev-1"}]

    async def failing_import(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("boom")

    log_calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def capture_exception(msg: str, *args: Any, **kwargs: Any) -> None:
        log_calls.append((msg, args, kwargs))

    monkeypatch.setattr(termoweb_init, "RESTClient", ServiceClient)
    monkeypatch.setattr(termoweb_init, "_async_import_energy_history", failing_import)
    monkeypatch.setattr(termoweb_init._LOGGER, "exception", capture_exception)

    entry1 = ConfigEntry("svc1", data={"username": "user", "password": "pw"})
    entry2 = ConfigEntry("svc2", data={"username": "user", "password": "pw"})
    stub_hass.config_entries.add(entry1)
    stub_hass.config_entries.add(entry2)

    async def _run() -> None:
        assert await termoweb_init.async_setup_entry(stub_hass, entry1)
        await _drain_tasks(stub_hass)
        service = stub_hass.services.get(termoweb_init.DOMAIN, "import_energy_history")
        assert service is not None

        assert stub_hass.services.has_service(
            termoweb_init.DOMAIN, "import_energy_history"
        )
        assert await termoweb_init.async_setup_entry(stub_hass, entry2)
        await _drain_tasks(stub_hass)
        assert (
            stub_hass.services.get(termoweb_init.DOMAIN, "import_energy_history")
            is service
        )

        registry.add(
            "sensor.svc1_energy",
            unique_id=build_heater_energy_unique_id("dev-1", "htr", "A"),
            platform=termoweb_init.DOMAIN,
            config_entry_id=entry1.entry_id,
        )
        registry.add(
            "sensor.svc2_energy",
            unique_id=build_heater_energy_unique_id("dev-1", "htr", "B"),
            platform=termoweb_init.DOMAIN,
            config_entry_id=entry2.entry_id,
        )

        call = SimpleNamespace(
            data={"entity_id": ["sensor.svc1_energy", "sensor.svc2_energy"]}
        )
        await service(call)
        assert len(log_calls) == 2


def test_import_energy_history_service_logs_global_task_errors(
    termoweb_init: Any, stub_hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ServiceClient(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            return [{"dev_id": "dev-1"}]

    async def failing_import(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("task boom")

    log_calls: list[str] = []

    def capture_exception(msg: str, *args: Any, **kwargs: Any) -> None:
        log_calls.append(msg % args if args else msg)

    monkeypatch.setattr(termoweb_init, "RESTClient", ServiceClient)
    monkeypatch.setattr(termoweb_init, "_async_import_energy_history", failing_import)
    energy_mod = importlib.import_module("custom_components.termoweb.energy")
    monkeypatch.setattr(energy_mod._LOGGER, "exception", capture_exception)

    entry = ConfigEntry("svc-global", data={"username": "user", "password": "pw"})
    stub_hass.config_entries.add(entry)

    async def _run() -> None:
        assert await termoweb_init.async_setup_entry(stub_hass, entry)
        await _drain_tasks(stub_hass)

        service = stub_hass.services.get(termoweb_init.DOMAIN, "import_energy_history")
        assert service is not None

        await service(SimpleNamespace(data={}))

    asyncio.run(_run())

    assert any("task failed" in msg for msg in log_calls)


def test_import_energy_history_service_logs_entry_task_exception(
    termoweb_init: Any,
    stub_hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class ServiceClient(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            return [{"dev_id": "dev-1"}]

    async def failing_import(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("entry task boom")

    monkeypatch.setattr(termoweb_init, "RESTClient", ServiceClient)
    monkeypatch.setattr(termoweb_init, "_async_import_energy_history", failing_import)

    entry = ConfigEntry("svc-entry", data={"username": "user", "password": "pw"})
    stub_hass.config_entries.add(entry)

    caplog.set_level(logging.ERROR, logger=termoweb_init.__name__)

    async def _run() -> None:
        assert await termoweb_init.async_setup_entry(stub_hass, entry)
        await _drain_tasks(stub_hass)

        service = stub_hass.services.get(termoweb_init.DOMAIN, "import_energy_history")
        assert service is not None

        await service(SimpleNamespace(data={}))

    asyncio.run(_run())

    assert "import_energy_history task failed" in caplog.text


def test_start_ws_skips_when_task_running(
    termoweb_init: Any, stub_hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    class HappyClient(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            return [{"dev_id": "dev-1"}]

    monkeypatch.setattr(termoweb_init, "RESTClient", HappyClient)
    entry = ConfigEntry("skip", data={"username": "user", "password": "pw"})
    stub_hass.config_entries.add(entry)

    async def _run() -> None:
        assert await termoweb_init.async_setup_entry(stub_hass, entry)
        await _drain_tasks(stub_hass)

        record = termoweb_init._test_helpers.get_record(stub_hass, entry)
        coordinator: FakeCoordinator = record["coordinator"]
        assert coordinator.listeners
        listener = coordinator.listeners[0]

        start_ws = None
        if listener.__closure__:
            for cell in listener.__closure__:
                candidate = cell.cell_contents
                if (
                    callable(candidate)
                    and getattr(candidate, "__name__", "") == "_start_ws"
                ):
                    start_ws = candidate
                    break
        assert start_ws is not None

        existing = record["ws_tasks"].get("dev-1")
        if existing:
            await existing

        blocker = asyncio.Event()
        pending = asyncio.create_task(blocker.wait())
        record["ws_tasks"]["dev-1"] = pending

        await start_ws("dev-1")
        assert record["ws_tasks"]["dev-1"] is pending

        blocker.set()
        await pending

    asyncio.run(_run())


def test_import_energy_history_service_handles_string_ids_and_cancelled(
    termoweb_init: Any, stub_hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = StubEntityRegistry()
    energy_module = importlib.import_module("custom_components.termoweb.energy")
    monkeypatch.setattr(
        energy_module.er, "async_get", lambda hass: registry, raising=False
    )
    monkeypatch.setattr(
        entity_registry_mod, "async_get", lambda hass: registry, raising=False
    )

    class ServiceClient(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            return [{"dev_id": "dev-1"}]

        async def get_nodes(self, dev_id: str) -> dict[str, Any]:
            return {"nodes": [{"addr": "A", "type": "htr"}]}

    cancel_import = AsyncMock(side_effect=asyncio.CancelledError())
    monkeypatch.setattr(termoweb_init, "RESTClient", ServiceClient)
    monkeypatch.setattr(termoweb_init, "_async_import_energy_history", cancel_import)

    entry = ConfigEntry("svc", data={"username": "user", "password": "pw"})
    stub_hass.config_entries.add(entry)

    async def _run() -> None:
        assert await termoweb_init.async_setup_entry(stub_hass, entry)
        await _drain_tasks(stub_hass)
        cancel_import.reset_mock()
        service = stub_hass.services.get(termoweb_init.DOMAIN, "import_energy_history")
        assert service is not None

        registry.add(
            "sensor.invalid",
            unique_id=build_heater_energy_unique_id("dev-1", "htr", "A"),
            platform="other",
            config_entry_id=entry.entry_id,
        )
        registry.add(
            "sensor.valid",
            unique_id=build_heater_energy_unique_id("dev-1", "htr", "A"),
            platform=termoweb_init.DOMAIN,
            config_entry_id=entry.entry_id,
        )

        with pytest.raises(asyncio.CancelledError):
            await service(SimpleNamespace(data={"entity_id": "sensor.valid"}))
        assert cancel_import.await_count == 1
        cancel_import.reset_mock()

        await service(SimpleNamespace(data={"entity_id": ["sensor.invalid"]}))
        assert cancel_import.await_count == 0

    asyncio.run(_run())


def test_async_unload_entry_handles_task_and_client_errors(
    termoweb_init: Any, stub_hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    class HappyClient(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            return [{"dev_id": "dev-1"}]

    monkeypatch.setattr(termoweb_init, "RESTClient", HappyClient)
    entry = ConfigEntry("unload", data={"username": "user", "password": "pw"})
    stub_hass.config_entries.add(entry)

    async def _run() -> None:
        assert await termoweb_init.async_setup_entry(stub_hass, entry)
        await _drain_tasks(stub_hass)

        record = termoweb_init._test_helpers.get_record(stub_hass, entry)

        class BadTask:
            def cancel(self) -> None:
                return None

            def __await__(self):
                async def _raise() -> None:
                    raise RuntimeError("task fail")

                return _raise().__await__()

        class BadClient:
            async def stop(self) -> None:
                raise RuntimeError("client fail")

        record["ws_tasks"]["dev-1"] = BadTask()
        record["ws_clients"]["dev-1"] = BadClient()

        log_calls: list[str] = []

        def capture_exception(msg: str, *args: Any, **kwargs: Any) -> None:
            log_calls.append(msg)

        monkeypatch.setattr(termoweb_init._LOGGER, "exception", capture_exception)

        assert await termoweb_init.async_unload_entry(stub_hass, entry)
        assert log_calls
        assert entry.entry_id not in stub_hass.data.get(termoweb_init.DOMAIN, {})

    asyncio.run(_run())


def test_async_setup_entry_cleans_up_on_hass_stop(
    termoweb_init: Any, stub_hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    class HappyClient(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            return [{"dev_id": "dev-stop"}]

    monkeypatch.setattr(termoweb_init, "RESTClient", HappyClient)
    entry = ConfigEntry("stop", data={"username": "user", "password": "pw"})
    stub_hass.config_entries.add(entry)

    async def _run() -> tuple[int, bool, bool]:
        assert await termoweb_init.async_setup_entry(stub_hass, entry)
        await _drain_tasks(stub_hass)

        listeners = [
            cb
            for event, cb in stub_hass.bus.listeners
            if event == termoweb_init.EVENT_HOMEASSISTANT_STOP
        ]
        assert listeners

        record = termoweb_init._test_helpers.get_record(stub_hass, entry)
        ws_task = next(iter(record["ws_tasks"].values()))
        client = next(iter(record["ws_clients"].values()))

        await listeners[0](None)

        return (
            client.stop_calls,
            ws_task.cancelled() or ws_task.done(),
            record.get("_shutdown_complete", False),
        )

    stop_calls, cancelled, shutdown_flag = asyncio.run(_run())
    assert stop_calls == 1
    assert cancelled is True
    assert shutdown_flag is True


def test_async_unload_entry_cleans_up(
    termoweb_init: Any, stub_hass: HomeAssistant
) -> None:
    entry = ConfigEntry("unload", data={})
    stub_hass.config_entries.add(entry)

    async def _run() -> tuple[bool, list[bool], int, bool, list[bool]]:
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
        return (
            result,
            cancel_events,
            client.stop_calls,
            ws_task.cancelled(),
            unsubscribed,
        )

    result, cancel_events, stop_calls, task_cancelled, unsubscribed = asyncio.run(
        _run()
    )
    assert result is True
    assert cancel_events == [True]
    assert stop_calls == 1
    assert task_cancelled is True
    assert stub_hass.config_entries.unloaded == [
        (entry, tuple(termoweb_init.PLATFORMS))
    ]
    assert entry.entry_id not in stub_hass.data.get(termoweb_init.DOMAIN, {})


def test_async_update_entry_options_recalculates_poll(
    termoweb_init: Any, stub_hass: HomeAssistant
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


def test_async_unload_entry_missing_returns_true(
    termoweb_init: Any, stub_hass: HomeAssistant
) -> None:
    entry = ConfigEntry("missing", data={})
    stub_hass.config_entries.add(entry)
    assert asyncio.run(termoweb_init.async_unload_entry(stub_hass, entry)) is True


def test_async_migrate_entry_returns_true(
    termoweb_init: Any, stub_hass: HomeAssistant
) -> None:
    entry = ConfigEntry("migrate", data={})
    stub_hass.config_entries.add(entry)
    assert asyncio.run(termoweb_init.async_migrate_entry(stub_hass, entry)) is True


@pytest.mark.asyncio
async def test_shutdown_entry_ignores_non_mapping(termoweb_init: Any) -> None:
    await termoweb_init._async_shutdown_entry(object())


@pytest.mark.asyncio
async def test_shutdown_entry_skips_completed_record(termoweb_init: Any) -> None:
    rec: dict[str, object] = {"_shutdown_complete": True, "ws_clients": {}}
    await termoweb_init._async_shutdown_entry(rec)
    assert rec["_shutdown_complete"] is True


@pytest.mark.asyncio
async def test_shutdown_entry_handles_client_without_stop(termoweb_init: Any) -> None:
    rec: dict[str, object] = {"ws_clients": {"dev": object()}}
    await termoweb_init._async_shutdown_entry(rec)
    assert rec["_shutdown_complete"] is True
