# ruff: noqa: D100,D101,D102,D103,D104,D105,D106,D107,INP001,E402
from __future__ import annotations

import asyncio
import importlib
import logging
import sys
from datetime import timedelta
from types import SimpleNamespace
from typing import Any, Callable, Coroutine, Mapping

import pytest
from unittest.mock import AsyncMock

from conftest import FakeCoordinator, _install_stubs, build_entry_runtime

_install_stubs()

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EVENT_HOMEASSISTANT_STARTED
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from homeassistant.helpers import entity_registry as entity_registry_mod

from custom_components.termoweb.backend.ws_health import WsHealthTracker
from custom_components.termoweb.identifiers import build_heater_energy_unique_id
from custom_components.termoweb.inventory import build_heater_address_map
import custom_components.termoweb.backend.ducaheat as ducaheat_module
import custom_components.termoweb.backend.factory as backend_factory
import custom_components.termoweb.const as const_module
import custom_components.termoweb.inventory as inventory_module


class FakeWSClient:
    def __init__(
        self,
        hass: HomeAssistant,
        *,
        entry_id: str,
        dev_id: str,
        api_client: Any,
        coordinator: Any,
        inventory: Any | None = None,
    ) -> None:
        self.hass = hass
        self.entry_id = entry_id
        self.dev_id = dev_id
        self.api_client = api_client
        self.coordinator = coordinator
        self.inventory = inventory
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

    async def async_close(self) -> None:
        self.closed = True


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

    monkeypatch.setattr(backend_factory, "RESTClient", DefaultClient)
    monkeypatch.setattr(ducaheat_module, "DucaheatRESTClient", DucaClient)

    default_client = backend_factory.create_rest_client(
        stub_hass,
        "user",
        "pw",
        termoweb_init.DEFAULT_BRAND,
    )
    duca_client = backend_factory.create_rest_client(
        stub_hass,
        "user2",
        "pw2",
        termoweb_init.BRAND_DUCAHEAT,
    )
    tevolve_client = backend_factory.create_rest_client(
        stub_hass,
        "user3",
        "pw3",
        termoweb_init.BRAND_TEVOLVE,
    )

    assert isinstance(default_client, DefaultClient)
    assert isinstance(duca_client, DucaClient)
    assert isinstance(tevolve_client, DucaClient)
    assert default_client.api_base == const_module.get_brand_api_base(
        termoweb_init.DEFAULT_BRAND
    )
    assert tevolve_client.api_base == const_module.get_brand_api_base(
        termoweb_init.BRAND_TEVOLVE
    )
    assert duca_client.basic_auth_b64 == const_module.get_brand_basic_auth(
        termoweb_init.BRAND_DUCAHEAT
    )
    assert tevolve_client.basic_auth_b64 == const_module.get_brand_basic_auth(
        termoweb_init.BRAND_TEVOLVE
    )
    assert stub_hass.client_session_calls == 3


async def _drain_tasks(hass: HomeAssistant) -> None:
    if hass.tasks:
        await asyncio.gather(*hass.tasks, return_exceptions=True)
        hass.tasks.clear()


@pytest.fixture
def termoweb_init(monkeypatch: pytest.MonkeyPatch) -> Any:
    for name in list(sys.modules):
        if name.startswith("custom_components.termoweb") and name not in {
            "custom_components.termoweb.inventory",
            "custom_components.termoweb.runtime",
        }:
            sys.modules.pop(name)

    FakeCoordinator.instances.clear()
    module = importlib.import_module("custom_components.termoweb.__init__")
    module = importlib.reload(module)
    globals()["backend_factory"] = importlib.import_module(
        "custom_components.termoweb.backend.factory"
    )
    globals()["ducaheat_module"] = importlib.import_module(
        "custom_components.termoweb.backend.ducaheat"
    )
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
        get_record=lambda hass, entry: importlib.import_module(
            "custom_components.termoweb.runtime"
        ).require_runtime(hass, entry.entry_id),
        get_ws_tasks=lambda hass, entry: importlib.import_module(
            "custom_components.termoweb.runtime"
        )
        .require_runtime(hass, entry.entry_id)
        .ws_tasks,
        get_ws_state=lambda hass, entry: importlib.import_module(
            "custom_components.termoweb.runtime"
        )
        .require_runtime(hass, entry.entry_id)
        .ws_state,
        get_recalc=lambda hass, entry: importlib.import_module(
            "custom_components.termoweb.runtime"
        )
        .require_runtime(hass, entry.entry_id)
        .recalc_poll,
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

    monkeypatch.setattr(backend_factory, "RESTClient", HappyClient)
    create_calls: list[tuple[Any, str, str, str]] = []
    orig_create = backend_factory.create_rest_client

    def fake_create(
        hass_in: HomeAssistant, username: str, password: str, brand: str
    ) -> Any:
        create_calls.append((hass_in, username, password, brand))
        return orig_create(hass_in, username, password, brand)

    monkeypatch.setattr(backend_factory, "create_rest_client", fake_create)

    list_calls: list[Any] = []
    orig_list_devices = termoweb_init.async_list_devices

    async def fake_async_list(client: Any) -> list[dict[str, Any]]:
        list_calls.append(client)
        return await orig_list_devices(client)

    monkeypatch.setattr(termoweb_init, "async_list_devices", fake_async_list)

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

    record = termoweb_init._test_helpers.get_record(stub_hass, entry)
    assert isinstance(record.client, HappyClient)
    assert isinstance(record.coordinator, FakeCoordinator)
    assert record.coordinator.refresh_calls == 1
    assert record.inventory is not None
    inventory_module = importlib.import_module("custom_components.termoweb.inventory")
    assert isinstance(record.inventory, inventory_module.Inventory)
    assert record.inventory.dev_id == "dev-1"
    assert record.coordinator.inventory is record.inventory
    node_list = list(record.inventory.nodes)
    assert node_list
    by_type, _ = build_heater_address_map(node_list)
    assert by_type == {"htr": ["A"], "acm": ["B"]}
    assert [node.addr for node in node_list] == ["A", "B"]
    assert [node.type for node in node_list] == ["htr", "acm"]
    assert not hasattr(record, "node_inventory")
    assert stub_hass.client_session_calls == 1
    assert stub_hass.config_entries.forwarded == [
        (entry, tuple(termoweb_init.PLATFORMS))
    ]


def test_async_setup_entry_logs_unknown_node_types_without_probing(
    termoweb_init: Any,
    stub_hass: HomeAssistant,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown node types (not thm) are logged at DEBUG; no extra probe GETs."""

    class ProbeClient(BaseFakeClient):
        instances: list["ProbeClient"] = []

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.probe_calls: list[str] = []
            ProbeClient.instances.append(self)

        async def list_devices(self) -> list[dict[str, Any]]:
            return [{"dev_id": "dev-1"}]

        async def get_nodes(self, dev_id: str) -> dict[str, Any]:
            await super().get_nodes(dev_id)
            return {
                "nodes": [
                    {"addr": "9", "type": "foo"},
                    {"addr": "9", "type": "foo"},
                    {"addr": "1", "type": "htr"},
                    {"addr": "2", "type": "thm"},
                ]
            }

        async def authed_headers(self) -> Mapping[str, str]:
            return {"Authorization": "Bearer token"}

        async def get_node_samples(self, *_args: Any, **_kwargs: Any) -> list[Any]:
            return []

        async def debug_probe_get(self, path: str, **_kwargs: Any) -> None:
            self.probe_calls.append(path)

    monkeypatch.setattr(backend_factory, "RESTClient", ProbeClient)

    entry = ConfigEntry("probe", data={"username": "user", "password": "pw"})
    stub_hass.config_entries.add(entry)

    caplog.set_level(logging.DEBUG, logger="custom_components.termoweb")

    async def _run() -> bool:
        result = await termoweb_init.async_setup_entry(stub_hass, entry)
        await _drain_tasks(stub_hass)
        return result

    assert asyncio.run(_run()) is True

    assert ProbeClient.instances, "REST client was not instantiated"
    assert ProbeClient.instances[0].probe_calls == []
    unknown = [
        record.message
        for record in caplog.records
        if record.name.startswith("custom_components.termoweb")
        and record.message.startswith("Unknown node type found")
    ]
    assert unknown == ["Unknown node type found: foo/9"]


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

    monkeypatch.setattr(backend_factory, "RESTClient", AuthClient)
    entry = ConfigEntry("auth", data={"username": "user", "password": "pw"})
    stub_hass.config_entries.add(entry)

    async def _run() -> None:
        try:
            await termoweb_init.async_setup_entry(stub_hass, entry)
        finally:
            await _drain_tasks(stub_hass)

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

    monkeypatch.setattr(backend_factory, "RESTClient", ErrorClient)
    entry = ConfigEntry("transient", data={"username": "user", "password": "pw"})
    stub_hass.config_entries.add(entry)

    async def _run() -> None:
        await termoweb_init.async_setup_entry(stub_hass, entry)

    with pytest.raises(ConfigEntryNotReady):
        asyncio.run(_run())


@pytest.mark.parametrize("error_case", ["link", "radio"])
def test_async_setup_entry_radio_gateway_unreachable(
    termoweb_init: Any,
    stub_hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    error_case: str,
) -> None:
    created: list[tuple[Any, ...]] = []

    class UnreachableRadio(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            if error_case == "link":
                raise termoweb_init.RadioLinkError("cannot connect")
            raise termoweb_init.RadioError("no MAC")

    def fake_create(*args: Any, power: Any = None) -> Any:
        created.append((*args, power))
        return UnreachableRadio(None, "", "")

    monkeypatch.setattr(backend_factory, "create_radio_client", fake_create)
    nodes = [{"type": "htr", "addr": "6", "name": "Heater 6"}]
    entry = ConfigEntry(
        "radio",
        data={
            "brand": "radio",
            "host": "10.0.0.5",
            "port": "2323",
            "dialect": "B",
            "network_id": "1234",
            "nodes": nodes,
        },
    )
    stub_hass.config_entries.add(entry)

    async def _run() -> None:
        await termoweb_init.async_setup_entry(stub_hass, entry)

    with pytest.raises(ConfigEntryNotReady):
        asyncio.run(_run())
    assert [args[:5] for args in created] == [
        ("10.0.0.5", 2323, "B", nodes, bytes.fromhex("1234"))
    ]
    power = created[0][5]
    power.set_power_limit(1800)  # saved into the entry options
    assert entry.options["radio_power"]["power_limit"] == 1800
    assert power.power_limit == 1800


def test_async_setup_entry_nanocul_builds_a_serial_client(
    termoweb_init: Any,
    stub_hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    class UnreachableStick(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            raise termoweb_init.RadioLinkError("unplugged")

    def fake_create(*args: Any, **kwargs: Any) -> Any:
        created.append((args, kwargs))
        return UnreachableStick(None, "", "")

    monkeypatch.setattr(backend_factory, "create_radio_client", fake_create)
    entry = ConfigEntry(
        "nanocul",
        data={
            "brand": "radio",
            "radio_type": "nanocul",
            "device": "/dev/ttyUSB0",
            "radio_device_id": "nanocul-a1b2c3",
            "dialect": "A",
            "network_id": "1B30",
            "nodes": [],
        },
    )
    stub_hass.config_entries.add(entry)

    async def _run() -> None:
        await termoweb_init.async_setup_entry(stub_hass, entry)

    with pytest.raises(ConfigEntryNotReady):
        asyncio.run(_run())
    ((args, kwargs),) = created
    assert args[:2] == ("/dev/ttyUSB0", 0)
    assert kwargs["serial_url"] == "/dev/ttyUSB0"
    assert kwargs["device_id"] == "nanocul-a1b2c3"


MONITOR_ESP32 = {"brand": "radio_monitor", "host": "10.0.0.5", "port": 2323}
MONITOR_STICK = {
    "brand": "radio_monitor",
    "radio_type": "nanocul",
    "device": "socket://10.0.0.7:5000",
    "radio_device_id": "nanocul-a1b2c3",
}


@pytest.mark.parametrize(
    ("data", "expected_args", "expected_kwargs"),
    [
        (MONITOR_ESP32, ("10.0.0.5", 2323), {"listen_only": True}),
        (
            MONITOR_STICK,
            ("socket://10.0.0.7:5000", 0),
            {
                "serial_url": "socket://10.0.0.7:5000",
                "device_id": "nanocul-a1b2c3",
                "listen_only": True,
            },
        ),
    ],
)
def test_async_setup_entry_monitor_builds_listen_only_client(
    termoweb_init: Any,
    stub_hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    data: dict[str, Any],
    expected_args: tuple[Any, ...],
    expected_kwargs: dict[str, Any],
) -> None:
    created: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    class Unreachable(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            raise termoweb_init.RadioLinkError("unplugged")

    def fake_create(*args: Any, **kwargs: Any) -> Any:
        created.append((args, kwargs))
        return Unreachable(None, "", "")

    monkeypatch.setattr(backend_factory, "create_radio_client", fake_create)
    entry = ConfigEntry("monitor", data=dict(data))
    stub_hass.config_entries.add(entry)

    with pytest.raises(ConfigEntryNotReady):
        asyncio.run(termoweb_init.async_setup_entry(stub_hass, entry))
    ((args, kwargs),) = created
    assert args == (*expected_args, "A", [], b"\x00\x00")  # no dialect, no heaters
    assert kwargs == expected_kwargs


def test_monitor_entry_sets_up_listen_only_and_unloads(
    termoweb_init: Any, stub_hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests_ha.fakes.radio_link import FakeRadioLink

    radio_client = importlib.import_module(
        "custom_components.termoweb.backend.radio_client"
    )
    links: list[Any] = []

    def link_factory(host: str, port: int, dialect: Any, **kwargs: Any) -> Any:
        link = FakeRadioLink(host, port, dialect, **kwargs)
        links.append(link)
        return link

    def fake_create(host, port, dialect, nodes, network_id, **kwargs: Any) -> Any:
        assert kwargs == {"listen_only": True}
        return radio_client.RadioClient(
            host,
            port,
            dialect,
            nodes,
            network_id=network_id,
            station_id=radio_client.LISTEN_ONLY_STATION_ID,
            link_factory=link_factory,
            listen_only=True,
        )

    monkeypatch.setattr(backend_factory, "create_radio_client", fake_create)
    entry = ConfigEntry("monitor", data=dict(MONITOR_ESP32))
    stub_hass.config_entries.add(entry)

    async def _run() -> tuple[Any, ...]:
        assert await termoweb_init.async_setup_entry(stub_hass, entry) is True
        await _drain_tasks(stub_hass)
        for _ in range(20):
            await asyncio.sleep(0)
        record = termoweb_init._test_helpers.get_record(stub_hass, entry)
        ws_client = record.ws_clients["aabbcc001122"]
        running = ws_client._task is not None and not ws_client._task.done()
        unloaded = await termoweb_init.async_unload_entry(stub_hass, entry)
        return record, ws_client, running, unloaded

    record, ws_client, running, unloaded = asyncio.run(_run())
    assert type(ws_client).__name__ == "RadioMonitor" and running
    assert list(record.inventory.nodes) == []
    assert stub_hass.config_entries.forwarded == [(entry, ("binary_sensor", "sensor"))]
    assert unloaded is True and not (
        ws_client._task is not None and not ws_client._task.done()
    )
    assert stub_hass.config_entries.unloaded == [(entry, ("binary_sensor", "sensor"))]
    (link,) = links
    assert link.kwargs == {"listen_only": True, "auto_ack": False}
    assert link.station_id == 0xFE and link.network_id == b"\x00\x00"
    assert link.sent == [] and link.closes >= 1


def test_async_setup_entry_no_devices(
    termoweb_init: Any, stub_hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    class EmptyClient(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            return []

    monkeypatch.setattr(backend_factory, "RESTClient", EmptyClient)
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

    monkeypatch.setattr(backend_factory, "RESTClient", PartialClient)

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

    monkeypatch.setattr(backend_factory, "RESTClient", InvalidClient)
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

    monkeypatch.setattr(backend_factory, "RESTClient", MappingClient)

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

    monkeypatch.setattr(backend_factory, "RESTClient", WeirdClient)
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

    monkeypatch.setattr(backend_factory, "RESTClient", HappyClient)

    entry = ConfigEntry("startup", data={"username": "user", "password": "pw"})
    stub_hass.config_entries.add(entry)
    stub_hass.is_running = False

    async def _run() -> None:
        await termoweb_init.async_setup_entry(stub_hass, entry)
        await _drain_tasks(stub_hass)
        assert all(
            event != EVENT_HOMEASSISTANT_STARTED for event, _ in stub_hass.bus.listeners
        )

    asyncio.run(_run())


def test_recalc_poll_interval_transitions(
    termoweb_init: Any, stub_hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    class PollClient(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            return [{"dev_id": "dev-1"}]

    monkeypatch.setattr(backend_factory, "RESTClient", PollClient)
    entry = ConfigEntry("poll", data={"username": "user", "password": "pw"})
    stub_hass.config_entries.add(entry)

    async def _run() -> None:
        assert await termoweb_init.async_setup_entry(stub_hass, entry)
        await _drain_tasks(stub_hass)

        record = termoweb_init._test_helpers.get_record(stub_hass, entry)
        coordinator: FakeCoordinator = record.coordinator

        if record.ws_tasks:
            await asyncio.gather(*record.ws_tasks.values(), return_exceptions=True)
        record.ws_tasks.clear()
        record.ws_state.clear()
        record.ws_trackers.clear()

        base_interval = record.base_poll_interval
        current_time = 1_000.0

        def fake_time() -> float:
            return current_time

        monkeypatch.setattr(termoweb_init.time, "time", fake_time)

        scheduled: list[float | str] = []

        def fake_async_call_later(_hass: Any, delay: float, _cb: Callable[[Any], None]):
            scheduled.append(delay)

            def _cancel() -> None:
                scheduled.append("cancelled")

            return _cancel

        monkeypatch.setattr(termoweb_init, "async_call_later", fake_async_call_later)

        # (a) No running tasks while suspended restores base interval
        record.poll_suspended = True
        coordinator.update_interval = timedelta(seconds=999)
        record.recalc_poll()
        assert record.poll_suspended is False
        assert coordinator.update_interval == timedelta(seconds=base_interval)

        # (b) Healthy trackers with fresh payloads suspend polling
        healthy_event = asyncio.Event()
        healthy_task = asyncio.create_task(healthy_event.wait())
        tracker = WsHealthTracker("dev-healthy")
        tracker.update_status(
            "healthy", healthy_since=current_time, timestamp=current_time
        )
        tracker.mark_payload(timestamp=current_time, stale_after=300)
        record.ws_tasks["dev-healthy"] = healthy_task
        record.ws_trackers["dev-healthy"] = tracker
        record.poll_suspended = False
        coordinator.update_interval = timedelta(seconds=base_interval)
        current_time = 1_010.0
        record.recalc_poll()
        assert record.poll_suspended is True
        assert coordinator.update_interval is None
        assert scheduled and scheduled[0] > 0

        # (c) Stale payloads resume the base polling interval
        current_time = 1_400.0
        record.recalc_poll()
        assert record.poll_suspended is False
        assert coordinator.update_interval == timedelta(seconds=base_interval)
        assert scheduled[-1] == "cancelled"

        # (d) Fresh payloads restore suspension after a resume
        tracker.mark_payload(timestamp=current_time, stale_after=300)
        tracker.update_status(
            "healthy", healthy_since=current_time, timestamp=current_time
        )
        current_time = 1_410.0
        record.recalc_poll()
        assert record.poll_suspended is True
        assert coordinator.update_interval is None

        # (e) Unhealthy status resumes polling immediately
        tracker.update_status("degraded", timestamp=current_time + 5, reset_health=True)
        current_time = 1_420.0
        record.recalc_poll()
        assert record.poll_suspended is False
        assert coordinator.update_interval == timedelta(seconds=base_interval)
        assert scheduled[-1] == "cancelled"

        healthy_event.set()
        await healthy_task

    asyncio.run(_run())


def test_recalc_poll_interval_edge_cases(
    termoweb_init: Any, stub_hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    class PollClient(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            return [{"dev_id": "dev-edge"}]

    monkeypatch.setattr(backend_factory, "RESTClient", PollClient)
    entry = ConfigEntry("poll-edge", data={"username": "user", "password": "pw"})
    stub_hass.config_entries.add(entry)

    async def _run() -> None:
        assert await termoweb_init.async_setup_entry(stub_hass, entry)
        await _drain_tasks(stub_hass)

        record = termoweb_init._test_helpers.get_record(stub_hass, entry)
        coordinator: FakeCoordinator = record.coordinator
        base_interval = record.base_poll_interval

        scheduled: list[float] = []
        callbacks: list[Callable[[Any], None]] = []
        cancellations: list[bool] = []

        def fake_async_call_later(
            _hass: HomeAssistant, delay: float, callback: Callable[[Any], None]
        ) -> Callable[[], None]:
            scheduled.append(delay)
            callbacks.append(callback)

            def _cancel() -> None:
                cancellations.append(True)

            return _cancel

        monkeypatch.setattr(termoweb_init, "async_call_later", fake_async_call_later)

        current_time = {"value": 1_000.0}

        def fake_time() -> float:
            return current_time["value"]

        monkeypatch.setattr(termoweb_init.time, "time", fake_time)

        # (a) Completed tasks resume polling even when task dict is populated
        loop = asyncio.get_running_loop()
        done_task = loop.create_future()
        done_task.set_result(None)
        record.ws_tasks["dev-edge"] = done_task
        record.ws_trackers.clear()
        record.poll_suspended = True
        coordinator.update_interval = None
        record.recalc_poll()
        assert record.poll_suspended is False
        assert coordinator.update_interval == timedelta(seconds=base_interval)

        record.ws_tasks.clear()

        # (b) Missing trackers mark payloads stale and keep polling active
        orphan_event = asyncio.Event()
        orphan_task = asyncio.create_task(orphan_event.wait())
        record.ws_tasks["dev-missing"] = orphan_task
        record.ws_trackers.clear()
        record.poll_suspended = False
        coordinator.update_interval = timedelta(seconds=base_interval)
        record.recalc_poll()
        orphan_event.set()
        await orphan_task
        record.ws_tasks.clear()

        # (c) Trackers without payload timestamps trigger the empty timestamp branch
        tracker_event = asyncio.Event()
        tracker_task = asyncio.create_task(tracker_event.wait())
        no_payload_tracker = WsHealthTracker("dev-no-payload")
        no_payload_tracker.update_status("healthy")
        record.ws_tasks["dev-no-payload"] = tracker_task
        record.ws_trackers["dev-no-payload"] = no_payload_tracker
        record.poll_suspended = False
        coordinator.update_interval = timedelta(seconds=base_interval)
        record.recalc_poll()
        tracker_event.set()
        await tracker_task
        record.ws_tasks.clear()
        record.ws_trackers.clear()

        # (d) Trackers with legacy call signatures raise TypeError paths
        legacy_event = asyncio.Event()
        legacy_task = asyncio.create_task(legacy_event.wait())

        class LegacyTracker:
            status = "healthy"

            def __init__(self, timestamp: float) -> None:
                self.last_payload_at = timestamp

            def is_payload_stale(self) -> bool:
                return True

            def stale_deadline(self, _now: float) -> float:
                return self.last_payload_at + 10

        current_time["value"] = 2_000.0
        legacy_tracker = LegacyTracker(current_time["value"])
        record.ws_tasks["dev-legacy"] = legacy_task
        record.ws_trackers["dev-legacy"] = legacy_tracker
        record.poll_suspended = False
        coordinator.update_interval = timedelta(seconds=base_interval)
        record.recalc_poll()
        legacy_event.set()
        await legacy_task
        record.ws_tasks.clear()
        record.ws_trackers.clear()

        # (e) Healthy trackers schedule resume callbacks using async_call_later
        resume_event = asyncio.Event()
        resume_task = asyncio.create_task(resume_event.wait())

        class FreshTracker:
            status = "healthy"

            def __init__(self, timestamp: float) -> None:
                self.last_payload_at = timestamp

            def is_payload_stale(self, now: float | None = None) -> bool:
                return False

            def stale_deadline(self) -> float:
                return self.last_payload_at + 30

        current_time["value"] = 3_000.0
        fresh_tracker = FreshTracker(current_time["value"])
        record.ws_tasks["dev-fresh"] = resume_task
        record.ws_trackers["dev-fresh"] = fresh_tracker
        record.poll_suspended = False
        coordinator.update_interval = timedelta(seconds=base_interval)
        record.recalc_poll()
        assert record.poll_suspended is True
        assert record.poll_resume_unsub is not None
        assert scheduled and scheduled[-1] == 30

        resume_event.set()
        await resume_task
        record.ws_tasks.clear()
        record.ws_trackers.clear()

        resume_callback = callbacks[-1]
        assert callable(resume_callback)
        resume_callback(None)
        assert record.poll_suspended is False
        assert record.poll_resume_unsub is None

        assert cancellations in ([], [True])

    asyncio.run(_run())


def test_ws_status_dispatcher_filters_entry(
    termoweb_init: Any, stub_hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    class DispatchClient(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            return [{"dev_id": "dev-1"}]

    monkeypatch.setattr(backend_factory, "RESTClient", DispatchClient)
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
        coordinator1: FakeCoordinator = record1.coordinator
        base_interval = record1.base_poll_interval

        callbacks = {
            signal: callback for signal, callback in stub_hass.dispatcher_connections
        }
        cb1 = callbacks[termoweb_init.signal_ws_status(entry1.entry_id)]
        cb2 = callbacks[termoweb_init.signal_ws_status(entry2.entry_id)]

        if record1.ws_tasks:
            await asyncio.gather(*record1.ws_tasks.values(), return_exceptions=True)
        record1.ws_tasks.clear()
        record1.ws_state.clear()
        record1.ws_trackers.clear()

        # Matching payload triggers recalc for entry1
        healthy_event = asyncio.Event()
        healthy_task = asyncio.create_task(healthy_event.wait())
        record1.ws_tasks["dev-1"] = healthy_task
        healthy_tracker = WsHealthTracker("dev-1")
        healthy_tracker.update_status("healthy")
        healthy_tracker.mark_payload(stale_after=300)
        record1.ws_trackers["dev-1"] = healthy_tracker
        record1.poll_suspended = False
        coordinator1.update_interval = timedelta(seconds=base_interval)
        cb1({"entry_id": entry1.entry_id, "payload_changed": True})
        assert record1.poll_suspended is True
        assert coordinator1.update_interval is None
        healthy_event.set()
        await healthy_task

        # Mismatching payload (other entry callback) does not affect entry1
        record1.ws_tasks.clear()
        record1.ws_state.clear()
        record1.ws_trackers.clear()
        other_event = asyncio.Event()
        other_task = asyncio.create_task(other_event.wait())
        record1.ws_tasks["dev-1"] = other_task
        other_tracker = WsHealthTracker("dev-1")
        other_tracker.update_status("healthy")
        other_tracker.mark_payload(stale_after=300)
        record1.ws_trackers["dev-1"] = other_tracker
        record1.poll_suspended = False
        coordinator1.update_interval = timedelta(seconds=base_interval)
        cb2({"entry_id": entry1.entry_id, "payload_changed": True})
        assert record1.poll_suspended is False
        assert coordinator1.update_interval == timedelta(seconds=base_interval)
        other_event.set()
        await other_task

        # Status-only updates still trigger recalculation
        record1.ws_tasks.clear()
        record1.ws_state.clear()
        record1.ws_trackers.clear()
        status_event = asyncio.Event()
        status_task = asyncio.create_task(status_event.wait())
        record1.ws_tasks["dev-1"] = status_task
        status_tracker = WsHealthTracker("dev-1")
        status_tracker.update_status("healthy")
        status_tracker.mark_payload(stale_after=300)
        record1.ws_trackers["dev-1"] = status_tracker
        cb1({"reason": "status"})
        status_event.set()
        await status_task

        # Non-mapping payloads still trigger recalculation for the owning entry
        record1.ws_tasks.clear()
        record1.ws_state.clear()
        record1.ws_trackers.clear()
        fallback_event = asyncio.Event()
        fallback_task = asyncio.create_task(fallback_event.wait())
        record1.ws_tasks["dev-1"] = fallback_task
        cb1(object())
        fallback_event.set()
        await fallback_task

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

    monkeypatch.setattr(backend_factory, "RESTClient", ListenerClient)
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
        coordinator: FakeCoordinator = record.coordinator

        existing_task = record.ws_tasks.get("dev-1")
        assert isinstance(existing_task, asyncio.Task)
        assert not existing_task.done()

        stub_hass.tasks.clear()
        coordinator.data = dict(coordinator.data)
        coordinator.data["dev-2"] = {"dev_id": "dev-2"}
        assert not coordinator.listeners
        assert not stub_hass.tasks
        assert set(record.ws_tasks) == {"dev-1"}
        assert record.ws_tasks["dev-1"] is existing_task

        for event in start_events:
            event.set()
        await asyncio.gather(*record.ws_tasks.values(), return_exceptions=True)

    asyncio.run(_run())


def test_async_unload_entry_handles_task_and_client_errors(
    termoweb_init: Any, stub_hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    class HappyClient(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            return [{"dev_id": "dev-1"}]

    monkeypatch.setattr(backend_factory, "RESTClient", HappyClient)
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

        record.ws_tasks["dev-1"] = BadTask()
        record.ws_clients["dev-1"] = BadClient()

        log_calls: list[str] = []

        def capture_exception(msg: str, *args: Any, **kwargs: Any) -> None:
            log_calls.append(msg)

        monkeypatch.setattr(termoweb_init._LOGGER, "exception", capture_exception)

        assert await termoweb_init.async_unload_entry(stub_hass, entry)
        assert log_calls
        assert record._shutdown_complete

    asyncio.run(_run())


def test_async_setup_entry_cleans_up_on_hass_stop(
    termoweb_init: Any, stub_hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    class HappyClient(BaseFakeClient):
        async def list_devices(self) -> list[dict[str, Any]]:
            return [{"dev_id": "dev-stop"}]

    monkeypatch.setattr(backend_factory, "RESTClient", HappyClient)
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
        ws_task = next(iter(record.ws_tasks.values()))
        client = next(iter(record.ws_clients.values()))

        await listeners[0](None)

        return (
            client.stop_calls,
            ws_task.cancelled() or ws_task.done(),
            record._shutdown_complete,
        )

    stop_calls, cancelled, shutdown_flag = asyncio.run(_run())
    assert stop_calls == 1
    assert cancelled is True
    assert shutdown_flag is True


def test_async_shutdown_entry_cleans_up(
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

        record = build_entry_runtime(hass=stub_hass, entry_id=entry.entry_id)
        record.ws_tasks["dev"] = ws_task
        record.ws_clients["dev"] = client
        record.unsub_ws_status = lambda: unsubscribed.append(True)
        record.recalc_poll = lambda: None

        await termoweb_init._async_shutdown_entry(record)
        await termoweb_init._async_shutdown_entry(record)  # runs once
        return (
            record._shutdown_complete,
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
    assert unsubscribed == [True]


def test_async_unload_entry_missing_returns_true(
    termoweb_init: Any, stub_hass: HomeAssistant
) -> None:
    entry = ConfigEntry("missing", data={})
    stub_hass.config_entries.add(entry)
    assert asyncio.run(termoweb_init.async_unload_entry(stub_hass, entry)) is True


@pytest.mark.asyncio
async def test_shutdown_entry_skips_completed_record(termoweb_init: Any) -> None:
    rec = build_entry_runtime()
    rec._shutdown_complete = True
    await termoweb_init._async_shutdown_entry(rec)
    assert rec._shutdown_complete is True


@pytest.mark.asyncio
async def test_shutdown_entry_handles_client_without_stop(termoweb_init: Any) -> None:
    rec = build_entry_runtime()
    rec.ws_clients["dev"] = object()
    await termoweb_init._async_shutdown_entry(rec)
    assert rec._shutdown_complete is True


@pytest.mark.asyncio
async def test_shutdown_entry_cancels_poll_timer(termoweb_init: Any) -> None:
    cancelled: list[bool] = []

    def fake_cancel() -> None:
        cancelled.append(True)

    rec = build_entry_runtime()
    rec.ws_clients.clear()
    rec.ws_tasks.clear()
    rec.poll_resume_unsub = fake_cancel

    await termoweb_init._async_shutdown_entry(rec)

    assert rec._shutdown_complete is True
    assert rec.poll_resume_unsub is None
    assert cancelled == [True]


def test_platforms_for_brand_filters_lock_by_backend(termoweb_init: Any) -> None:
    """Lock platform should be enabled only for Ducaheat-backed brands."""

    assert termoweb_init._platforms_for_brand("termoweb") == [
        "button",
        "binary_sensor",
        "climate",
        "number",
        "sensor",
    ]
    assert termoweb_init._platforms_for_brand("ducaheat") == [
        "button",
        "binary_sensor",
        "climate",
        "number",
        "sensor",
        "lock",
    ]
    assert termoweb_init._platforms_for_brand("tevolve") == [
        "button",
        "binary_sensor",
        "climate",
        "number",
        "sensor",
        "lock",
    ]
