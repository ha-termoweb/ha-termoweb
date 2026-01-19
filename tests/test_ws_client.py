"""Unit tests for websocket client helpers."""

from __future__ import annotations

import asyncio
import importlib
import gzip
from types import MappingProxyType, ModuleType, SimpleNamespace
from typing import Any, Callable, Mapping
from urllib.parse import parse_qsl, urlsplit
from unittest.mock import AsyncMock, MagicMock

import pytest

import logging
import sys

from conftest import build_entry_runtime
from custom_components.termoweb.backend import ducaheat_ws
from custom_components.termoweb.backend import termoweb_ws as module
from custom_components.termoweb.backend import ws_client as base_ws
from custom_components.termoweb.backend.sanitize import (
    mask_identifier,
    redact_token_fragment,
)
from custom_components.termoweb.inventory import Inventory, build_node_inventory


class DummyREST:
    """Minimal REST client stub for websocket tests."""

    def __init__(self, *, is_ducaheat: bool = False) -> None:
        self._session = SimpleNamespace()
        self._headers = {"Authorization": "Bearer token"}
        self._ensure_token = AsyncMock()
        self._is_ducaheat = is_ducaheat
        self._access_token = "token"

    async def authed_headers(self) -> dict[str, str]:
        return self._headers

    async def refresh_token(self) -> None:
        self._access_token = None
        await self._ensure_token()


class DummyTask:
    """Track coroutine execution for idle restart tests."""

    def __init__(self, coro: Any) -> None:
        self.coro = coro
        self._cancelled = False
        self._completed = False

    def cancel(self) -> None:
        self._cancelled = True
        self._completed = True
        try:
            self.coro.close()
        except AttributeError:
            pass

    def done(self) -> bool:
        return self._completed

    def __await__(self) -> Any:
        async def _finished() -> None:
            return None

        if self._completed:
            return _finished().__await__()
        return self.run().__await__()

    async def run(self) -> Any:
        try:
            return await self.coro
        finally:
            self._completed = True


class DummyLoop:
    """Simple event loop stub recording created tasks."""

    def __init__(self) -> None:
        self.created_tasks: list[DummyTask] = []

    def create_task(self, coro: Any, **_: Any) -> DummyTask:
        task = DummyTask(coro)
        self.created_tasks.append(task)
        return task

    def call_soon_threadsafe(self, callback: Any, *args: Any) -> None:
        callback(*args)


@pytest.fixture(autouse=True)
def patch_async_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ``socketio.AsyncClient`` with a controllable stub."""

    class StubAsyncClient:
        def __init__(self, **_: Any) -> None:
            self.events: dict[tuple[str, str | None], Any] = {}
            self.connected = False

        def on(self, event: str, *, handler: Any, namespace: str | None = None) -> None:
            self.events[(event, namespace)] = handler

        async def disconnect(self) -> None:
            self.connected = False

        async def emit(
            self,
            event: str,
            data: Any | None = None,
            *,
            namespace: str | None = None,
        ) -> None:  # pragma: no cover - only used for safety
            self.events[(event, namespace)] = (event, data)

    monkeypatch.setattr(module.socketio, "AsyncClient", StubAsyncClient)


@pytest.fixture(autouse=True)
def reload_ws_modules(patch_async_client: None) -> None:
    """Reload websocket modules to avoid stale references after other tests."""

    global base_ws, module
    base_ws = importlib.reload(
        importlib.import_module("custom_components.termoweb.backend.ws_client")
    )
    module = importlib.reload(
        importlib.import_module("custom_components.termoweb.backend.termoweb_ws")
    )


@pytest.fixture
def ws_common_stub() -> Callable[..., base_ws._WSCommon]:
    """Provide a configurable ``_WSCommon`` test double."""

    def _factory(
        *,
        hass: Any | None = None,
        entry_id: str = "entry",
        dev_id: str = "dev",
        coordinator: Any | None = None,
        inventory: Inventory | None = None,
        call_base_init: bool = True,
    ) -> base_ws._WSCommon:
        class Stub(base_ws._WSCommon):
            def __init__(self) -> None:
                self.hass = hass or SimpleNamespace(data={base_ws.DOMAIN: {}})
                self.entry_id = entry_id
                self.dev_id = dev_id
                self._coordinator = coordinator or SimpleNamespace(
                    update_nodes=MagicMock()
                )
                if getattr(
                    self.hass, "data", None
                ) is not None and entry_id not in self.hass.data.get(
                    base_ws.DOMAIN, {}
                ):
                    build_entry_runtime(
                        hass=self.hass,
                        entry_id=entry_id,
                        dev_id=dev_id,
                        inventory=inventory,
                        coordinator=self._coordinator,
                    )
                if call_base_init:
                    super().__init__(inventory=inventory)
                else:
                    self._inventory = inventory

        return Stub()

    return _factory


def _make_termoweb_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    hass_loop: Any | None = None,
) -> module.WebSocketClient:
    """Instantiate a TermoWeb websocket client for tests."""

    if hass_loop is None:
        hass_loop = SimpleNamespace(
            create_task=lambda coro, **_: SimpleNamespace(done=lambda: True),
            call_soon_threadsafe=lambda cb, *args: cb(*args),
        )

    hass = SimpleNamespace(loop=hass_loop, data={module.DOMAIN: {}})
    build_entry_runtime(
        hass=hass,
        entry_id="entry",
        dev_id="device",
    )
    coordinator = SimpleNamespace(update_nodes=MagicMock(), dev_id="dev")
    dispatcher = MagicMock()
    monkeypatch.setattr(module, "async_dispatcher_send", dispatcher)
    client = module.WebSocketClient(
        hass,
        entry_id="entry",
        dev_id="device",
        api_client=DummyREST(),
        coordinator=coordinator,
        session=SimpleNamespace(),
    )
    client._dispatcher_mock = dispatcher  # type: ignore[attr-defined]
    return client


def _make_ducaheat_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    hass_loop: Any | None = None,
) -> ducaheat_ws.DucaheatWSClient:
    """Instantiate a Ducaheat websocket client for tests."""

    if hass_loop is None:
        hass_loop = SimpleNamespace(
            create_task=lambda coro, **_: SimpleNamespace(done=lambda: True),
            call_soon_threadsafe=lambda cb, *args: cb(*args),
        )
    hass = SimpleNamespace(loop=hass_loop, data={module.DOMAIN: {}})
    build_entry_runtime(
        hass=hass,
        entry_id="entry",
        dev_id="device",
    )
    rest_client = DummyREST(is_ducaheat=True)
    dispatcher = MagicMock()
    monkeypatch.setattr(ducaheat_ws, "async_dispatcher_send", dispatcher, raising=False)
    client = ducaheat_ws.DucaheatWSClient(
        hass,
        entry_id="entry",
        dev_id="device",
        api_client=rest_client,
        coordinator=SimpleNamespace(update_nodes=MagicMock()),
        session=SimpleNamespace(),
    )
    client._dispatcher_mock = dispatcher  # type: ignore[attr-defined]
    return client


@pytest.mark.asyncio
async def test_ws_state_cleanup_and_reuse() -> None:
    """Stop cycles should clean up state buckets without duplicating metadata."""

    loop = DummyLoop()
    hass = SimpleNamespace(loop=loop, data={base_ws.DOMAIN: {}})
    raw_nodes = {"nodes": [{"type": "htr", "addr": "1"}]}
    inventory = Inventory("device", build_node_inventory(raw_nodes))
    runtime = build_entry_runtime(
        hass=hass,
        entry_id="entry",
        dev_id="device",
        inventory=inventory,
    )

    client = module.WebSocketClient(
        hass,
        entry_id="entry",
        dev_id="device",
        api_client=DummyREST(),
        coordinator=SimpleNamespace(update_nodes=MagicMock()),
        session=SimpleNamespace(),
        inventory=inventory,
    )

    for _ in range(3):
        client.start()
        client._ws_state_bucket()
        client._ws_health_tracker()
        assert client._ws_bucket_sizes() == (1, 1)
        assert runtime.inventory is inventory
        assert set(runtime.ws_state.keys()) == {"device"}
        assert set(runtime.ws_trackers.keys()) == {"device"}

        await client.stop()
        assert runtime.ws_state == {}
        assert runtime.ws_trackers == {}
        assert client._ws_bucket_sizes() == (0, 0)


@pytest.mark.asyncio
async def test_ducaheat_ws_cleanup_and_buckets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure Ducaheat websocket cleanup removes tracker buckets across retries."""

    loop = DummyLoop()
    hass = SimpleNamespace(loop=loop, data={base_ws.DOMAIN: {}})
    raw_nodes = {"nodes": [{"type": "pmo", "addr": "7"}]}
    inventory = Inventory("device", build_node_inventory(raw_nodes))
    runtime = build_entry_runtime(
        hass=hass,
        entry_id="entry",
        dev_id="device",
        inventory=inventory,
    )

    rest_client = DummyREST(is_ducaheat=True)
    dispatcher = MagicMock()
    monkeypatch.setattr(ducaheat_ws, "async_dispatcher_send", dispatcher, raising=False)
    client = ducaheat_ws.DucaheatWSClient(
        hass,
        entry_id="entry",
        dev_id="device",
        api_client=rest_client,
        coordinator=SimpleNamespace(update_nodes=MagicMock()),
        session=SimpleNamespace(),
        inventory=inventory,
    )

    for _ in range(2):
        client.start()
        client._ws_state_bucket()
        client._ws_health_tracker()
        assert client._ws_bucket_sizes() == (1, 1)
        assert runtime.inventory is inventory

        await client.stop()
        assert runtime.ws_state == {}
        assert runtime.ws_trackers == {}
        assert client._ws_bucket_sizes() == (0, 0)


def _ensure_inventory_record(
    hass: Any,
    entry_id: str,
    *,
    dev_id: str = "dev",
    inventory: Inventory | None = None,
) -> Inventory:
    """Populate ``hass`` domain data with a default inventory if needed."""

    if not isinstance(inventory, Inventory):
        payload = {
            "nodes": [
                {"type": "htr", "addr": "1"},
                {"type": "pmo", "addr": "7"},
            ]
        }
        inventory = Inventory(dev_id, build_node_inventory(payload))
    hass.data.setdefault(base_ws.DOMAIN, {})
    runtime = hass.data[base_ws.DOMAIN].get(entry_id)
    runtime_module = importlib.import_module("custom_components.termoweb.runtime")
    if not isinstance(runtime, runtime_module.EntryRuntime):
        runtime = build_entry_runtime(
            hass=hass,
            entry_id=entry_id,
            dev_id=dev_id,
            inventory=inventory,
        )
    else:
        runtime.inventory = inventory
    return inventory


def test_forward_ws_sample_updates_guards_and_invalid_lease() -> None:
    """Guard clauses and invalid lease values should be handled safely."""

    hass = SimpleNamespace(data={base_ws.DOMAIN: {}})
    base_ws.forward_ws_sample_updates(
        hass,
        "entry",
        "dev",
        {"pmo": {"samples": {"7": {"power": 1}}}},
    )

    runtime = build_entry_runtime(
        hass=hass,
        entry_id="entry",
        dev_id="dev",
        energy_coordinator=SimpleNamespace(),
    )
    _ensure_inventory_record(hass, "entry", dev_id="dev")
    base_ws.forward_ws_sample_updates(
        hass,
        "entry",
        "dev",
        {"pmo": {"samples": {"7": {"power": 2}}}},
    )

    class CoordinatorStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any], Any]] = []

        def handle_ws_samples(
            self,
            dev_id: str,
            updates: dict[str, dict[str, Any]],
            *,
            lease_seconds: float | None = None,
        ) -> None:
            self.calls.append((dev_id, updates, lease_seconds))

    coordinator = CoordinatorStub()
    runtime.energy_coordinator = coordinator

    base_ws.forward_ws_sample_updates(
        hass,
        "entry",
        "dev",
        {"pmo": {"samples": {"7": {"power": 3}}, "lease_seconds": "bad"}},
    )

    assert coordinator.calls == [
        ("dev", {"pmo": {"7": {"power": 3}}}, None),
    ]


def test_forward_ws_sample_updates_handles_power_monitors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """forward_ws_sample_updates should normalise power monitor payloads."""

    hass = SimpleNamespace(data={base_ws.DOMAIN: {}})
    raw_nodes = {"nodes": [{"type": "pmo", "addr": "7", "name": "PM"}]}
    inventory = Inventory("dev", build_node_inventory(raw_nodes))
    handler = MagicMock()
    build_entry_runtime(
        hass=hass,
        entry_id="entry",
        dev_id="dev",
        inventory=inventory,
        energy_coordinator=SimpleNamespace(handle_ws_samples=handler),
    )

    base_ws.forward_ws_sample_updates(
        hass,
        "entry",
        "dev",
        {
            "power_monitor": {
                "samples": {"7": {"power": 100}},
                "lease_seconds": 90,
            }
        },
    )

    handler.assert_called_once()
    args = handler.call_args[0]
    assert args[0] == "dev"
    assert args[1] == {"pmo": {"7": {"power": 100}}}
    assert handler.call_args.kwargs.get("lease_seconds") == 90


def test_forward_ws_sample_updates_skips_thermostats() -> None:
    """Thermostat sample payloads should be ignored."""

    hass = SimpleNamespace(data={base_ws.DOMAIN: {}})
    raw_nodes = {"nodes": [{"type": "thm", "addr": "1"}]}
    inventory = Inventory("dev", build_node_inventory(raw_nodes))
    handler = MagicMock()
    build_entry_runtime(
        hass=hass,
        entry_id="entry",
        dev_id="dev",
        inventory=inventory,
        energy_coordinator=SimpleNamespace(handle_ws_samples=handler),
    )

    base_ws.forward_ws_sample_updates(
        hass,
        "entry",
        "dev",
        {"thm": {"samples": {"1": {"counter": 1}}}},
    )

    handler.assert_not_called()


def test_forward_ws_sample_updates_respect_inventory_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Samples for disallowed node types should be ignored."""

    hass = SimpleNamespace(data={base_ws.DOMAIN: {}})
    raw_nodes = {"nodes": [{"type": "htr", "addr": "5"}]}
    inventory = Inventory("dev", build_node_inventory(raw_nodes))
    object.__setattr__(inventory, "_energy_sample_types_cache", frozenset({"pmo"}))
    handler = MagicMock()
    build_entry_runtime(
        hass=hass,
        entry_id="entry",
        dev_id="dev",
        inventory=inventory,
        energy_coordinator=SimpleNamespace(handle_ws_samples=handler),
    )

    base_ws.forward_ws_sample_updates(
        hass,
        "entry",
        "dev",
        {"htr": {"samples": {"5": {"counter": 1}}}},
    )

    handler.assert_not_called()


def test_forward_ws_sample_updates_uses_coordinator_inventory(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Coordinator inventory aliases and logging should be applied."""

    raw_nodes = {"nodes": [{"type": "htr", "addr": "5"}]}
    inventory = Inventory("dev", build_node_inventory(raw_nodes))

    monkeypatch.setattr(
        Inventory,
        "heater_sample_address_map",
        property(lambda self: ({"htr": ["5"]}, {"heater": "htr"})),
    )
    monkeypatch.setattr(
        Inventory,
        "power_monitor_sample_address_map",
        property(lambda self: ({}, {})),
    )

    handler = MagicMock(side_effect=RuntimeError("boom"))
    hass = SimpleNamespace(data={base_ws.DOMAIN: {}})
    build_entry_runtime(
        hass=hass,
        entry_id="entry",
        dev_id="dev",
        energy_coordinator=SimpleNamespace(handle_ws_samples=handler),
        coordinator=SimpleNamespace(inventory=inventory),
    )

    logger = logging.getLogger("test_forward_ws_samples")
    caplog.set_level(logging.DEBUG, logger=logger.name)

    base_ws.forward_ws_sample_updates(
        hass,
        "entry",
        "dev",
        {
            "heater": {
                "samples": {"5": {"temp": 21}, "lease_seconds": 10},
                "lease_seconds": 30,
            },
            "acm": {"lease_seconds": -5},
        },
        logger=logger,
        log_prefix="tester",
    )

    handler.assert_called_once()
    args = handler.call_args[0]
    assert args[0] == "dev"
    assert args[1] == {"htr": {"5": {"temp": 21}}}
    assert handler.call_args.kwargs.get("lease_seconds") == 30
    assert any(
        record.name == logger.name
        and record.levelno == logging.DEBUG
        and record.message == "tester: forwarding heater samples failed"
        for record in caplog.records
    )


def test_forward_ws_sample_updates_skips_invalid_sections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid update payloads should be ignored without calling the handler."""

    handler = MagicMock()
    hass = SimpleNamespace(data={base_ws.DOMAIN: {}})
    build_entry_runtime(
        hass=hass,
        entry_id="entry",
        dev_id="dev",
        inventory=Inventory("dev", []),
        energy_coordinator=SimpleNamespace(handle_ws_samples=handler),
    )

    base_ws.forward_ws_sample_updates(
        hass,
        "entry",
        "dev",
        {"pmo": ["invalid"], None: {"1": {}}},
    )

    handler.assert_not_called()


def test_forward_ws_sample_updates_skips_non_mapping_samples() -> None:
    """Sample sections that are not mappings should be skipped."""

    handler = MagicMock()
    hass = SimpleNamespace(data={base_ws.DOMAIN: {}})
    build_entry_runtime(
        hass=hass,
        entry_id="entry",
        dev_id="dev",
        inventory=Inventory("dev", []),
        energy_coordinator=SimpleNamespace(handle_ws_samples=handler),
    )

    class WeirdMapping(dict):
        def get(self, key: Any, default: Any | None = None) -> Any:
            if key == "samples":
                return {"7": {"power": 1}}
            return super().get(key, default)

        def __getitem__(self, key: Any) -> Any:
            if key == "samples":
                return ["invalid"]
            return super().__getitem__(key)

    base_ws.forward_ws_sample_updates(
        hass,
        "entry",
        "dev",
        {"htr": WeirdMapping({"samples": None, "lease_seconds": 30})},
    )

    handler.assert_not_called()


def test_forward_ws_sample_updates_inventory_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inventory-derived alias data should tolerate malformed updates."""

    raw_nodes = {"nodes": [{"type": "pmo", "addr": "3"}]}
    inventory = Inventory("dev", build_node_inventory(raw_nodes))

    monkeypatch.setattr(
        Inventory,
        "heater_sample_address_map",
        property(lambda self: ({"htr": ["1"]}, {"bad": "htr"})),
    )
    monkeypatch.setattr(
        Inventory,
        "power_monitor_sample_address_map",
        property(lambda self: ({"pmo": ["3"]}, {"invalid": "pmo"})),
    )

    handler = MagicMock()
    hass = SimpleNamespace(data={base_ws.DOMAIN: {}})
    build_entry_runtime(
        hass=hass,
        entry_id="entry",
        dev_id="dev",
        inventory=inventory,
        energy_coordinator=SimpleNamespace(handle_ws_samples=handler),
    )

    base_ws.forward_ws_sample_updates(
        hass,
        "entry",
        "dev",
        {
            None: {"1": {"power": 10}},
            "acm": "ignored",
            "pmo": {
                "samples": {"": {"power": 3}, "3": {"power": 5}},
                "lease_seconds": 15,
            },
        },
    )

    handler.assert_called_once()
    args = handler.call_args[0]
    assert args[0] == "dev"
    assert args[1] == {"pmo": {"3": {"power": 5}}}
    assert handler.call_args.kwargs.get("lease_seconds") == 15


def test_getattr_exposes_ducaheat_ws_client() -> None:
    """__getattr__ should expose the Ducaheat websocket client class."""

    cls = base_ws.__getattr__("DucaheatWSClient")
    assert cls.__name__ == ducaheat_ws.DucaheatWSClient.__name__
    assert cls.__module__ == ducaheat_ws.DucaheatWSClient.__module__


def test_getattr_exposes_termoweb_ws_client() -> None:
    """__getattr__ should expose the TermoWeb websocket client class."""

    cls = base_ws.__getattr__("TermoWebWSClient")
    assert cls.__name__ == module.TermoWebWSClient.__name__
    assert cls.__module__ == module.TermoWebWSClient.__module__


def test_getattr_raises_for_unknown_client() -> None:
    """Unknown attributes should raise AttributeError via __getattr__."""

    with pytest.raises(AttributeError) as exc_info:
        base_ws.__getattr__("UnknownClient")

    assert exc_info.value.args == ("UnknownClient",)


def test_termoweb_client_initialises_namespace_and_handlers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure the TermoWeb client uses the default namespace and registers events."""

    client = _make_termoweb_client(monkeypatch)
    assert client._namespace == module.WS_NAMESPACE
    expected = {
        ("connect", None),
        ("disconnect", None),
        ("reconnect", None),
        ("connect", module.WS_NAMESPACE),
        ("dev_data", module.WS_NAMESPACE),
        ("update", module.WS_NAMESPACE),
    }
    assert expected.issubset(client._sio.events.keys())


def test_ws_state_bucket_initialises_missing_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify hass.data is created when absent."""

    client = _make_termoweb_client(monkeypatch)
    bucket = client._ws_state_bucket()
    runtime = client.hass.data[module.DOMAIN]["entry"]
    runtime_module = importlib.import_module("custom_components.termoweb.runtime")
    assert isinstance(runtime, runtime_module.EntryRuntime)
    assert runtime.ws_state["device"] is bucket


def test_handshake_error_exposes_status_and_url() -> None:
    """Ensure ``HandshakeError`` forwards the status and URL details."""

    error = base_ws.HandshakeError(
        470,
        "https://example/ws",
        "nope",
        response_snippet="snippet",
    )
    assert str(error) == "handshake failed: status=470, detail=nope"
    assert error.status == 470
    assert error.url == "https://example/ws"
    assert error.detail == "nope"
    assert error.response_snippet == "snippet"


def test_dispatch_nodes_reuses_record_inventory(
    monkeypatch: pytest.MonkeyPatch,
    ws_common_stub: Callable[..., base_ws._WSCommon],
) -> None:
    """Node dispatch should rely on the immutable inventory stored during setup."""

    payload = {"nodes": [{"addr": "1", "type": "htr"}]}
    node_inventory = build_node_inventory(payload["nodes"])
    inventory = Inventory("device", node_inventory)

    hass = SimpleNamespace(data={base_ws.DOMAIN: {}})
    build_entry_runtime(
        hass=hass,
        entry_id="entry",
        dev_id="device",
        inventory=inventory,
    )
    coordinator = SimpleNamespace(update_nodes=MagicMock(), dev_id="dev")
    dispatcher = MagicMock()
    monkeypatch.setattr(base_ws, "async_dispatcher_send", dispatcher)

    def _fail(*_: Any, **__: Any) -> Any:
        raise AssertionError("resolve_record_inventory should not be called")

    monkeypatch.setattr(base_ws, "resolve_record_inventory", _fail, raising=False)

    dummy = ws_common_stub(
        hass=hass,
        entry_id="entry",
        dev_id="device",
        coordinator=coordinator,
    )
    dummy._inventory = None
    dummy._dispatch_nodes(payload)

    coordinator.update_nodes.assert_not_called()
    dispatcher.assert_called_once()
    dispatched_payload = dispatcher.call_args.args[2]
    assert dispatched_payload["inventory"] is inventory
    assert "addresses_by_type" not in dispatched_payload
    assert "nodes" not in dispatched_payload


def test_prepare_nodes_dispatch_uses_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Existing inventory objects should be reused by the dispatch helper."""

    hass = SimpleNamespace(data={base_ws.DOMAIN: {}})
    coordinator = SimpleNamespace(update_nodes=MagicMock(), dev_id="dev")
    node_inventory = build_node_inventory([{"type": "htr", "addr": "4"}])
    inventory = Inventory(
        "dev",
        node_inventory,
    )
    context = base_ws._prepare_nodes_dispatch(
        hass,
        entry_id="entry",
        coordinator=coordinator,
        raw_nodes=None,
        inventory=inventory,
    )

    assert context.inventory is inventory
    coordinator.update_nodes.assert_not_called()


def test_prepare_nodes_dispatch_prefers_runtime_inventory() -> None:
    """Runtime inventory should be preferred over coordinator inventory."""

    inventory = Inventory("dev", [])
    hass = SimpleNamespace(data={base_ws.DOMAIN: {}})
    build_entry_runtime(
        hass=hass,
        entry_id="entry",
        dev_id="raw",
        inventory=inventory,
    )
    coordinator = SimpleNamespace(update_nodes=MagicMock())

    context = base_ws._prepare_nodes_dispatch(
        hass,
        entry_id="entry",
        coordinator=coordinator,
        raw_nodes={},
    )
    assert context.inventory is inventory
    coordinator.update_nodes.assert_not_called()

    hass_numeric = SimpleNamespace(data={base_ws.DOMAIN: {}})
    build_entry_runtime(
        hass=hass_numeric,
        entry_id="entry",
        dev_id="99",
    )
    coordinator_numeric = SimpleNamespace(update_nodes=MagicMock(), inventory=inventory)

    context_numeric = base_ws._prepare_nodes_dispatch(
        hass_numeric,
        entry_id="entry",
        coordinator=coordinator_numeric,
        raw_nodes={"nodes": []},
    )

    assert context_numeric.inventory is not inventory
    assert context_numeric.inventory is not None
    assert context_numeric.inventory.dev_id == "99"
    coordinator_numeric.update_nodes.assert_not_called()


def test_termoweb_nodes_to_deltas(monkeypatch: pytest.MonkeyPatch) -> None:
    """Node payloads should translate into domain deltas."""

    client = _make_termoweb_client(monkeypatch)
    raw_nodes = {"nodes": [{"type": "htr", "addr": "1"}]}
    inventory = Inventory("device", build_node_inventory(raw_nodes))
    client._inventory = inventory

    nodes_payload = {
        "htr": {
            "settings": {"1": {"mode": "manual", "unknown": "drop"}},
            "status": {"1": {"stemp": "18.0", "online": True}},
            "prog": {"1": {"0": 1}},
            "samples": {"1": {"temp": 12}},
            "advanced": {"1": {"misc": "skip"}},
        }
    }

    deltas = client._nodes_to_deltas(nodes_payload, inventory=inventory)
    assert len(deltas) == 1
    delta = deltas[0]
    domain_module = importlib.import_module("custom_components.termoweb.domain")
    assert isinstance(delta, domain_module.NodeSettingsDelta)
    assert delta.node_id.addr == "1"
    assert delta.payload["mode"] == "manual"
    assert delta.payload["stemp"] == "18.0"
    assert delta.payload["prog"] == {"0": 1}
    assert "unknown" not in delta.payload
    assert "samples" not in delta.payload


def test_termoweb_nodes_to_deltas_validates_inventory(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Unknown nodes should be logged and ignored during delta translation."""

    client = _make_termoweb_client(monkeypatch)
    raw_nodes = {"nodes": [{"type": "htr", "addr": "1"}]}
    inventory = Inventory("device", build_node_inventory(raw_nodes))
    client._inventory = inventory

    with caplog.at_level(logging.WARNING, module._LOGGER.name):
        deltas = client._nodes_to_deltas(
            {"htr": {"settings": {"2": {"mode": "auto"}}}},
            inventory=inventory,
        )

    assert deltas == []
    assert any("unknown node_type" in record.message for record in caplog.records)


def test_termoweb_translate_path_deltas(monkeypatch: pytest.MonkeyPatch) -> None:
    """Path frames should yield node mappings and typed deltas."""

    client = _make_termoweb_client(monkeypatch)
    raw_nodes = {"nodes": [{"type": "htr", "addr": "1"}]}
    inventory = Inventory("device", build_node_inventory(raw_nodes))
    client._inventory = inventory

    payload = {"path": "/devs/device/htr/1/settings", "body": {"mode": "auto"}}
    nodes, deltas = client._translate_path_deltas(payload, inventory=inventory)

    assert nodes is not None
    assert nodes["htr"]["settings"]["1"]["mode"] == "auto"
    assert len(deltas) == 1
    assert deltas[0].payload["mode"] == "auto"


def test_ws_status_tracker_applies_default_cadence_hint() -> None:
    """Creating a tracker should immediately apply the cadence hint."""

    class Dummy(base_ws._WSStatusMixin):
        def __init__(self) -> None:
            self.hass = SimpleNamespace(data={base_ws.DOMAIN: {}})
            self.entry_id = "entry"
            self.dev_id = "dev"
            self._apply_payload_window_hint = MagicMock()
            build_entry_runtime(
                hass=self.hass,
                entry_id=self.entry_id,
                dev_id=self.dev_id,
            )

    dummy = Dummy()
    tracker = dummy._ws_health_tracker()

    assert isinstance(tracker, base_ws.WsHealthTracker)
    dummy._apply_payload_window_hint.assert_called_once_with(
        source="cadence",
        lease_seconds=120,
        candidates=[30, 75, "90"],
    )


def test_ws_status_tracker_processes_pending_cadence_hint() -> None:
    """Deferred cadence hints should wait until suppression is lifted."""

    class Dummy(base_ws._WSStatusMixin):
        def __init__(self) -> None:
            self.hass = SimpleNamespace(data={base_ws.DOMAIN: {}})
            self.entry_id = "entry"
            self.dev_id = "dev"
            self._apply_payload_window_hint = MagicMock()
            self._suppress_default_cadence_hint = True
            self._pending_default_cadence_hint = True
            build_entry_runtime(
                hass=self.hass,
                entry_id=self.entry_id,
                dev_id=self.dev_id,
            )

    dummy = Dummy()
    dummy._ws_health_tracker()

    dummy._apply_payload_window_hint.assert_not_called()
    assert dummy._pending_default_cadence_hint is True

    dummy._suppress_default_cadence_hint = False
    dummy._ws_health_tracker()
    dummy._apply_payload_window_hint.assert_called_once_with(
        source="cadence",
        lease_seconds=120,
        candidates=[30, 75, "90"],
    )
    assert dummy._pending_default_cadence_hint is False


def test_ws_common_ensure_type_bucket_handles_invalid_inputs(
    ws_common_stub: Callable[..., base_ws._WSCommon],
) -> None:
    """Ensure type bucket helper should guard against invalid structures."""

    dummy = ws_common_stub(
        hass=SimpleNamespace(data={base_ws.DOMAIN: {}}),
        coordinator=SimpleNamespace(),
    )
    build_entry_runtime(
        hass=dummy.hass,
        entry_id="entry",
        dev_id="dev",
    )
    assert dummy._ensure_type_bucket([], "htr") is None
    assert dummy._ensure_type_bucket({}, "", dev_map=None) is None

    bucket = dummy._ensure_type_bucket({}, "htr", dev_map=None)
    assert bucket == {}


def test_ws_common_ensure_type_bucket_uses_inventory_without_clones(
    ws_common_stub: Callable[..., base_ws._WSCommon],
) -> None:
    """Ensure type bucket helper reuses immutable metadata containers."""

    dummy = ws_common_stub(
        hass=SimpleNamespace(data={base_ws.DOMAIN: {}}),
        coordinator=SimpleNamespace(),
    )
    raw_nodes = {
        "nodes": [
            {"type": "htr", "addr": "1"},
            {"type": "acm", "addr": "2"},
        ]
    }
    inventory = Inventory("dev", build_node_inventory(raw_nodes))
    build_entry_runtime(
        hass=dummy.hass,
        entry_id="entry",
        dev_id="dev",
        inventory=inventory,
    )

    nodes_by_type = {
        "htr": MappingProxyType(
            {
                "settings": MappingProxyType({"1": MappingProxyType({"mode": "auto"})}),
                "samples": MappingProxyType({"1": MappingProxyType({"temp": 20})}),
                "status": MappingProxyType({"1": MappingProxyType({"on": True})}),
            }
        )
    }
    dev_map = {"settings": MappingProxyType({})}
    original_settings = dev_map["settings"]

    bucket = dummy._ensure_type_bucket(nodes_by_type, "htr", dev_map=dev_map)
    assert "addrs" not in bucket
    assert bucket is nodes_by_type["htr"]
    assert bucket["settings"]["1"]["mode"] == "auto"
    assert bucket["samples"]["1"]["temp"] == 20
    assert bucket["status"]["1"]["on"] is True
    assert dev_map["settings"] is original_settings
    assert dev_map["inventory"] is inventory
    assert "addresses_by_type" not in dev_map

    dev_map_second = {"settings": {"htr": {"existing": 1}}, "inventory": inventory}
    bucket_again = dummy._ensure_type_bucket(
        {"htr": bucket}, "htr", dev_map=dev_map_second
    )
    assert bucket_again is bucket
    assert dev_map_second["settings"]["htr"] == {"existing": 1}

    dev_map_non_mapping_settings = {"settings": None, "inventory": inventory}
    dummy._ensure_type_bucket(
        {"htr": bucket}, "htr", dev_map=dev_map_non_mapping_settings
    )
    assert dev_map_non_mapping_settings["settings"]["htr"] == {}
    assert "addresses_by_type" not in dev_map_non_mapping_settings


def test_ws_common_apply_heater_addresses_uses_inventory(
    ws_common_stub: Callable[..., base_ws._WSCommon],
) -> None:
    """Heater address helper should reuse the immutable inventory data."""

    energy_coordinator = SimpleNamespace(update_addresses=MagicMock())
    raw_nodes = {
        "nodes": [
            {"type": "htr", "addr": "1"},
            {"type": "acm", "addr": "2"},
            {"type": "pmo", "addr": "7"},
        ]
    }
    inventory = Inventory("dev", build_node_inventory(raw_nodes))
    hass = SimpleNamespace(data={base_ws.DOMAIN: {}})
    hass_record = build_entry_runtime(
        hass=hass,
        entry_id="entry",
        dev_id="dev",
        inventory=inventory,
        energy_coordinator=energy_coordinator,
    )

    dummy = ws_common_stub(
        hass=hass,
        coordinator=SimpleNamespace(),
    )
    dummy._apply_heater_addresses({}, inventory=inventory)

    assert not hasattr(hass_record, "sample_aliases")
    energy_coordinator.update_addresses.assert_called_once_with(inventory)

    energy_coordinator.update_addresses.reset_mock()
    hass_record.inventory = None  # type: ignore[assignment]
    dummy._apply_heater_addresses({}, inventory=None)
    energy_coordinator.update_addresses.assert_called_once_with(inventory)


@pytest.mark.asyncio
async def test_websocket_client_reuses_delegate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Starting twice should reuse the already created delegate."""

    hass = SimpleNamespace(loop=asyncio.get_event_loop(), data={base_ws.DOMAIN: {}})
    client = base_ws.WebSocketClient(
        hass,
        entry_id="entry",
        dev_id="device",
        api_client=DummyREST(),
        coordinator=SimpleNamespace(),
    )
    delegate = SimpleNamespace(
        start=MagicMock(return_value="task"),
        stop=AsyncMock(return_value=None),
        is_running=MagicMock(return_value=True),
        ws_url=AsyncMock(return_value="wss://example/ws"),
    )
    client._delegate = delegate  # type: ignore[assignment]

    assert client.start() == "task"
    delegate.start.assert_called_once_with()
    assert client.is_running() is True

    await client.stop()
    delegate.stop.assert_awaited_once_with()
    assert await client.ws_url() == "wss://example/ws"

    client._delegate = None
    assert await client.ws_url() == ""


def test_ducaheat_brand_headers_include_expected_fields() -> None:
    """Verify Ducaheat brand headers contain required keys."""

    headers = ducaheat_ws._brand_headers("agent", "requested")
    assert headers["User-Agent"] == "agent"
    assert headers["X-Requested-With"] == "requested"
    assert headers["Origin"].startswith("https://")


def test_encode_polling_packet_formats_payload() -> None:
    """Encoding should prefix the payload length using ASCII digits."""

    packet = "40/message"
    encoded = ducaheat_ws._encode_polling_packet(packet)
    assert encoded == b"10:40/message"


def test_decode_polling_packets_handles_gzip() -> None:
    """Compressed Engine.IO payloads should be decompressed before decoding."""

    payload = b"40/message"
    length = len(payload)
    digits: list[int] = []
    while length:
        digits.insert(0, length % 10)
        length //= 10
    if not digits:
        digits = [0]
    body = bytes([0] + digits + [0xFF]) + payload
    decoded = ducaheat_ws._decode_polling_packets(body)
    assert decoded == ["40/message"]

    compressed = gzip.compress(body)
    decoded_gzip = ducaheat_ws._decode_polling_packets(compressed)
    assert decoded_gzip == ["40/message"]


def test_ducaheat_base_host_uses_brand_api_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The base host helper should derive the scheme and host from brand configuration."""

    client = _make_ducaheat_client(monkeypatch)
    monkeypatch.setattr(
        ducaheat_ws, "get_brand_api_base", lambda _: "https://ducaheat.example/api/v2"
    )
    assert client._base_host() == "https://ducaheat.example"


@pytest.mark.asyncio
async def test_ducaheat_ws_url_includes_token_and_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generating the websocket URL should include the token and device parameters."""

    client = _make_ducaheat_client(monkeypatch)
    monkeypatch.setattr(ducaheat_ws, "_rand_t", lambda: "Pabcdefg")
    monkeypatch.setattr(client, "_get_token", AsyncMock(return_value="token"))
    monkeypatch.setattr(
        ducaheat_ws, "get_brand_api_base", lambda _: "https://ducaheat.example"
    )

    ws_url = await client.ws_url()
    assert "token=token" in ws_url
    assert "dev_id=device" in ws_url
    assert ws_url.startswith("https://ducaheat.example")


def test_ducaheat_log_nodes_summary_includes_counts(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Logging nodes should record the node types and address counts."""

    client = _make_ducaheat_client(monkeypatch)
    caplog.set_level("INFO")
    client._log_nodes_summary({"htr": {"settings": {"1": {}, "2": {}}}})
    assert "htr" in caplog.text
    assert "2" in caplog.text


def test_termoweb_brand_headers_optional_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Brand headers should include requested-with and optional origin."""

    client = _make_termoweb_client(monkeypatch)
    headers = client._brand_headers(origin="https://app.example")
    assert headers["User-Agent"]
    assert headers["X-Requested-With"]
    assert headers["Origin"] == "https://app.example"


def test_termoweb_value_redaction_behaviour(monkeypatch: pytest.MonkeyPatch) -> None:
    """Redaction helpers should mask tokens and identifiers consistently."""

    client = _make_termoweb_client(monkeypatch)
    assert redact_token_fragment("  ") == ""
    assert redact_token_fragment("abcd") == "***"
    assert redact_token_fragment("abcdefgh") == "ab***gh"
    assert redact_token_fragment("abcdefghijk") == "abcd...hijk"
    assert mask_identifier("   ") == ""
    assert mask_identifier("xy") == "***"
    assert mask_identifier("abcdefgh") == "ab...gh"
    assert mask_identifier("abcdefghijkl") == "abcdef...ijkl"

    sanitised = client._sanitise_params({"token": "  ", "dev_id": "  ", "sid": "  "})
    assert sanitised["token"] == "{token}"
    assert sanitised["dev_id"] == "{dev_id}"
    assert sanitised["sid"] == "{sid}"


def test_termoweb_sanitise_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sensitive header, parameter and URL values should be redacted."""

    client = _make_termoweb_client(monkeypatch)
    headers = {
        "Authorization": "Bearer secret-token-value",
        "Cookie": "sid=123456789",
        "X-Test": "value",
        "Binary": b"token",
    }
    sanitised_headers = client._sanitise_headers(headers)
    assert sanitised_headers["Authorization"].startswith("Bearer ")
    assert "..." in sanitised_headers["Authorization"]
    assert sanitised_headers["Cookie"] != "sid=123456789"
    assert "..." in sanitised_headers["Cookie"]
    assert sanitised_headers["Binary"] == "token"
    params = {
        "token": "secrettoken",
        "dev_id": "device123456",
        "sid": "session123",
        "other": "keep",
    }
    sanitised_params = client._sanitise_params(params)
    assert sanitised_params["token"] == "{token}"
    assert sanitised_params["dev_id"] == "{dev_id}"
    assert sanitised_params["sid"] == "{sid}"
    url = "wss://example/ws?token=abc123456&dev_id=device123456&flag=1&sid=session123"
    sanitised_url = client._sanitise_url(url)
    sanitised_query = dict(parse_qsl(urlsplit(sanitised_url).query))
    assert sanitised_query["token"] == "{token}"
    assert sanitised_query["dev_id"] == "{dev_id}"
    assert sanitised_query["sid"] == "{sid}"
    ws_url = "wss://example/socket.io/1/websocket/session123?sid=session123"
    sanitised_ws_url = client._sanitise_url(ws_url)
    parsed_ws_url = urlsplit(sanitised_ws_url)
    ws_query = dict(parse_qsl(parsed_ws_url.query))
    assert parsed_ws_url.path.endswith("/socket.io/1/websocket/{sid}")
    assert ws_query["sid"] == "{sid}"
    assert client._sanitise_url("://bad url") == "://bad url"


def test_termoweb_mark_event_updates_state(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Receiving an event should refresh health tracking and state buckets."""

    loop = DummyLoop()
    client = _make_termoweb_client(monkeypatch, hass_loop=loop)
    caplog.set_level("DEBUG", logger=module._LOGGER.name)
    state_before = client._ws_state_bucket().copy()
    client._mark_event(paths=["/node/1"])
    state_after = client._ws_state_bucket()
    assert state_after["last_event_at"] != state_before.get("last_event_at")
    assert state_after["events_total"] == 1
    assert client._healthy_since is not None
    assert client._stats.last_paths == ["/node/1"]


@pytest.mark.asyncio
async def test_termoweb_idle_restart_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Idle restart scheduling should close the socket and clear pending flags."""

    loop = DummyLoop()
    client = _make_termoweb_client(monkeypatch, hass_loop=loop)
    client._disconnect = AsyncMock()  # type: ignore[attr-defined]
    client._closing = False
    client._schedule_idle_restart(idle_for=300.0, source="test idle")
    assert client._idle_restart_pending is True
    assert client._ws_state_bucket()["idle_restart_pending"] is True
    assert loop.created_tasks
    task = loop.created_tasks[0]
    await task.run()
    client._disconnect.assert_awaited()
    assert client._idle_restart_task is None
    assert client._idle_restart_pending is False
    assert client._ws_state_bucket()["idle_restart_pending"] is False


@pytest.mark.asyncio
async def test_termoweb_cancel_idle_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancelling an idle restart should reset pending flags."""

    loop = DummyLoop()
    client = _make_termoweb_client(monkeypatch, hass_loop=loop)
    client._closing = False
    client._schedule_idle_restart(idle_for=120.0, source="test idle")
    assert client._idle_restart_pending is True
    client._cancel_idle_restart()
    assert client._idle_restart_pending is False
    assert client._ws_state_bucket()["idle_restart_pending"] is False


@pytest.mark.asyncio
async def test_termoweb_get_token_from_rest(monkeypatch: pytest.MonkeyPatch) -> None:
    """The websocket client should reuse REST client authorization tokens."""

    client = _make_termoweb_client(monkeypatch)
    rest_client = client._client
    rest_client.authed_headers = AsyncMock(  # type: ignore[attr-defined]
        return_value={"Authorization": "Bearer newtoken"}
    )
    token = await client._get_token()
    assert token == "newtoken"


@pytest.mark.asyncio
async def test_termoweb_get_token_missing_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing authorization headers should raise an error."""

    client = _make_termoweb_client(monkeypatch)
    rest_client = client._client
    rest_client.authed_headers = AsyncMock(return_value={})  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError):
        await client._get_token()


def test_termoweb_wrap_background_task_handles_sync_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Background task wrapper should handle synchronous callables."""

    loop = DummyLoop()
    client = _make_termoweb_client(monkeypatch, hass_loop=loop)

    task = client._wrap_background_task(lambda value: value + 1, 4)
    assert isinstance(task, DummyTask)
    assert loop.created_tasks
    assert task is loop.created_tasks[0]
    assert task.done() is False

    async def _async_target(value: int) -> int:
        return value * 2

    async_task = client._wrap_background_task(_async_target, 6)
    assert len(loop.created_tasks) == 2
    assert async_task is loop.created_tasks[1]

    for created in loop.created_tasks:
        created.cancel()


def test_termoweb_start_reuses_existing_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """Starting when already running should return the existing task."""

    loop = DummyLoop()
    client = _make_termoweb_client(monkeypatch, hass_loop=loop)

    async def _noop() -> None:
        return None

    existing = DummyTask(_noop())
    client._task = existing

    task = client.start()
    assert task is existing
    assert not loop.created_tasks
    existing.cancel()


def test_termoweb_start_creates_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """Starting without a running task should schedule the runner."""

    loop = DummyLoop()
    client = _make_termoweb_client(monkeypatch, hass_loop=loop)
    runner = AsyncMock()
    client._runner = runner  # type: ignore[assignment]

    task = client.start()
    assert task is loop.created_tasks[0]
    assert runner.await_count == 0
    task.cancel()


@pytest.mark.asyncio
async def test_termoweb_stop_cancels_background_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stopping should cancel scheduled background tasks and disconnect."""

    running_loop = asyncio.get_running_loop()
    hass_loop = SimpleNamespace(
        create_task=lambda coro, **kwargs: running_loop.create_task(coro, **kwargs),
        call_soon_threadsafe=lambda cb, *args: running_loop.call_soon(cb, *args),
    )
    client = _make_termoweb_client(monkeypatch, hass_loop=hass_loop)
    client._disconnect = AsyncMock()  # type: ignore[attr-defined]
    client._idle_restart_task = asyncio.create_task(asyncio.sleep(0))
    client._idle_monitor_task = asyncio.create_task(asyncio.sleep(0))
    client._task = asyncio.create_task(asyncio.sleep(0))
    client._idle_restart_pending = True
    client._subscription_refresh_failed = True

    await asyncio.sleep(0)
    await client.stop()

    assert client._idle_restart_task is None
    assert client._idle_monitor_task is None
    assert client._task is None
    assert client._idle_restart_pending is False
    assert client._subscription_refresh_failed is False
    client._disconnect.assert_awaited()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_termoweb_debug_probe_emits_when_debug_enabled(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Debug probe should emit when debug logging is enabled."""

    client = _make_termoweb_client(monkeypatch)
    emit = AsyncMock()
    client._sio.emit = emit  # type: ignore[attr-defined]
    caplog.set_level(logging.DEBUG, logger=module._LOGGER.name)

    await client.debug_probe()

    emit.assert_awaited_once_with("dev_data", namespace=module.WS_NAMESPACE)


def test_termoweb_update_status_records_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Status updates should populate the hass data bucket and dispatch events."""

    client = _make_termoweb_client(monkeypatch)
    client._stats.frames_total = 4  # type: ignore[attr-defined]
    client._stats.events_total = 2  # type: ignore[attr-defined]
    client._stats.last_event_ts = 50.0  # type: ignore[attr-defined]
    client._healthy_since = 40.0
    monkeypatch.setattr(module.time, "time", lambda: 100.0)

    client._update_status("connected")

    state = client._ws_state_bucket()
    assert state["status"] == "connected"
    assert state["frames_total"] == 4
    assert state["events_total"] == 2
    assert state["healthy_minutes"] == 1
    client._dispatcher_mock.assert_called()  # type: ignore[attr-defined]


def test_termoweb_mark_event_without_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Count-only events should still trigger healthy transitions."""

    client = _make_termoweb_client(monkeypatch)
    client._update_status = MagicMock()  # type: ignore[attr-defined]
    monkeypatch.setattr(module.time, "time", lambda: 300.0)

    client._mark_event(paths=None, count_event=True)

    assert client._stats.events_total == 1  # type: ignore[attr-defined]
    assert client._healthy_since == 300.0
    client._update_status.assert_called_once_with("healthy")  # type: ignore[attr-defined]


def test_termoweb_update_status_prefers_stats_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Healthy updates should fall back to stats timestamps when available."""

    client = _make_termoweb_client(monkeypatch)
    client._stats.frames_total = 1  # type: ignore[attr-defined]
    client._stats.events_total = 1  # type: ignore[attr-defined]
    client._stats.last_event_ts = 75.0  # type: ignore[attr-defined]
    client._last_event_at = None
    monkeypatch.setattr(module.time, "time", lambda: 100.0)

    client._update_status("healthy")

    state = client._ws_state_bucket()
    assert state["last_event_at"] == 75.0
    assert state["healthy_since"] == 75.0
    client._dispatcher_mock.assert_called()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_termoweb_force_refresh_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force refreshing tokens should clear cached access tokens."""

    client = _make_termoweb_client(monkeypatch)
    rest_client = client._client
    rest_client._access_token = "cached"  # type: ignore[attr-defined]

    await client._force_refresh_token()

    assert rest_client._access_token is None  # type: ignore[attr-defined]
    rest_client._ensure_token.assert_awaited()  # type: ignore[attr-defined]


def test_termoweb_api_base_prefers_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """API base helper should prefer the REST client's configured base."""

    client = _make_termoweb_client(monkeypatch)
    client._client.api_base = "https://example/api"  # type: ignore[attr-defined]
    assert client._api_base() == "https://example/api"


def test_termoweb_api_base_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """API base helper should fall back to the integration constant."""

    client = _make_termoweb_client(monkeypatch)
    client._client.api_base = ""  # type: ignore[attr-defined]
    assert client._api_base() == module.API_BASE


def test_termoweb_schedule_idle_restart_skips_when_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Idle restart scheduling should not occur when closing or already pending."""

    loop = DummyLoop()
    client = _make_termoweb_client(monkeypatch, hass_loop=loop)
    client._closing = True
    client._schedule_idle_restart(idle_for=10.0, source="closing")
    assert not loop.created_tasks

    client._closing = False
    client._idle_restart_pending = True
    client._schedule_idle_restart(idle_for=10.0, source="pending")
    assert not loop.created_tasks


def test_ducaheat_client_start_reuses_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ducaheat client should reuse existing tasks when running."""

    loop = DummyLoop()
    client = _make_ducaheat_client(monkeypatch, hass_loop=loop)

    async def _noop() -> None:
        return None

    existing = DummyTask(_noop())
    client._task = existing  # type: ignore[attr-defined]

    task = client.start()
    assert task is existing
    assert not loop.created_tasks
    existing.cancel()


def test_ducaheat_client_start_creates_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ducaheat client start should create a task when idle."""

    loop = DummyLoop()
    client = _make_ducaheat_client(monkeypatch, hass_loop=loop)
    runner = AsyncMock()
    client._runner = runner  # type: ignore[attr-defined]

    task = client.start()
    assert task is loop.created_tasks[0]
    task.cancel()


@pytest.mark.asyncio
async def test_ducaheat_client_stop_cancels_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stopping the Ducaheat client should cancel background tasks."""

    running_loop = asyncio.get_running_loop()
    hass_loop = SimpleNamespace(
        create_task=lambda coro, **kwargs: running_loop.create_task(coro, **kwargs),
        call_soon_threadsafe=lambda cb, *args: running_loop.call_soon(cb, *args),
    )
    client = _make_ducaheat_client(monkeypatch, hass_loop=hass_loop)
    client._disconnect = AsyncMock()  # type: ignore[attr-defined]
    client._task = asyncio.create_task(asyncio.sleep(0))

    await asyncio.sleep(0)
    await client.stop()

    assert client._task is None
    client._disconnect.assert_awaited_once_with("stop")  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_connection_rate_limiter_enforces_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Connection limiter should enforce spacing and rolling windows."""

    sleeps: list[float] = []
    now = 0.0

    async def _fake_sleep(delay: float) -> None:
        nonlocal now
        sleeps.append(delay)
        now += delay

    def _fake_clock() -> float:
        return now

    limiter = base_ws.ConnectionRateLimiter(
        min_interval=1.0,
        max_attempts=2,
        window_seconds=3.0,
        clock=_fake_clock,
        sleeper=_fake_sleep,
    )

    await limiter.wait_for_slot()
    await limiter.wait_for_slot()
    await limiter.wait_for_slot()

    assert sleeps == [1.0, 2.0]


def test_ducaheat_path_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ducaheat websocket path helper should return the Engine.IO path."""

    client = _make_ducaheat_client(monkeypatch)
    assert client._path() == "/socket.io/"


def test_ws_lease_backoff_sequence() -> None:
    """Backoff helper should iterate through the configured sequence."""

    lease = base_ws._WsLeaseMixin()
    values = [lease._next_backoff() for _ in range(6)]
    assert values == [5, 10, 30, 120, 300, 300]
    lease._reset_backoff()
    assert lease._next_backoff() == 5


@pytest.mark.asyncio
async def test_termoweb_runner_uses_connection_limiter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TermoWeb runner should throttle connection attempts before dialing."""

    client = _make_termoweb_client(monkeypatch)
    limiter = SimpleNamespace(wait_for_slot=AsyncMock())
    client._connect_limiter = limiter  # type: ignore[attr-defined]

    attempts = 0

    async def _connect_once() -> None:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("boom")

    async def _disconnect(reason: str) -> None:
        return None

    async def _handle_connection_lost(error: Exception | None) -> None:
        client._closing = True

    monkeypatch.setattr(client, "_connect_once", _connect_once)
    monkeypatch.setattr(client, "_wait_for_events", AsyncMock())
    monkeypatch.setattr(client, "_disconnect", _disconnect)
    monkeypatch.setattr(client, "_handle_connection_lost", _handle_connection_lost)
    monkeypatch.setattr(client, "_update_status", lambda *_args, **_kwargs: None)
    client._backoff_seq = [0]

    await client._runner()

    limiter.wait_for_slot.assert_awaited_once()
    assert attempts == 1


@pytest.mark.asyncio
async def test_ducaheat_runner_uses_connection_limiter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ducaheat runner should throttle connection attempts before dialing."""

    client = _make_ducaheat_client(monkeypatch)
    limiter = SimpleNamespace(wait_for_slot=AsyncMock())
    client._connect_limiter = limiter  # type: ignore[attr-defined]

    attempts = 0

    async def _connect_once() -> None:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("boom")

    async def _disconnect(reason: str) -> None:
        raise asyncio.CancelledError()

    monkeypatch.setattr(client, "_connect_once", _connect_once)
    monkeypatch.setattr(client, "_read_loop_ws", AsyncMock())
    monkeypatch.setattr(client, "_disconnect", _disconnect)
    monkeypatch.setattr(client, "_update_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(client, "_next_backoff", lambda: 0)

    with pytest.raises(asyncio.CancelledError):
        await client._runner()

    limiter.wait_for_slot.assert_awaited_once()
    assert attempts == 1


def test_ws_common_state_bucket(
    monkeypatch: pytest.MonkeyPatch,
    ws_common_stub: Callable[..., base_ws._WSCommon],
) -> None:
    """WS common helper should create domain buckets when missing."""

    hass = SimpleNamespace(data={base_ws.DOMAIN: {}})
    runtime = build_entry_runtime(
        hass=hass,
        entry_id="entry",
        dev_id="dev",
    )

    dummy = ws_common_stub(
        hass=hass,
    )
    bucket = dummy._ws_state_bucket()
    assert bucket == {}
    assert runtime.ws_state == {"dev": {}}


def test_ws_common_update_status_dispatches(
    monkeypatch: pytest.MonkeyPatch,
    ws_common_stub: Callable[..., base_ws._WSCommon],
) -> None:
    """WS common status helper should forward dispatcher signals."""

    hass = SimpleNamespace(data={base_ws.DOMAIN: {}})
    build_entry_runtime(
        hass=hass,
        entry_id="entry",
        dev_id="dev",
    )
    coordinator = SimpleNamespace(
        update_nodes=MagicMock(),
        update_gateway_connection=MagicMock(),
    )
    dispatcher = MagicMock()
    monkeypatch.setattr(base_ws, "async_dispatcher_send", dispatcher)

    dummy = ws_common_stub(
        hass=hass,
        coordinator=coordinator,
    )
    dummy._update_status("connected")

    dispatcher.assert_called_once()
    coordinator.update_gateway_connection.assert_called_once()


def test_ws_common_dispatch_nodes(
    monkeypatch: pytest.MonkeyPatch,
    ws_common_stub: Callable[..., base_ws._WSCommon],
) -> None:
    """WS common dispatch should update coordinator and emit dispatcher events."""

    raw_nodes = {"nodes": [{"type": "htr", "addr": "1"}]}
    inventory_nodes = build_node_inventory(raw_nodes)
    inventory_obj = Inventory("dev", inventory_nodes)

    hass = SimpleNamespace(data={base_ws.DOMAIN: {}})
    build_entry_runtime(
        hass=hass,
        entry_id="entry",
        dev_id="dev",
        inventory=inventory_obj,
    )
    coordinator = SimpleNamespace(update_nodes=MagicMock(), dev_id="dev")
    dispatcher = MagicMock()
    monkeypatch.setattr(base_ws, "async_dispatcher_send", dispatcher)

    monkeypatch.setattr(
        base_ws, "resolve_record_inventory", lambda *_, **__: None, raising=False
    )

    dummy = ws_common_stub(
        hass=hass,
        coordinator=coordinator,
    )
    dummy._inventory = None
    payload = {"nodes": raw_nodes}
    dummy._dispatch_nodes(payload)

    coordinator.update_nodes.assert_not_called()
    dispatcher.assert_called_once()
    dispatched_payload = dispatcher.call_args.args[2]
    assert dispatched_payload == {
        "dev_id": "dev",
        "node_type": None,
        "inventory": inventory_obj,
    }


def test_ws_client_start_selects_delegate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Top-level websocket client should instantiate the appropriate delegate."""

    hass = SimpleNamespace(
        loop=SimpleNamespace(create_task=lambda coro, **_: DummyTask(coro))
    )
    rest_client = DummyREST()
    coordinator = SimpleNamespace(update_nodes=MagicMock())

    start_mock = MagicMock(return_value="started")

    class StubTermo:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

        def start(self) -> str:
            return start_mock()

    fake_termoweb = ModuleType(module.__name__)
    fake_termoweb.__dict__.update(module.__dict__)
    fake_termoweb.TermoWebWSClient = StubTermo
    monkeypatch.setitem(sys.modules, module.__name__, fake_termoweb)
    monkeypatch.setattr(module, "TermoWebWSClient", StubTermo)
    monkeypatch.setattr(base_ws, "TermoWebWSClient", StubTermo, raising=False)

    client = base_ws.WebSocketClient(
        hass,
        entry_id="entry",
        dev_id="dev",
        api_client=rest_client,
        coordinator=coordinator,
        session=SimpleNamespace(),
    )

    result = client.start()
    assert result == "started"
    start_mock.assert_called_once()


def test_ws_client_start_selects_ducaheat(monkeypatch: pytest.MonkeyPatch) -> None:
    """Top-level websocket client should delegate to the Ducaheat client when flagged."""

    hass = SimpleNamespace(
        loop=SimpleNamespace(create_task=lambda coro, **_: DummyTask(coro))
    )
    rest_client = DummyREST(is_ducaheat=True)
    coordinator = SimpleNamespace(update_nodes=MagicMock())

    start_mock = MagicMock(return_value="started")

    class StubDucaheat:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

        def start(self) -> str:
            return start_mock()

    fake_ducaheat = ModuleType(ducaheat_ws.__name__)
    fake_ducaheat.__dict__.update(ducaheat_ws.__dict__)
    fake_ducaheat.DucaheatWSClient = StubDucaheat
    monkeypatch.setitem(sys.modules, ducaheat_ws.__name__, fake_ducaheat)
    monkeypatch.setattr(ducaheat_ws, "DucaheatWSClient", StubDucaheat)
    monkeypatch.setattr(base_ws, "DucaheatWSClient", StubDucaheat, raising=False)

    client = base_ws.WebSocketClient(
        hass,
        entry_id="entry",
        dev_id="dev",
        api_client=rest_client,
        coordinator=coordinator,
        session=SimpleNamespace(),
    )

    result = client.start()
    assert result == "started"
    start_mock.assert_called_once()


def test_ws_client_is_running_and_ws_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Top-level helpers should proxy to the delegate when available."""

    hass = SimpleNamespace(
        loop=SimpleNamespace(create_task=lambda coro, **_: DummyTask(coro))
    )
    rest_client = DummyREST()
    coordinator = SimpleNamespace(update_nodes=MagicMock())

    async def _noop() -> None:
        return None

    class StubDelegate:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._running = True

        def start(self) -> DummyTask:
            return DummyTask(_noop())

        def is_running(self) -> bool:
            return self._running

        async def ws_url(self) -> str:
            return "wss://example"

        async def stop(self) -> None:
            self._running = False

    fake_termoweb = ModuleType(module.__name__)
    fake_termoweb.__dict__.update(module.__dict__)
    fake_termoweb.TermoWebWSClient = StubDelegate
    monkeypatch.setitem(sys.modules, module.__name__, fake_termoweb)
    monkeypatch.setattr(module, "TermoWebWSClient", StubDelegate)
    monkeypatch.setattr(base_ws, "TermoWebWSClient", StubDelegate, raising=False)

    client = base_ws.WebSocketClient(
        hass,
        entry_id="entry",
        dev_id="dev",
        api_client=rest_client,
        coordinator=coordinator,
        session=SimpleNamespace(),
    )

    task = client.start()
    assert client.is_running() is True
    url = asyncio.run(client.ws_url())
    assert url == "wss://example"
    task.cancel()
