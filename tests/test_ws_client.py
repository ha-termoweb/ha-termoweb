"""Unit tests for websocket client helpers."""

from __future__ import annotations

import asyncio
import gzip
from types import MappingProxyType, ModuleType, SimpleNamespace
from typing import Any, Mapping
from urllib.parse import parse_qsl, urlsplit
from unittest.mock import AsyncMock, MagicMock

import pytest

import logging
import sys

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


class DummyLoop:
    """Simple event loop stub recording created tasks."""

    def __init__(self) -> None:
        self.created_tasks: list[DummyTask] = []

    def create_task(self, coro: Any, **_: Any) -> "DummyTask":
        task = DummyTask(coro)
        self.created_tasks.append(task)
        return task

    def call_soon_threadsafe(self, callback: Any, *args: Any) -> None:
        callback(*args)


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

    async def run(self) -> Any:
        try:
            return await self.coro
        finally:
            self._completed = True


@pytest.fixture(autouse=True)
def patch_async_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ``socketio.AsyncClient`` with a controllable stub."""

    class StubAsyncClient:
        def __init__(self, **_: Any) -> None:
            self.events: dict[tuple[str, str | None], Any] = {}

        def on(self, event: str, *, handler: Any, namespace: str | None = None) -> None:
            self.events[(event, namespace)] = handler

        async def emit(
            self,
            event: str,
            data: Any | None = None,
            *,
            namespace: str | None = None,
        ) -> None:  # pragma: no cover - only used for safety
            self.events[(event, namespace)] = (event, data)

    monkeypatch.setattr(module.socketio, "AsyncClient", StubAsyncClient)


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

    hass = SimpleNamespace(loop=hass_loop, data={module.DOMAIN: {"entry": {}}})
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
    hass = SimpleNamespace(loop=hass_loop, data={module.DOMAIN: {"entry": {}}})
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


def test_forward_ws_sample_updates_guards_and_invalid_lease() -> None:
    """Guard clauses and invalid lease values should be handled safely."""

    hass = SimpleNamespace(data={base_ws.DOMAIN: {}})
    base_ws.forward_ws_sample_updates(
        hass,
        "entry",
        "dev",
        {"pmo": {"samples": {"7": {"power": 1}}}},
    )

    hass.data[base_ws.DOMAIN]["entry"] = {
        "energy_coordinator": SimpleNamespace(),
    }
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
    hass.data[base_ws.DOMAIN]["entry"]["energy_coordinator"] = coordinator

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

    hass = SimpleNamespace(data={base_ws.DOMAIN: {"entry": {}}})
    raw_nodes = {"nodes": [{"type": "pmo", "addr": "7", "name": "PM"}]}
    inventory = Inventory("dev", raw_nodes, build_node_inventory(raw_nodes))
    handler = MagicMock()
    hass.data[base_ws.DOMAIN]["entry"] = {
        "energy_coordinator": SimpleNamespace(handle_ws_samples=handler),
        "inventory": inventory,
        "nodes": raw_nodes,
    }

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


def test_forward_ws_sample_updates_uses_coordinator_inventory(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Coordinator inventory aliases and logging should be applied."""

    raw_nodes = {"nodes": [{"type": "htr", "addr": "5"}]}
    inventory = Inventory("dev", raw_nodes, build_node_inventory(raw_nodes))

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
    hass = SimpleNamespace(
        data={
            base_ws.DOMAIN: {
                "entry": {
                    "energy_coordinator": SimpleNamespace(handle_ws_samples=handler),
                    "coordinator": SimpleNamespace(inventory=inventory),
                }
            }
        }
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
    hass = SimpleNamespace(
        data={
            base_ws.DOMAIN: {
                "entry": {
                    "energy_coordinator": SimpleNamespace(handle_ws_samples=handler),
                    "inventory": Inventory("dev", {}, []),
                }
            }
        }
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
    hass = SimpleNamespace(
        data={
            base_ws.DOMAIN: {
                "entry": {
                    "energy_coordinator": SimpleNamespace(handle_ws_samples=handler),
                    "inventory": Inventory("dev", {}, []),
                }
            }
        }
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
    inventory = Inventory("dev", raw_nodes, build_node_inventory(raw_nodes))

    monkeypatch.setattr(
        Inventory,
        "heater_sample_address_map",
        property(lambda self: ({"htr": ["1"]}, ["bad"])),
    )
    monkeypatch.setattr(
        Inventory,
        "power_monitor_sample_address_map",
        property(lambda self: ({"pmo": ["3"]}, ["invalid"])),
    )

    handler = MagicMock()
    hass = SimpleNamespace(
        data={
            base_ws.DOMAIN: {
                "entry": {
                    "energy_coordinator": SimpleNamespace(handle_ws_samples=handler),
                    "inventory": inventory,
                }
            }
        }
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
    assert module.DOMAIN in client.hass.data
    assert client.hass.data[module.DOMAIN]["entry"]["ws_state"]["device"] is bucket


def test_collect_update_addresses_extracts() -> None:
    """Ensure update address extraction returns sorted node/type pairs."""

    nodes = {
        "htr": {"settings": {"1": {"temp": 20}, "2": None}},
        "aux": {"samples": {"3": 10}},
    }
    addresses = module.WebSocketClient._collect_update_addresses(nodes)
    assert addresses == [("aux", "3"), ("htr", "1")]


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


def test_dispatch_nodes_without_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve inventory records when no snapshot exists."""

    hass_record: dict[str, Any] = {"dev_id": "device"}
    hass = SimpleNamespace(data={base_ws.DOMAIN: {"entry": hass_record}})
    coordinator = SimpleNamespace(update_nodes=MagicMock(), dev_id="dev")
    dispatcher = MagicMock()
    monkeypatch.setattr(base_ws, "async_dispatcher_send", dispatcher)
    seen: dict[str, Any] = {}

    payload = {"nodes": [{"addr": "1", "type": "htr"}]}
    node_inventory = build_node_inventory(payload["nodes"])
    inventory = Inventory("device", payload["nodes"], node_inventory)

    def fake_resolve(
        record_map: Mapping[str, Any] | None,
        *,
        dev_id: str | None = None,
        nodes_payload: Any | None = None,
        **_: Any,
    ) -> Any:
        seen["record"] = record_map
        seen["dev_id"] = dev_id
        seen["nodes_payload"] = nodes_payload
        return SimpleNamespace(
            inventory=inventory,
            source="test",
            raw_count=len(node_inventory),
            filtered_count=len(node_inventory),
        )

    monkeypatch.setattr(base_ws, "resolve_record_inventory", fake_resolve)
    monkeypatch.setattr(
        base_ws,
        "addresses_by_node_type",
        lambda nodes, **_: ({"htr": ["1"]}, set()),
    )

    class DummyCommon(base_ws._WSCommon):
        def __init__(self) -> None:
            self.hass = hass
            self.entry_id = "entry"
            self.dev_id = "device"
            self._coordinator = coordinator
            self._inventory = None

    DummyCommon()._dispatch_nodes(payload)

    assert seen["record"] is hass_record
    assert seen["dev_id"] == "dev"
    assert seen["nodes_payload"] is payload["nodes"]
    coordinator.update_nodes.assert_called_once()
    update_args = coordinator.update_nodes.call_args.args
    assert update_args[0] is None
    inventory_arg = update_args[1]
    assert isinstance(inventory_arg, Inventory)
    assert inventory_arg.payload is payload["nodes"]

    def _extract_addr(node: Any) -> str:
        addr_attr = getattr(node, "addr", None)
        if addr_attr is not None:
            return addr_attr
        if isinstance(node, Mapping):
            return str(node.get("addr", ""))
        return str(node)

    assert [_extract_addr(node) for node in inventory_arg.nodes] == ["1"]
    dispatcher.assert_called_once()
    dispatched_payload = dispatcher.call_args.args[2]
    assert dispatched_payload["addresses_by_type"] == {"htr": ["1"]}
    assert dispatched_payload.get("addr_map") is None
    assert "nodes" not in dispatched_payload


def test_prepare_nodes_dispatch_handles_non_mapping_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Node dispatch helper should tolerate non-mapping hass records."""

    hass = SimpleNamespace(data={base_ws.DOMAIN: {"entry": []}})
    coordinator = SimpleNamespace(update_nodes=MagicMock(), dev_id="dev")
    monkeypatch.setattr(
        base_ws, "addresses_by_node_type", lambda nodes, **_: ({}, set())
    )
    seen: dict[str, Any] = {}

    def fake_resolve(
        record_map: Mapping[str, Any] | None,
        *,
        dev_id: str | None = None,
        nodes_payload: Any | None = None,
        **_: Any,
    ) -> Any:
        seen["record"] = record_map
        seen["dev_id"] = dev_id
        seen["nodes_payload"] = nodes_payload
        return SimpleNamespace(inventory=None, source="fallback", raw_count=0, filtered_count=0)

    monkeypatch.setattr(base_ws, "resolve_record_inventory", fake_resolve)

    context = base_ws._prepare_nodes_dispatch(
        hass,
        entry_id="entry",
        coordinator=coordinator,
        raw_nodes={},
    )

    assert seen["record"] is None
    assert seen["dev_id"] == "dev"
    assert seen["nodes_payload"] == {}
    assert context.record is None
    assert isinstance(context.inventory, Inventory)
    assert context.inventory.nodes == ()


def test_prepare_nodes_dispatch_populates_record_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback inventory creation should store the container on the record."""

    hass_record: dict[str, Any] = {"dev_id": "dev"}
    hass = SimpleNamespace(data={base_ws.DOMAIN: {"entry": hass_record}})
    coordinator = SimpleNamespace(update_nodes=MagicMock(), dev_id="dev")
    monkeypatch.setattr(
        base_ws, "addresses_by_node_type", lambda nodes, **_: ({}, set())
    )

    def fake_resolve(
        record_map: Mapping[str, Any] | None,
        *,
        dev_id: str | None = None,
        nodes_payload: Any | None = None,
        **_: Any,
    ) -> Any:
        assert record_map is hass_record
        assert dev_id == "dev"
        assert nodes_payload == {}
        return SimpleNamespace(inventory=None, source="fallback", raw_count=0, filtered_count=0)

    monkeypatch.setattr(base_ws, "resolve_record_inventory", fake_resolve)

    context = base_ws._prepare_nodes_dispatch(
        hass,
        entry_id="entry",
        coordinator=coordinator,
        raw_nodes={},
    )

    assert isinstance(hass_record.get("inventory"), Inventory)
    assert context.inventory is hass_record["inventory"]


def test_prepare_nodes_dispatch_uses_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Existing inventory objects should be reused by the dispatch helper."""

    hass = SimpleNamespace(data={base_ws.DOMAIN: {"entry": {}}})
    coordinator = SimpleNamespace(update_nodes=MagicMock(), dev_id="dev")
    node_inventory = build_node_inventory([{"type": "htr", "addr": "4"}])
    inventory = Inventory("dev", {"nodes": [{"type": "htr", "addr": "4"}]}, node_inventory)
    monkeypatch.setattr(
        base_ws, "addresses_by_node_type", lambda nodes, **_: ({"htr": ["4"]}, set())
    )

    def _fail(*_: Any, **__: Any) -> Any:
        raise AssertionError("resolve_record_inventory should not be called")

    monkeypatch.setattr(base_ws, "resolve_record_inventory", _fail)

    context = base_ws._prepare_nodes_dispatch(
        hass,
        entry_id="entry",
        coordinator=coordinator,
        raw_nodes=None,
        inventory=inventory,
    )

    assert context.inventory is inventory
    coordinator.update_nodes.assert_called_once_with(None, inventory)


def test_prepare_nodes_dispatch_resolves_record_dev_id_and_coordinator_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Record dev IDs and coordinator inventory should be applied."""

    monkeypatch.setattr(
        base_ws,
        "addresses_by_node_type",
        lambda nodes, **_: ({}, set()),
    )
    resolve_calls: list[tuple[Mapping[str, Any] | None, str | None, Any]] = []

    def fake_resolve(
        record_map: Mapping[str, Any] | None,
        *,
        dev_id: str | None = None,
        nodes_payload: Any | None = None,
        **_: Any,
    ) -> Any:
        resolve_calls.append((record_map, dev_id, nodes_payload))
        return SimpleNamespace(inventory=None, source="fallback", raw_count=0, filtered_count=0)

    monkeypatch.setattr(base_ws, "resolve_record_inventory", fake_resolve)

    hass_str = SimpleNamespace(data={base_ws.DOMAIN: {"entry": {"dev_id": "raw"}}})
    coordinator_str = SimpleNamespace(update_nodes=MagicMock())
    context_str = base_ws._prepare_nodes_dispatch(
        hass_str,
        entry_id="entry",
        coordinator=coordinator_str,
        raw_nodes={},
    )
    assert context_str.inventory.dev_id == "raw"

    inventory = Inventory("dev", {}, [])
    hass_numeric_record: dict[str, Any] = {"dev_id": 99}
    hass_numeric = SimpleNamespace(data={base_ws.DOMAIN: {"entry": hass_numeric_record}})
    coordinator_numeric = SimpleNamespace(update_nodes=MagicMock(), inventory=inventory)
    context_numeric = base_ws._prepare_nodes_dispatch(
        hass_numeric,
        entry_id="entry",
        coordinator=coordinator_numeric,
        raw_nodes={"nodes": []},
    )

    assert context_numeric.inventory is inventory
    assert hass_numeric_record["inventory"] is inventory
    update_args = coordinator_numeric.update_nodes.call_args.args
    assert update_args[0] is None
    assert update_args[1] is inventory
    assert resolve_calls[0] == (hass_str.data[base_ws.DOMAIN]["entry"], "raw", {})
    assert len(resolve_calls) == 1


def test_ws_status_tracker_applies_default_cadence_hint() -> None:
    """Creating a tracker should immediately apply the cadence hint."""

    class Dummy(base_ws._WSStatusMixin):
        def __init__(self) -> None:
            self.hass = SimpleNamespace(data={base_ws.DOMAIN: {"entry": {}}})
            self.entry_id = "entry"
            self.dev_id = "dev"
            self._apply_payload_window_hint = MagicMock()

    dummy = Dummy()
    tracker = dummy._ws_health_tracker()

    assert isinstance(tracker, base_ws.WsHealthTracker)
    dummy._apply_payload_window_hint.assert_called_once_with(
        source="cadence",
        lease_seconds=120,
        candidates=[30, 75, "90"],
    )


def test_ws_status_tracker_processes_pending_cadence_hint() -> None:
    """Deferred cadence hints should execute when suppression is active."""

    class Dummy(base_ws._WSStatusMixin):
        def __init__(self) -> None:
            self.hass = SimpleNamespace(data={base_ws.DOMAIN: {"entry": {}}})
            self.entry_id = "entry"
            self.dev_id = "dev"
            self._apply_payload_window_hint = MagicMock()
            self._suppress_default_cadence_hint = True
            self._pending_default_cadence_hint = True

    dummy = Dummy()
    dummy._ws_health_tracker()

    dummy._apply_payload_window_hint.assert_called_once()
    assert dummy._pending_default_cadence_hint is False


def test_ws_common_ensure_type_bucket_handles_invalid_inputs() -> None:
    """Ensure type bucket helper should guard against invalid structures."""

    class Dummy(base_ws._WSCommon):
        def __init__(self) -> None:
            self.hass = SimpleNamespace(data={base_ws.DOMAIN: {"entry": {}}})
            self.entry_id = "entry"
            self.dev_id = "dev"
            self._coordinator = SimpleNamespace()
            base_ws._WSCommon.__init__(self, inventory=None)

    dummy = Dummy()
    assert dummy._ensure_type_bucket([], "htr") == {}
    assert dummy._ensure_type_bucket({}, "", dev_map=None) == {}

    bucket = dummy._ensure_type_bucket({}, "htr", dev_map=None)
    assert bucket["addrs"] == []
    assert bucket["settings"] == {}
    assert bucket["samples"] == {}
    assert bucket["status"] == {}
    assert bucket["advanced"] == {}


def test_ws_common_ensure_type_bucket_populates_dev_map() -> None:
    """Ensure type bucket helper should normalise buckets and dev maps."""

    class Dummy(base_ws._WSCommon):
        def __init__(self) -> None:
            self.hass = SimpleNamespace(data={base_ws.DOMAIN: {"entry": {}}})
            self.entry_id = "entry"
            self.dev_id = "dev"
            self._coordinator = SimpleNamespace()
            base_ws._WSCommon.__init__(self, inventory=None)

    dummy = Dummy()
    nodes_by_type = {
        "htr": MappingProxyType(
            {
                "addrs": ("1",),
                "settings": MappingProxyType({"mode": "auto"}),
                "samples": {"temp": 20},
                "status": MappingProxyType({"on": True}),
            }
        )
    }
    dev_map = {
        "settings": MappingProxyType({}),
        "nodes_by_type": MappingProxyType({}),
    }

    bucket = dummy._ensure_type_bucket(nodes_by_type, "htr", dev_map=dev_map)
    assert bucket["addrs"] == ["1"]
    assert bucket["samples"] == {"temp": 20}
    assert bucket["settings"] == {"mode": "auto"}
    assert bucket["advanced"] == {}
    assert dev_map["addresses_by_type"]["htr"] == []
    assert dev_map["settings"]["htr"] == {}
    assert dev_map["nodes_by_type"]["htr"] is bucket

    dev_map_second = {
        "addresses_by_type": MappingProxyType({"htr": ("2",)}),
        "settings": {"htr": {"existing": 1}},
        "nodes_by_type": {},
    }
    bucket_again = dummy._ensure_type_bucket({"htr": bucket}, "htr", dev_map=dev_map_second)
    assert bucket_again is bucket
    assert dev_map_second["addresses_by_type"]["htr"] == ["2"]
    assert dev_map_second["nodes_by_type"]["htr"] is bucket

    dev_map_third = {
        "addresses_by_type": {"htr": ["3"]},
        "settings": {},
        "nodes_by_type": {},
    }
    dummy._ensure_type_bucket({"htr": bucket}, "htr", dev_map=dev_map_third)
    assert dev_map_third["addresses_by_type"]["htr"] == ["3"]

    dev_map_missing = {
        "addresses_by_type": {},
        "settings": MappingProxyType({"htr": MappingProxyType({"mode": "sched"})}),
    }
    bucket_missing = dummy._ensure_type_bucket({"htr": bucket}, "htr", dev_map=dev_map_missing)
    assert bucket_missing is bucket
    assert dev_map_missing["addresses_by_type"]["htr"] == []
    assert dev_map_missing["settings"]["htr"] == {"mode": "sched"}
    assert dev_map_missing["nodes_by_type"]["htr"] is bucket

    dev_map_non_mapping_settings = {
        "addresses_by_type": {},
        "settings": None,
        "nodes_by_type": {},
    }
    dummy._ensure_type_bucket({"htr": bucket}, "htr", dev_map=dev_map_non_mapping_settings)
    assert dev_map_non_mapping_settings["settings"]["htr"] == {}


def test_ws_common_apply_heater_addresses_normalizes_inputs() -> None:
    """Heater address helper should normalise inputs and update energy state."""

    energy_coordinator = SimpleNamespace(update_addresses=MagicMock())
    hass_record: dict[str, Any] = {"energy_coordinator": energy_coordinator, "inventory": None}
    hass = SimpleNamespace(data={base_ws.DOMAIN: {"entry": hass_record}})

    class Dummy(base_ws._WSCommon):
        def __init__(self) -> None:
            self.hass = hass
            self.entry_id = "entry"
            self.dev_id = "dev"
            self._coordinator = SimpleNamespace()
            base_ws._WSCommon.__init__(self, inventory=None)

    dummy = Dummy()
    cleaned = dummy._apply_heater_addresses({"htr": "1", "pmo": "7"})
    assert cleaned["htr"] == ["1"]
    assert cleaned["pmo"] == ["7"]
    sample_aliases = hass_record["sample_aliases"]
    assert sample_aliases["htr"] == "htr"
    assert sample_aliases["pmo"] == "pmo"
    energy_coordinator.update_addresses.assert_called_with(cleaned)

    energy_coordinator.update_addresses.reset_mock()
    cleaned_iterable = dummy._apply_heater_addresses(
        {"htr": ["2", "2", "  "], "pmo": ["", "8", "8"]}
    )
    assert cleaned_iterable["htr"] == ["2"]
    assert cleaned_iterable["pmo"] == ["8"]
    sample_aliases = hass_record["sample_aliases"]
    assert sample_aliases["htr"] == "htr"
    assert sample_aliases["pmo"] == "pmo"
    energy_coordinator.update_addresses.assert_called_with(cleaned_iterable)


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


def test_collect_update_addresses_skips_invalid() -> None:
    """Non-mapping payloads should be ignored when collecting addresses."""

    nodes: Mapping[str, Any] = {"htr": [1, 2], 10: {"settings": {"1": {}}}}
    assert module.WebSocketClient._collect_update_addresses(nodes) == []



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


def test_ws_common_state_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    """WS common helper should create domain buckets when missing."""

    hass = SimpleNamespace(data={})

    class Dummy(base_ws._WSCommon):
        def __init__(self) -> None:
            self.hass = hass
            self.entry_id = "entry"
            self.dev_id = "dev"
            self._coordinator = SimpleNamespace(update_nodes=MagicMock())

    dummy = Dummy()
    bucket = dummy._ws_state_bucket()
    assert bucket == {}
    assert base_ws.DOMAIN in hass.data


def test_ws_common_update_status_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    """WS common status helper should forward dispatcher signals."""

    hass = SimpleNamespace(data={base_ws.DOMAIN: {"entry": {}}})
    dispatcher = MagicMock()
    monkeypatch.setattr(base_ws, "async_dispatcher_send", dispatcher)

    class Dummy(base_ws._WSCommon):
        def __init__(self) -> None:
            self.hass = hass
            self.entry_id = "entry"
            self.dev_id = "dev"
            self._coordinator = SimpleNamespace(update_nodes=MagicMock())

    dummy = Dummy()
    dummy._update_status("connected")

    dispatcher.assert_called_once()


def test_ws_common_dispatch_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    """WS common dispatch should update coordinator and emit dispatcher events."""

    hass = SimpleNamespace(data={base_ws.DOMAIN: {"entry": {}}})
    coordinator = SimpleNamespace(update_nodes=MagicMock(), dev_id="dev")
    dispatcher = MagicMock()
    monkeypatch.setattr(base_ws, "async_dispatcher_send", dispatcher)

    raw_nodes = {"nodes": [{"type": "htr", "addr": "1"}]}
    inventory_nodes = build_node_inventory(raw_nodes)
    inventory_obj = Inventory("dev", raw_nodes, inventory_nodes)

    def fake_resolve(
        record: Mapping[str, Any] | None,
        *,
        dev_id: str | None = None,
        nodes_payload: Any | None = None,
        **_: Any,
    ) -> Any:
        assert dev_id == "dev"
        assert nodes_payload == raw_nodes
        return SimpleNamespace(inventory=inventory_obj, source="test", raw_count=1, filtered_count=1)

    monkeypatch.setattr(base_ws, "resolve_record_inventory", fake_resolve)
    monkeypatch.setattr(
        base_ws,
        "addresses_by_node_type",
        lambda inventory, known_types=None: ({"htr": ["1"]}, {}),
    )

    class Dummy(base_ws._WSCommon):
        def __init__(self) -> None:
            self.hass = hass
            self.entry_id = "entry"
            self.dev_id = "dev"
            self._coordinator = coordinator
            self._inventory = None

    dummy = Dummy()
    payload = {"nodes": raw_nodes}
    dummy._dispatch_nodes(payload)

    coordinator.update_nodes.assert_called_once()
    update_args, update_kwargs = coordinator.update_nodes.call_args
    assert not update_kwargs
    assert update_args[0] is None
    assert update_args[1] is inventory_obj
    dispatcher.assert_called_once()
    record = hass.data[base_ws.DOMAIN]["entry"]
    assert record.get("inventory") is inventory_obj
    dispatched_payload = dispatcher.call_args.args[2]
    assert dispatched_payload == {
        "dev_id": "dev",
        "node_type": None,
        "addresses_by_type": {"htr": ["1"]},
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
