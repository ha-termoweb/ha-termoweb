"""Unit tests for websocket client helpers."""

from __future__ import annotations

import asyncio
import copy
import importlib
import gzip
from types import MappingProxyType, ModuleType, SimpleNamespace
from typing import Any, Callable, Mapping
from urllib.parse import parse_qsl, urlsplit
from unittest.mock import AsyncMock, MagicMock

import pytest

import logging
import sys

from conftest import (
    CoordinatorStub,
    DummyREST,
    build_entry_runtime,
    listen_ws_status,
)
from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.backend import ducaheat_ws
from custom_components.termoweb.backend import termoweb_ws as module
from custom_components.termoweb.backend import ws_client as base_ws
from custom_components.termoweb.backend.sanitize import (
    mask_identifier,
    redact_token_fragment,
)
from custom_components.termoweb.inventory import Inventory, build_node_inventory


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
def reload_ws_modules() -> None:
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
) -> module.TermoWebWSClient:
    """Instantiate a TermoWeb websocket client for tests."""

    if hass_loop is None:
        hass_loop = SimpleNamespace(
            create_task=lambda coro, **_: SimpleNamespace(done=lambda: True),
            call_soon_threadsafe=lambda cb, *args: cb(*args),
        )

    hass = SimpleNamespace(loop=hass_loop, data={DOMAIN: {}})
    build_entry_runtime(
        hass=hass,
        entry_id="entry",
        dev_id="device",
    )
    coordinator = SimpleNamespace(update_nodes=MagicMock(), dev_id="dev")
    client = module.TermoWebWSClient(
        hass,
        entry_id="entry",
        dev_id="device",
        api_client=DummyREST(),
        coordinator=coordinator,
        session=SimpleNamespace(),
    )
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
    hass = SimpleNamespace(loop=hass_loop, data={DOMAIN: {}})
    build_entry_runtime(
        hass=hass,
        entry_id="entry",
        dev_id="device",
    )
    rest_client = DummyREST(is_ducaheat=True)
    client = ducaheat_ws.DucaheatWSClient(
        hass,
        entry_id="entry",
        dev_id="device",
        api_client=rest_client,
        coordinator=SimpleNamespace(update_nodes=MagicMock()),
        session=SimpleNamespace(),
    )
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

    client = module.TermoWebWSClient(
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
            "pmo": {
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
        and record.levelno == logging.ERROR
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


def test_ws_state_bucket_initialises_missing_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify hass.data is created when absent."""

    client = _make_termoweb_client(monkeypatch)
    bucket = client._ws_state_bucket()
    runtime = client.hass.data[DOMAIN]["entry"]
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


def test_ducaheat_polling_headers_extend_brand_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ducaheat polling headers add browser-like fields to the shared brand set."""

    client = _make_ducaheat_client(monkeypatch)
    headers = client._polling_headers()
    assert headers.items() >= client._brand_headers(origin="https://localhost").items()
    assert headers["Referer"] == "https://localhost/"
    assert headers["Connection"] == "keep-alive"


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


def test_ducaheat_log_nodes_summary_includes_counts(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Logging nodes should record the node types and address counts."""

    client = _make_ducaheat_client(monkeypatch)
    caplog.set_level("DEBUG")
    client._log_nodes_summary({"htr": {"settings": {"1": {}, "2": {}}}})
    assert "htr" in caplog.text
    assert "2" in caplog.text


def test_termoweb_value_redaction_behaviour(monkeypatch: pytest.MonkeyPatch) -> None:
    """Redaction helpers should mask tokens and identifiers consistently."""

    assert redact_token_fragment("  ") == ""
    assert redact_token_fragment("abcd") == "***"
    assert redact_token_fragment("abcdefgh") == "ab***gh"
    assert redact_token_fragment("abcdefghijk") == "abcd...hijk"
    assert mask_identifier("   ") == ""
    assert mask_identifier("xy") == "***"
    assert mask_identifier("abcdefgh") == "ab...gh"
    assert mask_identifier("abcdefghijkl") == "abcdef...ijkl"


def test_termoweb_sanitise_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sensitive URL values should be redacted."""

    client = _make_termoweb_client(monkeypatch)
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
    client._mark_event(count_event=True)
    state_after = client._ws_state_bucket()
    assert state_after["last_event_at"] != state_before.get("last_event_at")
    assert state_after["events_total"] == 1
    assert client._healthy_since is not None


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


def test_termoweb_update_status_records_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Status updates should populate the hass data bucket and dispatch events."""

    client = _make_termoweb_client(monkeypatch)
    listener = listen_ws_status(monkeypatch, client)
    client._stats.frames_total = 4  # type: ignore[attr-defined]
    client._stats.events_total = 2  # type: ignore[attr-defined]
    client._stats.last_event_ts = 50.0  # type: ignore[attr-defined]
    client._ws_health_tracker().healthy_since = 40.0
    monkeypatch.setattr(module.time, "time", lambda: 100.0)

    client._update_status("connected")

    state = client._ws_state_bucket()
    assert state["status"] == "connected"
    assert state["frames_total"] == 4
    assert state["events_total"] == 2
    assert state["healthy_minutes"] == 1
    listener.assert_called()


def test_termoweb_mark_event_without_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Count-only events should still trigger healthy transitions."""

    client = _make_termoweb_client(monkeypatch)
    client._update_status = MagicMock()  # type: ignore[attr-defined]
    monkeypatch.setattr(module.time, "time", lambda: 300.0)

    client._mark_event(count_event=True)

    assert client._stats.events_total == 1  # type: ignore[attr-defined]
    assert client._healthy_since == 300.0
    client._update_status.assert_called_once_with("healthy")  # type: ignore[attr-defined]


def test_termoweb_update_status_prefers_stats_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Healthy updates should fall back to stats timestamps when available."""

    client = _make_termoweb_client(monkeypatch)
    listener = listen_ws_status(monkeypatch, client)
    client._stats.frames_total = 1  # type: ignore[attr-defined]
    client._stats.events_total = 1  # type: ignore[attr-defined]
    client._stats.last_event_ts = 75.0  # type: ignore[attr-defined]
    client._last_event_at = None
    monkeypatch.setattr(module.time, "time", lambda: 100.0)

    client._update_status("healthy")

    state = client._ws_state_bucket()
    assert state["last_event_at"] == 75.0
    assert state["healthy_since"] == 75.0
    listener.assert_called()


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


@pytest.mark.parametrize("brand", ["termoweb", "ducaheat"])
def test_ws_backoff_sequence(
    monkeypatch: pytest.MonkeyPatch, brand: str
) -> None:
    """Both clients share one backoff sequence that restarts on reset."""

    maker = _make_termoweb_client if brand == "termoweb" else _make_ducaheat_client
    client = maker(monkeypatch)
    values = [client._next_backoff() for _ in range(6)]
    assert values == [5, 10, 30, 120, 300, 300]
    client._reset_backoff()
    assert client._next_backoff() == 5


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


# ---------------------------------------------------------------------------
# Helpers shared by both websocket clients, exercised through each of them
# ---------------------------------------------------------------------------

_CLIENT_FACTORIES = {
    "termoweb": _make_termoweb_client,
    "ducaheat": _make_ducaheat_client,
}
both_clients = pytest.mark.parametrize("brand", sorted(_CLIENT_FACTORIES))


def _client_with_inventory(
    monkeypatch: pytest.MonkeyPatch, brand: str, *, hass_loop: Any | None = None
) -> tuple[Any, Inventory]:
    """Return a client of ``brand`` bound to a one-heater inventory."""

    client = _CLIENT_FACTORIES[brand](monkeypatch, hass_loop=hass_loop)
    inventory = Inventory(
        "device", build_node_inventory({"nodes": [{"type": "htr", "addr": "1"}]})
    )
    client._inventory = inventory
    return client, inventory


@both_clients
def test_common_brand_headers(monkeypatch: pytest.MonkeyPatch, brand: str) -> None:
    """Brand headers carry the brand identity and an optional origin."""

    client = _CLIENT_FACTORIES[brand](monkeypatch)
    headers = client._brand_headers(origin="https://app.example")
    assert headers["User-Agent"]
    assert headers["X-Requested-With"]
    assert headers["Origin"] == "https://app.example"
    assert "Origin" not in client._brand_headers()


@both_clients
def test_common_nodes_to_deltas(monkeypatch: pytest.MonkeyPatch, brand: str) -> None:
    """Node payloads become deltas; unknown nodes and missing inventory are skipped."""

    client, inventory = _client_with_inventory(monkeypatch, brand)
    nodes = {
        "htr": {
            "settings": {"1": {"mode": "manual", "unknown": "drop"}},
            "status": {"1": {"stemp": "18.0", "online": True}},
            "prog": {"1": {"0": 1}},
            "samples": {"1": {"temp": 12}},
            "capabilities": {"1": {"x": 1}},
            7: {"1": {"mode": "auto"}},
            "extra": {"": {"mode": "auto"}},
        },
        "bogus": {"settings": {"1": {"mode": "auto"}}},
    }

    deltas = client._nodes_to_deltas(nodes, inventory=inventory)

    assert len(deltas) == 1
    payload = deltas[0].payload
    assert deltas[0].node_id.addr == "1"
    assert payload["mode"] == "manual"
    assert payload["stemp"] == "18.0"
    assert payload["prog"] == {"0": 1}
    assert "unknown" not in payload
    assert "samples" not in payload
    assert (
        client._nodes_to_deltas(
            {"htr": {"settings": {"2": {"mode": "auto"}}}}, inventory=inventory
        )
        == []
    )
    assert client._nodes_to_deltas(nodes, inventory=None) == []


@both_clients
def test_common_apply_deltas_to_store(
    monkeypatch: pytest.MonkeyPatch, brand: str
) -> None:
    """Deltas reach the coordinator handler; handler failures do not propagate."""

    client, inventory = _client_with_inventory(monkeypatch, brand)
    deltas = client._nodes_to_deltas(
        {"htr": {"settings": {"1": {"mode": "auto"}}}}, inventory=inventory
    )
    client._apply_deltas_to_store(deltas, replace=True)  # no handler: no-op

    handler = MagicMock()
    client._coordinator = SimpleNamespace(handle_ws_deltas=handler)
    client._apply_deltas_to_store(deltas, replace=True)
    handler.assert_called_once_with("device", tuple(deltas), replace=True)

    handler.side_effect = RuntimeError("boom")
    client._apply_deltas_to_store(deltas, replace=False)


@both_clients
def test_common_translate_path_update(
    monkeypatch: pytest.MonkeyPatch, brand: str
) -> None:
    """Path frames map onto node sections; malformed frames are rejected."""

    client = _CLIENT_FACTORIES[brand](monkeypatch)
    assert client._translate_path_update(
        {"path": "/api/v2/devs/device/htr/2/settings/setup", "body": {"mode": "auto"}}
    ) == {"htr": {"settings": {"2": {"setup": {"mode": "auto"}}}}}
    assert client._translate_path_update(
        {"path": "/api/v2/devs/device/htr/2/setup", "body": {"mode": "eco"}}
    ) == {"htr": {"settings": {"2": {"setup": {"mode": "eco"}}}}}
    for payload in (
        {"path": "/", "body": {}},
        {"path": "/api/v2/devs/device/htr", "body": {}},
        "not a mapping",
        {"nodes": {}},
        {"path": "/api/v2/devs/device/htr/2/status"},
        {"path": "/api/v2/devs/device/htr/ /status", "body": {"temp": 1}},
    ):
        assert client._translate_path_update(payload) is None


@both_clients
@pytest.mark.asyncio
async def test_common_get_token(monkeypatch: pytest.MonkeyPatch, brand: str) -> None:
    """The bearer token is reused from the REST client; a missing one raises."""

    client = _CLIENT_FACTORIES[brand](monkeypatch)
    client._client.authed_headers = AsyncMock(
        return_value={"Authorization": "Bearer newtoken"}
    )
    assert await client._get_token() == "newtoken"
    client._client.authed_headers = AsyncMock(return_value={})
    with pytest.raises(RuntimeError):
        await client._get_token()


@both_clients
def test_common_start_reuses_or_creates_task(
    monkeypatch: pytest.MonkeyPatch, brand: str
) -> None:
    """``start`` reuses a live task and otherwise schedules the runner."""

    loop = DummyLoop()
    client = _CLIENT_FACTORIES[brand](monkeypatch, hass_loop=loop)

    async def _noop() -> None:
        return None

    existing = DummyTask(_noop())
    client._task = existing
    assert client.start() is existing
    assert not loop.created_tasks
    existing.cancel()

    client._task = None
    client._runner = AsyncMock()  # type: ignore[assignment]
    task = client.start()
    assert task is loop.created_tasks[0]
    task.cancel()


@both_clients
def test_common_normalise_nodes(monkeypatch: pytest.MonkeyPatch, brand: str) -> None:
    """Normalisation delegates to the REST codec and tolerates codec failures."""

    client = _CLIENT_FACTORIES[brand](monkeypatch)
    nodes = {"htr": {"status": {"1": {}}}}

    assert client._normalise_nodes(nodes) == nodes
    client._client.normalise_ws_nodes = lambda n: MappingProxyType({"htr": {}})
    assert client._normalise_nodes(nodes) == {"htr": {}}
    assert isinstance(client._normalise_nodes(nodes), dict)
    client._client.normalise_ws_nodes = lambda n: ["ok"]
    assert client._normalise_nodes(nodes) == ["ok"]

    def _raise(_nodes: Any) -> Any:
        raise RuntimeError

    client._client.normalise_ws_nodes = _raise
    assert client._normalise_nodes(nodes) == nodes


@both_clients
def test_common_coerce_nodes_list(monkeypatch: pytest.MonkeyPatch, brand: str) -> None:
    """List-shaped node snapshots become ``{type: {section: {addr: value}}}``."""

    client, _ = _client_with_inventory(monkeypatch, brand)
    entries = [
        {"type": "htr", "addr": "1", "name": "skip", "lease_seconds": 60},
        {
            "type": "htr",
            "addr": "1",
            "settings": {"stemp": "20"},
            "setup": {"program": 1},
            "status": {"mode": "auto"},
            3: "non-string key",
            "": "empty key",
        },
        {"type": "htr", "addr": "9", "settings": {"x": 1}},
        "not a mapping",
    ]
    original = copy.deepcopy(entries)

    assert client._coerce_nodes_list(entries) == {
        "htr": {
            "lease_seconds": 60,
            "settings": {"1": {"stemp": "20", "setup": {"program": 1}}},
            "status": {"1": {"mode": "auto"}},
        }
    }
    assert entries == original
    assert client._coerce_nodes_list([{"type": "htr", "addr": "9"}]) is None
    for not_a_list in (None, {"htr": {}}, "text", b"bytes", 5):
        assert client._coerce_nodes_list(not_a_list) is None
    client._inventory = None
    assert client._coerce_nodes_list(entries) is None


@both_clients
def test_common_collect_sample_updates(
    monkeypatch: pytest.MonkeyPatch, brand: str
) -> None:
    """Sample extraction filters node types, addresses and malformed entries."""

    client = _CLIENT_FACTORIES[brand](monkeypatch)
    payload: dict[Any, Any] = {
        "htr": {
            "samples": {"": {"power": 1}, "1": {"power": 2}},
            "lease_seconds": 90,
        },
        123: {"samples": {"1": {"power": 3}}},
        "acm": {"status": {"1": {}}},
        "thm": {"samples": {"1": {"temp": 1}}},
        "": {"samples": {"1": {"power": 5}}},
        "pmo": {"samples": {"1": {"power": 4}}},
    }

    assert client._collect_sample_updates(payload) == {
        "htr": {"samples": {"1": {"power": 2}}, "lease_seconds": 90},
        "pmo": {"samples": {"1": {"power": 4}}, "lease_seconds": None},
    }
    assert set(client._collect_sample_updates(payload, allowed_types=["HTR"])) == {
        "htr"
    }
