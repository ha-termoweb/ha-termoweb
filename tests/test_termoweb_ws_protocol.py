"""Extended tests for TermoWeb websocket protocol flows."""

from __future__ import annotations

import asyncio
import importlib
import logging
import threading
from types import MappingProxyType, SimpleNamespace
from typing import Any, Mapping
from urllib.parse import parse_qsl, urlsplit
from unittest.mock import AsyncMock, MagicMock

import pytest

from conftest import build_entry_runtime
from custom_components.termoweb.backend import termoweb_ws as module
from custom_components.termoweb.backend import ws_client as ws_client_module
from custom_components.termoweb.backend.sanitize import (
    mask_identifier,
    redact_token_fragment,
)
from custom_components.termoweb.inventory import Inventory, build_node_inventory
from homeassistant.core import HomeAssistant


@pytest.fixture(autouse=True)
def reload_ws_modules() -> None:
    """Reload websocket modules to avoid stale references after other tests."""

    global module, ws_client_module
    module = importlib.reload(
        importlib.import_module("custom_components.termoweb.backend.termoweb_ws")
    )
    ws_client_module = importlib.reload(
        importlib.import_module("custom_components.termoweb.backend.ws_client")
    )


def translate_update(payload: Any) -> Any:
    """Translate websocket path updates using the default namespace resolver."""

    return ws_client_module.translate_path_update(
        payload,
        resolve_section=module.TermoWebWSClient._resolve_update_section,
    )


INVALID_TRANSLATION_PAYLOADS: list[Any] = [
    "invalid",
    {"nodes": {}},
    {"path": 123, "body": {}},
    {"path": "/", "body": {}},
    {"path": "/api/devs/device", "body": {}},
    {"path": "/api/htr", "body": {}},
    {"path": "/htr", "body": {}},
    {"path": "/api/devs/device/htr/", "body": {}},
    {"path": "/api/devs/device/htr//settings", "body": {}},
    {"path": "/api/devs/device/ /settings", "body": {}},
    {"path": "/api/devs/device/htr/ /settings", "body": {}},
]


class DummyREST:
    """Provide just enough of the REST client interface for websocket tests."""

    def __init__(
        self,
        *,
        requested_with: str | None = "requested",
        api_base: str | None = "https://api.termoweb",
        authed_headers: dict[str, str] | None = None,
    ) -> None:
        self._session = SimpleNamespace(closed=True)
        self._ensure_token = AsyncMock()
        headers = authed_headers or {"Authorization": "Bearer token"}
        self.authed_headers = AsyncMock(return_value=headers)
        self.api_base = api_base
        self.user_agent = "agent"
        self.requested_with = requested_with


def _make_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    hass_loop: Any | None = None,
    rest_headers: dict[str, str] | None = None,
    session: Any | None = None,
    api_base: str | None = "https://api.termoweb",
) -> tuple[module.TermoWebWSClient, MagicMock]:
    """Instantiate the production ``TermoWebWSClient`` with test doubles."""

    dispatcher = MagicMock()

    if hass_loop is None:
        hass_loop = SimpleNamespace(
            create_task=lambda coro, **_: SimpleNamespace(done=lambda: False),
            call_soon_threadsafe=lambda cb, *args: cb(*args),
            is_running=lambda: False,
        )

    hass = HomeAssistant()
    hass.loop = hass_loop
    hass.loop_thread_id = threading.get_ident()
    inventory_payload = {"nodes": [{"type": "htr", "addr": "1"}]}
    default_inventory = Inventory(
        "device",
        build_node_inventory(inventory_payload),
    )
    coordinator = SimpleNamespace(
        update_nodes=MagicMock(), data={}, inventory=default_inventory
    )
    build_entry_runtime(
        hass=hass,
        entry_id="entry",
        dev_id="device",
        inventory=default_inventory,
        coordinator=coordinator,
    )
    client = module.TermoWebWSClient(
        hass,
        entry_id="entry",
        dev_id="device",
        api_client=DummyREST(
            api_base=api_base,
            authed_headers=rest_headers,
        ),
        coordinator=coordinator,
        session=session or SimpleNamespace(closed=True),
        inventory=default_inventory,
    )
    return client, dispatcher


def test_handshake_error_exposes_fields() -> None:
    """The TermoWeb handshake error should record status, URL and body."""

    error = module.HandshakeError(
        503,
        "https://example/ws",
        "body",
        response_snippet="body",
    )
    assert str(error) == "handshake failed: status=503, detail=body"
    assert error.status == 503
    assert error.url == "https://example/ws"
    assert error.detail == "body"
    assert error.response_snippet == "body"


@pytest.mark.asyncio
async def test_get_token_requires_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_get_token should raise when the Authorization header is missing."""

    client, _ = _make_client(monkeypatch, rest_headers={"Authorization": ""})
    with pytest.raises(RuntimeError):
        await client._get_token()


@pytest.mark.asyncio
async def test_force_refresh_token_resets_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_force_refresh_token should clear cached credentials and ensure tokens."""

    client, _ = _make_client(monkeypatch)
    client._client._access_token = "token"  # type: ignore[attr-defined]
    await client._force_refresh_token()
    client._client._ensure_token.assert_awaited()  # type: ignore[attr-defined]


def test_api_base_defaults_to_constant(monkeypatch: pytest.MonkeyPatch) -> None:
    """_api_base should fall back to the default when the client lacks one."""

    client, _ = _make_client(monkeypatch, api_base=None)
    assert client._api_base() == module.API_BASE


def test_ws_state_bucket_initialises_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    """_ws_state_bucket should create storage on hass when missing."""

    client, _ = _make_client(monkeypatch)
    client.hass = SimpleNamespace(loop=None)
    client._ws_state = None
    bucket = client._ws_state_bucket()
    assert isinstance(bucket, dict)


@pytest.mark.asyncio
async def test_refresh_subscription_requires_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refreshing while disconnected should raise an error."""

    client, _ = _make_client(monkeypatch)
    with pytest.raises(RuntimeError):
        await client._refresh_subscription(reason="disconnected")


def test_translate_path_update_and_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Path based updates should map onto node sections."""

    client, _ = _make_client(monkeypatch)
    payload = {
        "path": "/api/devs/device/htr/1/settings/temp",
        "body": {"value": 20},
    }
    translated = translate_update(payload)
    assert translated == {"htr": {"settings": {"1": {"temp": {"value": 20}}}}}
    assert client._translate_path_update(payload) == translated
    setup_payload = {
        "path": "/api/devs/device/htr/1/setup/program",
        "body": {"foo": 1},
    }
    translated_setup = translate_update(setup_payload)
    assert translated_setup == {
        "htr": {"settings": {"1": {"setup": {"program": {"foo": 1}}}}}
    }
    assert client._translate_path_update(setup_payload) == translated_setup
    assert module.TermoWebWSClient._resolve_update_section("advanced_setup") == (
        "advanced",
        "advanced_setup",
    )
    assert module.TermoWebWSClient._resolve_update_section("prog") == (
        "settings",
        "prog",
    )
    assert module.TermoWebWSClient._resolve_update_section(None) == (None, None)


def test_translate_path_update_invalid_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid payloads should return None from the path translator."""

    client, _ = _make_client(monkeypatch)
    for payload in INVALID_TRANSLATION_PAYLOADS:
        assert translate_update(payload) is None
        assert client._translate_path_update(payload) is None


def test_translate_path_update_rejects_unknown_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Path translation should ignore unknown node types and addresses."""

    client, _ = _make_client(monkeypatch)
    for payload in (
        {"path": "/api/devs/device/ /1/settings", "body": {"v": 1}},
        {"path": "/api/devs/device/htr/ /settings", "body": {"v": 1}},
    ):
        assert translate_update(payload) is None
        assert client._translate_path_update(payload) is None


def test_handle_handshake_logging(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Handshake handling should log keys and ignore invalid payloads."""

    client, _ = _make_client(monkeypatch)
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(module._LOGGER, "isEnabledFor", lambda level: True)
    monkeypatch.setattr(module.time, "time", lambda: 123.0)
    client._handle_handshake({"alpha": 1, "beta": 2})
    assert client._handshake_payload == {
        "keys": ("alpha", "beta"),
        "received_at": 123.0,
    }
    assert client._ws_state_bucket().get("handshake_keys") == ("alpha", "beta")
    client._handle_handshake("invalid")


def test_apply_nodes_payload_debug_branches(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Applying node payloads should log diagnostic information and filter invalid data."""

    client, _ = _make_client(monkeypatch)
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(module._LOGGER, "isEnabledFor", lambda level: True)
    client._forward_sample_updates = MagicMock()
    client._mark_event = MagicMock()
    client._collect_update_addresses = MagicMock(
        side_effect=[[("htr", "1")], [], [], []]
    )
    client._client.normalise_ws_nodes = lambda nodes: nodes

    client._apply_nodes_payload({}, merge=False, event="dev_data")

    nodes_payload = {
        "nodes": {
            1: {"samples": {"1": {"power": 5}}},
            "htr": {"samples": {"bad": {"power": 3}, "1": {"power": 10}}},
            "acm": {"samples": []},
        }
    }
    client._apply_nodes_payload(nodes_payload, merge=True, event="update")

    client._apply_nodes_payload(
        {"nodes": {"htr": {"samples": {"1": {"power": 7}}}}},
        merge=True,
        event="update",
    )

    client._apply_nodes_payload(
        {"nodes": {"htr": {"samples": {"1": {"power": 8}}}}},
        merge=False,
        event="dev_data",
    )

    client._apply_nodes_payload(
        {"nodes": {"htr": {"samples": {"": {"power": 9}}}}},
        merge=True,
        event="update",
    )

    assert client._forward_sample_updates.called


def test_handle_dev_data_and_update(monkeypatch: pytest.MonkeyPatch) -> None:
    """Direct handlers should call into the payload merger."""

    client, _ = _make_client(monkeypatch)
    client._apply_nodes_payload = MagicMock()  # type: ignore[attr-defined]
    client._handle_dev_data({"nodes": {"htr": {}}})
    client._apply_nodes_payload.assert_called_with(
        {"nodes": {"htr": {}}}, merge=False, event="dev_data"
    )
    client._apply_nodes_payload.reset_mock()
    client._handle_update({"path": "value"})
    client._apply_nodes_payload.assert_called_with(
        {"path": "value"}, merge=True, event="update"
    )


def test_apply_nodes_payload_merges_and_forwards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Applying node payloads should normalize and forward updates."""

    client, dispatcher = _make_client(monkeypatch)
    client._collect_update_addresses = MagicMock(return_value=[("htr", "1")])  # type: ignore[attr-defined]
    client._forward_sample_updates = MagicMock()  # type: ignore[attr-defined]
    client._mark_event = MagicMock()  # type: ignore[attr-defined]

    snapshot_payload = {"nodes": {"htr": {"settings": {"1": {"temp": 20}}}}}
    client._apply_nodes_payload(snapshot_payload, merge=False, event="dev_data")

    update_payload = {"path": "/api/devs/device/htr/1/samples", "body": {"power": 5}}
    client._apply_nodes_payload(update_payload, merge=True, event="update")
    client._forward_sample_updates.assert_called()
    client._mark_event.assert_called()


def test_heater_sample_subscription_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Subscription helper should return inventory-derived targets."""

    client, _ = _make_client(monkeypatch)
    raw_nodes = {"nodes": [{"type": "htr", "addr": "1"}, {"type": "acm", "addr": "2"}]}
    inventory = Inventory(
        client.dev_id,
        build_node_inventory(raw_nodes),
    )
    client._inventory = inventory

    targets = list(client._heater_sample_subscription_targets())

    assert targets == inventory.heater_sample_targets


def test_heater_sample_subscription_targets_logs_missing_inventory(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Missing inventory should be logged when resolving subscription targets."""

    client, _ = _make_client(monkeypatch)
    client._inventory = None

    with caplog.at_level(logging.ERROR):
        targets = list(client._heater_sample_subscription_targets())

    assert not targets
    assert any("missing inventory" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_schedule_idle_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scheduling an idle restart should create a task and reset flags afterwards."""

    loop = asyncio.get_running_loop()
    hass_loop = SimpleNamespace(
        create_task=lambda coro, **kwargs: loop.create_task(coro, **kwargs),
        call_soon_threadsafe=lambda cb, *args: loop.call_soon(cb, *args),
    )
    client, _ = _make_client(monkeypatch, hass_loop=hass_loop)
    client._closing = False
    client._schedule_idle_restart(idle_for=10, source="test")
    assert client._idle_restart_pending is True
    task = client._idle_restart_task
    assert task is not None
    await asyncio.sleep(0)
    await task
    assert client._idle_restart_pending is False


def test_schedule_idle_restart_ignored_when_closing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scheduling should be skipped when already closing."""

    client, _ = _make_client(monkeypatch)
    client._closing = True
    client._schedule_idle_restart(idle_for=10, source="closing")
    assert client._idle_restart_task is None


@pytest.mark.asyncio
async def test_cancel_idle_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancelling an idle restart should cancel the task."""

    loop = asyncio.get_running_loop()
    hass_loop = SimpleNamespace(
        create_task=lambda coro, **kwargs: loop.create_task(coro, **kwargs),
        call_soon_threadsafe=lambda cb, *args: loop.call_soon(cb, *args),
    )
    client, _ = _make_client(monkeypatch, hass_loop=hass_loop)
    client._closing = False
    client._schedule_idle_restart(idle_for=10, source="test")
    task = client._idle_restart_task
    assert task is not None
    client._cancel_idle_restart()
    assert client._idle_restart_task is None


def test_header_sanitizers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Header and URL sanitisation helpers should redact sensitive values."""

    client, _ = _make_client(monkeypatch)

    headers = client._brand_headers(origin="https://app")
    assert headers["X-Requested-With"] == module.get_brand_requested_with(
        module.BRAND_TERMOWEB
    )
    client._requested_with = ""
    headers = client._brand_headers(origin="https://app")
    assert headers["Origin"] == "https://app"
    assert headers["User-Agent"] == module.get_brand_user_agent(module.BRAND_TERMOWEB)
    assert headers["Accept-Language"] == module.ACCEPT_LANGUAGE

    assert redact_token_fragment("   ") == ""
    assert redact_token_fragment("") == ""
    assert redact_token_fragment("abc") == "***"
    assert redact_token_fragment("abcdefgh") == "ab***gh"
    assert redact_token_fragment("abcdefghijklmnop") == "abcd...mnop"

    assert mask_identifier("abcd") == "***"
    assert mask_identifier("abcdefgh") == "ab...gh"
    assert mask_identifier("abcdefghijklmnop") == "abcdef...mnop"

    sanitised_url = client._sanitise_url(
        "https://host/socket?token=abc&dev_id=12345&sid=session"
    )
    sanitised_query = dict(parse_qsl(urlsplit(sanitised_url).query))
    assert sanitised_query["token"] == "{token}"
    assert sanitised_query["dev_id"] == "{dev_id}"
    assert sanitised_query["sid"] == "{sid}"
    sanitised_ws_url = client._sanitise_url(
        "https://host/socket.io/1/websocket/abc123?transport=websocket&sid=session"
    )
    parsed_ws_url = urlsplit(sanitised_ws_url)
    ws_query = dict(parse_qsl(parsed_ws_url.query))
    assert parsed_ws_url.path.endswith("/socket.io/1/websocket/{sid}")
    assert ws_query["sid"] == "{sid}"
    assert client._sanitise_url("not a url") == "not a url"
    assert client._sanitise_url("http://[::1") == "http://[::1"


def test_redaction_helpers_handle_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Token and identifier masking should treat whitespace as empty."""

    client, _ = _make_client(monkeypatch)
    assert redact_token_fragment("   ") == ""
    assert mask_identifier("   ") == ""


@pytest.mark.asyncio
async def test_start_returns_existing_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """start should return the original task when invoked multiple times."""

    loop = asyncio.get_running_loop()
    hass_loop = SimpleNamespace(
        create_task=lambda coro, **kwargs: loop.create_task(coro, **kwargs),
        call_soon_threadsafe=lambda cb, *args: loop.call_soon(cb, *args),
    )
    client, _ = _make_client(monkeypatch, hass_loop=hass_loop)

    ready = asyncio.Event()

    async def fake_runner() -> None:
        await ready.wait()

    monkeypatch.setattr(client, "_runner", fake_runner)

    task1 = client.start()
    task2 = client.start()
    assert task1 is task2

    ready.set()
    await asyncio.wait_for(task1, timeout=0.1)


@pytest.mark.asyncio
async def test_start_and_stop_manage_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    """start should spawn tasks and stop should cancel them cleanly."""

    loop = asyncio.get_running_loop()
    hass_loop = SimpleNamespace(
        create_task=lambda coro, **kwargs: loop.create_task(coro, **kwargs),
        call_soon_threadsafe=lambda cb, *args: loop.call_soon(cb, *args),
    )
    client, _ = _make_client(monkeypatch, hass_loop=hass_loop)

    runner_gate = asyncio.Event()

    async def runner() -> None:
        try:
            await runner_gate.wait()
        except asyncio.CancelledError:
            raise

    monkeypatch.setattr(client, "_runner", runner)
    monkeypatch.setattr(client, "_disconnect", AsyncMock())
    client._idle_restart_task = loop.create_task(asyncio.sleep(0))
    client._idle_monitor_task = loop.create_task(asyncio.sleep(0))

    task = client.start()
    assert (client._task is not None and not client._task.done()) is True
    runner_gate.set()
    await asyncio.sleep(0)
    await client.stop()
    assert (client._task is not None and not client._task.done()) is False
    assert task.cancelled() or task.done()


def test_apply_nodes_payload_translation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Node payload application should merge data and notify listeners."""

    client, _dispatcher = _make_client(monkeypatch)
    raw_nodes = {"nodes": [{"type": "htr", "addr": "1"}]}
    client._inventory = Inventory(
        client.dev_id,
        build_node_inventory(raw_nodes),
    )
    client._handshake_payload = {"keys": ("nodes",), "received_at": 0}
    client._handle_handshake({"nodes": {"htr": {"status": {"1": {"temp": 20}}}}})
    client._forward_sample_updates = MagicMock()
    client._apply_nodes_payload(
        {"nodes": {"htr": {"status": {"1": {"temp": 25}}}}}, merge=True, event="update"
    )
    client._forward_sample_updates.assert_not_called()


def test_forward_sample_updates_invokes_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forwarding sample updates should notify the energy coordinator handler."""

    client, _ = _make_client(monkeypatch)
    handler_called: dict[str, Any] = {}
    energy_handler = SimpleNamespace(
        handle_ws_samples=lambda dev_id, payload, **kwargs: handler_called.update(
            {
                "dev_id": dev_id,
                "payload": payload,
                "lease": kwargs.get("lease_seconds"),
            }
        )
    )
    runtime = importlib.import_module(
        "custom_components.termoweb.runtime"
    ).require_runtime(client.hass, "entry")
    runtime.energy_coordinator = energy_handler
    client._forward_sample_updates(
        {"htr": {"samples": {"1": {"temp": 20}}, "lease_seconds": 30}}
    )
    assert handler_called["dev_id"] == "device"
    assert handler_called["payload"]["htr"]["1"]["temp"] == 20
    assert handler_called["lease"] == 30


def test_extract_nodes_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    """_extract_nodes should handle dicts, lists, and invalid payloads."""

    client, _ = _make_client(monkeypatch)
    inventory = Inventory(
        "device",
        build_node_inventory([{"type": "htr", "addr": "1"}]),
    )
    client._inventory = inventory
    assert client._extract_nodes({"nodes": {"htr": {}}}) == {"htr": {}}
    converted = client._extract_nodes(
        {"nodes": [{"type": "htr", "addr": "1", "status": {}}]}
    )
    assert "htr" in converted
    assert client._extract_nodes("not a dict") is None


def test_resolve_update_section_variants() -> None:
    """Update section resolver should map known segments consistently."""

    assert module.TermoWebWSClient._resolve_update_section(None) == (None, None)
    assert module.TermoWebWSClient._resolve_update_section("status") == ("status", None)
    assert module.TermoWebWSClient._resolve_update_section("advanced_setup") == (
        "advanced",
        "advanced_setup",
    )
    assert module.TermoWebWSClient._resolve_update_section("setup") == (
        "settings",
        "setup",
    )
    assert module.TermoWebWSClient._resolve_update_section("unknown") == (
        "settings",
        "unknown",
    )
