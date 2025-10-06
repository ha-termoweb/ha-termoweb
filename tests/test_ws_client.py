"""Unit tests for websocket client helpers."""

from __future__ import annotations

import asyncio
import gzip
from types import SimpleNamespace
from typing import Any, Mapping
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.termoweb.backend import ducaheat_ws
from custom_components.termoweb.backend import termoweb_ws as module


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
    coordinator = SimpleNamespace(update_nodes=MagicMock())
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
) -> ducaheat_ws.DucaheatWSClient:
    """Instantiate a Ducaheat websocket client for tests."""

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


def test_collect_update_addresses_skips_invalid() -> None:
    """Non-mapping payloads should be ignored when collecting addresses."""

    nodes: Mapping[str, Any] = {"htr": [1, 2], 10: {"settings": {"1": {}}}}
    assert module.WebSocketClient._collect_update_addresses(nodes) == []


def test_merge_nodes_combines_nested_payloads() -> None:
    """The merge helper should combine nested dictionaries in-place."""

    target = {"htr": {"1": {"temp": 20}, "2": None}}
    module.WebSocketClient._merge_nodes(target, {"htr": {"1": {"temp": 21}, "3": {"temp": 19}}})
    assert target == {"htr": {"1": {"temp": 21}, "2": None, "3": {"temp": 19}}}


def test_dispatch_nodes_updates_hass_and_coordinator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dispatching nodes should update the coordinator and hass data buckets."""

    client = _make_termoweb_client(monkeypatch)
    payload = {"nodes": {"htr": {"settings": {"1": {"temp": 20}}}}}
    addr_map = client._dispatch_nodes(payload)

    client._coordinator.update_nodes.assert_called_once()  # type: ignore[attr-defined]
    assert isinstance(addr_map, dict)
    entry_state = client.hass.data[module.DOMAIN]["entry"]
    assert "node_inventory" in entry_state
    client._dispatcher_mock.assert_called()  # type: ignore[attr-defined]


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


def test_ducaheat_base_host_uses_brand_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
    """The base host helper should derive the scheme and host from brand configuration."""

    client = _make_ducaheat_client(monkeypatch)
    monkeypatch.setattr(ducaheat_ws, "get_brand_api_base", lambda _: "https://ducaheat.example/api/v2")
    assert client._base_host() == "https://ducaheat.example"


@pytest.mark.asyncio
async def test_ducaheat_ws_url_includes_token_and_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Generating the websocket URL should include the token and device parameters."""

    client = _make_ducaheat_client(monkeypatch)
    monkeypatch.setattr(ducaheat_ws, "_rand_t", lambda: "Pabcdefg")
    monkeypatch.setattr(client, "_get_token", AsyncMock(return_value="token"))
    monkeypatch.setattr(ducaheat_ws, "get_brand_api_base", lambda _: "https://ducaheat.example")

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
