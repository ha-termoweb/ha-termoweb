"""Additional Ducaheat websocket client coverage tests."""

from __future__ import annotations

import asyncio
import copy
import logging
import json
from types import MappingProxyType, SimpleNamespace
from typing import Any, AsyncIterator, Iterable, Mapping
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from custom_components.termoweb.backend import ducaheat_ws, ws_client
from custom_components.termoweb.inventory import (
    Inventory,
    build_node_inventory,
)
from homeassistant.core import HomeAssistant


class DummyREST:
    """Provide the minimal interface required by the websocket client."""

    def __init__(self) -> None:
        self._session = SimpleNamespace()
        self._headers = {"Authorization": "Bearer rest-token"}

    async def authed_headers(self) -> dict[str, str]:
        """Return cached REST headers with an access token."""

        return self._headers


class StubResponse:
    """Async context manager returning a predetermined payload."""

    def __init__(self, *, status: int = 200, body: bytes = b"") -> None:
        self.status = status
        self._body = body
        self._read = AsyncMock(return_value=body)

    async def read(self) -> bytes:
        """Return the response body."""

        return await self._read()

    async def __aenter__(self) -> "StubResponse":
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None


class StubWebSocket:
    """Simple websocket stub capturing sent frames."""

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.closed = False
        self._receive = asyncio.Queue[str]()
        self._receive.put_nowait("3probe")

    async def send_str(self, payload: str) -> None:
        """Record outgoing websocket frames."""

        self.sent.append(payload)

    async def receive_str(self) -> str:
        """Return the next queued incoming frame."""

        return await self._receive.get()

    async def close(self, *, code: int, message: bytes) -> None:
        """Record that the websocket has been closed."""

        self.closed = True


class QueueWebSocket:
    """Queue-based websocket stub providing receive semantics."""

    def __init__(self, frames: Iterable[SimpleNamespace]) -> None:
        self.closed = False
        self.sent: list[str] = []
        self._frames = list(frames)
        self._index = 0

    def __aiter__(self) -> Any:
        async def _iterate() -> AsyncIterator[Any]:
            for frame in self._frames[self._index :]:
                yield frame

        return _iterate()

    async def receive(self) -> Any:
        if self._index < len(self._frames):
            frame = self._frames[self._index]
            self._index += 1
            return frame
        return SimpleNamespace(type=aiohttp.WSMsgType.CLOSE, data=None)

    async def send_str(self, payload: str) -> None:
        self.sent.append(payload)


async def _run_read_loop(client: Any) -> None:
    """Execute ``_read_loop_ws`` and ignore websocket closure errors."""

    try:
        await client._read_loop_ws()
    except RuntimeError as err:
        if "websocket closed" not in str(err).lower():
            raise


class StubSession:
    """Sequence driven aiohttp session stub for handshake operations."""

    def __init__(self, ws: StubWebSocket) -> None:
        self.calls: list[tuple[str, str]] = []
        self._ws = ws
        self._open_body = ducaheat_ws._encode_polling_packet(
            '0{"sid":"abc","pingInterval":25000,"pingTimeout":60000}'
        )

    def get(self, url: str, *, headers: dict[str, str]) -> StubResponse:
        """Return responses for the open and drain polling calls."""

        self.calls.append(("GET", url))
        if "sid=" not in url:
            return StubResponse(body=self._open_body)
        return StubResponse(body=b"6:40[]")

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        data: bytes,
    ) -> StubResponse:
        """Record POST invocations during handshake."""

        self.calls.append(("POST", url))
        return StubResponse(body=b"")

    async def ws_connect(self, url: str, **_: Any) -> StubWebSocket:
        """Return the preconfigured websocket stub."""

        self.calls.append(("WS", url))
        return self._ws


class DummyCoordinator:
    """Coordinator stub exposing shared data buckets."""

    def __init__(self) -> None:
        self.update_nodes = MagicMock()
        self.data: dict[str, Any] = {
            "device": {
                "nodes_by_type": {},
                "settings": {},
            }
        }


def _make_client(monkeypatch: pytest.MonkeyPatch) -> ducaheat_ws.DucaheatWSClient:
    """Create a websocket client with deterministic helpers."""

    ws = StubWebSocket()
    session = StubSession(ws)
    hass = HomeAssistant()
    hass.data.setdefault(ducaheat_ws.DOMAIN, {})["entry"] = {}
    coordinator = DummyCoordinator()
    monkeypatch.setattr(ducaheat_ws, "resolved_nodes", None, raising=False)
    client = ducaheat_ws.DucaheatWSClient(
        hass,
        entry_id="entry",
        dev_id="device",
        api_client=DummyREST(),
        coordinator=coordinator,
        session=session,  # type: ignore[arg-type]
    )
    monkeypatch.setattr(client, "_get_token", AsyncMock(return_value="token"))
    monkeypatch.setattr(ducaheat_ws, "_rand_t", lambda: "P123456")
    raw_nodes = {"nodes": [{"type": "htr", "addr": "1"}]}
    inventory = Inventory(
        "device",
        build_node_inventory(raw_nodes),
    )
    hass.data[ducaheat_ws.DOMAIN]["entry"]["inventory"] = inventory
    return client


@pytest.mark.asyncio
async def test_connect_once_performs_full_handshake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exhaust the polling and websocket upgrade handshake."""

    statuses: list[str] = []
    monkeypatch.setattr(
        ducaheat_ws.DucaheatWSClient,
        "_update_status",
        lambda self, status: statuses.append(status),
    )
    monkeypatch.setattr(
        ducaheat_ws,
        "_decode_polling_packets",
        lambda body: ['0{"sid":"abc","pingInterval":25000,"pingTimeout":60000}'],
    )
    client = _make_client(monkeypatch)
    await client._connect_once()

    assert client._ws is not None
    assert statuses[-1] == "connected"
    assert client._pending_dev_data is True
    assert all("dev_data" not in frame for frame in client._ws.sent)
    assert (
        client._ws.sent.count("3") == 0
    )  # handshake should not issue pong during setup
    await client._disconnect("test")


def test_update_status_records_health(monkeypatch: pytest.MonkeyPatch) -> None:
    """Healthy websocket updates should refresh the shared state bucket."""

    client = _make_client(monkeypatch)
    hass = client.hass

    first_ts = 1_000.0
    monkeypatch.setattr(ducaheat_ws.time, "time", lambda: first_ts)
    client._stats.frames_total = 1
    client._stats.events_total = 1
    client._stats.last_event_ts = first_ts
    client._last_event_at = first_ts

    client._update_status("healthy")

    ws_state = hass.data[ducaheat_ws.DOMAIN]["entry"]["ws_state"][client.dev_id]
    assert ws_state["status"] == "healthy"
    assert ws_state["healthy_since"] == first_ts
    assert ws_state["healthy_minutes"] == 0
    assert ws_state["last_event_at"] == first_ts

    later_ts = first_ts + 600
    monkeypatch.setattr(ducaheat_ws.time, "time", lambda: later_ts)
    client._stats.frames_total = 5
    client._stats.events_total = 3
    client._stats.last_event_ts = later_ts
    client._last_event_at = later_ts

    client._update_status("healthy")

    ws_state = hass.data[ducaheat_ws.DOMAIN]["entry"]["ws_state"][client.dev_id]
    assert ws_state["healthy_since"] == first_ts
    assert ws_state["healthy_minutes"] == 10
    assert ws_state["frames_total"] == 5


def _build_inventory_payload(addr: str = "1") -> dict[str, Any]:
    """Return a minimal node payload for inventory construction."""

    return {"nodes": [{"type": "htr", "addr": addr}]}


def _set_inventory(
    client: ducaheat_ws.DucaheatWSClient,
    payload: Mapping[str, Any],
) -> Inventory:
    """Bind a fresh inventory container to the client and hass record."""

    inventory = Inventory(client.dev_id, build_node_inventory(payload))
    hass_record = client.hass.data[ducaheat_ws.DOMAIN][client.entry_id]
    hass_record["inventory"] = inventory
    client._inventory = inventory
    return inventory


def test_dispatch_nodes_includes_inventory_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Snapshots should publish inventory metadata payloads."""

    client = _make_client(monkeypatch)
    hass = client.hass
    coordinator = client._coordinator
    client._dispatcher = MagicMock()

    energy_coordinator = SimpleNamespace(update_addresses=MagicMock())
    hass.data[ducaheat_ws.DOMAIN][client.entry_id]["energy_coordinator"] = (
        energy_coordinator
    )

    payload = {
        "htr": {
            "settings": {"1": {"target_temp": 21}},
            "samples": {"1": {"power": 1200}},
        }
    }
    inventory = _set_inventory(client, _build_inventory_payload())

    client._dispatch_nodes(payload)

    assert client._dispatcher.call_count == 1
    dispatched = client._dispatcher.call_args[0][2]
    assert "nodes" not in dispatched
    assert "nodes_by_type" not in dispatched
    assert "addresses_by_type" not in dispatched
    assert "addr_map" not in dispatched
    assert dispatched["inventory"] is inventory
    coordinator.update_nodes.assert_not_called()
    assert client._inventory.addresses_by_type["htr"] == ["1"]

    record = hass.data[ducaheat_ws.DOMAIN][client.entry_id]
    assert record.get("inventory") is inventory
    assert "sample_aliases" not in record
    energy_coordinator.update_addresses.assert_called_once_with(inventory)


def test_incremental_updates_preserve_address_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Incremental websocket merges should continue to publish addresses."""

    client = _make_client(monkeypatch)
    coordinator = client._coordinator
    client._dispatcher = MagicMock()

    _set_inventory(client, _build_inventory_payload())

    base = {"htr": {"settings": {"1": {"target_temp": 20}}}}
    client._dispatch_nodes(base)

    first_payload = client._dispatcher.call_args_list[-1][0][2]
    assert "nodes" not in first_payload
    assert "nodes_by_type" not in first_payload
    assert "addresses_by_type" not in first_payload
    assert "addr_map" not in first_payload
    assert first_payload["inventory"] is client._inventory

    update = {"htr": {"settings": {"1": {"target_temp": 23}}}}
    client._dispatch_nodes(update)

    dispatched = client._dispatcher.call_args_list[-1][0][2]
    assert "nodes" not in dispatched
    assert "nodes_by_type" not in dispatched
    assert "addresses_by_type" not in dispatched
    assert "addr_map" not in dispatched
    assert dispatched["inventory"] is client._inventory
    assert coordinator.update_nodes.call_count == 0


def test_nodes_to_deltas_translates_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Websocket node payloads should become domain deltas."""

    client = _make_client(monkeypatch)
    inventory = _set_inventory(client, _build_inventory_payload())

    nodes = {
        "htr": {
            "settings": {"1": {"mode": "auto", "ignored": "value"}},
            "status": {"1": {"online": True, "extra": "keep"}},
            "samples": {"1": {"temp": 25}},
        }
    }

    deltas = client._nodes_to_deltas(nodes, inventory=inventory)

    assert len(deltas) == 1
    delta = deltas[0]
    assert delta.node_id.node_type.value == "htr"
    assert delta.node_id.addr == "1"
    assert delta.payload["mode"] == "auto"
    assert delta.payload["status"]["online"] is True
    assert "ignored" not in delta.payload
    assert "samples" not in delta.payload


def test_nodes_to_deltas_validates_inventory(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Unknown nodes should be ignored with a warning."""

    client = _make_client(monkeypatch)
    inventory = _set_inventory(client, _build_inventory_payload())

    with caplog.at_level(logging.WARNING):
        deltas = client._nodes_to_deltas(
            {"htr": {"settings": {"9": {"mode": "eco"}}}},
            inventory=inventory,
        )

    assert deltas == []
    assert "ignoring update for unknown" in caplog.text


@pytest.mark.asyncio
async def test_emit_sio_logs_subscribe(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Debug logging should include subscription path summaries."""

    client = _make_client(monkeypatch)
    client._ws = StubWebSocket()
    caplog.set_level(
        logging.DEBUG, logger="custom_components.termoweb.backend.ducaheat_ws"
    )

    await client._emit_sio("subscribe", "/htr/1/status")
    await client._emit_sio("message", "pong")

    assert client._ws.sent == [
        '42/api/v2/socket_io,["subscribe","/htr/1/status"]',
        '42/api/v2/socket_io,["message","pong"]',
    ]
    assert any(
        "-> 42 subscribe" in record.message and "path=/htr/1/status" in record.message
        for record in caplog.records
    )
    assert any(
        "-> 42 message" in record.message and "args=('pong',)" in record.message
        for record in caplog.records
    )


def test_rand_t_token_format() -> None:
    """Random polling tokens should be eight alphanumeric characters with a P prefix."""

    token = ducaheat_ws._rand_t()

    assert token.startswith("P")
    assert len(token) == 8
    assert token[1:].isalnum()


def test_decode_polling_packets_additional_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise defensive Engine.IO polling decoder branches."""

    # Short buffers should trigger the early break condition.
    assert ducaheat_ws._decode_polling_packets(b"\x00\x00\x00") == []

    # Invalid digit bytes should abort parsing without raising.
    assert ducaheat_ws._decode_polling_packets(b"\x00\x0a") == []

    # Missing digit markers should also result in an empty decode.
    assert ducaheat_ws._decode_polling_packets(b"\x00\xff\x00\x00") == []

    # Length overruns should be handled gracefully.
    assert ducaheat_ws._decode_polling_packets(b"\x00\x02\xff\x00") == []

    class FailingPayload(bytes):
        def decode(
            self, *_: Any, **__: Any
        ) -> str:  # pragma: no cover - exercised via test
            raise ValueError

    class ByteLike:
        def __init__(self, data: bytes) -> None:
            self._data = data

        def __len__(self) -> int:
            return len(self._data)

        def __getitem__(self, item: Any) -> Any:
            result = self._data[item]
            if isinstance(item, slice) and item.start and item.start >= 3:
                return FailingPayload(result)
            return result

    # Emulate a decode failure to reach the exception branch.
    assert ducaheat_ws._decode_polling_packets(ByteLike(b"\x00\x01\xff\x00")) == []

    # Trigger gzip decompression fallback by raising from gzip.decompress.
    monkeypatch.setattr(
        ducaheat_ws, "gzip", SimpleNamespace(decompress=MagicMock(side_effect=OSError))
    )
    assert ducaheat_ws._decode_polling_packets(b"\x1f\x8bbad") == []


def test_brand_headers_apply_defaults() -> None:
    """Brand headers should retain required defaults when optional values are missing."""

    headers = ducaheat_ws._brand_headers("", "")

    assert headers["User-Agent"] == ducaheat_ws.USER_AGENT
    assert headers["X-Requested-With"] == ""
    assert headers["Accept-Language"] == ducaheat_ws.ACCEPT_LANGUAGE


def test_client_requires_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing without an aiohttp session should fail."""

    hass = SimpleNamespace(loop=asyncio.new_event_loop(), data={})
    with pytest.raises(RuntimeError):
        ducaheat_ws.DucaheatWSClient(
            hass,
            entry_id="entry",
            dev_id="device",
            api_client=SimpleNamespace(_session=None),  # type: ignore[arg-type]
            coordinator=SimpleNamespace(),
        )


@pytest.mark.asyncio
async def test_start_and_runner_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    """_runner should cycle through connect, read, and disconnect before stopping."""

    client = _make_client(monkeypatch)
    statuses: list[str] = []
    monkeypatch.setattr(client, "_disconnect", AsyncMock())
    monkeypatch.setattr(
        client, "_update_status", lambda status: statuses.append(status)
    )

    async def _connect_once() -> None:
        statuses.append("connect_once")

    async def _read_loop() -> None:
        statuses.append("read_loop")
        raise asyncio.CancelledError

    monkeypatch.setattr(client, "_connect_once", AsyncMock(side_effect=_connect_once))
    monkeypatch.setattr(client, "_read_loop_ws", AsyncMock(side_effect=_read_loop))

    await client._runner()

    assert statuses[0] == "starting"
    assert "connect_once" in statuses
    assert "read_loop" in statuses
    assert statuses[-1] == "stopped"


@pytest.mark.asyncio
async def test_start_method_reuses_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """start should cache the created asyncio task."""

    client = _make_client(monkeypatch)
    monkeypatch.setattr(client, "_runner", AsyncMock(return_value=None))

    task_one = client.start()
    task_two = client.start()

    assert task_one is task_two
    await asyncio.wait_for(task_one, 0.1)


def test_is_running_reflects_task_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """is_running should track whether the background task is active."""

    client = _make_client(monkeypatch)
    monkeypatch.setattr(client, "_runner", AsyncMock(return_value=None))

    assert client.is_running() is False
    task = client.start()
    assert client.is_running() is True
    loop = client._loop
    loop.run_until_complete(asyncio.sleep(0))
    assert client.is_running() is False


def test_extract_dev_data_payload_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    """dev_data payload extraction should walk nested containers safely."""

    client = _make_client(monkeypatch)
    client._inventory = Inventory(
        client.dev_id,
        build_node_inventory([{"type": "htr", "addr": "1"}]),
    )
    nodes = {"nodes": {"htr": {}}}
    wrapper = {"data": nodes}

    result = client._extract_dev_data_payload([wrapper, wrapper])
    assert result is nodes

    list_wrapper = [[nodes]]
    result = client._extract_dev_data_payload(list_wrapper)
    assert result is nodes

    tuple_wrapper = ({"body": nodes},)
    result = client._extract_dev_data_payload([tuple_wrapper])
    assert result is nodes

    list_payload = client._extract_dev_data_payload(
        [[{"nodes": [{"type": "htr", "addr": "1", "status": {"p": 1}}]}]]
    )
    assert isinstance(list_payload, Mapping)
    nodes_map = list_payload.get("nodes") if isinstance(list_payload, Mapping) else None
    assert isinstance(nodes_map, Mapping)
    assert nodes_map["htr"]["status"]["1"]["p"] == 1

    assert client._extract_dev_data_payload(["not json"]) is None


@pytest.mark.asyncio
async def test_connect_once_open_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Different handshake failures should surface as HandshakeError instances."""

    async def run(
        expected_status: int,
        *,
        open_status: int = 200,
        post_status: int = 200,
        drain_status: int = 200,
        decode_return: list[str] | None = None,
        open_body: bytes | None = None,
    ) -> None:
        with monkeypatch.context() as patch:
            client = _make_client(patch)
            ws = client._session._ws  # type: ignore[attr-defined]
            call_count = 0

            if decode_return is not None:
                patch.setattr(
                    ducaheat_ws, "_decode_polling_packets", lambda _body: decode_return
                )

            def fake_get(url: str, *, headers: dict[str, str]) -> StubResponse:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return StubResponse(
                        status=open_status, body=open_body or client._session._open_body
                    )  # type: ignore[attr-defined]
                return StubResponse(status=drain_status)

            def fake_post(
                url: str, *, headers: dict[str, str], data: bytes
            ) -> StubResponse:
                return StubResponse(status=post_status)

            patch.setattr(client._session, "get", fake_get)
            patch.setattr(client._session, "post", fake_post)
            patch.setattr(client._session, "ws_connect", AsyncMock(return_value=ws))

            with pytest.raises(ducaheat_ws.HandshakeError) as err:
                await client._connect_once()
            assert err.value.status == expected_status

    await run(403, open_status=403)
    await run(590, decode_return=[])
    await run(592, decode_return=['0{"pingInterval":1,"pingTimeout":1}'])
    await run(500, decode_return=['0{"sid":"abc"}'], post_status=500)
    await run(504, decode_return=['0{"sid":"abc"}'], drain_status=504)


@pytest.mark.asyncio
async def test_connect_once_probe_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unexpected probe acknowledgements should not abort the handshake."""

    statuses: list[str] = []
    monkeypatch.setattr(
        ducaheat_ws.DucaheatWSClient,
        "_update_status",
        lambda self, status: statuses.append(status),
    )
    client = _make_client(monkeypatch)
    monkeypatch.setattr(
        ducaheat_ws,
        "_decode_polling_packets",
        lambda _body: ['0{"sid":"abc"}'],
    )
    assert client._session._ws is not None  # type: ignore[attr-defined]
    client._session._ws._receive = asyncio.Queue()  # type: ignore[attr-defined]
    client._session._ws._receive.put_nowait("weird")  # type: ignore[attr-defined]

    await client._connect_once()

    assert statuses[-1] == "connected"
    assert any(frame == "5" for frame in client._ws.sent)  # type: ignore[union-attr]
    await client._disconnect("test")


@pytest.mark.asyncio
async def test_keepalive_loop_sends_engineio_pings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keepalive loop should honour the negotiated ping cadence."""

    original_sleep = asyncio.sleep
    sleeps: list[float] = []

    async def fast_sleep(delay: float) -> None:
        sleeps.append(delay)
        await original_sleep(0)

    client = _make_client(monkeypatch)
    monkeypatch.setattr(ducaheat_ws.asyncio, "sleep", fast_sleep)
    monkeypatch.setattr(
        ducaheat_ws,
        "_decode_polling_packets",
        lambda _body: ['0{"sid":"abc","pingInterval":200,"pingTimeout":600}'],
    )

    await client._connect_once()
    client._start_keepalive()  # should be a no-op while the loop task is active
    await original_sleep(0)
    await original_sleep(0)

    assert any(frame == "2" for frame in client._ws.sent)
    assert sleeps and pytest.approx(0.18, rel=0.05) == sleeps[0]

    await client._disconnect("test")
    assert client._keepalive_task is None
    client._start_keepalive()  # should be a no-op without websocket context


@pytest.mark.asyncio
async def test_read_loop_marks_healthy_on_engineio_pong(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Receiving a pong frame should mark the websocket as healthy."""

    client = _make_client(monkeypatch)
    statuses: list[str] = []
    monkeypatch.setattr(
        client, "_update_status", lambda status: statuses.append(status)
    )

    class PongWS(QueueWebSocket):
        def __init__(self) -> None:
            super().__init__(
                [
                    SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="3"),
                    SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="43"),
                ]
            )

    client._ws = PongWS()  # type: ignore[assignment]

    await _run_read_loop(client)

    assert statuses and statuses[-1] == "healthy"


@pytest.mark.asyncio
async def test_read_loop_updates_ws_state_on_dev_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Processing dev_data payloads should mark the websocket healthy."""

    client = _make_client(monkeypatch)
    hass = client.hass
    client._coordinator.update_nodes = MagicMock()
    monkeypatch.setattr(client, "_subscribe_feeds", AsyncMock(return_value=0))

    base_ts = 1_234.0
    monkeypatch.setattr(ducaheat_ws.time, "time", lambda: base_ts)

    payload = json.dumps(
        ["dev_data", {"nodes": {"htr": {"status": {"1": {"power": 1}}}}}],
        separators=(",", ":"),
    )

    class DevDataWS(QueueWebSocket):
        def __init__(self, frame: str) -> None:
            super().__init__([SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=frame)])

    frame = f"42{client._namespace},{payload}"
    client._ws = DevDataWS(frame)  # type: ignore[assignment]

    await _run_read_loop(client)

    ws_state = hass.data[ducaheat_ws.DOMAIN]["entry"]["ws_state"][client.dev_id]
    assert ws_state["status"] == "healthy"
    assert ws_state["healthy_since"] == base_ts
    assert ws_state["last_event_at"] == base_ts
    client._coordinator.update_nodes.assert_not_called()


@pytest.mark.asyncio
async def test_read_loop_handles_stringified_dev_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The client should decode string-wrapped dev_data payloads."""

    client = _make_client(monkeypatch)
    monkeypatch.setattr(client, "_subscribe_feeds", AsyncMock(return_value=1))
    dispatched: list[dict[str, Any]] = []
    client._dispatcher = lambda *_args: dispatched.append(_args[2])

    nodes = {"htr": {"status": {"1": {"power": 5}}}}
    wrapped = json.dumps({"nodes": nodes}, separators=(",", ":"))
    payload = json.dumps(["dev_data", wrapped], separators=(",", ":"))

    class DevDataWS(QueueWebSocket):
        def __init__(self, frame: str) -> None:
            super().__init__([SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=frame)])

    frame = f"42{client._namespace},{payload}"
    client._ws = DevDataWS(frame)  # type: ignore[assignment]

    await _run_read_loop(client)

    assert getattr(client, "_nodes_raw", None) is None
    assert dispatched
    payload = dispatched[-1]
    assert payload["inventory"] is client._inventory
    assert "addresses_by_type" not in payload
    assert "addr_map" not in payload
    client._subscribe_feeds.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_read_loop_handles_list_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """List-based dev_data snapshots should be coerced into mapping payloads."""

    client = _make_client(monkeypatch)
    client._inventory = Inventory(
        client.dev_id,
        build_node_inventory([{"type": "htr", "addr": "1"}]),
    )
    monkeypatch.setattr(client, "_subscribe_feeds", AsyncMock(return_value=2))
    dispatched: list[dict[str, Any]] = []
    client._dispatcher = lambda *_args: dispatched.append(_args[2])

    nodes_list = [
        {"type": "htr", "addr": "1", "status": {"power": 7}, "lease_seconds": 90}
    ]
    payload = json.dumps(["dev_data", {"nodes": nodes_list}], separators=(",", ":"))

    class DevDataWS(QueueWebSocket):
        def __init__(self, frame: str) -> None:
            super().__init__([SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=frame)])

    frame = f"42{client._namespace},{payload}"
    client._ws = DevDataWS(frame)  # type: ignore[assignment]

    await _run_read_loop(client)

    assert getattr(client, "_nodes_raw", None) is None
    assert dispatched
    payload = dispatched[-1]
    assert payload["inventory"] is client._inventory
    assert "addresses_by_type" not in payload
    assert "addr_map" not in payload
    client._subscribe_feeds.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_keepalive_loop_handles_ws_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keepalive loop should tolerate websocket swaps and errors."""

    original_sleep = asyncio.sleep

    class ErrorWS(StubWebSocket):
        def __init__(self) -> None:
            super().__init__()
            self.closed = False
            self.raise_error = False

        async def send_str(self, payload: str) -> None:
            if self.raise_error:
                raise RuntimeError("boom")
            await super().send_str(payload)

    client = _make_client(monkeypatch)
    first_ws = ErrorWS()
    second_ws = ErrorWS()
    client._ws = first_ws  # type: ignore[assignment]
    client._ping_interval = 0.2

    counter = {"count": 0}

    async def fast_sleep(delay: float) -> None:
        counter["count"] += 1
        if counter["count"] == 2:
            client._ws = second_ws  # type: ignore[assignment]
            second_ws.raise_error = True
        await original_sleep(0)

    monkeypatch.setattr(ducaheat_ws.asyncio, "sleep", fast_sleep)

    task = asyncio.create_task(client._keepalive_loop())
    client._keepalive_task = task

    await original_sleep(0)
    await original_sleep(0)
    await original_sleep(0)

    assert first_ws.sent.count("2") == 1
    assert second_ws.raise_error is True
    await task
    assert client._keepalive_task is None

    client._ws = second_ws  # type: ignore[assignment]
    client._ping_interval = None
    await client._keepalive_loop()


@pytest.mark.asyncio
async def test_read_loop_additional_flows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise ping, namespace, and event handling branches."""

    client = _make_client(monkeypatch)
    send_history: list[str] = []

    class RichWS(QueueWebSocket):
        def __init__(self) -> None:
            super().__init__(
                [
                    SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="1ignore"),
                    SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="42"),
                    SimpleNamespace(
                        type=aiohttp.WSMsgType.TEXT, data='42/api,["ping"]'
                    ),
                    SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="442/invalid"),
                    SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="442invalid"),
                    SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="442[]"),
                    SimpleNamespace(
                        type=aiohttp.WSMsgType.TEXT, data='442["message","ping"]'
                    ),
                    SimpleNamespace(
                        type=aiohttp.WSMsgType.TEXT, data='442["other",{}]'
                    ),
                    SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="440"),
                    SimpleNamespace(
                        type=aiohttp.WSMsgType.TEXT,
                        data='442["dev_data",{"nodes":{"htr":{"samples":{"1":{}}}}}]',
                    ),
                    SimpleNamespace(
                        type=aiohttp.WSMsgType.TEXT,
                        data='442["update",{"body":{"temp":1},"path":"/path"}]',
                    ),
                ]
            )

        async def send_str(self, payload: str) -> None:
            send_history.append(payload)
            await super().send_str(payload)

    monkeypatch.setattr(client, "_dispatch_nodes", MagicMock())
    emit_mock = AsyncMock()
    monkeypatch.setattr(client, "_emit_sio", emit_mock)
    updates: list[str] = []
    monkeypatch.setattr(client, "_update_status", lambda status: updates.append(status))
    client._ws = RichWS()  # type: ignore[assignment]

    await _run_read_loop(client)

    assert send_history.count("3/api") == 1
    assert any(call.args == ("message", "pong") for call in emit_mock.await_args_list)
    assert "healthy" in updates


@pytest.mark.asyncio
async def test_read_loop_handles_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Error frames should raise runtime exceptions."""

    client = _make_client(monkeypatch)

    class ErrorWS(QueueWebSocket):
        def __init__(self) -> None:
            super().__init__([SimpleNamespace(type=aiohttp.WSMsgType.ERROR, data=None)])

        def exception(self) -> str:
            return "boom"

    client._ws = ErrorWS()  # type: ignore[assignment]

    with pytest.raises(RuntimeError):
        await client._read_loop_ws()


@pytest.mark.asyncio
async def test_read_loop_handles_close(monkeypatch: pytest.MonkeyPatch) -> None:
    """Close frames should surface as runtime errors."""

    client = _make_client(monkeypatch)

    class CloseWS(QueueWebSocket):
        def __init__(self) -> None:
            super().__init__([SimpleNamespace(type=aiohttp.WSMsgType.CLOSE, data=None)])

    client._ws = CloseWS()  # type: ignore[assignment]

    with pytest.raises(RuntimeError):
        await client._read_loop_ws()


@pytest.mark.asyncio
async def test_read_loop_processes_update_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Update events should merge payloads and mark the connection healthy."""

    client = _make_client(monkeypatch)
    log_calls: list[Any] = []
    monkeypatch.setattr(
        client, "_log_update_brief", lambda body: log_calls.append(body)
    )
    statuses: list[str] = []
    monkeypatch.setattr(
        client, "_update_status", lambda status: statuses.append(status)
    )
    dispatched: list[dict[str, Any]] = []
    client._dispatcher = lambda *_args: dispatched.append(_args[2])
    forwarded: list[Mapping[str, Mapping[str, Any]]] = []
    monkeypatch.setattr(
        client,
        "_forward_sample_updates",
        lambda updates: forwarded.append(updates),
    )

    class UpdateWS(QueueWebSocket):
        def __init__(self) -> None:
            super().__init__(
                [
                    SimpleNamespace(
                        type=aiohttp.WSMsgType.TEXT,
                        data='442["update",{"body":{"temp":1},"path":"/api/v2/devs/device/htr/1/status"}]',
                    )
                ]
            )

    client._ws = UpdateWS()  # type: ignore[assignment]

    await _run_read_loop(client)

    assert log_calls and log_calls[0]["body"]["temp"] == 1
    assert statuses and statuses[-1] == "healthy"
    assert getattr(client, "_nodes_raw", None) is None
    assert dispatched
    payload = dispatched[-1]
    assert payload["inventory"] is client._inventory
    assert "addresses_by_type" not in payload
    assert "addr_map" not in payload
    assert not forwarded


@pytest.mark.asyncio
async def test_read_loop_applies_deltas_to_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Update events should translate into domain deltas for the coordinator."""

    client = _make_client(monkeypatch)
    _set_inventory(client, _build_inventory_payload())
    applied: list[tuple[str, tuple[Any, ...], bool]] = []
    client._coordinator.handle_ws_deltas = (
        lambda dev_id, deltas, *, replace=False: applied.append(
            (dev_id, deltas, replace)
        )
    )
    client._dispatcher = lambda *_args: None
    client._forward_sample_updates = lambda updates: None

    class UpdateWS(QueueWebSocket):
        def __init__(self) -> None:
            super().__init__(
                [
                    SimpleNamespace(
                        type=aiohttp.WSMsgType.TEXT,
                        data='442["update",{"body":{"mode":"eco"},"path":"/api/v2/devs/device/htr/1/settings"}]',
                    )
                ]
            )

    client._ws = UpdateWS()  # type: ignore[assignment]

    await _run_read_loop(client)

    assert applied
    dev_id, deltas, replace = applied[-1]
    assert dev_id == "device"
    assert replace is False
    assert deltas[0].payload["mode"] == "eco"


def test_translate_path_update_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    """Path-based websocket updates should translate into node payloads."""

    client = _make_client(monkeypatch)
    translate = lambda payload: ws_client.translate_path_update(
        payload,
        resolve_section=ducaheat_ws.DucaheatWSClient._resolve_update_section,
    )
    payload = {
        "path": "/api/v2/devs/device/htr/2/settings/setup",
        "body": {"mode": "auto"},
    }

    translated = translate(payload)

    assert translated == {"htr": {"settings": {"2": {"setup": {"mode": "auto"}}}}}
    assert client._translate_path_update(payload) == translated

    nested = translate(
        {"path": "/api/v2/devs/device/htr/2/setup", "body": {"mode": "eco"}}
    )
    assert nested == {"htr": {"settings": {"2": {"setup": {"mode": "eco"}}}}}
    assert (
        client._translate_path_update(
            {"path": "/api/v2/devs/device/htr/2/setup", "body": {"mode": "eco"}}
        )
        == nested
    )

    invalid_payloads = [
        {"path": "/", "body": {}},
        {"path": "/htr", "body": {}},
        {"path": "/api/v2/devs/device/htr", "body": {}},
        "not a mapping",
        {"nodes": {}},
        {"path": "/api/v2/devs/device/htr/2", "body": {}},
        {"path": "/api/v2/devs/device/htr/2/status"},
        {"path": None, "body": {}},
        {"path": "/api/v2/devs/device/htr//status", "body": {"temp": 1}},
        {"path": "/api/v2/devs/device/htr/ /status", "body": {"temp": 1}},
    ]
    for payload in invalid_payloads:
        assert translate(payload) is None
        assert client._translate_path_update(payload) is None

    assert ducaheat_ws.DucaheatWSClient._resolve_update_section(None) == (None, None)
    assert ducaheat_ws.DucaheatWSClient._resolve_update_section("ADVANCED_SETUP") == (
        "advanced",
        "advanced_setup",
    )
    assert ducaheat_ws.DucaheatWSClient._resolve_update_section("prog") == (
        "settings",
        "prog",
    )
    assert ducaheat_ws.DucaheatWSClient._resolve_update_section("unknown") == (
        "settings",
        "unknown",
    )


@pytest.mark.asyncio
async def test_read_loop_forwards_sample_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sample updates should be forwarded to the energy coordinator."""

    client = _make_client(monkeypatch)
    forwarded: list[tuple[str, Mapping[str, Any]]] = []

    def _handler(dev_id: str, payload: Mapping[str, Any], **kwargs: Any) -> None:
        forwarded.append((dev_id, payload, kwargs))

    client.hass.data[ducaheat_ws.DOMAIN]["entry"]["energy_coordinator"] = (
        SimpleNamespace(handle_ws_samples=_handler)
    )
    statuses: list[str] = []
    monkeypatch.setattr(
        client, "_update_status", lambda status: statuses.append(status)
    )
    dispatched: list[dict[str, Any]] = []
    client._dispatcher = lambda *_args: dispatched.append(_args[2])

    class UpdateWS(QueueWebSocket):
        def __init__(self) -> None:
            super().__init__(
                [
                    SimpleNamespace(
                        type=aiohttp.WSMsgType.TEXT,
                        data='442["update",{"body":{"power":10},"path":"/api/v2/devs/device/htr/1/samples"}]',
                    )
                ]
            )

    client._ws = UpdateWS()  # type: ignore[assignment]

    await _run_read_loop(client)

    assert statuses and statuses[-1] == "healthy"
    assert getattr(client, "_nodes_raw", None) is None
    assert dispatched
    payload = dispatched[-1]
    assert payload["inventory"] is client._inventory
    assert "addresses_by_type" not in payload
    assert "addr_map" not in payload
    assert forwarded and forwarded[-1][0] == "device"
    assert forwarded[-1][2].get("lease_seconds") is None
    assert forwarded[-1][1]["htr"]["1"]["power"] == 10


@pytest.mark.asyncio
async def test_read_loop_dev_data_uses_raw_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback to raw nodes when the normalised payload is not a mapping."""

    client = _make_client(monkeypatch)
    monkeypatch.setattr(client, "_normalise_nodes_payload", lambda nodes: "bad")
    monkeypatch.setattr(client, "_subscribe_feeds", AsyncMock(return_value=0))
    dispatched: list[dict[str, Any]] = []
    client._dispatcher = lambda *_args: dispatched.append(_args[2])

    class DevDataWS(QueueWebSocket):
        def __init__(self) -> None:
            super().__init__(
                [
                    SimpleNamespace(
                        type=aiohttp.WSMsgType.TEXT,
                        data='442["dev_data",{"nodes":{"htr":{"settings":{"1":{}}}}}]',
                    )
                ]
            )

    client._ws = DevDataWS()  # type: ignore[assignment]

    await _run_read_loop(client)

    assert dispatched
    payload = dispatched[-1]
    assert payload["inventory"] is client._inventory
    assert "addresses_by_type" not in payload
    assert "addr_map" not in payload
    assert getattr(client, "_nodes_raw", None) is None


@pytest.mark.asyncio
async def test_read_loop_does_not_cache_incremental_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Incremental updates should not mutate an internal snapshot cache."""

    client = _make_client(monkeypatch)
    monkeypatch.setattr(client, "_forward_sample_updates", lambda updates: None)
    dispatched: list[dict[str, Any]] = []
    client._dispatcher = lambda *_args: dispatched.append(_args[2])

    class MergeWS(QueueWebSocket):
        def __init__(self) -> None:
            super().__init__(
                [
                    SimpleNamespace(
                        type=aiohttp.WSMsgType.TEXT,
                        data='442["update",{"body":{"power":5},"path":"/api/v2/devs/device/htr/1/status"}]',
                    )
                ]
            )

    client._ws = MergeWS()  # type: ignore[assignment]

    await _run_read_loop(client)

    assert getattr(client, "_nodes_raw", None) is None
    assert dispatched
    payload = dispatched[-1]
    assert payload["inventory"] is client._inventory
    assert "addresses_by_type" not in payload
    assert "addr_map" not in payload


def test_normalise_nodes_payload_handles_mappings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The normaliser helper should coerce mappings and handle errors."""

    client = _make_client(monkeypatch)

    class DummyMapping(Mapping[str, Any]):
        def __init__(self, data: Mapping[str, Any]) -> None:
            self._data = data

        def __getitem__(self, key: str) -> Any:
            return self._data[key]

        def __iter__(self):  # type: ignore[override]
            return iter(self._data)

        def __len__(self) -> int:
            return len(self._data)

        def __deepcopy__(self, memo: dict[int, Any]) -> "DummyMapping":
            return DummyMapping(copy.deepcopy(self._data, memo))

    proxy = DummyMapping({"htr": {}})
    result = client._normalise_nodes_payload(proxy)
    assert isinstance(result, dict)

    class NormalisingREST(DummyREST):
        def normalise_ws_nodes(self, nodes: Mapping[str, Any]) -> Mapping[str, Any]:
            return MappingProxyType({"htr": {"status": {}}})

    client._client = NormalisingREST()
    result = client._normalise_nodes_payload({"htr": {}})
    assert result == {"htr": {"status": {}}}

    class RaisingREST(DummyREST):
        def normalise_ws_nodes(self, nodes: Mapping[str, Any]) -> Mapping[str, Any]:
            raise RuntimeError

    client._client = RaisingREST()
    result = client._normalise_nodes_payload({"htr": {}})
    assert isinstance(result, dict)

    class ListREST(DummyREST):
        def normalise_ws_nodes(self, nodes: Mapping[str, Any]) -> list[str]:
            return ["ok"]

    client._client = ListREST()
    result = client._normalise_nodes_payload({"htr": {}})
    assert result == ["ok"]


def test_collect_sample_updates_filters_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sample extraction should ignore invalid keys and addresses."""

    client = _make_client(monkeypatch)
    payload: Mapping[str, Any] = {
        "htr": {"samples": {"": {"power": 1}, "1": {"power": 2}}},
        123: {"samples": {"1": {"power": 3}}},
        "acm": {"status": {"1": {}}},
    }

    result = client._collect_sample_updates(payload)

    assert result == {"htr": {"samples": {"1": {"power": 2}}, "lease_seconds": None}}


def test_collect_sample_updates_updates_payload_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cadence hints should tune the websocket payload stale window."""

    client = _make_client(monkeypatch)

    assert client._payload_stale_after == pytest.approx(240.0)

    payload: Mapping[str, Any] = {
        "htr": {
            "samples": {"1": {"power": 5}},
            "lease_seconds": 60,
        }
    }

    result = client._collect_sample_updates(payload)

    assert result["htr"]["lease_seconds"] == 60
    assert client._payload_window_hint == pytest.approx(60.0)
    assert client._payload_stale_after == pytest.approx(75.0)
    state = client._ws_state_bucket()
    assert state["payload_stale_after"] == pytest.approx(75.0)
    assert state["payload_window_source"] == "sample_updates"


def test_forward_sample_updates_handles_guard_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guard clauses in the sample forwarder should short-circuit cleanly."""

    client = _make_client(monkeypatch)
    client.hass.data = {}
    client._forward_sample_updates({"htr": {"samples": {"1": {"power": 1}}}})

    client = _make_client(monkeypatch)
    client.hass.data[ducaheat_ws.DOMAIN]["entry"]["energy_coordinator"] = object()
    client._forward_sample_updates({"htr": {"samples": {"1": {"power": 1}}}})


def test_forward_sample_updates_handles_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exceptions raised by the coordinator should be swallowed."""

    client = _make_client(monkeypatch)

    class FailingCoordinator:
        def handle_ws_samples(self, *_: Any, **__: Any) -> None:
            raise RuntimeError

    client.hass.data[ducaheat_ws.DOMAIN]["entry"]["energy_coordinator"] = (
        FailingCoordinator()
    )
    client._forward_sample_updates({"htr": {"samples": {"1": {"power": 1}}}})


@pytest.mark.asyncio
async def test_disconnect_resets_payload_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disconnecting should restore the default payload window."""

    client = _make_client(monkeypatch)
    client._apply_payload_window_hint(source="test", lease_seconds=30)

    assert client._payload_stale_after == pytest.approx(45.0)

    await client._disconnect("testing")

    assert client._payload_stale_after == pytest.approx(240.0)
    state = client._ws_state_bucket()
    assert state["payload_window_hint"] is None
    assert state["payload_window_source"] == "disconnect"


def test_log_nodes_summary_branches(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Ensure node summary logging honours the configured log level."""

    client = _make_client(monkeypatch)
    nodes = {"htr": {"samples": {"1": {}}, "settings": {"2": {}}, "status": 3}}

    monkeypatch.setattr(ducaheat_ws._LOGGER, "isEnabledFor", lambda level: False)
    client._log_nodes_summary(nodes)

    monkeypatch.setattr(ducaheat_ws._LOGGER, "isEnabledFor", lambda level: True)
    with caplog.at_level(logging.INFO):
        client._log_nodes_summary(nodes)

    assert "htr=2" in caplog.text


def test_log_update_brief_variants(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """_log_update_brief should handle mapping and non-mapping payloads."""

    client = _make_client(monkeypatch)
    monkeypatch.setattr(ducaheat_ws._LOGGER, "isEnabledFor", lambda level: True)

    with caplog.at_level(logging.DEBUG):
        client._log_update_brief(
            {"path": "/x", "body": {"one": 1, "two": 2, "three": 3}}
        )
        client._log_update_brief("not a mapping")

    assert "path=/x" in caplog.text


@pytest.mark.asyncio
async def test_emit_sio_requires_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Attempting to emit without a websocket should raise a runtime error."""

    client = _make_client(monkeypatch)
    client._ws = None

    with pytest.raises(RuntimeError):
        await client._emit_sio("evt")


@pytest.mark.asyncio
async def test_emit_sio_sends_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """_emit_sio should serialise payloads and forward them to the websocket."""

    client = _make_client(monkeypatch)

    class RecorderWS:
        def __init__(self) -> None:
            self.closed = False
            self.sent: list[str] = []

        async def send_str(self, payload: str) -> None:
            self.sent.append(payload)

    ws = RecorderWS()
    client._ws = ws  # type: ignore[assignment]

    await client._emit_sio("sample", {"x": 1})

    assert ws.sent == ["42" + ducaheat_ws.DUCAHEAT_NAMESPACE + ',["sample",{"x":1}]']


@pytest.mark.asyncio
async def test_get_token_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """The access token should be extracted from the Authorization header."""

    client = _make_client(monkeypatch)
    client._get_token = ducaheat_ws.DucaheatWSClient._get_token.__get__(  # type: ignore[method-assign]
        client, ducaheat_ws.DucaheatWSClient
    )
    token = await client._get_token()

    assert token == "rest-token"


@pytest.mark.asyncio
async def test_read_loop_returns_when_websocket_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The websocket reader should no-op when no connection is bound."""

    client = _make_client(monkeypatch)
    client._ws = None

    await _run_read_loop(client)


@pytest.mark.asyncio
async def test_read_loop_handles_engineio_and_socketio_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Feed assorted Engine.IO frames through the websocket reader."""

    client = _make_client(monkeypatch)
    send_history: list[str] = []
    client._ws = SimpleNamespace(closed=False)

    class FakeWS(QueueWebSocket):
        def __init__(self) -> None:
            super().__init__(
                [
                    SimpleNamespace(
                        type=aiohttp.WSMsgType.TEXT,
                        data="2",
                    ),
                    SimpleNamespace(
                        type=aiohttp.WSMsgType.TEXT,
                        data='42["message","ping"]',
                    ),
                    SimpleNamespace(
                        type=aiohttp.WSMsgType.TEXT,
                        data="40",
                    ),
                    SimpleNamespace(
                        type=aiohttp.WSMsgType.TEXT,
                        data='42{"evt":"ignored"}',
                    ),
                    SimpleNamespace(
                        type=aiohttp.WSMsgType.TEXT,
                        data='42["dev_data",{"nodes":{"htr":{"settings":{"1":{}}}}}]',
                    ),
                    SimpleNamespace(
                        type=aiohttp.WSMsgType.TEXT,
                        data='42["update",{"body":{"path":{}}}]',
                    ),
                ]
            )

        async def send_str(self, payload: str) -> None:
            send_history.append(payload)
            await super().send_str(payload)

    fake_ws = FakeWS()
    client._ws = fake_ws  # type: ignore[assignment]
    client._pending_dev_data = True

    monkeypatch.setattr(client, "_dispatch_nodes", MagicMock())
    emit_mock = AsyncMock()
    monkeypatch.setattr(client, "_emit_sio", emit_mock)
    monkeypatch.setattr(client, "_update_status", lambda status: None)
    await _run_read_loop(client)

    assert any(frame == "3" for frame in send_history)
    assert any(call.args == ("dev_data",) for call in emit_mock.await_args_list)


@pytest.mark.asyncio
async def test_namespace_ack_replays_cached_subscriptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Namespace acknowledgements should trigger dev_data and subscription replay."""

    client = _make_client(monkeypatch)

    class AckWS(QueueWebSocket):
        def __init__(self) -> None:
            super().__init__(
                [
                    SimpleNamespace(
                        type=aiohttp.WSMsgType.TEXT,
                        data="40/api/v2/socket_io",
                    )
                ]
            )

        async def send_str(self, payload: str) -> None:
            await super().send_str(payload)

    client._ws = AckWS()  # type: ignore[assignment]
    client._pending_dev_data = True
    client._subscription_paths = {"/htr/1/status", "/htr/1/samples"}

    emit_calls: list[tuple[Any, ...]] = []

    async def _record_emit(event: str, *args: Any) -> None:
        emit_calls.append((event, *args))

    monkeypatch.setattr(client, "_emit_sio", AsyncMock(side_effect=_record_emit))
    statuses: list[str] = []
    monkeypatch.setattr(
        client, "_update_status", lambda status: statuses.append(status)
    )

    await _run_read_loop(client)

    assert client._pending_dev_data is False
    assert statuses and statuses[-1] == "healthy"
    assert ("dev_data",) == emit_calls[0]
    assert emit_calls[1:] == [
        ("subscribe", "/htr/1/samples"),
        ("subscribe", "/htr/1/status"),
    ]


@pytest.mark.asyncio
async def test_namespace_ack_processes_embedded_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Namespace frames with embedded events should be processed."""

    client = _make_client(monkeypatch)

    class AckWS(QueueWebSocket):
        def __init__(self) -> None:
            super().__init__(
                [
                    SimpleNamespace(
                        type=aiohttp.WSMsgType.TEXT,
                        data=(
                            "40/api/v2/socket_io,42/api/v2/socket_io,"
                            '["dev_data",{"nodes":{"htr":{"status":{"1":{}}}}}]'
                        ),
                    )
                ]
            )

        async def send_str(self, payload: str) -> None:
            await super().send_str(payload)

    client._ws = AckWS()  # type: ignore[assignment]
    client._pending_dev_data = True

    emit_calls: list[tuple[Any, ...]] = []

    async def _record_emit(event: str, *args: Any) -> None:
        emit_calls.append((event, *args))

    monkeypatch.setattr(client, "_emit_sio", AsyncMock(side_effect=_record_emit))
    monkeypatch.setattr(client, "_replay_subscription_paths", AsyncMock())
    monkeypatch.setattr(client, "_log_nodes_summary", lambda nodes: None)
    monkeypatch.setattr(client, "_normalise_nodes_payload", lambda nodes: nodes)
    dispatch = MagicMock()
    monkeypatch.setattr(client, "_dispatch_nodes", dispatch)
    subscribe_mock = AsyncMock(return_value=2)
    monkeypatch.setattr(client, "_subscribe_feeds", subscribe_mock)
    statuses: list[str] = []
    monkeypatch.setattr(
        client, "_update_status", lambda status: statuses.append(status)
    )

    await _run_read_loop(client)

    assert emit_calls and emit_calls[0] == ("dev_data",)
    subscribe_mock.assert_awaited_once()
    dispatch.assert_called_once()
    assert statuses and statuses[0] == "healthy"
    assert statuses[-1] == "healthy"


@pytest.mark.asyncio
async def test_namespace_ack_skips_unexpected_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected namespaces should be skipped while replaying embedded events."""

    client = _make_client(monkeypatch)

    class UnexpectedNamespaceWS(QueueWebSocket):
        def __init__(self) -> None:
            super().__init__(
                [
                    SimpleNamespace(
                        type=aiohttp.WSMsgType.TEXT,
                        data=(
                            "40/wrong,42/api/v2/socket_io,"
                            '["dev_data",{"nodes":{"htr":{"status":{"1":{}}}}}]'
                        ),
                    )
                ]
            )

        async def send_str(self, payload: str) -> None:
            await super().send_str(payload)

    client._ws = UnexpectedNamespaceWS()  # type: ignore[assignment]
    client._pending_dev_data = False

    monkeypatch.setattr(client, "_log_nodes_summary", lambda nodes: None)
    monkeypatch.setattr(client, "_normalise_nodes_payload", lambda nodes: nodes)
    dispatch = MagicMock()
    monkeypatch.setattr(client, "_dispatch_nodes", dispatch)
    subscribe_mock = AsyncMock(return_value=0)
    monkeypatch.setattr(client, "_subscribe_feeds", subscribe_mock)
    monkeypatch.setattr(client, "_update_status", lambda status: None)

    await _run_read_loop(client)

    dispatch.assert_called_once()
    subscribe_mock.assert_awaited_once()
    assert getattr(client, "_nodes_raw", None) is None


@pytest.mark.asyncio
async def test_namespace_ack_ignores_unexpected_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Namespace acknowledgements for other namespaces should be ignored."""

    client = _make_client(monkeypatch)

    class AckWS(QueueWebSocket):
        def __init__(self) -> None:
            super().__init__(
                [
                    SimpleNamespace(
                        type=aiohttp.WSMsgType.TEXT,
                        data="40/other",
                    )
                ]
            )

        async def send_str(self, payload: str) -> None:
            await super().send_str(payload)

    client._ws = AckWS()  # type: ignore[assignment]
    client._pending_dev_data = True

    emit_mock = AsyncMock()
    monkeypatch.setattr(client, "_emit_sio", emit_mock)
    statuses: list[str] = []
    monkeypatch.setattr(
        client, "_update_status", lambda status: statuses.append(status)
    )

    await _run_read_loop(client)

    assert client._pending_dev_data is True
    emit_mock.assert_not_awaited()
    assert not statuses


@pytest.mark.asyncio
async def test_disconnect_closes_websocket(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure _disconnect invokes the aiohttp close sequence."""

    client = _make_client(monkeypatch)
    close_called: dict[str, Any] = {}

    class ClosingWS:
        closed = False

        async def close(self, *, code: int, message: bytes) -> None:
            close_called["code"] = code
            close_called["message"] = message

    client._ws = ClosingWS()  # type: ignore[assignment]
    client._pending_dev_data = True
    await client._disconnect("reason")

    assert close_called["code"] == aiohttp.WSCloseCode.GOING_AWAY
    assert close_called["message"] == b"reason"
    assert client._ws is None
    assert client._pending_dev_data is False


@pytest.mark.asyncio
async def test_subscribe_feeds_stores_inventory_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Subscription state should persist the resolved inventory without nodes."""

    client = _make_client(monkeypatch)
    bucket = client.hass.data[ducaheat_ws.DOMAIN]["entry"]
    bucket.pop("inventory", None)

    raw_nodes = {"nodes": [{"type": "htr", "addr": "1"}]}
    inventory = Inventory(
        client.dev_id,
        build_node_inventory(raw_nodes),
    )
    client._inventory = inventory

    emissions: list[tuple[str, str]] = []
    monkeypatch.setattr(
        client,
        "_emit_sio",
        AsyncMock(side_effect=lambda evt, path: emissions.append((evt, path))),
    )

    count = await client._subscribe_feeds()

    assert count == 2
    assert {path for _evt, path in emissions} == {"/htr/1/samples", "/htr/1/status"}
    assert client._subscription_paths == {"/htr/1/samples", "/htr/1/status"}
    bucket = client.hass.data[ducaheat_ws.DOMAIN]["entry"]
    stored = bucket.get("inventory")
    assert stored is inventory


@pytest.mark.asyncio
async def test_subscribe_feeds_uses_inventory_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing inventory metadata should drive subscription target selection."""

    client = _make_client(monkeypatch)
    node_inventory = build_node_inventory([{"type": "htr", "addr": "7"}])
    inventory = Inventory(
        client.dev_id,
        node_inventory,
    )
    client._inventory = inventory

    emissions: list[tuple[str, str]] = []
    monkeypatch.setattr(
        client,
        "_emit_sio",
        AsyncMock(side_effect=lambda evt, path: emissions.append((evt, path))),
    )

    count = await client._subscribe_feeds()

    assert count == 2
    assert emissions == [
        ("subscribe", "/htr/7/samples"),
        ("subscribe", "/htr/7/status"),
    ]
    assert client._subscription_paths == {"/htr/7/samples", "/htr/7/status"}
    record = client.hass.data[ducaheat_ws.DOMAIN]["entry"]
    inventory = record.get("inventory")
    assert isinstance(inventory, Inventory)
    assert any(getattr(node, "addr", "") == "7" for node in inventory.nodes)
    assert "node_inventory" not in record


@pytest.mark.asyncio
async def test_subscribe_feeds_uses_record_inventory_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stored record inventory should be reused when the client cache is empty."""

    client = _make_client(monkeypatch)
    client._inventory = None
    record = client.hass.data[ducaheat_ws.DOMAIN]["entry"]
    node_inventory = build_node_inventory([{"type": "htr", "addr": "9"}])
    record["inventory"] = Inventory(
        client.dev_id,
        node_inventory,
    )

    emissions: list[str] = []

    async def _capture(event: str, path: str) -> None:
        emissions.append(path)

    monkeypatch.setattr(client, "_emit_sio", _capture)

    count = await client._subscribe_feeds()

    assert count == 2
    assert set(emissions) == {"/htr/9/samples", "/htr/9/status"}
    assert client._subscription_paths == {"/htr/9/samples", "/htr/9/status"}


@pytest.mark.asyncio
async def test_subscribe_feeds_handles_missing_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no subscription targets exist the helper should no-op."""

    client = _make_client(monkeypatch)
    payload = {"nodes": []}
    client.hass.data[ducaheat_ws.DOMAIN]["entry"]["inventory"] = Inventory(
        "device",
        build_node_inventory(payload),
    )
    emit_mock = AsyncMock()
    monkeypatch.setattr(client, "_emit_sio", emit_mock)

    count = await client._subscribe_feeds()

    assert count == 0
    emit_mock.assert_not_awaited()
    assert client._subscription_paths == set()


@pytest.mark.asyncio
async def test_subscribe_feeds_prefers_coordinator_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coordinator inventory should avoid resolver lookups."""

    client = _make_client(monkeypatch)
    client.hass.data[ducaheat_ws.DOMAIN]["entry"].pop("inventory", None)
    client.hass.data[ducaheat_ws.DOMAIN]["entry"].pop("nodes", None)
    inventory_nodes = build_node_inventory([{"type": "htr", "addr": "3"}])
    coordinator_inventory = Inventory(
        client.dev_id,
        inventory_nodes,
    )
    client._coordinator._inventory = coordinator_inventory

    emissions: list[str] = []

    async def _capture(event: str, path: str) -> None:
        emissions.append(path)

    monkeypatch.setattr(client, "_emit_sio", _capture)

    count = await client._subscribe_feeds()

    assert count == 2
    assert set(emissions) == {"/htr/3/samples", "/htr/3/status"}


@pytest.mark.asyncio
async def test_subscribe_feeds_handles_mapping_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mapping entries should be normalised to mutable state."""

    client = _make_client(monkeypatch)
    client.hass.data[ducaheat_ws.DOMAIN]["entry"].pop("inventory", None)
    raw_nodes = {"nodes": [{"addr": "8", "type": "htr"}]}
    inventory = Inventory(
        client.dev_id,
        build_node_inventory(raw_nodes),
    )
    mapping_record = MappingProxyType({"inventory": inventory})
    client.hass.data[ducaheat_ws.DOMAIN]["entry"] = mapping_record
    client._inventory = None

    emissions: list[str] = []

    async def _capture(event: str, path: str) -> None:
        emissions.append(path)

    monkeypatch.setattr(client, "_emit_sio", _capture)

    count = await client._subscribe_feeds()

    assert count == 2
    assert set(emissions) == {"/htr/8/samples", "/htr/8/status"}
    entry_record = client.hass.data[ducaheat_ws.DOMAIN]["entry"]
    assert isinstance(entry_record, dict)
    assert entry_record.get("inventory") is inventory


@pytest.mark.asyncio
async def test_subscribe_feeds_handles_missing_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """None records should be replaced with mutable containers."""

    client = _make_client(monkeypatch)
    client.hass.data[ducaheat_ws.DOMAIN]["entry"] = None

    raw_nodes = {"nodes": [{"addr": "6", "type": "htr"}]}
    inventory = Inventory(
        client.dev_id,
        build_node_inventory(raw_nodes),
    )
    client._inventory = inventory

    emissions: list[str] = []

    async def _capture(event: str, path: str) -> None:
        emissions.append(path)

    monkeypatch.setattr(client, "_emit_sio", _capture)

    count = await client._subscribe_feeds()

    assert count == 2
    assert set(emissions) == {"/htr/6/samples", "/htr/6/status"}
    entry_record = client.hass.data[ducaheat_ws.DOMAIN]["entry"]
    assert isinstance(entry_record, dict)
    assert entry_record.get("inventory") is inventory


@pytest.mark.asyncio
async def test_subscribe_feeds_logs_error_when_inventory_missing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Inventory resolution failures should be logged and abort subscriptions."""

    client = _make_client(monkeypatch)
    client.hass.data[ducaheat_ws.DOMAIN]["entry"].pop("inventory", None)
    client.hass.data[ducaheat_ws.DOMAIN]["entry"].pop("nodes", None)
    client._inventory = None
    client._coordinator._inventory = None
    caplog.set_level(logging.ERROR)

    emit_mock = AsyncMock()
    monkeypatch.setattr(client, "_emit_sio", emit_mock)

    count = await client._subscribe_feeds()

    assert count == 0
    emit_mock.assert_not_awaited()
    assert any("missing inventory" in message for message in caplog.messages)


@pytest.mark.asyncio
async def test_get_token_requires_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing Authorization header should raise a runtime error."""

    client = _make_client(monkeypatch)
    client._get_token = ducaheat_ws.DucaheatWSClient._get_token.__get__(  # type: ignore[method-assign]
        client, ducaheat_ws.DucaheatWSClient
    )
    monkeypatch.setattr(client._client, "authed_headers", AsyncMock(return_value={}))

    with pytest.raises(RuntimeError):
        await client._get_token()
