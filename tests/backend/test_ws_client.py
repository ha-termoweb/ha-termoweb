"""Tests for the shared websocket client and health tracking."""

from __future__ import annotations

import asyncio
import copy
import logging
import time
from types import MappingProxyType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
import pytest

from custom_components.termoweb.backend import (
    ducaheat_ws,
    termoweb_ws,
    ws_client as base_ws,
)
from custom_components.termoweb.backend.sanitize import (
    mask_identifier,
    redact_token_fragment,
)
from custom_components.termoweb.backend.ws_health import WsHealthTracker
from custom_components.termoweb.const import signal_ws_status
from custom_components.termoweb.domain.ids import NodeId, NodeType
from custom_components.termoweb.runtime import EntryRuntime
from tests.fakes.ws import DEV_ID, ENTRY_ID, DummyREST, make_inventory, make_runtime

_CLIENT_CLASSES = {
    "termoweb": termoweb_ws.TermoWebWSClient,
    "ducaheat": ducaheat_ws.DucaheatWSClient,
}
both_clients = pytest.mark.parametrize("brand", sorted(_CLIENT_CLASSES))


def _client(hass: HomeAssistant, brand: str, runtime: EntryRuntime) -> Any:
    """Return a websocket client of ``brand`` bound to ``runtime``."""
    return _CLIENT_CLASSES[brand](
        hass,
        entry_id=ENTRY_ID,
        dev_id=DEV_ID,
        api_client=DummyREST(),
        coordinator=runtime.coordinator,
        session=SimpleNamespace(),
        inventory=runtime.inventory,
    )


def _status_listener(hass: HomeAssistant) -> list[dict[str, Any]]:
    """Collect the ws-status dispatcher payloads of the test entry."""
    payloads: list[dict[str, Any]] = []

    @callback
    def _record(payload: dict[str, Any]) -> None:
        # A plain (non-callback) target would run in the executor, out of order.
        payloads.append(payload)

    async_dispatcher_connect(hass, signal_ws_status(ENTRY_ID), _record)
    return payloads


# ---------------------------------------------------------------------------
# Lifecycle and status bookkeeping
# ---------------------------------------------------------------------------


@both_clients
async def test_stop_clears_ws_state_across_restarts(
    hass: HomeAssistant, brand: str
) -> None:
    """Each start/stop cycle leaves no websocket state or tracker behind."""
    runtime = make_runtime(hass)
    client = _client(hass, brand, runtime)
    client._runner = lambda: asyncio.Event().wait()  # never connects

    for _ in range(2):
        client.start()
        client._ws_state_bucket()
        client._ws_health_tracker()
        assert set(runtime.ws_state) == {DEV_ID}
        assert set(runtime.ws_trackers) == {DEV_ID}

        await client.stop()
        assert runtime.ws_state == {}
        assert runtime.ws_trackers == {}
        assert client._task is None


@both_clients
async def test_start_reuses_a_live_task(hass: HomeAssistant, brand: str) -> None:
    """``start`` returns the running task instead of scheduling a second runner."""
    runtime = make_runtime(hass)
    client = _client(hass, brand, runtime)
    runs: list[None] = []

    async def _runner() -> None:
        runs.append(None)
        await asyncio.Event().wait()

    client._runner = _runner

    task = client.start()
    assert client.start() is task
    await asyncio.sleep(0)
    assert len(runs) == 1
    await client.stop()
    assert task.cancelled()


@both_clients
async def test_update_status_publishes_state_and_gateway_connection(
    hass: HomeAssistant, brand: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Status changes reach the ws-state bucket, the dispatcher and the store."""
    runtime = make_runtime(hass)
    client = _client(hass, brand, runtime)
    payloads = _status_listener(hass)
    client._stats.frames_total = 4
    client._stats.events_total = 2
    client._stats.last_event_ts = 75.0
    monkeypatch.setattr(base_ws.time, "time", lambda: 100.0)

    client._update_status("healthy")
    await hass.async_block_till_done()

    state = runtime.ws_state[DEV_ID]
    assert state["status"] == "healthy"
    assert state["healthy_since"] == 75.0  # falls back to the last event time
    assert state["last_event_at"] == 75.0
    assert state["frames_total"] == 4
    assert state["events_total"] == 2
    assert payloads[-1]["status"] == "healthy"
    assert payloads[-1]["health_changed"] is True
    connection = runtime.coordinator.domain_view.get_gateway_connection_state()
    assert connection.status == "healthy"
    assert connection.connected is True
    assert connection.healthy_since == 75.0

    payloads.clear()
    client._update_status("healthy")  # nothing changed: no dispatch
    await hass.async_block_till_done()
    assert payloads == []


async def test_ducaheat_resets_health_when_leaving_healthy(
    hass: HomeAssistant,
) -> None:
    """Ducaheat drops ``healthy_since`` on any non-healthy status."""
    runtime = make_runtime(hass)
    client = _client(hass, "ducaheat", runtime)

    client._update_status("healthy")
    assert runtime.ws_state[DEV_ID]["healthy_since"] is not None
    client._update_status("connected")
    assert runtime.ws_state[DEV_ID]["healthy_since"] is None
    assert runtime.ws_state[DEV_ID]["healthy_minutes"] == 0


@pytest.mark.parametrize(
    ("health_changed", "payload_changed"),
    [(False, False), (True, False), (False, True), (True, True)],
)
async def test_notify_ws_status_payload(
    hass: HomeAssistant, health_changed: bool, payload_changed: bool
) -> None:
    """The dispatcher payload carries metadata plus only the flags that are set."""
    runtime = make_runtime(hass)
    client = _client(hass, "ducaheat", runtime)
    payloads = _status_listener(hass)
    tracker = WsHealthTracker(DEV_ID)

    client._notify_ws_status(
        tracker,
        reason="unit-test",
        health_changed=health_changed,
        payload_changed=payload_changed,
    )
    await hass.async_block_till_done()

    expected = {
        "dev_id": DEV_ID,
        "status": tracker.status,
        "reason": "unit-test",
        "payload_stale": tracker.payload_stale,
    }
    if health_changed:
        expected["health_changed"] = True
    if payload_changed:
        expected["payload_changed"] = True
    assert payloads == [expected]


async def test_payload_heartbeat_and_refresh_track_staleness(
    hass: HomeAssistant,
) -> None:
    """Payload, heartbeat and refresh marks keep state, listeners and store aligned."""
    runtime = make_runtime(hass)
    client = _client(hass, "termoweb", runtime)
    payloads = _status_listener(hass)
    view = runtime.coordinator.domain_view

    client._mark_ws_payload(timestamp=100.0, stale_after=15.0, reason="payload")
    await hass.async_block_till_done()
    state = runtime.ws_state[DEV_ID]
    assert state["last_payload_at"] == 100.0
    assert state["last_heartbeat_at"] == 100.0
    assert state["payload_stale"] is False
    assert state["payload_stale_after"] == 15.0
    assert payloads[-1]["reason"] == "payload"
    assert payloads[-1]["payload_changed"] is True
    assert view.get_gateway_connection_state().last_payload_at == 100.0

    payloads.clear()
    client._mark_ws_heartbeat(timestamp=105.0, reason="beat")
    await hass.async_block_till_done()
    assert state["last_heartbeat_at"] == 105.0
    assert payloads == []  # still fresh: nothing to announce

    client._refresh_ws_payload_state(now=200.0, reason="refresh")
    await hass.async_block_till_done()
    assert state["payload_stale"] is True
    assert payloads[-1]["reason"] == "refresh"
    assert payloads[-1]["payload_stale"] is True
    assert view.get_gateway_connection_state().payload_stale is True


async def test_termoweb_event_marks_client_healthy(hass: HomeAssistant) -> None:
    """The first counted event marks the connection healthy at that time."""
    runtime = make_runtime(hass)
    client = _client(hass, "termoweb", runtime)

    client._mark_event(count_event=True)

    state = runtime.ws_state[DEV_ID]
    assert state["events_total"] == 1
    assert state["status"] == "healthy"
    assert state["last_event_at"] == client._healthy_since
    assert state["last_payload_at"] == client._healthy_since


async def test_termoweb_idle_restart_disconnects_once(hass: HomeAssistant) -> None:
    """An idle restart disconnects; it is not scheduled twice or while closing."""
    runtime = make_runtime(hass)
    client = _client(hass, "termoweb", runtime)
    client._disconnect = AsyncMock()

    client._schedule_idle_restart(idle_for=300.0, source="test")
    assert runtime.ws_state[DEV_ID]["idle_restart_pending"] is True
    view = runtime.coordinator.domain_view
    assert view.get_gateway_connection_state().idle_restart_pending is True
    first = client._idle_restart_task
    client._schedule_idle_restart(idle_for=300.0, source="again")
    assert client._idle_restart_task is first

    await first
    client._disconnect.assert_awaited_once_with(reason="idle restart")
    assert client._idle_restart_task is None
    assert runtime.ws_state[DEV_ID]["idle_restart_pending"] is False

    client._closing = True
    client._schedule_idle_restart(idle_for=300.0, source="closing")
    assert client._idle_restart_task is None


async def test_termoweb_payload_cancels_pending_idle_restart(
    hass: HomeAssistant,
) -> None:
    """Fresh activity cancels a scheduled idle restart."""
    runtime = make_runtime(hass)
    client = _client(hass, "termoweb", runtime)
    client._disconnect = AsyncMock()

    client._schedule_idle_restart(idle_for=120.0, source="test")
    task = client._idle_restart_task
    client._cancel_idle_restart()
    await asyncio.sleep(0)

    assert task.cancelled()
    client._disconnect.assert_not_awaited()
    assert runtime.ws_state[DEV_ID]["idle_restart_pending"] is False


async def test_termoweb_stop_cancels_idle_restart(hass: HomeAssistant) -> None:
    """Stopping cancels a pending idle restart and disconnects."""
    runtime = make_runtime(hass)
    client = _client(hass, "termoweb", runtime)
    client._disconnect = AsyncMock()

    client._schedule_idle_restart(idle_for=120.0, source="test")
    task = client._idle_restart_task
    await client.stop()
    await hass.async_block_till_done()

    assert task.cancelled()
    assert client._idle_restart_pending is False
    client._disconnect.assert_awaited_with(reason="client stop")
    assert runtime.ws_state == {}


async def test_termoweb_force_refresh_token(hass: HomeAssistant) -> None:
    """A forced refresh drops the cached REST token and fetches a new one."""
    runtime = make_runtime(hass)
    client = _client(hass, "termoweb", runtime)

    await client._force_refresh_token()

    assert client._client._access_token is None
    client._client._ensure_token.assert_awaited_once()


@pytest.mark.parametrize(
    ("api_base", "expected"),
    [("https://example/api/", "https://example/api"), ("", termoweb_ws.API_BASE)],
)
async def test_termoweb_api_base(
    hass: HomeAssistant, api_base: str, expected: str
) -> None:
    """The REST client's base URL wins; an empty one falls back to the default."""
    runtime = make_runtime(hass)
    client = _client(hass, "termoweb", runtime)
    client._client.api_base = api_base
    assert client._api_base() == expected


# ---------------------------------------------------------------------------
# Connection pacing
# ---------------------------------------------------------------------------


async def test_connection_rate_limiter_enforces_window() -> None:
    """The limiter spaces attempts and caps them per rolling window."""
    sleeps: list[float] = []
    now = 0.0

    async def _fake_sleep(delay: float) -> None:
        nonlocal now
        sleeps.append(delay)
        now += delay

    limiter = base_ws.ConnectionRateLimiter(
        min_interval=1.0,
        max_attempts=2,
        window_seconds=3.0,
        clock=lambda: now,
        sleeper=_fake_sleep,
    )

    for _ in range(3):
        await limiter.wait_for_slot()

    assert sleeps == [1.0, 2.0]


@both_clients
async def test_backoff_sequence_restarts_on_reset(
    hass: HomeAssistant, brand: str
) -> None:
    """Both clients share one backoff sequence that restarts on reset."""
    client = _client(hass, brand, make_runtime(hass))
    assert [client._next_backoff() for _ in range(6)] == [5, 10, 30, 120, 300, 300]
    client._reset_backoff()
    assert client._next_backoff() == 5


# ---------------------------------------------------------------------------
# Frame translation shared by both clients
# ---------------------------------------------------------------------------


@both_clients
async def test_brand_headers(hass: HomeAssistant, brand: str) -> None:
    """Brand headers carry the brand identity and an optional origin."""
    client = _client(hass, brand, make_runtime(hass))
    headers = client._brand_headers(origin="https://app.example")
    assert headers["User-Agent"]
    assert headers["X-Requested-With"]
    assert headers["Origin"] == "https://app.example"
    assert "Origin" not in client._brand_headers()


@both_clients
async def test_nodes_to_deltas(hass: HomeAssistant, brand: str) -> None:
    """Node payloads become deltas; unknown nodes and missing inventory are skipped."""
    runtime = make_runtime(hass)
    client = _client(hass, brand, runtime)
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

    deltas = client._nodes_to_deltas(nodes, inventory=runtime.inventory)

    assert len(deltas) == 1
    assert deltas[0].node_id == NodeId(NodeType.HEATER, "1")
    payload = deltas[0].payload
    assert payload["mode"] == "manual"
    assert payload["stemp"] == "18.0"
    assert payload["prog"] == {"0": 1}
    assert "unknown" not in payload
    assert "samples" not in payload
    unknown_node = {"htr": {"settings": {"2": {"mode": "auto"}}}}
    assert client._nodes_to_deltas(unknown_node, inventory=runtime.inventory) == []
    assert client._nodes_to_deltas(nodes, inventory=None) == []


@both_clients
async def test_apply_deltas_reaches_the_state_store(
    hass: HomeAssistant, brand: str
) -> None:
    """Deltas land in the coordinator's store; a failing handler is contained."""
    runtime = make_runtime(hass)
    client = _client(hass, brand, runtime)
    deltas = client._nodes_to_deltas(
        {"htr": {"settings": {"1": {"mode": "auto", "stemp": "21.5"}}}},
        inventory=runtime.inventory,
    )

    client._apply_deltas_to_store(deltas, replace=False)

    state = runtime.coordinator.domain_view.get_heater_state("htr", "1")
    assert state.mode == "auto"
    assert state.stemp == "21.5"

    client._coordinator = SimpleNamespace(
        handle_ws_deltas=MagicMock(side_effect=RuntimeError("boom"))
    )
    client._apply_deltas_to_store(deltas, replace=False)  # must not raise


@both_clients
async def test_translate_path_update(hass: HomeAssistant, brand: str) -> None:
    """Path frames map onto node sections; malformed frames are rejected."""
    client = _client(hass, brand, make_runtime(hass))
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


@pytest.mark.parametrize(
    ("section", "expected"),
    [
        (None, (None, None)),
        ("status", ("status", None)),
        ("advanced_setup", ("advanced", "advanced_setup")),
        ("setup", ("settings", "setup")),
        ("custom", ("settings", "custom")),
    ],
)
def test_resolve_ws_update_section(
    section: str | None, expected: tuple[str | None, str | None]
) -> None:
    """Websocket section names map onto store sections."""
    assert base_ws.resolve_ws_update_section(section) == expected


@both_clients
async def test_get_token(hass: HomeAssistant, brand: str) -> None:
    """The bearer token is reused from the REST client; a missing one raises."""
    client = _client(hass, brand, make_runtime(hass))
    client._client.authed_headers = AsyncMock(
        return_value={"Authorization": "Bearer newtoken"}
    )
    assert await client._get_token() == "newtoken"
    client._client.authed_headers = AsyncMock(return_value={})
    with pytest.raises(RuntimeError):
        await client._get_token()


@both_clients
async def test_normalise_nodes(hass: HomeAssistant, brand: str) -> None:
    """Normalisation delegates to the REST codec and tolerates codec failures."""
    client = _client(hass, brand, make_runtime(hass))
    nodes = {"htr": {"status": {"1": {}}}}

    assert client._normalise_nodes(nodes) == nodes
    client._client.normalise_ws_nodes = lambda n: MappingProxyType({"htr": {}})
    resolved = client._normalise_nodes(nodes)
    assert resolved == {"htr": {}}
    assert isinstance(resolved, dict)
    client._client.normalise_ws_nodes = lambda n: ["ok"]
    assert client._normalise_nodes(nodes) == ["ok"]

    def _raise(_nodes: Any) -> Any:
        raise RuntimeError

    client._client.normalise_ws_nodes = _raise
    assert client._normalise_nodes(nodes) == nodes


@both_clients
async def test_coerce_nodes_list(hass: HomeAssistant, brand: str) -> None:
    """List-shaped node snapshots become ``{type: {section: {addr: value}}}``."""
    client = _client(hass, brand, make_runtime(hass))
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
async def test_collect_sample_updates(hass: HomeAssistant, brand: str) -> None:
    """Sample extraction filters node types, addresses and malformed entries."""
    client = _client(hass, brand, make_runtime(hass))
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


# ---------------------------------------------------------------------------
# forward_ws_sample_updates
# ---------------------------------------------------------------------------

_NODES = (
    {"type": "htr", "addr": "1"},
    {"type": "pmo", "addr": "7"},
    {"type": "thm", "addr": "3"},
)


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        pytest.param(
            {"pmo": {"samples": {"7": {"power": 100}}, "lease_seconds": 90}},
            ("device", {"pmo": {"7": {"power": 100}}}, 90.0),
            id="power-monitor-with-lease",
        ),
        pytest.param(
            {"pmo": {"samples": {"7": {"power": 3}}, "lease_seconds": "bad"}},
            ("device", {"pmo": {"7": {"power": 3}}}, None),
            id="invalid-lease-ignored",
        ),
        pytest.param(
            {
                "htr": {"samples": {"1": {"temp": 23}}, "lease_seconds": 30},
                "pmo": {"samples": {"7": {"power": 1}}, "lease_seconds": 120},
                "acm": {"lease_seconds": -5},
            },
            ("device", {"htr": {"1": {"temp": 23}}, "pmo": {"7": {"power": 1}}}, 120.0),
            id="largest-lease-wins",
        ),
        pytest.param(
            {"htr": {"samples": {"lease_seconds": 15, "1": {"t": 1}}}},
            ("device", {"htr": {"1": {"t": 1}}}, None),
            id="pseudo-lease-address-dropped",
        ),
        pytest.param(
            {
                "htr": {"samples": {"1": {"t": 1}, "": {"t": 2}}, "lease_seconds": 15},
                "scalar": "ignored",
                "number": 5,
                "none": None,
                "list": ["bad"],
                None: {"1": {}},
            },
            ("device", {"htr": {"1": {"t": 1}}}, 15.0),
            id="malformed-sections-skipped",
        ),
        pytest.param(
            {"thm": {"samples": {"3": {"counter": 1}}}}, None, id="thermostat-skipped"
        ),
        pytest.param(
            {"acm": {"samples": {"2": {"counter": 1}}}},
            None,
            id="type-without-samples-in-inventory",
        ),
        pytest.param({"pmo": ["invalid"]}, None, id="nothing-valid"),
    ],
)
async def test_forward_ws_sample_updates(
    hass: HomeAssistant,
    updates: dict[Any, Any],
    expected: tuple[str, dict[str, Any], float | None] | None,
) -> None:
    """Samples are normalised per inventory and relayed with the largest lease."""
    handler = MagicMock()
    make_runtime(
        hass,
        make_inventory(_NODES),
        energy_coordinator=SimpleNamespace(handle_ws_samples=handler),
    )

    base_ws.forward_ws_sample_updates(hass, ENTRY_ID, DEV_ID, updates)

    if expected is None:
        handler.assert_not_called()
    else:
        dev_id, payload, lease = expected
        handler.assert_called_once_with(dev_id, payload, lease_seconds=lease)


async def test_forward_ws_sample_updates_without_a_receiver(
    hass: HomeAssistant,
) -> None:
    """No runtime, or an energy coordinator without a handler, is a no-op."""
    updates = {"htr": {"samples": {"1": {"temp": 1}}}}
    base_ws.forward_ws_sample_updates(hass, ENTRY_ID, DEV_ID, updates)

    runtime = make_runtime(hass, energy_coordinator=SimpleNamespace())
    base_ws.forward_ws_sample_updates(hass, ENTRY_ID, DEV_ID, updates)
    assert runtime.energy_coordinator == SimpleNamespace()


async def test_forward_ws_sample_updates_contains_handler_errors(
    hass: HomeAssistant, caplog: pytest.LogCaptureFixture
) -> None:
    """A failing energy handler is logged, never raised into the read loop."""
    handler = MagicMock(side_effect=RuntimeError("boom"))
    make_runtime(hass, energy_coordinator=SimpleNamespace(handle_ws_samples=handler))
    logger = logging.getLogger("test_forward_ws_samples")

    base_ws.forward_ws_sample_updates(
        hass,
        ENTRY_ID,
        DEV_ID,
        {"htr": {"samples": {"1": {"temp": 21}}}},
        logger=logger,
        log_prefix="tester",
    )

    handler.assert_called_once()
    assert any(
        record.name == logger.name
        and record.levelno == logging.ERROR
        and record.message == "tester: forwarding heater samples failed"
        for record in caplog.records
    )


# ---------------------------------------------------------------------------
# Log redaction helpers used by both clients
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "token", "identifier"),
    [
        ("   ", "", ""),
        ("xy", "***", "***"),
        ("abcd", "***", "***"),
        ("abcdefgh", "ab***gh", "ab...gh"),
        ("abcdefghijk", "abcd...hijk", "abcdef...hijk"),
    ],
)
def test_redaction_helpers(value: str, token: str, identifier: str) -> None:
    """Tokens and identifiers are masked by length."""
    assert redact_token_fragment(value) == token
    assert mask_identifier(value) == identifier


def test_mark_payload_and_heartbeat_updates() -> None:
    """Test payload updates refresh heartbeat timestamps and staleness."""

    tracker = WsHealthTracker("dev")
    tracker.set_payload_window(120)
    changed = tracker.mark_payload(timestamp=1_000.0, stale_after=120)
    assert changed is True
    assert tracker.last_payload_at == 1_000.0
    assert tracker.last_heartbeat_at == 1_000.0
    assert tracker.payload_stale is False

    changed = tracker.mark_heartbeat(timestamp=1_050.0)
    assert changed is False
    assert tracker.last_heartbeat_at == 1_050.0
    assert tracker.last_payload_at == 1_000.0
    assert tracker.payload_stale is False


def test_staleness_detection_and_refresh() -> None:
    """Test payload staleness detection transitions across the threshold."""

    tracker = WsHealthTracker("dev")
    tracker.mark_payload(timestamp=1_000.0, stale_after=30)
    assert tracker.payload_stale is False
    assert tracker.is_payload_stale(now=1_029.0) is False
    assert tracker.is_payload_stale(now=1_030.0) is True

    changed = tracker.refresh_payload_state(now=1_030.0)
    assert changed is True
    assert tracker.payload_stale is True

    changed = tracker.mark_payload(timestamp=1_040.0)
    assert changed is True
    assert tracker.payload_stale is False
    assert tracker.stale_deadline() == pytest.approx(1_070.0)


def test_stale_deadline_requires_positive_threshold() -> None:
    """Stale deadline should be None without a positive payload window."""

    tracker = WsHealthTracker("dev")
    tracker.mark_payload(timestamp=1_500.0)
    assert tracker.last_payload_at == 1_500.0
    # ``payload_stale_after`` is ``None`` until explicitly configured.
    assert tracker.payload_stale_after is None
    assert tracker.stale_deadline() is None


def test_update_status_resets_health_state() -> None:
    """Test status transitions update healthy timestamps and reset state."""

    tracker = WsHealthTracker("dev")
    status_changed, health_changed = tracker.update_status(
        "healthy", healthy_since=1_000.0, timestamp=1_000.0
    )
    assert status_changed is True
    assert health_changed is True
    assert tracker.healthy_since == 1_000.0
    assert tracker.healthy_minutes(now=1_120.0) == 2

    snapshot = tracker.snapshot(now=1_120.0)
    assert snapshot["status"] == "healthy"
    assert snapshot["healthy_minutes"] == 2

    status_changed, health_changed = tracker.update_status(
        "degraded", timestamp=1_130.0, reset_health=True
    )
    assert status_changed is True
    assert health_changed is True
    assert tracker.healthy_since is None
    assert tracker.healthy_minutes(now=1_200.0) == 0

    status_changed, health_changed = tracker.update_status(
        "degraded", timestamp=1_150.0
    )
    assert status_changed is False
    assert health_changed is False


def test_ws_health_tracker_payload_flow() -> None:
    """Exercise the happy-path lifecycle for payload freshness tracking."""

    tracker = WsHealthTracker("dev01")
    base = 1_000.0

    assert tracker.payload_stale is True
    assert tracker.set_payload_window(30.0) is False
    assert tracker.payload_stale_after == 30.0

    changed = tracker.mark_payload(timestamp=base)
    assert changed is True
    assert tracker.last_payload_at == base
    assert tracker.last_heartbeat_at == base
    assert tracker.payload_stale is False

    tracker.update_status("healthy", healthy_since=base - 120, timestamp=base - 60)
    assert tracker.healthy_minutes(now=base + 10.0) == 2

    assert tracker.mark_heartbeat(timestamp=base + 10.0) is False
    assert tracker.last_heartbeat_at == base + 10.0
    assert tracker.payload_stale is False

    assert tracker.refresh_payload_state(now=base + 40.0) is True
    assert tracker.payload_stale is True

    assert tracker.stale_deadline() == pytest.approx(base + 30.0)

    snapshot = tracker.snapshot(now=base + 40.0)
    assert snapshot == {
        "status": "healthy",
        "healthy_since": base - 120,
        "healthy_minutes": 2,
        "last_status_at": base - 60,
        "last_heartbeat_at": base + 10.0,
        "last_payload_at": base,
        "payload_stale": True,
        "payload_stale_after": 30.0,
    }


def test_ws_health_tracker_rejects_invalid_stale_after() -> None:
    """Ensure invalid staleness windows are ignored across setter paths."""

    tracker = WsHealthTracker("dev01")

    assert tracker.set_payload_window(None) is False
    assert tracker.payload_stale_after is None

    for invalid in (-5, 0, "bad-input"):
        assert tracker.set_payload_window(invalid) is False
        assert tracker.payload_stale_after is None

    base = time.time()
    assert tracker.mark_payload(timestamp=base, stale_after="noop") is True
    assert tracker.payload_stale_after is None

    assert tracker.set_payload_window(15.0) is False
    assert tracker.payload_stale_after == 15.0

    # Confirm reapplying the same positive value leaves the tracker unchanged.
    assert tracker.set_payload_window(15.0) is False
    assert tracker.payload_stale_after == 15.0

    assert tracker.set_payload_window("still-bad") is False
    assert tracker.payload_stale_after == 15.0

    assert tracker.mark_payload(timestamp=base + 1.0, stale_after=-3) is False
    assert tracker.payload_stale_after == 15.0


def test_ws_health_tracker_future_healthy_since() -> None:
    """Healthy minutes should not go negative when the transition is in the future."""

    tracker = WsHealthTracker("dev01")
    base = time.time()

    tracker.update_status("healthy", healthy_since=base + 60.0, timestamp=base)

    assert tracker.healthy_minutes(now=base) == 0
