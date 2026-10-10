"""StateCoordinator: REST polling, pending writes, WS deltas and patches."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from aiohttp import ClientError
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import UpdateFailed
from homeassistant.util import dt as dt_util
import pytest

from custom_components.termoweb import coordinator as coord_module
from custom_components.termoweb.backend.rest_client import (
    BackendAuthError,
    BackendRateLimitError,
)
from custom_components.termoweb.coordinator import (
    StateCoordinator,
    build_device_metadata,
)
from custom_components.termoweb.domain import NodeId, NodeSettingsDelta, NodeType
from custom_components.termoweb.domain.energy import EnergySnapshot
from custom_components.termoweb.domain.state import (
    AccumulatorState,
    HeaterState,
    PowerMonitorState,
    ThermostatState,
)
from tests_ha.fakes.coordinator import (
    DEV_ID,
    Clock,
    inventory,
    node_state,
    rest_client,
    state_coordinator,
)

RTC_MIDNIGHT = {"y": 2024, "n": 1, "d": 1, "h": 0, "m": 0, "s": 0}


def _listener(coord: StateCoordinator) -> MagicMock:
    """Subscribe a mock listener to ``coord`` and return it."""
    listener = MagicMock()
    coord.async_add_listener(listener)
    return listener


# --- construction and metadata ---------------------------------------------


async def test_coordinator_requires_an_inventory(hass: HomeAssistant) -> None:
    """The inventory is mandatory and immutable: anything else is a TypeError."""
    with pytest.raises(TypeError, match="Inventory instance"):
        StateCoordinator(hass, rest_client(), 30, DEV_ID, None, "not-an-inventory")


@pytest.mark.parametrize(
    ("device", "name", "model"),
    [
        ({"name": " Home ", "model": " TW100 "}, "Home", "TW100"),
        ({"name": "", "model": ""}, f"Device {DEV_ID}", None),
        ({"name": 1234}, "1234", None),
        (None, f"Device {DEV_ID}", None),
    ],
)
async def test_gateway_name_and_model_are_normalised(
    hass: HomeAssistant, device: dict | None, name: str, model: str | None
) -> None:
    """Names are trimmed and fall back to the device id; empty models are None."""
    metadata = build_device_metadata(DEV_ID, device)
    coord = StateCoordinator(hass, rest_client(), 30, DEV_ID, metadata, inventory({}))

    assert coord.device_metadata is metadata
    assert coord.gateway_name == name
    assert coord.gateway_model == model


async def test_poll_publishes_a_minimal_device_record(hass: HomeAssistant) -> None:
    """Coordinator data holds gateway facts only; node state lives in the store."""
    client = rest_client(get_node_settings=AsyncMock(return_value={"mode": "auto"}))
    coord = state_coordinator(
        hass, client, {"htr": ["1"]}, device={"name": " Home ", "model": "M"}
    )

    await coord.async_refresh()

    assert coord.data == {
        DEV_ID: {
            "dev_id": DEV_ID,
            "name": "Home",
            "model": "M",
            "connected": False,
            "backend": "termoweb",
        }
    }
    assert node_state(coord, "htr", "1") == {"mode": "auto"}


async def test_gateway_connection_updates_state_and_listeners(
    hass: HomeAssistant,
) -> None:
    """WS health updates land in the store and in the published record."""
    coord = state_coordinator(hass, rest_client(), {})
    listener = _listener(coord)

    coord.update_gateway_connection(
        status="connected",
        connected=True,
        last_event_at=12.0,
        healthy_since=10.0,
        healthy_minutes=2.0,
        last_payload_at=11.0,
        last_heartbeat_at=11.5,
        payload_stale=False,
        payload_stale_after=120.0,
        idle_restart_pending=False,
    )

    assert coord.gateway_connected is True
    assert coord.data[DEV_ID]["connected"] is True
    assert coord.domain_view.get_gateway_connection_state().status == "connected"
    listener.assert_called_once()


async def test_manual_update_debug_noise_is_suppressed(
    hass: HomeAssistant, caplog: pytest.LogCaptureFixture
) -> None:
    """HA's per-push 'Manually updated' debug line is filtered out (WS pushes)."""
    coord = state_coordinator(hass, rest_client(), {"htr": ["1"]})
    with caplog.at_level(logging.DEBUG, logger=coord_module.__name__):
        coord.update_gateway_connection(
            status="connected",
            connected=True,
            last_event_at=None,
            healthy_since=None,
            healthy_minutes=None,
            last_payload_at=None,
            last_heartbeat_at=None,
            payload_stale=None,
            payload_stale_after=None,
            idle_restart_pending=None,
        )
    assert "Manually updated" not in caplog.text


# --- REST polling ------------------------------------------------------------


async def test_poll_fetches_every_node_in_inventory_order(
    hass: HomeAssistant,
) -> None:
    """One settings GET per node; non-dict payloads are skipped, not stored."""
    client = rest_client(
        get_node_settings=AsyncMock(
            side_effect=[{"mode": "auto"}, "unexpected", {"mode": "charge"}]
        )
    )
    coord = state_coordinator(hass, client, {"htr": ["1", "3"], "acm": ["2"]})

    await coord.async_refresh()

    assert [c.args for c in client.get_node_settings.await_args_list] == [
        (DEV_ID, ("htr", "1")),
        (DEV_ID, ("htr", "3")),
        (DEV_ID, ("acm", "2")),
    ]
    assert node_state(coord, "htr", "1") == {"mode": "auto"}
    assert node_state(coord, "htr", "3") is None
    assert node_state(coord, "acm", "2") == {"mode": "charge"}


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (TimeoutError(), "API timeout"),
        (ClientError("boom"), "API error: boom"),
        (BackendAuthError("denied"), "API error: denied"),
    ],
)
async def test_poll_errors_raise_update_failed(
    hass: HomeAssistant, error: Exception, message: str
) -> None:
    """Transport and auth errors surface as UpdateFailed to manual refreshes."""
    client = rest_client(get_node_settings=AsyncMock(side_effect=error))
    coord = state_coordinator(hass, client, {"htr": ["1", "2"]})

    with pytest.raises(UpdateFailed, match=message):
        await coord.async_refresh()

    assert client.get_node_settings.await_count == 1
    assert coord.last_update_success is False


async def test_rate_limit_backs_off_exponentially_and_resets(
    hass: HomeAssistant,
) -> None:
    """429s double the poll interval up to an hour; success restores the base."""
    client = rest_client(
        get_node_settings=AsyncMock(side_effect=BackendRateLimitError("429"))
    )
    coord = state_coordinator(hass, client, {"htr": ["1", "2"]})

    for backoff in (60, 120, 240, 480, 960, 1920, 3600, 3600):
        with pytest.raises(UpdateFailed, match=f"backing off to {backoff}s"):
            await coord.async_refresh()
        assert coord.update_interval == timedelta(seconds=backoff)
    # The 429 stops the node loop: the second node is never asked.
    assert {c.args[1] for c in client.get_node_settings.await_args_list} == {
        ("htr", "1")
    }

    client.get_node_settings.side_effect = None
    client.get_node_settings.return_value = {"mode": "auto"}
    await coord.async_refresh()
    assert coord.update_interval == timedelta(seconds=30)


async def test_resume_polling_honours_a_pending_rate_limit_backoff(
    hass: HomeAssistant,
) -> None:
    """Resuming after WS suspension waits out a 429 backoff first."""
    client = rest_client(
        get_node_settings=AsyncMock(side_effect=BackendRateLimitError("429"))
    )
    coord = state_coordinator(hass, client, {"htr": ["1"]}, base_interval=60)
    with pytest.raises(UpdateFailed):
        await coord.async_refresh()
    coord.update_interval = None  # WS healthy: polling suspended

    coord.resume_polling(60)
    assert coord.update_interval == timedelta(seconds=120)

    with pytest.raises(UpdateFailed):
        await coord.async_refresh()
    coord.resume_polling(60)
    assert coord.update_interval == timedelta(seconds=240)


async def test_successful_poll_does_not_unsuspend_polling(
    hass: HomeAssistant,
) -> None:
    """Clearing a backoff must not re-enable polling that WS suspended."""
    client = rest_client(
        get_node_settings=AsyncMock(side_effect=BackendRateLimitError("429"))
    )
    coord = state_coordinator(hass, client, {"htr": ["1"]})
    with pytest.raises(UpdateFailed):
        await coord.async_refresh()

    coord.update_interval = None
    client.get_node_settings.side_effect = None
    client.get_node_settings.return_value = {"mode": "auto"}
    await coord.async_refresh()

    assert coord.update_interval is None
    coord.resume_polling(30)
    assert coord.update_interval == timedelta(seconds=30)  # backoff was cleared


# --- accumulator boost metadata ----------------------------------------------


async def test_poll_derives_accumulator_boost_end_from_device_clock(
    hass: HomeAssistant,
) -> None:
    """Boost day/minute fields resolve against the gateway RTC, fetched once."""
    client = rest_client(
        get_node_settings=AsyncMock(
            return_value={"mode": "boost", "boost_end_day": 1, "boost_end_min": 90}
        ),
        get_rtc_time=AsyncMock(return_value=RTC_MIDNIGHT),
    )
    coord = state_coordinator(hass, client, {"acm": ["1", "2"]})

    await coord.async_refresh()

    tz = dt_util.now().tzinfo
    for addr in ("1", "2"):
        state = node_state(coord, "acm", addr)
        assert state["boost_end_datetime"] == datetime(2024, 1, 1, 1, 30, tzinfo=tz)
        assert state["boost_minutes_delta"] == 90
    client.get_rtc_time.assert_awaited_once_with(DEV_ID)
    # The device clock reference keeps answering without another RTC fetch.
    end, minutes = coord.resolve_boost_end(1, 120)
    assert minutes is not None and 119 <= minutes <= 120
    assert end == datetime(2024, 1, 1, 2, 0, tzinfo=tz)


@pytest.mark.parametrize(
    ("day", "minute", "expected_minutes"),
    [(None, 10, None), (-5, 10, None), (2, 60, 1500), (1, 90, 90)],
)
async def test_resolve_boost_end_validates_day_and_minute(
    hass: HomeAssistant, day: Any, minute: Any, expected_minutes: int | None
) -> None:
    """Boost end day/minute resolve to an end time and minutes left, or nothing."""
    coord = state_coordinator(hass, rest_client(), {"acm": ["1"]})
    now = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)

    end, minutes = coord.resolve_boost_end(day, minute, now=now)

    assert minutes == expected_minutes
    if expected_minutes is None:
        assert end is None
    else:
        assert end == now + timedelta(minutes=expected_minutes)


@pytest.mark.parametrize(
    "rtc",
    [
        AsyncMock(side_effect=ClientError("boom")),
        AsyncMock(side_effect=BackendRateLimitError("429")),
        AsyncMock(return_value={"y": 2024, "n": 13, "d": 1}),  # invalid month
        AsyncMock(return_value={}),
        AsyncMock(return_value=None),
    ],
)
async def test_refresh_without_device_clock_falls_back_to_local_time(
    hass: HomeAssistant, rtc: AsyncMock
) -> None:
    """An unusable RTC never blocks a refresh; boost end uses HA's clock."""
    client = rest_client(
        get_node_settings=AsyncMock(
            return_value={"mode": "boost", "boost_end_day": 1, "boost_end_min": 30}
        ),
        get_rtc_time=rtc,
    )
    coord = state_coordinator(hass, client, {"acm": ["1"]})

    await coord.async_refresh_heater(("acm", "1"))

    state = node_state(coord, "acm", "1")
    assert state["mode"] == "boost"
    assert state["boost_end_datetime"] > dt_util.now()
    assert state["boost_minutes_delta"] > 0
    rtc.assert_awaited_once()

    # No device clock reference was cached: the next refresh asks again.
    await coord.async_refresh_heater(("acm", "1"))
    assert rtc.await_count == 2


async def test_refresh_skips_rtc_when_accumulator_reports_no_boost_end(
    hass: HomeAssistant,
) -> None:
    """Without boost day/minute fields there is nothing to resolve: no RTC GET."""
    client = rest_client(get_node_settings=AsyncMock(return_value={"mode": "auto"}))
    coord = state_coordinator(hass, client, {"acm": ["1"]})

    await coord.async_refresh_heater(("acm", "1"))

    assert node_state(coord, "acm", "1") == {"mode": "auto"}
    client.get_rtc_time.assert_not_awaited()


# --- single-node refresh -----------------------------------------------------


async def test_refresh_heater_stores_one_node_and_notifies(
    hass: HomeAssistant,
) -> None:
    """A single-node refresh replaces that node's state and pushes listeners."""
    client = rest_client(get_node_settings=AsyncMock(return_value={"mode": "eco"}))
    coord = state_coordinator(hass, client, {"htr": ["1"], "acm": ["2"]})
    coord.handle_ws_deltas(
        DEV_ID, [NodeSettingsDelta(NodeId(NodeType.HEATER, "1"), {"mode": "manual"})]
    )
    listener = _listener(coord)

    await coord.async_refresh_heater((" acm ", "2"))

    client.get_node_settings.assert_awaited_once_with(DEV_ID, ("acm", "2"))
    assert node_state(coord, "acm", "2") == {"mode": "eco"}
    assert node_state(coord, "htr", "1") == {"mode": "manual"}
    listener.assert_called_once()


@pytest.mark.parametrize("node", [("htr", ""), ("", "1")])
async def test_refresh_heater_without_type_or_address_does_nothing(
    hass: HomeAssistant, node: tuple[str, str], caplog: pytest.LogCaptureFixture
) -> None:
    """A refresh needs both a node type and an address."""
    client = rest_client()
    coord = state_coordinator(hass, client, {"htr": ["1"]})

    await coord.async_refresh_heater(node)

    client.get_node_settings.assert_not_awaited()
    assert "without a node type and address" in caplog.text


@pytest.mark.parametrize(
    ("result", "log"),
    [
        ("not-a-dict", None),
        (TimeoutError("slow"), "Timeout refreshing heater settings"),
        (ClientError("boom"), "Failed to refresh heater settings"),
        (BackendAuthError("denied"), "Failed to refresh heater settings"),
    ],
)
async def test_refresh_heater_failures_keep_state_and_warn(
    hass: HomeAssistant,
    result: Any,
    log: str | None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failed single-node refresh keeps the stored state and never raises."""
    side_effect = result if isinstance(result, Exception) else None
    client = rest_client(
        get_node_settings=AsyncMock(return_value=result, side_effect=side_effect)
    )
    coord = state_coordinator(hass, client, {"htr": ["1"]})
    coord.handle_ws_deltas(
        DEV_ID, [NodeSettingsDelta(NodeId(NodeType.HEATER, "1"), {"mode": "manual"})]
    )
    listener = _listener(coord)

    await coord.async_refresh_heater(("htr", "1"))

    assert node_state(coord, "htr", "1") == {"mode": "manual"}
    listener.assert_not_called()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    if log is None:
        assert warnings == []
    else:
        assert log in caplog.text


# --- pending writes ----------------------------------------------------------


@pytest.mark.parametrize("refresh", ["poll", "single"])
async def test_pending_write_defers_stale_reads_until_confirmed(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch, refresh: str
) -> None:
    """After a write, reads disagreeing with it are not merged until they match."""
    Clock(100.0).install(monkeypatch)
    client = rest_client(
        get_node_settings=AsyncMock(return_value={"mode": "auto", "stemp": "20.0"})
    )
    coord = state_coordinator(hass, client, {"htr": ["1"]})
    coord.handle_ws_deltas(
        DEV_ID,
        [NodeSettingsDelta(NodeId(NodeType.HEATER, "1"), {"mode": "manual"})],
    )
    coord.register_pending_setting(" htr ", " 1 ", mode="Manual", stemp=21.0, ttl=60)

    async def _read() -> None:
        if refresh == "poll":
            await coord.async_refresh()
        else:
            await coord.async_refresh_heater(("htr", "1"))

    await _read()
    assert node_state(coord, "htr", "1") == {"mode": "manual"}  # stale read deferred

    # Case-insensitive mode and a setpoint within tolerance confirm the write.
    client.get_node_settings.return_value = {"mode": "MANUAL", "stemp": "21.04"}
    await _read()
    assert node_state(coord, "htr", "1") == {"mode": "MANUAL", "stemp": "21.04"}

    # Confirmed: the pending entry is gone, so later reads merge immediately.
    client.get_node_settings.return_value = {"mode": "auto", "stemp": "20.0"}
    await _read()
    assert node_state(coord, "htr", "1") == {"mode": "auto", "stemp": "20.0"}


@pytest.mark.parametrize(
    ("pending", "payload", "merged"),
    [
        ({"mode": "auto", "stemp": None}, {"mode": "auto"}, True),
        ({"mode": None, "stemp": 21.0}, {"mode": "eco", "stemp": "21.0"}, True),
        ({"mode": None, "stemp": 21.0}, {"mode": "eco"}, False),  # stemp missing
        ({"mode": "auto", "stemp": 18.0}, {"mode": "eco", "stemp": "16.0"}, False),
    ],
)
async def test_pending_write_match_rules(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    pending: dict[str, Any],
    payload: dict[str, Any],
    merged: bool,
) -> None:
    """Only the fields that were written must match; a missing setpoint does not."""
    Clock(100.0).install(monkeypatch)
    client = rest_client(get_node_settings=AsyncMock(return_value=payload))
    coord = state_coordinator(hass, client, {"htr": ["1"]})
    coord.register_pending_setting("htr", "1", ttl=60, **pending)

    await coord.async_refresh()

    assert (node_state(coord, "htr", "1") == payload) is merged


async def test_pending_write_expires_after_its_ttl(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unconfirmed write stops masking reads once its TTL has passed."""
    clock = Clock(100.0).install(monkeypatch)
    client = rest_client(get_node_settings=AsyncMock(return_value={"mode": "eco"}))
    coord = state_coordinator(hass, client, {"htr": ["1"]})
    coord.register_pending_setting("htr", "1", mode="auto", stemp=None, ttl=5)

    clock.now = 104.0
    await coord.async_refresh()
    assert node_state(coord, "htr", "1") is None

    clock.now = 105.0
    await coord.async_refresh()
    assert node_state(coord, "htr", "1") == {"mode": "eco"}


async def test_pending_write_for_invalid_node_is_ignored(
    hass: HomeAssistant,
) -> None:
    """A pending write without a node type or address never defers reads."""
    client = rest_client(get_node_settings=AsyncMock(return_value={"mode": "eco"}))
    coord = state_coordinator(hass, client, {"htr": ["1"]})
    coord.register_pending_setting("", "", mode="auto", stemp=21.0)

    await coord.async_refresh()

    assert node_state(coord, "htr", "1") == {"mode": "eco"}


# --- websocket deltas --------------------------------------------------------


async def test_ws_deltas_replace_or_merge_node_state(hass: HomeAssistant) -> None:
    """Full snapshots replace a node's state; deltas merge into it."""
    coord = state_coordinator(hass, rest_client(), {"htr": ["1"]})
    listener = _listener(coord)
    node = NodeId(NodeType.HEATER, "1")

    coord.handle_ws_deltas(
        DEV_ID,
        [NodeSettingsDelta(node, {"mode": "auto", "status": {"stemp": "21.0"}})],
        replace=True,
    )
    assert node_state(coord, "htr", "1") == {"mode": "auto", "stemp": "21.0"}

    coord.handle_ws_deltas(DEV_ID, [NodeSettingsDelta(node, {"stemp": "19.0"})])
    assert node_state(coord, "htr", "1") == {"mode": "auto", "stemp": "19.0"}
    assert "settings" not in coord.data[DEV_ID]
    assert listener.call_count == 2


@pytest.mark.parametrize(
    ("dev_id", "deltas"),
    [
        (
            "fedcba9876543210",
            [NodeSettingsDelta(NodeId(NodeType.HEATER, "1"), {"mode": "manual"})],
        ),
        (DEV_ID, ["not-a-delta"]),
        (DEV_ID, []),
    ],
)
async def test_ws_deltas_for_another_gateway_or_invalid_are_ignored(
    hass: HomeAssistant, dev_id: str, deltas: list
) -> None:
    """Foreign or malformed deltas change nothing and notify nobody."""
    coord = state_coordinator(hass, rest_client(), {"htr": ["1"]})
    listener = _listener(coord)

    coord.handle_ws_deltas(dev_id, deltas)

    assert node_state(coord, "htr", "1") is None
    listener.assert_not_called()


# --- optimistic entity patches -----------------------------------------------


@pytest.mark.parametrize(
    ("node_type", "state_cls"),
    [
        ("htr", HeaterState),
        ("acm", AccumulatorState),
        ("thm", ThermostatState),
        ("pmo", PowerMonitorState),
    ],
)
async def test_entity_patch_creates_typed_state_for_new_node(
    hass: HomeAssistant, node_type: str, state_cls: type
) -> None:
    """Patching a node without state starts from that node type's state class."""
    coord = state_coordinator(hass, rest_client(), {node_type: ["1"]})
    seen: list[Any] = []

    assert coord.apply_entity_patch(node_type, "1", seen.append) is True

    assert type(seen[0]) is state_cls
    assert coord.domain_view.get_heater_state(node_type, "1") is not None


async def test_entity_patch_mutates_a_copy_and_notifies(hass: HomeAssistant) -> None:
    """A patch replaces the stored state with a mutated copy and pushes listeners."""
    coord = state_coordinator(hass, rest_client(), {"htr": ["1"], "acm": ["1"]})
    coord.handle_ws_deltas(
        DEV_ID,
        [NodeSettingsDelta(NodeId(NodeType.HEATER, "1"), {"mode": "manual"})],
        replace=True,
    )
    before = coord.domain_view.get_heater_state("htr", "1")
    listener = _listener(coord)

    assert coord.apply_entity_patch("htr", "1", lambda s: setattr(s, "mode", "auto"))

    assert node_state(coord, "htr", "1") == {"mode": "auto"}
    assert before.mode == "manual"  # readers never see a half-mutated state
    # Addresses are unique per gateway: the patch never fans out to the acm.
    assert coord.domain_view.get_heater_state("acm", "1") is None
    listener.assert_called_once()


async def test_entity_patch_clears_derived_boost_end_when_boost_changes(
    hass: HomeAssistant,
) -> None:
    """Changing boost fields invalidates the derived end time until next read."""
    client = rest_client(
        get_node_settings=AsyncMock(
            return_value={"mode": "boost", "boost_end_day": 1, "boost_end_min": 90}
        ),
        get_rtc_time=AsyncMock(return_value=RTC_MIDNIGHT),
    )
    coord = state_coordinator(hass, client, {"acm": ["1"]})
    await coord.async_refresh_heater(("acm", "1"))
    assert node_state(coord, "acm", "1")["boost_minutes_delta"] == 90

    coord.apply_entity_patch("acm", "1", lambda s: setattr(s, "mode", "auto"))
    assert node_state(coord, "acm", "1")["boost_minutes_delta"] == 90

    coord.apply_entity_patch("acm", "1", lambda s: setattr(s, "boost_end_min", 120))
    state = node_state(coord, "acm", "1")
    assert "boost_end_datetime" not in state
    assert "boost_minutes_delta" not in state


async def test_entity_patch_rejections(hass: HomeAssistant) -> None:
    """Unknown nodes and failing mutators leave the store untouched."""
    coord = state_coordinator(hass, rest_client(), {"htr": ["1"]})
    coord.apply_entity_patch("htr", "1", lambda s: setattr(s, "mode", "auto"))
    listener = _listener(coord)

    def _bad(state: Any) -> None:
        state["mode"] = "heat"

    assert coord.apply_entity_patch("", "1", lambda s: None) is False
    assert coord.apply_entity_patch("htr", "2", lambda s: None) is False
    assert coord.apply_entity_patch("htr", "1", _bad) is False

    assert node_state(coord, "htr", "1") == {"mode": "auto"}
    listener.assert_not_called()


async def test_entity_patch_propagates_cancellation(hass: HomeAssistant) -> None:
    """A cancelled mutator is re-raised, not swallowed as a failed patch."""
    coord = state_coordinator(hass, rest_client(), {"htr": ["1"]})

    def _cancel(_state: Any) -> None:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        coord.apply_entity_patch("htr", "1", _cancel)


# --- energy snapshots --------------------------------------------------------


@pytest.mark.parametrize(
    "snapshot",
    [
        "not-a-snapshot",
        None,
        EnergySnapshot(dev_id="other", metrics={}, updated_at=1.0, ws_deadline=None),
    ],
)
async def test_foreign_or_invalid_energy_snapshots_are_ignored(
    hass: HomeAssistant, snapshot: Any
) -> None:
    """Only this gateway's energy snapshots reach the store and listeners."""
    coord = state_coordinator(hass, rest_client(), {"htr": ["1"]})
    listener = _listener(coord)

    coord.apply_energy_snapshot(snapshot)

    assert coord.domain_view.get_energy_snapshot() is None
    listener.assert_not_called()


async def test_energy_snapshot_does_not_mark_a_failed_poll_successful(
    hass: HomeAssistant,
) -> None:
    """Energy pushes refresh listeners but leave the state poll result alone."""
    client = rest_client(get_node_settings=AsyncMock(side_effect=TimeoutError))
    coord = state_coordinator(hass, client, {"htr": ["1"]})
    with pytest.raises(UpdateFailed):
        await coord.async_refresh()
    listener = _listener(coord)

    coord.apply_energy_snapshot(
        EnergySnapshot(dev_id=DEV_ID, metrics={}, updated_at=1.0, ws_deadline=None)
    )

    assert coord.domain_view.get_energy_snapshot() is not None
    assert coord.last_update_success is False
    listener.assert_called_once()


async def test_rtc_reference_uses_home_assistant_timezone(
    hass: HomeAssistant,
) -> None:
    """The device clock is read in Home Assistant's configured timezone."""
    await hass.config.async_set_time_zone("Europe/Athens")
    client = rest_client(
        get_node_settings=AsyncMock(
            return_value={"boost_end_day": 1, "boost_end_min": 30}
        ),
        get_rtc_time=AsyncMock(return_value=RTC_MIDNIGHT),
    )
    coord = state_coordinator(hass, client, {"acm": ["1"]})

    await coord.async_refresh_heater(("acm", "1"))

    end = node_state(coord, "acm", "1")["boost_end_datetime"]
    assert end.utcoffset() == timedelta(hours=2)
    assert end.astimezone(UTC) == datetime(2023, 12, 31, 22, 30, tzinfo=UTC)
