from __future__ import annotations

import datetime as dt

from aiohttp import ClientError
import logging
from typing import Any, Callable, Iterable, Mapping
from unittest.mock import AsyncMock

import pytest

from conftest import build_device_metadata_payload
from homeassistant.core import HomeAssistant

from custom_components.termoweb import (
    boost as boost_module,
    coordinator as coord_module,
    inventory as inventory_module,
)
from custom_components.termoweb.domain import (
    NodeId,
    NodeSettingsDelta,
    NodeType,
    state_to_dict,
)
from custom_components.termoweb.inventory import AccumulatorNode, HeaterNode


class ExplodingStr:
    """Helper that raises when stringified to test defensive paths."""

    def __str__(self) -> str:
        raise RuntimeError("boom")


def _state_payload(
    coordinator: coord_module.StateCoordinator, node_type: str, addr: str
) -> dict[str, Any] | None:
    """Return the stored state payload for ``(node_type, addr)``."""

    view = getattr(coordinator, "domain_view", None)
    if view is None:
        return None
    state = view.get_heater_state(node_type, addr)
    return state_to_dict(state) if state is not None else None


def test_coerce_int_variants() -> None:
    """``coerce_int`` should normalise primitives and guard against errors."""

    assert boost_module.coerce_int(None) is None
    assert boost_module.coerce_int(True) == 1
    assert boost_module.coerce_int(False) == 0
    assert boost_module.coerce_int(5.7) == 5
    assert boost_module.coerce_int(float("inf")) is None
    assert boost_module.coerce_int(ExplodingStr()) is None
    assert boost_module.coerce_int("   ") is None
    assert boost_module.coerce_int(" 7.2 ") == 7


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (True, True),
        (False, False),
        (None, None),
        ("Yes", True),
        ("off", False),
        (1, True),
        (0, False),
        (2, None),
        (ExplodingStr(), None),
    ),
)
def test_coerce_boost_bool_variants(value: Any, expected: bool | None) -> None:
    """``coerce_boost_bool`` should normalise truthy and falsey values."""

    assert boost_module.coerce_boost_bool(value) is expected


def test_resolve_boost_end_from_fields_variants(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """Boost end resolver should validate ranges and return offsets."""

    base_now = dt.datetime(2024, 1, 1, 0, 0, tzinfo=dt.timezone.utc)

    dt_value, minutes = boost_module.resolve_boost_end_from_fields(None, 10)
    assert dt_value is None and minutes is None

    dt_value, minutes = boost_module.resolve_boost_end_from_fields(-5, 10, now=base_now)
    assert dt_value is None and minutes is None

    dt_value, minutes = boost_module.resolve_boost_end_from_fields(2, 60, now=base_now)
    assert isinstance(dt_value, dt.datetime)
    assert minutes == 1500

    hass = HomeAssistant()
    inventory_payload = {
        "nodes": [{"type": "acm", "addr": "1"}, {"type": "acm", "addr": "2"}]
    }
    inventory = inventory_builder("dev", inventory_payload)
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev"),
        nodes=None,
        inventory=inventory,
    )

    derived_dt, derived_minutes = coordinator.resolve_boost_end(2, 60, now=base_now)
    assert isinstance(derived_dt, dt.datetime)
    assert derived_minutes == 1500


def test_rtc_payload_to_datetime(monkeypatch: pytest.MonkeyPatch) -> None:
    """RTC payload helper should construct timezone-aware datetimes."""

    base_now = dt.datetime(2024, 5, 1, 12, 0, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(coord_module.dt_util, "now", lambda: base_now)

    payload = {"y": 2024, "n": 5, "d": 1, "h": 13, "m": 15, "s": 30}
    dt_value = coord_module.StateCoordinator._rtc_payload_to_datetime(payload)
    assert dt_value == dt.datetime(2024, 5, 1, 13, 15, 30, tzinfo=dt.timezone.utc)

    assert coord_module.StateCoordinator._rtc_payload_to_datetime(None) is None
    invalid_month = {"y": 2024, "n": 13, "d": 1, "h": 0, "m": 0, "s": 0}
    assert coord_module.StateCoordinator._rtc_payload_to_datetime(invalid_month) is None
    assert coord_module.StateCoordinator._rtc_payload_to_datetime({}) is None


def test_coordinator_updates_gateway_connection_state(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """Gateway connection updates should flow into coordinator data."""

    hass = HomeAssistant()
    inventory = inventory_builder("dev", {"nodes": []})
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev"),
        nodes=None,
        inventory=inventory,
    )

    coordinator.update_gateway_connection(
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

    assert coordinator.gateway_connected is True
    record = coordinator.data["dev"]
    assert record["connected"] is True
    connection_state = coordinator.domain_view.get_gateway_connection_state()
    assert connection_state.status == "connected"


@pytest.mark.asyncio
async def test_async_fetch_rtc_datetime_updates_reference(
    monkeypatch: pytest.MonkeyPatch,
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """Fetching RTC time should update the cached reference."""

    hass = HomeAssistant()
    client = AsyncMock()
    base_now = dt.datetime(2024, 5, 1, 12, 0, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(coord_module.dt_util, "now", lambda: base_now)

    client.get_rtc_time = AsyncMock(
        return_value={"y": 2024, "n": 5, "d": 1, "h": 12, "m": 5, "s": 0}
    )

    inventory_payload = {
        "nodes": [{"type": "acm", "addr": "1"}, {"type": "acm", "addr": "2"}]
    }
    inventory = inventory_builder("dev", inventory_payload)
    coordinator = coord_module.StateCoordinator(
        hass,
        client=client,
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev"),
        nodes=None,
        inventory=inventory,
    )

    result = await coordinator._async_fetch_rtc_datetime()
    assert result == dt.datetime(2024, 5, 1, 12, 5, 0, tzinfo=dt.timezone.utc)
    assert coordinator._device_now_estimate() is not None


@pytest.mark.asyncio
async def test_async_fetch_rtc_datetime_handles_error(
    monkeypatch: pytest.MonkeyPatch,
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """RTC fetch helper should swallow client errors and keep reference unset."""

    hass = HomeAssistant()
    client = AsyncMock()
    base_now = dt.datetime(2024, 5, 1, 12, 0, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(coord_module.dt_util, "now", lambda: base_now)

    client.get_rtc_time = AsyncMock(side_effect=ClientError("boom"))

    inventory = inventory_builder("dev", {})
    coordinator = coord_module.StateCoordinator(
        hass,
        client=client,
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev"),
        nodes=None,
        inventory=inventory,
    )

    result = await coordinator._async_fetch_rtc_datetime()
    assert result is None
    assert coordinator._device_now_estimate() is None


def test_boost_helpers_guard_against_invalid_sections(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """Derived metadata helpers should ignore unsupported payload shapes."""

    hass = HomeAssistant()
    inventory = inventory_builder("dev", {})
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev"),
        nodes=None,
        inventory=inventory,
    )

    calls: list[tuple[Mapping[str, Any], datetime | None]] = []

    def _record(payload: Mapping[str, Any], *, now: datetime | None) -> None:
        calls.append((payload, now))

    coordinator._apply_accumulator_boost_metadata = _record  # type: ignore[assignment]
    coordinator._apply_boost_metadata_for_settings(None, now=None)
    coordinator._apply_boost_metadata_for_settings({"1": []}, now=None)

    assert calls == []
    assert coord_module.StateCoordinator._requires_boost_resolution(None) is False


def test_apply_accumulator_boost_metadata_updates_payload(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """Accumulator boost helper should add and remove derived keys."""

    hass = HomeAssistant()
    inventory = inventory_builder("dev", {})
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev"),
        nodes=None,
        inventory=inventory,
    )

    payload: dict[str, Any] = {"boost_end_day": 1, "boost_end_min": 90}
    now = dt.datetime(2024, 1, 1, 0, 0, tzinfo=dt.timezone.utc)
    coordinator._apply_accumulator_boost_metadata(payload, now=now)

    assert payload["boost_end_datetime"] == dt.datetime(
        2024, 1, 1, 1, 30, tzinfo=dt.timezone.utc
    )
    assert payload["boost_minutes_delta"] == 90

    payload.clear()
    coordinator._apply_accumulator_boost_metadata(payload, now=now)

    assert "boost_end_datetime" not in payload
    assert "boost_minutes_delta" not in payload


@pytest.mark.asyncio
async def test_async_update_data_adds_boost_metadata(
    monkeypatch: pytest.MonkeyPatch,
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """Accumulator settings should expose derived boost metadata."""

    hass = HomeAssistant()
    client = AsyncMock()
    base_now = dt.datetime(2024, 1, 1, 0, 0, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(coord_module.dt_util, "now", lambda: base_now)

    nodes_payload = {"nodes": [{"addr": "1", "type": "acm"}]}
    nodes_list = [AccumulatorNode(name="Accumulator 1", addr="1")]
    inventory = inventory_builder("dev", nodes_payload, nodes_list)
    coordinator = coord_module.StateCoordinator(
        hass,
        client=client,
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev"),
        nodes=nodes_payload,
        inventory=inventory,
    )

    client.get_node_settings = AsyncMock(
        return_value={"mode": "boost", "boost_end_day": 1, "boost_end_min": 90}
    )
    client.get_rtc_time = AsyncMock(
        return_value={"y": 2024, "n": 1, "d": 1, "h": 0, "m": 0, "s": 0}
    )

    await coordinator._async_update_data()
    settings = _state_payload(coordinator, "acm", "1") or {}
    derived_dt = settings.get("boost_end_datetime")
    derived_minutes = settings.get("boost_minutes_delta")

    assert isinstance(derived_dt, dt.datetime)
    assert derived_dt == dt.datetime(2024, 1, 1, 1, 30, tzinfo=dt.timezone.utc)
    assert derived_minutes == 90
    client.get_rtc_time.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_update_data_skips_without_inventory(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """Coordinator should skip polling when inventory metadata is missing."""

    hass = HomeAssistant()
    client = AsyncMock()
    inventory = inventory_builder("dev", {})
    coord = coord_module.StateCoordinator(
        hass,
        client=client,
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev"),
        nodes=None,
        inventory=inventory,
    )

    coord._inventory = None
    result = await coord._async_update_data()

    assert coord._inventory is None
    assert result == {}


@pytest.mark.asyncio
async def test_async_update_data_requires_inventory(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """Coordinator should not rebuild inventory when the cache is cleared."""

    hass = HomeAssistant()
    client = AsyncMock()
    nodes = {"nodes": [{"addr": "1", "type": "htr"}]}
    inventory = inventory_builder("dev", nodes)
    coord = coord_module.StateCoordinator(
        hass,
        client=client,
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev"),
        nodes=None,
        inventory=inventory,
    )

    coord._inventory = None
    client.get_node_settings = AsyncMock(return_value={})

    result = await coord._async_update_data()

    assert coord._inventory is None
    assert result == {}


@pytest.mark.asyncio
async def test_async_update_data_omits_raw_nodes(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """Coordinator snapshots should omit raw node payloads and settings maps."""

    hass = HomeAssistant()
    client = AsyncMock()
    nodes_payload = {"nodes": [{"addr": "1", "type": "htr"}]}
    node_list = list(inventory_module.build_node_inventory(nodes_payload))
    inventory = inventory_builder("dev", nodes_payload, node_list)
    coord = coord_module.StateCoordinator(
        hass,
        client=client,
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev"),
        nodes=nodes_payload,
        inventory=inventory,
    )

    client.get_node_settings = AsyncMock(return_value={})

    result = await coord._async_update_data()

    record = result["dev"]
    assert "nodes" not in record
    assert "settings" not in record
    assert record["inventory"] is inventory
    assert record["inventory"].addresses_by_type["htr"] == ["1"]
    assert _state_payload(coord, "htr", "1") == {}


@pytest.mark.asyncio
async def test_async_fetch_settings_by_address_pending_and_boost(
    monkeypatch: pytest.MonkeyPatch,
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """Per-address fetch helper should defer pending payloads and resolve boost."""

    hass = HomeAssistant()
    client = AsyncMock()
    inventory = inventory_builder(
        "dev",
        {"nodes": [{"type": "acm", "addr": "1"}, {"type": "acm", "addr": "2"}]},
    )
    coord = coord_module.StateCoordinator(
        hass,
        client=client,
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev"),
        nodes=None,
        inventory=inventory,
    )

    responses = [
        {"mode": "boost", "boost_end_day": 1, "boost_end_min": 30},
        {"mode": "auto", "boost_end_day": 1, "boost_end_min": 60},
    ]
    client.get_node_settings = AsyncMock(side_effect=responses)

    pending_calls = 0

    def _fake_defer(_type: str, addr: str, payload: Mapping[str, Any] | None) -> bool:
        nonlocal pending_calls
        pending_calls += 1
        return pending_calls == 1

    monkeypatch.setattr(
        coord,
        "_should_defer_pending_setting",
        _fake_defer,
    )
    monkeypatch.setattr(
        coord_module.StateCoordinator,
        "_requires_boost_resolution",
        staticmethod(lambda payload: True),
    )
    rtc_value = dt.datetime(2024, 1, 1, 0, 0, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(
        coord,
        "_async_fetch_rtc_datetime",
        AsyncMock(return_value=rtc_value),
    )

    addr_map = {"acm": ["1", "2"]}
    reverse = {"1": {"acm"}, "2": {"acm"}}
    store = coord._state_store or coord._ensure_state_store(inventory)

    rtc_now = await coord._async_fetch_settings_to_store(
        "dev",
        addr_map,
        reverse,
        store,
        rtc_now=None,
    )

    assert pending_calls == 2
    assert store.get_state("acm", "1") is None
    state = store.get_state("acm", "2")
    assert state is not None
    assert state.mode == "auto"
    assert rtc_now == rtc_value
    coord._async_fetch_rtc_datetime.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_refresh_heater_errors_without_inventory(
    caplog: pytest.LogCaptureFixture,
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """Heater refresh should log an error when inventory cannot be rebuilt."""

    hass = HomeAssistant()
    client = AsyncMock()
    inventory = inventory_builder("dev", {})
    coord = coord_module.StateCoordinator(
        hass,
        client=client,
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev"),
        nodes=None,
        inventory=inventory,
    )

    coord._inventory = None
    assert coord._inventory is None
    client.get_node_settings = AsyncMock()

    with caplog.at_level(logging.ERROR):
        await coord.async_refresh_heater("1")

    client.get_node_settings.assert_not_called()
    assert any("inventory metadata" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_async_refresh_heater_fetches_rtc(
    monkeypatch: pytest.MonkeyPatch,
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """Refreshing an accumulator should resolve boost metadata using RTC."""

    hass = HomeAssistant()
    client = AsyncMock()
    base_now = dt.datetime(2024, 1, 1, 0, 0, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(coord_module.dt_util, "now", lambda: base_now)

    nodes_list = [AccumulatorNode(name="Accumulator 1", addr="1")]
    inventory = inventory_builder("dev", {}, nodes_list)
    coordinator = coord_module.StateCoordinator(
        hass,
        client=client,
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev"),
        nodes=None,
        inventory=inventory,
    )

    client.get_node_settings = AsyncMock(
        return_value={"boost_end_day": 1, "boost_end_min": 30}
    )
    client.get_rtc_time = AsyncMock(
        return_value={"y": 2024, "n": 1, "d": 1, "h": 0, "m": 0, "s": 0}
    )

    await coordinator.async_refresh_heater(("acm", "1"))
    client.get_rtc_time.assert_awaited_once()


def test_device_display_name_helper() -> None:
    """Helpers should trim names and fall back to the device id."""

    metadata = build_device_metadata_payload("dev", name=" Device ")
    assert coord_module._device_display_name(metadata, "dev") == "Device"
    assert (
        coord_module._device_display_name(
            build_device_metadata_payload("dev", name=""), "dev"
        )
        == "Device dev"
    )
    assert coord_module._device_display_name(None, "dev") == "Device dev"
    assert (
        coord_module._device_display_name(
            build_device_metadata_payload("dev", name=1234),
            "dev",
        )
        == "1234"
    )


def test_state_coordinator_omits_raw_device_payload(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """Coordinator should store typed metadata instead of raw device payloads."""

    hass = HomeAssistant()
    inventory = inventory_builder("dev", {})
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload(
            "dev",
            name=" Typed Device ",
            model=" Model X ",
        ),
        nodes=None,
        inventory=inventory,
    )

    assert not hasattr(coordinator, "_device")
    assert not isinstance(getattr(coordinator, "_device_metadata"), Mapping)

    device_record = coordinator._device_record()["dev"]
    assert device_record["name"] == "Typed Device"
    assert device_record["model"] == "Model X"


def test_mode_and_pending_key_helpers(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """Mode normalisation and pending key helpers should validate input."""

    assert coord_module.StateCoordinator._normalize_mode_value(None) is None
    assert coord_module.StateCoordinator._normalize_mode_value("  Auto ") == "auto"
    assert coord_module.StateCoordinator._normalize_mode_value(123) == "123"

    hass = HomeAssistant()
    inventory = inventory_builder("dev", {})
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev", name="Device"),
        nodes=None,
        inventory=inventory,
    )

    assert coordinator._pending_key("", "") is None
    assert coordinator._pending_key("htr", "1") == ("htr", "1")


def test_prune_and_register_pending_settings(
    monkeypatch: pytest.MonkeyPatch,
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """Pending settings helpers should drop expired entries and skip invalid keys."""

    hass = HomeAssistant()
    inventory = inventory_builder("dev", {})
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev", name="Device"),
        nodes=None,
        inventory=inventory,
    )

    coordinator.register_pending_setting("", "", mode="auto", stemp=21.0)
    assert coordinator._pending_settings == {}

    fake_time = {"value": 0.0}

    def _fake_time() -> float:
        return fake_time["value"]

    monkeypatch.setattr(coord_module.time, "time", _fake_time)
    monkeypatch.setattr(coord_module, "time_mod", _fake_time)

    coordinator._pending_settings[("htr", "1")] = coord_module.PendingSetting(
        mode="auto",
        stemp=None,
        expires_at=10.0,
    )
    fake_time["value"] = 20.0
    coordinator._prune_pending_settings()
    assert coordinator._pending_settings == {}


def test_should_defer_pending_setting_branches(
    monkeypatch: pytest.MonkeyPatch,
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """The pending settings deferral logic should cover all validation branches."""

    hass = HomeAssistant()
    inventory = inventory_builder("dev", {})
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev", name="Device"),
        nodes=None,
        inventory=inventory,
    )

    assert coordinator._should_defer_pending_setting("", "", None) is False
    assert coordinator._should_defer_pending_setting("htr", "1", {}) is False

    fake_time = {"value": 0.0}

    def _fake_time() -> float:
        return fake_time["value"]

    monkeypatch.setattr(coord_module.time, "time", _fake_time)
    monkeypatch.setattr(coord_module, "time_mod", _fake_time)

    coordinator._pending_settings[("htr", "1")] = coord_module.PendingSetting(
        mode="auto",
        stemp=None,
        expires_at=1.0,
    )
    fake_time["value"] = 5.0
    assert coordinator._should_defer_pending_setting("htr", "1", {}) is False
    assert coordinator._pending_settings == {}

    coordinator._pending_settings[("htr", "1")] = coord_module.PendingSetting(
        mode=None,
        stemp=None,
        expires_at=10.0,
    )
    fake_time["value"] = 0.0
    assert coordinator._should_defer_pending_setting("htr", "1", None) is True

    coordinator._pending_settings[("htr", "1")] = coord_module.PendingSetting(
        mode="auto",
        stemp=21.0,
        expires_at=10.0,
    )
    payload = {"mode": "manual"}
    assert coordinator._should_defer_pending_setting("htr", "1", payload) is True


@pytest.mark.asyncio
async def test_refresh_skips_pending_settings_merge(
    monkeypatch: pytest.MonkeyPatch,
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """Heater refresh should defer merging stale payloads while pending."""

    hass = HomeAssistant()
    client = AsyncMock()
    nodes = {"nodes": [{"type": "htr", "addr": "1", "name": "Heater"}]}
    node_list = [HeaterNode(name="Heater", addr="1")]
    inventory = inventory_builder("dev", nodes, node_list)
    coordinator = coord_module.StateCoordinator(
        hass,
        client=client,
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev", name="Device"),
        nodes=None,
        inventory=inventory,
    )

    store = coordinator._state_store or coordinator._ensure_state_store(inventory)
    assert store is not None
    store.apply_full_snapshot("htr", "1", {"mode": "manual", "stemp": "21.0"})

    client.get_node_settings = AsyncMock(return_value={"mode": "auto", "stemp": "20.0"})
    coordinator.register_pending_setting(
        "htr", "1", mode="manual", stemp=21.0, ttl=60.0
    )

    await coordinator.async_refresh_heater(("htr", "1"))

    settings = _state_payload(coordinator, "htr", "1")
    assert settings == {"mode": "manual", "stemp": "21.0"}
    assert ("htr", "1") in coordinator._pending_settings

    client.get_node_settings.return_value = {"mode": "manual", "stemp": "21.0"}

    await coordinator.async_refresh_heater(("htr", "1"))

    assert ("htr", "1") not in coordinator._pending_settings
    assert client.get_node_settings.await_count == 2


@pytest.mark.asyncio
async def test_poll_skips_pending_settings_merge(
    monkeypatch: pytest.MonkeyPatch,
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """Polling should defer merges until pending settings match."""

    hass = HomeAssistant()
    client = AsyncMock()
    nodes = {"nodes": [{"type": "htr", "addr": "1", "name": "Heater"}]}
    node_list = [HeaterNode(name="Heater", addr="1")]
    inventory = inventory_builder("dev", nodes, node_list)
    coordinator = coord_module.StateCoordinator(
        hass,
        client=client,
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev", name="Device"),
        nodes=None,
        inventory=inventory,
    )

    store = coordinator._state_store or coordinator._ensure_state_store(inventory)
    assert store is not None
    store.apply_full_snapshot("htr", "1", {"mode": "manual", "stemp": "21.0"})

    client.get_node_settings = AsyncMock(return_value={"mode": "auto", "stemp": "20.0"})
    coordinator.register_pending_setting(
        "htr", "1", mode="manual", stemp=21.0, ttl=60.0
    )

    await coordinator._async_update_data()

    settings = _state_payload(coordinator, "htr", "1")
    assert settings == {"mode": "manual", "stemp": "21.0"}
    assert ("htr", "1") in coordinator._pending_settings

    client.get_node_settings.return_value = {"mode": "manual", "stemp": "21.0"}

    await coordinator._async_update_data()

    assert ("htr", "1") not in coordinator._pending_settings
    assert client.get_node_settings.await_count == 2
    settings_second = _state_payload(coordinator, "htr", "1")
    assert settings_second == {"mode": "manual", "stemp": "21.0"}


def test_handle_ws_deltas_updates_store(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """Websocket deltas should update the coordinator data via the store."""

    hass = HomeAssistant()
    client = AsyncMock()
    nodes = {"nodes": [{"type": "htr", "addr": "1", "name": "Heater"}]}
    node_list = [HeaterNode(name="Heater", addr="1")]
    inventory = inventory_builder("dev", nodes, node_list)
    coordinator = coord_module.StateCoordinator(
        hass,
        client=client,
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev", name="Device"),
        nodes=None,
        inventory=inventory,
    )

    deltas = [
        NodeSettingsDelta(
            node_id=NodeId(NodeType.HEATER, "1"),
            changes={"mode": "auto", "status": {"stemp": "21.0"}},
        )
    ]
    coordinator.handle_ws_deltas("dev", deltas, replace=True)

    device_record = coordinator.data["dev"]
    assert "settings" not in device_record
    first_settings = _state_payload(coordinator, "htr", "1")
    assert first_settings is not None
    assert first_settings["mode"] == "auto"
    assert first_settings["stemp"] == "21.0"
    assert "status" not in first_settings

    coordinator.handle_ws_deltas(
        "dev",
        [NodeSettingsDelta(NodeId(NodeType.HEATER, "1"), {"stemp": "19.0"})],
        replace=False,
    )

    merged_settings = _state_payload(coordinator, "htr", "1")
    assert merged_settings is not None
    assert merged_settings["mode"] == "auto"
    assert merged_settings["stemp"] == "19.0"


def test_apply_entity_patch_uses_typed_state(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """Optimistic patches should mutate typed state without dict fallbacks."""

    hass = HomeAssistant()
    client = AsyncMock()
    nodes = {"nodes": [{"type": "htr", "addr": "1", "name": "Heater"}]}
    node_list = [HeaterNode(name="Heater", addr="1")]
    inventory = inventory_builder("dev", nodes, node_list)
    coordinator = coord_module.StateCoordinator(
        hass,
        client=client,
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev", name="Device"),
        nodes=None,
        inventory=inventory,
    )

    store = coordinator._state_store or coordinator._ensure_state_store(inventory)
    assert store is not None
    store.apply_full_snapshot("htr", "1", {"mode": "manual"})

    assert coordinator.apply_entity_patch(
        "htr",
        "1",
        lambda cur: setattr(cur, "mode", "auto"),
    )
    state = store.get_state(NodeType.HEATER, "1")
    assert state_to_dict(state) == {"mode": "auto"}

    assert (
        coordinator.apply_entity_patch(
            "htr",
            "2",
            lambda cur: setattr(cur, "mode", "manual"),
        )
        is False
    )

    def _bad_mutator(cur: Any) -> None:
        cur["mode"] = "heat"  # type: ignore[index]

    assert coordinator.apply_entity_patch("htr", "1", _bad_mutator) is False
    state_after = store.get_state(NodeType.HEATER, "1")
    assert state_to_dict(state_after) == {"mode": "auto"}


def test_wrap_logger_proxies_missing_helpers() -> None:
    """Ensure logger proxies expose inner attributes and ``isEnabledFor``."""

    class DummyLogger:
        def __init__(self) -> None:
            self.value = "logger"

    proxy = coord_module._wrap_logger(DummyLogger())

    assert proxy.value == "logger"
    assert proxy.isEnabledFor(10) is False


def test_wrap_logger_suppresses_manual_update_debug() -> None:
    """Ensure wrapped loggers skip DataUpdateCoordinator manual update noise."""

    class DummyLogger:
        def __init__(self) -> None:
            self.debug_calls: list[tuple[object, tuple[object, ...]]] = []

        def debug(self, msg: object, *args: object, **_kwargs: object) -> None:
            self.debug_calls.append((msg, args))

        def isEnabledFor(self, _level: int) -> bool:
            return True

    inner = DummyLogger()
    proxy = coord_module._wrap_logger(inner)

    proxy.debug("Manually updated %s data", "termoweb")
    proxy.debug("Retained %s", "entry")

    assert inner.debug_calls == [("Retained %s", ("entry",))]


# ---------------------------------------------------------------------------
# StateCoordinator constructor: inventory type guard (lines 217-218)
# ---------------------------------------------------------------------------


def test_coordinator_requires_inventory_instance(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """StateCoordinator should raise TypeError when inventory is not an Inventory."""

    hass = HomeAssistant()
    with pytest.raises(TypeError, match="Inventory instance"):
        coord_module.StateCoordinator(
            hass,
            client=AsyncMock(),
            base_interval=30,
            dev_id="dev",
            device=build_device_metadata_payload("dev"),
            nodes=None,
            inventory="not-an-inventory",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Properties: device_metadata, gateway_name, gateway_model (lines 239, 245, 251)
# ---------------------------------------------------------------------------


def test_coordinator_metadata_properties(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """device_metadata, gateway_name, gateway_model properties should return expected values."""

    hass = HomeAssistant()
    inventory = inventory_builder("dev", {"nodes": []})
    device = build_device_metadata_payload("dev", name="My Device", model="TW100")
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device=device,
        nodes=None,
        inventory=inventory,
    )

    assert coordinator.device_metadata is device
    assert coordinator.gateway_name == "My Device"
    assert coordinator.gateway_model == "TW100"


# ---------------------------------------------------------------------------
# apply_energy_snapshot edge cases (lines 278, 299, 301, 304, 307)
# ---------------------------------------------------------------------------


def test_apply_energy_snapshot_rejects_non_snapshot(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """apply_energy_snapshot should silently ignore non-EnergySnapshot values."""

    hass = HomeAssistant()
    inventory = inventory_builder("dev", {"nodes": []})
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev"),
        nodes=None,
        inventory=inventory,
    )
    # Should not raise
    coordinator.apply_energy_snapshot("not-a-snapshot")  # type: ignore[arg-type]
    coordinator.apply_energy_snapshot(None)  # type: ignore[arg-type]


def test_apply_energy_snapshot_wrong_dev_id(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """apply_energy_snapshot should ignore snapshots for a different device."""

    from custom_components.termoweb.domain.energy import EnergySnapshot

    hass = HomeAssistant()
    inventory = inventory_builder("dev", {"nodes": []})
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev"),
        nodes=None,
        inventory=inventory,
    )
    snapshot = EnergySnapshot(dev_id="other", metrics={}, updated_at=1.0, ws_deadline=None)
    coordinator.apply_energy_snapshot(snapshot)


def test_apply_energy_snapshot_no_inventory(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """apply_energy_snapshot should skip when inventory is missing."""

    from custom_components.termoweb.domain.energy import EnergySnapshot

    hass = HomeAssistant()
    inventory = inventory_builder("dev", {"nodes": []})
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev"),
        nodes=None,
        inventory=inventory,
    )
    coordinator._inventory = None
    snapshot = EnergySnapshot(dev_id="dev", metrics={}, updated_at=1.0, ws_deadline=None)
    coordinator.apply_energy_snapshot(snapshot)


# ---------------------------------------------------------------------------
# _filtered_settings_payload (line 345)
# ---------------------------------------------------------------------------


def test_filtered_settings_payload_non_mapping(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """_filtered_settings_payload should return empty dict for non-Mapping."""

    hass = HomeAssistant()
    inventory = inventory_builder("dev", {"nodes": []})
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev"),
        nodes=None,
        inventory=inventory,
    )
    assert coordinator._filtered_settings_payload("not a dict") == {}  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _instant_power_key empty normalization (line 361)
# ---------------------------------------------------------------------------


def test_instant_power_key_empty_values(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """_instant_power_key should return None for empty type or addr."""

    hass = HomeAssistant()
    inventory = inventory_builder("dev", {"nodes": []})
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev"),
        nodes=None,
        inventory=inventory,
    )
    assert coordinator._instant_power_key("", "01") is None
    assert coordinator._instant_power_key("htr", "") is None


# ---------------------------------------------------------------------------
# _record_instant_power: non-numeric watts, NaN, same-value-same-ts (lines 396-424)
# ---------------------------------------------------------------------------


def test_record_instant_power_non_numeric_watts(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """_record_instant_power should reject non-numeric watts (line 399)."""

    hass = HomeAssistant()
    inventory = inventory_builder("dev", {"nodes": []})
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev"),
        nodes=None,
        inventory=inventory,
    )
    assert coordinator._record_instant_power("htr", "01", "not-a-number", source="rest") is False  # type: ignore[arg-type]


def test_record_instant_power_nan_and_negative(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """_record_instant_power should reject NaN and negative watts (lines 403, 406)."""

    hass = HomeAssistant()
    inventory = inventory_builder("dev", {"nodes": []})
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev"),
        nodes=None,
        inventory=inventory,
    )
    assert coordinator._record_instant_power("htr", "01", float("nan"), source="rest") is False
    assert coordinator._record_instant_power("htr", "01", -5.0, source="rest") is False


def test_record_instant_power_duplicate_same_ts_source(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """_record_instant_power: same source + older/same timestamp skips (lines 412, 424)."""

    hass = HomeAssistant()
    inventory = inventory_builder("dev", {"nodes": []})
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev"),
        nodes=None,
        inventory=inventory,
    )
    assert coordinator._record_instant_power("htr", "01", 50.0, timestamp=100.0, source="rest") is True
    # Same source, same timestamp, same watts => skip (line 424)
    assert coordinator._record_instant_power("htr", "01", 50.0, timestamp=100.0, source="rest") is False
    # Same source, older timestamp => skip (line 412)
    assert coordinator._record_instant_power("htr", "01", 60.0, timestamp=99.0, source="rest") is False


def test_record_instant_power_default_timestamp(
    monkeypatch: pytest.MonkeyPatch,
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """_record_instant_power should use time.time() when no timestamp given (line 406)."""

    import time

    hass = HomeAssistant()
    inventory = inventory_builder("dev", {"nodes": []})
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev"),
        nodes=None,
        inventory=inventory,
    )
    monkeypatch.setattr(time, "time", lambda: 5000.0)
    assert coordinator._record_instant_power("htr", "01", 75.0, source="rest") is True
    entry = coordinator.instant_power_entry("htr", "01")
    assert entry is not None
    assert entry.timestamp == 5000.0


# ---------------------------------------------------------------------------
# handle_instant_power_update: non-int/float watts (line 449)
# ---------------------------------------------------------------------------


def test_handle_instant_power_update_non_numeric(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """handle_instant_power_update should ignore non-numeric watts (line 449)."""

    hass = HomeAssistant()
    inventory = inventory_builder("dev", {"nodes": []})
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev"),
        nodes=None,
        inventory=inventory,
    )
    coordinator.handle_instant_power_update("dev", "htr", "01", "not-a-number")  # type: ignore[arg-type]
    assert coordinator.instant_power_entry("htr", "01") is None


# ---------------------------------------------------------------------------
# instant_power_entry with None key (line 470)
# ---------------------------------------------------------------------------


def test_instant_power_entry_bad_key(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """instant_power_entry should return None for invalid node key (line 470)."""

    hass = HomeAssistant()
    inventory = inventory_builder("dev", {"nodes": []})
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev"),
        nodes=None,
        inventory=inventory,
    )
    assert coordinator.instant_power_entry("", "01") is None


# ---------------------------------------------------------------------------
# _should_skip_rest_power (line 483)
# ---------------------------------------------------------------------------


def test_should_skip_rest_power_ws_fresh(
    monkeypatch: pytest.MonkeyPatch,
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """_should_skip_rest_power should return True when WS data is fresh (line 483+)."""

    import time as time_mod

    hass = HomeAssistant()
    inventory = inventory_builder("dev", {"nodes": []})
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev"),
        nodes=None,
        inventory=inventory,
    )
    monkeypatch.setattr(time_mod, "time", lambda: 1000.0)
    coordinator.handle_instant_power_update("dev", "htr", "01", 50.0, timestamp=1000.0)

    monkeypatch.setattr(time_mod, "time", lambda: 1010.0)
    assert coordinator._should_skip_rest_power("htr", "01") is True

    # After the interval, should not skip
    monkeypatch.setattr(time_mod, "time", lambda: 2000.0)
    assert coordinator._should_skip_rest_power("htr", "01") is False


# ---------------------------------------------------------------------------
# update_nodes: inventory rebinding and None inventory (lines 826-828)
# ---------------------------------------------------------------------------


def test_update_nodes_inventory_rebinding_raises(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """update_nodes should raise ValueError when rebinding inventory (line 819-820)."""

    hass = HomeAssistant()
    inv1 = inventory_builder("dev", {"nodes": [{"type": "htr", "addr": "1"}]})
    inv2 = inventory_builder("dev", {"nodes": [{"type": "htr", "addr": "2"}]})
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev"),
        nodes=None,
        inventory=inv1,
    )

    with pytest.raises(ValueError, match="rebinding"):
        coordinator.update_nodes(inventory=inv2)


def test_update_nodes_none_inventory_clears_state(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """update_nodes with non-Inventory should clear state (lines 826-828)."""

    hass = HomeAssistant()
    inventory = inventory_builder("dev", {"nodes": [{"type": "htr", "addr": "1"}]})
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev"),
        nodes=None,
        inventory=inventory,
    )

    # Force a re-init with inventory=None by removing existing inventory first
    coordinator._inventory = None
    coordinator.update_nodes(inventory=None)
    assert coordinator._inventory is None
    assert coordinator._state_store is None


# ---------------------------------------------------------------------------
# _node_ids_from_inventory: various node shapes (lines 843-859)
# ---------------------------------------------------------------------------


def test_node_ids_from_inventory_with_mapping_nodes(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """_node_ids_from_inventory should handle mapping-style nodes (lines 843-844)."""

    hass = HomeAssistant()
    # Build inventory with valid and invalid nodes
    inventory = inventory_builder("dev", {
        "nodes": [
            {"type": "htr", "addr": "1"},
            {"type": "acm", "addr": "2"},
        ]
    })
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev"),
        nodes=None,
        inventory=inventory,
    )

    node_ids = coordinator._node_ids_from_inventory(inventory)
    types = {nid.node_type.value for nid in node_ids}
    assert "htr" in types
    assert "acm" in types


# ---------------------------------------------------------------------------
# _ensure_state_store: reset path (line 869)
# ---------------------------------------------------------------------------


def test_ensure_state_store_resets_existing(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """_ensure_state_store should reset existing store (line 869)."""

    hass = HomeAssistant()
    inv = inventory_builder("dev", {"nodes": [{"type": "htr", "addr": "1"}]})
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev"),
        nodes=None,
        inventory=inv,
    )
    store = coordinator._state_store
    assert store is not None
    # Call again -- should reset the same store
    result = coordinator._ensure_state_store(inv)
    assert result is store  # same object


# ---------------------------------------------------------------------------
# handle_ws_deltas (lines 883-908)
# ---------------------------------------------------------------------------


def test_handle_ws_deltas_wrong_dev_id(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """handle_ws_deltas should skip when dev_id does not match (line 883)."""

    hass = HomeAssistant()
    inventory = inventory_builder("dev", {"nodes": [{"type": "htr", "addr": "1"}]})
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev"),
        nodes=None,
        inventory=inventory,
    )
    coordinator.handle_ws_deltas("other", [])
    # No error means it returned early


def test_handle_ws_deltas_no_inventory(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """handle_ws_deltas should skip when inventory is missing (line 887)."""

    hass = HomeAssistant()
    inventory = inventory_builder("dev", {"nodes": []})
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev"),
        nodes=None,
        inventory=inventory,
    )
    coordinator._inventory = None
    coordinator.handle_ws_deltas("dev", [])


def test_handle_ws_deltas_no_store(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """handle_ws_deltas should create store when missing (line 891)."""

    hass = HomeAssistant()
    inventory = inventory_builder("dev", {"nodes": [{"type": "htr", "addr": "1"}]})
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev"),
        nodes=None,
        inventory=inventory,
    )
    # Force store to None
    coordinator._state_store = None
    delta = NodeSettingsDelta(
        node_id=NodeId(NodeType.HEATER, "1"),
        changes={"mode": "auto"},
    )
    coordinator.handle_ws_deltas("dev", [delta])
    assert coordinator._state_store is not None


def test_handle_ws_deltas_skips_non_settings_delta(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """handle_ws_deltas should skip non-NodeSettingsDelta (line 896)."""

    hass = HomeAssistant()
    inventory = inventory_builder("dev", {"nodes": [{"type": "htr", "addr": "1"}]})
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev"),
        nodes=None,
        inventory=inventory,
    )
    # Should not raise and should not publish (no applied deltas)
    coordinator.handle_ws_deltas("dev", ["not-a-delta"])  # type: ignore[list-item]


def test_handle_ws_deltas_replace_mode(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """handle_ws_deltas with replace=True should replace full snapshot (line 908)."""

    hass = HomeAssistant()
    inventory = inventory_builder("dev", {"nodes": [{"type": "htr", "addr": "1"}]})
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev"),
        nodes={"nodes": [{"type": "htr", "addr": "1"}]},
        inventory=inventory,
    )
    delta = NodeSettingsDelta(
        node_id=NodeId(NodeType.HEATER, "1"),
        changes={"mode": "manual", "stemp": "21.0"},
    )
    coordinator.handle_ws_deltas("dev", [delta], replace=True)
    state = _state_payload(coordinator, "htr", "1")
    assert state is not None
    assert state["mode"] == "manual"


# ---------------------------------------------------------------------------
# apply_entity_patch: new state creation for acm/thm/pmo (lines 948-966, 976, 993-994)
# ---------------------------------------------------------------------------


def test_apply_entity_patch_creates_accumulator_state(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """apply_entity_patch should create AccumulatorState for acm node (lines 959-960)."""

    hass = HomeAssistant()
    inventory = inventory_builder("dev", {"nodes": [{"type": "acm", "addr": "1"}]})
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev"),
        nodes={"nodes": [{"type": "acm", "addr": "1"}]},
        inventory=inventory,
    )

    def _mutator(state: Any) -> None:
        state.mode = "boost"
        state.boost_active = True
        state.boost_end_day = 2
        state.boost_end_min = 60

    result = coordinator.apply_entity_patch("acm", "1", _mutator)
    assert result is True
    state = _state_payload(coordinator, "acm", "1")
    assert state is not None
    assert state["mode"] == "boost"
    assert state["boost_active"] is True
    # boost fields changed => boost_end_datetime and boost_minutes_delta should be cleared (line 993-994)
    assert state.get("boost_end_datetime") is None
    assert state.get("boost_minutes_delta") is None


def test_apply_entity_patch_creates_thermostat_state(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """apply_entity_patch should create ThermostatState for thm node (lines 961-962)."""

    hass = HomeAssistant()
    inventory = inventory_builder("dev", {"nodes": [{"type": "thm", "addr": "1"}]})
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev"),
        nodes={"nodes": [{"type": "thm", "addr": "1"}]},
        inventory=inventory,
    )

    def _mutator(state: Any) -> None:
        state.mode = "auto"

    result = coordinator.apply_entity_patch("thm", "1", _mutator)
    assert result is True


def test_apply_entity_patch_creates_power_monitor_state(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """apply_entity_patch should create PowerMonitorState for pmo node (lines 963-964)."""

    hass = HomeAssistant()
    inventory = inventory_builder("dev", {"nodes": [{"type": "pmo", "addr": "1"}]})
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev"),
        nodes={"nodes": [{"type": "pmo", "addr": "1"}]},
        inventory=inventory,
    )

    def _mutator(state: Any) -> None:
        state.power = 150.0

    result = coordinator.apply_entity_patch("pmo", "1", _mutator)
    assert result is True


def test_apply_entity_patch_creates_heater_state(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """apply_entity_patch should create HeaterState for htr node (line 966)."""

    hass = HomeAssistant()
    inventory = inventory_builder("dev", {"nodes": [{"type": "htr", "addr": "1"}]})
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev"),
        nodes={"nodes": [{"type": "htr", "addr": "1"}]},
        inventory=inventory,
    )

    def _mutator(state: Any) -> None:
        state.mode = "auto"

    result = coordinator.apply_entity_patch("htr", "1", _mutator)
    assert result is True


def test_apply_entity_patch_no_store_returns_false(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """apply_entity_patch should return False when store or inventory is missing (line 920)."""

    hass = HomeAssistant()
    inventory = inventory_builder("dev", {"nodes": []})
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev"),
        nodes=None,
        inventory=inventory,
    )
    coordinator._inventory = None
    result = coordinator.apply_entity_patch("htr", "1", lambda s: None)
    assert result is False


def test_apply_entity_patch_unknown_type_returns_false(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """apply_entity_patch should return False for unresolvable node type (line 927)."""

    hass = HomeAssistant()
    inventory = inventory_builder("dev", {"nodes": [{"type": "htr", "addr": "1"}]})
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev"),
        nodes=None,
        inventory=inventory,
    )
    result = coordinator.apply_entity_patch("", "1", lambda s: None)
    assert result is False


def test_apply_entity_patch_multi_type_node(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """apply_entity_patch with shared addr should patch multiple types (lines 948-951)."""

    hass = HomeAssistant()
    inventory = inventory_builder("dev", {
        "nodes": [
            {"type": "htr", "addr": "1"},
            {"type": "acm", "addr": "1"},
        ]
    })
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device=build_device_metadata_payload("dev"),
        nodes={"nodes": [{"type": "htr", "addr": "1"}, {"type": "acm", "addr": "1"}]},
        inventory=inventory,
    )

    def _mutator(state: Any) -> None:
        state.mode = "manual"

    result = coordinator.apply_entity_patch("htr", "1", _mutator)
    assert result is True


# ---------------------------------------------------------------------------
# _wrap_logger: isEnabledFor with real logger (line 1884)
# ---------------------------------------------------------------------------


def test_wrap_logger_is_enabled_for_with_real_logger() -> None:
    """_wrap_logger.isEnabledFor should delegate to inner logger (line 1884)."""

    import logging

    logger = logging.getLogger("test_coordinator_wrap")
    logger.setLevel(logging.DEBUG)
    proxy = coord_module._wrap_logger(logger)
    assert proxy.isEnabledFor(logging.DEBUG) is True
    assert proxy.isEnabledFor(logging.CRITICAL) is True
