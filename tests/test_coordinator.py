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
from custom_components.termoweb.domain import NodeId, NodeSettingsDelta, NodeType
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
    return state.to_legacy() if state is not None else None


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
            changes={"mode": "auto", "status": {"online": True}},
        )
    ]
    coordinator.handle_ws_deltas("dev", deltas, replace=True)

    device_record = coordinator.data["dev"]
    assert "settings" not in device_record
    first_settings = _state_payload(coordinator, "htr", "1")
    assert first_settings is not None
    assert first_settings["mode"] == "auto"
    assert first_settings["status"]["online"] is True

    coordinator.handle_ws_deltas(
        "dev",
        [NodeSettingsDelta(NodeId(NodeType.HEATER, "1"), {"stemp": "19.0"})],
        replace=False,
    )

    merged_settings = _state_payload(coordinator, "htr", "1")
    assert merged_settings is not None
    assert merged_settings["mode"] == "auto"
    assert merged_settings["stemp"] == "19.0"


def test_wrap_logger_proxies_missing_helpers() -> None:
    """Ensure logger proxies expose inner attributes and ``isEnabledFor``."""

    class DummyLogger:
        def __init__(self) -> None:
            self.value = "logger"

    proxy = coord_module._wrap_logger(DummyLogger())

    assert proxy.value == "logger"
    assert proxy.isEnabledFor(10) is False
