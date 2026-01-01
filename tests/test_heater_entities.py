from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Iterator

import pytest
from unittest.mock import MagicMock

from conftest import _install_stubs, make_ws_payload

_install_stubs()

from custom_components.termoweb import boost as boost_module, heater as heater_module
from custom_components.termoweb.domain import (
    DomainStateStore,
    NodeId as DomainNodeId,
    NodeType as DomainNodeType,
)
from custom_components.termoweb.domain.state import HeaterState
from custom_components.termoweb.inventory import (
    Inventory,
    InventoryNodeMetadata,
    build_node_inventory,
)
from custom_components.termoweb.const import DOMAIN, signal_ws_data
from custom_components.termoweb.binary_sensor import HeaterBoostActiveBinarySensor
from custom_components.termoweb.sensor import (
    HeaterBoostEndSensor,
    HeaterBoostMinutesRemainingSensor,
)
from homeassistant.const import STATE_UNKNOWN
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

HeaterNodeBase = heater_module.HeaterNodeBase


def test_heater_node_base_normalizes_address(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, dict]] = []

    original = heater_module.normalize_node_addr

    def _record_normalize(value, **kwargs):
        calls.append((value, kwargs))
        return original(value, **kwargs)

    monkeypatch.setattr(heater_module, "normalize_node_addr", _record_normalize)

    coordinator = SimpleNamespace(hass=None)
    heater = HeaterNodeBase(coordinator, "entry", "dev", " 01 ", " Heater 01 ")

    assert heater._addr == "01"
    assert heater.device_info["identifiers"] == {(heater_module.DOMAIN, "dev", "01")}
    assert heater._attr_unique_id == f"{heater_module.DOMAIN}:dev:htr:01"
    assert calls == [(" 01 ", {})]


def test_heater_node_base_payload_matching_normalizes_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, dict]] = []

    original = heater_module.normalize_node_addr

    def _record_normalize(value, **kwargs):
        calls.append((value, kwargs))
        return original(value, **kwargs)

    monkeypatch.setattr(heater_module, "normalize_node_addr", _record_normalize)

    coordinator = SimpleNamespace(hass=None)
    heater = HeaterNodeBase(coordinator, "entry", "dev", " 01 ", "Heater 01")

    assert heater._payload_matches_heater(make_ws_payload("dev", " 01 "))
    assert not heater._payload_matches_heater(make_ws_payload("dev", "02"))
    assert not heater._payload_matches_heater(make_ws_payload("dev", "  "))
    assert calls == [(" 01 ", {}), (" 01 ", {}), ("02", {}), ("  ", {})]


def test_heater_async_will_remove_without_listener_resets_unsub() -> None:
    heater = HeaterNodeBase(SimpleNamespace(hass=None), "entry", "dev", "1", "Heater 1")
    unsub = MagicMock()
    heater._async_unsub_coordinator_update = unsub

    asyncio.run(heater.async_will_remove_from_hass())

    unsub.assert_called_once_with()
    assert getattr(heater, "_async_unsub_coordinator_update") is None


@pytest.mark.asyncio
async def test_async_added_to_hass_without_coordinator_listener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hass = HomeAssistant()
    coordinator = SimpleNamespace(hass=hass)
    heater = HeaterNodeBase(coordinator, "entry", "dev", "1", "Heater 1")
    heater.hass = hass

    recorded: list[tuple[HomeAssistant, str, Any]] = []

    def _record_subscribe(self, hass_param, signal, handler):
        recorded.append((hass_param, signal, handler))

    monkeypatch.setattr(
        heater_module.DispatcherSubscriptionHelper,
        "subscribe",
        _record_subscribe,
    )

    await heater.async_added_to_hass()

    assert getattr(heater, "_async_unsub_coordinator_update", None) is None
    assert recorded == [
        (hass, signal_ws_data("entry"), heater._handle_ws_message),
    ]


def test_heater_handle_ws_event_skips_removed_entity() -> None:
    hass = HomeAssistant()
    coordinator = SimpleNamespace(hass=hass)
    heater = HeaterNodeBase(coordinator, "entry", "dev", "1", "Heater 1")
    heater.hass = hass
    callback = MagicMock()
    heater.schedule_update_ha_state = callback  # type: ignore[assignment]
    heater._removed = True

    heater._handle_ws_event({"dev_id": "dev", "addr": "1", "node_type": "htr"})

    callback.assert_not_called()


def test_heater_handle_ws_event_requires_callable_callback() -> None:
    hass = HomeAssistant()
    coordinator = SimpleNamespace(hass=hass)
    heater = HeaterNodeBase(coordinator, "entry", "dev", "1", "Heater 1")
    heater.hass = hass
    heater.schedule_update_ha_state = None  # type: ignore[assignment]
    heater._removed = False

    heater._handle_ws_event({"dev_id": "dev", "addr": "1", "node_type": "htr"})


def test_heater_handle_ws_event_requires_loop_or_mock() -> None:
    hass = SimpleNamespace(loop=None)
    heater = HeaterNodeBase(SimpleNamespace(hass=hass), "entry", "dev", "1", "Heater 1")
    heater.hass = hass
    called = False

    def _callback() -> None:
        nonlocal called
        called = True

    heater.schedule_update_ha_state = _callback  # type: ignore[assignment]
    heater._removed = False

    heater._handle_ws_event({"dev_id": "dev", "addr": "1", "node_type": "htr"})

    assert called is False


def test_heater_section_requires_inventory_for_name() -> None:
    """Heater metadata should not fabricate inventory-backed details."""

    coordinator = SimpleNamespace(
        data={"dev": {"settings": {"htr": {"1": {"mode": "auto"}}}}},
    )
    store = DomainStateStore([DomainNodeId(DomainNodeType.HEATER, "1")])
    store.apply_full_snapshot("htr", "1", {"mode": "auto"})
    coordinator.domain_view = heater_module.DomainStateView("dev", store)
    heater = HeaterNodeBase(coordinator, "entry", "dev", "1", "Heater 1")

    section = heater._heater_section()

    assert section == {"settings": {"1": {"mode": "auto"}}}
    state = heater.heater_state()
    assert isinstance(state, HeaterState)
    assert state.mode == "auto"
    assert heater.available is False


def test_heater_section_includes_inventory_details() -> None:
    """Heater metadata should expose inventory-backed name and availability."""

    raw_nodes = {"nodes": [{"type": "htr", "addr": "1", "name": "Living"}]}
    inventory = Inventory("dev", build_node_inventory(raw_nodes))
    coordinator = SimpleNamespace(
        data={"dev": {"settings": {"htr": {"1": {"mode": "auto"}}}}},
        inventory=inventory,
    )
    store = DomainStateStore([DomainNodeId(DomainNodeType.HEATER, "1")])
    store.apply_full_snapshot("htr", "1", {"mode": "auto"})
    coordinator.domain_view = heater_module.DomainStateView("dev", store)
    heater = HeaterNodeBase(coordinator, "entry", "dev", "1", None)

    section = heater._heater_section()

    assert section["settings"] == {"1": {"mode": "auto"}}
    assert section["name"] == "Living"
    state = heater.heater_state()
    assert isinstance(state, HeaterState)
    assert state.mode == "auto"
    assert heater.available is True


def test_heater_resolve_inventory_logs_missing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Inventory resolution should raise when the immutable cache is missing."""

    caplog.set_level("ERROR")

    coordinator = SimpleNamespace(data={"dev": {}}, inventory=None)
    heater = HeaterNodeBase(coordinator, "entry", "dev", "1", "Heater 1")

    with pytest.raises(ValueError):
        heater._resolve_inventory()

    assert "missing immutable inventory cache" in caplog.text


def test_heater_platform_details_missing_inventory(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Inventory resolution should raise when device metadata is absent."""

    caplog.set_level("ERROR")

    with pytest.raises(ValueError):
        heater_module.heater_platform_details_for_entry(
            {"dev_id": "dev-1"},
            default_name_simple=lambda addr: addr,
        )

    assert "missing inventory" in caplog.text


def test_heater_client_handles_missing_hass_data() -> None:
    heater = HeaterNodeBase(SimpleNamespace(hass=None), "entry", "dev", "1", "Heater 1")

    assert heater._client() is None

    heater.hass = SimpleNamespace(data=None)
    assert heater._client() is None

    heater.hass = SimpleNamespace(data={DOMAIN: []})
    assert heater._client() is None

    heater.hass = SimpleNamespace(data={DOMAIN: {"entry": []}})
    assert heater._client() is None


def test_boost_runtime_storage_roundtrip(heater_hass_data) -> None:
    """Ensure boost runtime helpers normalise addresses and defaults."""

    hass = HomeAssistant()
    entry_id = "entry-store"
    heater_hass_data(
        hass,
        entry_id,
        "dev-store",
        SimpleNamespace(),
        boost_runtime={},
    )

    assert heater_module.get_boost_runtime_minutes(hass, entry_id, "acm", "01") is None

    heater_module.set_boost_runtime_minutes(hass, entry_id, "acm", "01", 120)
    assert heater_module.get_boost_runtime_minutes(hass, entry_id, "ACM", " 01 ") == 120
    assert (
        heater_module.resolve_boost_runtime_minutes(
            hass,
            entry_id,
            "ACM",
            "01",
        )
        == 120
    )

    heater_module.set_boost_runtime_minutes(hass, entry_id, "acm", "01", None)
    assert heater_module.get_boost_runtime_minutes(hass, entry_id, "acm", "01") is None
    assert (
        heater_module.resolve_boost_runtime_minutes(
            hass,
            entry_id,
            "acm",
            "01",
        )
        == heater_module.DEFAULT_BOOST_DURATION
    )


def test_derive_boost_state_uses_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify boost metadata derivation favours the coordinator resolver."""

    base_now = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(dt_util, "now", lambda: base_now)

    expected_end = base_now + timedelta(hours=1)

    class _Coordinator(SimpleNamespace):
        def resolve_boost_end(self, day, minute, *, now=None):
            return expected_end, 60

    settings = {"mode": "boost", "boost_end_day": 1, "boost_end_min": 60}
    state = heater_module.derive_boost_state(settings, _Coordinator())

    assert state.active is True
    assert state.minutes_remaining == 60
    assert state.end_datetime == expected_end
    assert state.end_iso == expected_end.isoformat()
    assert state.end_label is None


def test_derive_boost_state_from_remaining(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure boost metadata uses remaining minutes when resolver unavailable."""

    base_now = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(dt_util, "now", lambda: base_now)

    settings = {"boost_remaining": "15"}
    state = heater_module.derive_boost_state(settings, SimpleNamespace())

    assert state.active is False
    assert state.minutes_remaining == 15
    assert state.end_iso == (base_now + timedelta(minutes=15)).isoformat()
    assert state.end_datetime == base_now + timedelta(minutes=15)
    assert state.end_label is None


def test_derive_boost_state_handles_now_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure defensive handling when ``dt_util.now`` raises."""

    def _failing_now() -> datetime:
        raise RuntimeError("boom")

    monkeypatch.setattr(dt_util, "now", _failing_now)

    settings = {"boost_remaining": 10}
    state = heater_module.derive_boost_state(settings, SimpleNamespace())

    assert state.minutes_remaining == 10
    assert state.end_datetime is None
    assert state.end_iso is None
    assert state.end_label == "Never"


def test_derive_boost_state_parses_iso_without_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback ISO parsing should use datetime.fromisoformat when needed."""

    monkeypatch.setattr(dt_util, "parse_datetime", lambda value: None)
    settings = {
        "boost_active": True,
        "boost_end_datetime": "2024-01-01T00:30:00+00:00",
    }

    state = heater_module.derive_boost_state(settings, SimpleNamespace())

    assert state.end_iso == "2024-01-01T00:30:00+00:00"
    assert state.end_datetime == datetime.fromisoformat("2024-01-01T00:30:00+00:00")


def test_derive_boost_state_ignores_placeholder_iso(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Placeholder 1970-era timestamps should be treated as unset."""

    monkeypatch.setattr(dt_util, "parse_datetime", lambda value: None)
    settings = {
        "mode": "auto",
        "boost": False,
        "boost_end_datetime": "1970-01-02T00:00:00UTC",
    }

    state = heater_module.derive_boost_state(settings, SimpleNamespace())

    assert state.end_datetime is None
    assert state.end_iso is None
    assert state.end_label == "Never"


def test_derive_boost_state_parses_string_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """Boost metadata should parse ISO formatted end timestamps."""

    base_now = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(dt_util, "now", lambda: base_now)

    iso = "2024-01-01T00:30:00+00:00"
    settings = {"boost_active": True, "boost_end_datetime": iso}
    state = heater_module.derive_boost_state(settings, SimpleNamespace())

    assert state.active is True
    assert state.minutes_remaining == 30
    assert state.end_iso == iso
    assert state.end_datetime == datetime(2024, 1, 1, 0, 30, tzinfo=timezone.utc)
    assert state.end_label is None


def test_derive_boost_state_handles_now_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure boost end derivation tolerates errors from ``dt_util.now``."""

    def _raise_now() -> datetime:
        raise RuntimeError("boom")

    monkeypatch.setattr(dt_util, "now", _raise_now)

    coordinator = SimpleNamespace(resolve_boost_end=None)
    state = heater_module.derive_boost_state({"boost_remaining": "30"}, coordinator)

    assert state.minutes_remaining == 30
    assert state.end_datetime is None
    assert state.end_iso is None
    assert state.end_label == "Never"


def test_derive_boost_state_normalises_epoch_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Epoch placeholders should become a "Never" label without timestamps."""

    base_now = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(dt_util, "now", lambda: base_now)

    def _resolver(day: int, minute: int, *, now=None):
        return boost_module.resolve_boost_end_from_fields(day, minute, now=base_now)

    coordinator = SimpleNamespace(resolve_boost_end=_resolver)
    settings = {"boost": False, "boost_end_day": 0, "boost_end_min": 0}

    state = heater_module.derive_boost_state(settings, coordinator)

    assert state.active is False
    assert state.minutes_remaining is None
    assert state.end_datetime is None
    assert state.end_iso is None
    assert state.end_label == "Never"


def test_derive_boost_state_uses_parse_datetime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify ``dt_util.parse_datetime`` is leveraged when available."""

    iso_value = "2024-01-02T03:04:05+00:00"
    parsed = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    calls: list[str] = []

    def _fake_parse(value: str) -> datetime:
        calls.append(value)
        return parsed

    monkeypatch.setattr(dt_util, "parse_datetime", _fake_parse, raising=False)

    coordinator = SimpleNamespace(resolve_boost_end=None)
    state = heater_module.derive_boost_state(
        {"boost_end_datetime": iso_value}, coordinator
    )

    assert calls == [iso_value]
    assert state.end_datetime == parsed
    assert state.end_iso == iso_value


def test_boost_entities_expose_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure boost binary and sensor entities expose derived metadata."""

    base_now = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(dt_util, "now", lambda: base_now)

    expected_end = base_now + timedelta(minutes=30)
    settings = {
        "mode": "boost",
        "boost_end_day": 1,
        "boost_end_min": 30,
        "boost_end_datetime": expected_end,
        "boost_minutes_delta": 30,
    }

    def _resolver(day: int, minute: int, *, now=None) -> tuple[datetime, int]:
        raise AssertionError(
            "resolver should not be used when derived metadata present"
        )

    raw_nodes = [{"type": "acm", "addr": "1", "name": "Accumulator"}]
    inventory = Inventory("dev", build_node_inventory(raw_nodes))
    settings_map = {"acm": {"1": settings}}

    coordinator = SimpleNamespace(
        data={"dev": {"settings": settings_map}},
        resolve_boost_end=_resolver,
    )
    store = DomainStateStore([DomainNodeId(DomainNodeType.ACCUMULATOR, "1")])
    store.apply_full_snapshot("acm", "1", settings)
    coordinator.domain_view = heater_module.DomainStateView("dev", store)

    def _settings_resolver() -> dict[str, Any] | None:
        return settings_map["acm"].get("1")

    boost_binary = HeaterBoostActiveBinarySensor(
        coordinator,
        "entry",
        "dev",
        "acm",
        "1",
        "Accumulator Boost Active",
        f"{DOMAIN}:dev:acm:1:boost_active",
        inventory=inventory,
        settings_resolver=_settings_resolver,
        device_name="Accumulator",
    )
    minutes_sensor = HeaterBoostMinutesRemainingSensor(
        coordinator,
        "entry",
        "dev",
        "1",
        "Accumulator Boost Minutes Remaining",
        f"{DOMAIN}:dev:acm:1:boost:minutes_remaining",
        device_name="Accumulator",
        node_type="acm",
        inventory=inventory,
    )
    end_sensor = HeaterBoostEndSensor(
        coordinator,
        "entry",
        "dev",
        "1",
        "Accumulator Boost End",
        f"{DOMAIN}:dev:acm:1:boost:end",
        device_name="Accumulator",
        node_type="acm",
        inventory=inventory,
    )

    assert boost_binary.is_on is True
    assert minutes_sensor.native_value == 30
    assert end_sensor.native_value == expected_end
    assert end_sensor.state == expected_end.isoformat()

    assert boost_binary.extra_state_attributes["boost_minutes_remaining"] == 30
    assert boost_binary.extra_state_attributes["boost_end"] == expected_end.isoformat()
    assert boost_binary.extra_state_attributes["boost_end_label"] is None
    assert (
        minutes_sensor.extra_state_attributes["boost_end"] == expected_end.isoformat()
    )
    assert minutes_sensor.extra_state_attributes["boost_end_label"] is None
    assert end_sensor.extra_state_attributes["boost_minutes_remaining"] == 30
    assert end_sensor.extra_state_attributes["boost_end_label"] is None


def test_boost_entities_handle_missing_data() -> None:
    """Boost entities should gracefully handle missing settings."""

    raw_nodes = [{"type": "acm", "addr": "1", "name": "Accumulator"}]
    inventory = Inventory("dev", build_node_inventory(raw_nodes))
    settings_map = {"acm": {"1": {}}}

    coordinator = SimpleNamespace(data={"dev": {"settings": settings_map}})

    def _settings_resolver() -> dict[str, Any] | None:
        return settings_map["acm"].get("1")

    boost_binary = HeaterBoostActiveBinarySensor(
        coordinator,
        "entry",
        "dev",
        "acm",
        "1",
        "Accumulator Boost Active",
        f"{DOMAIN}:dev:acm:1:boost_active",
        inventory=inventory,
        settings_resolver=_settings_resolver,
        device_name="Accumulator",
    )
    minutes_sensor = HeaterBoostMinutesRemainingSensor(
        coordinator,
        "entry",
        "dev",
        "1",
        "Accumulator Boost Minutes Remaining",
        f"{DOMAIN}:dev:acm:1:boost:minutes_remaining",
        device_name="Accumulator",
        node_type="acm",
        inventory=inventory,
    )
    end_sensor = HeaterBoostEndSensor(
        coordinator,
        "entry",
        "dev",
        "1",
        "Accumulator Boost End",
        f"{DOMAIN}:dev:acm:1:boost:end",
        device_name="Accumulator",
        node_type="acm",
        inventory=inventory,
    )

    assert boost_binary.is_on is False
    assert minutes_sensor.native_value is None
    assert end_sensor.native_value is None
    assert end_sensor.state == "Never"
    assert boost_binary.extra_state_attributes["boost_end"] is None
    assert boost_binary.extra_state_attributes["boost_end_label"] == "Never"
    assert minutes_sensor.extra_state_attributes["boost_end"] is None
    assert minutes_sensor.extra_state_attributes["boost_end_label"] == "Never"
    assert end_sensor.extra_state_attributes["boost_minutes_remaining"] is None
    assert end_sensor.extra_state_attributes["boost_end_label"] == "Never"


def test_boost_end_sensor_returns_base_state_when_available() -> None:
    coordinator = SimpleNamespace(
        data={
            "dev": {
                "settings": {"acm": {"1": {}}},
            }
        }
    )

    payload: dict[str, Any] = {"nodes": []}
    inventory = Inventory("dev", build_node_inventory(payload))

    sensor = HeaterBoostEndSensor(
        coordinator,
        "entry",
        "dev",
        "1",
        "Accumulator Boost End",
        f"{DOMAIN}:dev:acm:1:boost:end",
        node_type="acm",
        inventory=inventory,
    )

    sensor.boost_state = MagicMock(  # type: ignore[assignment]
        return_value=heater_module.BoostState(
            active=None,
            minutes_remaining=None,
            end_datetime=None,
            end_iso=None,
            end_label=None,
        )
    )
    sensor._attr_state = "ready"

    assert sensor.state == "ready"


def test_boost_end_sensor_handles_isoformat_error() -> None:
    coordinator = SimpleNamespace(
        data={
            "dev": {
                "settings": {"acm": {"1": {}}},
            }
        }
    )

    payload: dict[str, Any] = {"nodes": []}
    inventory = Inventory("dev", build_node_inventory(payload))

    sensor = HeaterBoostEndSensor(
        coordinator,
        "entry",
        "dev",
        "1",
        "Accumulator Boost End",
        f"{DOMAIN}:dev:acm:1:boost:end",
        node_type="acm",
        inventory=inventory,
    )

    def _raise() -> None:
        raise TypeError("bad isoformat")

    faulty = SimpleNamespace(isoformat=_raise)

    sensor.boost_state = MagicMock(  # type: ignore[assignment]
        return_value=heater_module.BoostState(
            active=True,
            minutes_remaining=None,
            end_datetime=faulty,
            end_iso=None,
            end_label="Later",
        )
    )
    sensor._attr_state = STATE_UNKNOWN

    assert sensor.state == STATE_UNKNOWN


def test_coerce_boost_minutes_edge_cases() -> None:
    """Exercise defensive conversions for boost duration inputs."""

    coerce = boost_module.coerce_boost_minutes
    assert coerce(None) is None
    assert coerce(True) is None
    assert coerce(0) is None
    assert coerce(-5) is None
    assert coerce("   ") is None
    assert coerce("invalid") is None
    assert coerce("90") == 90
    assert coerce(120.7) == 120


def test_coerce_boost_minutes_filters_non_positive() -> None:
    """Ensure boost minute coercion rejects falsey and negative values."""

    coerce = boost_module.coerce_boost_minutes
    assert coerce(None) is None
    assert coerce(False) is None
    assert coerce(0) is None
    assert coerce(-1) is None
    assert coerce(" ") is None
    assert coerce("15") == 15


def test_coerce_boost_minutes_non_positive_values() -> None:
    """Ensure boost minute coercion rejects non-positive values."""

    coerce = boost_module.coerce_boost_minutes
    assert coerce(None) is None
    assert coerce(False) is None
    assert coerce(0) is None
    assert coerce(-3) is None
    assert coerce(" ") is None
    assert coerce(45) == 45


def test_boost_runtime_store_handles_non_mapping() -> None:
    """Verify boost runtime store creation tolerates unexpected inputs."""

    assert heater_module._boost_runtime_store(None, create=False) == {}
    entry_data: dict[str, Any] = {}
    assert heater_module._boost_runtime_store(entry_data, create=False) == {}
    created = heater_module._boost_runtime_store(entry_data, create=True)
    assert created == {}
    assert entry_data[heater_module._BOOST_RUNTIME_KEY] is created


def test_iter_nodes_metadata_uses_inventory() -> None:
    """Inventory helper should yield metadata with resolved names."""

    raw_nodes = {
        "nodes": [
            {"type": "htr", "addr": "1", "name": " Living "},
            {"type": "acm", "addr": "2"},
            {"type": "acm", "addr": "3", "name": " Storage "},
            {"type": "pmo", "addr": "99"},
        ]
    }
    inventory = Inventory("dev", build_node_inventory(raw_nodes))

    results = list(
        inventory.iter_nodes_metadata(
            node_types=("htr", "acm"),
            default_name_simple=lambda addr: f"Heater {addr}",
        )
    )

    pairs = sorted((meta.node_type, meta.addr) for meta in results)
    assert pairs == sorted(
        [
            ("htr", "1"),
            ("acm", "2"),
            ("acm", "3"),
        ]
    )
    names = {meta.addr: meta.name for meta in results}
    assert names["1"] == "Living"
    assert names["2"] == "Accumulator 2"
    assert names["3"] == "Storage"
    supports = {meta.addr: boost_module.supports_boost(meta.node) for meta in results}
    assert supports == {"1": False, "2": True, "3": True}


def test_iter_nodes_metadata_uses_inventory_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generator should source node data from inventory iterator."""

    inventory = Inventory("dev", [])
    default_factory = lambda addr: f"Heater {addr}"

    metadata = InventoryNodeMetadata(
        node_type="acm",
        node=SimpleNamespace(addr="2", type="acm", supports_boost=lambda: "true"),
        addr="2",
        name="Resolved acm 2",
    )

    def _fake_iter(
        self: Inventory,
        *,
        node_types: Iterable[str] | None = None,
        default_name_simple: Callable[[str], str] | None = None,
    ) -> Iterator[InventoryNodeMetadata]:
        assert self is inventory
        assert node_types == ("htr", "acm") or set(node_types or []) == {"htr", "acm"}
        assert default_name_simple is default_factory
        yield metadata

    monkeypatch.setattr(Inventory, "iter_nodes_metadata", _fake_iter)

    results = list(
        inventory.iter_nodes_metadata(
            node_types=("htr", "acm"),
            default_name_simple=default_factory,
        )
    )

    assert results == [metadata]
    assert boost_module.supports_boost(results[0].node) is True


def test_boost_runtime_helpers_guard_invalid_structures() -> None:
    """Ensure get/set helpers short-circuit when data is malformed."""

    hass = HomeAssistant()
    entry_id = "entry-invalid"

    # Domain data missing prevents persistence.
    heater_module.set_boost_runtime_minutes(hass, entry_id, "acm", "01", 30)
    assert heater_module.get_boost_runtime_minutes(hass, entry_id, "acm", "01") is None

    # Non-mapping entry data is ignored for get/set operations.
    hass.data = {DOMAIN: {entry_id: []}}
    assert heater_module.get_boost_runtime_minutes(hass, entry_id, "acm", "01") is None
    heater_module.set_boost_runtime_minutes(hass, entry_id, "acm", "01", 45)

    # Missing identifiers or invalid minutes are ignored.
    hass.data[DOMAIN][entry_id] = {}
    heater_module.set_boost_runtime_minutes(hass, entry_id, "", "01", 45)
    heater_module.set_boost_runtime_minutes(hass, entry_id, "acm", "", 45)
    heater_module.set_boost_runtime_minutes(hass, entry_id, "acm", "01", -10)

    # Stored garbage values should not be returned.
    hass.data[DOMAIN][entry_id][heater_module._BOOST_RUNTIME_KEY] = {
        "acm": {"01": "oops"}
    }
    assert heater_module.get_boost_runtime_minutes(hass, entry_id, "acm", "01") is None
    assert (
        heater_module.resolve_boost_runtime_minutes(
            hass,
            entry_id,
            "",
            "",
        )
        == heater_module.DEFAULT_BOOST_DURATION
    )
