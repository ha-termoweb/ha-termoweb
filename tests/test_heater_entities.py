from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Iterator

import pytest
from unittest.mock import MagicMock

from conftest import _install_stubs, build_entry_runtime

_install_stubs()

from custom_components.termoweb import boost as boost_module, heater as heater_module
from custom_components.termoweb.entities import heater as entities_heater_module
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
from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.binary_sensor import HeaterBoostActiveBinarySensor
from custom_components.termoweb.sensor import (
    HeaterBoostEndSensor,
    HeaterBoostMinutesRemainingSensor,
)
from homeassistant.const import STATE_UNKNOWN
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

HeaterNodeBase = heater_module.HeaterNodeBase


def _patch_heater_attr(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: Any,
    *,
    raising: bool | None = None,
) -> None:
    """Patch a heater module attribute across shim + entity modules."""

    if raising is None:
        monkeypatch.setattr(heater_module, name, value)
        monkeypatch.setattr(entities_heater_module, name, value)
    else:
        monkeypatch.setattr(heater_module, name, value, raising=raising)
        monkeypatch.setattr(entities_heater_module, name, value, raising=raising)


def test_heater_node_base_normalizes_address(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, dict]] = []

    original = heater_module.normalize_node_addr

    def _record_normalize(value, **kwargs):
        calls.append((value, kwargs))
        return original(value, **kwargs)

    _patch_heater_attr(monkeypatch, "normalize_node_addr", _record_normalize)

    coordinator = SimpleNamespace(hass=None)
    heater = HeaterNodeBase(coordinator, "entry", "dev", " 01 ", " Heater 01 ")

    assert heater._addr == "01"
    assert heater.device_info["identifiers"] == {(heater_module.DOMAIN, "dev", "01")}
    assert heater._attr_unique_id == f"{heater_module.DOMAIN}:dev:htr:01"
    assert calls == [(" 01 ", {})]


def test_heater_async_will_remove_without_listener_resets_unsub() -> None:
    heater = HeaterNodeBase(SimpleNamespace(hass=None), "entry", "dev", "1", "Heater 1")
    unsub = MagicMock()
    heater._async_unsub_coordinator_update = unsub

    asyncio.run(heater.async_will_remove_from_hass())

    unsub.assert_called_once_with()
    assert getattr(heater, "_async_unsub_coordinator_update") is None


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

    runtime = build_entry_runtime(
        entry_id="entry",
        dev_id="dev-1",
        allow_missing_inventory=True,
    )
    runtime.inventory = None  # type: ignore[assignment]
    with pytest.raises(ValueError):
        heater_module.heater_platform_details_for_entry(
            runtime,
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
    settings = {"boost_end_day": 0, "boost_end_min": 0}

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
    hass = HomeAssistant()
    runtime = build_entry_runtime(
        hass=hass,
        entry_id="entry",
        dev_id="dev",
    )
    assert heater_module._boost_runtime_store(runtime, create=False) == {}
    created = heater_module._boost_runtime_store(runtime, create=True)
    assert created == {}
    assert runtime.boost_runtime is created


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

    runtime = build_entry_runtime(
        hass=hass,
        entry_id=entry_id,
        dev_id="dev",
    )

    # Missing identifiers or invalid minutes are ignored.
    heater_module.set_boost_runtime_minutes(hass, entry_id, "", "01", 45)
    heater_module.set_boost_runtime_minutes(hass, entry_id, "acm", "", 45)
    heater_module.set_boost_runtime_minutes(hass, entry_id, "acm", "01", -10)
    assert runtime.boost_runtime == {}

    # Stored garbage values should not be returned.
    runtime.boost_runtime = {"acm": {"01": "oops"}}
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


# ---------------------------------------------------------------------------
# Coverage expansion: HeaterPlatformDetails properties
# ---------------------------------------------------------------------------


def test_heater_platform_details_addrs_by_type() -> None:
    """addrs_by_type should return addresses grouped by node type."""
    raw_nodes = [
        {"type": "htr", "addr": "1", "name": "Living"},
        {"type": "htr", "addr": "2", "name": "Bedroom"},
        {"type": "acm", "addr": "3"},
    ]
    inventory = Inventory("dev", build_node_inventory(raw_nodes))
    details = heater_module.HeaterPlatformDetails(
        inventory=inventory,
        default_name_simple=lambda addr: f"Heater {addr}",
    )

    addrs = details.addrs_by_type
    assert "htr" in addrs
    assert set(addrs["htr"]) == {"1", "2"}
    assert "acm" in addrs
    assert addrs["acm"] == ["3"]


def test_heater_platform_details_resolve_name() -> None:
    """resolve_name should delegate to inventory.resolve_heater_name."""
    raw_nodes = [{"type": "htr", "addr": "1", "name": "Living Room"}]
    inventory = Inventory("dev", build_node_inventory(raw_nodes))
    details = heater_module.HeaterPlatformDetails(
        inventory=inventory,
        default_name_simple=lambda addr: f"Heater {addr}",
    )
    assert details.resolve_name("htr", "1") == "Living Room"


# ---------------------------------------------------------------------------
# Coverage expansion: Boost temperature storage
# ---------------------------------------------------------------------------


def test_boost_temperature_store_create_false() -> None:
    """_boost_temperature_store should return empty when create=False."""
    hass = HomeAssistant()
    runtime = build_entry_runtime(hass=hass, entry_id="entry-t", dev_id="dev-t")
    store = entities_heater_module._boost_temperature_store(runtime, create=False)
    assert store == {}


def test_boost_temperature_store_create_true() -> None:
    """_boost_temperature_store should return runtime store when create=True."""
    hass = HomeAssistant()
    runtime = build_entry_runtime(hass=hass, entry_id="entry-t", dev_id="dev-t")
    store = entities_heater_module._boost_temperature_store(runtime, create=True)
    assert store is runtime.boost_temperature


def test_get_boost_temperature_roundtrip() -> None:
    """get/set boost temperature should roundtrip correctly."""
    hass = HomeAssistant()
    entry_id = "entry-bt"
    runtime = build_entry_runtime(hass=hass, entry_id=entry_id, dev_id="dev-bt")

    assert heater_module.get_boost_temperature(hass, entry_id, "acm", "01") is None

    heater_module.set_boost_temperature(hass, entry_id, "acm", "01", 25.5)
    assert heater_module.get_boost_temperature(hass, entry_id, "ACM", " 01 ") == 25.5


def test_get_boost_temperature_missing_runtime() -> None:
    """get_boost_temperature should return None when runtime is absent."""
    hass = HomeAssistant()
    assert heater_module.get_boost_temperature(hass, "missing", "acm", "01") is None


def test_set_boost_temperature_missing_runtime() -> None:
    """set_boost_temperature should silently return when runtime is absent."""
    hass = HomeAssistant()
    heater_module.set_boost_temperature(hass, "missing", "acm", "01", 20.0)
    # No exception raised


def test_get_boost_temperature_invalid_identifiers() -> None:
    """get_boost_temperature should return None for empty identifiers."""
    hass = HomeAssistant()
    runtime = build_entry_runtime(hass=hass, entry_id="entry-inv", dev_id="dev")
    assert heater_module.get_boost_temperature(hass, "entry-inv", "", "01") is None
    assert heater_module.get_boost_temperature(hass, "entry-inv", "acm", "") is None


def test_set_boost_temperature_invalid_identifiers() -> None:
    """set_boost_temperature should silently return for empty identifiers."""
    hass = HomeAssistant()
    runtime = build_entry_runtime(hass=hass, entry_id="entry-inv2", dev_id="dev")
    heater_module.set_boost_temperature(hass, "entry-inv2", "", "01", 20.0)
    heater_module.set_boost_temperature(hass, "entry-inv2", "acm", "", 20.0)
    assert runtime.boost_temperature == {}


def test_get_boost_temperature_non_mapping_bucket() -> None:
    """get_boost_temperature should return None when bucket is not a mapping."""
    hass = HomeAssistant()
    runtime = build_entry_runtime(hass=hass, entry_id="entry-nm", dev_id="dev")
    runtime.boost_temperature = {"acm": "not-a-dict"}  # type: ignore[assignment]
    assert heater_module.get_boost_temperature(hass, "entry-nm", "acm", "01") is None


def test_resolve_boost_temperature() -> None:
    """resolve_boost_temperature should return stored or default."""
    hass = HomeAssistant()
    entry_id = "entry-rt"
    runtime = build_entry_runtime(hass=hass, entry_id=entry_id, dev_id="dev")

    assert heater_module.resolve_boost_temperature(hass, entry_id, "acm", "01") is None
    assert heater_module.resolve_boost_temperature(
        hass, entry_id, "acm", "01", default=22.0
    ) == 22.0

    heater_module.set_boost_temperature(hass, entry_id, "acm", "01", 30.0)
    assert heater_module.resolve_boost_temperature(hass, entry_id, "acm", "01") == 30.0


# ---------------------------------------------------------------------------
# Coverage expansion: Climate entity ID storage
# ---------------------------------------------------------------------------


def test_climate_entity_store_create_false() -> None:
    """_climate_entity_store should return empty when create=False."""
    hass = HomeAssistant()
    runtime = build_entry_runtime(hass=hass, entry_id="entry-c", dev_id="dev")
    store = entities_heater_module._climate_entity_store(runtime, create=False)
    assert store == {}


def test_register_climate_entity_id_roundtrip() -> None:
    """register/resolve climate entity ID should roundtrip."""
    hass = HomeAssistant()
    entry_id = "entry-climate"
    runtime = build_entry_runtime(hass=hass, entry_id=entry_id, dev_id="dev")

    assert heater_module.resolve_climate_entity_id(hass, entry_id, "htr", "01") is None

    heater_module.register_climate_entity_id(
        hass, entry_id, "htr", "01", "climate.heater_01"
    )
    assert (
        heater_module.resolve_climate_entity_id(hass, entry_id, "HTR", " 01 ")
        == "climate.heater_01"
    )


def test_clear_climate_entity_id() -> None:
    """clear_climate_entity_id should remove the stored entity ID."""
    hass = HomeAssistant()
    entry_id = "entry-clear"
    runtime = build_entry_runtime(hass=hass, entry_id=entry_id, dev_id="dev")

    heater_module.register_climate_entity_id(
        hass, entry_id, "htr", "01", "climate.heater_01"
    )
    assert heater_module.resolve_climate_entity_id(hass, entry_id, "htr", "01") is not None

    heater_module.clear_climate_entity_id(hass, entry_id, "htr", "01")
    assert heater_module.resolve_climate_entity_id(hass, entry_id, "htr", "01") is None


def test_register_climate_entity_id_empty_entity_id() -> None:
    """register_climate_entity_id should do nothing when entity_id is empty."""
    hass = HomeAssistant()
    entry_id = "entry-empty-eid"
    runtime = build_entry_runtime(hass=hass, entry_id=entry_id, dev_id="dev")

    heater_module.register_climate_entity_id(hass, entry_id, "htr", "01", "")
    heater_module.register_climate_entity_id(hass, entry_id, "htr", "01", None)
    assert runtime.climate_entities == {}


def test_register_climate_entity_id_missing_runtime() -> None:
    """register_climate_entity_id should silently return on missing runtime."""
    hass = HomeAssistant()
    heater_module.register_climate_entity_id(hass, "missing", "htr", "01", "climate.x")
    # No error raised


def test_clear_climate_entity_id_missing_runtime() -> None:
    """clear_climate_entity_id should silently return on missing runtime."""
    hass = HomeAssistant()
    heater_module.clear_climate_entity_id(hass, "missing", "htr", "01")
    # No error raised


def test_resolve_climate_entity_id_missing_runtime() -> None:
    """resolve_climate_entity_id should return None on missing runtime."""
    hass = HomeAssistant()
    assert heater_module.resolve_climate_entity_id(hass, "missing", "htr", "01") is None


def test_register_climate_entity_id_invalid_identifiers() -> None:
    """register_climate_entity_id should skip empty type/addr."""
    hass = HomeAssistant()
    entry_id = "entry-inv-clim"
    runtime = build_entry_runtime(hass=hass, entry_id=entry_id, dev_id="dev")

    heater_module.register_climate_entity_id(hass, entry_id, "", "01", "climate.x")
    heater_module.register_climate_entity_id(hass, entry_id, "htr", "", "climate.x")
    assert runtime.climate_entities == {}


def test_clear_climate_entity_id_invalid_identifiers() -> None:
    """clear_climate_entity_id should skip empty type/addr."""
    hass = HomeAssistant()
    entry_id = "entry-clr-inv"
    runtime = build_entry_runtime(hass=hass, entry_id=entry_id, dev_id="dev")

    heater_module.register_climate_entity_id(
        hass, entry_id, "htr", "01", "climate.x"
    )
    # These should be no-ops
    heater_module.clear_climate_entity_id(hass, entry_id, "", "01")
    heater_module.clear_climate_entity_id(hass, entry_id, "htr", "")
    assert heater_module.resolve_climate_entity_id(hass, entry_id, "htr", "01") == "climate.x"


def test_resolve_climate_entity_id_invalid_identifiers() -> None:
    """resolve_climate_entity_id should return None for empty type/addr."""
    hass = HomeAssistant()
    entry_id = "entry-res-inv"
    runtime = build_entry_runtime(hass=hass, entry_id=entry_id, dev_id="dev")
    assert heater_module.resolve_climate_entity_id(hass, entry_id, "", "01") is None
    assert heater_module.resolve_climate_entity_id(hass, entry_id, "htr", "") is None


def test_resolve_climate_entity_id_non_mapping_bucket() -> None:
    """resolve_climate_entity_id should return None for non-mapping bucket."""
    hass = HomeAssistant()
    entry_id = "entry-nm-clim"
    runtime = build_entry_runtime(hass=hass, entry_id=entry_id, dev_id="dev")
    runtime.climate_entities = {"htr": "not-a-dict"}  # type: ignore[assignment]
    assert heater_module.resolve_climate_entity_id(hass, entry_id, "htr", "01") is None


def test_clear_climate_entity_id_non_mapping_bucket() -> None:
    """clear_climate_entity_id should handle non-mapping bucket gracefully."""
    hass = HomeAssistant()
    entry_id = "entry-nm-clr"
    runtime = build_entry_runtime(hass=hass, entry_id=entry_id, dev_id="dev")
    runtime.climate_entities = {"htr": "not-a-dict"}  # type: ignore[assignment]
    heater_module.clear_climate_entity_id(hass, entry_id, "htr", "01")
    # Should not raise


# ---------------------------------------------------------------------------
# Coverage expansion: iter_boostable_heater_nodes
# ---------------------------------------------------------------------------


def test_iter_boostable_heater_nodes_filters_by_type() -> None:
    """iter_boostable_heater_nodes should filter by node_types argument."""
    raw_nodes = [
        {"type": "htr", "addr": "1", "name": "Heater"},
        {"type": "acm", "addr": "2", "name": "Accumulator"},
    ]
    inventory = Inventory("dev", build_node_inventory(raw_nodes))
    details = heater_module.HeaterPlatformDetails(
        inventory=inventory,
        default_name_simple=lambda addr: f"Node {addr}",
    )

    # Only request acm type
    results = list(heater_module.iter_boostable_heater_nodes(details, node_types=["acm"]))
    assert len(results) == 1
    assert results[0][0] == "acm"


def test_iter_boostable_heater_nodes_string_node_types() -> None:
    """iter_boostable_heater_nodes should handle a single string node_types."""
    raw_nodes = [
        {"type": "acm", "addr": "1"},
    ]
    inventory = Inventory("dev", build_node_inventory(raw_nodes))
    details = heater_module.HeaterPlatformDetails(
        inventory=inventory,
        default_name_simple=lambda addr: f"Node {addr}",
    )

    results = list(heater_module.iter_boostable_heater_nodes(details, node_types="acm"))
    assert len(results) == 1
    assert results[0][0] == "acm"


def test_iter_boostable_heater_nodes_accumulators_only() -> None:
    """accumulators_only should only yield acm type nodes."""
    raw_nodes = [
        {"type": "htr", "addr": "1"},
        {"type": "acm", "addr": "2"},
    ]
    inventory = Inventory("dev", build_node_inventory(raw_nodes))
    details = heater_module.HeaterPlatformDetails(
        inventory=inventory,
        default_name_simple=lambda addr: f"Node {addr}",
    )

    results = list(heater_module.iter_boostable_heater_nodes(details, accumulators_only=True))
    # Only acm nodes should appear (htr doesn't have boost)
    for r in results:
        assert r[0] == "acm"


# ---------------------------------------------------------------------------
# Coverage expansion: log_skipped_nodes
# ---------------------------------------------------------------------------


def test_log_skipped_nodes_with_inventory(caplog: pytest.LogCaptureFixture) -> None:
    """log_skipped_nodes should log skipped addresses."""
    caplog.set_level("DEBUG")
    raw_nodes = [
        {"type": "thm", "addr": "T1"},
        {"type": "htr", "addr": "1"},
    ]
    inventory = Inventory("dev", build_node_inventory(raw_nodes))

    heater_module.log_skipped_nodes(
        "sensor",
        inventory,
        skipped_types=("thm",),
    )
    assert "thm" in caplog.text
    assert "T1" in caplog.text


def test_log_skipped_nodes_empty_platform_name(caplog: pytest.LogCaptureFixture) -> None:
    """log_skipped_nodes should handle empty platform name."""
    caplog.set_level("DEBUG")
    raw_nodes = [{"type": "pmo", "addr": "01"}]
    inventory = Inventory("dev", build_node_inventory(raw_nodes))

    heater_module.log_skipped_nodes("", inventory, skipped_types=("pmo",))
    assert "platform" in caplog.text


def test_log_skipped_nodes_with_details(caplog: pytest.LogCaptureFixture) -> None:
    """log_skipped_nodes should accept HeaterPlatformDetails."""
    caplog.set_level("DEBUG")
    raw_nodes = [{"type": "pmo", "addr": "01"}]
    inventory = Inventory("dev", build_node_inventory(raw_nodes))
    details = heater_module.HeaterPlatformDetails(
        inventory=inventory,
        default_name_simple=lambda addr: addr,
    )
    heater_module.log_skipped_nodes("climate", details, skipped_types=("pmo",))
    assert "pmo" in caplog.text


def test_log_skipped_nodes_none_inventory() -> None:
    """log_skipped_nodes should return early when inventory is None."""
    heater_module.log_skipped_nodes("sensor", None, skipped_types=("thm",))
    # No error raised


# ---------------------------------------------------------------------------
# Coverage expansion: build_settings_resolver
# ---------------------------------------------------------------------------


def test_build_settings_resolver_with_domain_view() -> None:
    """Settings resolver should read from domain view."""
    from custom_components.termoweb.domain import DomainStateStore
    from custom_components.termoweb.domain.ids import NodeId as DomainNodeId, NodeType as DomainNodeType
    from custom_components.termoweb.domain.view import DomainStateView

    store = DomainStateStore([DomainNodeId(DomainNodeType.HEATER, "01")])
    store.apply_full_snapshot("htr", "01", {"mode": "comfort"})
    view = DomainStateView("dev", store)
    coordinator = SimpleNamespace(domain_view=view)

    resolver = heater_module.build_settings_resolver(coordinator, "dev", "htr", "01")
    state = resolver()
    assert state is not None
    assert state.mode == "comfort"


def test_build_settings_resolver_no_domain_view() -> None:
    """Settings resolver should return None without domain view."""
    coordinator = SimpleNamespace(domain_view=None)
    resolver = heater_module.build_settings_resolver(coordinator, "dev", "htr", "01")
    assert resolver() is None


# ---------------------------------------------------------------------------
# Coverage expansion: HeaterNodeBase edge cases
# ---------------------------------------------------------------------------


def test_heater_node_base_thermostat_state() -> None:
    """thermostat_state should return ThermostatState when available."""
    from custom_components.termoweb.domain.state import ThermostatState, DomainStateStore
    from custom_components.termoweb.domain.ids import NodeId as DomainNodeId, NodeType as DomainNodeType

    raw_nodes = [{"type": "thm", "addr": "T1"}]
    inventory = Inventory("dev", build_node_inventory(raw_nodes))
    store = DomainStateStore([DomainNodeId(DomainNodeType.THERMOSTAT, "T1")])
    store.apply_full_snapshot("thm", "T1", {"batt_level": 4})
    view = heater_module.DomainStateView("dev", store)

    coordinator = SimpleNamespace(
        data={"dev": {"settings": {"thm": {"T1": {"batt_level": 4}}}}},
        domain_view=view,
        inventory=inventory,
    )
    heater = HeaterNodeBase(
        coordinator, "entry", "dev", "T1", None,
        node_type="thm", inventory=inventory,
    )
    ts = heater.thermostat_state()
    assert isinstance(ts, ThermostatState)


def test_heater_node_base_power_monitor_state() -> None:
    """power_monitor_state should return None when not a power monitor."""
    coordinator = SimpleNamespace(data={}, domain_view=None)
    heater = HeaterNodeBase(coordinator, "entry", "dev", "01", "Heater")
    assert heater.power_monitor_state() is None


def test_heater_node_base_units_default() -> None:
    """_units should default to 'C' when state units is absent."""
    coordinator = SimpleNamespace(data={})
    heater = HeaterNodeBase(coordinator, "entry", "dev", "01", "Heater")
    assert heater._units() == "C"


def test_heater_node_base_units_fahrenheit() -> None:
    """_units should return 'F' when state has units='F'."""
    from custom_components.termoweb.domain.state import DomainStateStore
    from custom_components.termoweb.domain.ids import NodeId as DomainNodeId, NodeType as DomainNodeType

    raw_nodes = [{"type": "htr", "addr": "01"}]
    inventory = Inventory("dev", build_node_inventory(raw_nodes))
    store = DomainStateStore([DomainNodeId(DomainNodeType.HEATER, "01")])
    store.apply_full_snapshot("htr", "01", {"units": "F"})
    view = heater_module.DomainStateView("dev", store)

    coordinator = SimpleNamespace(domain_view=view, inventory=inventory)
    heater = HeaterNodeBase(
        coordinator, "entry", "dev", "01", "Heater",
        node_type="htr", inventory=inventory,
    )
    assert heater._units() == "F"


def test_heater_node_base_units_unknown() -> None:
    """_units should default to 'C' for unknown unit values."""
    from custom_components.termoweb.domain.state import DomainStateStore
    from custom_components.termoweb.domain.ids import NodeId as DomainNodeId, NodeType as DomainNodeType

    raw_nodes = [{"type": "htr", "addr": "01"}]
    inventory = Inventory("dev", build_node_inventory(raw_nodes))
    store = DomainStateStore([DomainNodeId(DomainNodeType.HEATER, "01")])
    store.apply_full_snapshot("htr", "01", {"units": "K"})
    view = heater_module.DomainStateView("dev", store)

    coordinator = SimpleNamespace(domain_view=view, inventory=inventory)
    heater = HeaterNodeBase(
        coordinator, "entry", "dev", "01", "Heater",
        node_type="htr", inventory=inventory,
    )
    assert heater._units() == "C"


# ---------------------------------------------------------------------------
# Coverage expansion: heater_platform_details_for_entry non-standard inventory
# ---------------------------------------------------------------------------


def test_heater_platform_details_non_standard_inventory(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Non-Inventory objects with correct attributes should be accepted with warning."""
    caplog.set_level("ERROR")

    class FakeInventory:
        nodes_by_type = {}
        heater_address_map = ({}, {})

        def resolve_heater_name(self, *a, **k):
            return "name"

        def iter_heater_platform_metadata(self, *a, **k):
            return iter([])

    runtime = build_entry_runtime(
        entry_id="entry-ns",
        dev_id="dev-ns",
        allow_missing_inventory=True,
    )
    runtime.inventory = FakeInventory()  # type: ignore[assignment]

    details = heater_module.heater_platform_details_for_entry(
        runtime,
        default_name_simple=lambda addr: addr,
    )
    assert details is not None
    assert "non-standard inventory" in caplog.text


# ---------------------------------------------------------------------------
# Coverage expansion: derive_boost_state edge - boost_end_iso without dt
# ---------------------------------------------------------------------------


def test_derive_boost_state_iso_without_datetime(monkeypatch: pytest.MonkeyPatch) -> None:
    """When boost_end_iso is set but boost_end_dt is None, should parse it."""
    base_now = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(dt_util, "now", lambda: base_now)

    # Use a coordinator with isoformat that works but force ISO-only path
    settings = {
        "boost_active": True,
        "boost_end_datetime": datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc),
    }
    state = heater_module.derive_boost_state(settings, SimpleNamespace())
    assert state.end_datetime is not None
    assert state.end_iso is not None
