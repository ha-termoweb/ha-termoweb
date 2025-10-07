from __future__ import annotations

import datetime as dt

from aiohttp import ClientError
from types import MappingProxyType
from typing import Any, Mapping
from unittest.mock import AsyncMock

import pytest

from homeassistant.core import HomeAssistant

from custom_components.termoweb import boost as boost_module, coordinator as coord_module
from custom_components.termoweb.nodes import AccumulatorNode, HeaterNode


def test_coerce_int_variants() -> None:
    """``coerce_int`` should normalise primitives and guard against errors."""

    class BadString:
        def __str__(self) -> str:
            raise RuntimeError("boom")

    assert boost_module.coerce_int(None) is None
    assert boost_module.coerce_int(True) == 1
    assert boost_module.coerce_int(False) == 0
    assert boost_module.coerce_int(5.7) == 5
    assert boost_module.coerce_int(float("inf")) is None
    assert boost_module.coerce_int(BadString()) is None
    assert boost_module.coerce_int("   ") is None
    assert boost_module.coerce_int(" 7.2 ") == 7


def test_resolve_boost_end_from_fields_variants() -> None:
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
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device={},
        nodes={},
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
) -> None:
    """Fetching RTC time should update the cached reference."""

    hass = HomeAssistant()
    client = AsyncMock()
    base_now = dt.datetime(2024, 5, 1, 12, 0, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(coord_module.dt_util, "now", lambda: base_now)

    client.get_rtc_time = AsyncMock(
        return_value={"y": 2024, "n": 5, "d": 1, "h": 12, "m": 5, "s": 0}
    )

    coordinator = coord_module.StateCoordinator(
        hass,
        client=client,
        base_interval=30,
        dev_id="dev",
        device={},
        nodes={},
    )

    result = await coordinator._async_fetch_rtc_datetime()
    assert result == dt.datetime(2024, 5, 1, 12, 5, 0, tzinfo=dt.timezone.utc)
    assert coordinator._device_now_estimate() is not None


@pytest.mark.asyncio
async def test_async_fetch_rtc_datetime_handles_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RTC fetch helper should swallow client errors and keep reference unset."""

    hass = HomeAssistant()
    client = AsyncMock()
    base_now = dt.datetime(2024, 5, 1, 12, 0, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(coord_module.dt_util, "now", lambda: base_now)

    client.get_rtc_time = AsyncMock(side_effect=ClientError("boom"))

    coordinator = coord_module.StateCoordinator(
        hass,
        client=client,
        base_interval=30,
        dev_id="dev",
        device={},
        nodes={},
    )

    result = await coordinator._async_fetch_rtc_datetime()
    assert result is None
    assert coordinator._device_now_estimate() is None


def test_boost_helpers_guard_against_invalid_sections() -> None:
    """Derived metadata helpers should ignore unsupported payload shapes."""

    hass = HomeAssistant()
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device={},
        nodes={},
    )

    calls: list[tuple[Mapping[str, Any], datetime | None]] = []

    def _record(payload: Mapping[str, Any], *, now: datetime | None) -> None:
        calls.append((payload, now))

    coordinator._apply_accumulator_boost_metadata = _record  # type: ignore[assignment]
    coordinator._apply_boost_metadata_for_section(None, now=None)
    coordinator._apply_boost_metadata_for_section({"settings": []}, now=None)

    assert calls == []
    assert coord_module.StateCoordinator._requires_boost_resolution(None) is False


@pytest.mark.asyncio
async def test_async_update_data_adds_boost_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accumulator settings should expose derived boost metadata."""

    hass = HomeAssistant()
    client = AsyncMock()
    base_now = dt.datetime(2024, 1, 1, 0, 0, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(coord_module.dt_util, "now", lambda: base_now)

    inventory = [AccumulatorNode(name="Accumulator 1", addr="1")]
    coordinator = coord_module.StateCoordinator(
        hass,
        client=client,
        base_interval=30,
        dev_id="dev",
        device={},
        nodes={},
        node_inventory=inventory,
    )

    client.get_node_settings = AsyncMock(
        return_value={"mode": "boost", "boost_end_day": 1, "boost_end_min": 90}
    )
    client.get_rtc_time = AsyncMock(
        return_value={"y": 2024, "n": 1, "d": 1, "h": 0, "m": 0, "s": 0}
    )

    result = await coordinator._async_update_data()
    settings = result["dev"]["nodes_by_type"]["acm"]["settings"]["1"]
    derived_dt = settings.get("boost_end_datetime")
    derived_minutes = settings.get("boost_minutes_delta")

    assert isinstance(derived_dt, dt.datetime)
    assert derived_dt == dt.datetime(2024, 1, 1, 1, 30, tzinfo=dt.timezone.utc)
    assert derived_minutes == 90
    client.get_rtc_time.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_refresh_heater_fetches_rtc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refreshing an accumulator should resolve boost metadata using RTC."""

    hass = HomeAssistant()
    client = AsyncMock()
    base_now = dt.datetime(2024, 1, 1, 0, 0, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(coord_module.dt_util, "now", lambda: base_now)

    inventory = [AccumulatorNode(name="Accumulator 1", addr="1")]
    coordinator = coord_module.StateCoordinator(
        hass,
        client=client,
        base_interval=30,
        dev_id="dev",
        device={},
        nodes={},
        node_inventory=inventory,
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

    assert coord_module._device_display_name({"name": " Device "}, "dev") == "Device"
    assert coord_module._device_display_name({"name": ""}, "dev") == "Device dev"
    assert coord_module._device_display_name({}, "dev") == "Device dev"
    assert coord_module._device_display_name({"name": 1234}, "dev") == "1234"

    proxy_device: MappingProxyType[str, str] = MappingProxyType({"name": " Proxy "})
    assert coord_module._device_display_name(proxy_device, "dev") == "Proxy"


def test_ensure_heater_section_helper() -> None:
    """The helper must reuse existing sections or insert defaults."""

    nodes_by_type: dict[str, dict[str, Any]] = {
        "htr": {"addrs": ["1"], "settings": {"1": {}}}
    }
    existing = coord_module._ensure_heater_section(nodes_by_type, lambda: {})
    assert existing is nodes_by_type["htr"]

    proxy_nodes = MappingProxyType({"addrs": ("2",), "settings": {"2": {}}})
    nodes_by_type = {"htr": proxy_nodes}  # type: ignore[assignment]
    converted = coord_module._ensure_heater_section(nodes_by_type, lambda: {})
    assert converted == {"addrs": ["2"], "settings": {"2": {}}}
    assert nodes_by_type["htr"] == converted

    nodes_by_type = {}
    created = coord_module._ensure_heater_section(
        nodes_by_type,
        lambda: MappingProxyType(
            {"addrs": ("A",), "settings": {"A": {"mode": "auto"}}}
        ),
    )
    assert created == {"addrs": ["A"], "settings": {"A": {"mode": "auto"}}}
    assert nodes_by_type["htr"] == created


def test_mode_and_pending_key_helpers() -> None:
    """Mode normalisation and pending key helpers should validate input."""

    assert coord_module.StateCoordinator._normalize_mode_value(None) is None
    assert (
        coord_module.StateCoordinator._normalize_mode_value("  Auto ") == "auto"
    )
    assert (
        coord_module.StateCoordinator._normalize_mode_value(123) == "123"
    )

    hass = HomeAssistant()
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device={"name": "Device"},
        nodes={},
    )

    assert coordinator._pending_key("", "") is None
    assert coordinator._pending_key("htr", "1") == ("htr", "1")


def test_prune_and_register_pending_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pending settings helpers should drop expired entries and skip invalid keys."""

    hass = HomeAssistant()
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device={"name": "Device"},
        nodes={},
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
) -> None:
    """The pending settings deferral logic should cover all validation branches."""

    hass = HomeAssistant()
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device={"name": "Device"},
        nodes={},
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
) -> None:
    """Heater refresh should defer merging stale payloads while pending."""

    hass = HomeAssistant()
    client = AsyncMock()
    nodes = {"nodes": [{"type": "htr", "addr": "1", "name": "Heater"}]}
    inventory = [HeaterNode(name="Heater", addr="1")]
    coordinator = coord_module.StateCoordinator(
        hass,
        client=client,
        base_interval=30,
        dev_id="dev",
        device={"name": "Device"},
        nodes=nodes,
        node_inventory=inventory,
    )
    initial = {
        "dev": {
            "dev_id": "dev",
            "name": "Device",
            "raw": {"name": "Device"},
            "connected": True,
            "nodes": nodes,
            "nodes_by_type": {
                "htr": {
                    "addrs": ["1"],
                    "settings": {"1": {"mode": "manual", "stemp": "21.0"}},
                }
            },
            "htr": {
                "addrs": ["1"],
                "settings": {"1": {"mode": "manual", "stemp": "21.0"}},
            },
        }
    }
    coordinator.data = initial

    calls: list[Mapping[str, Any]] = []
    original_helper = coord_module._existing_nodes_map

    def recording_helper(source: Mapping[str, Any] | None) -> dict[str, Any]:
        calls.append(source or {})
        return original_helper(source)

    monkeypatch.setattr(coord_module, "_existing_nodes_map", recording_helper)

    client.get_node_settings = AsyncMock(return_value={"mode": "auto", "stemp": "20.0"})
    coordinator.register_pending_setting(
        "htr", "1", mode="manual", stemp=21.0, ttl=60.0
    )

    await coordinator.async_refresh_heater(("htr", "1"))

    settings = coordinator.data["dev"]["nodes_by_type"]["htr"]["settings"]["1"]
    assert settings == {"mode": "manual", "stemp": "21.0"}
    assert ("htr", "1") in coordinator._pending_settings
    assert not calls

    client.get_node_settings.return_value = {"mode": "manual", "stemp": "21.0"}

    await coordinator.async_refresh_heater(("htr", "1"))

    assert ("htr", "1") not in coordinator._pending_settings
    assert client.get_node_settings.await_count == 2
    assert calls
    assert calls[0]["htr"]["settings"]["1"]["mode"] == "manual"


@pytest.mark.asyncio
async def test_poll_skips_pending_settings_merge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Polling should defer merges until pending settings match."""

    hass = HomeAssistant()
    client = AsyncMock()
    nodes = {"nodes": [{"type": "htr", "addr": "1", "name": "Heater"}]}
    inventory = [HeaterNode(name="Heater", addr="1")]
    coordinator = coord_module.StateCoordinator(
        hass,
        client=client,
        base_interval=30,
        dev_id="dev",
        device={"name": "Device"},
        nodes=nodes,
        node_inventory=inventory,
    )
    initial = {
        "dev": {
            "dev_id": "dev",
            "name": "Device",
            "raw": {"name": "Device"},
            "connected": True,
            "nodes": nodes,
            "nodes_by_type": {
                "htr": {
                    "addrs": ["1"],
                    "settings": {"1": {"mode": "manual", "stemp": "21.0"}},
                }
            },
            "htr": {
                "addrs": ["1"],
                "settings": {"1": {"mode": "manual", "stemp": "21.0"}},
            },
        }
    }
    coordinator.data = initial

    calls: list[Mapping[str, Any]] = []
    original_helper = coord_module._existing_nodes_map

    def recording_helper(source: Mapping[str, Any] | None) -> dict[str, Any]:
        calls.append(source or {})
        return original_helper(source)

    monkeypatch.setattr(coord_module, "_existing_nodes_map", recording_helper)

    client.get_node_settings = AsyncMock(return_value={"mode": "auto", "stemp": "20.0"})
    coordinator.register_pending_setting(
        "htr", "1", mode="manual", stemp=21.0, ttl=60.0
    )

    result = await coordinator._async_update_data()

    settings = result["dev"]["nodes_by_type"]["htr"]["settings"]["1"]
    assert settings == {"mode": "manual", "stemp": "21.0"}
    assert ("htr", "1") in coordinator._pending_settings
    assert calls
    assert calls[0] is initial["dev"]

    coordinator.data = result
    client.get_node_settings.return_value = {"mode": "manual", "stemp": "21.0"}

    result_second = await coordinator._async_update_data()

    assert ("htr", "1") not in coordinator._pending_settings
    assert client.get_node_settings.await_count == 2
    settings_second = result_second["dev"]["nodes_by_type"]["htr"]["settings"]["1"]
    assert settings_second == {"mode": "manual", "stemp": "21.0"}
