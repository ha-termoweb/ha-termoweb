"""Tests for the frames-heard sensor of listen-only radio entries."""

from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace
from typing import Any

import pytest
from conftest import _install_stubs, build_entry_runtime

_install_stubs()

from custom_components.termoweb.const import DOMAIN, signal_radio_frames
from custom_components.termoweb.inventory import Inventory
from custom_components.termoweb.runtime import require_runtime
from homeassistant.core import HomeAssistant

DEV_ID = "aabbcc001122"


def _setup(brand: str) -> tuple[HomeAssistant, Any]:
    hass = HomeAssistant()
    hass.data = {DOMAIN: {}}
    entry = SimpleNamespace(entry_id=f"entry-{brand}")
    build_entry_runtime(
        hass=hass,
        entry_id=entry.entry_id,
        dev_id=DEV_ID,
        inventory=Inventory(DEV_ID, []),
        energy_coordinator=SimpleNamespace(async_add_listener=lambda *_a: None),
        brand=brand,
    )
    return hass, entry


@pytest.mark.asyncio
async def test_monitor_entry_gets_only_the_frames_sensor() -> None:
    """A listen-only entry has no heater or energy sensors, only frames heard."""

    module = importlib.import_module("custom_components.termoweb.sensor")
    added: list[Any] = []
    hass, entry = _setup("radio_monitor")
    await module.async_setup_entry(hass, entry, added.extend)
    assert [type(entity).__name__ for entity in added] == ["RadioFramesSensor"]

    added.clear()
    hass, entry = _setup("radio")
    await module.async_setup_entry(hass, entry, added.extend)
    assert "RadioFramesSensor" not in [type(entity).__name__ for entity in added]


def test_frames_sensor_follows_the_monitor(monkeypatch: pytest.MonkeyPatch) -> None:
    """The count and last frame time come from the monitor's dispatcher signal."""

    module = importlib.import_module("custom_components.termoweb.sensor")
    hass, entry = _setup("radio_monitor")
    sensor = module.RadioFramesSensor(entry.entry_id, DEV_ID)
    sensor.hass = hass
    writes: list[Any] = []
    sensor.async_write_ha_state = lambda: writes.append(sensor._attr_native_value)
    removers: list[Any] = []
    sensor.async_on_remove = removers.append

    assert sensor._attr_unique_id == f"{DOMAIN}:{DEV_ID}:frames_heard"
    assert sensor._attr_native_value == 0
    assert sensor._attr_extra_state_attributes == {"last_frame": None}
    assert sensor._attr_translation_key == "frames_heard"
    assert sensor.device_info["identifiers"] == {(DOMAIN, DEV_ID)}

    connected: list[tuple[str, Any]] = []

    def connect(target: Any, signal: str, callback: Any) -> Any:
        assert target is hass
        connected.append((signal, callback))
        return lambda: connected.clear()

    monkeypatch.setattr(module, "async_dispatcher_connect", connect)
    asyncio.run(sensor.async_added_to_hass())
    assert len(removers) == 1
    ((signal, handle),) = connected
    assert signal == signal_radio_frames(entry.entry_id)
    handle({"frames": 7, "last_frame": "2026-01-02T03:04:05+00:00"})
    assert writes == [7]
    assert sensor._attr_extra_state_attributes == {
        "last_frame": "2026-01-02T03:04:05+00:00"
    }
    removers[0]()
    assert connected == []


@pytest.mark.asyncio
@pytest.mark.parametrize("brand", ["radio_monitor", "radio"])
async def test_no_device_points_via_a_device_that_is_never_created(brand) -> None:
    """Every via_device of an entry's entities names a device the entry creates."""

    sensors = importlib.import_module("custom_components.termoweb.sensor")
    binary = importlib.import_module(
        "custom_components.termoweb.binary_sensor"
    )
    added: list[Any] = []
    hass, entry = _setup(brand)
    await binary.async_setup_entry(hass, entry, added.extend)
    await sensors.async_setup_entry(hass, entry, added.extend)
    for entity in added:
        entity.hass = hass
    termoweb = importlib.import_module("custom_components.termoweb")
    termoweb._register_hub_devices(hass, entry, require_runtime(hass, entry.entry_id))
    registry = importlib.import_module("homeassistant.helpers.device_registry")
    dev_reg = registry.async_get(hass)
    infos = [entity.device_info for entity in added]
    created = {
        dev_reg.async_get_or_create(config_entry_id=entry.entry_id, **info).id
        for info in infos
    }
    for info in infos:
        assert "via_device" not in info
        assert info.get("via_device_id") in created | {None}
        assert info["manufacturer"] == "Radio"
    gateway = next(i for i in infos if (DOMAIN, DEV_ID) in i["identifiers"])
    site = dev_reg.async_get_device_by_identifier(
        (DOMAIN, DEV_ID, "site"), entry.entry_id
    )
    if brand == "radio_monitor":
        assert "via_device_id" not in gateway
        assert site is None
    else:
        assert gateway["via_device_id"] == site.id
