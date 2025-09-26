from __future__ import annotations

import asyncio
from types import SimpleNamespace

from conftest import _install_stubs

_install_stubs()

from custom_components.termoweb import heater as heater_module
from homeassistant.core import HomeAssistant

HeaterNodeBase = heater_module.HeaterNodeBase
build_heater_name_map = heater_module.build_heater_name_map


def _make_heater(coordinator: SimpleNamespace) -> HeaterNodeBase:
    return HeaterNodeBase(coordinator, "entry", "dev", "A", "Heater A")


def test_build_heater_name_map_handles_invalid_entries() -> None:
    nodes = {
        "nodes": [
            123,
            {"type": "HTR", "addr": None, "name": "Ignored"},
            {"type": "foo", "addr": "B", "name": "Skip"},
            {"type": "htr", "addr": 5, "name": "  "},
            {"type": "htr", "addr": "6", "name": None},
        ]
    }

    result = build_heater_name_map(nodes, lambda addr: f"Heater {addr}")

    assert result == {"5": "Heater 5", "6": "Heater 6"}


def test_heater_base_async_added_without_hass() -> None:
    async def _run() -> None:
        coordinator = SimpleNamespace(hass=None)
        heater = _make_heater(coordinator)

        assert heater.hass is None
        await heater.async_added_to_hass()
        assert heater._unsub_ws is None

    asyncio.run(_run())


def test_device_available_requires_nodes_section() -> None:
    coordinator = SimpleNamespace(hass=None)
    heater = _make_heater(coordinator)

    assert not heater._device_available(None)
    assert not heater._device_available({})
    assert heater._device_available({"nodes": []})


class _FakeDict(dict):
    """Dictionary that exposes a non-callable ``get`` attribute."""

    get = "not-callable"


def test_device_record_fallback_dict() -> None:
    hass = HomeAssistant()
    coordinator = SimpleNamespace(
        hass=hass, data=_FakeDict({"dev": {"nodes": "ok"}})
    )
    heater = _make_heater(coordinator)

    assert heater._device_record() == {"nodes": "ok"}


def test_device_record_unknown_structure() -> None:
    hass = HomeAssistant()
    coordinator = SimpleNamespace(hass=hass, data=SimpleNamespace())
    heater = _make_heater(coordinator)

    assert heater._device_record() is None


def test_heater_section_handles_missing_device() -> None:
    hass = HomeAssistant()
    coordinator = SimpleNamespace(hass=hass, data={})
    heater = _make_heater(coordinator)

    assert heater._heater_section() == {}


def test_heater_settings_missing_mapping() -> None:
    hass = HomeAssistant()
    coordinator = SimpleNamespace(
        hass=hass, data={"dev": {"htr": {"settings": []}}}
    )
    heater = _make_heater(coordinator)

    assert heater.heater_settings() is None
