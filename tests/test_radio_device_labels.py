"""Radio entries name their devices after what they really are, on real Home Assistant."""

from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.termoweb.const import DOMAIN
from tests.fakes.radio_link import MAC
from tests.fakes.sensors import fake_radio_links, radio_entry, setup_entry

DEV_ID = MAC.replace(":", "").lower()
HEATER = {"type": "htr", "addr": "6", "name": "Heater 6"}


def _nanocul_entry() -> MockConfigEntry:
    """Return a nanoCUL (USB stick) radio entry."""
    return MockConfigEntry(
        domain=DOMAIN,
        minor_version=2,
        unique_id="radio:nanocul-test",
        data={
            "brand": "radio",
            "radio_type": "nanocul",
            "device": "socket://192.0.2.10:2323",
            "radio_device_id": DEV_ID,
            "dialect": "B",
            "network_id": "1234",
            "nodes": [HEATER],
        },
    )


def _devices(hass: HomeAssistant, entry: MockConfigEntry) -> dict[str, Any]:
    """Return the entry's devices keyed by model (gateway model varies, so by role)."""
    registry = dr.async_get(hass)
    return {
        "gateway": registry.async_get_device_by_identifier(
            (DOMAIN, DEV_ID), entry.entry_id
        ),
        "site": registry.async_get_device_by_identifier(
            (DOMAIN, DEV_ID, "site"), entry.entry_id
        ),
        "heater": registry.async_get_device_by_identifier(
            (DOMAIN, DEV_ID, "6"), entry.entry_id
        ),
    }


async def _unload(hass: HomeAssistant, entry: MockConfigEntry) -> None:
    """Unload the entry."""
    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()


async def test_esp32_entry_labels(hass: HomeAssistant) -> None:
    """An ESP32 entry: ESPHome gateway, TermoWeb heater, site named Home."""
    entry = radio_entry("radio", [HEATER])
    with fake_radio_links():
        await setup_entry(hass, entry)
        devices = _devices(hass, entry)
        assert (devices["gateway"].name, devices["gateway"].manufacturer) == (
            "Radio gateway",
            "ESPHome",
        )
        assert (devices["site"].name, devices["site"].manufacturer) == (
            "Home",
            "TermoWeb",
        )
        assert devices["heater"].manufacturer == "TermoWeb"
        await _unload(hass, entry)


async def test_nanocul_entry_labels(hass: HomeAssistant) -> None:
    """A nanoCUL entry: nanoCUL stick gateway, TermoWeb heater, site named Home."""
    entry = _nanocul_entry()
    with fake_radio_links():
        await setup_entry(hass, entry)
        devices = _devices(hass, entry)
        assert (devices["gateway"].name, devices["gateway"].manufacturer) == (
            "nanoCUL stick",
            "nanoCUL",
        )
        assert (devices["site"].name, devices["site"].manufacturer) == (
            "Home",
            "TermoWeb",
        )
        assert devices["heater"].manufacturer == "TermoWeb"
        await _unload(hass, entry)


async def test_existing_default_names_update_and_user_renames_stay(
    hass: HomeAssistant,
) -> None:
    """Old default names and labels are refreshed; a user's own name is kept."""
    entry = radio_entry("radio", [HEATER])
    entry.add_to_hass(hass)
    registry = dr.async_get(hass)
    old_site = registry.async_get_or_create(
        config_entry_id=entry.entry_id,
        identifiers={(DOMAIN, DEV_ID, "site")},
        name="Radio gateway",
        manufacturer="Radio",
        model="Site",
    )
    registry.async_get_or_create(
        config_entry_id=entry.entry_id,
        identifiers={(DOMAIN, DEV_ID)},
        name="Radio Gateway",
        manufacturer="Radio",
        model="Gateway/Controller",
    )
    registry.async_update_device(old_site.id, name_by_user="My house")

    with fake_radio_links():
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        devices = _devices(hass, entry)
        assert devices["site"].id == old_site.id
        assert devices["site"].name == "Home"
        assert devices["site"].name_by_user == "My house"
        assert devices["site"].manufacturer == "TermoWeb"
        assert devices["gateway"].name == "Radio gateway"
        assert devices["gateway"].manufacturer == "ESPHome"
        await _unload(hass, entry)
