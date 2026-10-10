"""Tests for device info helpers and the site device (utils.py)."""

from __future__ import annotations

from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr, entity_registry as er
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.domain.state import GeoData
from custom_components.termoweb.utils import (
    async_get_integration_version,
    build_gateway_device_info,
    build_installation_device_info,
    build_power_monitor_device_info,
    translate_default_device_name,
)
from tests.fakes.cloud import DEV_ID, VERSION, FakeCloud
from tests.fakes.radio_link import MAC
from tests.fakes.runtime import build_entry_runtime
from tests.fakes.sensors import fake_radio_links, radio_entry, setup_entry
from tests.fakes.setup import state_coordinator

PORTAL = "https://control.termoweb.net"


async def test_device_info_without_a_running_entry_uses_defaults(
    hass: HomeAssistant,
) -> None:
    """Without a loaded runtime the devices fall back to TermoWeb defaults."""
    gateway = build_gateway_device_info(hass, "missing", "dev")
    site = build_installation_device_info(hass, "missing", "dev")
    monitor = build_power_monitor_device_info(hass, "missing", "dev", "01")

    assert gateway["identifiers"] == {(DOMAIN, "dev")}
    assert (gateway["manufacturer"], gateway["model"]) == (
        "TermoWeb",
        "Gateway/Controller",
    )
    assert "sw_version" not in gateway
    assert "via_device_id" not in gateway
    assert site == {
        "identifiers": {(DOMAIN, "dev", "site")},
        "manufacturer": "TermoWeb",
        "name": "Site",
        "model": "Site",
        "configuration_url": PORTAL,
    }
    assert (monitor["manufacturer"], monitor["name"]) == (
        "TermoWeb",
        "Power Monitor 01",
    )
    assert "via_device_id" not in build_gateway_device_info(None, None, "dev")


async def test_gateway_and_site_take_their_details_from_the_coordinator(
    hass: HomeAssistant,
) -> None:
    """Gateway model, firmware, serial and site name come from the device list."""
    coordinator = state_coordinator(
        hass,
        device={
            "name": " My Home ",
            "model": "Controller",
            "serial_id": " SN-1 ",
            "fw_version": "3.0.1",
        },
    )
    build_entry_runtime(
        hass=hass,
        coordinator=coordinator,
        brand="  Ducaheat  ",
        version="7",
    )

    gateway = build_gateway_device_info(hass, "entry", "dev")
    site = build_installation_device_info(hass, "entry", "dev")

    assert gateway["manufacturer"] == "Ducaheat"
    assert gateway["model"] == "Controller"
    assert gateway["sw_version"] == "3.0.1"  # firmware wins over the version
    assert gateway["serial_number"] == "SN-1"
    assert (site["name"], site["manufacturer"]) == ("My Home", "Ducaheat")
    assert "via_device" not in site
    assert "via_device_id" not in site


async def test_gateway_without_metadata_reports_the_integration_version(
    hass: HomeAssistant,
) -> None:
    """A gateway without model or firmware keeps the default model and our version."""
    coordinator = state_coordinator(hass, device={"name": "Home"})
    build_entry_runtime(
        hass=hass,
        coordinator=coordinator,
        version="9.1",
    )

    info = build_gateway_device_info(hass, "entry", "dev")
    assert info["model"] == "Gateway/Controller"
    assert info["sw_version"] == "9.1"
    assert "serial_number" not in info
    assert "sw_version" not in build_gateway_device_info(
        hass, "entry", "dev", include_version=False
    )


@pytest.mark.parametrize(
    ("brand", "label", "has_site", "portal"),
    [
        ("termoweb", "TermoWeb", True, True),
        ("ducaheat", "Ducaheat", True, True),
        ("radio", "Radio", True, False),
        ("radio_monitor", "Radio", False, False),
    ],
)
async def test_brand_sets_labels_portal_link_and_site_parent(
    hass: HomeAssistant, brand: str, label: str, has_site: bool, portal: bool
) -> None:
    """Brand keys become labels; radio has no portal; a monitor has no site parent."""
    build_entry_runtime(hass=hass, brand=brand)
    site = dr.async_get(hass).async_get_or_create(
        config_entry_id="entry", identifiers={(DOMAIN, "dev", "site")}
    )

    gateway = build_gateway_device_info(hass, "entry", "dev")
    site_info = build_installation_device_info(hass, "entry", "dev")

    assert gateway["manufacturer"] == label
    assert gateway["name"] == f"{label} Gateway"
    assert (gateway.get("via_device_id") == site.id) is has_site
    assert "via_device" not in gateway
    assert ("configuration_url" in gateway) is portal
    assert ("configuration_url" in site_info) is portal


async def test_power_monitor_device_links_to_the_gateway(hass: HomeAssistant) -> None:
    """A power monitor sits under the registered gateway and follows the brand."""
    entry = MockConfigEntry(domain=DOMAIN, entry_id="entry")
    entry.add_to_hass(hass)
    build_entry_runtime(hass=hass, brand="ducaheat", config_entry=entry)
    gateway = dr.async_get(hass).async_get_or_create(
        config_entry_id="entry", identifiers={(DOMAIN, "dev")}
    )

    info = build_power_monitor_device_info(hass, "entry", "dev", " 01 ")

    assert info["identifiers"] == {(DOMAIN, "dev", "pmo", "01")}
    assert info["via_device_id"] == gateway.id
    assert info["manufacturer"] == "Ducaheat"
    assert info["name"] == "Power Monitor 01"
    assert info["translation_key"] == "power_monitor"
    assert info["translation_placeholders"] == {"addr": "01"}


def test_power_monitor_keeps_a_backend_name() -> None:
    """A backend-provided name is trimmed and used as-is, without a translation key."""
    info = build_power_monitor_device_info(None, None, "dev", "01", name=" Meter ")

    assert info["name"] == "Meter"
    assert "translation_key" not in info


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("Heater 7", "heater"),
        ("Accumulator 7", "accumulator"),
        ("Thermostat 7", "thermostat"),
        ("Power Monitor 7", "power_monitor"),
        ("Node 7", "node"),
    ],
)
def test_translate_default_device_name_matches_defaults(name: str, key: str) -> None:
    """Every English default name maps to its ``device`` translation key."""
    info = translate_default_device_name({"name": name}, "7")

    assert info == {
        "name": name,
        "translation_key": key,
        "translation_placeholders": {"addr": "7"},
    }


@pytest.mark.parametrize("name", ["Living Room", "Heater 8", "", None])
def test_translate_default_device_name_ignores_other_names(name: str | None) -> None:
    """Custom names and defaults for another address stay untranslated."""
    assert translate_default_device_name({"name": name}, "7") == {"name": name}


async def test_integration_version_comes_from_the_manifest(hass: HomeAssistant) -> None:
    """The reported integration version is the manifest version."""
    assert await async_get_integration_version(hass) == VERSION


SITE_ENTITIES = ("sensor.home_location", "sensor.home_total_energy")


async def _setup(hass: HomeAssistant, entry: MockConfigEntry) -> None:
    """Add the entry to hass and set it up."""
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()


async def test_site_entities_belong_to_the_site_device(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """Site-wide entities hang off the site device, not the gateway."""
    await _setup(hass, config_entry)

    ent_reg = er.async_get(hass)
    site = dr.async_get(hass).async_get_device_by_identifier(
        (DOMAIN, DEV_ID, "site"), config_entry.entry_id
    )
    power_limit = next(
        entity
        for entity in er.async_entries_for_config_entry(ent_reg, config_entry.entry_id)
        if entity.unique_id == f"{DOMAIN}:{DEV_ID}:site:power_limit"
    )
    for entity_id in (*SITE_ENTITIES, power_limit.entity_id):
        assert ent_reg.async_get(entity_id).device_id == site.id
    location = ent_reg.async_get("sensor.home_location")
    assert location.unique_id == f"{DOMAIN}:{DEV_ID}:site:info"


@pytest.mark.parametrize(
    ("geo", "state", "attributes"),
    [
        (
            GeoData(
                country="Testland",
                state="North",
                city="Sampleton",
                tz_code="Europe/X",
                zip="12345",
            ),
            "Sampleton, North, Testland",
            {
                "country": "Testland",
                "state": "North",
                "city": "Sampleton",
                "timezone": "Europe/X",
                "zip": "12345",
            },
        ),
        (GeoData(country="Testland"), "Testland", {"country": "Testland"}),
        (None, "unknown", {}),
    ],
    ids=["full", "country-only", "no-geo-data"],
)
async def test_location_sensor_summarises_geo_data(
    hass: HomeAssistant,
    cloud: FakeCloud,
    config_entry: MockConfigEntry,
    *,
    geo: GeoData | None,
    state: str,
    attributes: dict[str, str],
) -> None:
    """The location sensor shows city, state, country and only known fields."""
    cloud.get_geo_data.return_value = geo

    await _setup(hass, config_entry)

    location = hass.states.get("sensor.home_location")
    assert location.state == state
    extra = {
        key: value
        for key, value in location.attributes.items()
        if key not in ("friendly_name", "icon")
    }
    assert extra == attributes


async def test_radio_heater_device_reports_the_radio_brand(hass: HomeAssistant) -> None:
    """A heater behind the radio gateway is not labelled as TermoWeb hardware."""
    entry = radio_entry("radio", [{"type": "htr", "addr": "6", "name": "Heater 6"}])
    dev_id = MAC.replace(":", "").lower()
    with fake_radio_links():
        await setup_entry(hass, entry)
        heater = dr.async_get(hass).async_get_device_by_identifier(
            (DOMAIN, dev_id, "6"), entry.entry_id
        )
        assert heater is not None
        assert heater.manufacturer == "Radio"
        assert await hass.config_entries.async_unload(entry.entry_id)
        await hass.async_block_till_done()
