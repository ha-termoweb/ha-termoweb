"""The site device and its entities after a real setup."""

from __future__ import annotations

from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr, entity_registry as er
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.domain.state import GeoData

from .conftest import DEV_ID, FakeCloud

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
