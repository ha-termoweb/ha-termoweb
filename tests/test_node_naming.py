"""Node names are consistent across every platform, on real Home Assistant."""

from __future__ import annotations

from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.termoweb.const import DOMAIN
from tests.fakes.cloud import DEV_ID, FakeCloud
from tests.fakes.entities import FakeNodes, setup_entry


async def test_thermostat_named_like_a_heater_keeps_its_name_everywhere(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """A thermostat the user named "Heater 4" is called that by every entity."""
    nodes = FakeNodes(
        cloud, {("thm", "4"): ("Heater 4", {"mode": "auto", "units": "C"})}
    )
    with nodes.patched():
        entry = await setup_entry(hass, config_entry)

        names = {
            state.entity_id: state.attributes["friendly_name"]
            for state in hass.states.async_all()
            if "4" in state.entity_id
        }
        assert names == {
            "climate.heater_4": "Heater 4",
            "number.heater_4_priority": "Heater 4 Priority",
            "sensor.heater_4_temperature": "Heater 4 Temperature",
            "sensor.heater_4_battery": "Heater 4 Battery",
        }
        device = dr.async_get(hass).async_get_device_by_identifier(
            (DOMAIN, DEV_ID, "4"), entry.entry_id
        )
        assert device.name == "Heater 4"


async def test_node_devices_report_their_kind_as_model(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """Heater, accumulator and thermostat devices each carry their own model."""
    settings = {"mode": "auto", "units": "C"}
    nodes = FakeNodes(
        cloud,
        {
            ("htr", "1"): (None, dict(settings)),
            ("acm", "2"): (None, dict(settings)),
            ("thm", "4"): (None, dict(settings)),
        },
    )
    with nodes.patched():
        entry = await setup_entry(hass, config_entry)

        registry = dr.async_get(hass)
        models = {
            addr: registry.async_get_device_by_identifier(
                (DOMAIN, DEV_ID, addr), entry.entry_id
            ).model
            for addr in ("1", "2", "4")
        }
        assert models == {"1": "Heater", "2": "Accumulator", "4": "Thermostat"}
