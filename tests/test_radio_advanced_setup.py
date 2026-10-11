"""Advanced setup of radio heaters (offset, control mode, panel unit, window, radiant).

Real Home Assistant with a real radio client; only the gateway link is faked.
The heater cannot report these settings, so they are unknown until Home
Assistant writes them; heaters that do not answer ``DA`` have none.
"""

from __future__ import annotations

from homeassistant.const import STATE_OFF, STATE_ON, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.entity import EntityCategory
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.termoweb.const import DOMAIN
from tests.fakes.radio_link import (
    CLOCK_ACCEPTED,
    MAC,
    STATUS_SHORT,
    FakeRadioLink,
    dialect_a_heater,
    status_a,
)
from tests.fakes.sensors import fake_radio_links, radio_entry, setup_entry

DEV_ID = MAC.replace(":", "").lower()
HEATER = {"type": "htr", "addr": "6", "name": "Heater 6", "serial": "X123456"}
OFFSET = "number.heater_6_temperature_offset"
CONTROL = "select.heater_6_control_mode"
UNITS = "select.heater_6_panel_temperature_unit"
WINDOW = "switch.heater_6_open_window_detection"
RADIANT = "switch.heater_6_true_radiant"
ENTITIES = (OFFSET, CONTROL, UNITS, WINDOW, RADIANT)


def advanced_heater(link: FakeRadioLink) -> None:
    """Script a dialect-A heater that also accepts C4 writes."""

    dialect_a_heater(status_a(0x00))(link)
    link.reply(0xC4, b"\xc5\x55")


async def unload(hass: HomeAssistant, config: MockConfigEntry) -> None:
    """Unload the entry (stops the listener)."""

    assert await hass.config_entries.async_unload(config.entry_id)
    await hass.async_block_till_done()


async def call(hass: HomeAssistant, domain: str, service: str, entity_id: str, **data):
    """Call a service on ``entity_id`` and wait for it."""

    await hass.services.async_call(
        domain, service, {"entity_id": entity_id, **data}, blocking=True
    )


async def test_settings_are_config_entities_unknown_until_written(
    hass: HomeAssistant,
) -> None:
    """Each heater gets the five settings as config entities, unknown at first."""

    config = radio_entry("radio", [HEATER], dialect="A")
    with fake_radio_links(advanced_heater):
        await setup_entry(hass, config)
        registry = er.async_get(hass)
        for entity_id in ENTITIES:
            entry = registry.async_get(entity_id)
            assert entry.entity_category is EntityCategory.CONFIG
            assert entry.unique_id.startswith(f"{DOMAIN}:{DEV_ID}:htr:X123456:")
            assert hass.states.get(entity_id).state == STATE_UNKNOWN
        assert hass.states.get(CONTROL).attributes["options"] == [
            "pid",
            "hysteresis_0_25",
            "hysteresis_0_35",
            "hysteresis_0_5",
            "hysteresis_0_75",
        ]
        assert hass.states.get(UNITS).attributes["options"] == [
            "celsius",
            "fahrenheit",
        ]
        await unload(hass, config)


async def test_writes_send_c4_and_are_remembered(hass: HomeAssistant) -> None:
    """Every change sends the whole record; the record is kept in the options."""

    config = radio_entry("radio", [HEATER], dialect="A")
    with fake_radio_links(advanced_heater) as links:
        await setup_entry(hass, config)
        link = links[-1]
        link.sent.clear()

        await call(hass, "number", "set_value", OFFSET, value=-1.0)
        await call(hass, "select", "select_option", CONTROL, option="hysteresis_0_5")
        await call(hass, "select", "select_option", UNITS, option="fahrenheit")
        await call(hass, "switch", "turn_on", WINDOW)
        await call(hass, "switch", "turn_on", RADIANT)
        await call(hass, "switch", "turn_off", RADIANT)

        assert link.payloads() == [
            bytes.fromhex("C40400F60000000000"),
            bytes.fromhex("C40200F60000000000"),
            bytes.fromhex("C40201F60000000000"),
            bytes.fromhex("C40201F60000000100"),
            bytes.fromhex("C40201F60000000101"),
            bytes.fromhex("C40201F60000000100"),
        ]
        assert float(hass.states.get(OFFSET).state) == -1.0
        assert hass.states.get(CONTROL).state == "hysteresis_0_5"
        assert hass.states.get(UNITS).state == "fahrenheit"
        assert hass.states.get(WINDOW).state == STATE_ON
        assert hass.states.get(RADIANT).state == STATE_OFF
        assert config.options["radio_advanced"]["6"] == {
            "control_mode": 2,
            "units": 1,
            "offset_tenths": -10,
            "away_mode": 0,
            "away_offset": 0,
            "modified_auto_span": 0,
            "window_mode": 1,
            "true_radiant": 0,
        }
        # The heater's own readings stay Celsius whatever the panel shows.
        assert hass.states.get("climate.heater_6").attributes["units"] == "C"
        await unload(hass, config)

    # After a restart the stored record is shown again.
    with fake_radio_links(advanced_heater):
        assert await hass.config_entries.async_setup(config.entry_id)
        await hass.async_block_till_done()
        assert float(hass.states.get(OFFSET).state) == -1.0
        assert hass.states.get(WINDOW).state == STATE_ON
        await unload(hass, config)


async def test_a_rejected_write_fails_and_changes_nothing(hass: HomeAssistant) -> None:
    """``C5 56`` fails the call; the shown and stored values stay as they were."""

    config = radio_entry("radio", [HEATER], dialect="A")
    with fake_radio_links(advanced_heater) as links:
        await setup_entry(hass, config)
        links[-1].reply(0xC4, b"\xc5\x56")
        with pytest.raises(HomeAssistantError, match="Open window detection failed"):
            await call(hass, "switch", "turn_on", WINDOW)
        assert hass.states.get(WINDOW).state == STATE_UNKNOWN
        assert "radio_advanced" not in config.options
        await unload(hass, config)


def silent_heater(link: FakeRadioLink) -> None:
    """Script a dialect-B heater that answers status and clock but not DA."""

    link.reply(0xB8, STATUS_SHORT)
    link.reply(0x52, CLOCK_ACCEPTED)


async def test_heaters_without_the_advanced_record_show_unavailable(
    hass: HomeAssistant,
) -> None:
    """A heater that does not answer ``DA`` (seen in dialect B) has none of these."""

    config = radio_entry("radio", [HEATER])
    with fake_radio_links(silent_heater):
        await setup_entry(hass, config)
        for entity_id in ENTITIES:
            assert hass.states.get(entity_id).state == STATE_UNAVAILABLE
        await unload(hass, config)
