"""Runback and EASY switches of radio heaters, on real Home Assistant.

The radio entry runs its real client; only the gateway link is faked
(``tests.fakes.radio_link``). Dialect A heaters have the toggles, dialect B
heaters do not.
"""

from __future__ import annotations

from homeassistant.const import STATE_OFF, STATE_ON, STATE_UNKNOWN
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.termoweb.const import DOMAIN
from tests.fakes.radio_link import MAC, STATUS_SHORT, dialect_a_heater, status_a
from tests.fakes.sensors import entity_ids, fake_radio_links, radio_entry, setup_entry

DEV_ID = MAC.replace(":", "").lower()
HEATER = {"type": "htr", "addr": "6", "name": "Heater 6", "serial": "X123456"}
RUNBACK = "switch.heater_6_runback"
EASY = "switch.heater_6_easy_mode"


async def unload(hass: HomeAssistant, config: MockConfigEntry) -> None:
    """Unload the entry (stops the listener)."""

    assert await hass.config_entries.async_unload(config.entry_id)
    await hass.async_block_till_done()


async def turn(hass: HomeAssistant, service: str, entity_id: str) -> None:
    """Call switch.turn_on / switch.turn_off and wait for it."""

    await hass.services.async_call(
        "switch", service, {"entity_id": entity_id}, blocking=True
    )


async def test_switches_follow_the_heaters_status_flags(hass: HomeAssistant) -> None:
    """Flag 80 is runback, flag 40 is EASY; unique ids use the heater serial."""

    config = radio_entry("radio", [HEATER], dialect="A")
    with fake_radio_links(dialect_a_heater(status_a(0x80))):
        await setup_entry(hass, config)
        assert entity_ids(hass, config, "switch") == {RUNBACK, EASY}
        assert hass.states.get(RUNBACK).state == STATE_ON
        assert hass.states.get(EASY).state == STATE_OFF
        registry = er.async_get(hass)
        assert registry.async_get(RUNBACK).unique_id == (
            f"{DOMAIN}:{DEV_ID}:htr:X123456:runback"
        )
        assert registry.async_get(EASY).unique_id == (
            f"{DOMAIN}:{DEV_ID}:htr:X123456:easy"
        )
        await unload(hass, config)


async def test_switches_write_the_toggles(hass: HomeAssistant) -> None:
    """Turning a switch on or off sends D4 / D6 and shows the new state at once."""

    config = radio_entry("radio", [HEATER], dialect="A")
    with fake_radio_links(dialect_a_heater(status_a(0x00))) as links:
        await setup_entry(hass, config)
        link = links[-1]
        link.sent.clear()

        await turn(hass, "turn_on", EASY)
        assert hass.states.get(EASY).state == STATE_ON
        await turn(hass, "turn_off", EASY)
        assert hass.states.get(EASY).state == STATE_OFF
        await turn(hass, "turn_on", RUNBACK)
        assert hass.states.get(RUNBACK).state == STATE_ON
        await turn(hass, "turn_off", RUNBACK)
        assert hass.states.get(RUNBACK).state == STATE_OFF
        # Runback reads the status first, then restores manual 21.0 C after off.
        assert link.payloads() == [
            b"\xd6\x01",
            b"\xd6\x00",
            b"\xb8",
            b"\xd4\x01",
            b"\xd4\x00",
            bytes.fromhex("B4022A"),
        ]
        await unload(hass, config)


async def test_a_refused_toggle_fails_the_call(hass: HomeAssistant) -> None:
    """A ``<op+1> 56`` answer is an error and the state does not change."""

    config = radio_entry("radio", [HEATER], dialect="A")
    with fake_radio_links(dialect_a_heater(status_a(0x00))) as links:
        await setup_entry(hass, config)
        links[-1].reply(0xD6, b"\xd7\x56")
        with pytest.raises(HomeAssistantError, match="EASY mode failed"):
            await turn(hass, "turn_on", EASY)
        assert hass.states.get(EASY).state == STATE_OFF
        await unload(hass, config)


async def test_state_unknown_without_a_full_status(hass: HomeAssistant) -> None:
    """A heater that only gives the short status record has no flags to show."""

    config = radio_entry("radio", [HEATER], dialect="A")
    with fake_radio_links(dialect_a_heater(STATUS_SHORT)):
        await setup_entry(hass, config)
        assert hass.states.get(RUNBACK).state == STATE_UNKNOWN
        await unload(hass, config)


async def test_dialect_b_heaters_get_no_switches(hass: HomeAssistant) -> None:
    """Dialect B has no runback or EASY toggle: the platform adds nothing."""

    config = radio_entry("radio", [HEATER])
    with fake_radio_links(dialect_a_heater(STATUS_SHORT)):
        await setup_entry(hass, config)
        assert entity_ids(hass, config, "switch") == set()
        await unload(hass, config)
