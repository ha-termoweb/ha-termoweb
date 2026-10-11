"""Boost of radio heaters through the climate entity, on real Home Assistant.

Dialect-A heaters boost for their own boost time (``D2 01`` / ``D2 00``,
flag 20 in the status). The climate entity offers it as the ``boost`` preset
and through ``termoweb.start_boost`` / ``termoweb.cancel_boost``; only the
gateway link is faked.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from homeassistant.components.climate import ATTR_PRESET_MODE, ATTR_PRESET_MODES
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from homeassistant.util import dt as dt_util
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.termoweb import climate as climate_module
from custom_components.termoweb.codecs.radio_codec import boost_end
from custom_components.termoweb.const import DOMAIN
from tests.fakes.radio_link import STATUS_SHORT, dialect_a_heater, status_a
from tests.fakes.sensors import fake_radio_links, radio_entry, setup_entry

HEATER = {"type": "htr", "addr": "6", "name": "Heater 6", "serial": "X123456"}
CLIMATE = "climate.heater_6"


async def unload(hass: HomeAssistant, config: MockConfigEntry) -> None:
    """Unload the entry (stops the listener)."""

    assert await hass.config_entries.async_unload(config.entry_id)
    await hass.async_block_till_done()


async def call(hass: HomeAssistant, domain: str, service: str, **data) -> None:
    """Call a service on the heater's climate entity and wait for it."""

    await hass.services.async_call(
        domain, service, {"entity_id": CLIMATE, **data}, blocking=True
    )


async def test_boost_preset_starts_and_cancels_the_heaters_boost(
    hass: HomeAssistant,
) -> None:
    """Preset boost sends ``D2 01``; preset none during a boost sends ``D2 00``."""

    config = radio_entry("radio", [HEATER], dialect="A")
    with fake_radio_links(dialect_a_heater(status_a(0x00))) as links:
        await setup_entry(hass, config)
        state = hass.states.get(CLIMATE)
        assert state.attributes[ATTR_PRESET_MODES] == ["none", "boost"]
        assert state.attributes[ATTR_PRESET_MODE] == "none"
        assert state.attributes["boost_active"] is False
        link = links[-1]
        link.sent.clear()

        await call(hass, "climate", "set_preset_mode", preset_mode="boost")
        assert hass.states.get(CLIMATE).attributes[ATTR_PRESET_MODE] == "boost"
        await call(hass, "climate", "set_preset_mode", preset_mode="boost")  # no-op
        await call(hass, "climate", "set_preset_mode", preset_mode="none")
        assert hass.states.get(CLIMATE).attributes[ATTR_PRESET_MODE] == "none"
        await call(hass, "climate", "set_preset_mode", preset_mode="none")  # no-op
        assert link.payloads() == [b"\xd2\x01", b"\xd2\x00"]
        await unload(hass, config)


async def test_boost_services_work_for_radio_heaters(hass: HomeAssistant) -> None:
    """start_boost / cancel_boost toggle the heater's boost; minutes are refused."""

    config = radio_entry("radio", [HEATER], dialect="A")
    with fake_radio_links(dialect_a_heater(status_a(0x00))) as links:
        await setup_entry(hass, config)
        link = links[-1]
        link.sent.clear()

        await call(hass, DOMAIN, "start_boost")
        await call(hass, DOMAIN, "cancel_boost")
        with pytest.raises(ServiceValidationError, match="own boost time"):
            await call(hass, DOMAIN, "start_boost", minutes=60)
        assert link.payloads() == [b"\xd2\x01", b"\xd2\x00"]
        await unload(hass, config)


async def test_reported_boost_shows_the_preset_and_its_end(
    hass: HomeAssistant,
) -> None:
    """Flag 20 plus the E4 tail ``04 75`` (Sunday 19:01) give preset and end time."""

    config = radio_entry("radio", [HEATER], dialect="A")
    boosting = status_a(0x20, mode=0x04, tail=bytes.fromhex("0475"))
    with fake_radio_links(dialect_a_heater(boosting)):
        await setup_entry(hass, config)
        end = boost_end(0, 19 * 60 + 1, dt_util.now())  # the next Sunday 19:01
        attrs = hass.states.get(CLIMATE).attributes
        assert attrs[ATTR_PRESET_MODE] == "boost"
        assert attrs["boost_active"] is True
        assert attrs["boost_end"] == end.isoformat()
        remaining = (end - dt_util.now()).total_seconds() / 60
        assert abs(attrs["boost_minutes_remaining"] - remaining) <= 1
        await unload(hass, config)


async def test_dialect_b_heaters_have_no_boost(hass: HomeAssistant) -> None:
    """Dialect B toggles have no effect: no boost preset, boost services refused."""

    config = radio_entry("radio", [HEATER])
    with fake_radio_links(dialect_a_heater(STATUS_SHORT)) as links:
        await setup_entry(hass, config)
        state = hass.states.get(CLIMATE)
        assert ATTR_PRESET_MODES not in state.attributes
        assert "boost_active" not in state.attributes
        links[-1].sent.clear()
        with pytest.raises(ServiceValidationError, match="heaters with boost"):
            await call(hass, DOMAIN, "start_boost")
        with pytest.raises(ServiceValidationError):
            await call(hass, "climate", "set_preset_mode", preset_mode="boost")
        assert links[-1].sent == []
        await unload(hass, config)


async def test_preset_none_ends_a_temporary_override(hass: HomeAssistant) -> None:
    """A boost heater shows its override as a preset; preset none resumes Auto."""

    config = radio_entry("radio", [HEATER], dialect="A")
    with fake_radio_links(dialect_a_heater(status_a(0x00, mode=0x03))) as links:
        await setup_entry(hass, config)
        state = hass.states.get(CLIMATE)
        assert state.attributes[ATTR_PRESET_MODE] == "temporary_override"
        link = links[-1]
        link.sent.clear()

        with patch.object(climate_module, "_WRITE_DEBOUNCE", 0):
            await call(hass, "climate", "set_preset_mode", preset_mode="none")
            for _ in range(100):  # the write runs after a debounce task
                if link.sent:
                    break
                await asyncio.sleep(0.01)
        assert link.payloads() == [b"\xb4\x01"]  # mode Auto
        await unload(hass, config)
