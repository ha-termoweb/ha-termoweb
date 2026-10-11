"""Tests for the radio entry options flow."""

from __future__ import annotations

from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.termoweb.const import BRAND_RADIO, CONF_BRAND, DOMAIN

# Synthetic identifiers only.
NODES = [{"type": "htr", "addr": "6", "name": "Heater 6"}]


async def test_options_open_the_network_move_without_a_pairing_step(
    hass: HomeAssistant,
) -> None:
    """Heaters pair with the gateway's button, so the options only move heaters."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_BRAND: BRAND_RADIO, "network_id": "1234", "nodes": NODES},
        options={"radio_power": {"power_limit": 2000}},
    )
    entry.add_to_hass(hass)
    form = await hass.config_entries.options.async_init(entry.entry_id)
    assert form["type"] is FlowResultType.FORM
    assert form["step_id"] == "rehome"
    assert form["description_placeholders"] == {"heaters": "Heater 6"}
