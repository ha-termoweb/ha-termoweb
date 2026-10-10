"""Websocket section of the diagnostics download."""

from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.components.diagnostics import (
    get_diagnostics_for_config_entry,
)

from tests.fakes.cloud import DEV_ID, FakeCloud


async def test_websocket_state_before_any_health_report_has_no_health(
    hass: HomeAssistant,
    hass_client: Any,
    cloud: FakeCloud,
    config_entry: MockConfigEntry,
) -> None:
    """A client that stored state but has no health tracker yet lists no health."""
    config_entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(config_entry.entry_id)
    await hass.async_block_till_done()
    config_entry.runtime_data.ws_state[DEV_ID] = {"status": "connecting"}

    result = await get_diagnostics_for_config_entry(hass, hass_client, config_entry)

    (client,) = result["websocket"]["clients"]
    assert client["state"] == {"status": "connecting"}
    assert "health" not in client
