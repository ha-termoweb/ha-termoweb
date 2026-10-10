"""Config entry diagnostics through Home Assistant's real diagnostics endpoint."""

from __future__ import annotations

import json
from typing import Any

from homeassistant.components.diagnostics import REDACTED
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.components.diagnostics import (
    get_diagnostics_for_config_entry,
)

from custom_components.termoweb.domain.state import GeoData

from .conftest import DEV_ID, PASSWORD, USERNAME, FakeCloud

# Synthetic location only.
GEO = GeoData(
    country="Testland", state="North", city="Sampleton", tz_code="Europe/X", zip="12345"
)


async def _diagnostics(
    hass: HomeAssistant, hass_client: Any, cloud: FakeCloud, entry: MockConfigEntry
) -> dict[str, Any]:
    """Set the entry up and fetch its diagnostics over HTTP."""
    cloud.get_geo_data.return_value = GEO
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    entry.runtime_data.last_energy_import_summary = {
        "device_id": DEV_ID,
        "nodes": [{"dev_id": DEV_ID, "serial_id": "SN-TEST", "addr": "1"}],
    }
    return await get_diagnostics_for_config_entry(hass, hass_client, entry)


async def test_diagnostics_redact_location_and_ids(
    hass: HomeAssistant,
    hass_client: Any,
    cloud: FakeCloud,
    config_entry: MockConfigEntry,
) -> None:
    """B17: geo_data and any nested device id, serial or credential are redacted."""
    result = await _diagnostics(hass, hass_client, cloud, config_entry)

    assert result["site"]["geo_data"] == {
        "country": REDACTED,
        "state": REDACTED,
        "city": REDACTED,
        "zip": REDACTED,
        "tz_code": "Europe/X",
    }
    assert result["energy_import"]["last_run"] == {
        "device_id": REDACTED,
        "nodes": [{"dev_id": REDACTED, "serial_id": REDACTED, "addr": "1"}],
    }
    text = json.dumps(result)
    for secret in (DEV_ID, "SN-TEST", "Sampleton", "12345", "North", USERNAME):
        assert secret not in text
    assert PASSWORD not in text
    assert result["integration"]["brand"] == "TermoWeb"
    assert result["site"]["node_inventory"]
