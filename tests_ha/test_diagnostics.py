"""Config entry diagnostics through Home Assistant's real diagnostics endpoint."""

from __future__ import annotations

import json
import platform
from typing import Any

from homeassistant.components.diagnostics import REDACTED
from homeassistant.const import __version__ as HA_VERSION
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.components.diagnostics import (
    get_diagnostics_for_config_entry,
)

from custom_components.termoweb.backend.radio_backend import RadioBackend
from custom_components.termoweb.backend.radio_client import RadioClient
from custom_components.termoweb.backend.sanitize import mask_identifier
from custom_components.termoweb.backend.ws_health import WsHealthTracker
from custom_components.termoweb.const import BRAND_RADIO, CONF_BRAND, DOMAIN
from custom_components.termoweb.diagnostics import async_get_config_entry_diagnostics
from custom_components.termoweb.domain.state import GeoData
from tests_ha.fakes.radio_link import FakeRadioLink, gateway_info
from tests_ha.fakes.runtime import build_entry_runtime

from .conftest import DEV_ID, PASSWORD, USERNAME, VERSION, FakeCloud

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


async def test_diagnostics_report_versions_site_and_websocket(
    hass: HomeAssistant,
    hass_client: Any,
    cloud: FakeCloud,
    config_entry: MockConfigEntry,
) -> None:
    """Versions, gateway metadata and masked websocket health are reported."""
    await hass.config.async_set_time_zone("Europe/London")
    cloud.list_devices.return_value = [
        {"dev_id": DEV_ID, "name": "Home", "model": "TW-500", "fw_version": "2.1.0"}
    ]
    config_entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(config_entry.entry_id)
    await hass.async_block_till_done()
    runtime = config_entry.runtime_data
    tracker = WsHealthTracker(DEV_ID)
    tracker.update_status("healthy")
    tracker.mark_payload(stale_after=300)
    runtime.ws_state[DEV_ID] = {"status": "healthy"}
    runtime.ws_trackers[DEV_ID] = tracker

    result = await get_diagnostics_for_config_entry(hass, hass_client, config_entry)

    assert result["integration"]["version"] == VERSION
    assert result["home_assistant"]["version"] == HA_VERSION
    assert result["home_assistant"]["python_version"] == platform.python_version()
    assert result["home_assistant"]["time_zone"] == "Europe/London"
    site = result["site"]
    assert (site["name"], site["model"], site["fw_version"]) == (
        "Home",
        "TW-500",
        "2.1.0",
    )
    assert "geo_data" not in site
    assert site["node_inventory"] == [
        {"type": "htr", "addr": "1", "name": "Living room"}
    ]
    [client] = result["websocket"]["clients"]
    assert client["device"] == mask_identifier(DEV_ID)
    assert client["state"] == {"status": "healthy"}
    assert client["health"]["status"] == "healthy"
    assert "radio" not in result
    assert DEV_ID not in json.dumps(result)


async def test_diagnostics_radio_section_without_mac_or_network_id(
    hass: HomeAssistant,
) -> None:
    """Radio entries report gateway facts and the last survey, never MAC or net id."""

    def _link(host, port, dialect, **kwargs) -> FakeRadioLink:
        link = FakeRadioLink(host, port, dialect, **kwargs)
        link.info = gateway_info(version="3.7-esp32")
        return link

    entry = MockConfigEntry(
        domain=DOMAIN, data={CONF_BRAND: BRAND_RADIO, "network_id": "1234"}
    )
    entry.add_to_hass(hass)
    client = RadioClient(
        "10.0.0.5", 2323, "B", [], network_id=b"\x12\x34", link_factory=_link
    )
    runtime = build_entry_runtime(
        hass=hass,
        dev_id="aabbcc001122",
        brand=BRAND_RADIO,
        client=client,
        backend=RadioBackend(brand=BRAND_RADIO, client=client),
        config_entry=entry,
    )

    radio = (await async_get_config_entry_diagnostics(hass, entry))["radio"]
    assert radio == {
        "radio_type": "esp32",
        "dialect": "B",
        "connected": False,
        "listen_only": False,
        "gateway": None,
    }

    await client.async_connect()
    runtime.last_radio_survey = {"verdict": "silent"}
    runtime.last_radio_capture = {"frames": 3, "networks": ["1234"], "file": None}
    radio = (await async_get_config_entry_diagnostics(hass, entry))["radio"]
    await client.async_close()

    assert radio["connected"] is True
    assert radio["gateway"] == {
        "firmware": "3.7-esp32",
        "freq": "869.525",
        "sync": "2DD4",
        "dialect": None,
        "autoack": True,
        "station_id": 1,
        "survey": True,
    }
    assert radio["last_survey"] == {"verdict": "silent"}
    assert radio["last_capture"] == {"frames": 3, "file": None}
    text = repr(radio)
    assert "AA:BB" not in text
    assert "1234" not in text
