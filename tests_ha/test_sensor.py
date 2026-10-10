"""Sensor entities on real Home Assistant: node, energy, site and radio sensors."""

from __future__ import annotations

import time

from homeassistant.const import (
    ATTR_UNIT_OF_MEASUREMENT,
    STATE_UNKNOWN,
    UnitOfTemperature,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.util import dt as dt_util
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.termoweb.const import DOMAIN, signal_radio_frames
from custom_components.termoweb.domain import NodeId, NodeSettingsDelta, NodeType
from custom_components.termoweb.domain.state import GeoData

from .conftest import DEV_ID, FakeCloud
from .fakes.radio_link import MAC
from .fakes.sensors import (
    entity_ids,
    fake_radio_links,
    radio_entry,
    serve_nodes,
    setup_entry,
)

NODES = [
    {"type": "htr", "addr": 1, "name": "Living room"},
    {"type": "acm", "addr": 2, "name": "Store"},
    {"type": "pmo", "addr": 3, "name": "Meter"},
    {"type": "thm", "addr": 4, "name": "Hall"},
]
RADIO_DEV_ID = MAC.replace(":", "").lower()


async def test_node_sensors_report_device_settings(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """Temperature, accumulator charge and thermostat battery come from settings."""
    serve_nodes(
        cloud,
        NODES,
        {
            ("htr", "1"): {"mtemp": "19.5"},
            ("acm", "2"): {
                "charging": True,
                "current_charge_per": 150.5,  # clamped to 0..100
                "target_charge_per": -12,
            },
            ("thm", "4"): {"batt_level": 4},
        },
    )
    await setup_entry(hass, config_entry)

    temperature = hass.states.get("sensor.living_room_temperature")
    assert float(temperature.state) == 19.5
    assert temperature.attributes[ATTR_UNIT_OF_MEASUREMENT] == UnitOfTemperature.CELSIUS
    assert temperature.attributes["addr"] == "1"
    assert hass.states.get("sensor.store_charging").state == "True"
    assert hass.states.get("sensor.store_current_charge").state == "100"
    assert hass.states.get("sensor.store_target_charge").state == "0"
    battery = hass.states.get("sensor.hall_battery")
    assert battery.state == "80"
    assert battery.attributes["batt_level_steps"] == 4


async def test_sensor_set_depends_on_node_type(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """Thermostats meter no energy; accumulators add charge and boost sensors."""
    serve_nodes(cloud, NODES)
    await setup_entry(hass, config_entry)

    assert entity_ids(hass, config_entry, "sensor") == {
        "sensor.living_room_temperature",
        "sensor.living_room_energy",
        "sensor.living_room_power",
        "sensor.store_temperature",
        "sensor.store_charging",
        "sensor.store_current_charge",
        "sensor.store_target_charge",
        "sensor.store_energy",
        "sensor.store_power",
        "sensor.store_boost_minutes_remaining",
        "sensor.store_boost_end",
        "sensor.meter_energy",
        "sensor.meter_power",
        "sensor.hall_temperature",
        "sensor.hall_battery",
        "sensor.home_total_energy",
        "sensor.home_location",
    }


@pytest.mark.parametrize(
    ("node_type", "addr", "field", "value", "entity_id"),
    [
        ("thm", "4", "batt_level", "invalid", "sensor.hall_battery"),
        ("thm", "4", "batt_level", True, "sensor.hall_battery"),
        ("acm", "2", "charging", "unexpected", "sensor.store_charging"),
        ("acm", "2", "current_charge_per", None, "sensor.store_current_charge"),
    ],
)
async def test_unusable_device_values_show_unknown(
    *,
    hass: HomeAssistant,
    cloud: FakeCloud,
    config_entry: MockConfigEntry,
    node_type: str,
    addr: str,
    field: str,
    value: object,
    entity_id: str,
) -> None:
    """A value the sensor cannot interpret reads as unknown, not a wrong number."""
    serve_nodes(
        cloud,
        NODES,
        {
            ("thm", "4"): {"batt_level": 3},
            ("acm", "2"): {"charging": False, "current_charge_per": 40},
        },
    )
    await setup_entry(hass, config_entry)
    assert hass.states.get(entity_id).state != STATE_UNKNOWN

    config_entry.runtime_data.coordinator.handle_ws_deltas(
        DEV_ID,
        [
            NodeSettingsDelta(
                node_id=NodeId(NodeType(node_type), addr), changes={field: value}
            )
        ],
    )
    await hass.async_block_till_done()

    assert hass.states.get(entity_id).state == STATE_UNKNOWN
    if entity_id == "sensor.hall_battery":
        assert hass.states.get(entity_id).attributes["batt_level_steps"] is None


async def test_temperature_without_reading_is_unknown(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """A heater that reports no measured temperature has an unknown sensor."""
    serve_nodes(cloud, NODES[:1])
    await setup_entry(hass, config_entry)

    assert hass.states.get("sensor.living_room_temperature").state == STATE_UNKNOWN


async def test_energy_and_power_follow_websocket_samples(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """Counter samples become kWh energy and W power on heaters and the meter."""
    serve_nodes(cloud, NODES)
    await setup_entry(hass, config_entry)
    # A tracked meter without readings is available (unknown), not unavailable.
    assert hass.states.get("sensor.meter_power").state == STATE_UNKNOWN

    energy = config_entry.runtime_data.energy_coordinator
    now = time.time()
    for t, htr, pmo in ((now - 3600, 1000, 5000), (now, 1150, 5100)):
        energy.handle_ws_samples(
            DEV_ID,
            {
                "htr": {"1": {"t": t, "counter": htr}},
                "pmo": {"3": {"t": t, "counter": pmo}},
            },
        )
    await hass.async_block_till_done()

    assert float(hass.states.get("sensor.living_room_energy").state) == 1.15
    assert float(hass.states.get("sensor.living_room_power").state) == pytest.approx(
        150.0
    )
    view = config_entry.runtime_data.coordinator.domain_view
    meter = view.get_energy_metric("pmo", "3")
    assert float(hass.states.get("sensor.meter_energy").state) == pytest.approx(
        meter.energy_kwh
    )
    assert float(hass.states.get("sensor.meter_power").state) == pytest.approx(
        meter.power_w
    )
    # The accumulator has no reading yet, so the site total stays unknown.
    assert hass.states.get("sensor.store_energy").state == STATE_UNKNOWN
    assert hass.states.get("sensor.home_total_energy").state == STATE_UNKNOWN


async def test_boost_sensors_show_remaining_time_and_end(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """An active boost exposes its end time and remaining minutes."""
    served = serve_nodes(cloud, NODES[1:2])
    await setup_entry(hass, config_entry)
    assert hass.states.get("sensor.store_boost_minutes_remaining").state == (
        STATE_UNKNOWN
    )
    assert hass.states.get("sensor.store_boost_end").state == "Never"

    served[("acm", "2")].update(
        mode="boost", boost_active=True, boost_end_day=3, boost_end_min=600
    )
    await config_entry.runtime_data.coordinator.async_refresh()
    await hass.async_block_till_done()

    active = hass.states.get("binary_sensor.store_boost_active")
    minutes = hass.states.get("sensor.store_boost_minutes_remaining")
    end = hass.states.get("sensor.store_boost_end")
    assert int(minutes.state) == active.attributes["boost_minutes_remaining"] > 0
    assert dt_util.parse_datetime(end.state) == dt_util.parse_datetime(
        active.attributes["boost_end"]
    )
    assert end.attributes["boost_active"] is True


async def test_unnamed_nodes_get_type_names(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """Nodes without a name are named after their type and address."""
    serve_nodes(cloud, [{"type": "pmo", "addr": 3}, {"type": "thm", "addr": 4}])
    await setup_entry(hass, config_entry)

    devices = dr.async_get(hass)
    registry = er.async_get(hass)
    meter = registry.async_get("sensor.power_monitor_3_energy")
    assert devices.async_get(meter.device_id).name == "Power Monitor 3"
    battery = registry.async_get("sensor.thermostat_4_battery")
    assert devices.async_get(battery.device_id).name == "Thermostat 4"


async def test_location_sensor_shows_geo_data(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """The site location summarises the gateway's geo data."""
    serve_nodes(cloud, NODES[:1])
    cloud.get_geo_data.return_value = GeoData(
        country="Norway", state="Viken", city="Oslo", tz_code="CET", zip="0001"
    )
    await setup_entry(hass, config_entry)

    location = hass.states.get("sensor.home_location")
    assert location.state == "Oslo, Viken, Norway"
    assert {k: location.attributes[k] for k in ("city", "state", "country")} == {
        "city": "Oslo",
        "state": "Viken",
        "country": "Norway",
    }
    assert location.attributes["timezone"] == "CET"
    assert location.attributes["zip"] == "0001"


async def test_location_sensor_without_geo_data_is_unknown(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """Without geo data the location is unknown and has no attributes."""
    serve_nodes(cloud, NODES[:1])
    await setup_entry(hass, config_entry)

    location = hass.states.get("sensor.home_location")
    assert location.state == STATE_UNKNOWN
    assert "city" not in location.attributes


async def test_monitor_entry_counts_frames_heard(hass: HomeAssistant) -> None:
    """A listen-only entry has only the frames sensor, fed by the monitor signal."""
    entry = radio_entry("radio_monitor")
    with fake_radio_links():
        await setup_entry(hass, entry)
        assert entity_ids(hass, entry, "sensor") == {
            "sensor.radio_gateway_frames_heard"
        }
        frames = "sensor.radio_gateway_frames_heard"
        assert hass.states.get(frames).state == "0"
        assert hass.states.get(frames).attributes["last_frame"] is None
        assert er.async_get(hass).async_get(frames).unique_id == (
            f"{DOMAIN}:{RADIO_DEV_ID}:frames_heard"
        )

        async_dispatcher_send(
            hass,
            signal_radio_frames(entry.entry_id),
            {"frames": 7, "last_frame": "2026-01-02T03:04:05+00:00"},
        )
        await hass.async_block_till_done()

        assert hass.states.get(frames).state == "7"
        assert hass.states.get(frames).attributes["last_frame"] == (
            "2026-01-02T03:04:05+00:00"
        )
        assert await hass.config_entries.async_unload(entry.entry_id)
        await hass.async_block_till_done()


async def test_radio_gateway_has_estimated_energy_but_no_location(
    hass: HomeAssistant,
) -> None:
    """A radio gateway gets heater energy sensors, but no frames or geo sensor."""
    entry = radio_entry("radio", [{"type": "htr", "addr": "6", "name": "Heater 6"}])
    with fake_radio_links():
        await setup_entry(hass, entry)
        sensors = entity_ids(hass, entry, "sensor")
        assert {
            "sensor.heater_6_energy",
            "sensor.radio_gateway_total_energy",
        } <= sensors
        assert not any(s.endswith(("_location", "_frames_heard")) for s in sensors)
        assert await hass.config_entries.async_unload(entry.entry_id)
        await hass.async_block_till_done()


@pytest.mark.parametrize("brand", ["radio_monitor", "radio"])
async def test_radio_devices_link_only_to_devices_that_exist(
    hass: HomeAssistant, brand: str
) -> None:
    """Every via_device of a radio entry's devices is a device the entry created."""
    entry = radio_entry(brand, None if brand == "radio_monitor" else [])
    with fake_radio_links():
        await setup_entry(hass, entry)
        devices = dr.async_entries_for_config_entry(dr.async_get(hass), entry.entry_id)
        ids = {device.id for device in devices}
        assert all(d.via_device_id in ids | {None} for d in devices)
        assert {d.manufacturer for d in devices} == {"Radio"}
        registry = dr.async_get(hass)
        gateway = registry.async_get_device_by_identifier(
            (DOMAIN, RADIO_DEV_ID), entry.entry_id
        )
        site = registry.async_get_device_by_identifier(
            (DOMAIN, RADIO_DEV_ID, "site"), entry.entry_id
        )
        if brand == "radio_monitor":
            assert site is None
            assert gateway.via_device_id is None
        else:
            assert gateway.via_device_id == site.id
        assert await hass.config_entries.async_unload(entry.entry_id)
        await hass.async_block_till_done()
