"""Entity availability and the installation energy total on real Home Assistant."""

from __future__ import annotations

from collections.abc import Generator
from datetime import timedelta
import time
from typing import Any
from unittest.mock import patch

from aiohttp import ClientError
from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util
import pytest
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_fire_time_changed,
)

from custom_components.termoweb.backend.ducaheat import (
    DucaheatBackend,
    DucaheatRESTClient,
)
from custom_components.termoweb.const import BRAND_DUCAHEAT, CONF_BRAND, DOMAIN

from .conftest import PASSWORD, USERNAME, FakeCloud

TOTAL = "sensor.home_total_energy"
HTR_ENERGY = "sensor.living_room_energy"
ACM_ENERGY = "sensor.store_energy"
PMO_ENERGY = "sensor.meter_energy"
# Entities of the heater (addr 1) and accumulator (addr 2) nodes.
NODE_ENTITIES = (
    "climate.living_room",
    "climate.store",
    "sensor.living_room_temperature",
    "number.living_room_priority",
    "number.store_boost_temperature",
    "binary_sensor.store_boost_active",
    "button.store_start_boost",
    "button.living_room_flash_display",
)


@pytest.fixture
def samples(cloud: FakeCloud) -> Generator[dict[tuple[str, str], list[dict]]]:
    """Serve a heater, an accumulator, a power monitor and a thermostat (addr 1-4)."""
    names = {"htr": "Living room", "acm": "Store", "pmo": "Meter", "thm": "Hall"}
    settings = {"mode": "auto", "state": "off", "stemp": "20.0", "units": "C"}
    cloud.get_nodes.return_value = {
        "nodes": [
            {"type": node_type, "addr": addr, "name": name}
            for addr, (node_type, name) in enumerate(names.items(), start=1)
        ]
    }
    cloud.get_node_settings.side_effect = lambda *_: dict(settings)
    now = time.time()
    # The thermostat meters no energy; the total sums heaters only.
    by_node = {
        ("htr", "1"): [{"t": now - 60, "counter": 1500}],
        ("acm", "2"): [{"t": now - 60, "counter": 2500}],
        ("pmo", "3"): [{"t": now - 60, "counter": 9000}],
    }

    async def _samples(_dev_id: str, node: tuple[str, str], *_: Any) -> list[dict]:
        return by_node[(node[0], str(node[1]))]

    cloud.get_node_samples.side_effect = _samples
    return by_node


async def _setup(hass: HomeAssistant, entry: MockConfigEntry) -> None:
    """Add the entry to hass and set it up."""
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()


async def _refresh(hass: HomeAssistant, entry: MockConfigEntry, name: str) -> None:
    """Advance time past the coordinator's update interval and let it poll."""
    coordinator = getattr(hass.data[DOMAIN][entry.entry_id], name)
    async_fire_time_changed(
        hass, dt_util.utcnow() + coordinator.update_interval + timedelta(seconds=1)
    )
    await hass.async_block_till_done()


async def _fire_next_hourly_poll(hass: HomeAssistant) -> None:
    """Advance time to the next HH:05 energy poll and let it run."""
    now = dt_util.utcnow()
    target = (now + timedelta(hours=1)).replace(minute=5, second=0, microsecond=0)
    async_fire_time_changed(hass, target)
    await hass.async_block_till_done()


async def test_node_entities_unavailable_while_cloud_down(
    hass: HomeAssistant,
    cloud: FakeCloud,
    samples: dict,
    config_entry: MockConfigEntry,
) -> None:
    """A failed state poll marks node entities unavailable until it recovers."""
    await _setup(hass, config_entry)
    registry = er.async_get(hass)
    for entity_id in NODE_ENTITIES:
        assert registry.async_get(entity_id) is not None, entity_id
        assert hass.states.get(entity_id).state != STATE_UNAVAILABLE, entity_id

    # The cloud is down: every REST call fails.
    settings = cloud.get_node_settings.side_effect
    cloud.get_node_settings.side_effect = ClientError("cloud down")
    cloud.get_node_samples.side_effect = ClientError("cloud down")
    await _refresh(hass, config_entry, "coordinator")

    for entity_id in NODE_ENTITIES:
        assert hass.states.get(entity_id).state == STATE_UNAVAILABLE, entity_id

    cloud.get_node_settings.side_effect = settings
    await _refresh(hass, config_entry, "coordinator")

    for entity_id in NODE_ENTITIES:
        assert hass.states.get(entity_id).state != STATE_UNAVAILABLE, entity_id


async def test_energy_entities_unavailable_when_energy_poll_fails(
    hass: HomeAssistant,
    cloud: FakeCloud,
    samples: dict,
    config_entry: MockConfigEntry,
) -> None:
    """A failed energy poll marks the energy sensors and the total unavailable."""
    await _setup(hass, config_entry)
    assert float(hass.states.get(TOTAL).state) == pytest.approx(4.0)

    cloud.get_node_samples.side_effect = TimeoutError
    await _fire_next_hourly_poll(hass)

    for entity_id in (TOTAL, HTR_ENERGY, ACM_ENERGY, PMO_ENERGY):
        assert hass.states.get(entity_id).state == STATE_UNAVAILABLE, entity_id


async def test_installation_total_unknown_when_a_node_is_missing(
    hass: HomeAssistant,
    samples: dict,
    config_entry: MockConfigEntry,
) -> None:
    """The total is unknown, not a partial sum, while one heater has no reading."""
    samples[("acm", "2")] = []
    await _setup(hass, config_entry)

    assert float(hass.states.get(HTR_ENERGY).state) == pytest.approx(1.5)
    assert hass.states.get(TOTAL).state == STATE_UNKNOWN


async def test_installation_total_sums_every_node(
    hass: HomeAssistant,
    samples: dict,
    config_entry: MockConfigEntry,
) -> None:
    """With every heater reporting, the total is their sum."""
    await _setup(hass, config_entry)

    assert float(hass.states.get(TOTAL).state) == pytest.approx(4.0)


async def test_ducaheat_child_lock_unavailable_while_cloud_down(
    hass: HomeAssistant,
    cloud: FakeCloud,
    samples: dict,
) -> None:
    """The Ducaheat child lock also follows the state poll's success."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        unique_id=USERNAME,
        data={"username": USERNAME, "password": PASSWORD, CONF_BRAND: BRAND_DUCAHEAT},
    )
    with (
        patch.object(DucaheatRESTClient, "get_node_settings", cloud.get_node_settings),
        patch.object(DucaheatRESTClient, "get_node_samples", cloud.get_node_samples),
        patch.object(
            DucaheatBackend,
            "create_ws_client",
            lambda _self, hass, *a, **kw: cloud.create_ws_client(hass, *a, **kw),
        ),
    ):
        await _setup(hass, entry)
        locks = [
            e.entity_id
            for e in er.async_entries_for_config_entry(
                er.async_get(hass), entry.entry_id
            )
            if e.domain == "lock"
        ]
        assert locks
        for entity_id in locks:
            assert hass.states.get(entity_id).state != STATE_UNAVAILABLE

        cloud.get_node_settings.side_effect = ClientError("cloud down")
        cloud.get_node_samples.side_effect = ClientError("cloud down")
        await _refresh(hass, entry, "coordinator")

        for entity_id in locks:
            assert hass.states.get(entity_id).state == STATE_UNAVAILABLE
