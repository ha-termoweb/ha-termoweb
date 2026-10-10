"""Temperatures in the device's own unit (°C or °F) on real Home Assistant."""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, patch

from homeassistant.components.climate import ATTR_MAX_TEMP, ATTR_MIN_TEMP
from homeassistant.components.number import ATTR_MAX, ATTR_MIN, ATTR_STEP
from homeassistant.const import (
    ATTR_ENTITY_ID,
    ATTR_TEMPERATURE,
    ATTR_UNIT_OF_MEASUREMENT,
    UnitOfTemperature,
)
from homeassistant.core import HomeAssistant, State
from homeassistant.util.unit_system import US_CUSTOMARY_SYSTEM
import pytest
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    mock_restore_cache,
)

from custom_components.termoweb.backend.rest_client import RESTClient
from custom_components.termoweb.entities import climate as climate_module

from .conftest import FakeCloud

HTR = "climate.living_room"
ACM = "climate.store"
HTR_TEMP = "sensor.living_room_temperature"
ACM_TEMP = "sensor.store_temperature"
BOOST_TEMP = "number.store_boost_temperature"


def _settings(units: str, stemp: str | None, mtemp: str) -> dict[str, Any]:
    """Return node settings reported in ``units``."""
    return {
        "mode": "manual",
        "state": "off",
        "stemp": stemp,
        "mtemp": mtemp,
        "units": units,
    }


@pytest.fixture
def writes(cloud: FakeCloud) -> Generator[AsyncMock]:
    """Serve a °F heater (addr 1) and a °C accumulator (addr 2); record writes."""
    settings = {
        ("htr", "1"): _settings("F", "68.0", "70.0"),
        ("acm", "2"): _settings("C", "19.0", "18.5"),
    }
    names = {"htr": "Living room", "acm": "Store"}
    cloud.get_nodes.return_value = {
        "nodes": [
            {"type": node_type, "addr": int(addr), "name": names[node_type]}
            for node_type, addr in settings
        ]
    }

    async def _get(_dev_id: str, node: tuple[str, str]) -> dict[str, Any]:
        return dict(settings[(node[0], str(node[1]))])

    cloud.get_node_settings.side_effect = _get
    mock = AsyncMock(return_value=None)
    mock.settings = settings  # tests may change node settings before setup
    with (
        patch.object(RESTClient, "set_node_settings", mock),
        patch.object(climate_module, "_WRITE_DEBOUNCE", 0),
        patch.object(climate_module, "_WS_ECHO_FALLBACK_REFRESH", 0),
    ):
        yield mock


async def _setup(hass: HomeAssistant, entry: MockConfigEntry) -> None:
    """Add the entry to hass and set it up."""
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()


async def _set_temperature(hass: HomeAssistant, entity_id: str, value: float) -> None:
    """Call climate.set_temperature and wait for the debounced write."""
    await hass.services.async_call(
        "climate",
        "set_temperature",
        {ATTR_ENTITY_ID: entity_id, ATTR_TEMPERATURE: value},
        blocking=True,
    )
    # The write runs in a background task; yield until it has run.
    for _ in range(5):
        await asyncio.sleep(0)
        await hass.async_block_till_done()


async def test_fahrenheit_climate_converted_to_metric(
    hass: HomeAssistant, writes: AsyncMock, config_entry: MockConfigEntry
) -> None:
    """A °F heater shows correctly converted °C values on a metric HA."""
    await _setup(hass, config_entry)

    attrs = hass.states.get(HTR).attributes
    assert attrs[ATTR_TEMPERATURE] == 20.0  # 68 °F
    assert attrs["current_temperature"] == 21.1  # 70 °F
    assert attrs[ATTR_MIN_TEMP] == 5.0
    assert attrs[ATTR_MAX_TEMP] == 30.0

    celsius = hass.states.get(ACM).attributes
    assert celsius[ATTR_TEMPERATURE] == 19.0
    assert celsius["current_temperature"] == 18.5
    assert celsius[ATTR_MIN_TEMP] == 5.0
    assert celsius[ATTR_MAX_TEMP] == 30.0


async def test_fahrenheit_set_temperature_written_in_fahrenheit(
    hass: HomeAssistant, writes: AsyncMock, config_entry: MockConfigEntry
) -> None:
    """A metric setpoint reaches a °F heater converted to °F, units F."""
    await _setup(hass, config_entry)

    await _set_temperature(hass, HTR, 22.0)

    kwargs = writes.await_args.kwargs
    assert kwargs["stemp"] == pytest.approx(71.6)
    assert kwargs["units"] == "F"


async def test_fahrenheit_setpoint_not_clamped_to_celsius_range(
    hass: HomeAssistant, writes: AsyncMock, config_entry: MockConfigEntry
) -> None:
    """On a °F HA, 75 °F is written as 75, not clamped to 30."""
    await hass.config.async_update(unit_system="us_customary")
    assert hass.config.units is US_CUSTOMARY_SYSTEM
    await _setup(hass, config_entry)

    attrs = hass.states.get(HTR).attributes
    assert attrs[ATTR_TEMPERATURE] == 68
    assert attrs[ATTR_MIN_TEMP] == 41
    assert attrs[ATTR_MAX_TEMP] == 86

    await _set_temperature(hass, HTR, 75)

    kwargs = writes.await_args.kwargs
    assert kwargs["stemp"] == 75.0
    assert kwargs["units"] == "F"


async def test_celsius_setpoint_clamped_to_device_range(
    hass: HomeAssistant, writes: AsyncMock, config_entry: MockConfigEntry
) -> None:
    """A °C node still clamps the setpoint to 5-30 °C."""
    await _setup(hass, config_entry)
    entity = hass.data["climate"].get_entity(ACM)

    await entity.async_set_temperature(temperature=40.0)
    for _ in range(5):
        await asyncio.sleep(0)
        await hass.async_block_till_done()

    kwargs = writes.await_args.kwargs
    assert kwargs["stemp"] == 30.0
    assert kwargs["units"] == "C"


async def test_temperature_sensor_uses_device_unit(
    hass: HomeAssistant, writes: AsyncMock, config_entry: MockConfigEntry
) -> None:
    """The room temperature sensor declares °F and HA converts it to °C."""
    await _setup(hass, config_entry)

    fahrenheit = hass.states.get(HTR_TEMP)
    assert fahrenheit.attributes[ATTR_UNIT_OF_MEASUREMENT] == UnitOfTemperature.CELSIUS
    assert float(fahrenheit.state) == pytest.approx(21.1, abs=0.05)

    celsius = hass.states.get(ACM_TEMP)
    assert celsius.attributes[ATTR_UNIT_OF_MEASUREMENT] == UnitOfTemperature.CELSIUS
    assert float(celsius.state) == 18.5


async def test_boost_temperature_number_limits_follow_units(
    hass: HomeAssistant, writes: AsyncMock, config_entry: MockConfigEntry
) -> None:
    """The accumulator boost temperature slider uses the device's unit and range."""
    writes.settings[("acm", "2")] = _settings("F", "71.0", "70.0")
    await _setup(hass, config_entry)

    state = hass.states.get(BOOST_TEMP)
    assert state.attributes[ATTR_UNIT_OF_MEASUREMENT] == UnitOfTemperature.FAHRENHEIT
    assert state.attributes[ATTR_MIN] == 41.0
    assert state.attributes[ATTR_MAX] == 86.0
    assert state.attributes[ATTR_STEP] == 1.0
    assert float(state.state) == 71.0


async def test_boost_temperature_number_default_in_fahrenheit(
    hass: HomeAssistant, writes: AsyncMock, config_entry: MockConfigEntry
) -> None:
    """Without a device setpoint the °F default is 68 °F (20 °C), not 20 °F."""
    writes.settings[("acm", "2")] = _settings("F", None, "70.0")
    await _setup(hass, config_entry)

    assert float(hass.states.get(BOOST_TEMP).state) == 68.0


async def test_boost_temperature_number_celsius_limits(
    hass: HomeAssistant, writes: AsyncMock, config_entry: MockConfigEntry
) -> None:
    """A °C accumulator keeps the 5-30 °C slider in 0.5 steps."""
    await _setup(hass, config_entry)

    state = hass.states.get(BOOST_TEMP)
    assert state.attributes[ATTR_UNIT_OF_MEASUREMENT] == UnitOfTemperature.CELSIUS
    assert state.attributes[ATTR_MIN] == 5.0
    assert state.attributes[ATTR_MAX] == 30.0
    assert state.attributes[ATTR_STEP] == 0.5
    assert float(state.state) == 19.0


async def test_boost_temperature_restore_out_of_range_uses_default(
    hass: HomeAssistant, writes: AsyncMock, config_entry: MockConfigEntry
) -> None:
    """A restored value outside the device range falls back to the default."""
    mock_restore_cache(hass, [State(BOOST_TEMP, "150")])
    await _setup(hass, config_entry)

    assert float(hass.states.get(BOOST_TEMP).state) == 20.0
