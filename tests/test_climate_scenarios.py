"""Climate user scenarios on real Home Assistant, faked only at the REST boundary."""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, patch

from homeassistant.components.climate import HVACMode
from homeassistant.const import ATTR_ENTITY_ID, ATTR_TEMPERATURE
from homeassistant.core import HomeAssistant
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.termoweb import (
    climate as climate_module,
    entity as entity_module,
)
from custom_components.termoweb.backend.rest_client import RESTClient

from tests.fakes.cloud import FakeCloud

HTR = "climate.living_room"
ACM = "climate.store"
# Long enough that two service calls made back to back land in one batch.
DEBOUNCE = 0.05


class FakeHome:
    """A heater (addr 1) and an accumulator (addr 2) behind the fake cloud."""

    def __init__(self, cloud: FakeCloud) -> None:
        """Serve the node settings and record every write."""
        self.settings: dict[tuple[str, str], dict[str, Any]] = {
            ("htr", "1"): {
                "mode": "manual",
                "state": "on",
                "stemp": "21.0",
                "units": "C",
            },
            ("acm", "2"): {
                "mode": "auto",
                "state": "off",
                "stemp": "19.0",
                "units": "C",
                "boost_active": False,
                "boost_time": 120,
                "boost_temp": "24.0",
            },
        }
        cloud.get_nodes.return_value = {
            "nodes": [
                {"type": "htr", "addr": 1, "name": "Living room"},
                {"type": "acm", "addr": 2, "name": "Store"},
            ]
        }
        cloud.get_node_settings.side_effect = self._get
        self.settings_writes = AsyncMock(side_effect=self._set)
        self.boost_writes = AsyncMock(return_value=None)
        self.preset_writes = AsyncMock(return_value=None)

    async def _get(self, _dev_id: str, node: tuple[str, Any]) -> dict[str, Any]:
        """Return a copy of the node's current settings."""
        return dict(self.settings[(node[0], str(node[1]))])

    async def _set(self, _dev_id: str, node: tuple[str, Any], **kwargs: Any) -> None:
        """Apply a mode/setpoint write to the stored settings."""
        current = self.settings[(node[0], str(node[1]))]
        for key in ("mode", "stemp"):
            if kwargs.get(key) is not None:
                current[key] = kwargs[key]

    def last_settings_write(self) -> dict[str, Any]:
        """Return the keyword arguments of the most recent settings write."""
        return self.settings_writes.await_args.kwargs


@pytest.fixture
def home(cloud: FakeCloud) -> Generator[FakeHome]:
    """Patch the REST write methods and shorten the write debounce."""
    fake = FakeHome(cloud)
    with (
        patch.object(RESTClient, "set_node_settings", fake.settings_writes),
        patch.object(RESTClient, "set_acm_boost_state", fake.boost_writes),
        patch.object(RESTClient, "set_acm_extra_options", fake.preset_writes),
        patch.object(climate_module, "_WRITE_DEBOUNCE", DEBOUNCE),
        patch.object(entity_module, "WS_ECHO_FALLBACK_REFRESH", 0),
    ):
        yield fake


async def _setup(hass: HomeAssistant, entry: MockConfigEntry) -> None:
    """Add the entry to hass and set it up."""
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()


async def _service(
    hass: HomeAssistant, domain: str, service: str, entity_id: str, **data: Any
) -> None:
    """Call a service on ``entity_id`` without waiting for the debounced write."""
    await hass.services.async_call(
        domain, service, {ATTR_ENTITY_ID: entity_id, **data}, blocking=True
    )


async def _settle(hass: HomeAssistant) -> None:
    """Let the debounced write and its follow-up refresh run."""
    await asyncio.sleep(DEBOUNCE * 3)
    for _ in range(3):
        await asyncio.sleep(0)
        await hass.async_block_till_done()


async def test_off_right_after_new_setpoint_keeps_heater_off(
    hass: HomeAssistant, home: FakeHome, config_entry: MockConfigEntry
) -> None:
    """Off pressed while a new setpoint is still batched wins: the heater turns off."""
    await _setup(hass, config_entry)

    await _service(hass, "climate", "set_temperature", HTR, **{ATTR_TEMPERATURE: 23})
    await _service(hass, "climate", "turn_off", HTR)
    await _settle(hass)

    home.settings_writes.assert_awaited_once()
    assert home.last_settings_write()["mode"] == "off"
    assert hass.states.get(HTR).state == HVACMode.OFF


async def test_auto_right_after_new_setpoint_selects_auto(
    hass: HomeAssistant, home: FakeHome, config_entry: MockConfigEntry
) -> None:
    """Auto picked while a new setpoint is still batched runs the schedule."""
    await _setup(hass, config_entry)

    await _service(hass, "climate", "set_temperature", HTR, **{ATTR_TEMPERATURE: 23})
    await _service(hass, "climate", "set_hvac_mode", HTR, hvac_mode=HVACMode.AUTO)
    await _settle(hass)

    home.settings_writes.assert_awaited_once()
    assert home.last_settings_write()["mode"] == "auto"
    assert hass.states.get(HTR).state == HVACMode.AUTO
