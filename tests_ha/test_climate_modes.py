"""Climate HVAC modes, presets and turn_on/turn_off on real Home Assistant."""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, patch

from homeassistant.components.climate import (
    ATTR_HVAC_ACTION,
    ATTR_HVAC_MODES,
    ATTR_PRESET_MODE,
    ATTR_PRESET_MODES,
    ClimateEntityFeature,
    HVACMode,
)
from homeassistant.const import ATTR_ENTITY_ID, ATTR_SUPPORTED_FEATURES
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.termoweb.backend.rest_client import RESTClient
from custom_components.termoweb.entities import climate as climate_module
from custom_components.termoweb.entities import heater as heater_module

from .conftest import FakeCloud

HTR = "climate.living_room"
ACM = "climate.store"


class FakeDevices:
    """Node settings served by the fake cloud; writes update them."""

    def __init__(self, cloud: FakeCloud, settings: dict[tuple[str, str], dict]) -> None:
        """Serve ``settings`` keyed by (node type, addr) through ``cloud``."""
        self.settings = settings
        names = {"htr": "Living room", "acm": "Store"}
        cloud.get_nodes.return_value = {
            "nodes": [
                {"type": node_type, "addr": int(addr), "name": names[node_type]}
                for node_type, addr in settings
            ]
        }
        cloud.get_node_settings.side_effect = self._get
        self.writes = AsyncMock(side_effect=self._set)

    async def _get(self, _dev_id: str, node: tuple[str, str]) -> dict[str, Any]:
        """Return a copy of the node's current settings."""
        return dict(self.settings[(node[0], str(node[1]))])

    async def _set(self, _dev_id: str, node: tuple[str, str], **kwargs: Any) -> None:
        """Apply a settings write to the stored node settings."""
        current = self.settings[(node[0], str(node[1]))]
        for key in ("mode", "stemp"):
            if kwargs.get(key) is not None:
                current[key] = kwargs[key]


@pytest.fixture
def devices(cloud: FakeCloud) -> Generator[FakeDevices]:
    """Return a heater (addr 1) and an accumulator (addr 2) behind the fake cloud."""
    fake = FakeDevices(
        cloud,
        {
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
            },
        },
    )
    with (
        patch.object(RESTClient, "set_node_settings", fake.writes),
        patch.object(climate_module, "_WRITE_DEBOUNCE", 0),
        patch.object(heater_module, "WS_ECHO_FALLBACK_REFRESH", 0),
    ):
        yield fake


async def _setup(hass: HomeAssistant, entry: MockConfigEntry) -> None:
    """Add the entry to hass and set it up."""
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()


async def _call(hass: HomeAssistant, service: str, entity_id: str, **data: Any) -> None:
    """Call a climate service and wait for the debounced write and refresh."""
    await hass.services.async_call(
        "climate", service, {ATTR_ENTITY_ID: entity_id, **data}, blocking=True
    )
    # The write runs in a background task (the idle fake websocket is one too,
    # so wait_background_tasks would never return); yield until it has run.
    for _ in range(5):
        await asyncio.sleep(0)
        await hass.async_block_till_done()


async def test_turn_off_and_on_restore_previous_mode(
    hass: HomeAssistant, devices: FakeDevices, config_entry: MockConfigEntry
) -> None:
    """turn_off switches the heater off; turn_on brings back manual (Heat)."""
    await _setup(hass, config_entry)
    state = hass.states.get(HTR)
    features = state.attributes[ATTR_SUPPORTED_FEATURES]
    assert features & ClimateEntityFeature.TURN_ON
    assert features & ClimateEntityFeature.TURN_OFF
    assert state.state == HVACMode.HEAT

    await _call(hass, "turn_off", HTR)
    assert devices.writes.await_args.kwargs["mode"] == "off"
    assert hass.states.get(HTR).state == HVACMode.OFF

    await _call(hass, "turn_on", HTR)
    assert devices.writes.await_args.kwargs["mode"] == "manual"
    assert devices.writes.await_args.kwargs["stemp"] == 21.0
    assert hass.states.get(HTR).state == HVACMode.HEAT

    devices.writes.reset_mock()
    await _call(hass, "turn_on", HTR)
    devices.writes.assert_not_awaited()


async def test_turn_on_defaults_to_auto(
    hass: HomeAssistant, devices: FakeDevices, config_entry: MockConfigEntry
) -> None:
    """A heater that was off before HA started turns on in Auto."""
    devices.settings[("htr", "1")]["mode"] = "off"
    await _setup(hass, config_entry)

    await _call(hass, "toggle", HTR)

    assert devices.writes.await_args.kwargs["mode"] == "auto"
    assert hass.states.get(HTR).state == HVACMode.AUTO


async def test_accumulator_turn_on_uses_auto(
    hass: HomeAssistant, devices: FakeDevices, config_entry: MockConfigEntry
) -> None:
    """Accumulators turn off and back on into Auto."""
    await _setup(hass, config_entry)
    features = hass.states.get(ACM).attributes[ATTR_SUPPORTED_FEATURES]
    assert features & ClimateEntityFeature.TURN_ON
    assert features & ClimateEntityFeature.TURN_OFF

    await _call(hass, "turn_off", ACM)
    assert hass.states.get(ACM).state == HVACMode.OFF
    await _call(hass, "turn_on", ACM)

    assert devices.writes.await_args.kwargs["mode"] == "auto"
    assert hass.states.get(ACM).state == HVACMode.AUTO


async def test_unknown_mode_and_state_are_unknown(
    hass: HomeAssistant, devices: FakeDevices, config_entry: MockConfigEntry
) -> None:
    """An unrecognised mode is not reported as Heat, nor an odd state as heating."""
    devices.settings[("htr", "1")].update(mode="mystery", state="error")
    devices.settings[("acm", "2")].update(mode="mystery", state="error")
    await _setup(hass, config_entry)

    for entity_id in (HTR, ACM):
        state = hass.states.get(entity_id)
        assert state.state == "unknown"
        assert state.attributes.get(ATTR_HVAC_ACTION) is None


async def test_no_mode_reported_is_unknown(
    hass: HomeAssistant, devices: FakeDevices, config_entry: MockConfigEntry
) -> None:
    """A node that reports no mode has an unknown HVAC mode and preset."""
    devices.settings[("htr", "1")] = {"units": "C"}
    devices.settings[("acm", "2")] = {"units": "C"}
    await _setup(hass, config_entry)

    for entity_id in (HTR, ACM):
        state = hass.states.get(entity_id)
        assert state.state == "unknown"
        assert state.attributes[ATTR_PRESET_MODE] is None
        assert state.attributes.get(ATTR_HVAC_ACTION) is None


async def test_heating_and_idle_actions(
    hass: HomeAssistant, devices: FakeDevices, config_entry: MockConfigEntry
) -> None:
    """Known device states map to heating and idle."""
    devices.settings[("acm", "2")]["state"] = "idle"
    await _setup(hass, config_entry)

    assert hass.states.get(HTR).attributes[ATTR_HVAC_ACTION] == "heating"
    assert hass.states.get(ACM).attributes[ATTR_HVAC_ACTION] == "idle"


async def test_accumulator_manual_mode_is_listed(
    hass: HomeAssistant, devices: FakeDevices, config_entry: MockConfigEntry
) -> None:
    """An accumulator in manual mode reports Heat and lists it in hvac_modes."""
    devices.settings[("acm", "2")]["mode"] = "manual"
    await _setup(hass, config_entry)

    state = hass.states.get(ACM)
    assert state.state == HVACMode.HEAT
    assert state.state in state.attributes[ATTR_HVAC_MODES]


async def test_accumulator_without_manual_mode_lists_off_and_auto(
    hass: HomeAssistant, devices: FakeDevices, config_entry: MockConfigEntry
) -> None:
    """Heat is not offered for an accumulator that is not in manual mode."""
    await _setup(hass, config_entry)

    assert hass.states.get(ACM).attributes[ATTR_HVAC_MODES] == ["off", "auto"]


async def test_temporary_override_is_display_only(
    hass: HomeAssistant, devices: FakeDevices, config_entry: MockConfigEntry
) -> None:
    """temporary_override shows as the preset but cannot be selected."""
    devices.settings[("htr", "1")]["mode"] = "modified_auto"
    await _setup(hass, config_entry)

    state = hass.states.get(HTR)
    assert state.state == HVACMode.AUTO
    assert state.attributes[ATTR_PRESET_MODES] == ["none"]
    assert state.attributes[ATTR_PRESET_MODE] == "temporary_override"

    with pytest.raises(ServiceValidationError):
        await _call(hass, "set_preset_mode", HTR, preset_mode="temporary_override")
    devices.writes.assert_not_awaited()


async def test_preset_none_ends_temporary_override(
    hass: HomeAssistant, devices: FakeDevices, config_entry: MockConfigEntry
) -> None:
    """Selecting preset none during an override resumes the Auto program."""
    devices.settings[("htr", "1")]["mode"] = "modified_auto"
    await _setup(hass, config_entry)

    await _call(hass, "set_preset_mode", HTR, preset_mode="none")

    assert devices.writes.await_args.kwargs["mode"] == "auto"
    assert hass.states.get(HTR).attributes[ATTR_PRESET_MODE] == "none"

    devices.writes.reset_mock()
    await _call(hass, "set_preset_mode", HTR, preset_mode="none")
    devices.writes.assert_not_awaited()
