"""Tests for the climate platform, including temperature units."""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, patch

from freezegun.api import FrozenDateTimeFactory
from homeassistant.components.climate import (
    ATTR_CURRENT_TEMPERATURE,
    ATTR_HVAC_ACTION,
    ATTR_HVAC_MODES,
    ATTR_MAX_TEMP,
    ATTR_MIN_TEMP,
    ATTR_PRESET_MODE,
    ATTR_PRESET_MODES,
    ClimateEntityFeature,
    HVACMode,
)
from homeassistant.components.number import ATTR_MAX, ATTR_MIN, ATTR_STEP
from homeassistant.const import (
    ATTR_ENTITY_ID,
    ATTR_ICON,
    ATTR_SUPPORTED_FEATURES,
    ATTR_TEMPERATURE,
    ATTR_UNIT_OF_MEASUREMENT,
    UnitOfTemperature,
)
from homeassistant.core import HomeAssistant, State
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_system import US_CUSTOMARY_SYSTEM
import pytest
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    mock_restore_cache,
)
import voluptuous as vol

from custom_components.termoweb import (
    climate as climate_module,
    entity as entity_module,
)
from custom_components.termoweb.backend.rest_client import RESTClient
from custom_components.termoweb.const import BRAND_DUCAHEAT, DOMAIN
from tests.fakes.cloud import DEV_ID, FakeCloud

from .fakes.entities import FakeNodes, call, settle, setup_entry

HTR = "climate.living_room"
ACM = "climate.store"
THM = "climate.thermostat_4"
PROG = [0, 1, 2] * 56


def _heater(**settings: Any) -> dict[str, Any]:
    """Return heater settings in manual mode at 21 °C, updated by ``settings``."""
    return {"mode": "manual", "state": "on", "stemp": "21.0", "units": "C", **settings}


def _accumulator(**settings: Any) -> dict[str, Any]:
    """Return accumulator settings in auto mode with a boost setpoint."""
    return {
        "mode": "auto",
        "state": "off",
        "stemp": "19.0",
        "boost_temp": "23.0",
        "boost_time": 120,
        "units": "C",
        **settings,
    }


@pytest.fixture
def nodes(cloud: FakeCloud) -> Generator[FakeNodes]:
    """Serve a heater (1), an accumulator (3), a thermostat (4) and a power monitor."""
    fake = FakeNodes(
        cloud,
        {
            ("htr", "1"): ("Living room", _heater(ptemp=["7.0", "17.0", "21.0"])),
            ("acm", "3"): ("Store", _accumulator()),
            ("thm", "4"): (None, _heater(stemp="21.5", mtemp="20.3")),
            ("pmo", "5"): (None, {}),
        },
    )
    with fake.patched():
        yield fake


async def test_setup_creates_one_climate_per_heating_node(
    hass: HomeAssistant, nodes: FakeNodes, config_entry: MockConfigEntry
) -> None:
    """Heaters, accumulators and thermostats get a climate entity on their device."""
    entry = await setup_entry(hass, config_entry)

    registry = er.async_get(hass)
    climates = {
        e.unique_id: e
        for e in er.async_entries_for_config_entry(registry, entry.entry_id)
        if e.domain == "climate"
    }
    assert set(climates) == {
        f"{DOMAIN}:{DEV_ID}:htr:1:climate",
        f"{DOMAIN}:{DEV_ID}:acm:3:climate",
        f"{DOMAIN}:{DEV_ID}:thm:4:climate",
    }
    devices = dr.async_get(hass)
    gateway = devices.async_get_device_by_identifier((DOMAIN, DEV_ID), entry.entry_id)
    for (node_type, addr), model in {
        ("htr", "1"): "Heater",
        ("acm", "3"): "Accumulator",
    }.items():
        entity = climates[f"{DOMAIN}:{DEV_ID}:{node_type}:{addr}:climate"]
        device = devices.async_get(entity.device_id)
        assert (DOMAIN, DEV_ID, addr) in device.identifiers
        assert device.model == model
        assert device.via_device_id == gateway.id

    for service in (
        "set_schedule",
        "set_preset_temperatures",
        "set_acm_preset",
        "start_boost",
        "cancel_boost",
    ):
        assert hass.services.has_service(DOMAIN, service)


async def test_thermostat_reports_its_settings(
    hass: HomeAssistant, nodes: FakeNodes, config_entry: MockConfigEntry
) -> None:
    """A thermostat climate shows mode, action and both temperatures."""
    await setup_entry(hass, config_entry)

    state = hass.states.get(THM)
    assert state.state == HVACMode.HEAT
    assert state.attributes[ATTR_HVAC_ACTION] == "heating"
    assert state.attributes[ATTR_CURRENT_TEMPERATURE] == 20.3
    assert state.attributes[ATTR_TEMPERATURE] == 21.5


async def test_program_slot_attributes_follow_the_clock(
    hass: HomeAssistant,
    nodes: FakeNodes,
    config_entry: MockConfigEntry,
    freezer: FrozenDateTimeFactory,
) -> None:
    """The active weekly-program slot and its preset temperature are exposed."""
    # Monday 01:00 -> program index 1 -> slot 1 (night, 17 °C).
    freezer.move_to(datetime(2024, 1, 1, 1, 0, tzinfo=dt_util.get_default_time_zone()))
    nodes.settings[("htr", "1")]["prog"] = PROG
    await setup_entry(hass, config_entry)

    attrs = hass.states.get(HTR).attributes
    assert attrs["prog"] == PROG
    assert attrs["program_slot"] == "night"
    assert attrs["program_setpoint"] == 17.0


@pytest.mark.parametrize(
    ("mode", "device_state", "icon"),
    [
        ("off", "off", "mdi:radiator-off"),
        ("manual", "on", "mdi:radiator"),
        ("manual", "off", "mdi:radiator-disabled"),
    ],
)
async def test_icon_reflects_off_heating_and_idle(
    hass: HomeAssistant,
    nodes: FakeNodes,
    config_entry: MockConfigEntry,
    mode: str,
    device_state: str,
    icon: str,
) -> None:
    """The heater icon distinguishes off, heating and idle."""
    nodes.settings[("htr", "1")].update(mode=mode, state=device_state)
    await setup_entry(hass, config_entry)

    assert hass.states.get(HTR).attributes[ATTR_ICON] == icon


@pytest.mark.parametrize(
    ("mode", "written_mode"),
    [("auto", "modified_auto"), ("modified_auto", "modified_auto"), ("off", "manual")],
)
async def test_set_temperature_keeps_auto_programs_running(
    hass: HomeAssistant,
    nodes: FakeNodes,
    config_entry: MockConfigEntry,
    mode: str,
    written_mode: str,
) -> None:
    """A setpoint in Auto is a temporary override; otherwise it switches to manual."""
    nodes.settings[("htr", "1")]["mode"] = mode
    await setup_entry(hass, config_entry)

    await call(hass, "climate", "set_temperature", HTR, temperature=22.5)

    kwargs = nodes.set_settings.await_args.kwargs
    assert kwargs["mode"] == written_mode
    assert kwargs["stemp"] == 22.5
    assert hass.states.get(HTR).attributes[ATTR_TEMPERATURE] == 22.5


async def test_heat_mode_resends_the_current_setpoint(
    hass: HomeAssistant, nodes: FakeNodes, config_entry: MockConfigEntry
) -> None:
    """Switching to Heat sends manual together with the current setpoint."""
    nodes.settings[("htr", "1")]["mode"] = "auto"
    await setup_entry(hass, config_entry)

    await call(hass, "climate", "set_hvac_mode", HTR, hvac_mode="heat")

    kwargs = nodes.set_settings.await_args.kwargs
    assert (kwargs["mode"], kwargs["stemp"]) == ("manual", 21.0)


async def test_written_setpoint_survives_a_stale_poll(
    hass: HomeAssistant, nodes: FakeNodes, config_entry: MockConfigEntry
) -> None:
    """A poll still showing the old setpoint does not undo a fresh write."""
    entry = await setup_entry(hass, config_entry)
    nodes.set_settings.side_effect = None  # the cloud has not applied it yet

    await call(hass, "climate", "set_temperature", HTR, temperature=24)
    await entry.runtime_data.coordinator.async_refresh()
    await settle(hass)

    assert nodes.settings[("htr", "1")]["stemp"] == "21.0"
    assert hass.states.get(HTR).attributes[ATTR_TEMPERATURE] == 24


async def test_setpoint_queued_during_inflight_write_is_flushed(
    hass: HomeAssistant, nodes: FakeNodes, config_entry: MockConfigEntry
) -> None:
    """A setpoint made while the previous write is in flight is written after it."""
    await setup_entry(hass, config_entry)
    release = asyncio.Event()
    sent: list[float] = []

    async def _slow_write(*_args: Any, **kwargs: Any) -> None:
        sent.append(kwargs["stemp"])
        if len(sent) == 1:
            await release.wait()

    nodes.set_settings.side_effect = _slow_write
    await call(hass, "climate", "set_temperature", HTR, temperature=20)
    assert sent == [20.0]

    await call(hass, "climate", "set_temperature", HTR, temperature=22)
    assert sent == [20.0]
    release.set()
    await settle(hass)

    assert sent == [20.0, 22.0]
    assert hass.states.get(HTR).attributes[ATTR_TEMPERATURE] == 22.0


async def test_unloading_during_the_debounce_sends_nothing(
    hass: HomeAssistant, nodes: FakeNodes, config_entry: MockConfigEntry
) -> None:
    """Removing the entity before the debounced write runs cancels the write."""
    entry = await setup_entry(hass, config_entry)

    with patch.object(climate_module, "_WRITE_DEBOUNCE", 60):
        await call(hass, "climate", "set_temperature", HTR, temperature=25)
        assert await hass.config_entries.async_unload(entry.entry_id)
        await settle(hass)

    nodes.set_settings.assert_not_awaited()


async def test_set_schedule_writes_the_program(
    hass: HomeAssistant, nodes: FakeNodes, config_entry: MockConfigEntry
) -> None:
    """set_schedule sends the 168-slot program and shows it straight away."""
    await setup_entry(hass, config_entry)

    await call(hass, DOMAIN, "set_schedule", HTR, prog=PROG)

    args = nodes.set_settings.await_args
    assert args.args == (DEV_ID, ("htr", "1"))
    assert args.kwargs["prog"] == PROG
    assert args.kwargs["units"] == "C"
    assert hass.states.get(HTR).attributes["prog"] == PROG


@pytest.mark.parametrize(
    "prog", [[0] * 167, [3] * 168, ["x"] * 168], ids=["short", "range", "type"]
)
async def test_set_schedule_schema_rejects_bad_programs(
    hass: HomeAssistant,
    nodes: FakeNodes,
    config_entry: MockConfigEntry,
    prog: list[Any],
) -> None:
    """The service schema only accepts 168 values of 0, 1 or 2."""
    await setup_entry(hass, config_entry)

    with pytest.raises(vol.Invalid):
        await call(hass, DOMAIN, "set_schedule", HTR, prog=prog)
    nodes.set_settings.assert_not_awaited()


@pytest.mark.parametrize(("boosting", "cancel"), [(True, True), (False, False)])
async def test_ducaheat_accumulator_mode_change_cancels_a_running_boost(
    hass: HomeAssistant,
    cloud: FakeCloud,
    config_entry: MockConfigEntry,
    boosting: bool,
    cancel: bool,
) -> None:
    """On Ducaheat, a mode write during a boost also cancels the boost."""
    fake = FakeNodes(
        cloud,
        {("acm", "3"): ("Store", _accumulator(mode="boost" if boosting else "auto"))},
    )
    fake.settings[("acm", "3")]["boost_active"] = boosting
    with fake.patched(ducaheat=True):
        await setup_entry(hass, config_entry, brand=BRAND_DUCAHEAT)
        await call(hass, "climate", "set_hvac_mode", ACM, hvac_mode="off")

    kwargs = fake.set_settings.await_args.kwargs
    assert kwargs["mode"] == "off"
    assert kwargs["cancel_boost"] is cancel


async def test_accumulator_in_boost_is_auto_with_boost_preset(
    hass: HomeAssistant,
    nodes: FakeNodes,
    config_entry: MockConfigEntry,
    freezer: FrozenDateTimeFactory,
) -> None:
    """A boosting accumulator reports Auto, preset boost and the boost metadata."""
    freezer.move_to(datetime(2024, 1, 1, 12, 0, tzinfo=dt_util.UTC))
    nodes.settings[("acm", "3")].update(
        mode="boost",
        state="on",
        boost_active=True,
        boost_remaining=30,
        charging=True,
        current_charge_per=40,
        target_charge_per=80,
    )
    await setup_entry(hass, config_entry)

    state = hass.states.get(ACM)
    assert state.state == HVACMode.AUTO
    assert state.attributes[ATTR_PRESET_MODES] == ["none", "boost"]
    assert state.attributes[ATTR_PRESET_MODE] == "boost"
    assert state.attributes["boost_active"] is True
    assert state.attributes["boost_minutes_remaining"] == 30
    assert dt_util.parse_datetime(state.attributes["boost_end"]) == datetime(
        2024, 1, 1, 12, 30, tzinfo=dt_util.UTC
    )
    assert state.attributes["boost_end_label"] is None
    assert state.attributes["preferred_boost_minutes"] == 120
    assert state.attributes["charging"] is True
    assert state.attributes["current_charge_per"] == 40
    assert state.attributes["target_charge_per"] == 80


async def test_accumulator_without_boost_shows_never(
    hass: HomeAssistant, nodes: FakeNodes, config_entry: MockConfigEntry
) -> None:
    """With no boost running the boost end reads Never."""
    await setup_entry(hass, config_entry)

    attrs = hass.states.get(ACM).attributes
    assert attrs["boost_active"] is False
    assert attrs["boost_minutes_remaining"] is None
    assert attrs["boost_end"] is None
    assert attrs["boost_end_label"] == "Never"


async def test_boost_preset_starts_and_stops_a_boost(
    hass: HomeAssistant, nodes: FakeNodes, config_entry: MockConfigEntry
) -> None:
    """Preset boost starts a boost of the preferred length; none cancels it."""
    await setup_entry(hass, config_entry)

    await call(hass, "climate", "set_preset_mode", ACM, preset_mode="boost")
    nodes.set_boost.assert_awaited_once_with(
        DEV_ID, "3", boost=True, boost_time=120, stemp=23.0, units="C"
    )
    state = hass.states.get(ACM)
    assert state.attributes[ATTR_PRESET_MODE] == "boost"
    assert state.attributes["boost_minutes_remaining"] == 120

    await call(hass, "climate", "set_preset_mode", ACM, preset_mode="boost")
    assert nodes.set_boost.await_count == 1  # already boosting

    await call(hass, "climate", "set_preset_mode", ACM, preset_mode="none")
    assert nodes.set_boost.await_args.kwargs["boost"] is False
    assert nodes.set_settings.await_args.kwargs["mode"] == "auto"
    state = hass.states.get(ACM)
    assert state.state == HVACMode.AUTO
    assert state.attributes[ATTR_PRESET_MODE] == "none"


@pytest.mark.parametrize(("data", "minutes"), [({"minutes": 180}, 180), ({}, 120)])
async def test_start_boost_service(
    hass: HomeAssistant,
    nodes: FakeNodes,
    config_entry: MockConfigEntry,
    data: dict[str, Any],
    minutes: int,
) -> None:
    """start_boost uses the given minutes, or the accumulator's own boost time."""
    await setup_entry(hass, config_entry)

    await call(hass, DOMAIN, "start_boost", ACM, **data)

    kwargs = nodes.set_boost.await_args.kwargs
    assert (kwargs["boost"], kwargs["boost_time"], kwargs["stemp"]) == (
        True,
        minutes,
        23.0,
    )
    assert hass.states.get(ACM).attributes[ATTR_PRESET_MODE] == "boost"


async def test_cancel_boost_service(
    hass: HomeAssistant, nodes: FakeNodes, config_entry: MockConfigEntry
) -> None:
    """cancel_boost stops the boost and clears the boost metadata."""
    nodes.settings[("acm", "3")].update(
        mode="boost", boost_active=True, boost_remaining=30
    )
    await setup_entry(hass, config_entry)

    await call(hass, DOMAIN, "cancel_boost", ACM)

    nodes.set_boost.assert_awaited_once()
    assert nodes.set_boost.await_args.kwargs["boost"] is False
    attrs = hass.states.get(ACM).attributes
    assert attrs[ATTR_PRESET_MODE] == "none"
    assert attrs["boost_active"] is False
    assert attrs["boost_minutes_remaining"] is None


async def test_set_acm_preset_writes_boost_defaults(
    hass: HomeAssistant, nodes: FakeNodes, config_entry: MockConfigEntry
) -> None:
    """set_acm_preset stores the default boost length and temperature."""
    await setup_entry(hass, config_entry)

    await call(hass, DOMAIN, "set_acm_preset", ACM, minutes="60", temperature=22.5)

    nodes.set_extra_options.assert_awaited_once_with(
        DEV_ID, "3", boost_time=60, boost_temp=22.5
    )
    assert hass.states.get(ACM).attributes["preferred_boost_minutes"] == 60


@pytest.mark.parametrize("service", ["start_boost", "set_acm_preset"])
@pytest.mark.parametrize(
    ("minutes", "accepted"),
    [
        (60, True),
        ("300", True),
        (600, True),
        (30, False),
        (125, False),
        (0, False),
        (660, False),
    ],
)
async def test_boost_minutes_schema(
    hass: HomeAssistant,
    nodes: FakeNodes,
    config_entry: MockConfigEntry,
    service: str,
    minutes: Any,
    accepted: bool,
) -> None:
    """Both boost services accept exactly 60-600 minutes in steps of 60."""
    await setup_entry(hass, config_entry)
    written = nodes.set_boost if service == "start_boost" else nodes.set_extra_options

    if accepted:
        await call(hass, DOMAIN, service, ACM, minutes=minutes)
        assert written.await_args.kwargs["boost_time"] == int(minutes)
    else:
        with pytest.raises(vol.Invalid):
            await call(hass, DOMAIN, service, ACM, minutes=minutes)
        written.assert_not_awaited()


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
        patch.object(entity_module, "WS_ECHO_FALLBACK_REFRESH", 0),
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
        patch.object(entity_module, "WS_ECHO_FALLBACK_REFRESH", 0),
    ):
        yield mock


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
