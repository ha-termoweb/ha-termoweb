"""Climate platform on real Home Assistant: setup, writes, accumulator boost."""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from datetime import datetime
from typing import Any
from unittest.mock import patch

from freezegun.api import FrozenDateTimeFactory
from homeassistant.components.climate import (
    ATTR_CURRENT_TEMPERATURE,
    ATTR_HVAC_ACTION,
    ATTR_PRESET_MODE,
    ATTR_PRESET_MODES,
    HVACMode,
)
from homeassistant.const import ATTR_ICON, ATTR_TEMPERATURE
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.util import dt as dt_util
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry
import voluptuous as vol

from custom_components.termoweb import climate as climate_module
from custom_components.termoweb.const import BRAND_DUCAHEAT, DOMAIN

from .conftest import DEV_ID, FakeCloud
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
