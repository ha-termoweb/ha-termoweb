"""Entity services raise errors instead of logging, on real Home Assistant."""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, patch

from aiohttp import ClientError
from homeassistant.const import ATTR_ENTITY_ID
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.helpers import entity_registry as er
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.termoweb.backend.ducaheat import (
    DucaheatBackend,
    DucaheatRESTClient,
)
from custom_components.termoweb.backend.rest_client import RESTClient
from custom_components.termoweb.const import BRAND_DUCAHEAT, CONF_BRAND, DOMAIN
from custom_components.termoweb.entities import climate as climate_module
from custom_components.termoweb.entities import heater as heater_module
from custom_components.termoweb.entities.heater import get_boost_temperature

from .conftest import PASSWORD, USERNAME, FakeCloud

HTR = "climate.living_room"
ACM = "climate.store"
BOOST_TEMP = "number.store_boost_temperature"
BOOST_DURATION = "number.store_boost_duration"
PRIORITY = "number.living_room_priority"


class Writes:
    """Backend write mocks patched onto the REST client."""

    def __init__(self) -> None:
        """Create succeeding write mocks."""
        self.settings = AsyncMock(return_value=None)
        self.extra_options = AsyncMock(return_value=None)
        self.boost_state = AsyncMock(return_value=None)
        self.priority = AsyncMock(return_value=None)

    def fail_all(self) -> None:
        """Make every backend write fail as if the cloud rejected it."""
        for mock in (
            self.settings,
            self.extra_options,
            self.boost_state,
            self.priority,
        ):
            mock.side_effect = ClientError("rejected")


@pytest.fixture
def node_settings(cloud: FakeCloud) -> dict[tuple[str, str], dict[str, Any]]:
    """Serve a heater (addr 1) and an accumulator (addr 2) behind the fake cloud."""
    settings = {
        ("htr", "1"): {
            "mode": "manual",
            "state": "off",
            "stemp": "21.0",
            "units": "C",
            "ptemp": ["7.0", "17.0", "21.0"],
        },
        ("acm", "2"): {"mode": "auto", "state": "off", "stemp": "19.0", "units": "C"},
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
    return settings


@pytest.fixture
def writes(node_settings: dict) -> Generator[Writes]:
    """Patch the REST client's write methods."""
    fake = Writes()
    with (
        patch.object(RESTClient, "set_node_settings", fake.settings),
        patch.object(RESTClient, "set_acm_extra_options", fake.extra_options),
        patch.object(RESTClient, "set_acm_boost_state", fake.boost_state),
        patch.object(RESTClient, "set_node_priority", fake.priority),
        patch.object(climate_module, "_WRITE_DEBOUNCE", 0),
        patch.object(heater_module, "WS_ECHO_FALLBACK_REFRESH", 0),
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
    """Call a service on one entity and wait for follow-up work."""
    await hass.services.async_call(
        domain, service, {ATTR_ENTITY_ID: entity_id, **data}, blocking=True
    )
    for _ in range(5):
        await asyncio.sleep(0)
        await hass.async_block_till_done()


def _entity(hass: HomeAssistant, entity_id: str) -> Any:
    """Return the entity object behind ``entity_id``."""
    return hass.data[entity_id.partition(".")[0]].get_entity(entity_id)


@pytest.mark.parametrize(
    ("service", "data"),
    [
        ("set_acm_preset", {"minutes": 60}),
        ("start_boost", {}),
        ("cancel_boost", {}),
    ],
)
async def test_accumulator_services_reject_heaters(
    hass: HomeAssistant,
    writes: Writes,
    config_entry: MockConfigEntry,
    service: str,
    data: dict,
) -> None:
    """Accumulator-only services on a heater fail validation."""
    await _setup(hass, config_entry)

    with pytest.raises(ServiceValidationError, match="only applies to accumulator"):
        await _service(hass, DOMAIN, service, HTR, **data)


async def test_set_schedule_backend_failure_raises(
    hass: HomeAssistant, writes: Writes, config_entry: MockConfigEntry
) -> None:
    """A rejected schedule write fails the service call."""
    await _setup(hass, config_entry)
    prog = [1] * 168
    await _service(hass, DOMAIN, "set_schedule", HTR, prog=prog)
    assert writes.settings.await_args.kwargs["prog"] == prog

    writes.fail_all()
    with pytest.raises(HomeAssistantError, match="Schedule write"):
        await _service(hass, DOMAIN, "set_schedule", HTR, prog=[0] * 168)


async def test_set_schedule_rejects_invalid_programs(
    hass: HomeAssistant, writes: Writes, config_entry: MockConfigEntry
) -> None:
    """The entity validates programs that bypass the service schema."""
    await _setup(hass, config_entry)
    heater = _entity(hass, HTR)

    for prog in ([0] * 10, [5] * 168, ["x"] * 168):
        with pytest.raises(ServiceValidationError):
            await heater.async_set_schedule(prog)
    writes.settings.assert_not_awaited()


async def test_partial_preset_temperatures_merge_with_current(
    hass: HomeAssistant, writes: Writes, config_entry: MockConfigEntry
) -> None:
    """Setting only the day preset keeps the current cold and night presets."""
    await _setup(hass, config_entry)

    await _service(hass, DOMAIN, "set_preset_temperatures", HTR, day=22.5)

    assert writes.settings.await_args.kwargs["ptemp"] == [7.0, 17.0, 22.5]


async def test_preset_temperatures_validation(
    hass: HomeAssistant,
    writes: Writes,
    node_settings: dict,
    config_entry: MockConfigEntry,
) -> None:
    """Missing or invalid preset temperatures fail validation."""
    del node_settings[("htr", "1")]["ptemp"]
    await _setup(hass, config_entry)
    heater = _entity(hass, HTR)

    with pytest.raises(ServiceValidationError, match="at least one"):
        await _service(hass, DOMAIN, "set_preset_temperatures", HTR)
    with pytest.raises(ServiceValidationError, match="unknown"):
        await _service(hass, DOMAIN, "set_preset_temperatures", HTR, night=16)
    with pytest.raises(ServiceValidationError, match="3 values"):
        await heater.async_set_preset_temperatures(ptemp=[1.0, 2.0])
    with pytest.raises(ServiceValidationError, match="Invalid preset"):
        await heater.async_set_preset_temperatures(ptemp=["a", "b", "c"])
    writes.settings.assert_not_awaited()

    await _service(
        hass, DOMAIN, "set_preset_temperatures", HTR, cold=6, night=16, day=20
    )
    assert writes.settings.await_args.kwargs["ptemp"] == [6.0, 16.0, 20.0]


async def test_preset_temperatures_backend_failure_raises(
    hass: HomeAssistant, writes: Writes, config_entry: MockConfigEntry
) -> None:
    """A rejected preset write fails the service call."""
    await _setup(hass, config_entry)
    writes.fail_all()

    with pytest.raises(HomeAssistantError, match="Preset write"):
        await _service(hass, DOMAIN, "set_preset_temperatures", HTR, ptemp=[7, 17, 21])


async def test_set_acm_preset_validation_and_failure(
    hass: HomeAssistant, writes: Writes, config_entry: MockConfigEntry
) -> None:
    """set_acm_preset validates its input and raises when the write fails."""
    await _setup(hass, config_entry)
    acm = _entity(hass, ACM)

    with pytest.raises(ServiceValidationError, match="minutes and/or temperature"):
        await _service(hass, DOMAIN, "set_acm_preset", ACM)
    for minutes in (0, 90):
        with pytest.raises(ServiceValidationError, match="Boost duration"):
            await acm.async_set_acm_preset(minutes=minutes)
    with pytest.raises(ServiceValidationError, match="Invalid boost temperature"):
        await acm.async_set_acm_preset(temperature="warm")

    writes.fail_all()
    with pytest.raises(HomeAssistantError, match="Boost preset write"):
        await _service(hass, DOMAIN, "set_acm_preset", ACM, temperature=22)


async def test_boost_backend_failures_raise(
    hass: HomeAssistant, writes: Writes, config_entry: MockConfigEntry
) -> None:
    """Rejected boost start/cancel writes fail the service call."""
    await _setup(hass, config_entry)
    writes.fail_all()

    with pytest.raises(HomeAssistantError, match="Boost start"):
        await _service(hass, DOMAIN, "start_boost", ACM, minutes=60)
    with pytest.raises(HomeAssistantError, match="Boost cancel"):
        await _service(hass, DOMAIN, "cancel_boost", ACM)


async def test_start_boost_without_setpoint_raises(
    hass: HomeAssistant,
    writes: Writes,
    node_settings: dict,
    config_entry: MockConfigEntry,
) -> None:
    """Boost cannot start before the accumulator reported a setpoint."""
    del node_settings[("acm", "2")]["stemp"]
    await _setup(hass, config_entry)

    with pytest.raises(HomeAssistantError, match="setpoint"):
        await _service(hass, DOMAIN, "start_boost", ACM)
    writes.boost_state.assert_not_awaited()


async def test_write_without_backend_raises(
    hass: HomeAssistant, writes: Writes, config_entry: MockConfigEntry
) -> None:
    """A write while the entry's runtime is gone raises instead of passing."""
    await _setup(hass, config_entry)
    heater = _entity(hass, HTR)
    runtime = hass.data[DOMAIN].pop(config_entry.entry_id)
    try:
        with pytest.raises(HomeAssistantError, match="backend unavailable"):
            await heater.async_set_schedule([0] * 168)
    finally:
        hass.data[DOMAIN][config_entry.entry_id] = runtime


async def test_invalid_modes_and_temperatures_raise(
    hass: HomeAssistant, writes: Writes, config_entry: MockConfigEntry
) -> None:
    """Unsupported modes and bad temperatures raise validation errors."""
    await _setup(hass, config_entry)
    heater = _entity(hass, HTR)
    acm = _entity(hass, ACM)

    with pytest.raises(ServiceValidationError, match="Invalid temperature"):
        await heater.async_set_temperature(temperature="hot")
    with pytest.raises(ServiceValidationError, match="Unsupported hvac_mode"):
        await heater.async_set_hvac_mode("cool")
    with pytest.raises(ServiceValidationError, match="preset mode"):
        await acm.async_set_hvac_mode("boost")
    with pytest.raises(ServiceValidationError, match="for accumulator"):
        await acm.async_set_hvac_mode("heat")
    with pytest.raises(ServiceValidationError, match="Unsupported preset_mode"):
        await acm.async_set_preset_mode("turbo")
    writes.settings.assert_not_awaited()


async def test_debounced_write_failure_is_logged(
    hass: HomeAssistant,
    writes: Writes,
    config_entry: MockConfigEntry,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A setpoint write runs after the call returns, so its failure is logged."""
    await _setup(hass, config_entry)
    writes.fail_all()

    await _service(hass, "climate", "set_temperature", HTR, temperature=22)

    writes.settings.assert_awaited()
    assert "Mode/setpoint write for htr 1 failed" in caplog.text
    assert hass.states.get(HTR).attributes["temperature"] == 21.0


async def test_priority_errors(
    hass: HomeAssistant, writes: Writes, config_entry: MockConfigEntry
) -> None:
    """Priority writes raise on bad values and backend failures."""
    await _setup(hass, config_entry)
    await _service(hass, "number", "set_value", PRIORITY, value=5)
    assert writes.priority.await_args.kwargs["priority"] == 5

    with pytest.raises(ServiceValidationError, match="Priority"):
        await _entity(hass, PRIORITY).async_set_native_value(31)

    writes.fail_all()
    with pytest.raises(HomeAssistantError, match="Priority write"):
        await _service(hass, "number", "set_value", PRIORITY, value=6)

    # A backend that already raises HomeAssistantError is not wrapped again.
    writes.priority.side_effect = HomeAssistantError("not supported here")
    with pytest.raises(HomeAssistantError, match="^not supported here$"):
        await _service(hass, "number", "set_value", PRIORITY, value=7)


async def test_boost_temperature_not_kept_when_write_fails(
    hass: HomeAssistant, writes: Writes, config_entry: MockConfigEntry
) -> None:
    """The boost temperature number keeps its value when the device write fails."""
    await _setup(hass, config_entry)
    before = hass.states.get(BOOST_TEMP).state
    stored = get_boost_temperature(hass, config_entry.entry_id, "acm", "2")

    writes.fail_all()
    with pytest.raises(HomeAssistantError, match="Boost preset write"):
        await _service(hass, "number", "set_value", BOOST_TEMP, value=25)

    assert hass.states.get(BOOST_TEMP).state == before
    assert get_boost_temperature(hass, config_entry.entry_id, "acm", "2") == stored


async def test_boost_temperature_kept_when_write_succeeds(
    hass: HomeAssistant, writes: Writes, config_entry: MockConfigEntry
) -> None:
    """A successful device write updates the boost temperature number."""
    await _setup(hass, config_entry)

    await _service(hass, "number", "set_value", BOOST_TEMP, value=25)

    assert writes.extra_options.await_args.kwargs["boost_temp"] == 25.0
    assert float(hass.states.get(BOOST_TEMP).state) == 25.0


async def test_boost_numbers_reject_invalid_values(
    hass: HomeAssistant, writes: Writes, config_entry: MockConfigEntry
) -> None:
    """Out-of-range boost values raise validation errors."""
    await _setup(hass, config_entry)
    temperature = _entity(hass, BOOST_TEMP)

    with pytest.raises(ServiceValidationError, match="boost temperature"):
        await temperature.async_set_native_value(80)
    with pytest.raises(ServiceValidationError, match="boost duration"):
        await _entity(hass, BOOST_DURATION).async_set_native_value(0.5)
    writes.extra_options.assert_not_awaited()


async def test_child_lock_failure_raises(
    hass: HomeAssistant, cloud: FakeCloud, node_settings: dict
) -> None:
    """A rejected child-lock write fails the lock/unlock call."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        unique_id=USERNAME,
        data={"username": USERNAME, "password": PASSWORD, CONF_BRAND: BRAND_DUCAHEAT},
    )
    set_lock = AsyncMock(side_effect=ClientError("rejected"))
    with (
        patch.object(DucaheatRESTClient, "get_node_settings", cloud.get_node_settings),
        patch.object(DucaheatRESTClient, "get_node_samples", cloud.get_node_samples),
        patch.object(DucaheatRESTClient, "set_node_lock", set_lock),
        patch.object(
            DucaheatBackend,
            "create_ws_client",
            lambda _self, hass, *a, **kw: cloud.create_ws_client(hass, *a, **kw),
        ),
    ):
        await _setup(hass, entry)
        lock = next(
            e.entity_id
            for e in er.async_entries_for_config_entry(
                er.async_get(hass), entry.entry_id
            )
            if e.domain == "lock"
        )

        with pytest.raises(HomeAssistantError, match="Child lock"):
            await _service(hass, "lock", "lock", lock)
        with pytest.raises(HomeAssistantError, match="Child unlock"):
            await _service(hass, "lock", "unlock", lock)
