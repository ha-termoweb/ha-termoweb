"""Number entities (boost presets, priority, power limit) on real Home Assistant."""

from __future__ import annotations

import asyncio
from collections.abc import Generator
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from aiohttp import ClientError
from homeassistant.const import ATTR_ENTITY_ID, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import HomeAssistant, State
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.helpers.entity import EntityCategory
import pytest
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    mock_restore_cache,
)

from custom_components.termoweb import entity as entity_module, number
from custom_components.termoweb.backend.ducaheat import (
    DucaheatBackend,
    DucaheatRESTClient,
)
from custom_components.termoweb.backend.rest_client import RESTClient
from custom_components.termoweb.backend.termoweb_ws import TermoWebWSClient
from custom_components.termoweb.const import BRAND_DUCAHEAT, CONF_BRAND, DOMAIN
from custom_components.termoweb.coordinator import StateCoordinator
from custom_components.termoweb.identifiers import build_installation_entity_unique_id
from custom_components.termoweb.inventory import Inventory, build_node_inventory
from tests.fakes.cloud import DEV_ID, PASSWORD, USERNAME, FakeCloud

from .fakes.runtime import build_entry_runtime

ACM = "climate.store"
BOOST_TEMP = "number.store_boost_temperature"
BOOST_DURATION = "number.store_boost_duration"
PRIORITY = "number.living_room_priority"
POWER_LIMIT_PATH = f"/api/v2/devs/{DEV_ID}/htr_system/power_limit"
ACM_SETTINGS = {
    "mode": "auto",
    "state": "off",
    "stemp": "19.0",
    "units": "C",
    "boost_time": 120,
    "boost_temp": "22.0",
}


def _serve_accumulator(cloud: FakeCloud, settings: dict[str, Any]) -> None:
    """Serve one accumulator (addr 2) reporting ``settings``."""
    cloud.get_nodes.return_value = {
        "nodes": [{"type": "acm", "addr": 2, "name": "Store"}]
    }

    async def _get(_dev_id: str, _node: tuple[str, Any]) -> dict[str, Any]:
        return dict(settings)

    cloud.get_node_settings.side_effect = _get


@pytest.fixture
def extra_options(cloud: FakeCloud) -> Generator[AsyncMock]:
    """Serve one accumulator and patch the boost-defaults write."""
    _serve_accumulator(cloud, ACM_SETTINGS)
    write = AsyncMock(return_value=None)
    with patch.object(RESTClient, "set_acm_extra_options", write):
        yield write


async def _setup(hass: HomeAssistant, entry: MockConfigEntry) -> None:
    """Add the entry to hass and set it up."""
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()


async def _settle(hass: HomeAssistant) -> None:
    """Let follow-up tasks finish."""
    for _ in range(3):
        await asyncio.sleep(0)
        await hass.async_block_till_done()


async def _set(hass: HomeAssistant, entity_id: str, value: float) -> None:
    """Set a number value and let follow-up work finish."""
    await hass.services.async_call(
        "number", "set_value", {ATTR_ENTITY_ID: entity_id, "value": value}, True
    )
    await _settle(hass)


def _power_limit_entity(hass: HomeAssistant, entry: MockConfigEntry) -> str | None:
    """Return the entity_id of the power limit number, if it exists."""
    return next(
        (
            e.entity_id
            for e in er.async_entries_for_config_entry(
                er.async_get(hass), entry.entry_id
            )
            if e.unique_id.endswith("power_limit")
        ),
        None,
    )


# --- accumulator boost presets ---------------------------------------------


async def test_boost_numbers_are_config_entities(
    hass: HomeAssistant, extra_options: AsyncMock, config_entry: MockConfigEntry
) -> None:
    """Both boost presets are created as configuration entities."""
    await _setup(hass, config_entry)
    registry = er.async_get(hass)

    for entity_id in (BOOST_DURATION, BOOST_TEMP):
        assert registry.async_get(entity_id).entity_category is EntityCategory.CONFIG


async def test_boost_duration_writes_boost_time(
    hass: HomeAssistant, extra_options: AsyncMock, config_entry: MockConfigEntry
) -> None:
    """Setting the boost duration writes boost_time to the device."""
    await _setup(hass, config_entry)

    await _set(hass, BOOST_DURATION, 3)

    extra_options.assert_awaited_once()
    assert extra_options.await_args.kwargs["boost_time"] == 180
    assert extra_options.await_args.kwargs["boost_temp"] is None
    assert float(hass.states.get(BOOST_DURATION).state) == 3.0
    assert hass.states.get(ACM).attributes["preferred_boost_minutes"] == 180


@pytest.mark.parametrize("value", [1.01, float("nan")], ids=["off-step", "nan"])
async def test_boost_duration_rejects_unsupported_value(
    hass: HomeAssistant,
    extra_options: AsyncMock,
    config_entry: MockConfigEntry,
    value: float,
) -> None:
    """A slider value that is no allowed boost duration is refused unwritten."""
    await _setup(hass, config_entry)

    with pytest.raises(ServiceValidationError, match="Invalid boost duration"):
        await _set(hass, BOOST_DURATION, value)

    extra_options.assert_not_awaited()


async def test_boost_duration_not_kept_when_write_fails(
    hass: HomeAssistant, extra_options: AsyncMock, config_entry: MockConfigEntry
) -> None:
    """A rejected boost_time write raises and keeps the previous value."""
    await _setup(hass, config_entry)
    before = hass.states.get(BOOST_DURATION).state
    preferred = hass.states.get(ACM).attributes["preferred_boost_minutes"]
    extra_options.side_effect = ClientError("rejected")

    with pytest.raises(HomeAssistantError, match="Boost preset write"):
        await _set(hass, BOOST_DURATION, 5)

    assert hass.states.get(BOOST_DURATION).state == before
    assert hass.states.get(ACM).attributes["preferred_boost_minutes"] == preferred


async def test_boost_duration_follows_device_value(
    hass: HomeAssistant, extra_options: AsyncMock, config_entry: MockConfigEntry
) -> None:
    """A boost_time change reported by the device shows on the number."""
    await _setup(hass, config_entry)
    assert float(hass.states.get(BOOST_DURATION).state) == 2.0
    coordinator = config_entry.runtime_data.coordinator

    def _mutate(state: Any) -> None:
        state.boost_time = 240

    coordinator.apply_entity_patch("acm", "2", _mutate)
    await hass.async_block_till_done()

    assert float(hass.states.get(BOOST_DURATION).state) == 4.0
    assert hass.states.get(BOOST_DURATION).attributes["preferred_minutes"] == 240


@pytest.mark.parametrize(
    ("entity_id", "restored", "expected"),
    [(BOOST_DURATION, "3", 3.0), (BOOST_TEMP, "21.25", 21.3)],
)
async def test_boost_preset_restores_last_state_without_device_value(
    hass: HomeAssistant,
    cloud: FakeCloud,
    config_entry: MockConfigEntry,
    entity_id: str,
    restored: str,
    expected: float,
) -> None:
    """Until the device reports a boost preset, the last known value is shown."""
    settings = {
        k: v for k, v in ACM_SETTINGS.items() if k not in ("boost_time", "boost_temp")
    }
    _serve_accumulator(cloud, settings)
    mock_restore_cache(hass, [State(entity_id, restored)])

    await _setup(hass, config_entry)

    assert float(hass.states.get(entity_id).state) == expected


async def test_boost_duration_ignores_an_unavailable_last_state(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """A restored "unavailable" is no duration: the default is shown instead."""
    settings = {k: v for k, v in ACM_SETTINGS.items() if k != "boost_time"}
    _serve_accumulator(cloud, settings)
    mock_restore_cache(hass, [State(BOOST_DURATION, STATE_UNAVAILABLE)])

    await _setup(hass, config_entry)

    assert float(hass.states.get(BOOST_DURATION).state) == 1.0
    assert hass.states.get(BOOST_DURATION).attributes["preferred_minutes"] == 60


async def test_boost_temperature_uses_fahrenheit_device_units(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """A Fahrenheit accumulator shows and writes its boost preset in °F."""
    _serve_accumulator(
        cloud, {"mode": "auto", "state": "off", "stemp": "68", "units": "F"}
    )
    write = AsyncMock(return_value=None)
    with patch.object(RESTClient, "set_acm_extra_options", write):
        await _setup(hass, config_entry)
        state = hass.states.get(BOOST_TEMP)
        assert state.attributes["unit_of_measurement"] == "°F"
        assert state.attributes["step"] == 1.0
        assert (state.attributes["min"], state.attributes["max"]) == (41.0, 86.0)
        assert float(state.state) == 68.0

        await _set(hass, BOOST_TEMP, 75)

    assert write.await_args.kwargs["boost_temp"] == 75.0
    assert float(hass.states.get(BOOST_TEMP).state) == 75.0


async def test_boost_temperature_works_without_climate_entity(
    hass: HomeAssistant, extra_options: AsyncMock, config_entry: MockConfigEntry
) -> None:
    """The boost temperature writes to the device even if the climate is disabled."""
    await _setup(hass, config_entry)
    registry = er.async_get(hass)
    registry.async_update_entity(ACM, disabled_by=er.RegistryEntryDisabler.USER)
    assert await hass.config_entries.async_reload(config_entry.entry_id)
    await hass.async_block_till_done()
    assert hass.states.get(ACM) is None

    await _set(hass, BOOST_TEMP, 24)

    assert extra_options.await_args.kwargs["boost_temp"] == 24.0
    assert extra_options.await_args.kwargs["boost_time"] is None
    assert float(hass.states.get(BOOST_TEMP).state) == 24.0
    view = config_entry.runtime_data.coordinator.domain_view
    assert view.get_heater_state("acm", "2").boost_temp == "24.0"


async def test_boost_buttons_listen_to_coordinator_once(
    hass: HomeAssistant, extra_options: AsyncMock, config_entry: MockConfigEntry
) -> None:
    """Each boost button registers a single coordinator listener."""
    await _setup(hass, config_entry)
    coordinator = config_entry.runtime_data.coordinator
    buttons = [
        entity
        for entity in hass.data["button"].entities
        if type(entity).__name__.startswith("AccumulatorBoost")
    ]
    assert buttons

    for button in buttons:
        listeners = [
            callback
            for callback, _context in coordinator._listeners.values()  # noqa: SLF001
            if getattr(callback, "__self__", None) is button
        ]
        assert len(listeners) == 1, button.entity_id


def _set_ws(entry: MockConfigEntry, *, healthy: bool) -> None:
    """Report the WebSocket as healthy (recent payload) or disconnected."""
    coordinator = entry.runtime_data.coordinator
    now = time.time()
    coordinator.update_gateway_connection(
        status="healthy" if healthy else "disconnected",
        connected=healthy,
        last_event_at=now if healthy else None,
        healthy_since=now if healthy else None,
        healthy_minutes=1.0 if healthy else None,
        last_payload_at=now if healthy else None,
        last_heartbeat_at=now if healthy else None,
        payload_stale=not healthy,
        payload_stale_after=None,
        idle_restart_pending=False,
    )


@pytest.mark.parametrize(
    ("entity_id", "value"), [(BOOST_DURATION, 3), (BOOST_TEMP, 24)]
)
@pytest.mark.parametrize("healthy", [True, False])
async def test_boost_write_refreshes_node_only_when_ws_down(
    hass: HomeAssistant,
    extra_options: AsyncMock,
    config_entry: MockConfigEntry,
    entity_id: str,
    value: float,
    healthy: bool,
) -> None:
    """A boost write refreshes its node once after the delay, only if WS is down."""
    await _setup(hass, config_entry)
    _set_ws(config_entry, healthy=healthy)
    coordinator = config_entry.runtime_data.coordinator
    refresh = AsyncMock(return_value=None)

    with (
        patch.object(entity_module, "WS_ECHO_FALLBACK_REFRESH", 0.05),
        patch.object(coordinator, "async_refresh_heater", refresh),
    ):
        await _set(hass, entity_id, value)
        extra_options.assert_awaited_once()
        refresh.assert_not_awaited()
        await asyncio.sleep(0.1)
        await hass.async_block_till_done()

    if healthy:
        refresh.assert_not_awaited()
    else:
        refresh.assert_awaited_once_with(("acm", "2"))


# --- heater priority -------------------------------------------------------


@pytest.mark.parametrize(
    ("reported", "expected"),
    [(15, "15"), (0, "0"), (30, "30"), ("15", "15"), (None, STATE_UNKNOWN)],
)
async def test_priority_shows_device_value(
    hass: HomeAssistant,
    cloud: FakeCloud,
    config_entry: MockConfigEntry,
    reported: Any,
    expected: str,
) -> None:
    """The priority number shows the heater's priority; zero is a value."""
    settings = dict(cloud.get_node_settings.return_value)
    if reported is not None:
        settings["priority"] = reported
    cloud.get_node_settings.return_value = settings

    await _setup(hass, config_entry)

    assert hass.states.get(PRIORITY).state == expected


async def test_priority_write_shows_value_and_writes_device(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """A priority write goes to the heater's settings and shows at once."""
    write = AsyncMock(return_value=None)
    with patch.object(RESTClient, "set_node_priority", write):
        await _setup(hass, config_entry)
        await _set(hass, PRIORITY, 10)

    write.assert_awaited_once_with(DEV_ID, ("htr", "1"), priority=10)
    assert hass.states.get(PRIORITY).state == "10"


# --- installation power limit ----------------------------------------------


@pytest.mark.parametrize(
    ("reported", "expected"), [(5000, "5000"), (0, "0"), (None, STATE_UNAVAILABLE)]
)
async def test_power_limit_state(
    hass: HomeAssistant,
    cloud: FakeCloud,
    config_entry: MockConfigEntry,
    reported: int | None,
    expected: str,
) -> None:
    """A known limit (zero included) is shown; an unknown one is unavailable."""
    cloud.get_power_limit.return_value = reported

    await _setup(hass, config_entry)

    assert hass.states.get(_power_limit_entity(hass, config_entry)).state == expected


async def test_power_limit_belongs_to_the_site_device(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """The power limit is an installation entity on the site device."""
    cloud.get_power_limit.return_value = 5000
    await _setup(hass, config_entry)

    entry = er.async_get(hass).async_get(_power_limit_entity(hass, config_entry))
    assert entry.unique_id == build_installation_entity_unique_id(DEV_ID, "power_limit")
    assert entry.entity_category is EntityCategory.CONFIG
    device = dr.async_get(hass).async_get(entry.device_id)
    assert (DOMAIN, DEV_ID, "site") in device.identifiers


async def test_ws_update_event_power_limit_updates_number(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """An ``update`` event for the htr_system power limit reaches the number."""
    cloud.get_power_limit.return_value = 1000
    await _setup(hass, config_entry)
    runtime = config_entry.runtime_data
    client = TermoWebWSClient(
        hass,
        entry_id=config_entry.entry_id,
        dev_id=DEV_ID,
        api_client=runtime.client,
        coordinator=runtime.coordinator,
        inventory=runtime.inventory,
    )

    client._apply_nodes_payload(  # noqa: SLF001
        {"path": POWER_LIMIT_PATH, "body": {"power_limit": "9000"}},
        merge=True,
        event="update",
    )
    await _settle(hass)

    assert hass.states.get(_power_limit_entity(hass, config_entry)).state == "9000"


async def test_power_limit_store_notifies_only_on_change(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """Entities are told about a new limit, not about repeats or junk values."""
    await _setup(hass, config_entry)
    coordinator = config_entry.runtime_data.coordinator
    listener = MagicMock()
    unsub = coordinator.async_add_listener(listener)

    coordinator.apply_power_limit("4200")
    coordinator.apply_power_limit(4200)
    coordinator.apply_power_limit("abc")
    coordinator.apply_power_limit(None)
    unsub()

    listener.assert_called_once()
    assert coordinator.domain_view.get_power_limit() == 4200


async def test_ducaheat_has_priority_but_no_power_limit(
    hass: HomeAssistant, cloud: FakeCloud
) -> None:
    """Ducaheat has no installation power limit: no entity, no poll."""
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
        await entry.runtime_data.coordinator.async_refresh()
        await _settle(hass)

    assert _power_limit_entity(hass, entry) is None
    cloud.get_power_limit.assert_not_awaited()
    assert hass.states.get(PRIORITY) is not None


async def test_radio_gets_local_priority_and_power_limit(hass: HomeAssistant) -> None:
    """The radio backend's local power manager brings priority and limit numbers."""
    inventory = Inventory(
        DEV_ID, build_node_inventory([{"addr": "6", "name": "Heater 6", "type": "htr"}])
    )
    coordinator = StateCoordinator(
        hass, MagicMock(), 30, DEV_ID, None, inventory, brand="radio"
    )
    runtime = build_entry_runtime(
        hass=hass,
        dev_id=DEV_ID,
        inventory=inventory,
        coordinator=coordinator,
        brand="radio",
    )
    added: list[list[Any]] = []

    await number.async_setup_entry(hass, runtime.config_entry, added.append)

    assert [type(e) for e in added[0]] == [
        number.HeaterPriorityNumber,
        number.PowerLimitNumber,
    ]
