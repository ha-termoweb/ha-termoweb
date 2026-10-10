"""Setup and unload of a TermoWeb cloud entry on real Home Assistant."""

from __future__ import annotations

import asyncio
from datetime import timedelta

from aiohttp import ClientError
from homeassistant.config_entries import SOURCE_REAUTH, ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.util import dt as dt_util
import pytest
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_fire_time_changed,
)

from custom_components.termoweb.backend.rest_client import BackendAuthError
from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.runtime import require_runtime

from .conftest import DEV_ID, DEVICES, FakeCloud

PLATFORM_DOMAINS = {"binary_sensor", "button", "climate", "number", "sensor"}
SERVICES = {
    "import_energy_history",
    "radio_survey",
    "radio_capture",
    "radio_pair",
    "radio_factory_reset",
    "radio_rehome",
}


async def _setup(hass: HomeAssistant, entry: MockConfigEntry) -> bool:
    """Add the entry to hass and set it up."""
    entry.add_to_hass(hass)
    result = await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return result


async def _fire_next_hourly_poll(hass: HomeAssistant) -> None:
    """Advance time to the next HH:05 trigger and let the poller run."""
    now = dt_util.utcnow()
    target = (now + timedelta(hours=1)).replace(minute=5, second=0, microsecond=0)
    async_fire_time_changed(hass, target)
    # The poller schedules its run with call_soon_threadsafe + loop.create_task,
    # which async_block_till_done does not track; yield until it has run.
    for _ in range(10):
        await asyncio.sleep(0)
    await hass.async_block_till_done()


async def test_setup_entry_loads_and_forwards_platforms(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """Setup loads the entry, starts the websocket and creates every platform."""
    assert await _setup(hass, config_entry)

    assert config_entry.state is ConfigEntryState.LOADED
    runtime = config_entry.runtime_data
    assert runtime.dev_id == "0123456789abcdef"
    assert len(cloud.ws_clients) == 1
    assert not cloud.ws_clients[0].task.done()

    entities = er.async_entries_for_config_entry(
        er.async_get(hass), config_entry.entry_id
    )
    assert {entity.domain for entity in entities} == PLATFORM_DOMAINS
    climate = hass.states.get("climate.living_room")
    assert climate is not None
    assert climate.attributes["temperature"] == 21.0

    assert hass.services.has_service(DOMAIN, "import_energy_history")


async def test_unload_entry_stops_runtime(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """Unload removes the entities first, then stops the websocket and poller."""
    assert await _setup(hass, config_entry)
    ws_client = cloud.ws_clients[0]
    climate_at_stop: list[str] = []
    stop = ws_client.stop

    async def _stop() -> None:
        climate_at_stop.append(hass.states.get("climate.living_room").state)
        await stop()

    ws_client.stop = _stop

    assert await hass.config_entries.async_unload(config_entry.entry_id)
    await hass.async_block_till_done()

    assert config_entry.state is ConfigEntryState.NOT_LOADED
    assert not hasattr(config_entry, "runtime_data")
    with pytest.raises(LookupError):
        require_runtime(hass, config_entry.entry_id)
    assert ws_client.stop_calls == 1
    # B16: the platforms were already unloaded when the websocket stopped.
    assert climate_at_stop == ["unavailable"]
    assert ws_client.task.cancelled()
    assert hass.states.get("climate.living_room").state == "unavailable"

    cloud.get_node_samples.reset_mock()
    await _fire_next_hourly_poll(hass)
    cloud.get_node_samples.assert_not_awaited()


async def test_services_registered_once_and_refuse_without_loaded_entry(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """Services come from async_setup and fail clearly once no entry is loaded.

    B14: they used to be registered per entry and lingered after unload with
    nothing behind them. Quality-scale action-setup keeps them registered and
    raises ServiceValidationError instead.
    """
    assert await _setup(hass, config_entry)
    assert set(hass.services.async_services_for_domain(DOMAIN)) >= SERVICES

    assert await hass.config_entries.async_unload(config_entry.entry_id)
    await hass.async_block_till_done()

    with pytest.raises(ServiceValidationError, match="No loaded TermoWeb entry"):
        await hass.services.async_call(
            DOMAIN, "import_energy_history", {}, blocking=True
        )
    with pytest.raises(ServiceValidationError, match="No loaded TermoWeb entry"):
        await hass.services.async_call(
            DOMAIN,
            "radio_survey",
            {"entry_id": config_entry.entry_id},
            blocking=True,
            return_response=True,
        )


async def test_setup_retries_when_cloud_unreachable(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """A connection error during setup asks Home Assistant to retry."""
    cloud.list_devices.side_effect = ClientError("offline")

    assert not await _setup(hass, config_entry)

    assert config_entry.state is ConfigEntryState.SETUP_RETRY


@pytest.mark.parametrize("error", [ClientError("offline"), TimeoutError()])
async def test_setup_retries_when_nodes_unreachable(
    hass: HomeAssistant,
    cloud: FakeCloud,
    config_entry: MockConfigEntry,
    error: Exception,
) -> None:
    """A network error while reading the node list asks HA to retry."""
    cloud.get_nodes.side_effect = error

    assert not await _setup(hass, config_entry)

    assert config_entry.state is ConfigEntryState.SETUP_RETRY


async def test_setup_nodes_auth_failure_starts_reauth(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """Rejected credentials while reading the node list start a reauth flow."""
    cloud.get_nodes.side_effect = BackendAuthError("expired")

    assert not await _setup(hass, config_entry)

    assert config_entry.state is ConfigEntryState.SETUP_ERROR


async def test_setup_uses_first_gateway_and_warns(
    hass: HomeAssistant,
    cloud: FakeCloud,
    config_entry: MockConfigEntry,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An account with several gateways sets up the first and names the rest."""
    cloud.list_devices.return_value = [
        {"name": "No id"},
        *DEVICES,
        {"dev_id": "fedcba9876543210", "name": "Cabin"},
        {"dev_id": "00112233445566ff"},
    ]

    assert await _setup(hass, config_entry)

    assert config_entry.runtime_data.dev_id == DEV_ID
    assert "3 gateways; only the first (Home) is set up" in caplog.text
    assert "Ignored: Cabin, 00112233445566ff" in caplog.text


async def test_setup_retries_when_no_devices(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """An account without gateways is not ready yet."""
    cloud.list_devices.return_value = []

    assert not await _setup(hass, config_entry)

    assert config_entry.state is ConfigEntryState.SETUP_RETRY


async def test_setup_auth_failure_starts_reauth(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """Rejected credentials fail setup and start a reauth flow."""
    cloud.list_devices.side_effect = BackendAuthError("bad credentials")

    assert not await _setup(hass, config_entry)

    assert config_entry.state is ConfigEntryState.SETUP_ERROR
    flows = hass.config_entries.flow.async_progress_by_handler(DOMAIN)
    assert [flow["context"]["source"] for flow in flows] == [SOURCE_REAUTH]
    assert flows[0]["step_id"] == "reauth_confirm"


@pytest.mark.xfail(
    strict=True,
    reason="F2: setup writes a fictional 'supports_diagnostics' key into "
    "entry.data (PLAN Phase 2.4/3, setup_energy.md F2)",
)
async def test_setup_does_not_modify_entry_data(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """Setup leaves the stored credentials untouched."""
    before = dict(config_entry.data)

    assert await _setup(hass, config_entry)

    assert dict(config_entry.data) == before


async def test_failed_first_refresh_cleans_up(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """A setup that fails late leaves no runtime and no hourly poll behind."""
    cloud.get_node_settings.side_effect = TimeoutError

    assert not await _setup(hass, config_entry)
    assert config_entry.state is ConfigEntryState.SETUP_RETRY

    # Keep HA's setup retry from reaching the poller, so any sample fetch
    # below can only come from a listener the failed attempt left behind.
    cloud.list_devices.side_effect = ClientError("offline")
    cloud.get_node_samples.reset_mock()
    await _fire_next_hourly_poll(hass)
    cloud.get_node_samples.assert_not_awaited()
    with pytest.raises(LookupError):
        require_runtime(hass, config_entry.entry_id)
    assert config_entry.runtime_data._shutdown_complete  # noqa: SLF001


@pytest.mark.xfail(
    # Not strict: whether the site device exists before the gateway depends on
    # platform setup order, so the bug only shows up on some runs.
    strict=False,
    reason="The gateway device names a 'site' via_device that is not registered "
    "when the gateway is, so the link is dropped (HA logs it; still not an "
    "error in 2026.10). Order-dependent; fixed by the via_device_id migration.",
)
async def test_setup_registers_no_dangling_via_device(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """The gateway device is linked to the site device it names as via_device."""
    assert await _setup(hass, config_entry)

    # HA reports a dangling via_device only once per call site, and since
    # 2026.10 the via_device deprecation notice takes that slot, so check the
    # registry link itself rather than the log.
    dev_reg = dr.async_get(hass)
    dev_id = config_entry.runtime_data.dev_id
    entry_id = config_entry.entry_id
    gateway = dev_reg.async_get_device_by_identifier((DOMAIN, dev_id), entry_id)
    site = dev_reg.async_get_device_by_identifier((DOMAIN, dev_id, "site"), entry_id)
    assert gateway is not None
    assert site is not None
    assert gateway.via_device_id == site.id
