"""Setup and unload of a TermoWeb cloud entry on real Home Assistant."""

from __future__ import annotations

import asyncio
from datetime import timedelta
import logging
from unittest.mock import patch

from aiohttp import ClientError
from freezegun.api import FrozenDateTimeFactory
from homeassistant.config_entries import SOURCE_REAUTH, ConfigEntryState
from homeassistant.const import EVENT_HOMEASSISTANT_STOP
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.util import dt as dt_util
import pytest
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_fire_time_changed,
)

from custom_components.termoweb import PLATFORMS, _platforms_for_brand
from custom_components.termoweb.backend import factory
from custom_components.termoweb.backend.radio import RadioLinkError
from custom_components.termoweb.backend.radio_client import (
    LISTEN_ONLY_STATION_ID,
    RadioClient,
    RadioError,
)
from custom_components.termoweb.backend.radio_monitor import RadioMonitor
from custom_components.termoweb.backend.rest_client import (
    BackendAuthError,
    BackendRateLimitError,
)
from custom_components.termoweb.backend.ws_health import WsHealthTracker
from custom_components.termoweb.const import CONF_BRAND, DOMAIN, signal_ws_status
from custom_components.termoweb.runtime import require_runtime
from tests_ha.fakes.radio_link import FakeRadioLink

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
    assert "Ignored: Cabin, 001122...66ff" in caplog.text


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


async def test_setup_registers_no_dangling_via_device(
    hass: HomeAssistant,
    cloud: FakeCloud,
    config_entry: MockConfigEntry,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Site, gateway and node devices are linked by via_device_id, not via_device."""
    assert await _setup(hass, config_entry)

    dev_reg = dr.async_get(hass)
    dev_id = config_entry.runtime_data.dev_id
    entry_id = config_entry.entry_id
    gateway = dev_reg.async_get_device_by_identifier((DOMAIN, dev_id), entry_id)
    site = dev_reg.async_get_device_by_identifier((DOMAIN, dev_id, "site"), entry_id)
    assert gateway is not None
    assert site is not None
    assert gateway.via_device_id == site.id
    assert site.via_device_id is None
    nodes = [
        device
        for device in dr.async_entries_for_config_entry(dev_reg, entry_id)
        if device.id not in (gateway.id, site.id)
    ]
    assert nodes
    assert all(device.via_device_id == gateway.id for device in nodes)
    assert "via_device" not in caplog.text


@pytest.mark.parametrize(
    ("devices", "dev_id"),
    [
        (
            ["invalid", {"name": "No id"}, {"id": " 00aa11bb22cc33dd "}],
            "00aa11bb22cc33dd",
        ),
        ({"serial_id": " 00aa11bb22cc33ee "}, "00aa11bb22cc33ee"),
    ],
    ids=["list-skips-entries-without-id", "single-mapping"],
)
async def test_setup_picks_gateway_id_from_device_payload(
    hass: HomeAssistant,
    cloud: FakeCloud,
    config_entry: MockConfigEntry,
    devices: object,
    dev_id: str,
) -> None:
    """The gateway id comes from dev_id, id or serial_id; unusable entries are skipped."""
    cloud.list_devices.return_value = devices

    assert await _setup(hass, config_entry)

    assert config_entry.runtime_data.dev_id == dev_id
    cloud.get_nodes.assert_awaited_once_with(dev_id)


@pytest.mark.parametrize(
    "devices", [[{"name": "Missing"}, {}], "unexpected"], ids=["no-ids", "not-a-list"]
)
async def test_setup_retries_without_a_usable_gateway(
    hass: HomeAssistant,
    cloud: FakeCloud,
    config_entry: MockConfigEntry,
    devices: object,
) -> None:
    """Without any gateway id, setup is retried and no node list is requested."""
    cloud.list_devices.return_value = devices

    assert not await _setup(hass, config_entry)

    assert config_entry.state is ConfigEntryState.SETUP_RETRY
    cloud.get_nodes.assert_not_awaited()


@pytest.mark.parametrize(
    "error", [TimeoutError(), BackendRateLimitError("slow down")], ids=repr
)
async def test_setup_retries_on_transient_device_list_errors(
    hass: HomeAssistant,
    cloud: FakeCloud,
    config_entry: MockConfigEntry,
    error: Exception,
) -> None:
    """Timeouts and rate limiting while listing gateways ask HA to retry."""
    cloud.list_devices.side_effect = error

    assert not await _setup(hass, config_entry)

    assert config_entry.state is ConfigEntryState.SETUP_RETRY


async def test_setup_builds_one_inventory_and_survives_unknown_node_types(
    hass: HomeAssistant,
    cloud: FakeCloud,
    config_entry: MockConfigEntry,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The node list becomes the shared inventory; unknown types are logged once."""
    cloud.get_nodes.return_value = {
        "nodes": [
            {"addr": 9, "type": "foo"},
            {"addr": 9, "type": "foo"},
            {"addr": 1, "type": "htr", "name": "Living room"},
            {"addr": 2, "type": "acm", "name": "Hall"},
        ]
    }
    caplog.set_level(logging.DEBUG, logger="custom_components.termoweb")

    assert await _setup(hass, config_entry)

    runtime = config_entry.runtime_data
    known = [
        (node.type, node.addr)
        for node in runtime.inventory.nodes
        if node.type in ("htr", "acm")
    ]
    assert known == [("htr", "1"), ("acm", "2")]
    assert hass.states.get("climate.living_room") is not None
    unknown = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Unknown node type found")
    ]
    assert unknown == ["Unknown node type found: foo/9"]


async def test_setup_survives_geo_data_failure(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """The location lookup is best effort: an error leaves geo data unset."""
    cloud.get_geo_data.side_effect = ClientError("geo down")

    assert await _setup(hass, config_entry)

    assert config_entry.state is ConfigEntryState.LOADED
    assert config_entry.runtime_data.coordinator.device_metadata.geo_data is None


async def test_setup_registers_site_and_gateway_devices(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """Gateway metadata from the device list reaches the device registry."""
    cloud.list_devices.return_value = [
        {
            "dev_id": DEV_ID,
            "name": "Home",
            "serial_id": " SN-TEST ",
            "fw_version": " 3.0.1 ",
        }
    ]

    assert await _setup(hass, config_entry)

    dev_reg = dr.async_get(hass)
    entry_id = config_entry.entry_id
    site = dev_reg.async_get_device_by_identifier((DOMAIN, DEV_ID, "site"), entry_id)
    gateway = dev_reg.async_get_device_by_identifier((DOMAIN, DEV_ID), entry_id)
    assert (site.name, site.model, site.manufacturer) == ("Home", "Site", "TermoWeb")
    assert site.configuration_url == "https://control.termoweb.net"
    assert gateway.manufacturer == "TermoWeb"
    assert gateway.sw_version == "3.0.1"
    assert gateway.serial_number == "SN-TEST"


async def test_platforms_follow_the_backend() -> None:
    """Ducaheat-backed brands add the lock platform; monitors only report."""
    assert _platforms_for_brand("termoweb") == PLATFORMS
    assert _platforms_for_brand("ducaheat") == [*PLATFORMS, "lock"]
    assert _platforms_for_brand("tevolve") == [*PLATFORMS, "lock"]
    assert _platforms_for_brand("radio_monitor") == ["binary_sensor", "sensor"]


def _healthy_tracker(dev_id: str, *, stale_after: float = 300) -> WsHealthTracker:
    """Return a tracker that is healthy with a fresh payload."""
    tracker = WsHealthTracker(dev_id)
    tracker.update_status("healthy")
    tracker.mark_payload(stale_after=stale_after)
    return tracker


async def test_healthy_websocket_suspends_polling_until_payload_goes_stale(
    hass: HomeAssistant,
    cloud: FakeCloud,
    config_entry: MockConfigEntry,
    freezer: FrozenDateTimeFactory,
) -> None:
    """Fresh websocket payloads pause REST polling; staleness resumes it."""
    assert await _setup(hass, config_entry)
    runtime = config_entry.runtime_data
    coordinator = runtime.coordinator
    base = timedelta(seconds=runtime.base_poll_interval)
    signal = signal_ws_status(config_entry.entry_id)
    runtime.ws_trackers[DEV_ID] = tracker = _healthy_tracker(DEV_ID)

    # Another entry's status signal does nothing to this entry.
    async_dispatcher_send(hass, signal_ws_status("other"), {"payload_changed": True})
    assert coordinator.update_interval == base
    # Events that change neither health nor payload don't recalculate either.
    async_dispatcher_send(hass, signal, {"reason": "heartbeat"})
    assert not runtime.poll_suspended

    async_dispatcher_send(hass, signal, {"payload_changed": True})
    assert runtime.poll_suspended
    assert coordinator.update_interval is None

    # The resume timer fires when the payload goes stale.
    freezer.tick(timedelta(seconds=301))
    async_fire_time_changed(hass)
    await hass.async_block_till_done()
    assert not runtime.poll_suspended
    assert coordinator.update_interval == base
    assert runtime.poll_resume_unsub is None

    # A fresh payload suspends again; an unhealthy status resumes at once.
    tracker.mark_payload(stale_after=300)
    async_dispatcher_send(hass, signal, {"reason": "status"})
    assert runtime.poll_suspended
    tracker.update_status("degraded", reset_health=True)
    async_dispatcher_send(hass, signal, {"health_changed": True})
    assert not runtime.poll_suspended
    assert coordinator.update_interval == base
    assert runtime.poll_resume_unsub is None


async def test_polling_resumes_when_the_websocket_task_ends(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """A finished or missing websocket task brings REST polling back."""
    assert await _setup(hass, config_entry)
    runtime = config_entry.runtime_data
    base = timedelta(seconds=runtime.base_poll_interval)
    signal = signal_ws_status(config_entry.entry_id)
    runtime.ws_trackers[DEV_ID] = _healthy_tracker(DEV_ID)
    async_dispatcher_send(hass, signal, {"payload_changed": True})
    assert runtime.poll_suspended

    cloud.ws_clients[0].task.cancel()
    await hass.async_block_till_done()
    async_dispatcher_send(hass, signal, object())  # any non-mapping payload
    assert not runtime.poll_suspended
    assert runtime.coordinator.update_interval == base

    runtime.ws_tasks.clear()
    runtime.poll_suspended = True
    runtime.coordinator.update_interval = None
    async_dispatcher_send(hass, signal, {"reason": "status"})
    assert not runtime.poll_suspended
    assert runtime.coordinator.update_interval == base


async def test_unload_while_suspended_cancels_the_resume_timer(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """Unloading a suspended entry leaves no resume timer or listener behind."""
    assert await _setup(hass, config_entry)
    runtime = config_entry.runtime_data
    runtime.ws_trackers[DEV_ID] = _healthy_tracker(DEV_ID)
    async_dispatcher_send(
        hass, signal_ws_status(config_entry.entry_id), {"payload_changed": True}
    )
    assert runtime.poll_resume_unsub is not None

    assert await hass.config_entries.async_unload(config_entry.entry_id)
    await hass.async_block_till_done()

    assert runtime.poll_resume_unsub is None
    assert runtime.unsub_ws_status is None


async def test_home_assistant_stop_shuts_the_entry_down_once(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """Stopping HA stops the websocket; a later unload does not stop it again."""
    assert await _setup(hass, config_entry)
    ws_client = cloud.ws_clients[0]

    hass.bus.async_fire(EVENT_HOMEASSISTANT_STOP)
    await hass.async_block_till_done()

    assert ws_client.stop_calls == 1
    assert ws_client.task.cancelled()
    assert await hass.config_entries.async_unload(config_entry.entry_id)
    assert ws_client.stop_calls == 1


async def test_unload_survives_a_failing_websocket_client(
    hass: HomeAssistant,
    cloud: FakeCloud,
    config_entry: MockConfigEntry,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A websocket client that fails to stop is logged; unload still succeeds."""
    assert await _setup(hass, config_entry)
    ws_client = cloud.ws_clients[0]
    real_stop = ws_client.stop

    async def _failing_stop() -> None:
        await real_stop()
        raise RuntimeError("client fail")

    ws_client.stop = _failing_stop

    assert await hass.config_entries.async_unload(config_entry.entry_id)
    await hass.async_block_till_done()

    assert config_entry.state is ConfigEntryState.NOT_LOADED
    assert "failed to stop" in caplog.text


RADIO_GATEWAY = {
    CONF_BRAND: "radio",
    "host": "10.0.0.5",
    "port": "2323",
    "dialect": "B",
    "network_id": "1234",
    "nodes": [{"type": "htr", "addr": "6", "name": "Heater 6"}],
}
RADIO_STICK = {
    CONF_BRAND: "radio",
    "radio_type": "nanocul",
    "device": "/dev/ttyUSB0",
    "radio_device_id": "nanocul-a1b2c3",
    "dialect": "A",
    "network_id": "1B30",
    "nodes": [],
}
MONITOR_ESP32 = {CONF_BRAND: "radio_monitor", "host": "10.0.0.5", "port": 2323}
MONITOR_STICK = {
    CONF_BRAND: "radio_monitor",
    "radio_type": "nanocul",
    "device": "socket://10.0.0.7:5000",
    "radio_device_id": "nanocul-a1b2c3",
}


class _UnreachableRadio:
    """Radio client whose gateway cannot be reached."""

    def __init__(self, error: Exception) -> None:
        """Remember the error ``list_devices`` raises."""
        self.error = error

    async def list_devices(self) -> list[dict[str, object]]:
        """Fail like an unreachable gateway or a banner without a MAC."""
        raise self.error

    async def async_close(self) -> None:
        """Release nothing."""


@pytest.mark.parametrize(
    ("data", "args", "kwargs"),
    [
        (
            RADIO_GATEWAY,
            ("10.0.0.5", 2323, "B", RADIO_GATEWAY["nodes"], bytes.fromhex("1234")),
            {"power"},
        ),
        (
            RADIO_STICK,
            ("/dev/ttyUSB0", 0, "A", [], bytes.fromhex("1B30")),
            {"power", "serial_url", "device_id"},
        ),
        (MONITOR_ESP32, ("10.0.0.5", 2323, "A", [], b"\x00\x00"), {"listen_only"}),
        (
            MONITOR_STICK,
            ("socket://10.0.0.7:5000", 0, "A", [], b"\x00\x00"),
            {"serial_url", "device_id", "listen_only"},
        ),
    ],
    ids=["esp32", "nanocul", "monitor-esp32", "monitor-nanocul"],
)
@pytest.mark.parametrize(
    "error", [RadioLinkError("cannot connect"), RadioError("no MAC")], ids=repr
)
async def test_radio_setup_builds_the_client_and_retries_when_unreachable(
    hass: HomeAssistant,
    data: dict[str, object],
    args: tuple[object, ...],
    kwargs: set[str],
    error: Exception,
) -> None:
    """Each radio entry type gets its client; an unreachable radio is retried."""
    created: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _create(*a: object, **kw: object) -> _UnreachableRadio:
        created.append((a, kw))
        return _UnreachableRadio(error)

    entry = MockConfigEntry(domain=DOMAIN, minor_version=5, data=data)
    with patch.object(factory, "create_radio_client", _create):
        assert not await _setup(hass, entry)

    assert entry.state is ConfigEntryState.SETUP_RETRY
    [(call_args, call_kwargs)] = created
    assert call_args == args
    assert set(call_kwargs) == kwargs
    if "serial_url" in kwargs:
        assert call_kwargs["serial_url"] == data["device"]
        assert call_kwargs["device_id"] == "nanocul-a1b2c3"
    if "power" in kwargs:
        call_kwargs["power"].set_power_limit(1800)  # saved into the entry options
        assert entry.options["radio_power"]["power_limit"] == 1800


async def test_monitor_entry_listens_only_and_unloads(hass: HomeAssistant) -> None:
    """A monitor entry starts a listen-only radio, never transmits, and closes it."""
    links: list[FakeRadioLink] = []

    def _create(host, port, dialect, nodes, network_id, **kwargs) -> RadioClient:
        assert kwargs == {"listen_only": True}

        def _link(*a: object, **kw: object) -> FakeRadioLink:
            links.append(FakeRadioLink(*a, **kw))
            return links[-1]

        return RadioClient(
            host,
            port,
            dialect,
            nodes,
            network_id=network_id,
            station_id=LISTEN_ONLY_STATION_ID,
            link_factory=_link,
            listen_only=True,
        )

    entry = MockConfigEntry(domain=DOMAIN, minor_version=5, data=MONITOR_ESP32)
    with patch.object(factory, "create_radio_client", _create):
        assert await _setup(hass, entry)

    runtime = entry.runtime_data
    [ws_client] = runtime.ws_clients.values()
    assert isinstance(ws_client, RadioMonitor)
    task = runtime.ws_tasks[runtime.dev_id]
    assert not task.done()
    assert list(runtime.inventory.nodes) == []
    entities = er.async_entries_for_config_entry(er.async_get(hass), entry.entry_id)
    assert {entity.domain for entity in entities} == {"binary_sensor", "sensor"}

    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()

    assert task.done()
    [link] = links
    assert link.kwargs == {"listen_only": True, "auto_ack": False}
    assert (link.station_id, link.network_id) == (0xFE, b"\x00\x00")
    assert link.sent == []
    assert link.closes >= 1
