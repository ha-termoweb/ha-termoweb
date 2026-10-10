"""Gateway connectivity and boost binary sensors on real Home Assistant."""

from __future__ import annotations

from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr, entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.domain import NodeId, NodeSettingsDelta, NodeType

from .conftest import DEV_ID, FakeCloud
from .fakes.sensors import entity_ids, serve_nodes, setup_entry

GATEWAY = "binary_sensor.termoweb_gateway_gateway_online"
NODES = [
    {"type": "htr", "addr": 1, "name": "Living room"},
    {"type": "acm", "addr": 2, "name": "Store"},
    {"type": "thm", "addr": 4, "name": "Hall"},
]


async def test_gateway_online_follows_connection_state(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """The connectivity sensor and its attributes follow the websocket health."""
    serve_nodes(cloud, NODES[:1])
    await setup_entry(hass, config_entry)
    assert hass.states.get(GATEWAY).state == STATE_OFF

    config_entry.runtime_data.coordinator.update_gateway_connection(
        status="healthy",
        connected=True,
        last_event_at=171.0,
        healthy_since=111.0,
        healthy_minutes=42.0,
        last_payload_at=170.0,
        last_heartbeat_at=169.0,
        payload_stale=False,
        payload_stale_after=120.0,
        idle_restart_pending=False,
    )
    await hass.async_block_till_done()

    state = hass.states.get(GATEWAY)
    assert state.state == STATE_ON
    assert {
        key: state.attributes[key]
        for key in (
            "dev_id",
            "name",
            "connected",
            "ws_status",
            "ws_last_event_at",
            "ws_healthy_minutes",
        )
    } == {
        "dev_id": DEV_ID,
        "name": "Home",
        "connected": True,
        "ws_status": "healthy",
        "ws_last_event_at": 171.0,
        "ws_healthy_minutes": 42.0,
    }
    entry = er.async_get(hass).async_get(GATEWAY)
    assert entry.unique_id == f"{DOMAIN}:{DEV_ID}:online"
    gateway = dr.async_get(hass).async_get(entry.device_id)
    assert (DOMAIN, DEV_ID) in gateway.identifiers


async def test_only_accumulators_get_a_boost_sensor(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """Boost activity is exposed for boost-capable nodes only."""
    serve_nodes(cloud, NODES)
    await setup_entry(hass, config_entry)

    assert entity_ids(hass, config_entry, "binary_sensor") == {
        GATEWAY,
        "binary_sensor.store_boost_active",
    }


async def test_boost_sensor_follows_device_boost_state(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """The boost sensor turns on and off with the accumulator's boost flag."""
    serve_nodes(cloud, NODES[1:2], {("acm", "2"): {"boost_active": True}})
    await setup_entry(hass, config_entry)
    boost = "binary_sensor.store_boost_active"
    assert hass.states.get(boost).state == STATE_ON
    assert hass.states.get(boost).attributes["addr"] == "2"

    config_entry.runtime_data.coordinator.handle_ws_deltas(
        DEV_ID,
        [
            NodeSettingsDelta(
                node_id=NodeId(NodeType.ACCUMULATOR, "2"),
                changes={"boost_active": False},
            )
        ],
    )
    await hass.async_block_till_done()

    assert hass.states.get(boost).state == STATE_OFF


async def test_boost_sensor_belongs_to_the_accumulator_device(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """The boost sensor sits on the accumulator's device, named by type when unnamed."""
    serve_nodes(cloud, [{"type": "acm", "addr": 4}])
    await setup_entry(hass, config_entry)

    entry = er.async_get(hass).async_get("binary_sensor.accumulator_4_boost_active")
    device = dr.async_get(hass).async_get(entry.device_id)
    assert (DOMAIN, DEV_ID, "4") in device.identifiers
    assert device.name == "Accumulator 4"
    assert device.model == "Accumulator"
