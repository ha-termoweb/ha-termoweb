"""Tests for the shared entity base: boost state, availability and write refresh."""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from datetime import datetime, timedelta
import time
import types
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

from aiohttp import ClientError
from freezegun.api import FrozenDateTimeFactory
from homeassistant.const import (
    ATTR_ENTITY_ID,
    STATE_OFF,
    STATE_ON,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util
import pytest
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_fire_time_changed,
)

from custom_components.termoweb import (
    boost as boost_module,
    climate as climate_module,
    entity as entity_module,
)
from custom_components.termoweb.backend.ducaheat import (
    DucaheatBackend,
    DucaheatRESTClient,
)
from custom_components.termoweb.backend.rest_client import RESTClient
from custom_components.termoweb.boost import coerce_boost_minutes
from custom_components.termoweb.const import BRAND_DUCAHEAT, CONF_BRAND, DOMAIN
from custom_components.termoweb.entity import derive_boost_state
from custom_components.termoweb.inventory import Inventory
from tests.fakes.cloud import PASSWORD, USERNAME, FakeCloud

from .fakes.entities import FakeNodes, setup_entry

NOW = datetime(2024, 1, 1, 12, 0, tzinfo=dt_util.UTC)
BOOST_ACTIVE = "binary_sensor.store_boost_active"
BOOST_MINUTES = "sensor.store_boost_minutes_remaining"
BOOST_END = "sensor.store_boost_end"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (True, None),
        (False, None),
        (0, None),
        (-5, None),
        ("   ", None),
        ("invalid", None),
        ("90", 90),
        (120.7, 120),
        (45, 45),
    ],
)
def test_coerce_boost_minutes(value: Any, expected: int | None) -> None:
    """Only positive numbers (or numeric strings) are boost durations."""
    assert coerce_boost_minutes(value) == expected


def _resolver(day: Any, minute: Any) -> tuple[datetime | None, int | None]:
    """Resolve a boost end like the coordinator: day 1 + minutes past midnight."""
    end = datetime(2024, 1, day, 0, 0, tzinfo=dt_util.UTC) + timedelta(minutes=minute)
    return end, int((end - NOW).total_seconds() // 60)


def _failing_resolver(day: Any, minute: Any) -> tuple[datetime | None, int | None]:
    """Fail like a coordinator without a device clock estimate."""
    raise ValueError("no device clock")


@pytest.mark.parametrize(
    ("settings", "resolver", "active", "minutes", "end", "label"),
    [
        pytest.param({}, None, False, None, None, "Never", id="nothing"),
        pytest.param({"mode": "boost"}, None, True, None, None, None, id="mode"),
        pytest.param(
            {"boost_active": "off", "mode": "boost"},
            None,
            False,
            None,
            None,
            "Never",
            id="flag-wins-over-mode",
        ),
        pytest.param(
            {"boost_active": True, "boost_remaining": 30},
            None,
            True,
            30,
            NOW + timedelta(minutes=30),
            None,
            id="remaining",
        ),
        pytest.param(
            {"boost_active": 1, "boost_end_day": 1, "boost_end_min": 14 * 60},
            _resolver,
            True,
            120,
            NOW + timedelta(hours=2),
            None,
            id="day-minute",
        ),
        pytest.param(
            {
                "boost_active": 1,
                "boost_end_day": 1,
                "boost_end_min": 0,
                "boost_remaining": 15,
            },
            _failing_resolver,
            True,
            15,
            NOW + timedelta(minutes=15),
            None,
            id="resolver-fails",
        ),
        pytest.param(
            {"boost_active": True, "boost_end_datetime": "2024-01-01T13:00:00+00:00"},
            None,
            True,
            60,
            NOW + timedelta(hours=1),
            None,
            id="iso-end",
        ),
        pytest.param(
            {"boost_active": False, "boost_end_datetime": "1970-01-01T00:00:00+00:00"},
            None,
            False,
            None,
            None,
            "Never",
            id="epoch-placeholder",
        ),
        pytest.param(
            {"boost_active": True, "boost_remaining": 0},
            None,
            True,
            None,
            None,
            None,
            id="zero-remaining",
        ),
    ],
)
def test_derive_boost_state(
    freezer: FrozenDateTimeFactory,
    settings: dict[str, Any],
    resolver: Any,
    active: bool,
    minutes: int | None,
    end: datetime | None,
    label: str | None,
) -> None:
    """Boost metadata is derived from the flag, remaining minutes or end fields."""
    freezer.move_to(NOW)
    coordinator = SimpleNamespace(resolve_boost_end=resolver)

    state = derive_boost_state(settings, coordinator)

    assert state.active is active
    assert state.minutes_remaining == minutes
    assert state.end_datetime == end
    assert state.end_iso == (end.isoformat() if end else None)
    assert state.end_label == label


@pytest.fixture
def store(cloud: FakeCloud) -> Generator[FakeNodes]:
    """Serve one accumulator, "Store" (addr 3), behind the fake cloud."""
    fake = FakeNodes(
        cloud,
        {
            ("acm", "3"): (
                "Store",
                {"mode": "auto", "state": "off", "stemp": "19.0", "units": "C"},
            )
        },
    )
    with fake.patched():
        yield fake


async def test_boost_entities_show_a_running_boost(
    hass: HomeAssistant,
    store: FakeNodes,
    config_entry: MockConfigEntry,
    freezer: FrozenDateTimeFactory,
) -> None:
    """The boost binary sensor and sensors report a boost with 30 minutes left."""
    freezer.move_to(NOW)
    store.settings[("acm", "3")].update(
        mode="boost", boost_active=True, boost_remaining=30
    )
    await setup_entry(hass, config_entry)

    active = hass.states.get(BOOST_ACTIVE)
    assert active.state == STATE_ON
    assert active.attributes["boost_minutes_remaining"] == 30
    assert hass.states.get(BOOST_MINUTES).state == "30"
    end = hass.states.get(BOOST_END)
    assert dt_util.parse_datetime(end.state) == NOW + timedelta(minutes=30)
    assert end.attributes["boost_end_label"] is None


async def test_boost_entities_without_a_boost(
    hass: HomeAssistant, store: FakeNodes, config_entry: MockConfigEntry
) -> None:
    """Without a boost the binary sensor is off and nothing is left to run."""
    await setup_entry(hass, config_entry)

    assert hass.states.get(BOOST_ACTIVE).state == STATE_OFF
    assert hass.states.get(BOOST_MINUTES).state == STATE_UNKNOWN
    # The end sensor's state is review finding B14 (text "Never" in a timestamp
    # sensor); only its label attribute is a settled contract.
    assert hass.states.get(BOOST_END).attributes["boost_end_label"] == "Never"


TOTAL = "sensor.home_total_energy"
HTR_ENERGY = "sensor.living_room_energy"
ACM_ENERGY = "sensor.store_energy"
PMO_ENERGY = "sensor.meter_energy"
# Entities of the heater (addr 1) and accumulator (addr 2) nodes.
NODE_ENTITIES = (
    "climate.living_room",
    "climate.store",
    "sensor.living_room_temperature",
    "number.living_room_priority",
    "number.store_boost_temperature",
    "binary_sensor.store_boost_active",
    "button.store_start_boost",
    "button.living_room_flash_display",
)


@pytest.fixture
def samples(cloud: FakeCloud) -> Generator[dict[tuple[str, str], list[dict]]]:
    """Serve a heater, an accumulator, a power monitor and a thermostat (addr 1-4)."""
    names = {"htr": "Living room", "acm": "Store", "pmo": "Meter", "thm": "Hall"}
    settings = {"mode": "auto", "state": "off", "stemp": "20.0", "units": "C"}
    cloud.get_nodes.return_value = {
        "nodes": [
            {"type": node_type, "addr": addr, "name": name}
            for addr, (node_type, name) in enumerate(names.items(), start=1)
        ]
    }
    cloud.get_node_settings.side_effect = lambda *_: dict(settings)
    now = time.time()
    # The thermostat meters no energy; the total sums heaters only.
    by_node = {
        ("htr", "1"): [{"t": now - 60, "counter": 1500}],
        ("acm", "2"): [{"t": now - 60, "counter": 2500}],
        ("pmo", "3"): [{"t": now - 60, "counter": 9000}],
    }

    async def _samples(_dev_id: str, node: tuple[str, str], *_: Any) -> list[dict]:
        return by_node[(node[0], str(node[1]))]

    cloud.get_node_samples.side_effect = _samples
    return by_node


async def _setup(hass: HomeAssistant, entry: MockConfigEntry) -> None:
    """Add the entry to hass and set it up."""
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()


async def _refresh(hass: HomeAssistant, entry: MockConfigEntry, name: str) -> None:
    """Advance time past the coordinator's update interval and let it poll."""
    coordinator = getattr(entry.runtime_data, name)
    async_fire_time_changed(
        hass, dt_util.utcnow() + coordinator.update_interval + timedelta(seconds=1)
    )
    await hass.async_block_till_done()


async def _fire_next_hourly_poll(hass: HomeAssistant) -> None:
    """Advance time to the next HH:05 energy poll and let it run."""
    now = dt_util.utcnow()
    target = (now + timedelta(hours=1)).replace(minute=5, second=0, microsecond=0)
    async_fire_time_changed(hass, target)
    await hass.async_block_till_done()


async def test_node_entities_unavailable_while_cloud_down(
    hass: HomeAssistant,
    cloud: FakeCloud,
    samples: dict,
    config_entry: MockConfigEntry,
) -> None:
    """A failed state poll marks node entities unavailable until it recovers."""
    await _setup(hass, config_entry)
    registry = er.async_get(hass)
    for entity_id in NODE_ENTITIES:
        assert registry.async_get(entity_id) is not None, entity_id
        assert hass.states.get(entity_id).state != STATE_UNAVAILABLE, entity_id

    # The cloud is down: every REST call fails.
    settings = cloud.get_node_settings.side_effect
    cloud.get_node_settings.side_effect = ClientError("cloud down")
    cloud.get_node_samples.side_effect = ClientError("cloud down")
    await _refresh(hass, config_entry, "coordinator")

    for entity_id in NODE_ENTITIES:
        assert hass.states.get(entity_id).state == STATE_UNAVAILABLE, entity_id

    cloud.get_node_settings.side_effect = settings
    await _refresh(hass, config_entry, "coordinator")

    for entity_id in NODE_ENTITIES:
        assert hass.states.get(entity_id).state != STATE_UNAVAILABLE, entity_id


async def test_energy_entities_unavailable_when_energy_poll_fails(
    hass: HomeAssistant,
    cloud: FakeCloud,
    samples: dict,
    config_entry: MockConfigEntry,
) -> None:
    """A failed energy poll marks the energy sensors and the total unavailable."""
    await _setup(hass, config_entry)
    assert float(hass.states.get(TOTAL).state) == pytest.approx(4.0)

    cloud.get_node_samples.side_effect = TimeoutError
    await _fire_next_hourly_poll(hass)

    for entity_id in (TOTAL, HTR_ENERGY, ACM_ENERGY, PMO_ENERGY):
        assert hass.states.get(entity_id).state == STATE_UNAVAILABLE, entity_id


async def test_installation_total_unknown_when_a_node_is_missing(
    hass: HomeAssistant,
    samples: dict,
    config_entry: MockConfigEntry,
) -> None:
    """The total is unknown, not a partial sum, while one heater has no reading."""
    samples[("acm", "2")] = []
    await _setup(hass, config_entry)

    assert float(hass.states.get(HTR_ENERGY).state) == pytest.approx(1.5)
    assert hass.states.get(TOTAL).state == STATE_UNKNOWN


async def test_installation_total_sums_every_node(
    hass: HomeAssistant,
    samples: dict,
    config_entry: MockConfigEntry,
) -> None:
    """With every heater reporting, the total is their sum."""
    await _setup(hass, config_entry)

    assert float(hass.states.get(TOTAL).state) == pytest.approx(4.0)


async def test_ducaheat_child_lock_unavailable_while_cloud_down(
    hass: HomeAssistant,
    cloud: FakeCloud,
    samples: dict,
) -> None:
    """The Ducaheat child lock also follows the state poll's success."""
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
        locks = [
            e.entity_id
            for e in er.async_entries_for_config_entry(
                er.async_get(hass), entry.entry_id
            )
            if e.domain == "lock"
        ]
        assert locks
        for entity_id in locks:
            assert hass.states.get(entity_id).state != STATE_UNAVAILABLE

        cloud.get_node_settings.side_effect = ClientError("cloud down")
        cloud.get_node_samples.side_effect = ClientError("cloud down")
        await _refresh(hass, entry, "coordinator")

        for entity_id in locks:
            assert hass.states.get(entity_id).state == STATE_UNAVAILABLE


def test_supports_boost_accepts_boolean_attribute() -> None:
    """A boolean ``supports_boost`` attribute should be returned verbatim."""

    node = types.SimpleNamespace(supports_boost=True)

    assert boost_module.supports_boost(node) is True


def test_supports_boost_handles_callable_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Callable ``supports_boost`` errors should be logged and return ``False``."""

    class FailingNode:
        addr = "01"

        def supports_boost(self) -> bool:
            raise RuntimeError("boom")

    with caplog.at_level("DEBUG"):
        assert boost_module.supports_boost(FailingNode()) is False
    assert "Ignoring boost support probe failure" in caplog.text


def test_supports_boost_defaults_to_false_for_unknown_value() -> None:
    """Unsupported values should fall back to ``False``."""

    node = types.SimpleNamespace(supports_boost="maybe")

    assert boost_module.supports_boost(node) is False


def test_iter_nodes_metadata_covers_branch_variants(
    monkeypatch: pytest.MonkeyPatch, inventory_builder
) -> None:
    """Inventory metadata iterator should cope with mixed node structures."""

    inventory = inventory_builder("dev", {})

    htr_node = types.SimpleNamespace(addr="4", type="htr", supports_boost="no")
    acm_node = types.SimpleNamespace(
        addr="5", type="acm", supports_boost=lambda: "true"
    )
    pmo_node = types.SimpleNamespace(addr="6", type="pmo", supports_boost=None)

    object.__setattr__(
        inventory,
        "_nodes_by_type_cache",
        {
            " ": (types.SimpleNamespace(addr="ignored"),),
            "htr": (
                types.SimpleNamespace(addr=" ", type="htr"),
                htr_node,
            ),
            "acm": (
                types.SimpleNamespace(addr=None, type="acm"),
                acm_node,
            ),
            "pmo": (pmo_node,),
        },
    )
    object.__setattr__(
        inventory,
        "_heater_address_map_cache",
        (
            {
                " ": ("1",),
                "acm-empty": (),
                "htr": ("4",),
                "acm": ("", "5"),
                "thm": ("7",),
            },
            {},
        ),
    )

    def _fake_resolve(
        self: Inventory,
        node_type: str,
        addr: str,
        *,
        default_factory: Any | None = None,
    ) -> str:
        return f"{node_type}:{addr}"

    monkeypatch.setattr(Inventory, "resolve_heater_name", _fake_resolve)

    results = list(
        inventory.iter_nodes_metadata(
            node_types=("htr", "acm"),
            default_name_simple=lambda addr: f"Heater {addr}",
        )
    )

    assert [(meta.node_type, meta.addr) for meta in results] == [
        ("htr", "4"),
        ("acm", "5"),
    ]
    assert results[0].name == "htr:4"
    assert boost_module.supports_boost(results[0].node) is False
    assert results[1].name == "acm:5"
    assert boost_module.supports_boost(results[1].node) is True


HTR = "climate.living_room"
PRIORITY = "number.living_room_priority"
SETTINGS = {
    "mode": "manual",
    "state": "off",
    "stemp": "21.0",
    "units": "C",
    "lock": False,
    "priority": 1,
}


class Ducaheat:
    """Fake Ducaheat cloud with two heaters (addr 1 and 3)."""

    def __init__(self, cloud: FakeCloud) -> None:
        """Serve two heaters and record reads and writes."""
        cloud.get_nodes.return_value = {
            "nodes": [
                {"type": "htr", "addr": 1, "name": "Living room"},
                {"type": "htr", "addr": 3, "name": "Bedroom"},
            ]
        }
        self.reads: list[str] = []
        self.cloud = cloud
        self.settings = AsyncMock(return_value=None)
        self.lock = AsyncMock(return_value=None)
        self.priority = AsyncMock(return_value=None)

    async def get_node_settings(
        self, _dev_id: str, node: tuple[str, Any]
    ) -> dict[str, Any]:
        """Record which node was read by REST."""
        self.reads.append(str(node[1]))
        return dict(SETTINGS)


@pytest.fixture
def ducaheat(cloud: FakeCloud) -> Generator[Ducaheat]:
    """Patch the Ducaheat REST client reads and writes."""
    fake = Ducaheat(cloud)
    with (
        patch.object(DucaheatRESTClient, "get_node_settings", fake.get_node_settings),
        patch.object(DucaheatRESTClient, "get_node_samples", cloud.get_node_samples),
        patch.object(DucaheatRESTClient, "set_node_settings", fake.settings),
        patch.object(DucaheatRESTClient, "set_node_lock", fake.lock),
        patch.object(DucaheatRESTClient, "set_node_priority", fake.priority),
        patch.object(
            DucaheatBackend,
            "create_ws_client",
            lambda _self, hass, *a, **kw: cloud.create_ws_client(hass, *a, **kw),
        ),
        patch.object(climate_module, "_WRITE_DEBOUNCE", 0),
        patch.object(entity_module, "WS_ECHO_FALLBACK_REFRESH", 0),
    ):
        yield fake


async def _setup_cloud_entry(hass: HomeAssistant) -> MockConfigEntry:
    """Set up a Ducaheat entry and return it."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        unique_id=USERNAME,
        data={"username": USERNAME, "password": PASSWORD, CONF_BRAND: BRAND_DUCAHEAT},
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return entry


def _set_ws(hass: HomeAssistant, entry: MockConfigEntry, *, healthy: bool) -> None:
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


def _lock_entity_id(hass: HomeAssistant, entry: MockConfigEntry) -> str:
    """Return the child-lock entity of the living-room heater."""
    return next(
        e.entity_id
        for e in er.async_entries_for_config_entry(er.async_get(hass), entry.entry_id)
        if e.domain == "lock" and e.unique_id.endswith(":1:child_lock")
    )


async def _settle(hass: HomeAssistant) -> None:
    """Let write tasks and any fallback refresh run."""
    for _ in range(5):
        await asyncio.sleep(0)
        await hass.async_block_till_done()


async def _write(hass: HomeAssistant, entry: MockConfigEntry, target: str) -> None:
    """Perform one entity write of the given kind on heater 1."""
    if target == "climate":
        data = {ATTR_ENTITY_ID: HTR, "temperature": 23}
        await hass.services.async_call(
            "climate", "set_temperature", data, blocking=True
        )
    elif target == "lock":
        data = {ATTR_ENTITY_ID: _lock_entity_id(hass, entry)}
        await hass.services.async_call("lock", "lock", data, blocking=True)
    else:
        data = {ATTR_ENTITY_ID: PRIORITY, "value": 5}
        await hass.services.async_call("number", "set_value", data, blocking=True)
    await _settle(hass)


WRITES = ["climate", "lock", "priority"]


@pytest.mark.parametrize("target", WRITES)
async def test_write_with_healthy_ws_reads_nothing(
    hass: HomeAssistant, ducaheat: Ducaheat, target: str
) -> None:
    """With the WebSocket healthy a write triggers no REST read at all."""
    entry = await _setup_cloud_entry(hass)
    _set_ws(hass, entry, healthy=True)
    ducaheat.reads.clear()

    await _write(hass, entry, target)

    assert (
        ducaheat.settings.await_count
        + ducaheat.lock.await_count
        + (ducaheat.priority.await_count)
        == 1
    )
    assert ducaheat.reads == []


@pytest.mark.parametrize("target", WRITES)
async def test_write_with_ws_down_refreshes_only_that_node(
    hass: HomeAssistant, ducaheat: Ducaheat, target: str
) -> None:
    """With the WebSocket down a write refreshes only the written node, once."""
    entry = await _setup_cloud_entry(hass)
    _set_ws(hass, entry, healthy=False)
    ducaheat.reads.clear()

    await _write(hass, entry, target)

    assert ducaheat.reads == ["1"]


async def test_lock_write_is_optimistic(
    hass: HomeAssistant, ducaheat: Ducaheat
) -> None:
    """The lock shows the written state at once, without waiting for a read."""
    entry = await _setup_cloud_entry(hass)
    _set_ws(hass, entry, healthy=True)
    lock = _lock_entity_id(hass, entry)
    assert hass.states.get(lock).state == "unlocked"
    ducaheat.reads.clear()

    await _write(hass, entry, "lock")
    assert hass.states.get(lock).state == "locked"

    await hass.services.async_call("lock", "unlock", {ATTR_ENTITY_ID: lock}, True)
    await _settle(hass)
    assert hass.states.get(lock).state == "unlocked"
    assert ducaheat.reads == []


async def test_priority_write_is_optimistic(
    hass: HomeAssistant, ducaheat: Ducaheat
) -> None:
    """The priority number shows the written value at once."""
    entry = await _setup_cloud_entry(hass)
    _set_ws(hass, entry, healthy=True)

    await _write(hass, entry, "priority")

    assert hass.states.get(PRIORITY).state == "5"


async def test_fallback_cancelled_when_entity_removed(
    hass: HomeAssistant, ducaheat: Ducaheat
) -> None:
    """A pending fallback refresh does not run after the entry is unloaded."""
    entry = await _setup_cloud_entry(hass)
    _set_ws(hass, entry, healthy=False)
    ducaheat.reads.clear()

    with patch.object(entity_module, "WS_ECHO_FALLBACK_REFRESH", 3600):
        await _write(hass, entry, "lock")
        assert await hass.config_entries.async_unload(entry.entry_id)
        await hass.async_block_till_done()

    assert ducaheat.reads == []


async def test_power_limit_write_reads_nothing(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """Writing the power limit updates the number without any REST read."""
    cloud.get_power_limit.return_value = 1000
    set_limit = AsyncMock(return_value=None)
    with patch.object(RESTClient, "set_power_limit", set_limit):
        config_entry.add_to_hass(hass)
        assert await hass.config_entries.async_setup(config_entry.entry_id)
        await hass.async_block_till_done()
        number = next(
            e.entity_id
            for e in er.async_entries_for_config_entry(
                er.async_get(hass), config_entry.entry_id
            )
            if e.unique_id.endswith("power_limit")
        )
        cloud.get_node_settings.reset_mock()
        cloud.get_power_limit.reset_mock()

        await hass.services.async_call(
            "number", "set_value", {ATTR_ENTITY_ID: number, "value": 2000}, True
        )
        await _settle(hass)

    set_limit.assert_awaited_once()
    assert hass.states.get(number).state == "2000"
    cloud.get_node_settings.assert_not_awaited()
    cloud.get_power_limit.assert_not_awaited()


async def test_fallback_refresh_failure_is_logged(
    hass: HomeAssistant, ducaheat: Ducaheat, caplog: pytest.LogCaptureFixture
) -> None:
    """A failing fallback refresh is logged instead of crashing."""
    entry = await _setup_cloud_entry(hass)
    _set_ws(hass, entry, healthy=False)
    coordinator = entry.runtime_data.coordinator

    with patch.object(
        coordinator, "async_refresh_heater", AsyncMock(side_effect=RuntimeError("x"))
    ):
        await _write(hass, entry, "lock")

    assert "Refresh fallback failed" in caplog.text
