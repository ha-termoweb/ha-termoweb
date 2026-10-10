"""Child lock entities on real Home Assistant (Ducaheat)."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

from homeassistant.components.lock import LockState
from homeassistant.const import ATTR_ICON, STATE_UNKNOWN
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.helpers.entity import EntityCategory
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.termoweb.const import BRAND_DUCAHEAT, DOMAIN
from custom_components.termoweb.domain.ids import NodeId, NodeType
from custom_components.termoweb.domain.state import DomainStateStore, NodeSettingsDelta

from .conftest import DEV_ID, FakeCloud
from .fakes.entities import FakeNodes, call, setup_entry

HTR_LOCK = "lock.living_room_child_lock"
ACM_LOCK = "lock.store_child_lock"


def _settings(**extra: Any) -> dict[str, Any]:
    """Return node settings with an unlocked child lock."""
    return {"mode": "auto", "state": "off", "stemp": "20.0", "units": "C", **extra}


@pytest.fixture
def nodes(cloud: FakeCloud) -> Generator[FakeNodes]:
    """Serve a heater, an accumulator, a thermostat and a power monitor on Ducaheat."""
    fake = FakeNodes(
        cloud,
        {
            ("htr", "1"): ("Living room", _settings(lock=False)),
            ("acm", "2"): ("Store", _settings(lock=True)),
            ("thm", "3"): ("Hall", _settings(lock=False)),
            ("pmo", "4"): (None, {}),
        },
    )
    with fake.patched(ducaheat=True):
        yield fake


async def test_child_locks_exist_for_heaters_and_accumulators_only(
    hass: HomeAssistant, nodes: FakeNodes, config_entry: MockConfigEntry
) -> None:
    """Only heaters and accumulators get a child lock, as a config entity."""
    entry = await setup_entry(hass, config_entry, brand=BRAND_DUCAHEAT)

    registry = er.async_get(hass)
    locks = {
        e.unique_id: e
        for e in er.async_entries_for_config_entry(registry, entry.entry_id)
        if e.domain == "lock"
    }
    assert set(locks) == {
        f"{DOMAIN}:{DEV_ID}:htr:1:child_lock",
        f"{DOMAIN}:{DEV_ID}:acm:2:child_lock",
    }
    devices = dr.async_get(hass)
    for lock in locks.values():
        assert lock.entity_category is EntityCategory.CONFIG
    acm_device = devices.async_get(
        locks[f"{DOMAIN}:{DEV_ID}:acm:2:child_lock"].device_id
    )
    assert acm_device.model == "Accumulator"
    assert (DOMAIN, DEV_ID, "2") in acm_device.identifiers


async def test_lock_state_and_icon_follow_the_device(
    hass: HomeAssistant, nodes: FakeNodes, config_entry: MockConfigEntry
) -> None:
    """The lock mirrors the device's child lock, with a matching icon."""
    await setup_entry(hass, config_entry, brand=BRAND_DUCAHEAT)

    unlocked = hass.states.get(HTR_LOCK)
    assert unlocked.state == LockState.UNLOCKED
    assert unlocked.attributes[ATTR_ICON] == "mdi:lock-open-variant"
    locked = hass.states.get(ACM_LOCK)
    assert locked.state == LockState.LOCKED
    assert locked.attributes[ATTR_ICON] == "mdi:lock"


async def test_lock_state_unknown_until_reported(
    hass: HomeAssistant, nodes: FakeNodes, config_entry: MockConfigEntry
) -> None:
    """A node that never reported its child lock shows an unknown lock."""
    del nodes.settings[("htr", "1")]["lock"]
    await setup_entry(hass, config_entry, brand=BRAND_DUCAHEAT)

    assert hass.states.get(HTR_LOCK).state == STATE_UNKNOWN


async def test_lock_and_unlock_write_the_child_lock(
    hass: HomeAssistant, nodes: FakeNodes, config_entry: MockConfigEntry
) -> None:
    """lock/unlock write the child lock and show the new state."""
    await setup_entry(hass, config_entry, brand=BRAND_DUCAHEAT)

    await call(hass, "lock", "lock", HTR_LOCK)
    nodes.set_lock.assert_awaited_once_with(DEV_ID, ("htr", "1"), lock=True)
    assert hass.states.get(HTR_LOCK).state == LockState.LOCKED

    await call(hass, "lock", "unlock", ACM_LOCK)
    nodes.set_lock.assert_awaited_with(DEV_ID, ("acm", "2"), lock=False)
    assert hass.states.get(ACM_LOCK).state == LockState.UNLOCKED


def test_lock_values_on_and_off_are_booleans() -> None:
    """Devices that report the lock as "on"/"off" are read as booleans."""
    node = NodeId(NodeType.HEATER, "1")
    store = DomainStateStore([node])

    store.apply_full_snapshot("htr", "1", {"lock": "off"})
    assert store.get_state("htr", "1").lock is False

    store.apply_delta(NodeSettingsDelta(node_id=node, changes={"lock": "on"}))
    assert store.get_state("htr", "1").lock is True
