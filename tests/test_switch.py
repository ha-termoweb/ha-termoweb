"""Unit tests for child lock switch entities."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from custom_components.termoweb.domain import DomainStateStore, NodeId, NodeType
from custom_components.termoweb.domain.state import HeaterState
from custom_components.termoweb.entities.switch import ChildLockSwitch
from custom_components.termoweb.inventory import Inventory, build_node_inventory


@pytest.fixture
def lock_inventory() -> Inventory:
    """Return an inventory with one heater node."""

    payload = {"nodes": [{"type": "htr", "addr": "1", "name": "Master Bedroom"}]}
    return Inventory("dev", build_node_inventory(payload))


def test_child_lock_switch_available_without_lock_value(
    lock_inventory: Inventory,
) -> None:
    """Switch availability should only depend on immutable inventory membership."""

    coordinator = SimpleNamespace()
    switch = ChildLockSwitch(
        coordinator,
        "entry",
        "dev",
        "htr",
        "1",
        unique_id="dev_htr_1_child_lock",
        inventory=lock_inventory,
        settings_resolver=lambda: HeaterState(lock=None),
        device_name="Master Bedroom",
    )

    assert switch.available is True


def test_child_lock_switch_has_feature_name(lock_inventory: Inventory) -> None:
    """Switch name should describe the feature, not the node."""

    coordinator = SimpleNamespace()
    switch = ChildLockSwitch(
        coordinator,
        "entry",
        "dev",
        "htr",
        "1",
        unique_id="dev_htr_1_child_lock",
        inventory=lock_inventory,
        settings_resolver=lambda: HeaterState(lock=True),
        device_name="Master Bedroom",
    )

    assert switch.name == "Child lock"


def test_child_lock_switch_uses_locked_icon_when_engaged(
    lock_inventory: Inventory,
) -> None:
    """Switch icon should show a locked icon when child lock is enabled."""

    coordinator = SimpleNamespace()
    switch = ChildLockSwitch(
        coordinator,
        "entry",
        "dev",
        "htr",
        "1",
        unique_id="dev_htr_1_child_lock",
        inventory=lock_inventory,
        settings_resolver=lambda: HeaterState(lock=True),
        device_name="Master Bedroom",
    )

    assert switch.icon == "mdi:lock"


def test_child_lock_switch_uses_unlocked_icon_when_disengaged(
    lock_inventory: Inventory,
) -> None:
    """Switch icon should show an unlocked icon when child lock is disabled."""

    coordinator = SimpleNamespace()
    switch = ChildLockSwitch(
        coordinator,
        "entry",
        "dev",
        "htr",
        "1",
        unique_id="dev_htr_1_child_lock",
        inventory=lock_inventory,
        settings_resolver=lambda: HeaterState(lock=False),
        device_name="Master Bedroom",
    )

    assert switch.icon == "mdi:lock-open-variant"


@pytest.mark.asyncio
async def test_child_lock_switch_turn_on_off_writes_backend(
    lock_inventory: Inventory,
    runtime_factory,
) -> None:
    """Switch writes should call backend lock API for both states."""

    backend = SimpleNamespace(set_node_lock=AsyncMock())
    coordinator = SimpleNamespace(async_request_refresh=AsyncMock())
    switch = ChildLockSwitch(
        coordinator,
        "entry",
        "dev",
        "htr",
        "1",
        unique_id="dev_htr_1_child_lock",
        inventory=lock_inventory,
        settings_resolver=lambda: HeaterState(lock=False),
        device_name="Master Bedroom",
    )
    hass = SimpleNamespace(data={})
    runtime_factory(
        hass=hass,
        entry_id="entry",
        dev_id="dev",
        inventory=lock_inventory,
        coordinator=coordinator,
        backend=backend,
    )
    switch.hass = hass

    await switch.async_turn_on()
    await switch.async_turn_off()

    assert backend.set_node_lock.await_args_list == [
        (("dev", ("htr", "1")), {"lock": True}),
        (("dev", ("htr", "1")), {"lock": False}),
    ]
    assert coordinator.async_request_refresh.await_count == 2


def test_domain_state_lock_parses_on_off_strings() -> None:
    """Domain state snapshots should coerce string lock values to booleans."""

    store = DomainStateStore([NodeId(NodeType.HEATER, "1")])

    store.apply_full_snapshot("htr", "1", {"lock": "off"})
    state = store.get_state("htr", "1")
    assert isinstance(state, HeaterState)
    assert state.lock is False

    store.apply_patch("htr", "1", {"lock": "on"})
    patched = store.get_state("htr", "1")
    assert isinstance(patched, HeaterState)
    assert patched.lock is True
