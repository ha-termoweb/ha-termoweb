"""Unit tests for child lock entities in the lock domain."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from custom_components.termoweb.domain import DomainStateStore, NodeId, NodeType
from custom_components.termoweb.domain.state import HeaterState
from custom_components.termoweb.entities.lock import ChildLockEntity
from custom_components.termoweb.inventory import Inventory, build_node_inventory


@pytest.fixture
def lock_inventory() -> Inventory:
    """Return an inventory with one heater node."""

    payload = {"nodes": [{"type": "htr", "addr": "1", "name": "Master Bedroom"}]}
    return Inventory("dev", build_node_inventory(payload))


def test_child_lock_entity_available_without_lock_value(
    lock_inventory: Inventory,
) -> None:
    """Lock availability should only depend on immutable inventory membership."""

    coordinator = SimpleNamespace()
    lock_entity = ChildLockEntity(
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

    assert lock_entity.available is True


def test_child_lock_entity_has_feature_name(lock_inventory: Inventory) -> None:
    """Lock name should describe the feature, not the node."""

    coordinator = SimpleNamespace()
    lock_entity = ChildLockEntity(
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

    assert lock_entity.name == "Child lock"


def test_child_lock_entity_reports_is_locked(lock_inventory: Inventory) -> None:
    """Lock state should mirror the domain child lock value."""

    coordinator = SimpleNamespace()
    lock_entity = ChildLockEntity(
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

    assert lock_entity.is_locked is True


def test_child_lock_entity_uses_locked_icon_when_engaged(
    lock_inventory: Inventory,
) -> None:
    """Lock icon should show a locked icon when child lock is enabled."""

    coordinator = SimpleNamespace()
    lock_entity = ChildLockEntity(
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

    assert lock_entity.icon == "mdi:lock"


def test_child_lock_entity_uses_unlocked_icon_when_disengaged(
    lock_inventory: Inventory,
) -> None:
    """Lock icon should show an unlocked icon when child lock is disabled."""

    coordinator = SimpleNamespace()
    lock_entity = ChildLockEntity(
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

    assert lock_entity.icon == "mdi:lock-open-variant"


@pytest.mark.asyncio
async def test_child_lock_entity_lock_unlock_writes_backend(
    lock_inventory: Inventory,
    runtime_factory,
) -> None:
    """Lock commands should call backend lock API for both states."""

    backend = SimpleNamespace(set_node_lock=AsyncMock())
    coordinator = SimpleNamespace(async_request_refresh=AsyncMock())
    lock_entity = ChildLockEntity(
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
    lock_entity.hass = hass

    await lock_entity.async_lock()
    await lock_entity.async_unlock()

    assert backend.set_node_lock.await_args_list == [
        (("dev", ("htr", "1")), {"lock": True}),
        (("dev", ("htr", "1")), {"lock": False}),
    ]
    assert coordinator.async_request_refresh.await_count == 2


def test_child_lock_entity_reports_unlocked(lock_inventory: Inventory) -> None:
    """Lock state should report False when child lock is disabled."""

    coordinator = SimpleNamespace()
    lock_entity = ChildLockEntity(
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

    assert lock_entity.is_locked is False


def test_child_lock_entity_is_locked_none_when_resolver_returns_none(
    lock_inventory: Inventory,
) -> None:
    """Lock state should be None when settings resolver returns None."""

    coordinator = SimpleNamespace()
    lock_entity = ChildLockEntity(
        coordinator,
        "entry",
        "dev",
        "htr",
        "1",
        unique_id="dev_htr_1_child_lock",
        inventory=lock_inventory,
        settings_resolver=lambda: None,
        device_name="Master Bedroom",
    )

    assert lock_entity.is_locked is None


def test_child_lock_device_info_model_for_heater_and_accumulator(
    lock_inventory: Inventory,
) -> None:
    """device_info model should reflect node type (Heater vs Accumulator)."""

    coordinator = SimpleNamespace()

    # htr -> Heater
    lock_htr = ChildLockEntity(
        coordinator, "entry", "dev", "htr", "1",
        unique_id="dev_htr_1_lock",
        inventory=lock_inventory,
        settings_resolver=lambda: HeaterState(lock=True),
        device_name="Room",
    )
    info = lock_htr.device_info
    assert info["model"] == "Heater"

    # acm -> Accumulator
    acm_payload = {"nodes": [{"type": "acm", "addr": "2", "name": "Store"}]}
    acm_inventory = Inventory("dev", build_node_inventory(acm_payload))
    lock_acm = ChildLockEntity(
        coordinator, "entry", "dev", "acm", "2",
        unique_id="dev_acm_2_lock",
        inventory=acm_inventory,
        settings_resolver=lambda: HeaterState(lock=True),
        device_name="Store",
    )
    info_acm = lock_acm.device_info
    assert info_acm["model"] == "Accumulator"


def test_child_lock_not_available_when_addr_missing_from_inventory() -> None:
    """Lock entity should not be available when address removed from inventory."""

    empty_inventory = Inventory("dev", [])
    coordinator = SimpleNamespace()
    lock_entity = ChildLockEntity(
        coordinator,
        "entry",
        "dev",
        "htr",
        "99",
        unique_id="dev_htr_99_child_lock",
        inventory=empty_inventory,
        settings_resolver=lambda: HeaterState(lock=True),
        device_name="Missing",
    )

    assert lock_entity.available is False


def test_child_lock_entity_raises_on_blank_type_or_addr() -> None:
    """ChildLockEntity should reject blank node_type or addr."""

    import pytest as _pytest

    coordinator = SimpleNamespace()
    empty_inventory = Inventory("dev", [])

    with _pytest.raises(ValueError, match="node_type and addr must be provided"):
        ChildLockEntity(
            coordinator, "entry", "dev", "  ", "1",
            unique_id="u", inventory=empty_inventory,
            settings_resolver=lambda: None,
        )

    with _pytest.raises(ValueError, match="node_type and addr must be provided"):
        ChildLockEntity(
            coordinator, "entry", "dev", "htr", "  ",
            unique_id="u", inventory=empty_inventory,
            settings_resolver=lambda: None,
        )


@pytest.mark.asyncio
async def test_lock_async_setup_entry_creates_entities(
    runtime_factory,
) -> None:
    """async_setup_entry should create lock entities for htr and acm nodes."""

    from conftest import _install_stubs
    _install_stubs()
    from homeassistant.core import HomeAssistant
    from custom_components.termoweb.entities.lock import async_setup_entry

    payload = {
        "nodes": [
            {"type": "htr", "addr": "1", "name": "Living Room"},
            {"type": "acm", "addr": "2", "name": "Basement"},
            {"type": "pmo", "addr": "3", "name": "Power Module"},
        ]
    }
    inventory = Inventory("dev-lock", build_node_inventory(payload))
    coordinator = SimpleNamespace(
        hass=None,
        data={},
        async_request_refresh=AsyncMock(),
    )
    hass = HomeAssistant()
    runtime_factory(
        hass=hass,
        entry_id="entry-lock",
        dev_id="dev-lock",
        inventory=inventory,
        coordinator=coordinator,
    )

    added = []

    def _add_entities(entities):
        added.extend(entities)

    entry = SimpleNamespace(entry_id="entry-lock")
    await async_setup_entry(hass, entry, _add_entities)

    # Only htr and acm should get lock entities, not pmo
    assert len(added) == 2
    types_seen = {e._node_type for e in added}
    assert types_seen == {"htr", "acm"}


@pytest.mark.asyncio
async def test_lock_async_setup_entry_raises_without_inventory(
    runtime_factory,
) -> None:
    """async_setup_entry should raise TypeError when inventory is missing."""

    from conftest import _install_stubs
    _install_stubs()
    from homeassistant.core import HomeAssistant
    from custom_components.termoweb.entities.lock import async_setup_entry

    coordinator = SimpleNamespace(hass=None, data={})
    hass = HomeAssistant()
    runtime_factory(
        hass=hass,
        entry_id="entry-no-inv",
        dev_id="dev-no-inv",
        coordinator=coordinator,
        allow_missing_inventory=True,
    )

    entry = SimpleNamespace(entry_id="entry-no-inv")
    with pytest.raises(TypeError, match="inventory unavailable"):
        await async_setup_entry(hass, entry, lambda _: None)


def test_iter_lockable_inventory_nodes_filters_types() -> None:
    """_iter_lockable_inventory_nodes should yield only htr and acm nodes."""

    from custom_components.termoweb.entities.lock import _iter_lockable_inventory_nodes

    payload = {
        "nodes": [
            {"type": "htr", "addr": "1", "name": "Heater"},
            {"type": "acm", "addr": "2", "name": "Accumulator"},
            {"type": "thm", "addr": "3", "name": "Thermostat"},
            {"type": "pmo", "addr": "4", "name": "Power"},
            {"type": "htr", "addr": "  ", "name": "Blank addr"},
        ]
    }
    inventory = Inventory("dev", build_node_inventory(payload))

    results = list(_iter_lockable_inventory_nodes(inventory))

    # Only htr:1 and acm:2 should pass (blank addr filtered, thm/pmo excluded)
    assert len(results) == 2
    types_addrs = [(t, a) for t, a, _ in results]
    assert ("htr", "1") in types_addrs
    assert ("acm", "2") in types_addrs


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
