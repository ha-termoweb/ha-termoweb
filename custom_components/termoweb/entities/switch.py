"""Switch entities for TermoWeb child lock."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import logging
from typing import Any

from homeassistant.components.switch import SwitchDeviceClass, SwitchEntity
from homeassistant.helpers.entity import DeviceInfo, EntityCategory
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.coordinator import StateCoordinator
from custom_components.termoweb.domain import DomainStateView
from custom_components.termoweb.domain.state import DomainState
from custom_components.termoweb.entities.heater import log_skipped_nodes
from custom_components.termoweb.i18n import (
    async_get_fallback_translations,
    attach_fallbacks,
)
from custom_components.termoweb.identifiers import build_heater_entity_unique_id
from custom_components.termoweb.inventory import (
    Inventory,
    normalize_node_addr,
    normalize_node_type,
)
from custom_components.termoweb.runtime import require_runtime

_LOGGER = logging.getLogger(__name__)

_LOCK_NODE_TYPES: frozenset[str] = frozenset({"htr", "acm"})

SettingsResolver = Callable[[], DomainState | None]


async def async_setup_entry(hass, entry, async_add_entities):
    """Set up child lock switches for heater and accumulator nodes."""

    runtime = require_runtime(hass, entry.entry_id)
    coord: StateCoordinator = runtime.coordinator
    dev_id = runtime.dev_id

    fallbacks = await async_get_fallback_translations(hass, runtime)
    attach_fallbacks(coord, fallbacks)

    inventory = runtime.inventory
    if not isinstance(inventory, Inventory):
        _LOGGER.error("TermoWeb switch setup missing inventory for device %s", dev_id)
        raise ValueError("TermoWeb inventory unavailable for switch platform")

    entities: list[SwitchEntity] = []
    for node_type, addr_str, base_name in _iter_lockable_inventory_nodes(inventory):
        unique_id = build_heater_entity_unique_id(
            dev_id,
            node_type,
            addr_str,
            ":child_lock",
        )
        settings_resolver = _build_settings_resolver(
            coord,
            dev_id,
            node_type,
            addr_str,
        )
        entities.append(
            ChildLockSwitch(
                coord,
                entry.entry_id,
                dev_id,
                node_type,
                addr_str,
                unique_id=unique_id,
                inventory=inventory,
                settings_resolver=settings_resolver,
                device_name=base_name,
            )
        )

    if entities:
        _LOGGER.debug("Adding %d TermoWeb child lock switches", len(entities))

    log_skipped_nodes("switch", inventory, logger=_LOGGER)
    async_add_entities(entities)


class ChildLockSwitch(
    CoordinatorEntity[StateCoordinator],
    SwitchEntity,
):
    """Switch controlling the child lock on a heater or accumulator."""

    _attr_device_class = SwitchDeviceClass.SWITCH
    _attr_entity_category = EntityCategory.CONFIG
    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_translation_key = "child_lock"

    def __init__(
        self,
        coordinator: StateCoordinator,
        entry_id: str,
        dev_id: str,
        node_type: str,
        addr: str,
        unique_id: str,
        *,
        inventory: Inventory,
        settings_resolver: SettingsResolver,
        device_name: str | None = None,
    ) -> None:
        """Initialise the child lock switch entity."""

        super().__init__(coordinator)
        canonical_type = normalize_node_type(node_type, use_default_when_falsey=True)
        canonical_addr = normalize_node_addr(addr, use_default_when_falsey=True)
        if not canonical_type or not canonical_addr:
            msg = "node_type and addr must be provided"
            raise ValueError(msg)

        self._entry_id = entry_id
        self._dev_id = str(dev_id)
        self._node_type = canonical_type
        self._addr = canonical_addr
        self._attr_unique_id = unique_id
        self._inventory = inventory
        self._settings_resolver = settings_resolver
        self._device_name = device_name if device_name else ""

    @property
    def is_on(self) -> bool | None:
        """Return True when the child lock is engaged."""

        state = self._settings_resolver()
        if state is None:
            return None
        lock_value = getattr(state, "lock", None)
        if lock_value is None:
            return None
        return bool(lock_value)

    @property
    def available(self) -> bool:
        """Return True when the node exists and reports lock state."""

        forward_map, _ = self._inventory.heater_address_map
        addresses = forward_map.get(self._node_type, [])
        if self._addr not in addresses:
            return False
        state = self._settings_resolver()
        return state is not None and getattr(state, "lock", None) is not None

    @property
    def device_info(self) -> DeviceInfo:
        """Expose Home Assistant device metadata for the node."""

        model = "Accumulator" if self._node_type == "acm" else "Heater"
        return DeviceInfo(
            identifiers={(DOMAIN, self._dev_id, self._addr)},
            name=self._device_name,
            manufacturer="TermoWeb",
            model=model,
            via_device=(DOMAIN, self._dev_id),
        )

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Enable the child lock."""

        runtime = require_runtime(self.hass, self._entry_id)
        await runtime.backend.set_node_lock(
            self._dev_id,
            (self._node_type, self._addr),
            lock=True,
        )
        await self.coordinator.async_request_refresh()

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Disable the child lock."""

        runtime = require_runtime(self.hass, self._entry_id)
        await runtime.backend.set_node_lock(
            self._dev_id,
            (self._node_type, self._addr),
            lock=False,
        )
        await self.coordinator.async_request_refresh()


def _iter_lockable_inventory_nodes(
    inventory: Inventory,
) -> Iterable[tuple[str, str, str]]:
    """Yield htr and acm node metadata from ``inventory``."""

    for metadata in inventory.iter_nodes_metadata(node_types=_LOCK_NODE_TYPES):
        canonical_type = normalize_node_type(
            metadata.node_type,
            use_default_when_falsey=True,
        )
        canonical_addr = normalize_node_addr(
            metadata.addr,
            use_default_when_falsey=True,
        )
        if not canonical_type or not canonical_addr:
            continue
        yield (canonical_type, canonical_addr, metadata.name)


def _build_settings_resolver(
    coordinator: StateCoordinator,
    dev_id: str,
    node_type: str,
    addr: str,
) -> SettingsResolver:
    """Return callable resolving the domain state for a node."""

    def _resolver() -> DomainState | None:
        view = getattr(coordinator, "domain_view", None)
        if isinstance(view, DomainStateView):
            return view.get_heater_state(node_type, addr)
        return None

    return _resolver
