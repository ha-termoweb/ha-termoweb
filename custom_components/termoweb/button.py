"""Button platform entities for TermoWeb gateways."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
import logging
from typing import Any

from homeassistant.components.button import ButtonEntity
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity import DeviceInfo, EntityCategory
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .backend.sanitize import mask_identifier, redact_text
from .domain.ids import HEATING_NODE_TYPES
from .domain.state import DomainState
from .entity import (
    BOOST_BUTTON_METADATA,
    BoostButtonMetadata,
    async_cancel_acm_boost,
    async_start_acm_boost,
    derive_boost_state_from_domain,
    log_skipped_nodes,
    resolve_boost_runtime_minutes,
)
from .identifiers import build_gateway_entity_unique_id, build_heater_unique_id
from .inventory import (
    AccumulatorNode,
    Inventory,
)
from .runtime import require_runtime
from .utils import build_gateway_device_info, build_node_device_info

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DisplayFlashContext:
    """Inventory-backed context describing a node display-flash target."""

    entry_id: str
    dev_id: str
    node_type: str
    addr: str
    name: str

    @property
    def unique_id(self) -> str:
        """Return the stable unique ID for this flash button."""

        return build_heater_unique_id(
            self.dev_id,
            self.node_type,
            self.addr,
            suffix=":flash_display",
        )


def _iter_display_flash_contexts(
    entry_id: str,
    inventory: Inventory,
) -> Iterator[DisplayFlashContext]:
    """Yield flash-button contexts for flash-capable inventory nodes."""

    for metadata in inventory.iter_nodes_metadata(node_types=HEATING_NODE_TYPES):
        yield DisplayFlashContext(
            entry_id=entry_id,
            dev_id=inventory.dev_id,
            node_type=metadata.node_type,
            addr=metadata.addr,
            name=metadata.name,
        )


@dataclass(frozen=True, slots=True)
class AccumulatorBoostContext:
    """Inventory-backed context describing an accumulator boost target."""

    entry_id: str
    inventory: Inventory
    node: AccumulatorNode
    base_name: str

    @classmethod
    def from_inventory(
        cls,
        entry_id: str,
        inventory: Inventory,
        node: AccumulatorNode,
    ) -> AccumulatorBoostContext:
        """Build context for ``node`` using the shared inventory."""

        base_name = inventory.resolve_heater_name(node.type, node.addr)
        return cls(entry_id, inventory, node, base_name)

    @property
    def dev_id(self) -> str:
        """Return the gateway identifier for the accumulator node."""

        return self.inventory.dev_id

    @property
    def node_type(self) -> str:
        """Return the canonical node type for the accumulator node."""

        return self.node.type

    @property
    def addr(self) -> str:
        """Return the canonical address for the accumulator node."""

        return self.node.addr

    def unique_id(self, suffix: str) -> str:
        """Return the unique ID for a boost helper with ``suffix``."""

        return build_heater_unique_id(
            self.dev_id, self.node_type, self.addr, suffix=f"boost_{suffix}"
        )


def _iter_accumulator_contexts(
    entry_id: str,
    inventory: Inventory,
) -> Iterator[AccumulatorBoostContext]:
    """Yield boost contexts for accumulator nodes in ``inventory``."""

    for metadata in inventory.iter_nodes_metadata(node_types=("acm",)):
        yield AccumulatorBoostContext.from_inventory(entry_id, inventory, metadata.node)


async def async_setup_entry(hass, entry, async_add_entities):
    """Expose hub refresh and accumulator boost helper buttons."""
    runtime = require_runtime(hass, entry.entry_id)
    coordinator = runtime.coordinator
    dev_id = runtime.dev_id

    entities: list[ButtonEntity] = [
        StateRefreshButton(coordinator, entry.entry_id, dev_id)
    ]

    inventory = runtime.inventory

    log_skipped_nodes("button", inventory, logger=_LOGGER)

    boost_entities: list[ButtonEntity] = []
    for context in _iter_accumulator_contexts(entry.entry_id, inventory):
        boost_entities.extend(_create_boost_button_entities(coordinator, context))

    if boost_entities:
        entities.extend(boost_entities)

    flash_entities = [
        DisplayFlashButton(coordinator, context)
        for context in _iter_display_flash_contexts(entry.entry_id, inventory)
    ]
    entities.extend(flash_entities)

    async_add_entities(entities)


class StateRefreshButton(CoordinatorEntity, ButtonEntity):
    """Button that requests an immediate coordinator refresh."""

    _attr_has_entity_name = True
    _attr_translation_key = "force_refresh"

    def __init__(self, coordinator, entry_id: str, dev_id: str) -> None:
        """Initialise the force-refresh button entity."""
        super().__init__(coordinator)
        self._entry_id = entry_id
        self._dev_id = dev_id
        self._attr_unique_id = build_gateway_entity_unique_id(dev_id, "refresh")

    @property
    def device_info(self) -> DeviceInfo:
        """Return the Home Assistant device metadata for this gateway."""
        return build_gateway_device_info(
            self.hass,
            getattr(self, "_entry_id", None),
            self._dev_id,
        )

    async def async_press(self) -> None:
        """Request an immediate coordinator refresh when pressed."""
        await self.coordinator.async_request_refresh()


class AccumulatorBoostButtonBase(CoordinatorEntity, ButtonEntity):
    """Base entity for TermoWeb accumulator boost helper buttons."""

    _attr_entity_category = EntityCategory.CONFIG
    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator,
        context: AccumulatorBoostContext,
        *,
        unique_suffix: str,
        icon: str,
    ) -> None:
        """Initialise an accumulator boost helper button."""

        super().__init__(coordinator)
        self._boost_context = context
        self._attr_unique_id = context.unique_id(unique_suffix)
        self._attr_icon = icon

    @property
    def boost_context(self) -> AccumulatorBoostContext:
        """Return the inventory-derived context for this entity."""

        return self._boost_context

    @property
    def available(self) -> bool:
        """Return True when the last update succeeded and the node is in inventory."""

        if not super().available:
            return False
        forward_map, _ = self.boost_context.inventory.heater_address_map
        return self.boost_context.addr in forward_map.get(
            self.boost_context.node_type, ()
        )

    def _coordinator_state(self) -> DomainState | None:
        """Return cached coordinator state for this accumulator."""

        return self.coordinator.domain_view.get_heater_state(
            self.boost_context.node_type,
            self.boost_context.addr,
        )

    def _coordinator_boost_active(self) -> bool:
        """Return True when coordinator cache reports boost activity."""

        state = derive_boost_state_from_domain(
            self._coordinator_state(), getattr(self, "coordinator", None)
        )
        return bool(state.active)

    @property
    def device_info(self) -> DeviceInfo:
        """Return Home Assistant device metadata for the accumulator."""

        return build_node_device_info(
            self.hass,
            self.boost_context.entry_id,
            self.boost_context.dev_id,
            self.boost_context.addr,
            name=self.boost_context.base_name,
            node_type=self.boost_context.node_type,
        )

    async def _async_boost_request(
        self,
        hass: HomeAssistant,
        action: str,
        request: Callable[[Any], Awaitable[None]],
    ) -> None:
        """Run a boost backend request, raising HomeAssistantError on failure."""

        context = self.boost_context
        runtime = require_runtime(hass, context.entry_id)
        _LOGGER.info(
            "Requesting boost %s for %s/%s node %s",
            action,
            mask_identifier(context.dev_id),
            context.node_type,
            context.addr,
        )
        try:
            await request(runtime.backend)
        except Exception as err:
            _LOGGER.error(
                "Boost %s failed for %s/%s node %s: %s",
                action,
                mask_identifier(context.dev_id),
                context.node_type,
                context.addr,
                redact_text(str(err)),
            )
            raise HomeAssistantError(
                f"Unable to {action} the accumulator boost"
            ) from err


class AccumulatorBoostButton(AccumulatorBoostButtonBase):
    """Button that starts an accumulator boost using persisted presets."""

    _attr_icon = "mdi:flash-outline"
    _attr_translation_key = "accumulator_boost_start"

    def __init__(
        self,
        coordinator,
        context: AccumulatorBoostContext,
        metadata: BoostButtonMetadata,
    ) -> None:
        """Initialise the boost helper button that uses stored presets."""

        super().__init__(
            coordinator,
            context,
            unique_suffix=metadata.unique_suffix,
            icon=metadata.icon,
        )

    async def async_press(self) -> None:
        """Start a boost for the stored duration at the device boost setpoint."""

        hass = self.hass
        context = self.boost_context
        state = self._coordinator_state()
        minutes = resolve_boost_runtime_minutes(state)
        await self._async_boost_request(
            hass,
            "start",
            lambda backend: async_start_acm_boost(
                backend, context.dev_id, context.addr, state, minutes=minutes
            ),
        )


class AccumulatorBoostCancelButton(AccumulatorBoostButtonBase):
    """Button that cancels an active accumulator boost session."""

    _attr_icon = "mdi:flash-off"
    _attr_translation_key = "accumulator_boost_cancel"

    def __init__(
        self,
        coordinator,
        context: AccumulatorBoostContext,
        metadata: BoostButtonMetadata,
    ) -> None:
        """Initialise the boost cancellation helper button."""

        super().__init__(
            coordinator,
            context,
            unique_suffix=metadata.unique_suffix,
            icon=metadata.icon,
        )

    @property
    def available(self) -> bool:
        """Return True when an accumulator boost is active."""

        if not super().available:
            return False
        return self._coordinator_boost_active()

    async def async_press(self) -> None:
        """Cancel the active accumulator boost session."""

        hass = self.hass
        context = self.boost_context
        await self._async_boost_request(
            hass,
            "cancel",
            lambda backend: async_cancel_acm_boost(
                backend, context.dev_id, context.addr
            ),
        )


class DisplayFlashButton(CoordinatorEntity, ButtonEntity):
    """Button that triggers the backend display identify endpoint."""

    _attr_entity_category = EntityCategory.CONFIG
    _attr_has_entity_name = True
    _attr_icon = "mdi:gesture-tap-button"
    _attr_translation_key = "flash_display"

    def __init__(self, coordinator, context: DisplayFlashContext) -> None:
        """Initialise the display flash button entity."""

        super().__init__(coordinator)
        self._flash_context = context
        self._attr_unique_id = context.unique_id

    @property
    def available(self) -> bool:
        """Return True when the last update succeeded and the node is in inventory."""

        inventory = getattr(self.coordinator, "_inventory", None)
        return bool(
            super().available
            and isinstance(inventory, Inventory)
            and inventory.has_node(
                self._flash_context.node_type,
                self._flash_context.addr,
            )
        )

    @property
    def device_info(self) -> DeviceInfo:
        """Expose Home Assistant device metadata for the flash target."""

        return build_node_device_info(
            self.hass,
            self._flash_context.entry_id,
            self._flash_context.dev_id,
            self._flash_context.addr,
            name=self._flash_context.name,
            node_type=self._flash_context.node_type,
        )

    async def async_press(self) -> None:
        """Call the backend /select endpoint to flash the unit display."""

        hass = self.hass
        runtime = require_runtime(hass, self._flash_context.entry_id)
        _LOGGER.info(
            "Requesting display flash for %s/%s node %s",
            mask_identifier(self._flash_context.dev_id),
            self._flash_context.node_type,
            self._flash_context.addr,
        )
        try:
            await runtime.backend.set_node_display_select(
                self._flash_context.dev_id,
                (self._flash_context.node_type, self._flash_context.addr),
                select=True,
            )
        except Exception as err:
            _LOGGER.error(
                "Display flash failed for %s/%s node %s: %s",
                mask_identifier(self._flash_context.dev_id),
                self._flash_context.node_type,
                self._flash_context.addr,
                redact_text(str(err)),
            )
            raise HomeAssistantError("Unable to flash the unit display") from err


def _create_boost_button_entities(
    coordinator,
    context: AccumulatorBoostContext,
) -> list[ButtonEntity]:
    """Return boost helper buttons described by shared metadata."""

    return [
        _build_boost_button(
            metadata,
            coordinator,
            context,
        )
        for metadata in BOOST_BUTTON_METADATA
    ]


def _build_boost_button(
    metadata: BoostButtonMetadata,
    coordinator,
    context: AccumulatorBoostContext,
) -> ButtonEntity:
    """Instantiate a boost helper button for ``metadata``."""

    if metadata.action == "start":
        return AccumulatorBoostButton(coordinator, context, metadata)
    return AccumulatorBoostCancelButton(coordinator, context, metadata)
