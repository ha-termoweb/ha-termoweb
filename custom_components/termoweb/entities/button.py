"""Button platform entities for TermoWeb gateways."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import logging
from typing import Any

from homeassistant.components.button import ButtonEntity
from homeassistant.core import callback
from homeassistant.exceptions import HomeAssistantError, ServiceNotFound
from homeassistant.helpers.entity import DeviceInfo, EntityCategory
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from ..const import DOMAIN
from ..domain import DomainStateView
from ..domain.state import DomainState
from ..i18n import async_get_fallback_translations, attach_fallbacks
from ..identifiers import build_heater_entity_unique_id
from ..inventory import (
    AccumulatorNode,
    Inventory,
    normalize_node_addr,
    normalize_node_type,
)
from ..runtime import require_runtime
from ..utils import build_gateway_device_info
from .heater import (
    BOOST_BUTTON_METADATA,
    DEFAULT_BOOST_TEMPERATURE,
    BoostButtonMetadata,
    derive_boost_state_from_domain,
    log_skipped_nodes,
    resolve_boost_runtime_minutes,
    resolve_boost_temperature,
)

_LOGGER = logging.getLogger(__name__)

_SERVICE_REQUEST_ACCUMULATOR_BOOST = "request_accumulator_boost"
_SERVICE_CANCEL_ACCUMULATOR_BOOST = "cancel_accumulator_boost"


_FLASHABLE_NODE_TYPES: tuple[str, ...] = ("htr", "acm")


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

        return build_heater_entity_unique_id(
            self.dev_id,
            self.node_type,
            self.addr,
            ":flash_display",
        )


def _iter_display_flash_contexts(
    entry_id: str,
    inventory: Inventory,
) -> Iterator[DisplayFlashContext]:
    """Yield flash-button contexts for flash-capable inventory nodes."""

    for metadata in inventory.iter_nodes_metadata(node_types=_FLASHABLE_NODE_TYPES):
        node_type = normalize_node_type(
            metadata.node_type, use_default_when_falsey=True
        )
        addr = normalize_node_addr(metadata.addr, use_default_when_falsey=True)
        if not node_type or not addr:
            continue
        yield DisplayFlashContext(
            entry_id=entry_id,
            dev_id=inventory.dev_id,
            node_type=node_type,
            addr=addr,
            name=metadata.name,
        )


@dataclass(frozen=True, slots=True)
class AccumulatorBoostContext:
    """Inventory-backed context describing an accumulator boost target."""

    entry_id: str
    inventory: Inventory
    node: AccumulatorNode
    base_name: str
    unique_prefix: str

    @classmethod
    def from_inventory(
        cls,
        entry_id: str,
        inventory: Inventory,
        node: AccumulatorNode,
    ) -> AccumulatorBoostContext:
        """Build context for ``node`` using the shared inventory."""

        base_name = inventory.resolve_heater_name(node.type, node.addr)
        unique_prefix = build_heater_entity_unique_id(
            inventory.dev_id,
            node.type,
            node.addr,
            ":boost",
        )
        return cls(entry_id, inventory, node, base_name, unique_prefix)

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

        return f"{self.unique_prefix}_{suffix}"


def _iter_accumulator_contexts(
    entry_id: str,
    inventory: Inventory,
) -> Iterator[AccumulatorBoostContext]:
    """Yield boost contexts for accumulator nodes in ``inventory``."""

    for metadata in inventory.iter_nodes_metadata(node_types=("acm",)):
        node = metadata.node
        if not isinstance(node, AccumulatorNode):
            continue
        yield AccumulatorBoostContext.from_inventory(entry_id, inventory, node)


async def async_setup_entry(hass, entry, async_add_entities):
    """Expose hub refresh and accumulator boost helper buttons."""
    runtime = require_runtime(hass, entry.entry_id)
    coordinator = runtime.coordinator
    dev_id = runtime.dev_id

    fallbacks = await async_get_fallback_translations(hass, runtime)
    attach_fallbacks(coordinator, fallbacks)
    entities: list[ButtonEntity] = [
        StateRefreshButton(coordinator, entry.entry_id, dev_id)
    ]

    inventory = runtime.inventory
    if not isinstance(inventory, Inventory):
        _LOGGER.error("TermoWeb button setup missing inventory for device %s", dev_id)
        raise ValueError("TermoWeb inventory unavailable for button platform")

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
        self._attr_unique_id = f"{DOMAIN}:{dev_id}:refresh"

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
        label: str | None,
        unique_suffix: str,
        icon: str | None = None,
    ) -> None:
        """Initialise an accumulator boost helper button."""

        super().__init__(coordinator)
        self._boost_context = context
        if label:
            self._attr_name = label
        self._attr_unique_id = context.unique_id(unique_suffix)
        if icon is not None:
            self._attr_icon = icon

    @property
    def boost_context(self) -> AccumulatorBoostContext:
        """Return the inventory-derived context for this entity."""

        return self._boost_context

    @property
    def _service_minutes(self) -> int | None:
        """Return the minutes payload passed to the helper service."""

        return None

    @property
    def available(self) -> bool:
        """Return True when the inventory exposes this accumulator."""

        forward_map, _ = self.boost_context.inventory.heater_address_map
        return self.boost_context.addr in forward_map.get(
            self.boost_context.node_type, ()
        )

    async def async_added_to_hass(self) -> None:
        """Register coordinator listener hooks once the entity is added."""

        await super().async_added_to_hass()
        add_listener = getattr(self.coordinator, "async_add_listener", None)
        if callable(add_listener):
            remove = add_listener(self._handle_coordinator_update)
            if callable(remove):
                self.async_on_remove(remove)

    @callback
    def _handle_coordinator_update(self) -> None:
        """Refresh entity state when the coordinator updates."""

        self.async_write_ha_state()

    def _coordinator_state(self) -> DomainState | None:
        """Return cached coordinator state for this accumulator."""

        coordinator = getattr(self, "coordinator", None)
        view = getattr(coordinator, "domain_view", None)
        if not isinstance(view, DomainStateView):
            return None

        return view.get_heater_state(
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

        return DeviceInfo(
            identifiers={(DOMAIN, self.boost_context.dev_id, self.boost_context.addr)},
            name=self.boost_context.base_name,
            manufacturer="TermoWeb",
            model="Accumulator",
            via_device=(DOMAIN, self.boost_context.dev_id),
        )

    async def async_press(self) -> None:
        """Invoke the helper service to update the accumulator boost state."""

        hass = self.hass
        if hass is None:
            return

        data: dict[str, Any] = {
            "entry_id": self.boost_context.entry_id,
            "dev_id": self.boost_context.dev_id,
            "node_type": self.boost_context.node_type,
            "addr": self.boost_context.addr,
        }
        minutes = self._service_minutes
        if minutes is not None:
            data["minutes"] = minutes

        try:
            await hass.services.async_call(
                DOMAIN,
                _SERVICE_REQUEST_ACCUMULATOR_BOOST,
                data,
                blocking=True,
            )
        except ServiceNotFound as err:
            _LOGGER.error(
                "Boost helper service unavailable for %s (%s): %s",
                self.boost_context.addr,
                self.boost_context.node_type,
                err,
            )
        except HomeAssistantError as err:  # pragma: no cover - defensive logging
            _LOGGER.error(
                "Boost helper service failed for %s (%s): %s",
                self.boost_context.addr,
                self.boost_context.node_type,
                err,
            )


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

        self._metadata = metadata
        icon = metadata.icon or None
        super().__init__(
            coordinator,
            context,
            label=None,
            unique_suffix=metadata.unique_suffix,
            icon=icon,
        )

    async def async_press(self) -> None:
        """Trigger a boost using the persisted duration and temperature."""

        hass = self.hass
        if hass is None:
            return

        minutes = resolve_boost_runtime_minutes(
            hass,
            self.boost_context.entry_id,
            self.boost_context.node_type,
            self.boost_context.addr,
        )
        if minutes is None or minutes <= 0:
            _LOGGER.error(
                "Boost start requires a stored duration for %s (%s)",
                self.boost_context.addr,
                self.boost_context.node_type,
            )
            return

        temperature = resolve_boost_temperature(
            hass,
            self.boost_context.entry_id,
            self.boost_context.node_type,
            self.boost_context.addr,
            default=DEFAULT_BOOST_TEMPERATURE,
        )
        if temperature is None:
            _LOGGER.error(
                "Boost start requires a stored temperature for %s (%s)",
                self.boost_context.addr,
                self.boost_context.node_type,
            )
            return

        data: dict[str, Any] = {
            "entry_id": self.boost_context.entry_id,
            "dev_id": self.boost_context.dev_id,
            "node_type": self.boost_context.node_type,
            "addr": self.boost_context.addr,
            "minutes": minutes,
            "temperature": round(float(temperature), 1),
        }

        try:
            await hass.services.async_call(
                DOMAIN,
                _SERVICE_REQUEST_ACCUMULATOR_BOOST,
                data,
                blocking=True,
            )
        except ServiceNotFound as err:
            _LOGGER.error(
                "Boost helper service unavailable for %s (%s): %s",
                self.boost_context.addr,
                self.boost_context.node_type,
                err,
            )
        except HomeAssistantError as err:  # pragma: no cover - defensive logging
            _LOGGER.error(
                "Boost helper service failed for %s (%s): %s",
                self.boost_context.addr,
                self.boost_context.node_type,
                err,
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

        icon = metadata.icon or None
        super().__init__(
            coordinator,
            context,
            label=None,
            unique_suffix=metadata.unique_suffix,
            icon=icon,
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
        if hass is None:
            return

        data: dict[str, Any] = {
            "entry_id": self.boost_context.entry_id,
            "dev_id": self.boost_context.dev_id,
            "node_type": self.boost_context.node_type,
            "addr": self.boost_context.addr,
        }

        try:
            await hass.services.async_call(
                DOMAIN,
                _SERVICE_CANCEL_ACCUMULATOR_BOOST,
                data,
                blocking=True,
            )
        except ServiceNotFound as err:
            _LOGGER.error(
                "Boost cancel service unavailable for %s (%s): %s",
                self.boost_context.addr,
                self.boost_context.node_type,
                err,
            )
        except HomeAssistantError as err:  # pragma: no cover - defensive logging
            _LOGGER.error(
                "Boost cancel service failed for %s (%s): %s",
                self.boost_context.addr,
                self.boost_context.node_type,
                err,
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
        """Return True when the target node is still in immutable inventory."""

        inventory = getattr(self.coordinator, "_inventory", None)
        return bool(
            isinstance(inventory, Inventory)
            and inventory.has_node(
                self._flash_context.node_type,
                self._flash_context.addr,
            )
        )

    @property
    def device_info(self) -> DeviceInfo:
        """Expose Home Assistant device metadata for the flash target."""

        model = "Thermostat" if self._flash_context.node_type == "thm" else "Heater"
        if self._flash_context.node_type == "acm":
            model = "Accumulator"

        return DeviceInfo(
            identifiers={
                (
                    DOMAIN,
                    self._flash_context.dev_id,
                    self._flash_context.addr,
                )
            },
            name=self._flash_context.name,
            manufacturer="TermoWeb",
            model=model,
            via_device=(DOMAIN, self._flash_context.dev_id),
        )

    async def async_press(self) -> None:
        """Call the backend /select endpoint to flash the unit display."""

        hass = self.hass
        if hass is None:
            return

        runtime = require_runtime(hass, self._flash_context.entry_id)
        _LOGGER.info(
            "Requesting display flash for %s/%s node %s",
            self._flash_context.dev_id,
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
                self._flash_context.dev_id,
                self._flash_context.node_type,
                self._flash_context.addr,
                err,
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
        return AccumulatorBoostButton(
            coordinator,
            context,
            metadata,
        )
    if metadata.action == "cancel":
        return AccumulatorBoostCancelButton(
            coordinator,
            context,
            metadata,
        )

    raise ValueError(f"Unsupported boost button action: {metadata.action}")
