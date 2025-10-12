"""Button platform entities for TermoWeb gateways."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import logging
from typing import Any

from homeassistant.components.button import ButtonEntity
from homeassistant.core import callback

try:  # pragma: no cover - fallback for stripped Home Assistant stubs in tests
    from homeassistant.exceptions import HomeAssistantError, ServiceNotFound
except ImportError:  # pragma: no cover - executed in unit test stubs

    class HomeAssistantError(Exception):
        """Fallback Home Assistant error used in unit tests."""

    class ServiceNotFound(HomeAssistantError):
        """Fallback service lookup error used in unit tests."""


from homeassistant.helpers.entity import DeviceInfo, EntityCategory
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .boost import iter_inventory_heater_metadata
from .const import DOMAIN
from .heater import (
    BOOST_BUTTON_METADATA,
    DEFAULT_BOOST_TEMPERATURE,
    BoostButtonMetadata,
    derive_boost_state,
    log_skipped_nodes,
    resolve_boost_runtime_minutes,
    resolve_boost_temperature,
)
from .identifiers import build_heater_entity_unique_id
from .inventory import AccumulatorNode, Inventory
from .utils import build_gateway_device_info

_LOGGER = logging.getLogger(__name__)

_SERVICE_REQUEST_ACCUMULATOR_BOOST = "request_accumulator_boost"
_SERVICE_CANCEL_ACCUMULATOR_BOOST = "cancel_accumulator_boost"


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
    inventory: Inventory | None,
) -> Iterator[AccumulatorBoostContext]:
    """Yield boost contexts for accumulator nodes in ``inventory``."""

    if not isinstance(inventory, Inventory):
        return

    for node_type, _addr, _name, node in iter_inventory_heater_metadata(inventory):
        if node_type != "acm" or not isinstance(node, AccumulatorNode):
            continue
        yield AccumulatorBoostContext.from_inventory(entry_id, inventory, node)


async def async_setup_entry(hass, entry, async_add_entities):
    """Expose hub refresh and accumulator boost helper buttons."""

    data = hass.data[DOMAIN][entry.entry_id]
    coordinator = data["coordinator"]
    dev_id = data["dev_id"]
    entities: list[ButtonEntity] = [
        StateRefreshButton(coordinator, entry.entry_id, dev_id)
    ]

    inventory = data.get("inventory")
    log_skipped_nodes("button", inventory, logger=_LOGGER)

    boost_entities: list[ButtonEntity] = []
    for context in _iter_accumulator_contexts(entry.entry_id, inventory):
        boost_entities.extend(_create_boost_button_entities(coordinator, context))

    if boost_entities:
        entities.extend(boost_entities)

    async_add_entities(entities)


class StateRefreshButton(CoordinatorEntity, ButtonEntity):
    """Button that requests an immediate coordinator refresh."""

    _attr_name = "Force refresh"
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
        label: str,
        unique_suffix: str,
        icon: str | None = None,
    ) -> None:
        """Initialise an accumulator boost helper button."""

        super().__init__(coordinator)
        self._context = context
        self._attr_name = label
        self._attr_unique_id = context.unique_id(unique_suffix)
        if icon is not None:
            self._attr_icon = icon

    @property
    def _service_minutes(self) -> int | None:
        """Return the minutes payload passed to the helper service."""

        return None

    @property
    def available(self) -> bool:
        """Return True when the inventory exposes this accumulator."""

        forward_map, _ = self._context.inventory.heater_address_map
        return self._context.addr in forward_map.get(self._context.node_type, ())

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

    def _coordinator_settings(self) -> Mapping[str, Any] | None:
        """Return cached coordinator settings for this accumulator."""

        coordinator = getattr(self, "coordinator", None)
        data = getattr(coordinator, "data", None)
        if not isinstance(data, Mapping):
            return None

        record = data.get(self._context.dev_id)
        if not isinstance(record, Mapping):
            return None

        settings_by_type = record.get("settings")
        if isinstance(settings_by_type, Mapping):
            type_settings = settings_by_type.get(self._context.node_type)
            if isinstance(type_settings, Mapping):
                settings = type_settings.get(self._context.addr)
                if isinstance(settings, Mapping):
                    return dict(settings)

        return None

    def _coordinator_boost_active(self) -> bool:
        """Return True when coordinator cache reports boost activity."""

        settings = self._coordinator_settings() or {}
        coordinator = getattr(self, "coordinator", None)
        state = derive_boost_state(settings, coordinator)
        return bool(state.active)

    @property
    def device_info(self) -> DeviceInfo:
        """Return Home Assistant device metadata for the accumulator."""

        return DeviceInfo(
            identifiers={(DOMAIN, self._context.dev_id, self._context.addr)},
            name=self._context.base_name,
            manufacturer="TermoWeb",
            model="Accumulator",
            via_device=(DOMAIN, self._context.dev_id),
        )

    async def async_press(self) -> None:
        """Invoke the helper service to update the accumulator boost state."""

        hass = self.hass
        if hass is None:
            return

        data: dict[str, Any] = {
            "entry_id": self._context.entry_id,
            "dev_id": self._context.dev_id,
            "node_type": self._context.node_type,
            "addr": self._context.addr,
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
                self._context.addr,
                self._context.node_type,
                err,
            )
        except HomeAssistantError as err:  # pragma: no cover - defensive logging
            _LOGGER.error(
                "Boost helper service failed for %s (%s): %s",
                self._context.addr,
                self._context.node_type,
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
        label = metadata.label or "Start boost"
        icon = metadata.icon or None
        super().__init__(
            coordinator,
            context,
            label=label,
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
            self._context.entry_id,
            self._context.node_type,
            self._context.addr,
        )
        if minutes is None or minutes <= 0:
            _LOGGER.error(
                "Boost start requires a stored duration for %s (%s)",
                self._context.addr,
                self._context.node_type,
            )
            return

        temperature = resolve_boost_temperature(
            hass,
            self._context.entry_id,
            self._context.node_type,
            self._context.addr,
            default=DEFAULT_BOOST_TEMPERATURE,
        )
        if temperature is None:
            _LOGGER.error(
                "Boost start requires a stored temperature for %s (%s)",
                self._context.addr,
                self._context.node_type,
            )
            return

        data: dict[str, Any] = {
            "entry_id": self._context.entry_id,
            "dev_id": self._context.dev_id,
            "node_type": self._context.node_type,
            "addr": self._context.addr,
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
                self._context.addr,
                self._context.node_type,
                err,
            )
        except HomeAssistantError as err:  # pragma: no cover - defensive logging
            _LOGGER.error(
                "Boost helper service failed for %s (%s): %s",
                self._context.addr,
                self._context.node_type,
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

        label = metadata.label or "Cancel boost"
        icon = metadata.icon or None
        super().__init__(
            coordinator,
            context,
            label=label,
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
            "entry_id": self._context.entry_id,
            "dev_id": self._context.dev_id,
            "node_type": self._context.node_type,
            "addr": self._context.addr,
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
                self._context.addr,
                self._context.node_type,
                err,
            )
        except HomeAssistantError as err:  # pragma: no cover - defensive logging
            _LOGGER.error(
                "Boost cancel service failed for %s (%s): %s",
                self._context.addr,
                self._context.node_type,
                err,
            )


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
