"""Button platform entities for TermoWeb gateways."""

from __future__ import annotations

import logging
from typing import Any, Mapping

from homeassistant.components.button import ButtonEntity

try:  # pragma: no cover - fallback for stripped Home Assistant stubs in tests
    from homeassistant.exceptions import HomeAssistantError, ServiceNotFound
except ImportError:  # pragma: no cover - executed in unit test stubs

    class HomeAssistantError(Exception):
        """Fallback Home Assistant error used in unit tests."""

    class ServiceNotFound(HomeAssistantError):
        """Fallback service lookup error used in unit tests."""


from homeassistant.helpers.entity import DeviceInfo, EntityCategory
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .heater import (
    BoostButtonMetadata,
    HeaterNodeBase,
    HeaterPlatformDetails,
    heater_platform_details_from_inventory,
    heater_platform_details_for_entry,
    iter_boost_button_metadata,
    iter_boostable_heater_nodes,
    log_skipped_nodes,
)
from .identifiers import build_heater_entity_unique_id
from .inventory import Inventory
from .utils import build_gateway_device_info

_LOGGER = logging.getLogger(__name__)

_SERVICE_REQUEST_ACCUMULATOR_BOOST = "request_accumulator_boost"


def _resolve_inventory(entry_data: Mapping[str, Any]) -> Inventory | None:
    """Return the Inventory associated with ``entry_data`` when available."""

    candidate = entry_data.get("inventory")
    if isinstance(candidate, Inventory):
        return candidate

    coordinator = entry_data.get("coordinator")
    candidate = getattr(coordinator, "inventory", None)
    if isinstance(candidate, Inventory):
        return candidate

    return None


async def async_setup_entry(hass, entry, async_add_entities):
    """Expose hub refresh and accumulator boost helper buttons."""

    data = hass.data[DOMAIN][entry.entry_id]
    coordinator = data["coordinator"]
    dev_id = data["dev_id"]

    inventory = _resolve_inventory(data)
    def default_name(addr: str) -> str:
        """Return a placeholder name for heater nodes."""

        return f"Heater {addr}"
    if inventory is not None:
        heater_details = heater_platform_details_from_inventory(
            inventory,
            default_name_simple=default_name,
        )
    else:
        heater_details = heater_platform_details_for_entry(
            data,
            default_name_simple=default_name,
        )
    _, _, resolve_name = heater_details
    metadata_source: Inventory | HeaterPlatformDetails = (
        inventory if inventory is not None else heater_details
    )

    entities: list[ButtonEntity] = [
        StateRefreshButton(coordinator, entry.entry_id, dev_id)
    ]

    boost_entities: list[ButtonEntity] = []
    for node_type, _node, addr_str, base_name in iter_boostable_heater_nodes(
        metadata_source,
        resolve_name,
        accumulators_only=True,
    ):

        boost_entities.extend(
            _create_boost_button_entities(
                coordinator,
                entry.entry_id,
                dev_id,
                addr_str,
                base_name,
                node_type,
            )
        )

    if boost_entities:
        entities.extend(boost_entities)
    log_skipped_nodes("button", metadata_source, logger=_LOGGER)

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


class AccumulatorBoostButtonBase(HeaterNodeBase, ButtonEntity):
    """Base entity for TermoWeb accumulator boost helper buttons."""

    _attr_entity_category = EntityCategory.CONFIG
    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator,
        entry_id: str,
        dev_id: str,
        addr: str,
        base_name: str,
        unique_id: str,
        *,
        label: str,
        node_type: str | None = None,
    ) -> None:
        """Initialise an accumulator boost helper button."""

        HeaterNodeBase.__init__(
            self,
            coordinator,
            entry_id,
            dev_id,
            addr,
            f"{base_name} {label}",
            unique_id,
            device_name=base_name,
            node_type=node_type,
        )
        self._label = label
        self._attr_name = label

    @property
    def _service_minutes(self) -> int | None:
        """Return the minutes payload passed to the helper service."""

        return None

    async def async_press(self) -> None:
        """Invoke the helper service to update the accumulator boost state."""

        hass = self.hass
        if hass is None:
            return

        data: dict[str, Any] = {
            "entry_id": self._entry_id,
            "dev_id": self._dev_id,
            "node_type": self._node_type,
            "addr": self._addr,
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
                self._addr,
                self._node_type,
                err,
            )
        except HomeAssistantError as err:  # pragma: no cover - defensive logging
            _LOGGER.error(
                "Boost helper service failed for %s (%s): %s",
                self._addr,
                self._node_type,
                err,
            )


class AccumulatorBoostButton(AccumulatorBoostButtonBase):
    """Button that starts an accumulator boost for a fixed duration."""

    _attr_icon = "mdi:timer-play"
    _attr_translation_key = "accumulator_boost_minutes"

    def __init__(
        self,
        coordinator,
        entry_id: str,
        dev_id: str,
        addr: str,
        base_name: str,
        unique_id: str,
        *,
        minutes: int,
        node_type: str | None = None,
        label: str | None = None,
        icon: str | None = None,
    ) -> None:
        """Initialise the boost helper button for a fixed duration."""

        self._minutes = minutes
        super().__init__(
            coordinator,
            entry_id,
            dev_id,
            addr,
            base_name,
            unique_id,
            label=label or f"Boost {minutes} minutes",
            node_type=node_type,
        )
        if icon is not None:
            self._attr_icon = icon

    @property
    def _service_minutes(self) -> int | None:
        """Return the hard-coded boost duration for the button."""

        return self._minutes

    @property
    def translation_placeholders(self) -> dict[str, str]:
        """Expose the configured boost duration for translations."""

        return {"minutes": str(self._minutes)}


class AccumulatorBoostCancelButton(AccumulatorBoostButtonBase):
    """Button that stops the active accumulator boost session."""

    _attr_icon = "mdi:timer-off"
    _attr_translation_key = "accumulator_boost_cancel"

    def __init__(
        self,
        coordinator,
        entry_id: str,
        dev_id: str,
        addr: str,
        base_name: str,
        unique_id: str,
        *,
        node_type: str | None = None,
        label: str | None = None,
        icon: str | None = None,
    ) -> None:
        """Initialise the helper button that cancels an active boost."""

        super().__init__(
            coordinator,
            entry_id,
            dev_id,
            addr,
            base_name,
            unique_id,
            label=label or "Cancel boost",
            node_type=node_type,
        )
        if icon is not None:
            self._attr_icon = icon


def _create_boost_button_entities(
    coordinator,
    entry_id: str,
    dev_id: str,
    addr: str,
    base_name: str,
    node_type: str,
) -> list[ButtonEntity]:
    """Return boost helper buttons described by shared metadata."""

    unique_prefix = build_heater_entity_unique_id(
        dev_id,
        node_type,
        addr,
        ":boost",
    )
    return [
        _build_boost_button(
            metadata,
            coordinator,
            entry_id,
            dev_id,
            addr,
            base_name,
            node_type,
            unique_prefix,
        )
        for metadata in iter_boost_button_metadata()
    ]


def _build_boost_button(
    metadata: BoostButtonMetadata,
    coordinator,
    entry_id: str,
    dev_id: str,
    addr: str,
    base_name: str,
    node_type: str,
    unique_prefix: str,
) -> ButtonEntity:
    """Instantiate a boost helper button for ``metadata``."""

    unique_id = f"{unique_prefix}_{metadata.unique_suffix}"
    if metadata.minutes is None:
        return AccumulatorBoostCancelButton(
            coordinator,
            entry_id,
            dev_id,
            addr,
            base_name,
            unique_id,
            node_type=node_type,
            label=metadata.label,
            icon=metadata.icon,
        )

    return AccumulatorBoostButton(
        coordinator,
        entry_id,
        dev_id,
        addr,
        base_name,
        unique_id,
        minutes=metadata.minutes,
        node_type=node_type,
        label=metadata.label,
        icon=metadata.icon,
    )
