"""Switch entities for heater operating modes: runback and EASY (radio heaters)."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.switch import SwitchEntity
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from custom_components.termoweb.backend.sanitize import mask_identifier
from custom_components.termoweb.coordinator.state import StateCoordinator
from custom_components.termoweb.domain.state import DomainState
from custom_components.termoweb.entity import (
    NodeRefreshFallback,
    SettingsResolver,
    async_backend_write,
    build_settings_resolver,
)
from custom_components.termoweb.identifiers import build_heater_unique_id, node_key
from custom_components.termoweb.inventory import Inventory
from custom_components.termoweb.runtime import require_runtime
from custom_components.termoweb.utils import build_node_device_info

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass: Any, entry: Any, async_add_entities: Any) -> None:
    """Add runback and EASY switches for heaters whose backend supports them."""

    runtime = require_runtime(hass, entry.entry_id)
    if not runtime.backend.features.heater_toggles:
        return
    inventory = runtime.inventory
    entities: list[SwitchEntity] = [
        cls(
            runtime.coordinator,
            entry.entry_id,
            runtime.dev_id,
            metadata.addr,
            inventory=inventory,
            device_name=metadata.name,
        )
        for metadata in inventory.iter_nodes_metadata(node_types=("htr",))
        for cls in (RunbackSwitch, EasyModeSwitch)
    ]
    if entities:
        _LOGGER.debug("Adding %d TermoWeb heater mode switches", len(entities))
    async_add_entities(entities)


class HeaterModeSwitch(CoordinatorEntity[StateCoordinator], SwitchEntity):
    """A heater operating mode the heater reports in its status flags."""

    _attr_has_entity_name = True
    _attr_should_poll = False
    _field: str  # HeaterState field holding the reported flag
    _label: str  # for logs and errors
    _write: str  # backend method taking the field name as its keyword

    def __init__(
        self,
        coordinator: StateCoordinator,
        entry_id: str,
        dev_id: str,
        addr: str,
        *,
        inventory: Inventory,
        device_name: str,
    ) -> None:
        """Bind the switch to heater ``addr``."""

        super().__init__(coordinator)
        self._entry_id = entry_id
        self._dev_id = str(dev_id)
        self._addr = addr
        self._inventory = inventory
        self._device_name = device_name
        self._attr_unique_id = build_heater_unique_id(
            dev_id, "htr", node_key(inventory, "htr", addr), suffix=f":{self._field}"
        )
        self._settings_resolver: SettingsResolver = build_settings_resolver(
            coordinator, dev_id, "htr", addr
        )
        self._refresh_fallback = NodeRefreshFallback(self, "htr", addr)

    async def async_added_to_hass(self) -> None:
        """Cancel a pending fallback refresh when the entity is removed."""

        await super().async_added_to_hass()
        self.async_on_remove(self._refresh_fallback.cancel)

    @property
    def is_on(self) -> bool | None:
        """Return the mode flag the heater last reported (None before a full status)."""

        value = getattr(self._settings_resolver(), self._field, None)
        return None if value is None else bool(value)

    @property
    def device_info(self) -> DeviceInfo:
        """Attach the switch to its heater device."""

        return build_node_device_info(
            self.hass,
            self._entry_id,
            self._dev_id,
            node_key(self._inventory, "htr", self._addr),
            name=self._device_name,
            node_type="htr",
        )

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Switch the mode on."""

        await self._async_set(True)

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Switch the mode off."""

        await self._async_set(False)

    async def _async_set(self, on: bool) -> None:
        """Write the mode, then show it until the heater's next status confirms it."""

        _LOGGER.info(
            "Setting %s %s for %s heater %s",
            self._label,
            "on" if on else "off",
            mask_identifier(self._dev_id),
            self._addr,
        )
        backend = require_runtime(self.hass, self._entry_id).backend
        write = getattr(backend, self._write)
        await async_backend_write(
            self._label,
            write(self._dev_id, ("htr", self._addr), **{self._field: on}),
        )

        def _mutate(state: DomainState) -> None:
            setattr(state, self._field, on)

        self.coordinator.apply_entity_patch("htr", self._addr, _mutate)
        self._refresh_fallback.schedule()


class RunbackSwitch(HeaterModeSwitch):
    """Runback: the heater holds its anti-frost temperature; off restores the mode."""

    _attr_translation_key = "runback"
    _attr_icon = "mdi:snowflake-thermometer"
    _field = "runback"
    _label = "Runback"
    _write = "set_node_runback"


class EasyModeSwitch(HeaterModeSwitch):
    """EASY mode: forces manual heat, setpoint unchanged; off restores the mode."""

    _attr_translation_key = "easy_mode"
    _attr_icon = "mdi:gesture-tap-button"
    _field = "easy"
    _label = "EASY mode"
    _write = "set_node_easy"
