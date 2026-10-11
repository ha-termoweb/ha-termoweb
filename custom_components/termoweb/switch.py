"""Switch entities for heater settings: runback, EASY, open window, true radiant."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.switch import SwitchEntity
from homeassistant.helpers.entity import EntityCategory

from custom_components.termoweb.entity import HeaterSettingEntity
from custom_components.termoweb.runtime import require_runtime

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass: Any, entry: Any, async_add_entities: Any) -> None:
    """Add the heater switches the entry's backend supports."""

    runtime = require_runtime(hass, entry.entry_id)
    features = runtime.backend.features
    classes: list[type[HeaterSettingSwitch]] = []
    if features.heater_toggles:
        classes += [RunbackSwitch, EasyModeSwitch]
    if features.advanced_setup:
        classes += [OpenWindowSwitch, TrueRadiantSwitch]
    inventory = runtime.inventory
    entities = [
        cls(
            runtime.coordinator,
            entry.entry_id,
            runtime.dev_id,
            metadata.addr,
            inventory=inventory,
            device_name=metadata.name,
        )
        for metadata in inventory.iter_nodes_metadata(node_types=("htr",))
        for cls in classes
    ]
    if entities:
        _LOGGER.debug("Adding %d TermoWeb heater switches", len(entities))
    async_add_entities(entities)


class HeaterSettingSwitch(HeaterSettingEntity, SwitchEntity):
    """An on/off heater setting."""

    @property
    def is_on(self) -> bool | None:
        """Return the setting the heater last reported (None while unknown)."""

        value = self._setting()
        return None if value is None else bool(value)

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Switch the setting on."""

        await self._async_write_setting(True)

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Switch the setting off."""

        await self._async_write_setting(False)


class RunbackSwitch(HeaterSettingSwitch):
    """Runback: the heater holds its anti-frost temperature; off restores the mode."""

    _attr_translation_key = "runback"
    _attr_icon = "mdi:snowflake-thermometer"
    _field = "runback"
    _label = "Runback"
    _write = "set_node_runback"


class EasyModeSwitch(HeaterSettingSwitch):
    """EASY mode: forces manual heat, setpoint unchanged; off restores the mode."""

    _attr_translation_key = "easy_mode"
    _attr_icon = "mdi:gesture-tap-button"
    _field = "easy"
    _label = "EASY mode"
    _write = "set_node_easy"


class OpenWindowSwitch(HeaterSettingSwitch):
    """Open window detection: the heater stops heating when a window opens."""

    _attr_entity_category = EntityCategory.CONFIG
    _attr_translation_key = "open_window_detection"
    _attr_icon = "mdi:window-open-variant"
    _field = "window_detection"
    _gate = "advanced_setup"
    _label = "Open window detection"
    _write = "set_node_advanced"


class TrueRadiantSwitch(HeaterSettingSwitch):
    """True radiant: the heater's radiant-heat control (no effect on some panels)."""

    _attr_entity_category = EntityCategory.CONFIG
    _attr_translation_key = "true_radiant"
    _attr_icon = "mdi:radiator"
    _field = "true_radiant"
    _gate = "advanced_setup"
    _label = "True radiant"
    _write = "set_node_advanced"
