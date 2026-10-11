"""Select entities for heater settings: control mode and panel temperature unit."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.select import SelectEntity
from homeassistant.helpers.entity import EntityCategory

from custom_components.termoweb.entity import HeaterSettingEntity
from custom_components.termoweb.runtime import require_runtime

_LOGGER = logging.getLogger(__name__)

CONTROL_MODES = [
    "pid",
    "hysteresis_0_25",
    "hysteresis_0_35",
    "hysteresis_0_5",
    "hysteresis_0_75",
]
PANEL_UNITS = {"celsius": "C", "fahrenheit": "F"}


async def async_setup_entry(hass: Any, entry: Any, async_add_entities: Any) -> None:
    """Add control mode and panel unit selects for heaters with advanced setup."""

    runtime = require_runtime(hass, entry.entry_id)
    if not runtime.backend.features.advanced_setup:
        return
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
        for cls in (ControlModeSelect, PanelUnitsSelect)
    ]
    if entities:
        _LOGGER.debug("Adding %d TermoWeb heater selects", len(entities))
    async_add_entities(entities)


class ControlModeSelect(HeaterSettingEntity, SelectEntity):
    """How the heater regulates: PID, or on/off with a hysteresis band."""

    _attr_entity_category = EntityCategory.CONFIG
    _attr_translation_key = "control_mode"
    _attr_icon = "mdi:tune-variant"
    _attr_options = CONTROL_MODES
    _field = "control_mode"
    _gate = "advanced_setup"
    _label = "Control mode"
    _write = "set_node_advanced"

    @property
    def current_option(self) -> str | None:
        """Return the control mode last written (None until written)."""

        value = self._setting()
        return value if value in CONTROL_MODES else None

    async def async_select_option(self, option: str) -> None:
        """Write the control mode."""

        await self._async_write_setting(option)


class PanelUnitsSelect(HeaterSettingEntity, SelectEntity):
    """The temperature unit on the heater's own panel (Home Assistant is not affected)."""

    _attr_entity_category = EntityCategory.CONFIG
    _attr_translation_key = "panel_units"
    _attr_icon = "mdi:temperature-celsius"
    _attr_options = list(PANEL_UNITS)
    _field = "panel_units"
    _gate = "advanced_setup"
    _label = "Panel temperature unit"
    _write = "set_node_advanced"

    @property
    def current_option(self) -> str | None:
        """Return the panel unit last written (None until written)."""

        value = self._setting()
        for option, units in PANEL_UNITS.items():
            if value == units:
                return option
        return None

    async def async_select_option(self, option: str) -> None:
        """Write the panel unit (``C`` or ``F``)."""

        await self._async_write_setting(PANEL_UNITS[option])
