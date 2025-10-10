"""Select platform entities for configuring TermoWeb boost runtimes."""

from __future__ import annotations

import logging
from typing import Any, Mapping

from homeassistant.components.select import SelectEntity
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.restore_state import RestoreEntity

from .boost import coerce_boost_minutes
from .const import DOMAIN
from .heater import (
    BOOST_DURATION_OPTIONS,
    DEFAULT_BOOST_DURATION,
    HeaterNodeBase,
    HeaterPlatformDetails,
    get_boost_runtime_minutes,
    heater_platform_details_from_inventory,
    heater_platform_details_for_entry,
    iter_boostable_heater_nodes,
    log_skipped_nodes,
    resolve_boost_runtime_minutes,
    set_boost_runtime_minutes,
)
from .identifiers import build_heater_entity_unique_id
from .inventory import Inventory

_LOGGER = logging.getLogger(__name__)


def _resolve_inventory(entry_data: Mapping[str, Any]) -> Inventory | None:
    """Return the Inventory attached to ``entry_data`` when available."""

    candidate = entry_data.get("inventory")
    if isinstance(candidate, Inventory):
        return candidate

    coordinator = entry_data.get("coordinator")
    candidate = getattr(coordinator, "inventory", None)
    if isinstance(candidate, Inventory):
        return candidate

    return None


async def async_setup_entry(hass, entry, async_add_entities):
    """Set up boost duration selectors for accumulator nodes."""

    data = hass.data[DOMAIN][entry.entry_id]
    coordinator = data["coordinator"]
    dev_id = data["dev_id"]
    inventory = _resolve_inventory(data)
    default_name = lambda addr: f"Heater {addr}"
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

    new_entities: list[AccumulatorBoostDurationSelect] = []
    for node_type, _node, addr_str, base_name in iter_boostable_heater_nodes(
        metadata_source,
        resolve_name,
        accumulators_only=True,
    ):
        unique_id = build_heater_entity_unique_id(
            dev_id,
            node_type,
            addr_str,
            ":boost_duration",
        )
        new_entities.append(
            AccumulatorBoostDurationSelect(
                coordinator,
                entry.entry_id,
                dev_id,
                addr_str,
                base_name,
                unique_id,
                node_type=node_type,
            )
        )

    log_skipped_nodes("select", metadata_source, logger=_LOGGER)

    if new_entities:
        _LOGGER.debug("Adding %d TermoWeb boost selectors", len(new_entities))
        async_add_entities(new_entities)


class AccumulatorBoostDurationSelect(RestoreEntity, HeaterNodeBase, SelectEntity):
    """Select entity exposing preferred boost duration per accumulator."""

    _attr_entity_category = EntityCategory.CONFIG
    _attr_has_entity_name = True
    _attr_icon = "mdi:timer-cog-outline"
    _attr_translation_key = "accumulator_boost_duration"

    _OPTION_MAP = {str(option): option for option in BOOST_DURATION_OPTIONS}
    _REVERSE_OPTION_MAP = {value: key for key, value in _OPTION_MAP.items()}

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
    ) -> None:
        """Initialise the boost duration selector for an accumulator."""

        HeaterNodeBase.__init__(
            self,
            coordinator,
            entry_id,
            dev_id,
            addr,
            f"{base_name} Boost duration",
            unique_id,
            device_name=base_name,
            node_type=node_type,
        )
        self._attr_name = "Boost duration"
        self._attr_options = list(self._OPTION_MAP.keys())
        self._attr_current_option: str | None = None

    async def async_added_to_hass(self) -> None:
        """Restore the preferred duration once the entity is added."""

        await HeaterNodeBase.async_added_to_hass(self)
        await RestoreEntity.async_added_to_hass(self)

        hass = self.hass
        minutes: int | None = None
        if hass is not None:
            minutes = get_boost_runtime_minutes(
                hass,
                self._entry_id,
                self._node_type,
                self._addr,
            )

        if minutes is None:
            last_state = await self.async_get_last_state()
            if last_state is not None:
                minutes = self._option_to_minutes(last_state.state)

        if minutes is None:
            minutes = self._initial_minutes_from_settings()

        self._apply_minutes(minutes, persist=hass is not None)
        self.async_write_ha_state()

    async def async_select_option(self, option: str) -> None:
        """Handle a new boost duration selection from the user."""

        minutes = self._option_to_minutes(option)
        if minutes is None:
            _LOGGER.error(
                "Invalid boost duration option for %s: %s",
                self._addr,
                option,
            )
            return

        self._apply_minutes(minutes, persist=True)
        self.async_write_ha_state()

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Expose the preferred minutes as an attribute."""

        return {"preferred_minutes": self._current_minutes()}

    def _apply_minutes(self, minutes: int | None, *, persist: bool) -> None:
        """Update the active option and persist when requested."""

        resolved = self._validate_minutes(minutes)
        option = self._REVERSE_OPTION_MAP.get(resolved)
        if option is None:
            option = str(DEFAULT_BOOST_DURATION)
            resolved = DEFAULT_BOOST_DURATION
        self._attr_current_option = option
        if persist and self.hass is not None:
            set_boost_runtime_minutes(
                self.hass,
                self._entry_id,
                self._node_type,
                self._addr,
                resolved,
            )

    def _initial_minutes_from_settings(self) -> int:
        """Return the bootstrap value sourced from cached settings."""

        settings = self.heater_settings() or {}
        candidate = self._option_to_minutes(settings.get("boost_time"))
        if candidate is not None:
            return candidate
        return DEFAULT_BOOST_DURATION

    def _option_to_minutes(self, value: Any) -> int | None:
        """Translate ``value`` into a supported minute option."""

        candidate: int | None
        if isinstance(value, str):
            text = value.strip()
            if text in self._OPTION_MAP:
                return self._OPTION_MAP[text]
            candidate = coerce_boost_minutes(text)
        else:
            candidate = coerce_boost_minutes(value)

        if candidate is None:
            return None
        if candidate in self._REVERSE_OPTION_MAP:
            return candidate
        return None

    def _validate_minutes(self, minutes: int | None) -> int:
        """Return a supported minute value, falling back to the default."""

        resolved = self._option_to_minutes(minutes)
        if resolved is not None:
            return resolved
        return DEFAULT_BOOST_DURATION

    def _current_minutes(self) -> int:
        """Return the currently selected duration as minutes."""

        if self._attr_current_option in self._OPTION_MAP:
            return self._OPTION_MAP[self._attr_current_option]
        hass = self.hass
        if hass is None:
            return DEFAULT_BOOST_DURATION
        return resolve_boost_runtime_minutes(
            hass,
            self._entry_id,
            self._node_type,
            self._addr,
        )
