"""Select platform entities for configuring TermoWeb boost runtimes."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.select import SelectEntity
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.restore_state import RestoreEntity

from .const import DOMAIN
from .heater import (
    BOOST_DURATION_OPTIONS,
    DEFAULT_BOOST_DURATION,
    HeaterNodeBase,
    get_boost_runtime_minutes,
    iter_heater_nodes,
    log_skipped_nodes,
    prepare_heater_platform_data,
    resolve_boost_runtime_minutes,
    set_boost_runtime_minutes,
)

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass, entry, async_add_entities):
    """Set up boost duration selectors for accumulator nodes."""

    data = hass.data[DOMAIN][entry.entry_id]
    coordinator = data["coordinator"]
    dev_id = data["dev_id"]
    _, nodes_by_type, _, resolve_name = prepare_heater_platform_data(
        data,
        default_name_simple=lambda addr: f"Heater {addr}",
    )

    new_entities: list[AccumulatorBoostDurationSelect] = []
    for node_type, node, addr_str, base_name in iter_heater_nodes(
        nodes_by_type,
        resolve_name,
    ):
        if node_type != "acm":
            continue
        supports_boost = getattr(node, "supports_boost", None)
        if callable(supports_boost) and not supports_boost():
            continue
        unique_id = f"{DOMAIN}:{dev_id}:{node_type}:{addr_str}:boost_duration"
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

    log_skipped_nodes("select", nodes_by_type, logger=_LOGGER)

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

        if isinstance(value, str):
            text = value.strip()
            if text in self._OPTION_MAP:
                return self._OPTION_MAP[text]
            try:
                numeric = int(float(text))
            except (TypeError, ValueError):  # pragma: no cover - defensive
                return None
            if numeric in self._REVERSE_OPTION_MAP:
                return numeric
            return None
        if isinstance(value, (int, float)):
            numeric = int(value)
            if numeric in self._REVERSE_OPTION_MAP:
                return numeric
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
