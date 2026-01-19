"""Number platform entities for configuring TermoWeb boost presets."""

from __future__ import annotations

from collections.abc import Callable
import logging
import math
from typing import Any, TypeVar

from homeassistant.components.number import NumberEntity, NumberMode
from homeassistant.const import UnitOfTemperature, UnitOfTime
from homeassistant.exceptions import HomeAssistantError, ServiceNotFound
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.restore_state import RestoreEntity

from ..boost import ALLOWED_BOOST_MINUTES, coerce_boost_minutes
from ..const import DOMAIN
from .heater import (
    DEFAULT_BOOST_DURATION,
    DEFAULT_BOOST_TEMPERATURE,
    HeaterNodeBase,
    get_boost_runtime_minutes,
    get_boost_temperature,
    resolve_climate_entity_id,
    set_boost_runtime_minutes,
    set_boost_temperature,
)
from ..i18n import async_get_fallback_translations, attach_fallbacks, format_fallback
from ..identifiers import build_heater_entity_unique_id
from ..inventory import Inventory, boostable_accumulator_details_for_entry
from ..runtime import require_runtime
from ..utils import float_or_none

_LOGGER = logging.getLogger(__name__)


_T = TypeVar("_T")


async def _restore_boost_value(
    entity: RestoreEntity,
    hass,
    *,
    cached_lookup: Callable[[], _T | None],
    last_state_parser: Callable[[Any], _T | None],
    settings_lookup: Callable[[], _T],
    applier: Callable[[_T | None, bool], None],
) -> None:
    """Restore a boost preference from cache, state, or settings."""

    value: _T | None = None
    if hass is not None:
        value = cached_lookup()
    if value is None:
        last_state = await entity.async_get_last_state()
        if last_state is not None:
            value = last_state_parser(last_state.state)
    if value is None:
        value = settings_lookup()
    applier(value, persist=hass is not None)


async def async_setup_entry(hass, entry, async_add_entities):
    """Set up boost configuration number entities for accumulator nodes."""
    runtime = require_runtime(hass, entry.entry_id)
    coordinator = runtime.coordinator
    dev_id = runtime.dev_id

    fallbacks = await async_get_fallback_translations(hass, runtime)
    attach_fallbacks(coordinator, fallbacks)

    def default_name(addr: str) -> str:
        """Return the fallback name for an accumulator node."""

        return format_fallback(
            fallbacks,
            "fallbacks.heater_name",
            "Heater {addr}",
            addr=addr,
        )

    heater_details, accumulator_nodes = boostable_accumulator_details_for_entry(
        runtime,
        default_name_simple=default_name,
        platform_name="number",
        logger=_LOGGER,
    )

    new_entities: list[NumberEntity] = []
    for node_type, addr_str, base_name in accumulator_nodes:
        unique_prefix = build_heater_entity_unique_id(
            dev_id,
            node_type,
            addr_str,
            "",
        )
        new_entities.extend(
            (
                AccumulatorBoostDurationNumber(
                    coordinator,
                    entry.entry_id,
                    dev_id,
                    addr_str,
                    base_name,
                    f"{unique_prefix}:boost_duration",
                    node_type=node_type,
                    inventory=heater_details.inventory,
                ),
                AccumulatorBoostTemperatureNumber(
                    coordinator,
                    entry.entry_id,
                    dev_id,
                    addr_str,
                    base_name,
                    f"{unique_prefix}:boost_temperature",
                    node_type=node_type,
                    inventory=heater_details.inventory,
                ),
            )
        )

    if new_entities:
        _LOGGER.debug("Adding %d TermoWeb boost numbers", len(new_entities))
        async_add_entities(new_entities)


class AccumulatorBoostDurationNumber(RestoreEntity, HeaterNodeBase, NumberEntity):
    """Number entity exposing preferred boost duration per accumulator."""

    _attr_entity_category = EntityCategory.CONFIG
    _attr_has_entity_name = True
    _attr_icon = "mdi:timer-cog-outline"
    _attr_mode = NumberMode.SLIDER
    _attr_native_min_value = 1
    _attr_native_max_value = 10
    _attr_native_step = 1
    _attr_native_unit_of_measurement = UnitOfTime.HOURS
    _attr_translation_key = "accumulator_boost_duration"

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
        inventory: Inventory | None = None,
    ) -> None:
        """Initialise the boost duration slider for an accumulator."""

        HeaterNodeBase.__init__(
            self,
            coordinator,
            entry_id,
            dev_id,
            addr,
            None,
            unique_id,
            device_name=base_name,
            node_type=node_type,
            inventory=inventory,
        )
        self._minutes = DEFAULT_BOOST_DURATION

    async def async_added_to_hass(self) -> None:
        """Restore the preferred duration once the entity is added."""

        await HeaterNodeBase.async_added_to_hass(self)
        await RestoreEntity.async_added_to_hass(self)

        hass = self.hass
        await _restore_boost_value(
            self,
            hass,
            cached_lookup=lambda: None
            if hass is None
            else get_boost_runtime_minutes(
                hass,
                self._entry_id,
                self._node_type,
                self._addr,
            ),
            last_state_parser=self._hours_to_minutes,
            settings_lookup=self._initial_minutes_from_settings,
            applier=self._apply_minutes,
        )
        self.async_write_ha_state()

    @property
    def native_value(self) -> float:
        """Return the preferred duration in hours for the UI slider."""

        return self._minutes / 60

    async def async_set_native_value(self, value: float) -> None:
        """Handle slider updates from the user interface."""

        minutes = self._hours_to_minutes(value)
        if minutes is None or minutes not in ALLOWED_BOOST_MINUTES:
            _LOGGER.error(
                "Invalid boost duration for %s: %s",
                self._addr,
                value,
            )
            return

        self._apply_minutes(minutes, persist=True)
        self.async_write_ha_state()

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Expose the preferred minutes as an attribute."""

        return {"preferred_minutes": self._minutes}

    def _initial_minutes_from_settings(self) -> int:
        """Return the bootstrap value sourced from cached settings."""

        state = self.accumulator_state()
        candidate = coerce_boost_minutes(
            getattr(state, "boost_time", None) if state is not None else None
        )
        if candidate in ALLOWED_BOOST_MINUTES:
            return candidate
        return DEFAULT_BOOST_DURATION

    def _apply_minutes(self, minutes: int | None, *, persist: bool) -> None:
        """Update the cached minutes and persist when requested."""

        resolved = self._validate_minutes(minutes)
        self._minutes = resolved
        if persist and self.hass is not None:
            set_boost_runtime_minutes(
                self.hass,
                self._entry_id,
                self._node_type,
                self._addr,
                resolved,
            )

    def _validate_minutes(self, minutes: int | None) -> int:
        """Return a supported minute value, falling back to the default."""

        candidate = coerce_boost_minutes(minutes)
        if candidate in ALLOWED_BOOST_MINUTES:
            return candidate
        return DEFAULT_BOOST_DURATION

    @staticmethod
    def _hours_to_minutes(value: Any) -> int | None:
        """Translate a slider value in hours into whole minutes."""

        if value is None:
            return None
        try:
            hours = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(hours):
            return None
        minutes = int(round(hours * 60))
        return minutes if minutes > 0 else None


class AccumulatorBoostTemperatureNumber(RestoreEntity, HeaterNodeBase, NumberEntity):
    """Number entity exposing preferred boost temperature per accumulator."""

    _attr_entity_category = EntityCategory.CONFIG
    _attr_has_entity_name = True
    _attr_icon = "mdi:thermometer"
    _attr_mode = NumberMode.SLIDER
    _attr_native_min_value = 5.0
    _attr_native_max_value = 30.0
    _attr_native_step = 0.5
    _attr_translation_key = "accumulator_boost_temperature"

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
        inventory: Inventory | None = None,
    ) -> None:
        """Initialise the boost temperature slider for an accumulator."""

        HeaterNodeBase.__init__(
            self,
            coordinator,
            entry_id,
            dev_id,
            addr,
            None,
            unique_id,
            device_name=base_name,
            node_type=node_type,
            inventory=inventory,
        )
        self._temperature = DEFAULT_BOOST_TEMPERATURE
        self._climate_entity_id: str | None = None

    async def async_added_to_hass(self) -> None:
        """Restore the preferred temperature once the entity is added."""

        await HeaterNodeBase.async_added_to_hass(self)
        await RestoreEntity.async_added_to_hass(self)

        hass = self.hass
        await _restore_boost_value(
            self,
            hass,
            cached_lookup=lambda: None
            if hass is None
            else get_boost_temperature(
                hass,
                self._entry_id,
                self._node_type,
                self._addr,
            ),
            last_state_parser=float_or_none,
            settings_lookup=self._initial_temperature_from_settings,
            applier=self._apply_temperature,
        )
        self.async_write_ha_state()

    @property
    def native_unit_of_measurement(self) -> str:
        """Return the configured temperature units for the heater."""

        units = self._units()
        if units == "F":
            return UnitOfTemperature.FAHRENHEIT
        return UnitOfTemperature.CELSIUS

    @property
    def native_value(self) -> float:
        """Return the preferred boost temperature."""

        return self._temperature

    async def async_set_native_value(self, value: float) -> None:
        """Handle slider updates that adjust the boost temperature."""

        temperature = self._validate_temperature(value)
        if temperature is None:
            _LOGGER.error(
                "Invalid boost temperature for %s: %s",
                self._addr,
                value,
            )
            return

        hass = self.hass
        if hass is None:
            return

        entity_id = self._climate_entity_id or resolve_climate_entity_id(
            hass,
            self._entry_id,
            self._node_type,
            self._addr,
        )
        if not entity_id:
            _LOGGER.error(
                "Cannot resolve climate entity for boost temperature on %s",
                self._addr,
            )
            return
        self._climate_entity_id = entity_id

        data = {"entity_id": entity_id, "temperature": temperature}
        try:
            await hass.services.async_call(
                DOMAIN,
                "set_acm_preset",
                data,
                blocking=True,
            )
        except ServiceNotFound as err:
            _LOGGER.error(
                "Boost preset service unavailable for %s: %s",
                entity_id,
                err,
            )
            return
        except HomeAssistantError as err:  # pragma: no cover - defensive logging
            _LOGGER.error(
                "Boost preset service failed for %s: %s",
                entity_id,
                err,
            )
            return

        self._apply_temperature(temperature, persist=True)
        self.async_write_ha_state()

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Expose the preferred temperature as an attribute."""

        return {"preferred_temperature": self._temperature}

    def _initial_temperature_from_settings(self) -> float:
        """Return the bootstrap value sourced from cached settings."""

        state = self.accumulator_state()
        candidate = float_or_none(
            getattr(state, "boost_temp", None) if state is not None else None
        )
        if candidate is None:
            candidate = float_or_none(
                getattr(state, "stemp", None) if state is not None else None
            )
        if candidate is None:
            return DEFAULT_BOOST_TEMPERATURE
        return self._validate_temperature(candidate) or DEFAULT_BOOST_TEMPERATURE

    def _apply_temperature(self, value: float | None, *, persist: bool) -> None:
        """Update the cached temperature and persist when requested."""

        temperature = self._validate_temperature(value)
        if temperature is None:
            temperature = DEFAULT_BOOST_TEMPERATURE
        self._temperature = temperature
        if persist and self.hass is not None:
            set_boost_temperature(
                self.hass,
                self._entry_id,
                self._node_type,
                self._addr,
                temperature,
            )

    def _validate_temperature(self, value: Any) -> float | None:
        """Return a valid boost temperature within supported limits."""

        candidate = float_or_none(value)
        if candidate is None:
            return None
        if not math.isfinite(candidate):
            return None
        if candidate < float(self._attr_native_min_value) or candidate > float(
            self._attr_native_max_value
        ):
            return None
        return math.floor(candidate * 10 + 0.5) / 10.0
