"""Number platform entities for configuring TermoWeb boost presets."""

from __future__ import annotations

from collections.abc import Callable
import logging
import math
from typing import Any, TypeVar

from homeassistant.components.number import NumberEntity, NumberMode
from homeassistant.const import UnitOfPower, UnitOfTemperature, UnitOfTime
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers.entity import DeviceInfo, EntityCategory
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from ..boost import ALLOWED_BOOST_MINUTES, coerce_boost_minutes
from ..backend.factory import backend_capabilities
from ..domain.state import DomainState
from ..identifiers import build_gateway_entity_unique_id, build_heater_unique_id
from ..inventory import (
    Inventory,
    boostable_accumulator_details_for_entry,
    normalize_node_addr,
    normalize_node_type,
)
from ..runtime import require_runtime
from ..coerce import as_float
from ..utils import build_installation_device_info
from .heater import (
    DEFAULT_BOOST_DURATION,
    DEFAULT_BOOST_TEMPERATURE,
    HeaterNodeBase,
    NodeRefreshFallback,
    async_backend_write,
    heater_platform_details_for_entry,
    to_device_temperature,
)

_LOGGER = logging.getLogger(__name__)


_T = TypeVar("_T")


async def _restore_boost_value(
    entity: RestoreEntity,
    *,
    last_state_parser: Callable[[Any], _T | None],
    settings_lookup: Callable[[], _T],
    applier: Callable[[_T | None], None],
) -> None:
    """Restore the fallback boost value shown until the device reports one."""

    value: _T | None = None
    last_state = await entity.async_get_last_state()
    if last_state is not None:
        value = last_state_parser(last_state.state)
    if value is None:
        value = settings_lookup()
    applier(value)


async def _async_write_boost_preset(
    entity: HeaterNodeBase,
    *,
    boost_time: int | None = None,
    boost_temp: float | None = None,
) -> None:
    """Write accumulator boost defaults to the device and patch cached state."""

    runtime = require_runtime(entity.hass, entity._entry_id)  # noqa: SLF001
    await async_backend_write(
        "Boost preset write",
        runtime.backend.set_acm_extra_options(
            entity._dev_id,  # noqa: SLF001
            entity._addr,  # noqa: SLF001
            boost_time=boost_time,
            boost_temp=boost_temp,
        ),
    )

    def _mutate(state: DomainState) -> None:
        if boost_time is not None:
            state.boost_time = boost_time
        if boost_temp is not None:
            state.boost_temp = f"{boost_temp:.1f}"

    entity.coordinator.apply_entity_patch(
        entity._node_type,  # noqa: SLF001
        entity._addr,  # noqa: SLF001
        _mutate,
    )


async def async_setup_entry(hass, entry, async_add_entities):
    """Set up boost configuration number entities for accumulator nodes."""
    runtime = require_runtime(hass, entry.entry_id)
    coordinator = runtime.coordinator
    dev_id = runtime.dev_id

    def default_name(addr: str) -> str:
        """Return the fallback name for an accumulator node."""

        return f"Heater {addr}"

    heater_details, accumulator_nodes = boostable_accumulator_details_for_entry(
        runtime,
        default_name_simple=default_name,
        platform_name="number",
        logger=_LOGGER,
    )

    new_entities: list[NumberEntity] = []
    for node_type, addr_str, base_name in accumulator_nodes:
        unique_prefix = build_heater_unique_id(
            dev_id,
            node_type,
            addr_str,
            suffix="",
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

    heater_details = heater_platform_details_for_entry(
        runtime,
        default_name_simple=default_name,
    )
    priority_nodes = (
        heater_details.iter_metadata()
        if backend_capabilities(runtime.brand).priority
        else ()
    )
    for node_type, _node, addr_str, base_name in priority_nodes:
        canonical_type = normalize_node_type(node_type, use_default_when_falsey=True)
        canonical_addr = normalize_node_addr(addr_str, use_default_when_falsey=True)
        if not canonical_type or not canonical_addr:
            continue
        priority_unique_id = build_heater_unique_id(
            dev_id, canonical_type, canonical_addr, suffix=":priority"
        )
        new_entities.append(
            HeaterPriorityNumber(
                coordinator,
                entry.entry_id,
                dev_id,
                canonical_addr,
                priority_unique_id,
                device_name=base_name,
                node_type=canonical_type,
                inventory=heater_details.inventory,
            )
        )

    # Installation-wide power limit (TermoWeb only)
    if backend_capabilities(runtime.brand).power_limit:
        power_limit_uid = build_gateway_entity_unique_id(dev_id, "power_limit")
        new_entities.append(
            PowerLimitNumber(
                coordinator,
                entry.entry_id,
                dev_id,
                power_limit_uid,
            )
        )

    if new_entities:
        _LOGGER.debug("Adding %d TermoWeb number entities", len(new_entities))
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
        self._refresh_fallback = NodeRefreshFallback(self, self._node_type, self._addr)

    async def async_added_to_hass(self) -> None:
        """Restore the preferred duration once the entity is added."""

        await HeaterNodeBase.async_added_to_hass(self)
        await RestoreEntity.async_added_to_hass(self)
        self.async_on_remove(self._refresh_fallback.cancel)

        await _restore_boost_value(
            self,
            last_state_parser=self._hours_to_minutes,
            settings_lookup=self._initial_minutes_from_settings,
            applier=self._apply_minutes,
        )
        self.async_write_ha_state()

    @property
    def native_value(self) -> float:
        """Return the boost duration in hours for the UI slider."""

        return self._current_minutes() / 60

    async def async_set_native_value(self, value: float) -> None:
        """Handle slider updates from the user interface."""

        minutes = self._hours_to_minutes(value)
        if minutes is None or minutes not in ALLOWED_BOOST_MINUTES:
            raise ServiceValidationError(
                f"Invalid boost duration for {self._addr}: {value}"
            )

        # A failed device write raises here, so the value is only kept once
        # the device accepted it.
        await _async_write_boost_preset(self, boost_time=minutes)
        self._apply_minutes(minutes)
        self.async_write_ha_state()
        self._refresh_fallback.schedule()

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Expose the boost duration in minutes as an attribute."""

        return {"preferred_minutes": self._current_minutes()}

    def _current_minutes(self) -> int:
        """Return the device boost_time when known, else the stored preference."""

        state = self.accumulator_state()
        device = coerce_boost_minutes(getattr(state, "boost_time", None))
        return device if device in ALLOWED_BOOST_MINUTES else self._minutes

    def _initial_minutes_from_settings(self) -> int:
        """Return the bootstrap value sourced from cached settings."""

        state = self.accumulator_state()
        candidate = coerce_boost_minutes(
            getattr(state, "boost_time", None) if state is not None else None
        )
        if candidate in ALLOWED_BOOST_MINUTES:
            return candidate
        return DEFAULT_BOOST_DURATION

    def _apply_minutes(self, minutes: int | None) -> None:
        """Update the fallback minutes shown until the device reports boost_time."""

        self._minutes = self._validate_minutes(minutes)

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
        self._temperature = self._default_temperature()
        self._refresh_fallback = NodeRefreshFallback(self, self._node_type, self._addr)

    async def async_added_to_hass(self) -> None:
        """Restore the preferred temperature once the entity is added."""

        await HeaterNodeBase.async_added_to_hass(self)
        await RestoreEntity.async_added_to_hass(self)
        self.async_on_remove(self._refresh_fallback.cancel)

        await _restore_boost_value(
            self,
            last_state_parser=as_float,
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
    def native_min_value(self) -> float:
        """Return the lowest boost temperature in the device's units."""

        return self._setpoint_range()[0]

    @property
    def native_max_value(self) -> float:
        """Return the highest boost temperature in the device's units."""

        return self._setpoint_range()[1]

    @property
    def native_step(self) -> float:
        """Return the slider step: 0.5 degrees Celsius or 1 degree Fahrenheit."""

        return 1.0 if self._units() == "F" else 0.5

    def _default_temperature(self) -> float:
        """Return the default boost temperature in the device's units."""

        return to_device_temperature(DEFAULT_BOOST_TEMPERATURE, self._units())

    @property
    def native_value(self) -> float:
        """Return the preferred boost temperature."""

        return self._current_temperature()

    async def async_set_native_value(self, value: float) -> None:
        """Handle slider updates that adjust the boost temperature."""

        temperature = self._validate_temperature(value)
        if temperature is None:
            raise ServiceValidationError(
                f"Invalid boost temperature for {self._addr}: {value}"
            )

        # A failed device write raises here, so the value is only kept once
        # the device accepted it.
        await _async_write_boost_preset(self, boost_temp=temperature)
        self._apply_temperature(temperature)
        self.async_write_ha_state()
        self._refresh_fallback.schedule()

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Expose the preferred temperature as an attribute."""

        return {"preferred_temperature": self._current_temperature()}

    def _current_temperature(self) -> float:
        """Return the device boost_temp when valid, else the fallback value."""

        state = self.accumulator_state()
        device = self._validate_temperature(getattr(state, "boost_temp", None))
        return self._temperature if device is None else device

    def _initial_temperature_from_settings(self) -> float:
        """Return the bootstrap value sourced from cached settings."""

        state = self.accumulator_state()
        candidate = as_float(
            getattr(state, "boost_temp", None) if state is not None else None
        )
        if candidate is None:
            candidate = as_float(
                getattr(state, "stemp", None) if state is not None else None
            )
        if candidate is None:
            return self._default_temperature()
        return self._validate_temperature(candidate) or self._default_temperature()

    def _apply_temperature(self, value: float | None) -> None:
        """Update the fallback temperature shown until the device reports one."""

        temperature = self._validate_temperature(value)
        if temperature is None:
            temperature = self._default_temperature()
        self._temperature = temperature

    def _validate_temperature(self, value: Any) -> float | None:
        """Return a valid boost temperature within supported limits."""

        candidate = as_float(value)
        if candidate is None:
            return None
        if candidate < self.native_min_value or candidate > self.native_max_value:
            return None
        return math.floor(candidate * 10 + 0.5) / 10.0


class HeaterPriorityNumber(HeaterNodeBase, NumberEntity):
    """Number entity controlling the heater priority level."""

    _attr_entity_category = EntityCategory.CONFIG
    _attr_has_entity_name = True
    _attr_icon = "mdi:priority-high"
    _attr_mode = NumberMode.BOX
    _attr_native_min_value = 0
    _attr_native_max_value = 30
    _attr_native_step = 1
    _attr_translation_key = "heater_priority"

    def __init__(
        self,
        coordinator,
        entry_id: str,
        dev_id: str,
        addr: str,
        unique_id: str,
        *,
        device_name: str,
        node_type: str | None = None,
        inventory: Inventory | None = None,
    ) -> None:
        """Initialise the priority number entity."""

        super().__init__(
            coordinator,
            entry_id,
            dev_id,
            addr,
            None,
            unique_id,
            device_name=device_name,
            node_type=node_type,
            inventory=inventory,
        )
        self._refresh_fallback = NodeRefreshFallback(self, self._node_type, self._addr)

    async def async_added_to_hass(self) -> None:
        """Cancel a pending fallback refresh when the entity is removed."""

        await super().async_added_to_hass()
        self.async_on_remove(self._refresh_fallback.cancel)

    @property
    def native_value(self) -> int | None:
        """Return the current priority value from the domain state."""

        state = self.heater_state()
        return getattr(state, "priority", None) if state else None

    async def async_set_native_value(self, value: float) -> None:
        """Write the new priority value to the backend API."""

        priority = int(value)
        if priority < 0 or priority > 30:
            raise ServiceValidationError(f"Priority must be 0-30, got {priority}")
        runtime = require_runtime(self.hass, self._entry_id)
        await async_backend_write(
            "Priority write",
            runtime.backend.set_node_priority(
                self._dev_id,
                (self._node_type, self._addr),
                priority=priority,
            ),
        )

        def _mutate(state: DomainState) -> None:
            state.priority = priority

        # Show the value now; the WebSocket echo (or fallback refresh) confirms it.
        self.coordinator.apply_entity_patch(self._node_type, self._addr, _mutate)
        self._refresh_fallback.schedule()


class PowerLimitNumber(CoordinatorEntity, NumberEntity):
    """Number entity for the installation-wide power limit."""

    _attr_entity_category = EntityCategory.CONFIG
    _attr_has_entity_name = True
    _attr_icon = "mdi:flash-alert"
    _attr_mode = NumberMode.BOX
    _attr_native_min_value = 0
    _attr_native_max_value = 60000
    _attr_native_step = 100
    _attr_native_unit_of_measurement = UnitOfPower.WATT
    _attr_translation_key = "installation_power_limit"

    def __init__(
        self,
        coordinator,
        entry_id: str,
        dev_id: str,
        unique_id: str,
    ) -> None:
        """Initialise the installation power limit number entity."""
        super().__init__(coordinator)
        self._entry_id = entry_id
        self._dev_id = dev_id
        self._attr_unique_id = unique_id

    @property
    def device_info(self) -> DeviceInfo:
        """Return the installation device info."""
        return build_installation_device_info(self.hass, self._entry_id, self._dev_id)

    @property
    def available(self) -> bool:
        """Return True when the power limit value is known."""
        return self.native_value is not None

    @property
    def native_value(self) -> int | None:
        """Return the current power limit in watts from the domain state."""
        return self.coordinator.domain_view.get_power_limit()

    async def async_set_native_value(self, value: float) -> None:
        """Write the new power limit through the backend, then store it."""
        power_limit = int(value)
        runtime = require_runtime(self.hass, self._entry_id)
        await async_backend_write(
            "Power limit write",
            runtime.backend.set_power_limit(self._dev_id, power_limit=power_limit),
        )
        # Show the value now; a WebSocket push or the next poll confirms it.
        self.coordinator.apply_power_limit(power_limit)
