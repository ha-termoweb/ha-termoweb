# ruff: noqa: D100,BLE001,TID252

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
import logging
import typing
from typing import Any, cast

from homeassistant.components.climate import (
    ClimateEntity,
    ClimateEntityFeature,
    HVACAction,
    HVACMode,
)
from homeassistant.const import ATTR_TEMPERATURE
from homeassistant.core import ServiceCall
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.helpers import entity_platform
from homeassistant.util import dt as dt_util
import voluptuous as vol

from ..backend.base import BoostContext
from ..boost import (
    ALLOWED_BOOST_MINUTES,
    ALLOWED_BOOST_MINUTES_MESSAGE,
    coerce_boost_minutes,
    supports_boost,
    validate_boost_minutes,
)
from ..coerce import as_float
from ..domain import DomainState, HeaterState
from ..identifiers import build_heater_unique_id, thermostat_fallback_name
from ..inventory import HeaterNode, Inventory, normalize_node_addr, normalize_node_type
from ..runtime import require_runtime
from .heater import (
    HeaterNodeBase,
    HeaterPlatformDetails,
    NodeRefreshFallback,
    async_backend_write,
    async_cancel_acm_boost,
    async_start_acm_boost,
    derive_boost_state_from_domain,
    log_skipped_nodes,
    resolve_acm_boost_setpoint,
    resolve_boost_runtime_minutes,
)

_LOGGER = logging.getLogger(__name__)
_CANCELLED_ERROR = asyncio.CancelledError

# Shared boost-duration validator: 60-600 minutes in steps of 60.
BOOST_MINUTES_VALIDATOR = vol.All(vol.Coerce(int), vol.In(ALLOWED_BOOST_MINUTES))


def _is_cancelled_error(err: BaseException) -> bool:
    """Return ``True`` when ``err`` represents a cancellation."""

    if isinstance(err, _CANCELLED_ERROR):
        return True
    cancelled_type = asyncio.CancelledError
    if cancelled_type is not ValueError and isinstance(err, cancelled_type):
        return True
    return False


# Small debounce so multiple UI events coalesce
_WRITE_DEBOUNCE = 0.2


async def async_setup_entry(hass, entry, async_add_entities):
    """Discover heater nodes and create climate entities."""
    runtime = require_runtime(hass, entry.entry_id)
    coordinator = runtime.coordinator
    dev_id = runtime.dev_id

    inventory = runtime.inventory
    if not isinstance(inventory, Inventory):
        _LOGGER.error("TermoWeb climate setup missing inventory for device %s", dev_id)
        raise TypeError("TermoWeb inventory unavailable for climate platform")

    def default_name_simple(addr: str) -> str:
        """Return fallback name for heater nodes."""

        return f"Heater {addr}"

    new_entities: list[ClimateEntity] = []

    heater_details = HeaterPlatformDetails(
        inventory=inventory,
        default_name_simple=default_name_simple,
    )

    for node_type, node, addr_str, base_name in heater_details.iter_metadata():
        fallback_type = normalize_node_type(
            node_type,
            use_default_when_falsey=True,
        )
        canonical_type = normalize_node_type(
            getattr(node, "type", None),
            default=fallback_type,
            use_default_when_falsey=True,
        )
        fallback_addr = normalize_node_addr(
            addr_str,
            use_default_when_falsey=True,
        )
        addr = normalize_node_addr(
            getattr(node, "addr", None),
            default=fallback_addr,
            use_default_when_falsey=True,
        )
        if not canonical_type or not addr:
            continue
        if canonical_type == "thm":
            heater_fallback = default_name_simple(addr)
            if base_name == heater_fallback:
                base_name = thermostat_fallback_name(addr)
        unique_id = build_heater_unique_id(
            dev_id,
            canonical_type,
            addr,
            suffix=":climate",
        )
        entity_cls: type[HeaterClimateEntity]
        if canonical_type == "acm" or supports_boost(node):
            entity_cls = AccumulatorClimateEntity
        else:
            entity_cls = HeaterClimateEntity
        new_entities.append(
            entity_cls(
                coordinator,
                entry.entry_id,
                dev_id,
                addr,
                base_name,
                unique_id,
                node_type=canonical_type,
                inventory=heater_details.inventory,
            )
        )

    log_skipped_nodes("climate", heater_details, logger=_LOGGER)
    if new_entities:
        _LOGGER.debug("Adding %d TermoWeb heater entities", len(new_entities))
        async_add_entities(new_entities)

    # -------------------- Register entity services --------------------
    platform = entity_platform.async_get_current_platform()

    # Explicit callables ensure dispatch and let us add clear logs when invoked.
    async def _svc_set_schedule(entity: HeaterClimateEntity, call: ServiceCall) -> None:
        """Handle the set_schedule entity service."""
        prog = cast(list[int], call.data["prog"])
        _LOGGER.info(
            "entity-service termoweb.set_schedule -> %s prog_len=%s",
            getattr(entity, "entity_id", "<no-entity-id>"),
            len(prog) if isinstance(prog, list) else "<invalid>",
        )
        await entity.async_set_schedule(prog)

    async def _svc_set_preset_temperatures(
        entity: HeaterClimateEntity, call: ServiceCall
    ) -> None:
        """Handle the set_preset_temperatures entity service."""
        if "ptemp" in call.data:
            args = {"ptemp": call.data.get("ptemp")}
        else:
            args = {
                "cold": call.data.get("cold"),
                "night": call.data.get("night"),
                "day": call.data.get("day"),
            }
        _LOGGER.info(
            "entity-service termoweb.set_preset_temperatures -> %s",
            getattr(entity, "entity_id", "<no-entity-id>"),
        )
        await entity.async_set_preset_temperatures(**args)

    # termoweb.set_schedule
    platform.async_register_entity_service(
        "set_schedule",
        {
            vol.Required("prog"): vol.All(
                [vol.All(int, vol.In([0, 1, 2]))],
                vol.Length(min=168, max=168),
            )
        },
        _svc_set_schedule,
    )

    # termoweb.set_preset_temperatures
    preset_schema = {
        vol.Optional("ptemp"): vol.All([vol.Coerce(float)], vol.Length(min=3, max=3)),
        vol.Optional("cold"): vol.Coerce(float),
        vol.Optional("night"): vol.Coerce(float),
        vol.Optional("day"): vol.Coerce(float),
    }
    platform.async_register_entity_service(
        "set_preset_temperatures",
        preset_schema,
        _svc_set_preset_temperatures,
    )

    async def _svc_set_acm_preset(
        entity: HeaterClimateEntity, call: ServiceCall
    ) -> None:
        """Handle accumulator preset updates."""

        _require_accumulator(entity, "set_acm_preset")

        _LOGGER.info(
            "entity-service termoweb.set_acm_preset -> %s minutes=%s temperature=%s",
            getattr(entity, "entity_id", "<no-entity-id>"),
            call.data.get("minutes"),
            call.data.get("temperature"),
        )
        await entity.async_set_acm_preset(
            minutes=call.data.get("minutes"),
            temperature=call.data.get("temperature"),
        )

    async def _svc_start_boost(entity: HeaterClimateEntity, call: ServiceCall) -> None:
        """Handle accumulator boost start service."""

        _require_accumulator(entity, "start_boost")

        _LOGGER.info(
            "entity-service termoweb.start_boost -> %s minutes=%s",
            getattr(entity, "entity_id", "<no-entity-id>"),
            call.data.get("minutes"),
        )
        await entity.async_start_boost(minutes=call.data.get("minutes"))

    async def _svc_cancel_boost(entity: HeaterClimateEntity, call: ServiceCall) -> None:
        """Handle accumulator boost cancellation service."""

        _require_accumulator(entity, "cancel_boost")

        _LOGGER.info(
            "entity-service termoweb.cancel_boost -> %s",
            getattr(entity, "entity_id", "<no-entity-id>"),
        )
        await entity.async_cancel_boost()

    acm_preset_schema = {
        vol.Optional("minutes"): BOOST_MINUTES_VALIDATOR,
        vol.Optional("temperature"): vol.Coerce(float),
    }
    platform.async_register_entity_service(
        "set_acm_preset",
        acm_preset_schema,
        _svc_set_acm_preset,
    )

    start_boost_schema = {vol.Optional("minutes"): BOOST_MINUTES_VALIDATOR}
    platform.async_register_entity_service(
        "start_boost",
        start_boost_schema,
        _svc_start_boost,
    )

    platform.async_register_entity_service(
        "cancel_boost",
        {},
        _svc_cancel_boost,
    )


def _require_accumulator(entity: HeaterClimateEntity, service: str) -> None:
    """Raise ServiceValidationError when an acm-only service targets another node."""

    if not isinstance(entity, AccumulatorClimateEntity):
        raise ServiceValidationError(
            f"termoweb.{service} only applies to accumulator entities"
        )


class HeaterClimateEntity(HeaterNode, HeaterNodeBase, ClimateEntity):
    """HA climate entity representing a single TermoWeb heater."""

    _attr_supported_features = (
        ClimateEntityFeature.TARGET_TEMPERATURE
        | ClimateEntityFeature.PRESET_MODE
        | ClimateEntityFeature.TURN_ON
        | ClimateEntityFeature.TURN_OFF
    )
    _attr_hvac_modes = [HVACMode.OFF, HVACMode.HEAT, HVACMode.AUTO]
    # ``temporary_override`` (backend ``modified_auto``) is reported as the
    # current preset but cannot be selected: changing the target temperature
    # while in Auto starts it.
    _attr_preset_modes = ["none"]
    # Modes that turn_on may restore after a turn_off.
    _resume_modes: tuple[HVACMode, ...] = (HVACMode.HEAT, HVACMode.AUTO)

    def __init__(
        self,
        coordinator,
        entry_id: str,
        dev_id: str,
        addr: str,
        name: str,
        unique_id: str | None = None,
        *,
        node_type: str | None = None,
        inventory: Inventory | None = None,
    ) -> None:
        """Initialise the climate entity for a TermoWeb heater."""
        HeaterNode.__init__(self, name=name, addr=addr)

        default_type = (
            normalize_node_type(
                getattr(self, "type", None),
                default="htr",
                use_default_when_falsey=True,
            )
            or "htr"
        )
        resolved_type = (
            normalize_node_type(
                node_type,
                default=default_type,
                use_default_when_falsey=True,
            )
            or default_type
        )
        if resolved_type != getattr(self, "type", ""):
            self.type = resolved_type
        HeaterNodeBase.__init__(
            self,
            coordinator,
            entry_id,
            dev_id,
            addr,
            self.name,
            unique_id,
            node_type=resolved_type,
            inventory=inventory,
        )

        self._refresh_fallback = NodeRefreshFallback(self, self._node_type, self._addr)

        # pending write aggregation
        self._pending_mode: HVACMode | str | None = None
        self._pending_stemp: float | None = None
        self._write_task: asyncio.Task | None = None
        self._resume_mode: HVACMode | None = None

    async def async_will_remove_from_hass(self) -> None:
        """Clean up pending tasks when the entity is removed."""
        if self._write_task:
            self._write_task.cancel()
            self._write_task = None
        self._refresh_fallback.cancel()
        await super().async_will_remove_from_hass()

    @staticmethod
    def _slot_label(v: int) -> str | None:
        """Translate a program slot integer into a label."""
        return {0: "cold", 1: "night", 2: "day"}.get(v)

    def _current_prog_slot(self, state: HeaterState | DomainState | None) -> int | None:
        """Return the active program slot index for the heater."""

        prog = getattr(state, "prog", None)
        if not isinstance(prog, list) or len(prog) < 168:
            return None
        now = dt_util.now()
        idx = now.weekday() * 24 + now.hour
        try:
            return int(prog[idx])
        except asyncio.CancelledError:
            raise
        except Exception:
            return None

    def _shared_inventory(self) -> Inventory | None:
        """Return the shared immutable inventory for this coordinator."""

        coordinator = getattr(self, "coordinator", None)
        if coordinator is None:
            return None
        for attr in ("inventory", "_inventory"):
            candidate = getattr(coordinator, attr, None)
            if isinstance(candidate, Inventory):
                return candidate
        return None

    def _optimistic_update(self, mutator: Callable[[DomainState], None]) -> bool:
        """Apply ``mutator`` to cached state and write the entity state."""

        try:
            coordinator = getattr(self, "coordinator", None)
            apply_patch = getattr(coordinator, "apply_entity_patch", None)
            updated = False
            if callable(apply_patch):
                updated = bool(apply_patch(self._node_type, self._addr, mutator))
            if updated:
                self.async_write_ha_state()
            data_obj = getattr(self.coordinator, "data", None)
            if not isinstance(data_obj, dict):
                _LOGGER.debug(
                    "Optimistic update failed type=%s addr=%s: unexpected coordinator data %s",
                    self._node_type,
                    self._addr,
                    type(data_obj).__name__,
                )
                return False
        except BaseException as err:  # pragma: no cover - defensive
            if _is_cancelled_error(err):
                raise
            _LOGGER.debug(
                "Optimistic update failed type=%s addr=%s: %s",
                self._node_type,
                self._addr,
                err,
            )
            return False
        else:
            return updated

    async def _async_write_settings(
        self,
        *,
        log_context: str,
        mode: str | None = None,
        stemp: float | None = None,
        prog: list[int] | None = None,
        ptemp: list[float] | None = None,
    ) -> None:
        """Submit a settings update to the TermoWeb API."""

        async def _submit(client: Any) -> None:
            await self._async_submit_settings(
                client,
                mode=mode,
                stemp=stemp,
                prog=prog,
                ptemp=ptemp,
                units=self._units(),
            )

        await self._async_client_call(log_context=log_context, call=_submit)

    async def _async_client_call(
        self,
        *,
        log_context: str,
        call: Callable[[Any], Awaitable[Any]],
    ) -> None:
        """Call a backend helper, raising HomeAssistantError when it fails."""

        description = f"{log_context} for {self._node_type} {self._addr}"
        client = self._client()
        if client is None:
            raise HomeAssistantError(f"{description} failed: backend unavailable")
        await async_backend_write(description, call(client))

    async def _async_submit_settings(
        self,
        backend,
        *,
        mode: str | None,
        stemp: float | None,
        prog: list[int] | None,
        ptemp: list[float] | None,
        units: str,
    ) -> None:
        """Send settings for this heater to the backend."""

        await backend.set_node_settings(
            self._dev_id,
            (self._node_type, self._addr),
            mode=mode,
            stemp=stemp,
            prog=prog,
            ptemp=ptemp,
            units=units,
        )

    @property
    def hvac_mode(self) -> HVACMode | None:
        """Return the HA HVAC mode, or None when the device mode is unknown."""

        state = self.heater_state()
        mode = (getattr(state, "mode", None) or "").lower()
        if mode == "off":
            return HVACMode.OFF
        if mode in {"auto", "modified_auto"}:
            return HVACMode.AUTO
        if mode == "manual":
            return HVACMode.HEAT
        return None

    @property
    def preset_mode(self) -> str | None:
        """Return the preset: temporary_override while an Auto override is active."""

        state = self.heater_state()
        mode = (getattr(state, "mode", None) or "").lower()
        if not mode:
            return None
        if mode == "modified_auto":
            return "temporary_override"
        return "none"

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        """Select preset ``none``: end a temporary override by resuming Auto."""

        if self.preset_mode == "temporary_override":
            await self.async_set_hvac_mode(HVACMode.AUTO)

    async def async_turn_on(self) -> None:
        """Turn on, restoring the mode used before the last turn off (else Auto)."""

        if self.hvac_mode not in (None, HVACMode.OFF):
            return
        await self.async_set_hvac_mode(self._resume_mode or HVACMode.AUTO)

    async def async_turn_off(self) -> None:
        """Turn the heater off."""

        await self.async_set_hvac_mode(HVACMode.OFF)

    @property
    def hvac_action(self) -> HVACAction | None:
        """Return the HVAC action, or None when the device state is unknown."""

        state = self.heater_state()
        heater_state = (getattr(state, "state", None) or "").lower()
        if heater_state in ("off", "idle", "standby"):
            return HVACAction.IDLE if self.hvac_mode != HVACMode.OFF else HVACAction.OFF
        if heater_state in ("on", "heating"):
            return HVACAction.HEATING
        return None

    @property
    def current_temperature(self) -> float | None:
        """Return the measured ambient temperature."""

        state = self.heater_state()
        return as_float(getattr(state, "mtemp", None))

    @property
    def target_temperature(self) -> float | None:
        """Return the target temperature set on the heater."""

        state = self.heater_state()
        return as_float(getattr(state, "stemp", None))

    @property
    def temperature_unit(self) -> str:
        """Return the unit the device reports temperatures in."""
        return self._temperature_unit()

    @property
    def min_temp(self) -> float:
        """Return the minimum supported setpoint in the device's units."""
        return self._setpoint_range()[0]

    @property
    def max_temp(self) -> float:
        """Return the maximum supported setpoint in the device's units."""
        return self._setpoint_range()[1]

    @property
    def icon(self) -> str | None:
        """Return an icon reflecting the heater state."""
        if self.hvac_mode == HVACMode.OFF:
            return "mdi:radiator-off"
        if self.hvac_action == HVACAction.HEATING:
            return "mdi:radiator"
        if self.hvac_action == HVACAction.IDLE:
            return "mdi:radiator-disabled"
        return "mdi:radiator"

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return additional metadata about the heater."""
        state = self.heater_state()
        attrs: dict[str, Any] = {
            "dev_id": self._dev_id,
            "addr": self._addr,
            "units": getattr(state, "units", None),
            "max_power": getattr(state, "max_power", None),
            "ptemp": getattr(state, "ptemp", None),
            "prog": getattr(state, "prog", None),  # full weekly program (168 ints)
        }

        slot = self._current_prog_slot(state)
        if slot is not None:
            label = self._slot_label(slot)
            attrs["program_slot"] = label
            ptemp = getattr(state, "ptemp", None)
            try:
                if isinstance(ptemp, (list, tuple)) and 0 <= slot < len(ptemp):
                    attrs["program_setpoint"] = as_float(ptemp[slot])
            except asyncio.CancelledError:
                raise
            except Exception:
                pass

        return attrs

    # -------------------- Entity services: schedule & preset temps --------------------
    async def _commit_write(
        self,
        *,
        log_context: str,
        write_kwargs: Mapping[str, typing.Any],
        apply_fn: Callable[[DomainState], None],
        success_details: Mapping[str, typing.Any] | None = None,
    ) -> None:
        """Submit a heater write, update cached state, and schedule fallback."""
        await self._async_write_settings(
            log_context=log_context,
            **dict(write_kwargs),
        )

        detail_suffix = ""
        if success_details:
            parts = [f"{key}={value}" for key, value in success_details.items()]
            if parts:
                detail_suffix = f" ({', '.join(parts)})"

        _LOGGER.debug(
            "%s OK type=%s addr=%s%s",
            log_context,
            self._node_type,
            self._addr,
            detail_suffix,
        )

        self._optimistic_update(apply_fn)
        self._refresh_fallback.schedule()

    async def async_set_schedule(self, prog: list[int]) -> None:
        """Write the 7x24 tri-state program to the device."""
        # Validate defensively even though the schema should catch most issues
        if not isinstance(prog, list) or len(prog) != 168:
            raise ServiceValidationError("prog must be a list of 168 values")
        try:
            prog2 = [int(x) for x in prog]
        except (TypeError, ValueError) as err:
            raise ServiceValidationError(f"Invalid prog: {err}") from err
        if any(x not in (0, 1, 2) for x in prog2):
            raise ServiceValidationError("prog values must be 0, 1 or 2")

        def _apply(cur: DomainState) -> None:
            if hasattr(cur, "prog"):
                cur.prog = list(prog2)

        await self._commit_write(
            log_context="Schedule write",
            write_kwargs={"prog": prog2},
            apply_fn=_apply,
            success_details={"prog_len": len(prog2)},
        )

    async def async_set_preset_temperatures(self, **kwargs) -> None:
        """Write the cold/night/day presets; missing ones keep their current value."""
        if isinstance(kwargs.get("ptemp"), list):
            p = kwargs["ptemp"]
        else:
            p = [kwargs.get(key) for key in ("cold", "night", "day")]
            if all(value is None for value in p):
                raise ServiceValidationError(
                    "Give ptemp or at least one of cold, night and day"
                )
            if any(value is None for value in p):
                current = getattr(self.heater_state(), "ptemp", None)
                if not isinstance(current, (list, tuple)) or len(current) != 3:
                    raise ServiceValidationError(
                        "The current preset temperatures are unknown; "
                        "give all of cold, night and day"
                    )
                p = [
                    cur if new is None else new
                    for new, cur in zip(p, current, strict=True)
                ]

        if len(p) != 3:
            raise ServiceValidationError("ptemp must have 3 values")
        try:
            p2 = [float(x) for x in p]
        except (TypeError, ValueError) as err:
            raise ServiceValidationError(f"Invalid preset temperatures: {err}") from err

        def _apply(cur: DomainState) -> None:
            if hasattr(cur, "ptemp"):
                cur.ptemp = [f"{t:.1f}" if isinstance(t, float) else t for t in p2]

        await self._commit_write(
            log_context="Preset write",
            write_kwargs={"ptemp": p2},
            apply_fn=_apply,
            success_details={"ptemp": p2},
        )

    # -------------------- Existing write path (mode/setpoint) --------------------
    async def async_set_temperature(self, **kwargs: Any) -> None:
        """Set target temperature; server requires manual+stemp together (stemp string handled by API)."""
        raw = kwargs.get(ATTR_TEMPERATURE)
        try:
            t = float(raw)
        except (TypeError, ValueError) as err:
            raise ServiceValidationError(f"Invalid temperature: {raw!r}") from err

        low, high = self._setpoint_range()
        t = max(low, min(high, t))
        self._pending_stemp = t
        default_mode = self._default_mode_for_setpoint()
        if default_mode is not None:
            self._pending_mode = default_mode
        _LOGGER.info(
            "Queue write: addr=%s stemp=%.1f mode=%s (batching %.1fs)",
            self._addr,
            t,
            default_mode if default_mode is not None else "<unchanged>",
            _WRITE_DEBOUNCE,
        )
        await self._ensure_write_task()

    def _default_mode_for_setpoint(self) -> HVACMode | str | None:
        """Return the mode enforced when sending a bare setpoint."""

        backend_mode = str(getattr(self.heater_state(), "mode", "") or "").lower()
        if backend_mode in {"auto", "modified_auto"} or self.hvac_mode == HVACMode.AUTO:
            return "modified_auto"

        return HVACMode.HEAT

    def _requires_setpoint_with_mode(self, hvac_mode: HVACMode | str) -> bool:
        """Return whether the backend needs a target temperature for the mode."""

        return hvac_mode == HVACMode.HEAT

    def _allows_setpoint_in_mode(self, hvac_mode: HVACMode | str) -> bool:
        """Return whether a mode already supports standalone setpoint writes."""

        return hvac_mode in {HVACMode.HEAT, "modified_auto"}

    def _hvac_mode_to_backend(self, hvac_mode: HVACMode | str) -> str:
        """Translate an HA HVAC mode to the backend string representation."""

        mapping: dict[HVACMode | str, str] = {
            HVACMode.OFF: "off",
            HVACMode.AUTO: "auto",
            HVACMode.HEAT: "manual",
        }
        return mapping.get(hvac_mode, str(hvac_mode))

    async def async_set_hvac_mode(self, hvac_mode: HVACMode | str) -> None:
        """Post off/auto/manual."""
        if isinstance(hvac_mode, HVACMode):
            hvac_mode_value = hvac_mode.value
        else:
            hvac_mode_value = str(hvac_mode)
        hvac_mode_norm = hvac_mode_value.lower()

        if hvac_mode_norm == HVACMode.OFF:
            if self.hvac_mode in self._resume_modes:
                self._resume_mode = self.hvac_mode
            self._pending_mode = HVACMode.OFF
            _LOGGER.info(
                "Queue write: addr=%s mode=%s (batching %.1fs)",
                self._addr,
                HVACMode.OFF,
                _WRITE_DEBOUNCE,
            )
            await self._ensure_write_task()
            return

        if hvac_mode_norm == HVACMode.AUTO:
            self._pending_mode = HVACMode.AUTO
            _LOGGER.info(
                "Queue write: addr=%s mode=%s (batching %.1fs)",
                self._addr,
                HVACMode.AUTO,
                _WRITE_DEBOUNCE,
            )
            await self._ensure_write_task()
            return

        if hvac_mode_norm == HVACMode.HEAT:
            self._pending_mode = HVACMode.HEAT
            if self._pending_stemp is None:
                cur = self.target_temperature
                if cur is not None:
                    self._pending_stemp = float(cur)
            _LOGGER.info(
                "Queue write: addr=%s mode=%s stemp=%s (batching %.1fs)",
                self._addr,
                HVACMode.HEAT,
                self._pending_stemp,
                _WRITE_DEBOUNCE,
            )
            await self._ensure_write_task()
            return

        raise ServiceValidationError(f"Unsupported hvac_mode {hvac_mode}")

    async def _ensure_write_task(self) -> None:
        """Schedule a debounced write task if one is not running."""
        if self._write_task and not self._write_task.done():
            return
        self._write_task = self.hass.async_create_background_task(
            self._write_after_debounce(),
            f"termoweb-write-{self._dev_id}-{self._addr}",
        )

    async def _write_after_debounce(self) -> None:
        """Flush pending writes after the debounce, repeating while new ones queue."""
        while True:
            await asyncio.sleep(_WRITE_DEBOUNCE)
            await self._async_flush_pending_write()
            if self._pending_mode is None and self._pending_stemp is None:
                return

    async def _async_flush_pending_write(self) -> None:
        """Send the currently pending mode/setpoint as one backend write."""
        mode = self._pending_mode
        stemp = self._pending_stemp
        self._pending_mode = None
        self._pending_stemp = None

        # Normalize to backend rules using subclass hooks so accumulators can
        # avoid forcing an unsupported manual mode.
        if stemp is not None:
            if mode is None:
                default_mode = self._default_mode_for_setpoint()
                if default_mode is not None:
                    mode = default_mode
            elif not self._allows_setpoint_in_mode(mode):
                fallback_mode = self._default_mode_for_setpoint()
                if fallback_mode is not None:
                    mode = fallback_mode
        if (
            mode is not None
            and stemp is None
            and self._requires_setpoint_with_mode(mode)
        ):
            current = self.target_temperature
            if current is not None:
                stemp = float(current)

        if mode is None and stemp is None:
            return

        mode_api = None
        if mode is not None:
            mode_api = self._hvac_mode_to_backend(mode)
        _LOGGER.info(
            "POST %s settings addr=%s mode=%s stemp=%s",
            self._node_type,
            self._addr,
            mode_api,
            stemp,
        )

        # This runs in the debounced background task, after the service call
        # returned, so a failure can only be logged.
        try:
            await self._async_write_settings(
                log_context="Mode/setpoint write",
                mode=mode_api,
                stemp=stemp,
            )
        except HomeAssistantError as err:
            _LOGGER.error("%s", err)
            return

        register_pending = getattr(self.coordinator, "register_pending_setting", None)
        if callable(register_pending):
            try:
                register_pending(
                    self._node_type,
                    self._addr,
                    mode=mode_api,
                    stemp=as_float(stemp),
                )
            except Exception as err:  # pragma: no cover - defensive
                _LOGGER.debug(
                    "Failed to register pending settings type=%s addr=%s: %s",
                    self._node_type,
                    self._addr,
                    err,
                    exc_info=err,
                )

        def _apply(cur: DomainState) -> None:
            if mode_api is not None and hasattr(cur, "mode"):
                cur.mode = mode_api
            if stemp is not None and hasattr(cur, "stemp"):
                stemp_str: Any = stemp
                try:
                    stemp_float = float(stemp)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    pass
                else:
                    stemp_str = f"{stemp_float:.1f}"
                cur.stemp = stemp_str

        self._optimistic_update(_apply)
        _LOGGER.debug(
            "Optimistic mode/stemp applied type=%s addr=%s mode=%s stemp=%s",
            self._node_type,
            self._addr,
            mode_api,
            stemp,
        )

        # Expect WS echo; schedule refresh if it doesn't arrive soon.
        self._refresh_fallback.schedule()


class AccumulatorClimateEntity(HeaterClimateEntity):
    """HA climate entity for TermoWeb accumulator nodes."""

    _attr_hvac_modes: list[HVACMode] = [HVACMode.OFF, HVACMode.AUTO]
    _attr_preset_modes = ["none", "boost"]
    _resume_modes = (HVACMode.AUTO,)

    def __init__(
        self,
        coordinator,
        entry_id: str,
        dev_id: str,
        addr: str,
        name: str,
        unique_id: str | None = None,
        *,
        node_type: str | None = None,
        inventory: Inventory | None = None,
    ) -> None:
        """Initialise the accumulator climate entity."""

        super().__init__(
            coordinator,
            entry_id,
            dev_id,
            addr,
            name,
            unique_id,
            node_type=node_type,
            inventory=inventory,
        )
        self._boost_resume_mode: HVACMode | None = None

    def _default_mode_for_setpoint(self) -> HVACMode | str | None:
        """Accumulators keep their current mode when updating setpoints."""

        return None

    def _requires_setpoint_with_mode(self, hvac_mode: HVACMode | str) -> bool:
        """Boost does not rely on manual setpoint semantics."""

        return False

    def _allows_setpoint_in_mode(self, hvac_mode: HVACMode | str) -> bool:
        """Accumulators accept setpoints without forcing a manual mode."""

        return True

    def _preferred_boost_minutes(self) -> int:
        """Return the configured boost duration in minutes."""

        return resolve_boost_runtime_minutes(self.accumulator_state())

    @property
    def hvac_modes(self) -> list[HVACMode]:
        """Return Off/Auto, plus Heat while the device reports manual mode."""

        if self.hvac_mode == HVACMode.HEAT:
            return [*self._attr_hvac_modes, HVACMode.HEAT]
        return self._attr_hvac_modes

    @property
    def hvac_mode(self) -> HVACMode | None:
        """Return the current accumulator HVAC mode."""

        state = self.accumulator_state()
        mode = (getattr(state, "mode", None) or "").lower()
        if mode == "off":
            return HVACMode.OFF
        if mode == "auto":
            return HVACMode.AUTO
        if mode == "boost":
            return HVACMode.AUTO
        return super().hvac_mode

    async def async_set_hvac_mode(self, hvac_mode: HVACMode | str) -> None:
        """Handle accumulator HVAC modes, delegating boost to presets."""

        if isinstance(hvac_mode, HVACMode):
            value = hvac_mode.value.lower()
        else:
            value = str(hvac_mode).lower()
        if value == "boost":
            raise ServiceValidationError(
                "Boost is a preset mode for accumulators, not an HVAC mode"
            )
        if value == str(HVACMode.HEAT):
            raise ServiceValidationError(
                f"Unsupported hvac_mode {hvac_mode} for accumulator"
            )
        await super().async_set_hvac_mode(hvac_mode)

    @property
    def preset_mode(self) -> str | None:
        """Return the active preset mode, or None when the mode is unknown."""

        state = self.accumulator_state()
        mode = (getattr(state, "mode", None) or "").lower()
        if not mode:
            return None
        if mode == "boost":
            return "boost"
        return "none"

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        """Set the accumulator preset mode."""

        value = (preset_mode or "").lower()
        if value not in self._attr_preset_modes:
            raise ServiceValidationError(
                f"Unsupported preset_mode {preset_mode} for accumulator"
            )

        current_preset = self.preset_mode
        if value == current_preset:
            return

        if value == "boost":
            self._boost_resume_mode = self.hvac_mode
            await self.async_start_boost(minutes=self._preferred_boost_minutes())
            return

        resume_mode = self._boost_resume_mode or self.hvac_mode
        self._boost_resume_mode = None
        await self.async_cancel_boost()
        await super().async_set_hvac_mode(resume_mode)

    @property
    def extra_state_attributes(self) -> Mapping[str, typing.Any] | None:
        """Return accumulator attributes including boost and charge metadata."""

        base_attrs = super().extra_state_attributes
        attrs: dict[str, Any] = dict(base_attrs) if base_attrs is not None else {}
        state = self.accumulator_state()
        boost_state = derive_boost_state_from_domain(state, self.coordinator)

        attrs["boost_active"] = boost_state.active
        attrs["boost_minutes_remaining"] = boost_state.minutes_remaining
        attrs["boost_end"] = boost_state.end_iso
        attrs["boost_end_label"] = boost_state.end_label
        attrs["preferred_boost_minutes"] = self._preferred_boost_minutes()

        charging = getattr(state, "charging", None)
        if isinstance(charging, bool):
            attrs["charging"] = charging
        elif charging is not None:
            attrs["charging"] = bool(charging)

        for key in ("current_charge_per", "target_charge_per"):
            value = getattr(state, key, None) if state is not None else None
            if isinstance(value, (int, float)):
                attrs[key] = int(value)

        return attrs

    def _validate_boost_minutes(self, minutes: int | None) -> int | None:
        """Return a validated boost duration, or None when absent; raise if invalid."""

        if minutes is None:
            return None
        message = (
            f"Boost duration must be one of [{ALLOWED_BOOST_MINUTES_MESSAGE}] "
            f"minutes, got {minutes}"
        )
        value = coerce_boost_minutes(minutes)
        if value is None:
            raise ServiceValidationError(message)
        try:
            return validate_boost_minutes(value)
        except ValueError as err:
            raise ServiceValidationError(message) from err

    async def async_set_acm_preset(
        self,
        *,
        minutes: int | None = None,
        temperature: float | None = None,
    ) -> None:
        """Update the default boost duration and/or temperature."""

        if minutes is None and temperature is None:
            raise ServiceValidationError(
                "Accumulator preset update requires minutes and/or temperature"
            )

        validated_minutes = self._validate_boost_minutes(minutes)

        temp_value: float | None = None
        if temperature is not None:
            try:
                temp_value = float(temperature)
            except (TypeError, ValueError) as err:
                raise ServiceValidationError(
                    f"Invalid boost temperature: {temperature!r}"
                ) from err

        async def _call(client: Any) -> None:
            await client.set_acm_extra_options(
                self._dev_id,
                self._addr,
                boost_time=validated_minutes,
                boost_temp=temp_value,
            )

        await self._async_client_call(log_context="Boost preset write", call=_call)

        def _apply(cur: DomainState) -> None:
            if validated_minutes is not None and hasattr(cur, "boost_time"):
                cur.boost_time = validated_minutes
            if temp_value is not None and hasattr(cur, "boost_temp"):
                cur.boost_temp = f"{float(temp_value):.1f}"

        self._optimistic_update(_apply)
        detail_parts = []
        if validated_minutes is not None:
            detail_parts.append(f"minutes={validated_minutes}")
        if temp_value is not None:
            detail_parts.append(f"temperature={temp_value:.1f}")
        suffix = f" ({', '.join(detail_parts)})" if detail_parts else ""
        _LOGGER.debug(
            "Boost preset write OK type=%s addr=%s%s",
            self._node_type,
            self._addr,
            suffix,
        )
        self._refresh_fallback.schedule()

    async def async_start_boost(self, *, minutes: int | None = None) -> None:
        """Start an accumulator boost session."""

        validated_minutes = self._validate_boost_minutes(minutes)
        if validated_minutes is None:
            validated_minutes = self._preferred_boost_minutes()

        state = self.accumulator_state()
        if resolve_acm_boost_setpoint(state) is None:
            raise HomeAssistantError(
                f"Boost start needs a setpoint, but {self._node_type} "
                f"{self._addr} has not reported one"
            )

        async def _call(client: Any) -> None:
            await async_start_acm_boost(
                client,
                self._dev_id,
                self._addr,
                state,
                minutes=validated_minutes,
            )

        await self._async_client_call(log_context="Boost start", call=_call)

        def _apply(cur: DomainState) -> None:
            if hasattr(cur, "boost_active"):
                cur.boost_active = True
            if hasattr(cur, "boost_remaining"):
                cur.boost_remaining = validated_minutes
            if hasattr(cur, "mode"):
                cur.mode = "boost"

        self._optimistic_update(_apply)
        _LOGGER.debug(
            "Boost start OK type=%s addr=%s minutes=%s",
            self._node_type,
            self._addr,
            validated_minutes,
        )
        self._refresh_fallback.schedule()

    async def async_cancel_boost(self) -> None:
        """Cancel the active accumulator boost session."""

        async def _call(client: Any) -> None:
            await async_cancel_acm_boost(client, self._dev_id, self._addr)

        await self._async_client_call(log_context="Boost cancel", call=_call)

        def _apply(cur: DomainState) -> None:
            if hasattr(cur, "boost_active"):
                cur.boost_active = False
            if hasattr(cur, "boost_remaining"):
                cur.boost_remaining = None
            if hasattr(cur, "boost_end_day"):
                cur.boost_end_day = None
            if hasattr(cur, "boost_end_min"):
                cur.boost_end_min = None
            if getattr(cur, "mode", None) == "boost":
                cur.mode = "auto"

        self._optimistic_update(_apply)
        _LOGGER.debug(
            "Boost cancel OK type=%s addr=%s",
            self._node_type,
            self._addr,
        )
        self._refresh_fallback.schedule()

    async def _async_submit_settings(  # type: ignore[override]
        self,
        backend,
        *,
        mode: str | None,
        stemp: float | None,
        prog: list[int] | None,
        ptemp: list[float] | None,
        units: str,
    ) -> None:
        boost_context: BoostContext | None = None

        if self._node_type == "acm":
            boost_state = None
            try:
                boost_state = self.boost_state()
            except Exception as err:  # defensive
                _LOGGER.debug(
                    "Failed to derive boost state for cancel heuristic addr=%s: %s",
                    self._addr,
                    err,
                    exc_info=err,
                )
            state = self.accumulator_state()
            mode_value: str | None = None
            if state is not None:
                boost_flag = getattr(state, "boost_active", None)
                if not isinstance(boost_flag, bool):
                    boost_flag = None
                mode_value = getattr(state, "mode", None)
                if not isinstance(mode_value, str):
                    mode_value = None
            boost_context = BoostContext(
                active=boost_state.active if boost_state is not None else None,
                mode=mode_value,
            )

        await backend.set_node_settings(
            self._dev_id,
            (self._node_type, self._addr),
            mode=mode,
            stemp=stemp,
            prog=prog,
            ptemp=ptemp,
            units=units,
            boost_context=boost_context,
        )
