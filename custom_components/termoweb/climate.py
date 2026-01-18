# ruff: noqa: D100,BLE001,TRY301

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
import inspect
import logging
import time
from typing import Any, cast

from homeassistant.components.climate import (
    ClimateEntity,
    ClimateEntityFeature,
    HVACAction,
    HVACMode,
)
from homeassistant.const import ATTR_TEMPERATURE, UnitOfTemperature
from homeassistant.core import ServiceCall, callback
from homeassistant.helpers import entity_platform
from homeassistant.util import dt as dt_util
import voluptuous as vol

from .backend.base import BoostContext
from .boost import (
    ALLOWED_BOOST_MINUTES_MESSAGE,
    coerce_boost_minutes,
    supports_boost,
    validate_boost_minutes,
)
from .domain import DomainState, HeaterState
from .heater import (
    DEFAULT_BOOST_DURATION,
    HeaterNodeBase,
    HeaterPlatformDetails,
    clear_climate_entity_id,
    derive_boost_state_from_domain,
    log_skipped_nodes,
    register_climate_entity_id,
    resolve_boost_runtime_minutes,
)
from .i18n import async_get_fallback_translations, attach_fallbacks, format_fallback
from .identifiers import build_heater_entity_unique_id, thermostat_fallback_name
from .inventory import HeaterNode, Inventory, normalize_node_addr, normalize_node_type
from .runtime import require_runtime
from .utils import float_or_none

_LOGGER = logging.getLogger(__name__)
_CANCELLED_ERROR = asyncio.CancelledError


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
# If WS echo doesn't arrive quickly after a successful write, force a refresh
_WS_ECHO_FALLBACK_REFRESH = 4.0


async def async_setup_entry(hass, entry, async_add_entities):
    """Discover heater nodes and create climate entities."""
    runtime = require_runtime(hass, entry.entry_id)
    coordinator = runtime.coordinator
    dev_id = runtime.dev_id

    fallbacks = await async_get_fallback_translations(hass, runtime)
    attach_fallbacks(coordinator, fallbacks)

    inventory = runtime.inventory
    if not isinstance(inventory, Inventory):
        _LOGGER.error("TermoWeb climate setup missing inventory for device %s", dev_id)
        raise TypeError("TermoWeb inventory unavailable for climate platform")

    def default_name_simple(addr: str) -> str:
        """Return fallback name for heater nodes."""

        return format_fallback(
            fallbacks,
            "fallbacks.heater_name",
            "Heater {addr}",
            addr=addr,
        )

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
            thermostat_default = format_fallback(
                fallbacks,
                "fallbacks.thermostat_name",
                thermostat_fallback_name(addr),
                addr=addr,
            )
            if base_name == heater_fallback:
                base_name = thermostat_default
        unique_id = build_heater_entity_unique_id(
            dev_id,
            canonical_type,
            addr,
            ":climate",
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

        if not isinstance(entity, AccumulatorClimateEntity):
            _LOGGER.error(
                "termoweb.set_acm_preset only applies to accumulator entities"
            )
            return

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

        if not isinstance(entity, AccumulatorClimateEntity):
            _LOGGER.error("termoweb.start_boost only applies to accumulator entities")
            return

        _LOGGER.info(
            "entity-service termoweb.start_boost -> %s minutes=%s",
            getattr(entity, "entity_id", "<no-entity-id>"),
            call.data.get("minutes"),
        )
        await entity.async_start_boost(minutes=call.data.get("minutes"))

    async def _svc_cancel_boost(entity: HeaterClimateEntity, call: ServiceCall) -> None:
        """Handle accumulator boost cancellation service."""

        if not isinstance(entity, AccumulatorClimateEntity):
            _LOGGER.error("termoweb.cancel_boost only applies to accumulator entities")
            return

        _LOGGER.info(
            "entity-service termoweb.cancel_boost -> %s",
            getattr(entity, "entity_id", "<no-entity-id>"),
        )
        await entity.async_cancel_boost()

    acm_preset_schema = {
        vol.Optional("minutes"): vol.All(vol.Coerce(int), vol.Range(min=1, max=120)),
        vol.Optional("temperature"): vol.Coerce(float),
    }
    platform.async_register_entity_service(
        "set_acm_preset",
        acm_preset_schema,
        _svc_set_acm_preset,
    )

    start_boost_schema = {
        vol.Optional("minutes"): vol.All(vol.Coerce(int), vol.Range(min=1, max=120))
    }
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


class HeaterClimateEntity(HeaterNode, HeaterNodeBase, ClimateEntity):
    """HA climate entity representing a single TermoWeb heater."""

    _attr_supported_features = ClimateEntityFeature.TARGET_TEMPERATURE
    _attr_hvac_modes = [HVACMode.OFF, HVACMode.HEAT, HVACMode.AUTO]
    _attr_temperature_unit = UnitOfTemperature.CELSIUS

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

        self._refresh_fallback: asyncio.Task | None = None

        # pending write aggregation
        self._pending_mode: HVACMode | str | None = None
        self._pending_stemp: float | None = None
        self._write_task: asyncio.Task | None = None

    async def async_added_to_hass(self) -> None:
        """Register the entity ID for cross-platform helpers."""

        await super().async_added_to_hass()
        hass = self.hass
        if hass is not None:
            register_climate_entity_id(
                hass,
                self._entry_id,
                self._node_type,
                self._addr,
                getattr(self, "entity_id", None),
            )

    async def async_will_remove_from_hass(self) -> None:
        """Clean up pending tasks when the entity is removed."""
        if self._refresh_fallback:
            self._refresh_fallback.cancel()
            self._refresh_fallback = None
        hass = self.hass
        if hass is not None:
            clear_climate_entity_id(
                hass,
                self._entry_id,
                self._node_type,
                self._addr,
            )
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
        """Apply ``mutator`` to cached state and refresh state if changed."""

        try:
            coordinator = getattr(self, "coordinator", None)
            apply_patch = getattr(coordinator, "apply_entity_patch", None)
            updated = False
            refresh_needed = False
            if callable(apply_patch):
                applied = bool(apply_patch(self._node_type, self._addr, mutator))
                if applied:
                    updated = True
                    refresh_needed = True
            if updated:
                self.async_write_ha_state()
                hass = self.hass
                refresh = getattr(self.coordinator, "async_request_refresh", None)
                if refresh_needed and hass is not None and callable(refresh):
                    refresh_task = refresh()
                    if inspect.isawaitable(refresh_task):
                        try:
                            hass.async_create_task(refresh_task)
                        except Exception:
                            try:
                                loop = asyncio.get_running_loop()
                                self._last_refresh_task = loop.create_task(refresh_task)
                            except Exception:
                                if hasattr(refresh_task, "close"):
                                    refresh_task.close()
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
    ) -> bool:
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

        return await self._async_client_call(log_context=log_context, call=_submit)

    async def _async_client_call(
        self,
        *,
        log_context: str,
        call: Callable[[Any], Awaitable[Any]],
    ) -> bool:
        """Call a backend helper while applying standard error handling."""

        client = self._client()
        if client is None:
            _LOGGER.error(
                "%s failed type=%s addr=%s: client unavailable",
                log_context,
                self._node_type,
                self._addr,
            )
            return False

        try:
            await call(client)
        except asyncio.CancelledError:
            raise
        except Exception as err:
            status = getattr(err, "status", None)
            body = (
                getattr(err, "body", None) or getattr(err, "message", None) or str(err)
            )
            _LOGGER.error(
                "%s failed type=%s addr=%s: status=%s body=%s",
                log_context,
                self._node_type,
                self._addr,
                status,
                (str(body)[:200] if body else ""),
            )
            return False
        return True

    async def _async_submit_settings(
        self,
        client,
        *,
        mode: str | None,
        stemp: float | None,
        prog: list[int] | None,
        ptemp: list[float] | None,
        units: str,
    ) -> None:
        """Send settings for this heater to the backend."""

        await client.set_node_settings(
            self._dev_id,
            (self._node_type, self._addr),
            mode=mode,
            stemp=stemp,
            prog=prog,
            ptemp=ptemp,
            units=units,
        )

    # -------------------- WS updates --------------------
    @callback
    def _handle_ws_event(self, payload: dict) -> None:
        """React to websocket updates for this heater."""
        kind = payload.get("kind")
        addr = payload.get("addr")
        expected_kind = f"{self._node_type}_settings"
        cancel_fallback = False
        if kind == expected_kind:
            cancel_fallback = addr is None or (
                normalize_node_addr(addr) == self._addr if addr is not None else False
            )
        elif self._refresh_fallback and self._payload_mentions_heater(payload):
            cancel_fallback = True

        if cancel_fallback and self._refresh_fallback:
            if not self._refresh_fallback.done():
                self._refresh_fallback.cancel()
            self._refresh_fallback = None
        super()._handle_ws_event(payload)

    def _payload_mentions_heater(self, payload: Mapping[str, Any]) -> bool:
        """Return True when a websocket payload references this heater."""

        if not isinstance(payload, Mapping):
            return False

        node_type = self._node_type
        normalized_addr = normalize_node_addr(
            self._addr,
            use_default_when_falsey=True,
        )
        if not normalized_addr:
            return False

        direct_addr = normalize_node_addr(
            payload.get("addr"),
            use_default_when_falsey=True,
        )
        if direct_addr and direct_addr == normalized_addr:
            return True

        inventory = self._shared_inventory()
        if inventory is None:
            return False

        _, reverse_map = inventory.heater_address_map
        addr_types = reverse_map.get(normalized_addr)
        if addr_types and node_type in addr_types:
            return True

        for sample_type, sample_addr in inventory.heater_sample_targets:
            canonical_type = normalize_node_type(
                sample_type,
                use_default_when_falsey=True,
            )
            if canonical_type != node_type:
                continue
            if (
                normalize_node_addr(sample_addr, use_default_when_falsey=True)
                == normalized_addr
            ):
                return True

        return False

    @property
    def hvac_mode(self) -> HVACMode:
        """Return the HA HVAC mode derived from heater settings."""

        state = self.heater_state()
        mode = (getattr(state, "mode", None) or "").lower()
        if mode == "off":
            return HVACMode.OFF
        if mode == "auto":
            return HVACMode.AUTO
        if mode == "manual":
            return HVACMode.HEAT
        return HVACMode.HEAT

    @property
    def hvac_action(self) -> HVACAction | None:
        """Return the current HVAC action reported by the heater."""

        state = self.heater_state()
        heater_state = (getattr(state, "state", None) or "").lower()
        if not heater_state:
            return None
        if heater_state in ("off", "idle", "standby"):
            return HVACAction.IDLE if self.hvac_mode != HVACMode.OFF else HVACAction.OFF
        return HVACAction.HEATING

    @property
    def current_temperature(self) -> float | None:
        """Return the measured ambient temperature."""

        state = self.heater_state()
        return float_or_none(getattr(state, "mtemp", None))

    @property
    def target_temperature(self) -> float | None:
        """Return the target temperature set on the heater."""

        state = self.heater_state()
        return float_or_none(getattr(state, "stemp", None))

    @property
    def min_temp(self) -> float:
        """Return the minimum supported setpoint."""
        return 5.0

    @property
    def max_temp(self) -> float:
        """Return the maximum supported setpoint."""
        return 30.0

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
                    attrs["program_setpoint"] = float_or_none(ptemp[slot])
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
        write_kwargs: Mapping[str, Any],
        apply_fn: Callable[[DomainState], None],
        success_details: Mapping[str, Any] | None = None,
    ) -> None:
        """Submit a heater write, update cached state, and schedule fallback."""
        success = await self._async_write_settings(
            log_context=log_context,
            **dict(write_kwargs),
        )
        if not success:
            return

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
        self._schedule_refresh_fallback()

    async def async_set_schedule(self, prog: list[int]) -> None:
        """Write the 7x24 tri-state program to the device."""
        # Validate defensively even though the schema should catch most issues
        if not isinstance(prog, list) or len(prog) != 168:
            _LOGGER.error(
                "Invalid prog length for type=%s addr=%s",
                self._node_type,
                self._addr,
            )
            return
        try:
            prog2 = [int(x) for x in prog]
            if any(x not in (0, 1, 2) for x in prog2):
                raise ValueError("prog values must be 0/1/2")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _LOGGER.error(
                "Invalid prog for type=%s addr=%s: %s",
                self._node_type,
                self._addr,
                e,
            )
            return

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
        """Write the cold/night/day preset temperatures."""
        if "ptemp" in kwargs and isinstance(kwargs["ptemp"], list):
            p = kwargs["ptemp"]
        else:
            try:
                p = [kwargs["cold"], kwargs["night"], kwargs["day"]]
            except asyncio.CancelledError:
                raise
            except Exception:
                _LOGGER.error(
                    "Preset temperatures require either ptemp[3] or cold/night/day fields"
                )
                return

        if not isinstance(p, list) or len(p) != 3:
            _LOGGER.error(
                "Invalid ptemp length for type=%s addr=%s",
                self._node_type,
                self._addr,
            )
            return
        try:
            p2 = [float(x) for x in p]
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _LOGGER.error(
                "Invalid ptemp values for type=%s addr=%s: %s",
                self._node_type,
                self._addr,
                e,
            )
            return

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
        except (TypeError, ValueError):
            _LOGGER.error("Invalid temperature payload: %r", raw)
            return

        t = max(5.0, min(30.0, t))
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

        return HVACMode.HEAT

    def _requires_setpoint_with_mode(self, hvac_mode: HVACMode | str) -> bool:
        """Return whether the backend needs a target temperature for the mode."""

        return hvac_mode == HVACMode.HEAT

    def _allows_setpoint_in_mode(self, hvac_mode: HVACMode | str) -> bool:
        """Return whether a mode already supports standalone setpoint writes."""

        return hvac_mode == HVACMode.HEAT

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

        _LOGGER.error("Unsupported hvac_mode=%s", hvac_mode)

    async def _ensure_write_task(self) -> None:
        """Schedule a debounced write task if one is not running."""
        if self._write_task and not self._write_task.done():
            return
        self._write_task = asyncio.create_task(
            self._write_after_debounce(),
            name=f"termoweb-write-{self._dev_id}-{self._addr}",
        )

    async def _write_after_debounce(self) -> None:
        """Batch pending mode/setpoint writes after the debounce interval."""
        await asyncio.sleep(_WRITE_DEBOUNCE)
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

        success = await self._async_write_settings(
            log_context="Mode/setpoint write",
            mode=mode_api,
            stemp=stemp,
        )
        if not success:
            return

        register_pending = getattr(self.coordinator, "register_pending_setting", None)
        if callable(register_pending):
            try:
                register_pending(
                    self._node_type,
                    self._addr,
                    mode=mode_api,
                    stemp=float_or_none(stemp),
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
        self._schedule_refresh_fallback()

    def _schedule_refresh_fallback(self) -> None:
        """Schedule a refresh if the websocket echo does not arrive."""
        if self._refresh_fallback:
            if not self._refresh_fallback.done():
                self._refresh_fallback.cancel()
            self._refresh_fallback = None

        hass = self._hass_for_runtime()
        ws_record = None
        if hass is not None:
            try:
                runtime = require_runtime(hass, self._entry_id)
            except LookupError:
                runtime = None
            if runtime is not None:
                ws_record = runtime.ws_state
        if isinstance(ws_record, dict):
            ws_state = ws_record.get(self._dev_id)
            if isinstance(ws_state, dict):
                status = str(ws_state.get("status") or "").lower()
                if status in {"connected", "healthy"}:
                    last_payload_at = ws_state.get("last_payload_at")
                    idle_restart_pending = bool(ws_state.get("idle_restart_pending"))
                    recent_payload = False
                    if isinstance(last_payload_at, (int, float)):
                        recent_payload = (time.time() - last_payload_at) <= (
                            _WS_ECHO_FALLBACK_REFRESH
                        )
                    if idle_restart_pending or recent_payload:
                        _LOGGER.debug(
                            "Skipping refresh fallback addr=%s ws_status=%s",
                            self._addr,
                            status,
                        )
                        return

        async def _fallback() -> None:
            """Force a heater refresh after the fallback delay."""
            task = asyncio.current_task()
            await asyncio.sleep(_WS_ECHO_FALLBACK_REFRESH)
            try:
                hass = self._hass_for_runtime()
                is_stopping = getattr(hass, "is_stopping", False)
                is_running = getattr(hass, "is_running", True)
                if is_stopping or not is_running:
                    reason = "stopping" if is_stopping else "not running"
                    _LOGGER.debug(
                        "Skipping refresh fallback addr=%s: hass %s",
                        self._addr,
                        reason,
                    )
                    return

                await self.coordinator.async_refresh_heater(
                    (self._node_type, self._addr)
                )
                if task and self._refresh_fallback is task:
                    self._refresh_fallback = None
                    return
            except asyncio.CancelledError:
                raise
            except Exception as e:
                _LOGGER.error(
                    "Refresh fallback failed addr=%s: %s",
                    self._addr,
                    str(e),
                )
            finally:
                if task and self._refresh_fallback is task:
                    self._refresh_fallback = None

        self._refresh_fallback = asyncio.create_task(
            _fallback(), name=f"termoweb-fallback-{self._dev_id}-{self._addr}"
        )


class AccumulatorClimateEntity(HeaterClimateEntity):
    """HA climate entity for TermoWeb accumulator nodes."""

    _attr_hvac_modes: list[HVACMode] = [HVACMode.OFF, HVACMode.AUTO]
    _attr_supported_features = (
        ClimateEntityFeature.TARGET_TEMPERATURE | ClimateEntityFeature.PRESET_MODE
    )
    _attr_preset_modes = ["none", "boost"]

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

        hass = getattr(self, "hass", None)
        if hass is None:
            return DEFAULT_BOOST_DURATION
        return resolve_boost_runtime_minutes(
            hass,
            self._entry_id,
            self._node_type,
            self._addr,
        )

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
        fallback = super().hvac_mode
        return fallback if fallback is not None else None

    async def async_set_hvac_mode(self, hvac_mode: HVACMode | str) -> None:
        """Handle accumulator HVAC modes, delegating boost to presets."""

        if isinstance(hvac_mode, HVACMode):
            value = hvac_mode.value.lower()
        else:
            value = str(hvac_mode).lower()
        if value == "boost":
            _LOGGER.error(
                "Boost is exposed as a preset_mode for accumulators addr=%s",
                self._addr,
            )
            return
        if value == str(HVACMode.HEAT):
            _LOGGER.error("Unsupported hvac_mode=%s for accumulator", hvac_mode)
            return
        await super().async_set_hvac_mode(hvac_mode)

    @property
    def preset_mode(self) -> str:
        """Return the active preset mode."""

        state = self.accumulator_state()
        if (getattr(state, "mode", None) or "").lower() == "boost":
            return "boost"
        return "none"

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        """Set the accumulator preset mode."""

        value = (preset_mode or "").lower()
        if value not in self._attr_preset_modes:
            _LOGGER.error("Unsupported preset_mode=%s for accumulator", preset_mode)
            return

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
    def extra_state_attributes(self) -> Mapping[str, Any] | None:
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
        """Return a validated boost duration or ``None`` when absent."""

        if minutes is None:
            return None
        value = coerce_boost_minutes(minutes)
        if value is None:
            _LOGGER.error(
                "Invalid boost minutes for type=%s addr=%s: %s",
                self._node_type,
                self._addr,
                minutes,
            )
            return None
        try:
            return validate_boost_minutes(value)
        except ValueError:
            _LOGGER.error(
                "Boost duration must be one of [%s] minutes for type=%s addr=%s: %s",
                ALLOWED_BOOST_MINUTES_MESSAGE,
                self._node_type,
                self._addr,
                value,
            )
            return None

    async def async_set_acm_preset(
        self,
        *,
        minutes: int | None = None,
        temperature: float | None = None,
    ) -> None:
        """Update the default boost duration and/or temperature."""

        if minutes is None and temperature is None:
            _LOGGER.error(
                "Accumulator preset update requires minutes and/or temperature"
            )
            return

        validated_minutes = self._validate_boost_minutes(minutes)
        if minutes is not None and validated_minutes is None:
            return

        temp_value: float | None = None
        if temperature is not None:
            try:
                temp_value = float(temperature)
            except (TypeError, ValueError):
                _LOGGER.error(
                    "Invalid boost temperature for type=%s addr=%s: %s",
                    self._node_type,
                    self._addr,
                    temperature,
                )
                return

        async def _call(client: Any) -> None:
            await client.set_acm_extra_options(
                self._dev_id,
                self._addr,
                boost_time=validated_minutes,
                boost_temp=temp_value,
            )

        success = await self._async_client_call(
            log_context="Boost preset write",
            call=_call,
        )
        if not success:
            return

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
        self._schedule_refresh_fallback()

    async def async_start_boost(self, *, minutes: int | None = None) -> None:
        """Start an accumulator boost session."""

        validated_minutes = self._validate_boost_minutes(minutes)
        if minutes is not None and validated_minutes is None:
            return
        if validated_minutes is None:
            validated_minutes = self._preferred_boost_minutes()

        state = self.accumulator_state()
        boost_temp = float_or_none(getattr(state, "boost_temp", None))
        if boost_temp is None:
            boost_temp = float_or_none(getattr(state, "stemp", None))
        if boost_temp is None:
            _LOGGER.error(
                "Boost start requires a setpoint for type=%s addr=%s",
                self._node_type,
                self._addr,
            )
            return

        units = self._units()

        async def _call(client: Any) -> None:
            await client.set_acm_boost_state(
                self._dev_id,
                self._addr,
                boost=True,
                boost_time=validated_minutes,
                stemp=boost_temp,
                units=units,
            )

        success = await self._async_client_call(
            log_context="Boost start",
            call=_call,
        )
        if not success:
            return

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
        self._schedule_refresh_fallback()

    async def async_cancel_boost(self) -> None:
        """Cancel the active accumulator boost session."""

        async def _call(client: Any) -> None:
            await client.set_acm_boost_state(
                self._dev_id,
                self._addr,
                boost=False,
            )

        success = await self._async_client_call(
            log_context="Boost cancel",
            call=_call,
        )
        if not success:
            return

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
        self._schedule_refresh_fallback()

    async def _async_submit_settings(  # type: ignore[override]
        self,
        client,
        *,
        mode: str | None,
        stemp: float | None,
        prog: list[int] | None,
        ptemp: list[float] | None,
        units: str,
    ) -> None:
        boost_context: BoostContext | None = None
        backend = None
        hass = getattr(self, "hass", None)
        if hass is not None:
            try:
                runtime = require_runtime(hass, self._entry_id)
            except LookupError:
                runtime = None
            if runtime is not None:
                backend = runtime.backend

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
            legacy_active: bool | None = None
            mode_value: str | None = None
            if state is not None:
                boost_flag = getattr(state, "boost_active", None)
                if isinstance(boost_flag, bool):
                    legacy_active = boost_flag
                else:
                    legacy_flag = getattr(state, "boost", None)
                    if isinstance(legacy_flag, bool):
                        legacy_active = legacy_flag
                mode_value = getattr(state, "mode", None)
                if not isinstance(mode_value, str):
                    mode_value = None
            boost_context = BoostContext(
                active=boost_state.active if boost_state is not None else None,
                legacy_active=legacy_active,
                mode=mode_value,
            )

        if backend is not None:
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
            return

        await client.set_node_settings(
            self._dev_id,
            (self._node_type, self._addr),
            mode=mode,
            stemp=stemp,
            prog=prog,
            ptemp=ptemp,
            units=units,
        )
