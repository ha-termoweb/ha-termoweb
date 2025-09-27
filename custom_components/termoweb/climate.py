from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

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

from .const import DOMAIN
from .heater import HeaterNodeBase, log_skipped_nodes, prepare_heater_platform_data
from .nodes import HeaterNode
from .utils import HEATER_NODE_TYPES, float_or_none

_LOGGER = logging.getLogger(__name__)

# Small debounce so multiple UI events coalesce
_WRITE_DEBOUNCE = 0.2
# If WS echo doesn't arrive quickly after a successful write, force a refresh
_WS_ECHO_FALLBACK_REFRESH = 2.0


async def async_setup_entry(hass, entry, async_add_entities):
    """Discover heater nodes and create climate entities."""
    data = hass.data[DOMAIN][entry.entry_id]
    coordinator = data["coordinator"]
    dev_id = data["dev_id"]
    _, nodes_by_type, _, resolve_name = prepare_heater_platform_data(
        data,
        default_name_simple=lambda addr: f"Heater {addr}",
    )

    new_entities: list[ClimateEntity] = []
    for node_type in HEATER_NODE_TYPES:
        for node in nodes_by_type.get(node_type, []):
            addr_str = str(getattr(node, "addr", "")).strip()
            if not addr_str:
                continue
            resolved_name = resolve_name(node_type, addr_str)
            unique_id = f"{DOMAIN}:{dev_id}:{node_type}:{addr_str}:climate"
            entity_cls: type[HeaterClimateEntity]
            if node_type == "acm":
                entity_cls = AccumulatorClimateEntity
            else:
                entity_cls = HeaterClimateEntity
            new_entities.append(
                entity_cls(
                    coordinator,
                    entry.entry_id,
                    dev_id,
                    addr_str,
                    resolved_name,
                    unique_id,
                    node_type=node_type,
                )
            )

    log_skipped_nodes("climate", nodes_by_type, logger=_LOGGER)
    if new_entities:
        _LOGGER.debug("Adding %d TermoWeb heater entities", len(new_entities))
        async_add_entities(new_entities)

    # -------------------- Register entity services --------------------
    platform = entity_platform.async_get_current_platform()

    # Explicit callables ensure dispatch and let us add clear logs when invoked.
    async def _svc_set_schedule(entity: HeaterClimateEntity, call: ServiceCall) -> None:
        """Handle the set_schedule entity service."""
        prog = call.data.get("prog")
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
    ) -> None:
        """Initialise the climate entity for a TermoWeb heater."""
        HeaterNode.__init__(self, name=name, addr=addr)
        resolved_type = str(node_type or getattr(self, "type", "htr")).strip().lower()
        if resolved_type and resolved_type != getattr(self, "type", ""):
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
        )

        self._refresh_fallback: asyncio.Task | None = None

        # pending write aggregation
        self._pending_mode: HVACMode | None = None
        self._pending_stemp: float | None = None
        self._write_task: asyncio.Task | None = None

    async def async_will_remove_from_hass(self) -> None:
        """Clean up pending tasks when the entity is removed."""
        if self._refresh_fallback:
            self._refresh_fallback.cancel()
            self._refresh_fallback = None
        await super().async_will_remove_from_hass()

    @staticmethod
    def _slot_label(v: int) -> str | None:
        """Translate a program slot integer into a label."""
        return {0: "cold", 1: "night", 2: "day"}.get(v)

    def _current_prog_slot(self, s: dict[str, Any]) -> int | None:
        """Return the active program slot index for the heater."""
        prog = s.get("prog")
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

    def _settings_maps(self) -> list[dict[str, Any]]:
        """Return all cached settings maps referencing this node."""
        data = (self.coordinator.data or {}).get(self._dev_id, {})
        maps: list[dict[str, Any]] = []
        seen: set[int] = set()

        if isinstance(data, dict):
            by_type = data.get("nodes_by_type")
            if isinstance(by_type, dict):
                section = by_type.get(self._node_type)
                if isinstance(section, dict):
                    settings_map = section.get("settings")
                    if isinstance(settings_map, dict) and id(settings_map) not in seen:
                        maps.append(settings_map)
                        seen.add(id(settings_map))

            legacy = data.get("htr")
            if isinstance(legacy, dict):
                settings_map = legacy.get("settings")
                if isinstance(settings_map, dict) and id(settings_map) not in seen:
                    maps.append(settings_map)
                    seen.add(id(settings_map))

        return maps

    def _optimistic_update(self, mutator: Callable[[dict[str, Any]], None]) -> None:
        """Apply ``mutator`` to cached settings and refresh state if changed."""
        try:
            updated = False
            for settings_map in self._settings_maps():
                cur = settings_map.get(self._addr)
                if isinstance(cur, dict):
                    mutator(cur)
                    updated = True
            if updated:
                self.async_write_ha_state()
        except asyncio.CancelledError:
            raise
        except Exception as err:  # pragma: no cover - defensive
            _LOGGER.debug(
                "Optimistic update failed dev=%s type=%s addr=%s: %s",
                self._dev_id,
                self._node_type,
                self._addr,
                err,
            )

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
        client = self._client()
        if client is None:
            _LOGGER.error(
                "%s failed dev=%s type=%s addr=%s: client unavailable",
                log_context,
                self._dev_id,
                self._node_type,
                self._addr,
            )
            return False

        try:
            await self._async_submit_settings(
                client,
                mode=mode,
                stemp=stemp,
                prog=prog,
                ptemp=ptemp,
                units=self._units(),
            )
        except asyncio.CancelledError:
            raise
        except Exception as err:
            status = getattr(err, "status", None)
            body = (
                getattr(err, "body", None) or getattr(err, "message", None) or str(err)
            )
            _LOGGER.error(
                "%s failed dev=%s type=%s addr=%s: status=%s body=%s",
                log_context,
                self._dev_id,
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
        """Send settings via the heater-specific API."""

        await client.set_htr_settings(
            self._dev_id,
            self._addr,
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
        if kind == expected_kind and addr is not None and self._refresh_fallback:
            if not self._refresh_fallback.done():
                self._refresh_fallback.cancel()
            self._refresh_fallback = None
        super()._handle_ws_event(payload)

    @property
    def hvac_mode(self) -> HVACMode:
        """Return the HA HVAC mode derived from heater settings."""
        s = self.heater_settings() or {}
        mode = (s.get("mode") or "").lower()
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
        s = self.heater_settings() or {}
        state = (s.get("state") or "").lower()
        if not state:
            return None
        if state in ("off", "idle", "standby"):
            return HVACAction.IDLE if self.hvac_mode != HVACMode.OFF else HVACAction.OFF
        return HVACAction.HEATING

    @property
    def current_temperature(self) -> float | None:
        """Return the measured ambient temperature."""
        s = self.heater_settings() or {}
        return float_or_none(s.get("mtemp"))

    @property
    def target_temperature(self) -> float | None:
        """Return the target temperature set on the heater."""
        s = self.heater_settings() or {}
        return float_or_none(s.get("stemp"))

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
        s = self.heater_settings() or {}
        attrs: dict[str, Any] = {
            "dev_id": self._dev_id,
            "addr": self._addr,
            "units": s.get("units"),
            "max_power": s.get("max_power"),
            "ptemp": s.get("ptemp"),
            "prog": s.get("prog"),  # full weekly program (168 ints)
        }

        slot = self._current_prog_slot(s)
        if slot is not None:
            label = self._slot_label(slot)
            attrs["program_slot"] = label
            ptemp = s.get("ptemp")
            try:
                if isinstance(ptemp, (list, tuple)) and 0 <= slot < len(ptemp):
                    attrs["program_setpoint"] = float_or_none(ptemp[slot])
            except asyncio.CancelledError:
                raise
            except Exception:
                pass

        return attrs

    # -------------------- Entity services: schedule & preset temps --------------------
    async def async_set_schedule(self, prog: list[int]) -> None:
        """Write the 7x24 tri-state program to the device."""
        # Validate defensively even though the schema should catch most issues
        if not isinstance(prog, list) or len(prog) != 168:
            _LOGGER.error(
                "Invalid prog length for dev=%s addr=%s", self._dev_id, self._addr
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
                "Invalid prog for dev=%s addr=%s: %s", self._dev_id, self._addr, e
            )
            return

        success = await self._async_write_settings(
            log_context="Schedule write",
            prog=prog2,
        )
        if not success:
            return

        _LOGGER.debug(
            "Schedule write OK dev=%s type=%s addr=%s (prog_len=%d)",
            self._dev_id,
            self._node_type,
            self._addr,
            len(prog2),
        )

        def _apply(cur: dict[str, Any]) -> None:
            cur["prog"] = list(prog2)

        self._optimistic_update(_apply)

        # Expect WS echo; schedule refresh if it doesn't arrive soon.
        self._schedule_refresh_fallback()

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
                "Invalid ptemp length for dev=%s addr=%s", self._dev_id, self._addr
            )
            return
        try:
            p2 = [float(x) for x in p]
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _LOGGER.error(
                "Invalid ptemp values for dev=%s addr=%s: %s",
                self._dev_id,
                self._addr,
                e,
            )
            return

        success = await self._async_write_settings(
            log_context="Preset write",
            ptemp=p2,
        )
        if not success:
            return

        _LOGGER.debug(
            "Preset write OK dev=%s type=%s addr=%s ptemp=%s",
            self._dev_id,
            self._node_type,
            self._addr,
            p2,
        )

        def _apply(cur: dict[str, Any]) -> None:
            cur["ptemp"] = [f"{t:.1f}" if isinstance(t, float) else t for t in p2]

        self._optimistic_update(_apply)

        # Expect WS echo; schedule refresh if it doesn't arrive soon.
        self._schedule_refresh_fallback()

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
        self._pending_mode = (
            HVACMode.HEAT
        )  # required by backend for setpoint acceptance
        _LOGGER.info(
            "Queue write: dev=%s addr=%s stemp=%.1f mode=%s (batching %.1fs)",
            self._dev_id,
            self._addr,
            t,
            HVACMode.HEAT,
            _WRITE_DEBOUNCE,
        )
        await self._ensure_write_task()

    async def async_set_hvac_mode(self, hvac_mode: HVACMode) -> None:
        """Post off/auto/manual."""
        if hvac_mode == HVACMode.OFF:
            self._pending_mode = HVACMode.OFF
            _LOGGER.info(
                "Queue write: dev=%s addr=%s mode=%s (batching %.1fs)",
                self._dev_id,
                self._addr,
                HVACMode.OFF,
                _WRITE_DEBOUNCE,
            )
            await self._ensure_write_task()
            return

        if hvac_mode == HVACMode.AUTO:
            self._pending_mode = HVACMode.AUTO
            _LOGGER.info(
                "Queue write: dev=%s addr=%s mode=%s (batching %.1fs)",
                self._dev_id,
                self._addr,
                HVACMode.AUTO,
                _WRITE_DEBOUNCE,
            )
            await self._ensure_write_task()
            return

        if hvac_mode == HVACMode.HEAT:
            self._pending_mode = HVACMode.HEAT
            if self._pending_stemp is None:
                cur = self.target_temperature
                if cur is not None:
                    self._pending_stemp = float(cur)
            _LOGGER.info(
                "Queue write: dev=%s addr=%s mode=%s stemp=%s (batching %.1fs)",
                self._dev_id,
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

        # Normalize to backend rules:
        # - If stemp present but mode is not, force manual.
        # - If mode=manual but stemp missing, include current target.
        if stemp is not None and (mode is None or mode != HVACMode.HEAT):
            mode = HVACMode.HEAT
        if mode == HVACMode.HEAT and stemp is None:
            current = self.target_temperature
            if current is not None:
                stemp = float(current)

        if mode is None and stemp is None:
            return

        mode_api = None
        if mode is not None:
            mode_api = {
                HVACMode.OFF: "off",
                HVACMode.AUTO: "auto",
                HVACMode.HEAT: "manual",
            }.get(mode, str(mode))
        _LOGGER.info(
            "POST %s settings dev=%s addr=%s mode=%s stemp=%s",
            self._node_type,
            self._dev_id,
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

        def _apply(cur: dict[str, Any]) -> None:
            if mode_api is not None:
                cur["mode"] = mode_api
            if stemp is not None:
                stemp_str: Any = stemp
                try:
                    stemp_float = float(stemp)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    pass
                else:
                    stemp_str = f"{stemp_float:.1f}"
                cur["stemp"] = stemp_str

        self._optimistic_update(_apply)
        _LOGGER.debug(
            "Optimistic mode/stemp applied dev=%s type=%s addr=%s mode=%s stemp=%s",
            self._dev_id,
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

        ws_record = (
            self.hass.data.get(DOMAIN, {}).get(self._entry_id, {}).get("ws_state")
        )
        if isinstance(ws_record, dict):
            ws_state = ws_record.get(self._dev_id)
            if isinstance(ws_state, dict):
                status = str(ws_state.get("status") or "").lower()
                last_event_at = ws_state.get("last_event_at")
                if status in {"connected", "healthy"} and isinstance(
                    last_event_at, (int, float)
                ):
                    now = time.time()
                    age = now - float(last_event_at)
                    if age < _WS_ECHO_FALLBACK_REFRESH:
                        _LOGGER.debug(
                            "Skipping refresh fallback dev=%s addr=%s ws_status=%s age=%.3f",
                            self._dev_id,
                            self._addr,
                            status,
                            age,
                        )
                        return

        async def _fallback() -> None:
            """Force a heater refresh after the fallback delay."""
            task = asyncio.current_task()
            await asyncio.sleep(_WS_ECHO_FALLBACK_REFRESH)
            try:
                hass = self.hass
                is_stopping = getattr(hass, "is_stopping", False)
                is_running = getattr(hass, "is_running", True)
                if is_stopping or not is_running:
                    reason = "stopping" if is_stopping else "not running"
                    _LOGGER.debug(
                        "Skipping refresh fallback dev=%s addr=%s: hass %s",
                        self._dev_id,
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
                    "Refresh fallback failed dev=%s addr=%s: %s",
                    self._dev_id,
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
        await client.set_node_settings(
            self._dev_id,
            (self._node_type, self._addr),
            mode=mode,
            stemp=stemp,
            prog=prog,
            ptemp=ptemp,
            units=units,
        )
