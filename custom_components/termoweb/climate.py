from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from homeassistant.components.climate import (
    ClimateEntity,
    HVACMode,
    HVACAction,
    ClimateEntityFeature,
)
from homeassistant.const import UnitOfTemperature, ATTR_TEMPERATURE
from homeassistant.core import callback, ServiceCall
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.util import dt as dt_util
from homeassistant.helpers import entity_platform
import voluptuous as vol

from .const import DOMAIN, signal_ws_data

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
    nodes = data["nodes"]
    addrs: list[str] = data["htr_addrs"]

    name_map: dict[str, str] = {}
    node_list = nodes.get("nodes") if isinstance(nodes, dict) else None
    if isinstance(node_list, list):
        for node in node_list:
            if isinstance(node, dict) and (node.get("type") or "").lower() == "htr":
                addr = str(node.get("addr"))
                name = (node.get("name") or f"Heater {addr}").strip()
                name_map[addr] = name

    new_entities = [
        TermoWebHeater(coordinator, entry.entry_id, dev_id, addr, name_map.get(addr, f"Heater {addr}"))
        for addr in addrs
    ]
    if new_entities:
        _LOGGER.debug("Adding %d TermoWeb heater entities", len(new_entities))
        async_add_entities(new_entities)

    # -------------------- Register entity services --------------------
    platform = entity_platform.async_get_current_platform()

    # Explicit callables ensure dispatch and let us add clear logs when invoked.
    async def _svc_set_schedule(entity: "TermoWebHeater", call: ServiceCall) -> None:
        prog = call.data.get("prog")
        _LOGGER.info(
            "entity-service termoweb.set_schedule -> %s prog_len=%s",
            getattr(entity, "entity_id", "<no-entity-id>"),
            len(prog) if isinstance(prog, list) else "<invalid>",
        )
        await entity.async_set_schedule(prog)

    async def _svc_set_preset_temperatures(entity: "TermoWebHeater", call: ServiceCall) -> None:
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


class TermoWebHeater(CoordinatorEntity, ClimateEntity):
    """HA climate entity representing a single TermoWeb heater."""

    _attr_supported_features = ClimateEntityFeature.TARGET_TEMPERATURE | ClimateEntityFeature.PRESET_MODE
    _attr_hvac_modes = [HVACMode.HEAT, HVACMode.OFF]
    _attr_temperature_unit = UnitOfTemperature.CELSIUS
    # Server exposes "auto" (program) and "manual". "off" is an hvac_mode.
    _attr_preset_modes = ["auto", "manual"]

    def __init__(self, coordinator, entry_id: str, dev_id: str, addr: str, name: str) -> None:
        super().__init__(coordinator)
        self._entry_id = entry_id
        self._dev_id = dev_id
        self._addr = addr
        self._attr_name = name
        self._attr_unique_id = f"{DOMAIN}:{dev_id}:htr:{addr}"
        self._unsub_ws = None

        # pending write aggregation
        self._pending_mode: Optional[str] = None
        self._pending_stemp: Optional[float] = None
        self._write_task: Optional[asyncio.Task] = None

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        self._unsub_ws = async_dispatcher_connect(
            self.hass, signal_ws_data(self._entry_id), self._on_ws_data
        )
        self.async_on_remove(lambda: self._unsub_ws() if self._unsub_ws else None)

    @property
    def device_info(self) -> DeviceInfo:
        return DeviceInfo(identifiers={(DOMAIN, self._dev_id)})

    # -------------------- Helpers --------------------
    def _client(self):
        return self.hass.data[DOMAIN][self._entry_id]["client"]

    def _settings(self) -> dict[str, Any] | None:
        d = (self.coordinator.data or {}).get(self._dev_id, {})
        htr = d.get("htr") or {}
        settings = (htr.get("settings") or {}).get(self._addr)
        return settings if isinstance(settings, dict) else None

    @staticmethod
    def _f(v: Any) -> Optional[float]:
        try:
            if v is None:
                return None
            if isinstance(v, (int, float)):
                return float(v)
            s = str(v).strip()
            return float(s) if s else None
        except Exception:
            return None

    def _units(self) -> str:
        s = self._settings() or {}
        u = (s.get("units") or "C").upper()
        return "C" if u not in ("C", "F") else u

    @staticmethod
    def _slot_label(v: int) -> Optional[str]:
        return {0: "cold", 1: "night", 2: "day"}.get(v)

    def _current_prog_slot(self, s: dict[str, Any]) -> Optional[int]:
        prog = s.get("prog")
        if not isinstance(prog, list) or len(prog) < 168:
            return None
        now = dt_util.now()
        idx = now.weekday() * 24 + now.hour
        try:
            return int(prog[idx])
        except Exception:
            return None

    # -------------------- WS updates --------------------
    @callback
    def _on_ws_data(self, payload: dict) -> None:
        if payload.get("dev_id") != self._dev_id:
            return
        addr = payload.get("addr")
        if addr is not None and str(addr) != self._addr:
            return
        # Thread-safe state push
        self.schedule_update_ha_state()

    # -------------------- Read properties --------------------
    @property
    def should_poll(self) -> bool:
        return False

    @property
    def available(self) -> bool:
        d = (self.coordinator.data or {}).get(self._dev_id, {})
        return d.get("nodes") is not None

    @property
    def hvac_mode(self) -> HVACMode:
        s = self._settings() or {}
        mode = (s.get("mode") or "").lower()
        if mode == "off":
            return HVACMode.OFF
        return HVACMode.HEAT

    @property
    def hvac_action(self) -> Optional[HVACAction]:
        s = self._settings() or {}
        state = (s.get("state") or "").lower()
        if not state:
            return None
        if state in ("off", "idle", "standby"):
            return HVACAction.IDLE if self.hvac_mode != HVACMode.OFF else HVACAction.OFF
        return HVACAction.HEATING

    @property
    def preset_mode(self) -> Optional[str]:
        s = self._settings() or {}
        mode = (s.get("mode") or "").lower()
        return mode if mode in self._attr_preset_modes else None

    @property
    def current_temperature(self) -> Optional[float]:
        s = self._settings() or {}
        return self._f(s.get("mtemp"))

    @property
    def target_temperature(self) -> Optional[float]:
        s = self._settings() or {}
        return self._f(s.get("stemp"))

    @property
    def min_temp(self) -> float:
        return 5.0

    @property
    def max_temp(self) -> float:
        return 30.0

    @property
    def icon(self) -> str | None:
        """Dynamic radiator icon: disabled when OFF, radiator when heating, radiator (idle) otherwise."""
        if self.hvac_mode == HVACMode.OFF:
            return "mdi:radiator-disabled"
        if self.hvac_action == HVACAction.HEATING:
            return "mdi:radiator"
        return "mdi:radiator"

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        s = self._settings() or {}
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
                    attrs["program_setpoint"] = self._f(ptemp[slot])
            except Exception:
                pass

        return attrs

    # -------------------- Entity services: schedule & preset temps --------------------
    async def async_set_schedule(self, prog: list[int]) -> None:
        """Write the 7x24 tri-state program to the device."""
        # Validate defensively even though the schema should catch most issues
        if not isinstance(prog, list) or len(prog) != 168:
            _LOGGER.error("Invalid prog length for dev=%s addr=%s", self._dev_id, self._addr)
            return
        try:
            prog2 = [int(x) for x in prog]
            if any(x not in (0, 1, 2) for x in prog2):
                raise ValueError("prog values must be 0/1/2")
        except Exception as e:
            _LOGGER.error("Invalid prog for dev=%s addr=%s: %s", self._dev_id, self._addr, e)
            return

        client = self._client()
        try:
            await client.set_htr_settings(self._dev_id, self._addr, prog=prog2, units=self._units())
            _LOGGER.debug("Schedule write OK dev=%s addr=%s (prog_len=%d)", self._dev_id, self._addr, len(prog2))
        except Exception as e:
            status = getattr(e, "status", None)
            body = getattr(e, "body", None) or getattr(e, "message", None) or str(e)
            _LOGGER.error(
                "Schedule write failed dev=%s addr=%s: status=%s body=%s",
                self._dev_id,
                self._addr,
                status,
                (str(body)[:200] if body else ""),
            )
            return

        # Optimistic local update so HA shows the change immediately
        try:
            d = (self.coordinator.data or {}).get(self._dev_id, {})
            htr = d.get("htr") or {}
            settings_map = htr.get("settings") or {}
            cur = settings_map.get(self._addr)
            if isinstance(cur, dict):
                cur["prog"] = list(prog2)
                self.async_write_ha_state()
                _LOGGER.debug("Optimistic prog applied dev=%s addr=%s", self._dev_id, self._addr)
        except Exception as e:
            _LOGGER.debug("Optimistic prog update failed dev=%s addr=%s: %s", self._dev_id, self._addr, e)

        # Expect WS echo; schedule refresh if it doesn't arrive soon.
        self._schedule_refresh_fallback()

    async def async_set_preset_temperatures(self, **kwargs) -> None:
        """Write the cold/night/day preset temperatures."""
        if "ptemp" in kwargs and isinstance(kwargs["ptemp"], list):
            p = kwargs["ptemp"]
        else:
            try:
                p = [kwargs["cold"], kwargs["night"], kwargs["day"]]
            except Exception:
                _LOGGER.error("Preset temperatures require either ptemp[3] or cold/night/day fields")
                return

        if not isinstance(p, list) or len(p) != 3:
            _LOGGER.error("Invalid ptemp length for dev=%s addr=%s", self._dev_id, self._addr)
            return
        try:
            p2 = [float(x) for x in p]
        except Exception as e:
            _LOGGER.error("Invalid ptemp values for dev=%s addr=%s: %s", self._dev_id, self._addr, e)
            return

        client = self._client()
        try:
            await client.set_htr_settings(self._dev_id, self._addr, ptemp=p2, units=self._units())
            _LOGGER.debug("Preset write OK dev=%s addr=%s ptemp=%s", self._dev_id, self._addr, p2)
        except Exception as e:
            status = getattr(e, "status", None)
            body = getattr(e, "body", None) or getattr(e, "message", None) or str(e)
            _LOGGER.error(
                "Preset write failed dev=%s addr=%s: status=%s body=%s",
                self._dev_id,
                self._addr,
                status,
                (str(body)[:200] if body else ""),
            )
            return

        # Optimistic local update so HA shows the change immediately
        try:
            d = (self.coordinator.data or {}).get(self._dev_id, {})
            htr = d.get("htr") or {}
            settings_map = htr.get("settings") or {}
            cur = settings_map.get(self._addr)
            if isinstance(cur, dict):
                cur["ptemp"] = [f"{t:.1f}" if isinstance(t, float) else t for t in p2]
                self.async_write_ha_state()
                _LOGGER.debug("Optimistic ptemp applied dev=%s addr=%s", self._dev_id, self._addr)
        except Exception as e:
            _LOGGER.debug("Optimistic ptemp update failed dev=%s addr=%s: %s", self._dev_id, self._addr, e)

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
        self._pending_mode = "manual"  # required by backend for setpoint acceptance
        _LOGGER.info(
            "Queue write: dev=%s addr=%s stemp=%.1f mode=manual (batching %.1fs)",
            self._dev_id,
            self._addr,
            t,
            _WRITE_DEBOUNCE,
        )
        await self._ensure_write_task()

    async def async_set_hvac_mode(self, hvac_mode: HVACMode) -> None:
        """Set off/heat. For HEAT prefer 'auto' unless user changes setpoint or preset."""
        if hvac_mode == HVACMode.OFF:
            self._pending_mode = "off"
        else:
            self._pending_mode = "auto"
        _LOGGER.info(
            "Queue write: dev=%s addr=%s mode=%s (batching %.1fs)",
            self._dev_id,
            self._addr,
            self._pending_mode,
            _WRITE_DEBOUNCE,
        )
        await self._ensure_write_task()

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        """Set 'auto' or 'manual'. For 'manual' include a setpoint."""
        preset_mode = (preset_mode or "").lower()
        if preset_mode not in self._attr_preset_modes:
            _LOGGER.error("Unsupported preset_mode=%r", preset_mode)
            return

        self._pending_mode = preset_mode
        if preset_mode == "manual" and self._pending_stemp is None:
            cur = self.target_temperature
            if cur is not None:
                self._pending_stemp = float(cur)

        _LOGGER.info(
            "Queue write: dev=%s addr=%s mode=%s stemp=%s (batching %.1fs)",
            self._dev_id,
            self._addr,
            self._pending_mode,
            self._pending_stemp,
            _WRITE_DEBOUNCE,
        )
        await self._ensure_write_task()

    async def _ensure_write_task(self) -> None:
        if self._write_task and not self._write_task.done():
            return
        self._write_task = asyncio.create_task(
            self._write_after_debounce(), name=f"termoweb-write-{self._dev_id}-{self._addr}"
        )

    async def _write_after_debounce(self) -> None:
        await asyncio.sleep(_WRITE_DEBOUNCE)
        mode = self._pending_mode
        stemp = self._pending_stemp
        self._pending_mode = None
        self._pending_stemp = None

        # Normalize to backend rules:
        # - If stemp present but mode is not, force manual.
        # - If mode=manual but stemp missing, include current target.
        if stemp is not None and (mode is None or mode != "manual"):
            mode = "manual"
        if mode == "manual" and stemp is None:
            current = self.target_temperature
            if current is not None:
                stemp = float(current)

        if mode is None and stemp is None:
            return

        client = self._client()
        try:
            _LOGGER.info(
                "POST htr settings dev=%s addr=%s mode=%s stemp=%s",
                self._dev_id,
                self._addr,
                mode,
                stemp,
            )
            await client.set_htr_settings(
                self._dev_id, self._addr, mode=mode, stemp=stemp, units=self._units()
            )
        except Exception as e:
            status = getattr(e, "status", None)
            body = getattr(e, "body", None) or getattr(e, "message", None) or str(e)
            _LOGGER.error(
                "Write failed dev=%s addr=%s mode=%s stemp=%s: status=%s body=%s",
                self._dev_id,
                self._addr,
                mode,
                stemp,
                status,
                (str(body)[:200] if body else ""),
            )
            return

        # Expect WS echo; schedule refresh if it doesn't arrive soon.
        self._schedule_refresh_fallback()

    def _schedule_refresh_fallback(self) -> None:
        async def _fallback():
            await asyncio.sleep(_WS_ECHO_FALLBACK_REFRESH)
            try:
                await self.coordinator.async_request_refresh()
            except Exception as e:
                _LOGGER.error(
                    "Refresh fallback failed dev=%s addr=%s: %s",
                    self._dev_id,
                    self._addr,
                    str(e),
                )

        asyncio.create_task(_fallback())
