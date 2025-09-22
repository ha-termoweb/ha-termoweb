from __future__ import annotations

import asyncio
import datetime as dt
import enum
import importlib.util
import logging
from collections import deque
from collections.abc import Coroutine
from pathlib import Path
import sys
import types
from typing import Any, Callable, Deque
from unittest.mock import AsyncMock, MagicMock

import pytest

# -------------------- Minimal Home Assistant stubs --------------------

ha_core = types.ModuleType("homeassistant.core")


class HomeAssistant:  # pragma: no cover - lightweight container
    def __init__(self) -> None:
        self.data: dict[str, Any] = {}


def callback(func: Callable[..., Any]) -> Callable[..., Any]:  # pragma: no cover
    return func


class ServiceCall:  # pragma: no cover - simple data holder
    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self.data = data or {}


ha_core.HomeAssistant = HomeAssistant
ha_core.callback = callback
ha_core.ServiceCall = ServiceCall
sys.modules["homeassistant"] = types.ModuleType("homeassistant")
ha_root = sys.modules["homeassistant"]
sys.modules["homeassistant.core"] = ha_core

ha_components = types.ModuleType("homeassistant.components")
sys.modules["homeassistant.components"] = ha_components

ha_climate = types.ModuleType("homeassistant.components.climate")


class ClimateEntity:  # pragma: no cover - minimal implementation
    def __init__(self) -> None:
        self.hass: HomeAssistant | None = None

    async def async_added_to_hass(self) -> None:
        return None

    async def async_will_remove_from_hass(self) -> None:
        return None

    def schedule_update_ha_state(self) -> None:
        return None

    def async_write_ha_state(self) -> None:
        return None

    def async_on_remove(self, func: Callable[[], Any]) -> None:
        self._on_remove = func


class ClimateEntityFeature:  # pragma: no cover - placeholder container
    TARGET_TEMPERATURE = 1


class HVACMode(str, enum.Enum):  # pragma: no cover - simple enum
    OFF = "off"
    HEAT = "heat"
    AUTO = "auto"


class HVACAction(str, enum.Enum):  # pragma: no cover - simple enum
    OFF = "off"
    IDLE = "idle"
    HEATING = "heating"


ha_climate.ClimateEntity = ClimateEntity
ha_climate.ClimateEntityFeature = ClimateEntityFeature
ha_climate.HVACMode = HVACMode
ha_climate.HVACAction = HVACAction
ha_components.climate = ha_climate
sys.modules["homeassistant.components.climate"] = ha_climate

ha_const = types.ModuleType("homeassistant.const")
ha_const.ATTR_TEMPERATURE = "temperature"


class UnitOfTemperature:  # pragma: no cover - minimal
    CELSIUS = "C"


ha_const.UnitOfTemperature = UnitOfTemperature
sys.modules["homeassistant.const"] = ha_const

ha_entity = types.ModuleType("homeassistant.helpers.entity")


class DeviceInfo(dict):  # pragma: no cover - mapping subclass
    pass


ha_entity.DeviceInfo = DeviceInfo
sys.modules["homeassistant.helpers.entity"] = ha_entity

ha_helpers = types.ModuleType("homeassistant.helpers")
sys.modules["homeassistant.helpers"] = ha_helpers

ha_entity_platform = types.ModuleType("homeassistant.helpers.entity_platform")


class EntityPlatform:  # pragma: no cover - captures registrations
    def __init__(self) -> None:
        self.registered: list[tuple[str, Any, Callable[..., Any]]] = []

    def async_register_entity_service(
        self, name: str, schema: Any, func: Callable[..., Any]
    ) -> None:
        self.registered.append((name, schema, func))


_current_platform = EntityPlatform()


def async_get_current_platform() -> EntityPlatform:  # pragma: no cover
    return _current_platform


def _set_current_platform(platform: EntityPlatform) -> None:  # pragma: no cover
    global _current_platform
    _current_platform = platform


ha_entity_platform.EntityPlatform = EntityPlatform
ha_entity_platform.async_get_current_platform = async_get_current_platform
ha_entity_platform._set_current_platform = _set_current_platform
sys.modules["homeassistant.helpers.entity_platform"] = ha_entity_platform

ha_update_coordinator = types.ModuleType("homeassistant.helpers.update_coordinator")


class CoordinatorEntity:  # pragma: no cover - lightweight base
    def __init__(self, coordinator) -> None:
        self.coordinator = coordinator
        self.hass = getattr(coordinator, "hass", None)
        self._on_remove: Callable[[], Any] | None = None

    async def async_added_to_hass(self) -> None:
        return None

    async def async_will_remove_from_hass(self) -> None:
        if self._on_remove:
            self._on_remove()

    def schedule_update_ha_state(self) -> None:
        return None

    def async_write_ha_state(self) -> None:
        return None

    def async_on_remove(self, func: Callable[[], Any]) -> None:
        self._on_remove = func


ha_update_coordinator.CoordinatorEntity = CoordinatorEntity
sys.modules["homeassistant.helpers.update_coordinator"] = ha_update_coordinator
ha_helpers.update_coordinator = ha_update_coordinator

ha_dispatcher = types.ModuleType("homeassistant.helpers.dispatcher")
_dispatch_map: dict[str, list[Callable[[dict[str, Any]], None]]] = {}


def async_dispatcher_connect(
    _hass: HomeAssistant, signal: str, func: Callable[[dict[str, Any]], None]
) -> Callable[[], None]:  # pragma: no cover
    _dispatch_map.setdefault(signal, []).append(func)

    def _unsub() -> None:
        callbacks = _dispatch_map.get(signal, [])
        if func in callbacks:
            callbacks.remove(func)

    return _unsub


def dispatcher_send(signal: str, payload: dict[str, Any]) -> None:  # pragma: no cover
    for func in list(_dispatch_map.get(signal, [])):
        func(payload)


ha_dispatcher.async_dispatcher_connect = async_dispatcher_connect
ha_dispatcher.dispatcher_send = dispatcher_send
sys.modules["homeassistant.helpers.dispatcher"] = ha_dispatcher

ha_dt = types.ModuleType("homeassistant.util.dt")
ha_dt.NOW = dt.datetime(2024, 1, 1, 0, 0, tzinfo=dt.timezone.utc)


def now() -> dt.datetime:  # pragma: no cover - deterministic clock
    return ha_dt.NOW


ha_dt.now = now
ha_util = types.ModuleType("homeassistant.util")
ha_util.dt = ha_dt
sys.modules["homeassistant.util"] = ha_util
sys.modules["homeassistant.util.dt"] = ha_dt

vol = types.ModuleType("voluptuous")


def _identity(value: Any) -> Any:
    return value


def Required(_key: Any) -> Callable[[Any], Any]:  # pragma: no cover
    def _req(value: Any) -> Any:
        return value

    return _req


def Optional(_key: Any) -> Callable[[Any], Any]:  # pragma: no cover
    def _opt(value: Any) -> Any:
        return value

    return _opt


def In(container: list[Any]) -> Callable[[Any], Any]:  # pragma: no cover
    def _validator(value: Any) -> Any:
        if value not in container:
            raise ValueError("value not allowed")
        return value

    return _validator


def Coerce(type_: Callable[[Any], Any]) -> Callable[[Any], Any]:  # pragma: no cover
    def _validator(value: Any) -> Any:
        return type_(value)

    return _validator


def Length(*_args: Any, **_kwargs: Any) -> Callable[[Any], Any]:  # pragma: no cover
    return _identity


def All(*validators: Callable[[Any], Any]) -> Callable[[Any], Any]:  # pragma: no cover
    def _validator(value: Any) -> Any:
        result = value
        for validator in validators:
            result = validator(result)
        return result

    return _validator


vol.Required = Required
vol.Optional = Optional
vol.In = In
vol.Coerce = Coerce
vol.Length = Length
vol.All = All
sys.modules["voluptuous"] = vol


def _reset_stubs() -> None:
    sys.modules["homeassistant"] = ha_root
    sys.modules["homeassistant.core"] = ha_core
    sys.modules["homeassistant.components"] = ha_components
    sys.modules["homeassistant.components.climate"] = ha_climate
    sys.modules["homeassistant.const"] = ha_const
    sys.modules["homeassistant.helpers"] = ha_helpers
    sys.modules["homeassistant.helpers.entity"] = ha_entity
    sys.modules["homeassistant.helpers.entity_platform"] = ha_entity_platform
    sys.modules["homeassistant.helpers.update_coordinator"] = ha_update_coordinator
    sys.modules["homeassistant.helpers.dispatcher"] = ha_dispatcher
    sys.modules["homeassistant.util"] = ha_util
    sys.modules["homeassistant.util.dt"] = ha_dt
    sys.modules["voluptuous"] = vol

# -------------------- Load the real climate module --------------------

CLIMATE_PATH = (
    Path(__file__).resolve().parents[1]
    / "custom_components"
    / "termoweb"
    / "climate.py"
)
package = "custom_components.termoweb"

sys.modules.setdefault("custom_components", types.ModuleType("custom_components"))
termoweb_pkg = types.ModuleType(package)
termoweb_pkg.__path__ = [str(CLIMATE_PATH.parent)]
sys.modules[package] = termoweb_pkg

spec = importlib.util.spec_from_file_location(f"{package}.climate", CLIMATE_PATH)
climate_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[f"{package}.climate"] = climate_module
spec.loader.exec_module(climate_module)

TermoWebHeater = climate_module.TermoWebHeater
async_setup_entry = climate_module.async_setup_entry

const_module = importlib.import_module(f"{package}.const")
signal_ws_data = const_module.signal_ws_data
DOMAIN = const_module.DOMAIN

# -------------------- Helpers for tests --------------------


class FakeCoordinator:
    def __init__(self, hass: HomeAssistant, data: dict[str, Any]) -> None:
        self.hass = hass
        self.data = data
        self.async_request_refresh: AsyncMock = AsyncMock()

    def async_add_listener(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def test_async_setup_entry_creates_entities() -> None:
    async def _run() -> None:
        _reset_stubs()
        hass = HomeAssistant()
        entry_id = "entry"
        dev_id = "dev1"
        addrs = ["A1", "B2"]
        nodes = {
            "nodes": [
                {"type": "htr", "addr": "A1", "name": " Living Room "},
                {"type": "HTR", "addr": "B2"},
                {"type": "other", "addr": "X"},
            ]
        }
        coordinator_data = {dev_id: {"nodes": nodes, "htr": {"settings": {}}}}
        coordinator = FakeCoordinator(hass, coordinator_data)

        hass.data = {
            DOMAIN: {
                entry_id: {
                    "coordinator": coordinator,
                    "dev_id": dev_id,
                    "client": AsyncMock(),
                    "nodes": nodes,
                    "htr_addrs": addrs,
                }
            }
        }

        added: list[TermoWebHeater] = []

        def _async_add_entities(entities: list[TermoWebHeater]) -> None:
            added.extend(entities)

        platform = EntityPlatform()
        ha_entity_platform._set_current_platform(platform)

        entry = types.SimpleNamespace(entry_id=entry_id)
        await async_setup_entry(hass, entry, _async_add_entities)

        assert len(added) == 2
        assert all(isinstance(entity, TermoWebHeater) for entity in added)
        names = {entity._addr: entity._attr_name for entity in added}
        assert names["A1"] == "Living Room"
        assert names["B2"] == "Heater B2"

        registered = [name for name, _, _ in platform.registered]
        assert registered == ["set_schedule", "set_preset_temperatures"]

    asyncio.run(_run())


def test_heater_properties_and_ws_update() -> None:
    async def _run() -> None:
        _reset_stubs()
        from homeassistant.helpers.dispatcher import dispatcher_send
        from homeassistant.components.climate import HVACAction, HVACMode

        hass = HomeAssistant()
        entry_id = "entry"
        dev_id = "dev1"
        addr = "A1"
        prog: list[int] = [2] + [1] * 167
        settings = {
            "mode": "manual",
            "state": "heating",
            "mtemp": "19.5",
            "stemp": "21.0",
            "ptemp": ["15.0", "18.0", "21.0"],
            "prog": prog,
            "units": "C",
            "max_power": 1200,
        }
        coordinator_data = {
            dev_id: {
                "nodes": {"nodes": []},
                "htr": {"settings": {addr: settings}},
            }
        }
        coordinator = FakeCoordinator(hass, coordinator_data)
        hass.data = {
            DOMAIN: {
                entry_id: {
                    "client": AsyncMock(),
                    "coordinator": coordinator,
                    "dev_id": dev_id,
                }
            }
        }

        ha_dt.NOW = dt.datetime(2024, 1, 1, 0, 0, tzinfo=dt.timezone.utc)

        heater = TermoWebHeater(coordinator, entry_id, dev_id, addr, "Living")
        assert heater.hass is hass

        await heater.async_added_to_hass()

        assert heater.should_poll is False
        assert heater.available is True
        assert heater.hvac_mode == HVACMode.HEAT
        assert heater.hvac_action == HVACAction.HEATING
        assert heater.current_temperature == pytest.approx(19.5)
        assert heater.target_temperature == pytest.approx(21.0)
        assert heater.icon == "mdi:radiator"

        attrs = heater.extra_state_attributes
        assert attrs["dev_id"] == dev_id
        assert attrs["addr"] == addr
        assert attrs["ptemp"] == ["15.0", "18.0", "21.0"]
        assert attrs["prog"] == prog
        assert attrs["program_slot"] == "day"
        assert attrs["program_setpoint"] == pytest.approx(21.0)

        heater.schedule_update_ha_state = MagicMock()
        dispatcher_send(signal_ws_data(entry_id), {"dev_id": dev_id, "addr": addr})
        heater.schedule_update_ha_state.assert_called_once()

    asyncio.run(_run())


def test_heater_write_paths_and_errors(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def _run() -> None:
        _reset_stubs()
        from homeassistant.const import ATTR_TEMPERATURE
        from homeassistant.components.climate import HVACMode

        hass = HomeAssistant()
        entry_id = "entry"
        dev_id = "dev1"
        addr = "A1"
        base_prog: list[int] = [0, 1, 2] * 56
        settings = {
            "mode": "manual",
            "state": "heating",
            "mtemp": "19.5",
            "stemp": "21.0",
            "ptemp": ["15.0", "18.0", "21.0"],
            "prog": list(base_prog),
            "units": "C",
            "max_power": 1200,
        }
        coordinator_data = {
            dev_id: {
                "nodes": {"nodes": []},
                "htr": {"settings": {addr: settings}},
            }
        }

        coordinator = FakeCoordinator(hass, coordinator_data)
        client = AsyncMock()
        hass.data = {
            DOMAIN: {
                entry_id: {
                    "client": client,
                    "coordinator": coordinator,
                    "dev_id": dev_id,
                }
            }
        }

        heater = TermoWebHeater(coordinator, entry_id, dev_id, addr, "Heater")
        await heater.async_added_to_hass()

        fallback_waiters: Deque[asyncio.Future[None]] = deque()

        async def fake_sleep(delay: float) -> None:
            if delay == climate_module._WRITE_DEBOUNCE:
                return None
            if delay == climate_module._WS_ECHO_FALLBACK_REFRESH:
                loop = asyncio.get_running_loop()
                fut: asyncio.Future[None] = loop.create_future()
                fallback_waiters.append(fut)
                await fut
                return None
            return None

        real_create_task = asyncio.create_task
        created_tasks: list[asyncio.Task[Any]] = []

        def track_create_task(
            coro: Coroutine[Any, Any, Any], *, name: str | None = None
        ) -> asyncio.Task[Any]:
            task = real_create_task(coro, name=name)
            created_tasks.append(task)
            return task

        async def _pop_waiter() -> asyncio.Future[None]:
            for _ in range(10):
                if fallback_waiters:
                    return fallback_waiters.popleft()
                await asyncio.get_running_loop().run_in_executor(None, lambda: None)
            raise AssertionError("fallback waiter not created")

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)
        monkeypatch.setattr(asyncio, "create_task", track_create_task)

        caplog.set_level(logging.ERROR)

        # -------------------- async_set_schedule (valid) --------------------
        await heater.async_set_schedule(list(base_prog))
        call = client.set_htr_settings.await_args
        assert call.args == (dev_id, addr)
        assert call.kwargs["prog"] == list(base_prog)
        assert call.kwargs["units"] == "C"

        settings_after = coordinator.data[dev_id]["htr"]["settings"][addr]
        assert settings_after["prog"] == list(base_prog)

        assert heater._refresh_fallback is not None
        waiter = await _pop_waiter()
        assert coordinator.async_request_refresh.await_count == 0
        waiter.set_result(None)
        await heater._refresh_fallback
        assert coordinator.async_request_refresh.await_count == 1

        client.set_htr_settings.reset_mock()
        coordinator.async_request_refresh.reset_mock()

        # -------------------- async_set_schedule (invalid) --------------------
        caplog.clear()
        await heater.async_set_schedule([0, 1])
        assert client.set_htr_settings.await_count == 0
        assert "Invalid prog length" in caplog.text
        assert not fallback_waiters

        # -------------------- async_set_preset_temperatures (valid) --------------------
        caplog.clear()
        preset_payload = [18.5, 19.5, 20.5]
        await heater.async_set_preset_temperatures(ptemp=preset_payload)
        call = client.set_htr_settings.await_args
        assert call.args == (dev_id, addr)
        assert call.kwargs["ptemp"] == preset_payload
        assert call.kwargs["units"] == "C"

        settings_after = coordinator.data[dev_id]["htr"]["settings"][addr]
        assert settings_after["ptemp"] == ["18.5", "19.5", "20.5"]

        waiter = await _pop_waiter()
        assert coordinator.async_request_refresh.await_count == 0
        waiter.set_result(None)
        await heater._refresh_fallback
        assert coordinator.async_request_refresh.await_count == 1

        client.set_htr_settings.reset_mock()
        coordinator.async_request_refresh.reset_mock()

        # -------------------- async_set_preset_temperatures (invalid) --------------------
        caplog.clear()
        await heater.async_set_preset_temperatures(ptemp=[18.0, 19.0])
        assert client.set_htr_settings.await_count == 0
        assert "Invalid ptemp length" in caplog.text
        assert not fallback_waiters

        # -------------------- async_set_temperature --------------------
        caplog.clear()
        client.set_htr_settings.reset_mock()
        await heater.async_set_temperature(**{ATTR_TEMPERATURE: 23.7})
        assert heater._write_task is not None
        await heater._write_task
        call = client.set_htr_settings.await_args
        assert call.args == (dev_id, addr)
        assert call.kwargs["mode"] == "manual"
        assert call.kwargs["stemp"] == pytest.approx(23.7)
        assert call.kwargs["units"] == "C"

        waiter = await _pop_waiter()
        waiter.set_result(None)
        await heater._refresh_fallback
        assert coordinator.async_request_refresh.await_count == 1

        client.set_htr_settings.reset_mock()
        coordinator.async_request_refresh.reset_mock()

        # -------------------- async_set_hvac_mode (AUTO) --------------------
        await heater.async_set_hvac_mode(HVACMode.AUTO)
        assert heater._write_task is not None
        await heater._write_task
        call = client.set_htr_settings.await_args
        assert call.kwargs["mode"] == "auto"
        assert call.kwargs["stemp"] is None

        waiter = await _pop_waiter()
        waiter.set_result(None)
        await heater._refresh_fallback
        assert coordinator.async_request_refresh.await_count == 1

        client.set_htr_settings.reset_mock()
        coordinator.async_request_refresh.reset_mock()

        # -------------------- async_set_hvac_mode (OFF) --------------------
        await heater.async_set_hvac_mode(HVACMode.OFF)
        assert heater._write_task is not None
        await heater._write_task
        call = client.set_htr_settings.await_args
        assert call.kwargs["mode"] == "off"

        waiter = await _pop_waiter()
        waiter.set_result(None)
        await heater._refresh_fallback
        assert coordinator.async_request_refresh.await_count == 1

        client.set_htr_settings.reset_mock()
        coordinator.async_request_refresh.reset_mock()

        # -------------------- async_set_hvac_mode (HEAT) --------------------
        await heater.async_set_hvac_mode(HVACMode.HEAT)
        assert heater._write_task is not None
        await heater._write_task
        call = client.set_htr_settings.await_args
        assert call.kwargs["mode"] == "manual"
        assert call.kwargs["stemp"] == pytest.approx(21.0)

        waiter = await _pop_waiter()
        waiter.set_result(None)
        await heater._refresh_fallback
        assert coordinator.async_request_refresh.await_count == 1

        coordinator.async_request_refresh.reset_mock()

        # -------------------- _schedule_refresh_fallback behaviour --------------------
        heater._schedule_refresh_fallback()
        task_a = heater._refresh_fallback
        waiter_a = await _pop_waiter()
        heater._schedule_refresh_fallback()
        task_b = heater._refresh_fallback
        waiter_b = await _pop_waiter()

        assert task_a is not None and task_b is not None and task_a is not task_b
        with pytest.raises(asyncio.CancelledError):
            await task_a
        if not waiter_a.done():
            waiter_a.cancel()

        assert coordinator.async_request_refresh.await_count == 0

        waiter_b.set_result(None)
        await task_b
        assert coordinator.async_request_refresh.await_count == 1

        assert created_tasks, "Expected background tasks to be created"

    asyncio.run(_run())
