# ruff: noqa: D100, D101, D102, D103, D105, D107, INP001
from __future__ import annotations

import asyncio
from contextlib import suppress
import importlib.util
from pathlib import Path
import sys
import types
from unittest.mock import AsyncMock, MagicMock

# --- Minimal Home Assistant stubs -------------------------------------------------

aiohttp_stub = types.ModuleType("aiohttp")


class ClientError(Exception):
    """Placeholder for aiohttp.ClientError."""


aiohttp_stub.ClientError = ClientError
sys.modules["aiohttp"] = aiohttp_stub

ha_core = types.ModuleType("homeassistant.core")


class HomeAssistant:
    """Very small Home Assistant stand-in for tests."""

    def __init__(self) -> None:
        self.data: dict = {}


def callback(func):
    return func


ha_core.HomeAssistant = HomeAssistant
ha_core.callback = callback
sys.modules["homeassistant"] = types.ModuleType("homeassistant")
sys.modules["homeassistant.core"] = ha_core

ha_helpers = types.ModuleType("homeassistant.helpers")
ha_uc = types.ModuleType("homeassistant.helpers.update_coordinator")


class UpdateFailed(Exception):
    """Update error raised by the coordinator."""


class DataUpdateCoordinator:
    """Simplified DataUpdateCoordinator used by coordinator module."""

    def __init__(self, hass, *, logger=None, name=None, update_interval=None) -> None:
        self.hass = hass
        self.logger = logger
        self.name = name
        self.update_interval = update_interval
        self.data: dict | None = None

    async def async_refresh(self) -> None:  # pragma: no cover - not used here
        self.data = await self._async_update_data()

    async def async_config_entry_first_refresh(self) -> None:  # pragma: no cover
        await self.async_refresh()

    def async_add_listener(self, *_args, **_kwargs) -> None:  # pragma: no cover
        return None

    @classmethod
    def __class_getitem__(cls, _item):  # pragma: no cover - typing helper
        return cls


class CoordinatorEntity:
    """Minimal coordinator entity base for entities under test."""

    def __init__(self, coordinator) -> None:
        self.coordinator = coordinator
        self.hass = coordinator.hass
        self._remove_callbacks: list = []

    async def async_added_to_hass(self) -> None:  # pragma: no cover - noop
        return None

    def async_on_remove(self, func) -> None:
        self._remove_callbacks.append(func)

    def schedule_update_ha_state(self) -> None:  # pragma: no cover - noop
        return None

    @classmethod
    def __class_getitem__(cls, _item):  # pragma: no cover - typing helper
        return cls


ha_uc.UpdateFailed = UpdateFailed
ha_uc.DataUpdateCoordinator = DataUpdateCoordinator
ha_uc.CoordinatorEntity = CoordinatorEntity
ha_helpers.update_coordinator = ha_uc
sys.modules["homeassistant.helpers"] = ha_helpers
sys.modules["homeassistant.helpers.update_coordinator"] = ha_uc

ha_dispatcher = types.ModuleType("homeassistant.helpers.dispatcher")
_dispatchers: dict[str, list] = {}


def async_dispatcher_connect(_hass, signal: str, callback):
    callbacks = _dispatchers.setdefault(signal, [])
    callbacks.append(callback)

    def _remove() -> None:
        with suppress(ValueError):  # pragma: no cover - best effort cleanup
            callbacks.remove(callback)

    return _remove


def dispatcher_send(signal: str, payload: dict) -> None:
    for cb in list(_dispatchers.get(signal, [])):
        cb(payload)


ha_dispatcher.async_dispatcher_connect = async_dispatcher_connect
sys.modules["homeassistant.helpers.dispatcher"] = ha_dispatcher

ha_entity = types.ModuleType("homeassistant.helpers.entity")


class DeviceInfo(dict):
    """Dictionary-based DeviceInfo mimic."""


ha_entity.DeviceInfo = DeviceInfo
sys.modules["homeassistant.helpers.entity"] = ha_entity

ha_binary_sensor = types.ModuleType("homeassistant.components.binary_sensor")


class BinarySensorEntity:
    """Minimal binary sensor entity."""

    def __init__(self) -> None:
        self.hass = None

    async def async_added_to_hass(self) -> None:  # pragma: no cover - noop
        return None

    def async_on_remove(self, func) -> None:
        self._on_remove = func

    def schedule_update_ha_state(self) -> None:  # pragma: no cover - noop
        return None


class BinarySensorDeviceClass:
    CONNECTIVITY = "connectivity"


ha_binary_sensor.BinarySensorEntity = BinarySensorEntity
ha_binary_sensor.BinarySensorDeviceClass = BinarySensorDeviceClass
sys.modules.setdefault("homeassistant.components", types.ModuleType("homeassistant.components"))
sys.modules["homeassistant.components.binary_sensor"] = ha_binary_sensor

ha_button = types.ModuleType("homeassistant.components.button")


class ButtonEntity:
    """Minimal button entity."""

    def __init__(self) -> None:
        self.hass = None

    async def async_added_to_hass(self) -> None:  # pragma: no cover - noop
        return None


ha_button.ButtonEntity = ButtonEntity
sys.modules["homeassistant.components.button"] = ha_button

# --- Load integration modules -----------------------------------------------------

BASE_PATH = Path(__file__).resolve().parents[1] / "custom_components" / "termoweb"
PACKAGE = "custom_components.termoweb"

sys.modules.setdefault("custom_components", types.ModuleType("custom_components"))
termoweb_pkg = types.ModuleType(PACKAGE)
termoweb_pkg.__path__ = [str(BASE_PATH)]
sys.modules[PACKAGE] = termoweb_pkg


def _load_module(module: str, path: Path):
    spec = importlib.util.spec_from_file_location(module, path)
    module_obj = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module] = module_obj
    spec.loader.exec_module(module_obj)
    return module_obj


const_module = _load_module(f"{PACKAGE}.const", BASE_PATH / "const.py")
_load_module(f"{PACKAGE}.coordinator", BASE_PATH / "coordinator.py")
binary_sensor_module = _load_module(f"{PACKAGE}.binary_sensor", BASE_PATH / "binary_sensor.py")
button_module = _load_module(f"{PACKAGE}.button", BASE_PATH / "button.py")

DOMAIN = const_module.DOMAIN
signal_ws_status = const_module.signal_ws_status
TermoWebDeviceOnlineBinarySensor = binary_sensor_module.TermoWebDeviceOnlineBinarySensor
async_setup_binary_sensor_entry = binary_sensor_module.async_setup_entry
TermoWebRefreshButton = button_module.TermoWebRefreshButton


# --- Tests -----------------------------------------------------------------------


def test_binary_sensor_setup_and_dispatch() -> None:
    async def _run() -> None:
        hass = HomeAssistant()
        entry = types.SimpleNamespace(entry_id="entry-1")
        dev_id = "device-123"

        coordinator = types.SimpleNamespace(
            hass=hass,
            data={
                dev_id: {
                    "name": "Living Room",  # attributes
                    "connected": True,
                    "raw": {"model": "TW-GW"},
                }
            },
        )

        hass.data = {
            DOMAIN: {
                entry.entry_id: {
                    "coordinator": coordinator,
                    "dev_id": dev_id,
                    "version": "2.1.0",
                    "ws_state": {
                        dev_id: {
                            "status": "healthy",
                            "last_event_at": "2024-05-01T12:00:00Z",
                            "healthy_minutes": 42,
                        }
                    },
                }
            }
        }

        added: list = []

        def _add_entities(entities):
            added.extend(entities)

        await async_setup_binary_sensor_entry(hass, entry, _add_entities)

        assert len(added) == 1
        entity = added[0]
        assert isinstance(entity, TermoWebDeviceOnlineBinarySensor)

        entity.hass = hass
        await entity.async_added_to_hass()

        assert entity.is_on is True

        info = entity.device_info
        assert info["identifiers"] == {(DOMAIN, dev_id)}
        assert info["manufacturer"] == "TermoWeb"
        assert info["name"] == "TermoWeb Gateway"
        assert info["model"] == "TW-GW"
        assert info["sw_version"] == "2.1.0"

        attrs = entity.extra_state_attributes
        assert attrs == {
            "dev_id": dev_id,
            "name": "Living Room",
            "connected": True,
            "ws_status": "healthy",
            "ws_last_event_at": "2024-05-01T12:00:00Z",
            "ws_healthy_minutes": 42,
            "raw": {"model": "TW-GW"},
        }

        entity.schedule_update_ha_state = MagicMock()
        dispatcher_send(signal_ws_status(entry.entry_id), {"dev_id": dev_id})
        entity.schedule_update_ha_state.assert_called_once_with()

    asyncio.run(_run())


def test_refresh_button_device_info_and_press() -> None:
    async def _run() -> None:
        hass = HomeAssistant()
        dev_id = "device-123"
        coordinator = types.SimpleNamespace(
            hass=hass,
            async_request_refresh=AsyncMock(),
        )

        button = TermoWebRefreshButton(coordinator, dev_id)
        button.hass = hass

        info = button.device_info
        assert info["identifiers"] == {(DOMAIN, dev_id)}
        assert info["manufacturer"] == "TermoWeb"
        assert info["name"] == "TermoWeb Gateway"
        assert info["model"] == "Gateway/Controller"
        assert info["configuration_url"] == "https://control.termoweb.net"

        await button.async_press()
        coordinator.async_request_refresh.assert_awaited_once()

    asyncio.run(_run())
