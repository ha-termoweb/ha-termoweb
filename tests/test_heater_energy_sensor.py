from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Minimal aiohttp stub
aiohttp_stub = types.ModuleType("aiohttp")


class ClientError(Exception):  # pragma: no cover - placeholder
    pass


aiohttp_stub.ClientError = ClientError
sys.modules["aiohttp"] = aiohttp_stub

# Stub Home Assistant modules
ha_core = types.ModuleType("homeassistant.core")


class HomeAssistant:  # pragma: no cover - placeholder
    pass


def callback(fn):  # pragma: no cover - passthrough
    return fn

ha_core.HomeAssistant = HomeAssistant
ha_core.callback = callback
sys.modules["homeassistant"] = types.ModuleType("homeassistant")
sys.modules["homeassistant.core"] = ha_core

ha_helpers = types.ModuleType("homeassistant.helpers")
uc = types.ModuleType("homeassistant.helpers.update_coordinator")


class UpdateFailed(Exception):  # pragma: no cover - placeholder
    pass


class DataUpdateCoordinator:  # pragma: no cover - minimal stub
    def __init__(self, hass, *, logger=None, name=None, update_interval=None) -> None:
        self.hass = hass
        self.logger = logger
        self.name = name
        self.update_interval = update_interval
        self.data: dict | None = None

    async def async_refresh(self) -> None:
        self.data = await self._async_update_data()

    async def async_config_entry_first_refresh(self) -> None:
        await self.async_refresh()

    def async_add_listener(self, *_args) -> None:
        return None

    @classmethod
    def __class_getitem__(cls, _item) -> type:
        return cls


class CoordinatorEntity:  # pragma: no cover - minimal stub
    def __init__(self, coordinator) -> None:
        self.coordinator = coordinator
        self.hass = coordinator.hass

    async def async_added_to_hass(self) -> None:
        return None

    def schedule_update_ha_state(self) -> None:
        return None


uc.UpdateFailed = UpdateFailed
uc.DataUpdateCoordinator = DataUpdateCoordinator
uc.CoordinatorEntity = CoordinatorEntity
ha_helpers.update_coordinator = uc
sys.modules["homeassistant.helpers"] = ha_helpers
sys.modules["homeassistant.helpers.update_coordinator"] = uc

# Dispatcher stub
ha_dispatcher = types.ModuleType("homeassistant.helpers.dispatcher")
_dispatchers: dict[str, list] = {}


def async_dispatcher_connect(_hass, signal: str, callback) -> None:  # pragma: no cover - minimal
    _dispatchers.setdefault(signal, []).append(callback)
    return lambda: None


def dispatcher_send(signal: str, payload: dict) -> None:
    for cb in list(_dispatchers.get(signal, [])):
        cb(payload)


ha_dispatcher.async_dispatcher_connect = async_dispatcher_connect
sys.modules["homeassistant.helpers.dispatcher"] = ha_dispatcher

# Other HA stubs
ha_const = types.ModuleType("homeassistant.const")


class UnitOfTemperature:  # pragma: no cover - placeholder
    CELSIUS = "C"

ha_const.UnitOfTemperature = UnitOfTemperature
sys.modules["homeassistant.const"] = ha_const

ha_sensor = types.ModuleType("homeassistant.components.sensor")


class SensorEntity:  # pragma: no cover - minimal entity
    def __init__(self) -> None:
        self.hass = None

    async def async_added_to_hass(self) -> None:
        return None

    def schedule_update_ha_state(self) -> None:
        return None

    def async_on_remove(self, func) -> None:  # pragma: no cover - store callback
        self._on_remove = func

    @property
    def device_class(self) -> str | None:
        return getattr(self, "_attr_device_class", None)

    @property
    def state_class(self) -> str | None:
        return getattr(self, "_attr_state_class", None)

    @property
    def native_unit_of_measurement(self) -> str | None:
        return getattr(self, "_attr_native_unit_of_measurement", None)


class SensorDeviceClass:  # pragma: no cover - simple container
    ENERGY = "energy"
    POWER = "power"
    TEMPERATURE = "temp"


class SensorStateClass:  # pragma: no cover - simple container
    MEASUREMENT = "measurement"
    TOTAL_INCREASING = "total_increasing"


ha_sensor.SensorEntity = SensorEntity
ha_sensor.SensorDeviceClass = SensorDeviceClass
ha_sensor.SensorStateClass = SensorStateClass
sys.modules["homeassistant.components.sensor"] = ha_sensor

ha_entity = types.ModuleType("homeassistant.helpers.entity")


class DeviceInfo(dict):  # pragma: no cover - simple mapping
    pass


ha_entity.DeviceInfo = DeviceInfo
sys.modules["homeassistant.helpers.entity"] = ha_entity

# Load coordinator and sensor modules
COORD_PATH = (
    Path(__file__).resolve().parents[1]
    / "custom_components"
    / "termoweb"
    / "coordinator.py"
)
SENSOR_PATH = (
    Path(__file__).resolve().parents[1]
    / "custom_components"
    / "termoweb"
    / "sensor.py"
)
package = "custom_components.termoweb"

sys.modules.setdefault("custom_components", types.ModuleType("custom_components"))
termoweb_pkg = types.ModuleType(package)
termoweb_pkg.__path__ = [str(COORD_PATH.parent)]
sys.modules[package] = termoweb_pkg

spec = importlib.util.spec_from_file_location(f"{package}.coordinator", COORD_PATH)
coord_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[f"{package}.coordinator"] = coord_module
spec.loader.exec_module(coord_module)

spec2 = importlib.util.spec_from_file_location(f"{package}.sensor", SENSOR_PATH)
sensor_module = importlib.util.module_from_spec(spec2)
assert spec2.loader is not None
sys.modules[f"{package}.sensor"] = sensor_module
spec2.loader.exec_module(sensor_module)

TermoWebHeaterEnergyCoordinator = coord_module.TermoWebHeaterEnergyCoordinator
TermoWebHeaterEnergyTotal = sensor_module.TermoWebHeaterEnergyTotal
TermoWebHeaterPower = sensor_module.TermoWebHeaterPower
signal_ws_data = __import__(f"{package}.const", fromlist=["signal_ws_data"]).signal_ws_data


def test_coordinator_and_sensors() -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_htr_samples = AsyncMock(
            side_effect=[
                [{"t": 1000, "counter": "1.0"}],
                [{"t": 1900, "counter": "1.5"}],
            ]
        )

        hass = HomeAssistant()
        coord = TermoWebHeaterEnergyCoordinator(hass, client, "1", ["A"])  # type: ignore[arg-type]

        await coord.async_refresh()
        await coord.async_refresh()

        assert coord.data["1"]["htr"]["energy"]["A"] == 1.5
        power = coord.data["1"]["htr"]["power"]["A"]
        assert power == pytest.approx(2000.0, rel=1e-3)

        energy_sensor = TermoWebHeaterEnergyTotal(coord, "entry", "1", "A", "Energy", "e1")
        power_sensor = TermoWebHeaterPower(coord, "entry", "1", "A", "Power", "p1")
        energy_sensor.hass = power_sensor.hass = hass
        await energy_sensor.async_added_to_hass()
        await power_sensor.async_added_to_hass()

        assert energy_sensor.device_class == SensorDeviceClass.ENERGY
        assert energy_sensor.state_class == SensorStateClass.TOTAL_INCREASING
        assert energy_sensor.native_unit_of_measurement == "kWh"

        assert energy_sensor.native_value == pytest.approx(0.0015)
        assert power_sensor.native_value == pytest.approx(2000.0, rel=1e-3)

        first_value: float = energy_sensor.native_value  # type: ignore[assignment]

        energy_sensor.schedule_update_ha_state = MagicMock()
        power_sensor.schedule_update_ha_state = MagicMock()

        coord.data["1"]["htr"]["energy"]["A"] = 2.0
        coord.data["1"]["htr"]["power"]["A"] = 123.0
        dispatcher_send(signal_ws_data("entry"), {"dev_id": "1", "addr": "A"})

        energy_sensor.schedule_update_ha_state.assert_called_once()
        power_sensor.schedule_update_ha_state.assert_called_once()
        assert energy_sensor.native_value >= first_value
        assert energy_sensor.native_value == pytest.approx(0.002)
        assert power_sensor.native_value == pytest.approx(123.0)

    asyncio.run(_run())
