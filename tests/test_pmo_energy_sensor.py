import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Stub minimal Home Assistant and aiohttp modules for import
ha = types.ModuleType("homeassistant")
ha.__path__ = []
sys.modules["homeassistant"] = ha

ha_core = types.ModuleType("homeassistant.core")
ha_core.HomeAssistant = type("HomeAssistant", (), {})
ha_core.callback = lambda func: func
sys.modules["homeassistant.core"] = ha_core
ha.core = ha_core

ha_helpers = types.ModuleType("homeassistant.helpers")
ha_helpers.__path__ = []
sys.modules["homeassistant.helpers"] = ha_helpers

ha_uc = types.ModuleType("homeassistant.helpers.update_coordinator")
sys.modules["homeassistant.helpers.update_coordinator"] = ha_uc


class DataUpdateCoordinator:  # pragma: no cover - minimal stub
    def __init__(self, hass, logger=None, name=None, update_interval=None) -> None:
        self.hass = hass
        self.logger = logger
        self.name = name
        self.update_interval = update_interval
        self.data = None

    def __class_getitem__(cls, item):  # pragma: no cover - minimal stub
        return cls


class UpdateFailed(Exception):
    pass


class CoordinatorEntity:  # pragma: no cover - minimal stub
    def __init__(self, coordinator) -> None:
        self.coordinator = coordinator


aiohttp_stub = types.ModuleType("aiohttp")
aiohttp_stub.ClientError = Exception
sys.modules["aiohttp"] = aiohttp_stub

ha_uc.DataUpdateCoordinator = DataUpdateCoordinator
ha_uc.UpdateFailed = UpdateFailed
ha_uc.CoordinatorEntity = CoordinatorEntity
ha_helpers.update_coordinator = ha_uc

ha_dispatcher = types.ModuleType("homeassistant.helpers.dispatcher")
ha_dispatcher.async_dispatcher_connect = lambda *args, **kwargs: lambda: None
sys.modules["homeassistant.helpers.dispatcher"] = ha_dispatcher
ha_helpers.dispatcher = ha_dispatcher

ha_entity = types.ModuleType("homeassistant.helpers.entity")
ha_entity.DeviceInfo = type("DeviceInfo", (dict,), {})
sys.modules["homeassistant.helpers.entity"] = ha_entity
ha_helpers.entity = ha_entity

ha_components = types.ModuleType("homeassistant.components")
ha_components.__path__ = []
sys.modules["homeassistant.components"] = ha_components

ha_sensor = types.ModuleType("homeassistant.components.sensor")
ha_sensor.SensorEntity = type("SensorEntity", (), {})
ha_sensor.SensorDeviceClass = type("SensorDeviceClass", (), {"ENERGY": "energy", "TEMPERATURE": "temperature"})
ha_sensor.SensorStateClass = type("SensorStateClass", (), {"TOTAL_INCREASING": "total_increasing", "MEASUREMENT": "measurement"})
sys.modules["homeassistant.components.sensor"] = ha_sensor
ha_components.sensor = ha_sensor

ha_const = types.ModuleType("homeassistant.const")
ha_const.UnitOfTemperature = type("UnitOfTemperature", (), {"CELSIUS": "°C"})
sys.modules["homeassistant.const"] = ha_const
ha.const = ha_const

# Load integration modules dynamically
PACKAGE = "custom_components.termoweb"
COMP_PATH = Path(__file__).resolve().parents[1] / "custom_components" / "termoweb"

sys.modules.setdefault("custom_components", types.ModuleType("custom_components"))
termoweb_pkg = types.ModuleType(PACKAGE)
termoweb_pkg.__path__ = [str(COMP_PATH)]
sys.modules[PACKAGE] = termoweb_pkg

spec_coord = importlib.util.spec_from_file_location(f"{PACKAGE}.coordinator", COMP_PATH / "coordinator.py")
coordinator_module = importlib.util.module_from_spec(spec_coord)
assert spec_coord.loader is not None
sys.modules[f"{PACKAGE}.coordinator"] = coordinator_module
spec_coord.loader.exec_module(coordinator_module)
TermoWebPmoEnergyCoordinator = coordinator_module.TermoWebPmoEnergyCoordinator

spec_sensor = importlib.util.spec_from_file_location(f"{PACKAGE}.sensor", COMP_PATH / "sensor.py")
sensor_module = importlib.util.module_from_spec(spec_sensor)
assert spec_sensor.loader is not None
sys.modules[f"{PACKAGE}.sensor"] = sensor_module
spec_sensor.loader.exec_module(sensor_module)
TermoWebPmoEnergyTotal = sensor_module.TermoWebPmoEnergyTotal


def test_wh_to_kwh_conversion() -> None:
    async def run() -> None:
        client = AsyncMock()
        client.list_devices.return_value = [{"dev_id": "1"}]
        client.get_nodes.return_value = {"nodes": [{"type": "pmo", "addr": 1}]}
        client.get_pmo_samples.return_value = {"samples": [{"counter": "12345"}]}

        coordinator = TermoWebPmoEnergyCoordinator(MagicMock(), client)
        coordinator.data = await coordinator._async_update_data()

        sensor = TermoWebPmoEnergyTotal(coordinator, "entry", "1", "1", "name", "uid")
        assert sensor.native_value == pytest.approx(12.345)

    asyncio.run(run())


def test_missing_response() -> None:
    async def run() -> None:
        client = AsyncMock()
        client.list_devices.return_value = [{"dev_id": "1"}]
        client.get_nodes.return_value = {"nodes": [{"type": "pmo", "addr": 1}]}
        client.get_pmo_samples.side_effect = Exception("404")

        coordinator = TermoWebPmoEnergyCoordinator(MagicMock(), client)
        coordinator.data = await coordinator._async_update_data()
        assert coordinator.data["1"]["pmo"]["energy_total"] == {}

    asyncio.run(run())


def test_counter_reset_detection() -> None:
    async def run() -> None:
        client = AsyncMock()
        client.list_devices.return_value = [{"dev_id": "1"}]
        client.get_nodes.return_value = {"nodes": [{"type": "pmo", "addr": 1}]}

        client.get_pmo_samples.return_value = {"samples": [{"counter": "1000"}]}
        coordinator = TermoWebPmoEnergyCoordinator(MagicMock(), client)
        coordinator.data = await coordinator._async_update_data()
        assert coordinator.data["1"]["pmo"]["energy_total"]["1"] == pytest.approx(1.0)

        client.get_pmo_samples.return_value = {"samples": [{"counter": "100"}]}
        coordinator.data = await coordinator._async_update_data()
        assert coordinator.data["1"]["pmo"]["energy_total"]["1"] == pytest.approx(0.1)

    asyncio.run(run())
