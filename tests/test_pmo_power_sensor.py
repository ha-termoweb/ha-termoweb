from __future__ import annotations

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import asyncio
import sys
import types
from pathlib import Path


# Provide minimal Home Assistant stubs for module imports
ha_pkg = types.ModuleType("homeassistant")
core_mod = types.ModuleType("homeassistant.core")


class HomeAssistant:  # pragma: no cover - simple stub
    def __init__(self) -> None:
        self.data: Dict[str, Any] = {}
        self.loop = asyncio.get_event_loop()

    def async_create_task(self, coro):  # pragma: no cover - pass-through
        return self.loop.create_task(coro)


def callback(func):  # pragma: no cover - identity decorator
    return func


core_mod.HomeAssistant = HomeAssistant
core_mod.callback = callback

dispatcher_mod = types.ModuleType("homeassistant.helpers.dispatcher")


def async_dispatcher_send(hass, signal, payload) -> None:  # pragma: no cover
    for cb in hass.data.setdefault("_signals", {}).get(signal, []):
        cb(payload)


def async_dispatcher_connect(hass, signal, cb):  # pragma: no cover
    hass.data.setdefault("_signals", {}).setdefault(signal, []).append(cb)

    def _unsub() -> None:
        hass.data.get("_signals", {}).get(signal, []).remove(cb)

    return _unsub


dispatcher_mod.async_dispatcher_send = async_dispatcher_send
dispatcher_mod.async_dispatcher_connect = async_dispatcher_connect

update_mod = types.ModuleType("homeassistant.helpers.update_coordinator")


class UpdateFailed(Exception):  # pragma: no cover - simple placeholder
    pass


class DataUpdateCoordinator:  # pragma: no cover - minimal coordinator
    def __init__(self, hass, *, logger=None, name=None, update_interval=None) -> None:
        self.hass = hass
        self.logger = logger
        self.name = name
        self.update_interval = update_interval
        self.data: Dict[str, Any] | None = None
        self._listeners: list = []

    __class_getitem__ = classmethod(lambda cls, _item: cls)

    async def async_config_entry_first_refresh(self) -> None:
        self.data = await self._async_update_data()

    def async_set_updated_data(self, data) -> None:
        self.data = data
        for cb in list(self._listeners):
            cb()

    def async_add_listener(self, cb) -> None:
        self._listeners.append(cb)


class CoordinatorEntity:  # pragma: no cover - simple base
    def __init__(self, coordinator) -> None:
        self.coordinator = coordinator

    async def async_added_to_hass(self) -> None:  # pragma: no cover - no-op
        return None

    def async_on_remove(self, _func) -> None:  # pragma: no cover - no-op
        return None


update_mod.DataUpdateCoordinator = DataUpdateCoordinator
update_mod.UpdateFailed = UpdateFailed
update_mod.CoordinatorEntity = CoordinatorEntity

entity_mod = types.ModuleType("homeassistant.helpers.entity")


class DeviceInfo(dict):  # pragma: no cover - dict subclass
    pass


entity_mod.DeviceInfo = DeviceInfo

sensor_mod = types.ModuleType("homeassistant.components.sensor")


class SensorEntity:  # pragma: no cover - empty base
    pass


class SensorDeviceClass:  # pragma: no cover - enum placeholder
    POWER = "power"
    TEMPERATURE = "temperature"
    ENERGY = "energy"


class SensorStateClass:  # pragma: no cover - enum placeholder
    MEASUREMENT = "measurement"
    TOTAL_INCREASING = "total_increasing"


sensor_mod.SensorEntity = SensorEntity
sensor_mod.SensorDeviceClass = SensorDeviceClass
sensor_mod.SensorStateClass = SensorStateClass

const_mod = types.ModuleType("homeassistant.const")


class UnitOfTemperature:  # pragma: no cover - placeholder enum
    CELSIUS = "C"


const_mod.UnitOfTemperature = UnitOfTemperature

helpers_pkg = types.ModuleType("homeassistant.helpers")
helpers_pkg.__path__ = []  # pragma: no cover
components_pkg = types.ModuleType("homeassistant.components")
components_pkg.__path__ = []  # pragma: no cover

sys.modules.setdefault("homeassistant", ha_pkg)
sys.modules["homeassistant.core"] = core_mod
sys.modules["homeassistant.helpers"] = helpers_pkg
sys.modules["homeassistant.helpers.dispatcher"] = dispatcher_mod
sys.modules["homeassistant.helpers.update_coordinator"] = update_mod
sys.modules["homeassistant.helpers.entity"] = entity_mod
sys.modules["homeassistant.components"] = components_pkg
sys.modules["homeassistant.components.sensor"] = sensor_mod
sys.modules["homeassistant.const"] = const_mod

# Expose custom_components.termoweb package
package_path = Path(__file__).resolve().parents[1] / "custom_components" / "termoweb"
sys.modules.setdefault("custom_components", types.ModuleType("custom_components"))
termoweb_pkg = types.ModuleType("custom_components.termoweb")
termoweb_pkg.__path__ = [str(package_path)]
sys.modules["custom_components.termoweb"] = termoweb_pkg

# Ensure fresh imports for sensor module
sys.modules.pop("custom_components.termoweb.sensor", None)
sys.modules.pop("custom_components.termoweb.coordinator", None)

# Minimal aiohttp stub
aiohttp_stub = types.ModuleType("aiohttp")


class ClientSession:  # pragma: no cover - placeholder
    pass


class ClientTimeout:  # pragma: no cover - placeholder
    def __init__(self, total: int | None = None) -> None:
        self.total = total


class ClientResponseError(Exception):  # pragma: no cover - placeholder
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args)


aiohttp_stub.ClientSession = ClientSession
aiohttp_stub.ClientTimeout = ClientTimeout
aiohttp_stub.ClientResponseError = ClientResponseError
aiohttp_stub.ClientError = Exception
sys.modules.setdefault("aiohttp", aiohttp_stub)

from custom_components.termoweb.api import TermoWebClient  # noqa: E402
from custom_components.termoweb.const import DOMAIN, signal_ws_data  # noqa: E402
from custom_components.termoweb.coordinator import TermoWebPmoPowerCoordinator  # noqa: E402
from custom_components.termoweb import sensor as sensor_mod  # noqa: E402


def test_get_pmo_power_url() -> None:
    async def _run() -> None:
        client = TermoWebClient(MagicMock(), "u", "p")
        client._authed_headers = AsyncMock(return_value={})
        client._request = AsyncMock(return_value={"power": 1})
        await client.get_pmo_power("d1", "2")
        client._request.assert_called_once()
        method, path = client._request.call_args[0][:2]
        assert method == "GET"
        assert path == "/api/v2/devs/d1/pmo/2/power"

    asyncio.run(_run())


def test_ws_event_updates_sensor() -> None:
    async def _run() -> None:
        hass = HomeAssistant()
        base = MagicMock()
        base.data = {
            "dev1": {
                "nodes": {"nodes": [{"type": "pmo", "addr": "1"}]},
                "pmo": {"power": {"1": 0.0}},
            }
        }
        client = MagicMock()
        client.get_pmo_power = AsyncMock(return_value={"power": 5})
        coord = TermoWebPmoPowerCoordinator(hass, client, base, "entry")
        await coord.async_config_entry_first_refresh()
        assert coord.data["dev1"]["pmo"]["power"]["1"] == 5.0

        base.data["dev1"]["pmo"]["power"]["1"] = 7.0
        async_dispatcher_send(
            hass, signal_ws_data("entry"), {"dev_id": "dev1", "addr": "1", "kind": "pmo_power"}
        )
        assert coord.data["dev1"]["pmo"]["power"]["1"] == 7.0

        ent = sensor_mod.TermoWebPmoPower(coord, "entry", "dev1", "1", "Power", "uid")
        ent.hass = hass
        await ent.async_added_to_hass()
        ent.schedule_update_ha_state = MagicMock()
        async_dispatcher_send(
            hass, signal_ws_data("entry"), {"dev_id": "dev1", "addr": "1", "kind": "pmo_power"}
        )
        assert ent.schedule_update_ha_state.call_count == 1

    asyncio.run(_run())


def test_entity_registration() -> None:
    async def _run() -> None:
        hass = HomeAssistant()
        entry = MagicMock()
        entry.entry_id = "e1"
        data: Dict[str, Any] = {
            "client": MagicMock(),
            "coordinator": MagicMock(),
        }
        data["coordinator"].data = {
            "dev1": {
                "nodes": {"nodes": [{"type": "pmo", "addr": "1", "name": "PMO"}]}
            }
        }
        data["pmo_coordinator"] = MagicMock()
        data["pmo_coordinator"].data = {
            "dev1": {"pmo": {"energy_total": {"1": 0.0}}}
        }
        data["client"].get_pmo_power = AsyncMock(return_value={"power": 0})
        data["coordinator"].async_add_listener = MagicMock()
        hass.data.setdefault(DOMAIN, {})[entry.entry_id] = data

        added: list[Any] = []

        def _add(ents):
            added.extend(ents)

        await sensor_mod.async_setup_entry(hass, entry, _add)
        assert any(isinstance(ent, sensor_mod.TermoWebPmoPower) for ent in added)

    asyncio.run(_run())
