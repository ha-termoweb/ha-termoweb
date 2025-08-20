from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
import sys
import time as _time
import types
from typing import Any
from unittest.mock import AsyncMock

import pytest

# Minimal aiohttp stub
aiohttp_stub = types.ModuleType("aiohttp")


class ClientError(Exception):  # pragma: no cover - placeholder
    pass


aiohttp_stub.ClientError = ClientError
sys.modules["aiohttp"] = aiohttp_stub

# Stub minimal Home Assistant modules
ha_core = types.ModuleType("homeassistant.core")


class HomeAssistant:  # pragma: no cover - placeholder
    pass


ha_core.HomeAssistant = HomeAssistant
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
        self.data: dict[str, dict[str, Any]] | None = None

    async def async_refresh(self) -> None:
        self.data = await self._async_update_data()

    async def async_config_entry_first_refresh(self) -> None:
        await self.async_refresh()

    async def async_request_refresh(self) -> None:
        await self.async_refresh()

    def async_add_listener(self, *_args) -> None:
        return None

    @classmethod
    def __class_getitem__(cls, _item) -> type:
        return cls


uc.UpdateFailed = UpdateFailed
uc.DataUpdateCoordinator = DataUpdateCoordinator
ha_helpers.update_coordinator = uc
sys.modules["homeassistant.helpers"] = ha_helpers
sys.modules["homeassistant.helpers.update_coordinator"] = uc

# Stub API module
package = "custom_components.termoweb"
api_stub = types.ModuleType(f"{package}.api")


class TermoWebClient:  # pragma: no cover - placeholder
    pass


class TermoWebAuthError(Exception):  # pragma: no cover - placeholder
    pass


class TermoWebRateLimitError(Exception):  # pragma: no cover - placeholder
    pass


api_stub.TermoWebClient = TermoWebClient
api_stub.TermoWebAuthError = TermoWebAuthError
api_stub.TermoWebRateLimitError = TermoWebRateLimitError
api_stub.time = _time
sys.modules[f"{package}.api"] = api_stub

# Dispatcher stub
ha_dispatcher = types.ModuleType("homeassistant.helpers.dispatcher")
_dispatchers: dict[str, list] = {}


def async_dispatcher_connect(_hass, signal: str, callback):  # pragma: no cover
    _dispatchers.setdefault(signal, []).append(callback)
    return lambda: None


def dispatcher_send(signal: str, payload: dict) -> None:
    for cb in list(_dispatchers.get(signal, [])):
        cb(payload)


ha_dispatcher.async_dispatcher_connect = async_dispatcher_connect
ha_dispatcher.dispatcher_send = dispatcher_send
sys.modules["homeassistant.helpers.dispatcher"] = ha_dispatcher

# Load coordinator module
COORD_PATH = (
    Path(__file__).resolve().parents[1] / "custom_components" / "termoweb" / "coordinator.py"
)
sys.modules.setdefault("custom_components", types.ModuleType("custom_components"))
termoweb_pkg = types.ModuleType(package)
termoweb_pkg.__path__ = [str(COORD_PATH.parent)]
sys.modules[package] = termoweb_pkg

spec = importlib.util.spec_from_file_location(f"{package}.coordinator", COORD_PATH)
coord_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[f"{package}.coordinator"] = coord_module
spec.loader.exec_module(coord_module)

TermoWebHeaterEnergyCoordinator = coord_module.TermoWebHeaterEnergyCoordinator
TermoWebCoordinator = coord_module.TermoWebCoordinator
signal_ws_data = __import__(f"{package}.const", fromlist=["signal_ws_data"]).signal_ws_data
HTR_ENERGY_UPDATE_INTERVAL = __import__(
    f"{package}.const", fromlist=["HTR_ENERGY_UPDATE_INTERVAL"]
).HTR_ENERGY_UPDATE_INTERVAL


def test_power_calculation(monkeypatch: pytest.MonkeyPatch) -> None:
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

        fake_time = 1000.0

        def _fake_time() -> float:
            return fake_time

        monkeypatch.setattr(coord_module.time, "time", _fake_time)

        await coord.async_refresh()
        assert coord.data["1"]["htr"]["energy"]["A"] == 1.0
        assert "A" not in coord.data["1"]["htr"]["power"]

        fake_time = 1900.0
        await coord.async_refresh()
        assert coord.data["1"]["htr"]["energy"]["A"] == 1.5
        power = coord.data["1"]["htr"]["power"]["A"]
        assert power == pytest.approx(2000.0, rel=1e-3)

    asyncio.run(_run())


def test_counter_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_htr_samples = AsyncMock(
            side_effect=[
                [{"t": 1000, "counter": "5.0"}],
                [{"t": 1900, "counter": "1.0"}],
            ]
        )

        hass = HomeAssistant()
        coord = TermoWebHeaterEnergyCoordinator(hass, client, "1", ["A"])  # type: ignore[arg-type]

        fake_time = 1000.0

        def _fake_time() -> float:
            return fake_time

        monkeypatch.setattr(coord_module.time, "time", _fake_time)

        await coord.async_refresh()
        fake_time = 1900.0
        await coord.async_refresh()

        assert coord.data["1"]["htr"]["energy"]["A"] == 1.0
        assert "A" not in coord.data["1"]["htr"]["power"]

    asyncio.run(_run())


def test_update_interval_constant() -> None:
    hass = HomeAssistant()
    client = types.SimpleNamespace()
    coord = TermoWebHeaterEnergyCoordinator(hass, client, "1", ["A"])  # type: ignore[arg-type]
    assert coord.update_interval == HTR_ENERGY_UPDATE_INTERVAL


def test_ws_driven_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_htr_samples = AsyncMock(
            return_value=[{"t": 1000, "counter": "1.0"}]
        )

        hass = HomeAssistant()
        coord = TermoWebHeaterEnergyCoordinator(hass, client, "1", ["A"])  # type: ignore[arg-type]

        await coord.async_refresh()
        assert coord.data["1"]["htr"]["energy"]["A"] == 1.0

        client.get_htr_samples = AsyncMock(
            return_value=[{"t": 2000, "counter": "2.0"}]
        )

        async_dispatcher_connect(hass, signal_ws_data("entry"), lambda payload: asyncio.create_task(coord.async_request_refresh()) if payload.get("kind") == "htr_samples" else None)

        dispatcher_send(signal_ws_data("entry"), {"dev_id": "1", "addr": "A", "kind": "htr_samples"})
        await asyncio.sleep(0)

        assert coord.data["1"]["htr"]["energy"]["A"] == 2.0

    asyncio.run(_run())


def test_coordinator_timeout() -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_htr_settings = AsyncMock(side_effect=asyncio.TimeoutError)

        hass = HomeAssistant()
        nodes = {"nodes": [{"addr": "A", "type": "htr"}]}
        coord = TermoWebCoordinator(
            hass, client, 30, "1", {}, nodes  # type: ignore[arg-type]
        )

        with pytest.raises(UpdateFailed, match="API timeout"):
            await coord.async_refresh()

    asyncio.run(_run())


def test_heater_energy_timeout() -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_htr_samples = AsyncMock(side_effect=asyncio.TimeoutError)

        hass = HomeAssistant()
        coord = TermoWebHeaterEnergyCoordinator(
            hass, client, "1", ["A"]  # type: ignore[arg-type]
        )

        with pytest.raises(UpdateFailed, match="API timeout"):
            await coord.async_refresh()

    asyncio.run(_run())
