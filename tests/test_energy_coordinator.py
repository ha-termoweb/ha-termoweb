from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, Dict
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
        self.data: Dict[str, Dict[str, Any]] | None = None

    async def async_refresh(self) -> None:
        self.data = await self._async_update_data()

    async def async_config_entry_first_refresh(self) -> None:
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
import time as _time


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
