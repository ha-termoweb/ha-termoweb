from __future__ import annotations

import asyncio
import importlib.util
from datetime import timedelta
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
    Path(__file__).resolve().parents[1]
    / "custom_components"
    / "termoweb"
    / "coordinator.py"
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
signal_ws_data = __import__(
    f"{package}.const", fromlist=["signal_ws_data"]
).signal_ws_data
HTR_ENERGY_UPDATE_INTERVAL = __import__(
    f"{package}.const", fromlist=["HTR_ENERGY_UPDATE_INTERVAL"]
).HTR_ENERGY_UPDATE_INTERVAL


@pytest.mark.parametrize("value", ["", "  ", "not-a-number"])
def test_as_float_returns_none_for_invalid_strings(value: str) -> None:
    assert coord_module._as_float(value) is None


def test_as_float_returns_none_for_unhandled_types() -> None:
    assert coord_module._as_float(object()) is None


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
        assert coord.data["1"]["htr"]["energy"]["A"] == pytest.approx(0.001)
        assert "A" not in coord.data["1"]["htr"]["power"]

        fake_time = 1900.0
        await coord.async_refresh()
        assert coord.data["1"]["htr"]["energy"]["A"] == pytest.approx(0.0015)
        power = coord.data["1"]["htr"]["power"]["A"]
        assert power == pytest.approx(2.0, rel=1e-3)

    asyncio.run(_run())


def test_coordinator_success_resets_backoff() -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_htr_settings = AsyncMock(return_value={"mode": "auto"})

        hass = HomeAssistant()
        nodes = {"nodes": [{"addr": "A", "type": "htr"}]}
        coord = TermoWebCoordinator(
            hass,
            client,
            30,
            "dev",
            {"name": "Device"},
            nodes,  # type: ignore[arg-type]
        )
        coord._backoff = 120
        coord.update_interval = timedelta(seconds=999)

        await coord.async_refresh()

        assert coord.data["dev"]["htr"]["settings"]["A"] == {"mode": "auto"}
        assert coord._backoff == 0
        assert coord.update_interval == timedelta(seconds=coord._base_interval)

    asyncio.run(_run())


def test_refresh_heater_skips_invalid_inputs() -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_htr_settings = AsyncMock()

        hass = HomeAssistant()
        nodes = {"nodes": [{"addr": "A", "type": "htr"}]}
        coord = TermoWebCoordinator(
            hass,
            client,
            30,
            "dev",
            {"name": " Device "},
            nodes,  # type: ignore[arg-type]
        )

        updates: list[dict[str, dict[str, Any]]] = []

        def _set_updated_data(self, data: dict[str, dict[str, Any]]) -> None:
            updates.append(data)
            self.data = data

        coord.async_set_updated_data = types.MethodType(  # type: ignore[attr-defined]
            _set_updated_data,
            coord,
        )

        coord.data = {"dev": {"htr": {"settings": {"A": {"mode": "manual"}}}}}
        await coord.async_refresh_heater("")
        client.get_htr_settings.assert_not_called()
        assert updates == []

        coord.data = {"dev": {"htr": {"settings": {}}}}
        await coord.async_refresh_heater("A")
        client.get_htr_settings.assert_called_once()
        assert updates == []

    asyncio.run(_run())


def test_refresh_heater_updates_existing_and_new_data() -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_htr_settings = AsyncMock(
            side_effect=[{"mode": "auto"}, {"mode": "eco"}]
        )

        hass = HomeAssistant()
        nodes = {
            "nodes": [
                {"addr": "A", "type": "htr"},
                {"addr": "B", "type": "htr"},
            ]
        }
        coord = TermoWebCoordinator(
            hass,
            client,
            15,
            "dev",
            {"name": " Device "},
            nodes,  # type: ignore[arg-type]
        )

        updates: list[dict[str, dict[str, Any]]] = []

        def _set_updated_data(self, data: dict[str, dict[str, Any]]) -> None:
            updates.append(data)
            self.data = data

        coord.async_set_updated_data = types.MethodType(  # type: ignore[attr-defined]
            _set_updated_data,
            coord,
        )

        coord.data = None
        await coord.async_refresh_heater("A")
        assert len(updates) == 1
        first = updates[-1]
        dev = first["dev"]
        assert dev["dev_id"] == "dev"
        assert dev["name"] == "Device"
        assert dev["raw"] == {"name": " Device "}
        assert dev["nodes"] == nodes
        assert dev["connected"] is True
        htr = dev["htr"]
        assert htr["settings"]["A"] == {"mode": "auto"}
        assert htr["addrs"] == ["A", "B"]

        await coord.async_refresh_heater("B")
        assert len(updates) == 2
        second = updates[-1]
        htr_second = second["dev"]["htr"]
        assert htr_second["settings"]["A"] == {"mode": "auto"}
        assert htr_second["settings"]["B"] == {"mode": "eco"}
        assert htr_second["addrs"] == ["A", "B"]
        assert htr_second["addrs"] is not htr["addrs"]

    asyncio.run(_run())


def test_refresh_heater_populates_missing_metadata() -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_htr_settings = AsyncMock(return_value={"mode": "heat"})

        hass = HomeAssistant()
        nodes = {"nodes": [{"addr": "A", "type": "htr"}]}
        coord = TermoWebCoordinator(
            hass,
            client,
            45,
            "dev",
            {"name": " Device "},
            nodes,  # type: ignore[arg-type]
        )

        updates: list[dict[str, dict[str, Any]]] = []

        def _set_updated_data(self, data: dict[str, dict[str, Any]]) -> None:
            updates.append(data)
            self.data = data

        coord.async_set_updated_data = types.MethodType(  # type: ignore[attr-defined]
            _set_updated_data,
            coord,
        )

        coord.data = {"dev": {"htr": {"settings": {}}, "connected": False}}

        await coord.async_refresh_heater("A")

        assert updates, "Expected async_set_updated_data to be called"
        result = updates[-1]["dev"]
        assert result["name"] == "Device"
        assert result["raw"] == {"name": " Device "}
        assert result["nodes"] == nodes
        assert result["connected"] is False
        assert result["htr"]["settings"]["A"] == {"mode": "heat"}

    asyncio.run(_run())


def test_refresh_heater_handles_errors(caplog: pytest.LogCaptureFixture) -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_htr_settings = AsyncMock(
            side_effect=[
                "not-a-dict",
                TimeoutError("slow"),
                coord_module.TermoWebAuthError("denied"),
            ]
        )

        hass = HomeAssistant()
        nodes = {"nodes": [{"addr": "A", "type": "htr"}]}
        coord = TermoWebCoordinator(
            hass,
            client,
            30,
            "dev",
            {"name": "Device"},
            nodes,  # type: ignore[arg-type]
        )

        updates: list[dict[str, dict[str, Any]]] = []

        def _set_updated_data(self, data: dict[str, dict[str, Any]]) -> None:
            updates.append(data)
            self.data = data

        coord.async_set_updated_data = types.MethodType(  # type: ignore[attr-defined]
            _set_updated_data,
            coord,
        )

        coord.data = {"dev": {"htr": {"settings": {}}}}
        await coord.async_refresh_heater("A")
        assert updates == []

        caplog.clear()
        with caplog.at_level("ERROR"):
            await coord.async_refresh_heater("A")
        assert "Timeout refreshing heater settings" in caplog.text

        caplog.clear()
        with caplog.at_level("ERROR"):
            await coord.async_refresh_heater("A")
        assert "Failed to refresh heater settings" in caplog.text
        assert updates == []

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

        assert coord.data["1"]["htr"]["energy"]["A"] == pytest.approx(0.001)
        assert "A" not in coord.data["1"]["htr"]["power"]

    asyncio.run(_run())


def test_energy_regression_resets_last() -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_htr_samples = AsyncMock(
            side_effect=[
                [{"t": 1000, "counter": "1.0"}],
                [{"t": 1600, "counter": "2.0"}],
                [{"t": 1500, "counter": "1.5"}],
            ]
        )

        hass = HomeAssistant()
        coord = TermoWebHeaterEnergyCoordinator(
            hass,
            client,
            "1",
            ["A"],  # type: ignore[arg-type]
        )

        await coord.async_refresh()
        assert coord.data["1"]["htr"]["energy"]["A"] == pytest.approx(0.001)
        assert "A" not in coord.data["1"]["htr"]["power"]
        assert coord._last[("1", "A")] == (1000.0, pytest.approx(0.001))

        await coord.async_refresh()
        second_data = coord.data["1"]["htr"]
        assert second_data["energy"]["A"] == pytest.approx(0.002)
        assert second_data["power"]["A"] == pytest.approx(6.0, rel=1e-3)
        assert coord._last[("1", "A")] == (1600.0, pytest.approx(0.002))

        await coord.async_refresh()

        final_data = coord.data["1"]["htr"]
        assert final_data["energy"]["A"] == pytest.approx(0.0015)
        assert "A" not in final_data["power"]
        assert coord._last[("1", "A")] == (1500.0, pytest.approx(0.0015))

    asyncio.run(_run())


def test_energy_samples_missing_fields() -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_htr_samples = AsyncMock(
            return_value=[{"t": 1000, "counter": None}]
        )

        hass = HomeAssistant()
        coord = TermoWebHeaterEnergyCoordinator(
            hass,
            client,
            "dev",
            ["A"],  # type: ignore[arg-type]
        )

        await coord.async_refresh()
        data = coord.data["dev"]["htr"]
        assert data["energy"] == {}
        assert data["power"] == {}

    asyncio.run(_run())


def test_update_interval_constant() -> None:
    hass = HomeAssistant()
    client = types.SimpleNamespace()
    coord = TermoWebHeaterEnergyCoordinator(hass, client, "1", ["A"])  # type: ignore[arg-type]
    assert coord.update_interval == HTR_ENERGY_UPDATE_INTERVAL


def test_heater_energy_samples_empty_on_api_error() -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_htr_samples = AsyncMock(side_effect=ClientError("fail"))

        hass = HomeAssistant()
        coord = TermoWebHeaterEnergyCoordinator(
            hass,
            client,
            "1",
            ["A"],  # type: ignore[arg-type]
        )

        await coord.async_refresh()
        data = coord.data["1"]["htr"]
        assert data["energy"] == {}
        assert data["power"] == {}
        assert coord._last == {}

    asyncio.run(_run())


def test_heater_energy_client_error_update_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_htr_samples = AsyncMock(return_value=[{"t": 1000, "counter": "1.0"}])

        hass = HomeAssistant()
        coord = TermoWebHeaterEnergyCoordinator(
            hass,
            client,
            "1",
            ["A"],  # type: ignore[arg-type]
        )

        def _raise_client_error(_value: Any) -> float:
            raise ClientError("bad")

        monkeypatch.setattr(coord_module, "_as_float", _raise_client_error)

        with pytest.raises(UpdateFailed, match="API error: bad"):
            await coord.async_refresh()

    asyncio.run(_run())


def test_coordinator_rate_limit_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        async def _raise_rate_limit(*_args: Any, **_kwargs: Any) -> Any:
            raise coord_module.TermoWebRateLimitError("429")

        client = types.SimpleNamespace()
        client.get_htr_settings = AsyncMock(side_effect=_raise_rate_limit)

        hass = HomeAssistant()
        nodes = {"nodes": [{"addr": "A", "type": "htr"}, {"addr": "B", "type": "htr"}]}
        coord = TermoWebCoordinator(
            hass,
            client,
            30,
            "1",
            {},
            nodes,  # type: ignore[arg-type]
        )

        expected_backoffs = [60, 120, 240, 480, 960, 1920, 3600]
        for backoff in expected_backoffs:
            with pytest.raises(
                UpdateFailed, match=f"Rate limited; backing off to {backoff}s"
            ):
                await coord.async_refresh()
            assert coord._backoff == backoff
            assert coord.update_interval == timedelta(seconds=backoff)
            assert client.get_htr_settings.await_args_list[-1].args[1] == "A"

        with pytest.raises(UpdateFailed, match="Rate limited; backing off to 3600s"):
            await coord.async_refresh()
        assert coord._backoff == 3600
        assert coord.update_interval == timedelta(seconds=3600)
        assert client.get_htr_settings.await_args_list[-1].args[1] == "A"

    class RaisingLogger:
        def debug(
            self, *_args: Any, exc_info: Exception | None = None, **_kwargs: Any
        ) -> None:
            if exc_info is not None:
                raise exc_info

        def info(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def warning(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def error(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    monkeypatch.setattr(coord_module, "_LOGGER", RaisingLogger())

    asyncio.run(_run())


def test_coordinator_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_htr_settings = AsyncMock(side_effect=ClientError("boom"))

        hass = HomeAssistant()
        nodes = {"nodes": [{"addr": "A", "type": "htr"}]}
        coord = TermoWebCoordinator(
            hass,
            client,
            30,
            "1",
            {},
            nodes,  # type: ignore[arg-type]
        )

        with pytest.raises(UpdateFailed, match="API error: boom"):
            await coord.async_refresh()
        assert client.get_htr_settings.await_args_list[-1].args[1] == "A"

    class RaisingLogger:
        def debug(
            self, *_args: Any, exc_info: Exception | None = None, **_kwargs: Any
        ) -> None:
            if exc_info is not None:
                raise exc_info

        def info(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def warning(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def error(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    monkeypatch.setattr(coord_module, "_LOGGER", RaisingLogger())

    asyncio.run(_run())


def test_ws_driven_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_htr_samples = AsyncMock(return_value=[{"t": 1000, "counter": "1.0"}])

        hass = HomeAssistant()
        coord = TermoWebHeaterEnergyCoordinator(hass, client, "1", ["A"])  # type: ignore[arg-type]

        await coord.async_refresh()
        assert coord.data["1"]["htr"]["energy"]["A"] == pytest.approx(0.001)

        client.get_htr_samples = AsyncMock(return_value=[{"t": 2000, "counter": "2.0"}])

        async_dispatcher_connect(
            hass,
            signal_ws_data("entry"),
            lambda payload: asyncio.create_task(coord.async_request_refresh())
            if payload.get("kind") == "htr_samples"
            else None,
        )

        dispatcher_send(
            signal_ws_data("entry"), {"dev_id": "1", "addr": "A", "kind": "htr_samples"}
        )
        await asyncio.sleep(0)

        assert coord.data["1"]["htr"]["energy"]["A"] == pytest.approx(0.002)

    asyncio.run(_run())


def test_coordinator_timeout() -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_htr_settings = AsyncMock(side_effect=TimeoutError)

        hass = HomeAssistant()
        nodes = {"nodes": [{"addr": "A", "type": "htr"}]}
        coord = TermoWebCoordinator(
            hass,
            client,
            30,
            "1",
            {},
            nodes,  # type: ignore[arg-type]
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
            hass,
            client,
            "1",
            ["A"],  # type: ignore[arg-type]
        )

        with pytest.raises(UpdateFailed, match="API timeout"):
            await coord.async_refresh()

    asyncio.run(_run())
