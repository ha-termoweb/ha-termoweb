from __future__ import annotations

import asyncio
from datetime import timedelta
import types
from typing import Any
from unittest.mock import AsyncMock

import pytest

from conftest import _install_stubs

_install_stubs()

from aiohttp import ClientError
from custom_components.termoweb import coordinator as coord_module
from custom_components.termoweb.api import TermoWebAuthError, TermoWebRateLimitError
from custom_components.termoweb.const import (
    HTR_ENERGY_UPDATE_INTERVAL,
    signal_ws_data,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_connect, dispatcher_send
from homeassistant.helpers.update_coordinator import UpdateFailed

TermoWebHeaterEnergyCoordinator = coord_module.TermoWebHeaterEnergyCoordinator
TermoWebCoordinator = coord_module.TermoWebCoordinator


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
                TermoWebAuthError("denied"),
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


def test_inventory_handles_mixed_payloads() -> None:
    hass = HomeAssistant()
    nodes = {
        "nodes": [
            {"addr": " 1 ", "type": "HTR"},
            {"addr": None, "type": "acm"},
        ],
        "htrs": [{"addr": "2"}, "  "],
        "acms": {"addrs": ["4", "", "4"]},
        "thermostats": [{"addr": "5"}, "6", ""],
    }

    coord = TermoWebCoordinator(
        hass,
        types.SimpleNamespace(),
        30,
        "dev",
        {"name": " Device "},
        nodes,  # type: ignore[arg-type]
    )

    addr_map, reverse = coord._addr_lookup()

    assert addr_map["htr"] == ["1", "2", " 1 "]
    assert addr_map["acm"] == ["4"]
    assert addr_map["thermostat"] == ["5", "6"]
    assert reverse["1"] == "htr"
    assert reverse["4"] == "acm"
    assert coord._addrs() == ["1", "2", " 1 "]

def test_inventory_handles_node_list_payload() -> None:
    hass = HomeAssistant()
    coord = TermoWebCoordinator(
        hass,
        types.SimpleNamespace(),
        30,
        "dev",
        {},
        [
            {"addr": "7", "type": "THM"},
            {"addr": "8", "type": None},
            "ignored",
        ],  # type: ignore[arg-type]
    )

    addr_map, reverse = coord._addr_lookup()

    assert addr_map["thm"] == ["7"]
    assert addr_map["htr"] == ["8"]
    assert reverse == {"7": "thm", "8": "htr"}


def test_normalise_type_section_varied_inputs() -> None:
    hass = HomeAssistant()
    coord = TermoWebCoordinator(
        hass,
        types.SimpleNamespace(),
        30,
        "dev",
        {},
        {"nodes": []},  # type: ignore[arg-type]
    )

    tuple_section = {
        "addrs": (" 7 ", "8", ""),
        "settings": {"7": {"mode": "heat"}, "8": None},
    }
    result = coord._normalise_type_section("thm", tuple_section, ["9", "7", ""])
    assert result == {"addrs": ["9", "7", "8"], "settings": {"7": {"mode": "heat"}}}

    string_section = {
        "addrs": " 10 ",
        "settings": {"10": 1, 11: {"foo": "bar"}, "skip": None},
    }
    str_result = coord._normalise_type_section("acm", string_section, [])
    assert str_result == {"addrs": ["10"], "settings": {"10": 1, "11": {"foo": "bar"}}}

    list_result = coord._normalise_type_section("pmo", [" 12 ", "", "12"], ["13"])
    assert list_result == {"addrs": ["13", "12"], "settings": {}}


def test_fetch_settings_prefers_generic_getter() -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_node_settings = AsyncMock(return_value={"ok": True})
        client.get_htr_settings = AsyncMock(return_value={"wrong": True})

        hass = HomeAssistant()
        coord = TermoWebCoordinator(
            hass,
            client,
            30,
            "dev",
            {},
            {"nodes": []},  # type: ignore[arg-type]
        )

        result = await coord._fetch_settings("thm", "A")
        assert result == {"ok": True}
        client.get_node_settings.assert_awaited_once_with("dev", ("thm", "A"))
        client.get_htr_settings.assert_not_called()

        client.get_node_settings = AsyncMock(return_value="bad")
        assert await coord._fetch_settings("thm", "A") is None

        delattr(client, "get_node_settings")
        client.get_htr_settings = AsyncMock(return_value={"mode": "heat"})
        assert await coord._fetch_settings("htr", "A") == {"mode": "heat"}
        client.get_htr_settings.assert_awaited_with("dev", "A")

        client.get_htr_settings = AsyncMock(return_value="oops")
        assert await coord._fetch_settings("htr", "A") is None

        empty_client = types.SimpleNamespace()
        other = TermoWebCoordinator(
            hass,
            empty_client,
            30,
            "dev",
            {},
            {"nodes": []},  # type: ignore[arg-type]
        )
        assert await other._fetch_settings("thm", "A") is None

    asyncio.run(_run())


def test_refresh_heater_accepts_typed_address() -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_node_settings = AsyncMock(return_value={"mode": "auto"})

        hass = HomeAssistant()
        nodes = {"nodes": [{"addr": "A", "type": "thm"}]}
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

        await coord.async_refresh_heater(("THM", "A"))

        assert updates, "Expected async_set_updated_data to be called"
        dev = updates[-1]["dev"]
        assert dev["htr"]["settings"] == {}
        assert dev["nodes_by_type"]["thm"]["settings"]["A"] == {"mode": "auto"}

    asyncio.run(_run())


def test_async_update_data_merges_previous_sections() -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_node_settings = AsyncMock(return_value={"mode": "fresh"})

        hass = HomeAssistant()
        nodes = {
            "nodes": [
                {"addr": "A", "type": "htr"},
                {"addr": "B", "type": "thm"},
            ]
        }
        coord = TermoWebCoordinator(
            hass,
            client,
            30,
            "dev",
            {"name": " Device "},
            nodes,  # type: ignore[arg-type]
        )

        coord.data = {
            "dev": {
                "dev_id": "dev",
                "name": "Old",
                "raw": {"name": "Old"},
                "connected": False,
                "nodes": nodes,
                "nodes_by_type": {
                    "htr": {"addrs": ["A"], "settings": {"A": {"mode": "hold"}}},
                    "legacy": {
                        "addrs": ("L", "L2"),
                        "settings": {"L": {"mode": "legacy"}, "L2": None},
                    },
                    "thm": {"addrs": ["B"], "settings": {"B": {"mode": "cool"}}},
                },
                "htr": {"addrs": ["A"], "settings": {"A": {"mode": "hold"}}},
                "legacy": {"addrs": ["L"], "settings": {"L": {"mode": "legacy"}}},
                "auxiliary": {
                    "addrs": ["B1", ""],
                    "settings": {"B1": {"temp": 20}, "skip": None},
                },
                "ignored": "skip",
            }
        }

        result = await coord._async_update_data()
        data = result["dev"]

        assert data["dev_id"] == "dev"
        assert data["name"] == "Device"
        assert data["connected"] is True
        assert data["nodes_by_type"]["htr"]["settings"]["A"] == {"mode": "fresh"}
        assert data["nodes_by_type"]["thm"]["settings"]["B"] == {"mode": "cool"}
        assert data["nodes_by_type"]["legacy"] == {
            "addrs": ["L", "L2"],
            "settings": {"L": {"mode": "legacy"}},
        }
        assert data["nodes_by_type"]["auxiliary"] == {
            "addrs": ["B1"],
            "settings": {"B1": {"temp": 20}},
        }
        assert coord._rr_index["dev"] == 1

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


def test_energy_samples_invalid_strings() -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_htr_samples = AsyncMock(
            return_value=[{"t": " ", "counter": "garbage"}]
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

        monkeypatch.setattr(coord_module, "float_or_none", _raise_client_error)

        with pytest.raises(UpdateFailed, match="API error: bad"):
            await coord.async_refresh()

    asyncio.run(_run())


def test_coordinator_rate_limit_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        async def _raise_rate_limit(*_args: Any, **_kwargs: Any) -> Any:
            raise TermoWebRateLimitError("429")

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
