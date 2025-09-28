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
from custom_components.termoweb import nodes as nodes_module
from custom_components.termoweb.api import BackendAuthError, BackendRateLimitError
from custom_components.termoweb.const import (
    HTR_ENERGY_UPDATE_INTERVAL,
    signal_ws_data,
)
from custom_components.termoweb.utils import normalize_heater_addresses
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_connect, dispatcher_send
from homeassistant.helpers.update_coordinator import UpdateFailed

EnergyStateCoordinator = coord_module.EnergyStateCoordinator
StateCoordinator = coord_module.StateCoordinator


def test_ensure_inventory_rebuilds_and_refreshes_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = types.SimpleNamespace(get_node_settings=AsyncMock())
    hass = HomeAssistant()
    nodes = {"nodes": [{"addr": "1", "type": "htr"}]}
    coord = StateCoordinator(
        hass,
        client,
        30,
        "dev",
        {"name": "Device"},
        nodes,  # type: ignore[arg-type]
    )

    sentinel = [nodes_module.Node(name="Heater", addr="1", node_type="htr")]

    def _fake_builder(raw: Any) -> list[nodes_module.Node]:
        assert raw == nodes
        return sentinel

    monkeypatch.setattr(coord_module, "build_node_inventory", _fake_builder)

    coord._nodes = nodes
    coord._node_inventory = []

    inventory = coord._ensure_inventory()

    assert inventory is sentinel
    assert coord._nodes_by_type == {"htr": ["1"]}
    assert coord._addr_lookup == {"1": {"htr"}}


def test_update_nodes_uses_provided_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    client = types.SimpleNamespace(get_node_settings=AsyncMock())
    hass = HomeAssistant()
    coord = StateCoordinator(
        hass,
        client,
        30,
        "dev",
        {"name": "Device"},
        {},
    )

    builder_called = False

    def _fake_builder(raw: Any) -> list[nodes_module.Node]:
        nonlocal builder_called
        builder_called = True
        return []

    monkeypatch.setattr(coord_module, "build_node_inventory", _fake_builder)

    provided = [nodes_module.Node(name="Acc", addr="2", node_type="acm")]
    coord.update_nodes({"nodes": []}, provided)

    assert coord._node_inventory == provided
    assert coord._nodes_by_type == {"acm": ["2"]}
    assert coord._addr_lookup == {"2": {"acm"}}
    assert builder_called is False


def test_set_inventory_from_nodes_defaults_to_empty() -> None:
    client = types.SimpleNamespace(get_node_settings=AsyncMock())
    hass = HomeAssistant()
    coord = StateCoordinator(
        hass,
        client,
        30,
        "dev",
        {"name": "Device"},
        {},
    )

    result = coord._set_inventory_from_nodes(None)

    assert result == []
    assert coord._node_inventory == []


def test_power_calculation(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_node_samples = AsyncMock(
            side_effect=[
                [{"t": 1000, "counter": "1.0"}],
                [{"t": 1900, "counter": "1.5"}],
            ]
        )

        hass = HomeAssistant()
        coord = EnergyStateCoordinator(hass, client, "1", ["A"])  # type: ignore[arg-type]

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
        client.get_node_settings = AsyncMock(return_value={"mode": "auto"})

        hass = HomeAssistant()
        nodes = {
            "nodes": [
                {"addr": "A", "type": "htr"},
                {"addr": "B", "type": "acm"},
            ]
        }
        coord = StateCoordinator(
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

        dev = coord.data["dev"]
        client.get_node_settings.assert_called_once_with("dev", ("htr", "A"))
        assert dev["htr"]["settings"]["A"] == {"mode": "auto"}
        nodes_by_type = dev.get("nodes_by_type")
        assert nodes_by_type is not None
        assert nodes_by_type["htr"]["settings"]["A"] == {"mode": "auto"}
        assert nodes_by_type["htr"]["addrs"] == ["A"]
        assert nodes_by_type["acm"]["addrs"] == ["B"]
        assert nodes_by_type["acm"]["settings"] == {}
        assert dev["htr"] is nodes_by_type["htr"]
        assert coord._backoff == 0
        assert coord.update_interval == timedelta(seconds=coord._base_interval)

    asyncio.run(_run())


def test_state_coordinator_round_robin_mixed_types() -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_node_settings = AsyncMock(
            side_effect=[
                {"mode": "auto"},
                {"mode": "eco"},
                {"mode": "charge"},
            ]
        )

        hass = HomeAssistant()
        nodes = {
            "nodes": [
                {"addr": "A", "type": "htr"},
                {"addr": "C", "type": "htr"},
                {"addr": "B", "type": "acm"},
            ]
        }
        coord = StateCoordinator(
            hass,
            client,
            30,
            "dev",
            {"name": "Device"},
            nodes,  # type: ignore[arg-type]
        )

        await coord.async_refresh()
        await coord.async_refresh()
        await coord.async_refresh()

        dev = coord.data["dev"]
        assert coord._rr_index["dev"] == 0
        assert client.get_node_settings.await_args_list[0].args == (
            "dev",
            ("htr", "A"),
        )
        assert client.get_node_settings.await_args_list[1].args == (
            "dev",
            ("htr", "C"),
        )
        assert client.get_node_settings.await_args_list[2].args == (
            "dev",
            ("acm", "B"),
        )
        assert dev["nodes_by_type"]["htr"]["settings"]["A"] == {"mode": "auto"}
        assert dev["nodes_by_type"]["htr"]["settings"]["C"] == {"mode": "eco"}
        assert dev["nodes_by_type"]["acm"]["settings"]["B"] == {"mode": "charge"}
        assert dev["htr"] is dev["nodes_by_type"]["htr"]
        assert dev["acm"] is dev["nodes_by_type"]["acm"]

    asyncio.run(_run())


def test_refresh_heater_skips_invalid_inputs() -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_node_settings = AsyncMock()

        hass = HomeAssistant()
        nodes = {
            "nodes": [
                {"addr": "A", "type": "htr"},
                {"addr": "B", "type": "acm"},
            ]
        }
        coord = StateCoordinator(
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
        client.get_node_settings.assert_not_called()
        assert updates == []

        coord.data = {"dev": {"htr": {"settings": {}}}}
        await coord.async_refresh_heater("A")
        client.get_node_settings.assert_called_once_with("dev", ("htr", "A"))
        assert updates == []

    asyncio.run(_run())


def test_register_node_address_strips_and_skips_blank() -> None:
    client = types.SimpleNamespace(get_node_settings=AsyncMock())
    hass = HomeAssistant()
    nodes = {"nodes": []}
    coord = StateCoordinator(
        hass,
        client,
        30,
        "dev",
        {"name": "Device"},
        nodes,  # type: ignore[arg-type]
    )

    coord._nodes_by_type = {}
    coord._addr_lookup = {}

    coord._register_node_address("", "")
    coord._register_node_address("htr", " ")

    assert coord._nodes_by_type == {}
    assert coord._addr_lookup == {}

    coord._register_node_address(" htr ", " A ")

    assert coord._nodes_by_type == {"htr": ["A"]}
    assert coord._addr_lookup == {"A": {"htr"}}


def test_refresh_heater_updates_existing_and_new_data() -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_node_settings = AsyncMock(
            side_effect=[{"mode": "auto"}, {"mode": "eco"}]
        )

        hass = HomeAssistant()
        nodes = {
            "nodes": [
                {"addr": "A", "type": "htr"},
                {"addr": "B", "type": "htr"},
                {"addr": "C", "type": "acm"},
            ]
        }
        coord = StateCoordinator(
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
        client.get_node_settings.assert_called_with("dev", ("htr", "A"))
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
        nodes_by_type = dev.get("nodes_by_type")
        assert nodes_by_type is not None
        assert nodes_by_type["htr"]["settings"]["A"] == {"mode": "auto"}
        assert nodes_by_type["htr"]["addrs"] == ["A", "B"]

        await coord.async_refresh_heater("B")
        assert client.get_node_settings.await_args_list[-1].args == ("dev", ("htr", "B"))
        assert len(updates) == 2
        second = updates[-1]
        htr_second = second["dev"]["htr"]
        assert htr_second["settings"]["A"] == {"mode": "auto"}
        assert htr_second["settings"]["B"] == {"mode": "eco"}
        assert htr_second["addrs"] == ["A", "B"]
        assert htr_second["addrs"] is not htr["addrs"]
        nodes_by_type_second = second["dev"].get("nodes_by_type")
        assert nodes_by_type_second is not None
        assert nodes_by_type_second["htr"]["settings"]["B"] == {"mode": "eco"}
        assert nodes_by_type_second["htr"]["addrs"] == ["A", "B"]
        assert nodes_by_type_second["acm"]["addrs"] == ["C"]
        assert nodes_by_type_second["acm"]["settings"] == {}

    asyncio.run(_run())


def test_refresh_heater_handles_tuple_and_acm() -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_node_settings = AsyncMock(return_value={"mode": "auto"})

        hass = HomeAssistant()
        coord = StateCoordinator(
            hass,
            client,
            30,
            "dev",
            {"name": "Device"},
            {},
        )

        coord._node_inventory = [
            nodes_module.Node(name="Acc", addr="3", node_type="acm")
        ]
        coord._refresh_node_cache()
        coord.data = {
            "dev": {
                "nodes_by_type": {
                    "acm": {"addrs": "bad", "settings": {"1": {"prev": True}}},
                    "weird": [],
                }
            }
        }

        updates: list[dict[str, dict[str, Any]]] = []

        def _set_updated_data(self, data: dict[str, dict[str, Any]]) -> None:
            updates.append(data)
            self.data = data

        coord.async_set_updated_data = types.MethodType(  # type: ignore[attr-defined]
            _set_updated_data,
            coord,
        )

        await coord.async_refresh_heater(("acm", "2"))

        client.get_node_settings.assert_awaited_once()
        assert updates, "Expected coordinator data to be updated"
        latest = updates[-1]["dev"]
        acm_section = latest["nodes_by_type"]["acm"]
        addrs = acm_section["addrs"]
        assert addrs[0] == "3"
        assert "2" in addrs
        assert acm_section["settings"]["2"] == {"mode": "auto"}
        assert "htr" in latest["nodes_by_type"]

    asyncio.run(_run())


def test_async_refresh_heater_adds_missing_type() -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_node_settings = AsyncMock(return_value={"mode": "eco"})

        hass = HomeAssistant()
        nodes = {"nodes": [{"addr": "A", "type": "htr"}]}
        coord = StateCoordinator(
            hass,
            client,
            30,
            "dev",
            {"name": "Device"},
            nodes,  # type: ignore[arg-type]
        )

        coord._nodes_by_type = {"htr": ["A"]}
        coord._addr_lookup = {"A": {"htr"}}
        coord.data = {
            "dev": {
                "nodes_by_type": {"htr": {"addrs": ["A"], "settings": {"A": {"mode": "manual"}}}}
            }
        }

        await coord.async_refresh_heater(("acm", "B"))

        assert coord._nodes_by_type["acm"] == ["B"]
        dev_data = coord.data["dev"]
        assert dev_data["nodes_by_type"]["acm"]["settings"]["B"] == {"mode": "eco"}

    asyncio.run(_run())


def test_refresh_heater_populates_missing_metadata() -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_node_settings = AsyncMock(return_value={"mode": "heat"})

        hass = HomeAssistant()
        nodes = {
            "nodes": [
                {"addr": "A", "type": "htr"},
                {"addr": "B", "type": "acm"},
            ]
        }
        coord = StateCoordinator(
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
        client.get_node_settings.assert_called_once_with("dev", ("htr", "A"))
        assert result["name"] == "Device"
        assert result["raw"] == {"name": " Device "}
        assert result["nodes"] == nodes
        assert result["connected"] is False
        assert result["htr"]["settings"]["A"] == {"mode": "heat"}
        nodes_by_type = result.get("nodes_by_type")
        assert nodes_by_type is not None
        assert nodes_by_type["htr"]["settings"]["A"] == {"mode": "heat"}
        assert nodes_by_type["acm"]["addrs"] == ["B"]

    asyncio.run(_run())


def test_refresh_heater_handles_errors(caplog: pytest.LogCaptureFixture) -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_node_settings = AsyncMock(
            side_effect=[
                "not-a-dict",
                TimeoutError("slow"),
                BackendAuthError("denied"),
            ]
        )

        hass = HomeAssistant()
        nodes = {
            "nodes": [
                {"addr": "A", "type": "htr"},
                {"addr": "B", "type": "acm"},
            ]
        }
        coord = StateCoordinator(
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
        assert client.get_node_settings.await_args_list[-1].args == ("dev", ("htr", "A"))

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


def test_state_coordinator_async_update_data_reuses_previous() -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_node_settings = AsyncMock(return_value={"mode": "eco"})

        hass = HomeAssistant()
        coord = StateCoordinator(
            hass,
            client,
            30,
            "dev",
            {"name": " Device "},
            {},
        )

        coord._node_inventory = [
            nodes_module.Node(name="Acc", addr="7", node_type="acm")
        ]
        coord._refresh_node_cache()
        coord.data = {
            "dev": {
                "nodes_by_type": {
                    "acm": {"settings": {"7": {"prev": True}}},
                    "bad": [],
                },
                "htr": {"settings": {"legacy": {"mode": "auto"}}},
            }
        }

        result = await coord._async_update_data()

        client.get_node_settings.assert_awaited_once()
        dev_data = result["dev"]
        acm_section = dev_data["nodes_by_type"]["acm"]
        assert acm_section["settings"]["7"] == {"mode": "eco"}
        assert dev_data["htr"]["addrs"] == ["legacy"]

    asyncio.run(_run())


def test_async_update_data_skips_non_dict_sections() -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_node_settings = AsyncMock(return_value={"mode": "heat"})

        hass = HomeAssistant()
        nodes = {"nodes": [{"addr": "B", "type": "acm"}]}
        coord = StateCoordinator(
            hass,
            client,
            30,
            "dev",
            {"name": "Device"},
            nodes,  # type: ignore[arg-type]
        )

        coord.data = {
            "dev": {
                "nodes_by_type": {"acm": {"addrs": ["B"], "settings": {"B": {"mode": "manual"}}}},
                "misc": "invalid",
            }
        }

        result = await coord._async_update_data()

        dev_data = result["dev"]
        assert "htr" in dev_data["nodes_by_type"]
        assert client.get_node_settings.await_count == 1

    asyncio.run(_run())


def test_counter_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_node_samples = AsyncMock(
            side_effect=[
                [{"t": 1000, "counter": "5.0"}],
                [{"t": 1900, "counter": "1.0"}],
            ]
        )

        hass = HomeAssistant()
        coord = EnergyStateCoordinator(hass, client, "1", ["A"])  # type: ignore[arg-type]

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
        client.get_node_samples = AsyncMock(
            side_effect=[
                [{"t": 1000, "counter": "1.0"}],
                [{"t": 1600, "counter": "2.0"}],
                [{"t": 1500, "counter": "1.5"}],
            ]
        )

        hass = HomeAssistant()
        coord = EnergyStateCoordinator(
            hass,
            client,
            "1",
            ["A"],  # type: ignore[arg-type]
        )

        await coord.async_refresh()
        assert coord.data["1"]["htr"]["energy"]["A"] == pytest.approx(0.001)
        assert "A" not in coord.data["1"]["htr"]["power"]
        assert coord._last[("htr", "A")] == (1000.0, pytest.approx(0.001))

        await coord.async_refresh()
        second_data = coord.data["1"]["htr"]
        assert second_data["energy"]["A"] == pytest.approx(0.002)
        assert second_data["power"]["A"] == pytest.approx(6.0, rel=1e-3)
        assert coord._last[("htr", "A")] == (1600.0, pytest.approx(0.002))

        await coord.async_refresh()

        final_data = coord.data["1"]["htr"]
        assert final_data["energy"]["A"] == pytest.approx(0.0015)
        assert "A" not in final_data["power"]
        assert coord._last[("htr", "A")] == (1500.0, pytest.approx(0.0015))

    asyncio.run(_run())


def test_energy_samples_missing_fields() -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_node_samples = AsyncMock(
            return_value=[{"t": 1000, "counter": None}]
        )

        hass = HomeAssistant()
        coord = EnergyStateCoordinator(
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
        client.get_node_samples = AsyncMock(
            return_value=[{"t": " ", "counter": "garbage"}]
        )

        hass = HomeAssistant()
        coord = EnergyStateCoordinator(
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


def test_energy_coordinator_alias_creates_canonical_bucket() -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_node_samples = AsyncMock(return_value=[])

        hass = HomeAssistant()
        coord = EnergyStateCoordinator(hass, client, "dev", [])
        coord._addresses_by_type = {"legacy": []}
        coord._compat_aliases = {"htr": "htr", "legacy": "acm"}

        result = await coord._async_update_data()

        dev = result["dev"]
        assert "acm" in dev
        assert dev["legacy"] is dev["acm"]

    asyncio.run(_run())


def test_update_interval_constant() -> None:
    hass = HomeAssistant()
    client = types.SimpleNamespace()
    coord = EnergyStateCoordinator(hass, client, "1", ["A"])  # type: ignore[arg-type]
    assert coord.update_interval == HTR_ENERGY_UPDATE_INTERVAL


def test_heater_energy_samples_empty_on_api_error() -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_node_samples = AsyncMock(side_effect=ClientError("fail"))

        hass = HomeAssistant()
        coord = EnergyStateCoordinator(
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
        client.get_node_samples = AsyncMock(return_value=[{"t": 1000, "counter": "1.0"}])

        hass = HomeAssistant()
        coord = EnergyStateCoordinator(
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


def test_energy_state_coordinator_fetches_acm_samples() -> None:
    async def _run() -> None:
        hass = HomeAssistant()
        client = types.SimpleNamespace()
        client.get_node_samples = AsyncMock(
            return_value=[{"t": 1000, "counter": "1000"}]
        )

        coord = EnergyStateCoordinator(
            hass,
            client,
            "dev",
            {"acm": ["7"]},
        )

        result = await coord._async_update_data()
        client.get_node_samples.assert_awaited_once()
        call = client.get_node_samples.await_args
        assert call.args[0] == "dev"
        assert call.args[1] == ("acm", "7")
        acm_section = result["dev"]["acm"]
        assert acm_section["energy"]["7"] == pytest.approx(1.0)
        nodes_by_type = result["dev"]["nodes_by_type"]
        assert nodes_by_type["acm"] is acm_section
        assert nodes_by_type["htr"] is result["dev"]["htr"]

    asyncio.run(_run())


def test_energy_coordinator_caches_per_type() -> None:
    async def _run() -> None:
        hass = HomeAssistant()
        client = types.SimpleNamespace()
        client.get_node_samples = AsyncMock(
            side_effect=[
                [{"t": 0.0, "counter": 0}],
                [{"t": 0.0, "counter": 0}],
                [{"t": 3600.0, "counter": 1000}],
                [{"t": 3600.0, "counter": 2000}],
            ]
        )

        coord = EnergyStateCoordinator(
            hass,
            client,
            "dev",
            {"htr": ["A"], "acm": ["B"]},
        )

        await coord.async_refresh()
        await coord.async_refresh()

        dev_data = coord.data["dev"]
        assert dev_data["htr"]["energy"]["A"] == pytest.approx(1.0)
        assert dev_data["acm"]["energy"]["B"] == pytest.approx(2.0)
        assert dev_data["htr"]["power"]["A"] == pytest.approx(1000)
        assert dev_data["acm"]["power"]["B"] == pytest.approx(2000)
        nodes_by_type = dev_data["nodes_by_type"]
        assert nodes_by_type["htr"] is dev_data["htr"]
        assert nodes_by_type["acm"] is dev_data["acm"]

        last_htr = coord._last[("htr", "A")]
        assert last_htr[0] == pytest.approx(3600.0)
        assert last_htr[1] == pytest.approx(1.0)
        last_acm = coord._last[("acm", "B")]
        assert last_acm[0] == pytest.approx(3600.0)
        assert last_acm[1] == pytest.approx(2.0)

        assert client.get_node_samples.await_count == 4

    asyncio.run(_run())


def test_energy_coordinator_handles_rate_limit_per_node() -> None:
    async def _run() -> None:
        hass = HomeAssistant()
        client = types.SimpleNamespace()

        async def _side_effect(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            dev_id, descriptor, *_ = args
            node_type, _addr = descriptor
            if node_type == "htr":
                return [{"t": 0, "counter": 500}]
            raise BackendRateLimitError("429")

        client.get_node_samples = AsyncMock(side_effect=_side_effect)

        coord = EnergyStateCoordinator(
            hass,
            client,
            "dev",
            {"htr": ["A"], "acm": ["B"]},
        )

        await coord.async_refresh()

        dev_data = coord.data["dev"]
        assert dev_data["htr"]["energy"]["A"] == pytest.approx(0.5)
        assert dev_data["acm"]["energy"] == {}
        assert ("acm", "B") not in coord._last
        assert ("htr", "A") in coord._last

    asyncio.run(_run())


def test_energy_coordinator_uses_heater_alias() -> None:
    async def _run() -> None:
        hass = HomeAssistant()
        client = types.SimpleNamespace()
        client.get_node_samples = AsyncMock(
            side_effect=[
                [{"t": 0, "counter": 500}],
                [{"t": 0, "counter": 1500}],
            ]
        )

        coord = EnergyStateCoordinator(
            hass,
            client,
            "dev",
            {"heater": ["A"], "acm": ["B"]},
        )

        await coord.async_refresh()

        assert coord._addresses_by_type == {"htr": ["A"], "acm": ["B"]}
        assert coord._compat_aliases["heater"] == "htr"

        dev_data = coord.data["dev"]
        assert dev_data["htr"]["energy"]["A"] == pytest.approx(0.5)
        assert dev_data["heater"] is dev_data["htr"]
        assert dev_data["heater"]["energy"]["A"] == pytest.approx(0.5)
        assert dev_data["acm"]["energy"]["B"] == pytest.approx(1.5)
        nodes_by_type = dev_data["nodes_by_type"]
        assert nodes_by_type["htr"] is dev_data["htr"]
        assert nodes_by_type["heater"] is dev_data["htr"]
        assert nodes_by_type["acm"] is dev_data["acm"]

    asyncio.run(_run())


def test_state_coordinator_update_nodes_rebuilds_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hass = HomeAssistant()
    client = types.SimpleNamespace()
    nodes = {"nodes": [{"addr": "A", "type": "htr"}]}

    built_nodes = [types.SimpleNamespace(addr="A")]
    calls: list[dict[str, Any]] = []

    def fake_build(payload: dict[str, Any]) -> list[Any]:
        calls.append(payload)
        return built_nodes

    monkeypatch.setattr(coord_module, "build_node_inventory", fake_build)

    coord = StateCoordinator(
        hass,
        client,
        30,
        "dev",
        {"name": "Device"},
        {},
    )

    coord.update_nodes(nodes)

    assert calls == [nodes]
    assert coord._node_inventory == built_nodes


def test_state_coordinator_update_nodes_uses_provided_inventory() -> None:
    hass = HomeAssistant()
    client = types.SimpleNamespace()
    nodes = {"nodes": [{"addr": "A", "type": "htr"}]}
    provided_inventory = [nodes_module.Node(name="Heater", addr="A", node_type="htr")]

    coord = StateCoordinator(
        hass,
        client,
        30,
        "dev",
        {"name": "Device"},
        {},
    )

    coord.update_nodes(nodes, provided_inventory)

    assert coord._nodes == nodes
    assert coord._node_inventory is not provided_inventory
    assert coord._node_inventory == provided_inventory
    assert coord._node_inventory[0] is provided_inventory[0]


def test_energy_state_coordinator_update_addresses_filters_duplicates() -> None:
    hass = HomeAssistant()
    client = types.SimpleNamespace()
    coord = EnergyStateCoordinator(hass, client, "dev", ["orig"])  # type: ignore[arg-type]

    coord.update_addresses(["A", " ", "B", "A", "B ", ""])

    expected_map, _ = normalize_heater_addresses(["A", " ", "B", "A", "B ", ""])
    filtered_expected = {key: list(value) for key, value in expected_map.items() if value}
    assert coord._addresses_by_type == filtered_expected
    expected_flat = [addr for addrs in filtered_expected.values() for addr in addrs]
    assert coord._addrs == expected_flat


def test_energy_state_coordinator_update_addresses_ignores_invalid_types() -> None:
    hass = HomeAssistant()
    client = types.SimpleNamespace()
    coord = EnergyStateCoordinator(hass, client, "dev", [])  # type: ignore[arg-type]

    coord.update_addresses({" ": ["skip"], "htr": ["A"], "acm": ["", "B"], "foo": ["X"]})

    expected_map, _ = normalize_heater_addresses(
        {" ": ["skip"], "htr": ["A"], "acm": ["", "B"], "foo": ["X"]}
    )
    filtered_expected = {key: list(value) for key, value in expected_map.items() if value}
    assert coord._addresses_by_type == filtered_expected
    expected_flat = [addr for addrs in filtered_expected.values() for addr in addrs]
    assert coord._addrs == expected_flat


def test_energy_state_coordinator_update_addresses_accepts_map() -> None:
    hass = HomeAssistant()
    client = types.SimpleNamespace()
    coord = EnergyStateCoordinator(hass, client, "dev", [])  # type: ignore[arg-type]

    coord.update_addresses({"htr": ["A", "A"], "acm": ["B", ""], "foo": ["X"]})

    expected_map, _ = normalize_heater_addresses(
        {"htr": ["A", "A"], "acm": ["B", ""], "foo": ["X"]}
    )
    filtered_expected = {key: list(value) for key, value in expected_map.items() if value}
    assert coord._addresses_by_type == filtered_expected
    expected_flat = [addr for addrs in filtered_expected.values() for addr in addrs]
    assert coord._addrs == expected_flat


def test_energy_state_coordinator_update_addresses_uses_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hass = HomeAssistant()
    client = types.SimpleNamespace()
    calls: list[Any] = []

    def fake_normalize(addrs: Any) -> tuple[dict[str, list[str]], dict[str, str]]:
        calls.append(addrs)
        return {"htr": ["A"]}, {"htr": "htr"}

    monkeypatch.setattr(coord_module, "normalize_heater_addresses", fake_normalize)

    coord = EnergyStateCoordinator(hass, client, "dev", [])  # type: ignore[arg-type]
    assert calls == [[]]

    coord.update_addresses(["ignored"])

    assert calls[-1] == ["ignored"]
    assert coord._addresses_by_type == {"htr": ["A"]}
    assert coord._compat_aliases == {"htr": "htr"}


def test_energy_state_coordinator_async_update_adds_legacy_bucket() -> None:
    async def _run() -> None:
        hass = HomeAssistant()
        client = types.SimpleNamespace()
        coord = EnergyStateCoordinator(hass, client, "dev", [])  # type: ignore[arg-type]

        coord._addresses_by_type = {"acm": []}
        coord._addr_lookup = {}
        coord._addrs = []

        result = await coord._async_update_data()
        dev_data = result["dev"]
        assert dev_data["htr"] == {"energy": {}, "power": {}, "addrs": []}

    asyncio.run(_run())


def test_coordinator_rate_limit_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        async def _raise_rate_limit(*_args: Any, **_kwargs: Any) -> Any:
            raise BackendRateLimitError("429")

        client = types.SimpleNamespace()
        client.get_node_settings = AsyncMock(side_effect=_raise_rate_limit)

        hass = HomeAssistant()
        nodes = {"nodes": [{"addr": "A", "type": "htr"}, {"addr": "B", "type": "htr"}]}
        coord = StateCoordinator(
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
            assert client.get_node_settings.await_args_list[-1].args == (
                "1",
                ("htr", "A"),
            )

        with pytest.raises(UpdateFailed, match="Rate limited; backing off to 3600s"):
            await coord.async_refresh()
        assert coord._backoff == 3600
        assert coord.update_interval == timedelta(seconds=3600)
        assert client.get_node_settings.await_args_list[-1].args == (
            "1",
            ("htr", "A"),
        )

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
        client.get_node_settings = AsyncMock(side_effect=ClientError("boom"))

        hass = HomeAssistant()
        nodes = {
            "nodes": [
                {"addr": "A", "type": "htr"},
                {"addr": "B", "type": "acm"},
            ]
        }
        coord = StateCoordinator(
            hass,
            client,
            30,
            "1",
            {},
            nodes,  # type: ignore[arg-type]
        )

        with pytest.raises(UpdateFailed, match="API error: boom"):
            await coord.async_refresh()
        assert client.get_node_settings.await_args_list[-1].args == (
            "1",
            ("htr", "A"),
        )

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
        client.get_node_samples = AsyncMock(return_value=[{"t": 1000, "counter": "1.0"}])

        hass = HomeAssistant()
        coord = EnergyStateCoordinator(hass, client, "1", ["A"])  # type: ignore[arg-type]

        await coord.async_refresh()
        assert coord.data["1"]["htr"]["energy"]["A"] == pytest.approx(0.001)

        client.get_node_samples = AsyncMock(return_value=[{"t": 2000, "counter": "2.0"}])

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
        client.get_node_settings = AsyncMock(side_effect=TimeoutError)

        hass = HomeAssistant()
        nodes = {"nodes": [{"addr": "A", "type": "htr"}]}
        coord = StateCoordinator(
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
        client.get_node_samples = AsyncMock(side_effect=asyncio.TimeoutError)

        hass = HomeAssistant()
        coord = EnergyStateCoordinator(
            hass,
            client,
            "1",
            ["A"],  # type: ignore[arg-type]
        )

        with pytest.raises(UpdateFailed, match="API timeout"):
            await coord.async_refresh()

    asyncio.run(_run())
