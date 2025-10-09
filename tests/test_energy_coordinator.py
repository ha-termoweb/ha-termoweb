from __future__ import annotations

import asyncio
from datetime import timedelta
import types
from typing import Any, Callable, Iterable, Mapping
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
from custom_components.termoweb.inventory import (
    AccumulatorNode,
    HeaterNode,
    Node,
    normalize_heater_addresses,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_connect, dispatcher_send
from homeassistant.helpers.update_coordinator import UpdateFailed

EnergyStateCoordinator = coord_module.EnergyStateCoordinator
StateCoordinator = coord_module.StateCoordinator


def test_device_display_name_helper() -> None:
    """Helpers should trim names and fall back to device id."""

    assert coord_module._device_display_name({"name": " Device "}, "dev") == "Device"
    assert coord_module._device_display_name({"name": ""}, "dev") == "Device dev"
    assert coord_module._device_display_name({}, "dev") == "Device dev"
    assert coord_module._device_display_name({"name": 1234}, "dev") == "1234"


def test_ensure_heater_section_helper() -> None:
    """The helper must reuse existing sections or insert defaults."""

    nodes_by_type: dict[str, dict[str, Any]] = {
        "htr": {"addrs": ["1"], "settings": {"1": {}}}
    }
    existing = coord_module._ensure_heater_section(nodes_by_type, lambda: {})
    assert existing is nodes_by_type["htr"]

    nodes_by_type = {}
    created = coord_module._ensure_heater_section(
        nodes_by_type, lambda: {"addrs": ["A"], "settings": {}}
    )
    assert created == {"addrs": ["A"], "settings": {}}
    assert nodes_by_type["htr"] == {"addrs": ["A"], "settings": {}}

    nodes_by_type = {"htr": []}  # type: ignore[assignment]
    replaced = coord_module._ensure_heater_section(
        nodes_by_type,
        lambda: {"addrs": ["B"], "settings": {"B": {"mode": "auto"}}},
    )
    assert replaced == {"addrs": ["B"], "settings": {"B": {"mode": "auto"}}}
    assert nodes_by_type["htr"] == replaced


def test_merge_nodes_by_type_appends_new_addresses(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """Helper should append payload addresses not present in cache."""

    client = types.SimpleNamespace(get_node_settings=AsyncMock())
    hass = HomeAssistant()
    inventory = inventory_builder("dev", {})
    coord = StateCoordinator(
        hass,
        client,
        30,
        "dev",
        {"name": "Device"},
        inventory.payload,
        inventory=inventory,
    )

    cache_map = {"htr": ["A"]}
    current = {"htr": {"addrs": ["A"], "settings": {}}}
    payload = {"htr": {"B": {"mode": "auto"}}}

    merged = coord._merge_nodes_by_type(cache_map, current, payload)

    assert merged["htr"]["addrs"] == ["A", "B"]
    assert merged["htr"]["settings"]["B"] == {"mode": "auto"}


def test_merge_nodes_by_type_skips_invalid_addresses(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """Helper should ignore payload and cache entries without valid addresses."""

    client = types.SimpleNamespace(get_node_settings=AsyncMock())
    hass = HomeAssistant()
    inventory = inventory_builder("dev", {})
    coord = StateCoordinator(
        hass,
        client,
        30,
        "dev",
        {"name": "Device"},
        inventory.payload,
        inventory=inventory,
    )

    cache_map = {"htr": ["", "A"]}
    current = {"htr": {"addrs": ["  "], "settings": {}}}
    payload = {"htr": {"": {"bad": True}}}

    merged = coord._merge_nodes_by_type(cache_map, current, payload)

    assert merged["htr"]["addrs"] == ["A"]
    assert merged["htr"]["settings"] == {}
    assert "" not in merged["htr"]["settings"]


def test_inventory_addresses_by_type_handles_missing_inventory() -> None:
    """Inventory helper should return an empty mapping when unset."""

    client = types.SimpleNamespace(get_node_settings=AsyncMock())
    hass = HomeAssistant()
    coord = StateCoordinator(
        hass,
        client,
        30,
        "dev",
        {"name": "Device"},
        None,
    )

    assert coord._inventory is None
    assert coord._inventory_addresses_by_type() == {}


def test_inventory_addresses_by_type_merges_forward_map(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """Inventory helper should combine node cache and heater mapping results."""

    client = types.SimpleNamespace(get_node_settings=AsyncMock())
    hass = HomeAssistant()
    nodes = {"nodes": [{"addr": "1", "type": "htr"}]}
    inventory = inventory_builder("dev", nodes)
    coord = StateCoordinator(
        hass,
        client,
        30,
        "dev",
        {"name": "Device"},
        inventory.payload,
        inventory=inventory,
    )

    inventory = coord._inventory
    assert inventory is not None

    object.__setattr__(
        inventory,
        "_heater_address_map_cache",
        ({"htr": ("1", "2")}, {"1": frozenset({"htr"}), "2": frozenset({"htr"})}),
    )

    addresses = coord._inventory_addresses_by_type()
    assert addresses["htr"] == ["1", "2"]


def test_update_nodes_builds_inventory_from_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Updating nodes should build an inventory container from payload data."""

    client = types.SimpleNamespace(get_node_settings=AsyncMock())
    hass = HomeAssistant()
    coord = StateCoordinator(
        hass,
        client,
        30,
        "dev",
        {"name": "Device"},
        None,
    )

    payload = {
        "nodes": [
            {"addr": "1", "type": "htr", "name": "Heater"},
            {"addr": "2", "type": "acm", "name": "Accumulator"},
        ]
    }
    sentinel = [
        HeaterNode(name="Heater", addr="1"),
        AccumulatorNode(name="Accumulator", addr="2"),
    ]

    def _fake_builder(raw: Any) -> list[Node]:
        assert raw == payload
        return sentinel

    monkeypatch.setattr(coord_module, "build_node_inventory", _fake_builder)

    coord.update_nodes(payload)

    inventory = coord._inventory
    assert isinstance(inventory, coord_module.Inventory)
    assert inventory.nodes == tuple(sentinel)
    assert inventory.payload == payload


def test_update_nodes_handles_non_iterable_inventory_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-iterable inventory hints should fall back to building from payload."""

    client = types.SimpleNamespace(get_node_settings=AsyncMock())
    hass = HomeAssistant()
    coord = StateCoordinator(
        hass,
        client,
        30,
        "dev",
        {"name": "Device"},
        None,
    )

    payload = {"nodes": [{"addr": "5", "type": "htr"}]}
    sentinel = [HeaterNode(name="Heater", addr="5")]

    def _fake_builder(raw: Any) -> list[Node]:
        assert raw == payload
        return sentinel

    monkeypatch.setattr(coord_module, "build_node_inventory", _fake_builder)

    coord.update_nodes(payload, inventory=object())

    inventory = coord._inventory
    assert isinstance(inventory, coord_module.Inventory)
    assert inventory.nodes == tuple(sentinel)


def test_update_nodes_accepts_inventory_container(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    """Providing an inventory container should be reused directly."""

    client = types.SimpleNamespace(get_node_settings=AsyncMock())
    hass = HomeAssistant()
    payload = {"nodes": [{"addr": "3", "type": "acm"}]}
    nodes_list = [AccumulatorNode(name="Accumulator", addr="3")]
    container = inventory_builder("dev", payload, nodes_list)

    coord = StateCoordinator(
        hass,
        client,
        30,
        "dev",
        {"name": "Device"},
        container.payload,
        inventory=container,
    )

    assert coord._inventory is container
    assert coord._inventory.payload == payload


def test_normalise_type_section_cleans_addresses() -> None:
    section = {
        "addrs": [" 1 ", None, "2"],
        "settings": {" 1 ": {"mode": "auto"}, None: {"mode": "skip"}},
    }

    normalized = StateCoordinator._normalise_type_section("htr", section, [" 3 ", ""])

    assert normalized["addrs"] == ["1", "None", "2"]
    assert normalized["settings"] == {
        "1": {"mode": "auto"},
        "None": {"mode": "skip"},
    }


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
        monkeypatch.setattr(coord_module, "time_mod", _fake_time)

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
        assert client.get_node_settings.await_args_list[0].args == (
            "dev",
            ("htr", "A"),
        )
        assert client.get_node_settings.await_args_list[1].args == (
            "dev",
            ("acm", "B"),
        )
        assert dev["htr"]["settings"]["A"] == {"mode": "auto"}
        nodes_by_type = dev.get("nodes_by_type")
        assert nodes_by_type is not None
        assert nodes_by_type["htr"]["settings"]["A"] == {"mode": "auto"}
        assert nodes_by_type["htr"]["addrs"] == ["A"]
        assert nodes_by_type["acm"]["addrs"] == ["B"]
        assert nodes_by_type["acm"]["settings"]["B"] == {"mode": "auto"}
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

        dev = coord.data["dev"]
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


def test_state_coordinator_ignores_non_dict_payloads() -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_node_settings = AsyncMock(
            side_effect=[
                "unexpected",
                {"mode": "auto"},
            ]
        )

        hass = HomeAssistant()
        nodes = {
            "nodes": [
                {"addr": "A", "type": "htr"},
                {"addr": "B", "type": "htr"},
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

        dev = coord.data["dev"]
        assert client.get_node_settings.await_args_list[0].args == (
            "dev",
            ("htr", "A"),
        )
        assert client.get_node_settings.await_args_list[1].args == (
            "dev",
            ("htr", "B"),
        )
        assert dev["htr"]["settings"] == {"B": {"mode": "auto"}}

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


def test_register_pending_setting_normalizes_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    monkeypatch.setattr(coord_module, "time_mod", lambda: 100.0)

    coord.register_pending_setting(" htr ", " 01 ", mode="Auto", stemp="21.5", ttl=5)

    key = ("htr", "01")
    assert key in coord._pending_settings
    entry = coord._pending_settings[key]
    assert entry.mode == "auto"
    assert entry.stemp == pytest.approx(21.5)
    assert entry.expires_at == pytest.approx(105.0)


def test_should_defer_pending_setting_handles_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    monkeypatch.setattr(coord_module, "time_mod", lambda: 10.0)
    coord.register_pending_setting("htr", "1", mode="auto", stemp=20.0, ttl=0)

    monkeypatch.setattr(coord_module, "time_mod", lambda: 11.0)
    assert coord._should_defer_pending_setting("htr", "1", {"mode": "auto"}) is False
    assert coord._pending_settings == {}


def test_should_defer_pending_setting_defers_missing_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    monkeypatch.setattr(coord_module, "time_mod", lambda: 50.0)
    coord.register_pending_setting("acm", "2", mode="boost", stemp=None, ttl=5)

    monkeypatch.setattr(coord_module, "time_mod", lambda: 52.0)
    assert coord._should_defer_pending_setting("acm", "2", None) is True
    assert ("acm", "2") in coord._pending_settings


def test_should_defer_pending_setting_satisfied_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    monkeypatch.setattr(coord_module, "time_mod", lambda: 75.0)
    coord.register_pending_setting("htr", "3", mode="manual", stemp=19.5, ttl=5)

    payload = {"mode": "MANUAL", "stemp": 19.52}
    monkeypatch.setattr(coord_module, "time_mod", lambda: 76.0)
    assert coord._should_defer_pending_setting("htr", "3", payload) is False
    assert coord._pending_settings == {}


def test_should_defer_pending_setting_mismatch_defers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    monkeypatch.setattr(coord_module, "time_mod", lambda: 200.0)
    coord.register_pending_setting("htr", "4", mode="auto", stemp=18.0, ttl=10)

    payload = {"mode": "eco", "stemp": 16.0}
    monkeypatch.setattr(coord_module, "time_mod", lambda: 201.0)
    assert coord._should_defer_pending_setting("htr", "4", payload) is True
    assert ("htr", "4") in coord._pending_settings


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
        assert client.get_node_settings.await_args_list[-1].args == (
            "dev",
            ("htr", "B"),
        )
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
            inventory=[AccumulatorNode(name="Acc", addr="3")],
        )
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
        inventory = [
            HeaterNode(name="Heater", addr="A"),
            AccumulatorNode(name="Accumulator", addr="B"),
        ]
        coord = StateCoordinator(
            hass,
            client,
            30,
            "dev",
            {"name": "Device"},
            nodes,  # type: ignore[arg-type]
            inventory=inventory,
        )

        coord.data = {
            "dev": {
                "nodes_by_type": {
                    "htr": {"addrs": ["A"], "settings": {"A": {"mode": "manual"}}}
                }
            }
        }

        await coord.async_refresh_heater(("acm", "B"))

        dev_data = coord.data["dev"]
        acm_section = dev_data["nodes_by_type"]["acm"]
        assert "B" in acm_section["addrs"]
        assert acm_section["settings"]["B"] == {"mode": "eco"}

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
        assert client.get_node_settings.await_args_list[-1].args == (
            "dev",
            ("htr", "A"),
        )

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


def test_state_coordinator_async_update_data_reuses_previous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[dict[str, list[str]], dict[str, Any], dict[str, dict[str, Any]]]] = []
    original = StateCoordinator._merge_nodes_by_type

    def _spy(
        self: StateCoordinator,
        cache_map: dict[str, list[str]],
        current_sections: dict[str, Any] | None,
        new_payload: dict[str, dict[str, Any]] | None,
    ) -> dict[str, dict[str, Any]]:
        calls.append((dict(cache_map), dict(current_sections or {}), dict(new_payload or {})))
        return original(self, cache_map, current_sections, new_payload)

    monkeypatch.setattr(StateCoordinator, "_merge_nodes_by_type", _spy)

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

        coord.update_nodes({"nodes": [{"type": "acm", "addr": "7"}]})
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

    assert len(calls) == 1
    cache_map, _, payload = calls[0]
    assert cache_map.get("acm") == ["7"]
    assert payload.get("acm", {}).get("7") == {"mode": "eco"}


def test_async_refresh_heater_uses_merge_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[dict[str, list[str]], dict[str, Any], dict[str, dict[str, Any]]]] = []
    original = StateCoordinator._merge_nodes_by_type

    def _spy(
        self: StateCoordinator,
        cache_map: dict[str, list[str]],
        current_sections: dict[str, Any] | None,
        new_payload: dict[str, dict[str, Any]] | None,
    ) -> dict[str, dict[str, Any]]:
        calls.append((dict(cache_map), dict(current_sections or {}), dict(new_payload or {})))
        return original(self, cache_map, current_sections, new_payload)

    monkeypatch.setattr(StateCoordinator, "_merge_nodes_by_type", _spy)

    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_node_settings = AsyncMock(return_value={"mode": "heat"})

        hass = HomeAssistant()
        coord = StateCoordinator(
            hass,
            client,
            30,
            "dev",
            {"name": " Device "},
            {},
        )

        coord.update_nodes({"nodes": [{"type": "htr", "addr": "A"}]})

        await coord.async_refresh_heater("A")

    asyncio.run(_run())

    assert len(calls) == 1
    cache_map, current, payload = calls[0]
    assert cache_map == {"htr": ["A"]}
    assert current == {}
    assert payload == {"htr": {"A": {"mode": "heat"}}}


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
                "nodes_by_type": {
                    "acm": {"addrs": ["B"], "settings": {"B": {"mode": "auto"}}}
                },
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
        monkeypatch.setattr(coord_module, "time_mod", _fake_time)

        await coord.async_refresh()
        fake_time = 1900.0
        await coord.async_refresh()

        assert coord.data["1"]["htr"]["energy"]["A"] == pytest.approx(0.001)
        assert "A" not in coord.data["1"]["htr"]["power"]

    asyncio.run(_run())


def test_energy_processing_consistent_between_poll_and_ws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        monkeypatch.setattr(coord_module.time, "time", lambda: 4000.0)
        monkeypatch.setattr(coord_module, "time_mod", lambda: 4000.0)

        poll_client = types.SimpleNamespace()
        poll_client.get_node_samples = AsyncMock(
            side_effect=[
                [{"t": 1000.0, "counter": 1200.0}],
                [{"t": 1600.0, "counter": 2400.0}],
            ]
        )

        hass = HomeAssistant()
        poll_coord = EnergyStateCoordinator(
            hass,
            poll_client,
            "dev",
            ["A"],  # type: ignore[arg-type]
        )

        await poll_coord.async_refresh()
        await poll_coord.async_refresh()

        poll_bucket = poll_coord.data["dev"]["htr"]
        poll_energy = poll_bucket["energy"]["A"]
        poll_power = poll_bucket["power"]["A"]

        ws_client = types.SimpleNamespace()
        ws_client.get_node_samples = AsyncMock(
            return_value=[{"t": 1000.0, "counter": 1200.0}]
        )

        ws_coord = EnergyStateCoordinator(
            hass,
            ws_client,
            "dev",
            ["A"],  # type: ignore[arg-type]
        )

        await ws_coord.async_refresh()

        ws_coord.handle_ws_samples(
            "dev",
            {"htr": {"A": {"samples": [{"t": 1600.0, "counter": 2400.0}]}}},
        )

        ws_bucket = ws_coord.data["dev"]["htr"]

        assert ws_bucket["energy"]["A"] == pytest.approx(poll_energy)
        assert ws_bucket["power"]["A"] == pytest.approx(poll_power)
        assert ws_coord._last[("htr", "A")] == poll_coord._last[("htr", "A")]

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
        client.get_node_samples = AsyncMock(return_value=[{"t": 1000, "counter": None}])

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


def test_ws_samples_update_defers_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_node_samples = AsyncMock(
            return_value=[{"t": 0.0, "counter": 1000.0}]
        )

        hass = HomeAssistant()
        coord = EnergyStateCoordinator(hass, client, "dev", {"htr": ["A"]})

        fake_time = 0.0

        def _fake_time() -> float:
            return fake_time

        monkeypatch.setattr(coord_module.time, "time", _fake_time)
        monkeypatch.setattr(coord_module, "time_mod", _fake_time)

        await coord.async_refresh()

        data = coord.data["dev"]["htr"]
        assert data["energy"]["A"] == pytest.approx(1.0)
        assert "A" not in data["power"]

        client.get_node_samples.reset_mock()
        client.get_node_samples.return_value = [{"t": 7200.0, "counter": 3000.0}]

        fake_time = 3600.0
        coord.handle_ws_samples(
            "dev",
            {"htr": {" A ": {"samples": [{"t": 3600.0, "counter": 2000.0}]}}},
            lease_seconds=300.0,
        )

        updated = coord.data["dev"]["htr"]
        assert updated["energy"]["A"] == pytest.approx(2.0)
        assert updated["power"]["A"] == pytest.approx(1000.0)
        last_t, last_kwh = coord._last[("htr", "A")]
        assert last_t == pytest.approx(3600.0)
        assert last_kwh == pytest.approx(2.0)
        assert coord.update_interval == timedelta(seconds=375)

        fake_time = 3800.0
        await coord.async_refresh()
        assert client.get_node_samples.await_count == 0

        fake_time = 3976.0
        await coord.async_refresh()
        assert client.get_node_samples.await_count == 1
        assert coord.update_interval == HTR_ENERGY_UPDATE_INTERVAL

    asyncio.run(_run())


def test_should_skip_poll_conditions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise websocket-driven polling skip decisions."""

    hass = HomeAssistant()
    client = types.SimpleNamespace()
    coord = EnergyStateCoordinator(hass, client, "dev", {"htr": ["A"]})

    assert coord._should_skip_poll() is False

    coord.data = {}
    coord._ws_deadline = None
    assert coord._should_skip_poll() is False

    coord._ws_deadline = 10.0
    monkeypatch.setattr(coord_module, "time_mod", lambda: 15.0)
    assert coord._should_skip_poll() is False

    coord.data = {"dev": {}}
    coord._ws_deadline = 20.0
    monkeypatch.setattr(coord_module, "time_mod", lambda: 5.0)
    assert coord._should_skip_poll() is True


def test_async_update_data_uses_cached_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure websocket freshness skips API polling and returns cached data."""

    async def _run() -> None:
        hass = HomeAssistant()
        client = types.SimpleNamespace()
        coord = EnergyStateCoordinator(hass, client, "dev", {"htr": ["A"]})

        coord.data = {
            "dev": {
                "nodes_by_type": {"htr": {"addrs": ["A"], "energy": {"A": 1.0}, "power": {}}},
                "htr": {"addrs": ["A"], "energy": {"A": 1.0}, "power": {}},
            }
        }
        coord._ws_deadline = 100.0

        monkeypatch.setattr(coord_module, "time_mod", lambda: 50.0)

        cached = await coord._async_update_data()
        assert cached == coord.data

        coord.data = ["bad"]  # type: ignore[assignment]
        monkeypatch.setattr(coord, "_should_skip_poll", lambda: True)
        assert await coord._async_update_data() == {}

    asyncio.run(_run())


def test_ws_margin_seconds_bounds() -> None:
    """Verify the websocket margin respects defaults and upper bounds."""

    hass = HomeAssistant()
    client = types.SimpleNamespace()
    coord = EnergyStateCoordinator(hass, client, "dev", {"htr": ["A"]})

    coord._ws_lease = -1
    assert coord._ws_margin_seconds() == coord._ws_margin_default

    coord._ws_lease = 10_000.0
    assert coord._ws_margin_seconds() == 600.0


def test_extract_sample_point_variants() -> None:
    """Confirm sample extraction tolerates nested mappings and iterables."""

    nested = {"samples": {"samples": [{"t": 100.0, "counter": 2000.0}]}}
    assert EnergyStateCoordinator._extract_sample_point(nested) == (100.0, 2000.0)

    sequence = [{"t": 10.0, "counter": 1000.0}, {"t": 20.0, "counter": 3000.0}]
    assert EnergyStateCoordinator._extract_sample_point(sequence) == (20.0, 3000.0)

    assert EnergyStateCoordinator._extract_sample_point({"samples": []}) is None
    assert EnergyStateCoordinator._extract_sample_point("invalid") is None


def test_handle_ws_samples_branching(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the various guard paths when processing websocket samples."""

    hass = HomeAssistant()
    client = types.SimpleNamespace()
    coord = EnergyStateCoordinator(hass, client, "dev", {"htr": ["A"], "acm": ["B"]})

    coord.handle_ws_samples("other", {})

    coord.data = []  # type: ignore[assignment]
    coord.handle_ws_samples("dev", {})

    coord.data = {"dev": []}  # type: ignore[assignment]
    coord.handle_ws_samples("dev", {})

    coord.data = {"dev": {"nodes_by_type": []}}  # type: ignore[assignment]
    coord.handle_ws_samples("dev", {})

    coord.data = {
        "dev": {
            "nodes_by_type": {
                "htr": {"addrs": ["A"], "energy": {}, "power": {}},
                "acm": [],
            },
            "htr": {"addrs": ["A"], "energy": {}, "power": {}},
        }
    }
    coord._last = {("htr", "A"): (float("nan"), 1.0)}

    fake_now = 0.0

    def _fake_time() -> float:
        return fake_now

    monkeypatch.setattr(coord_module, "time_mod", _fake_time)
    monkeypatch.setattr(coord_module.time, "time", _fake_time)

    updates = {
        "": {"1": {"samples": []}},
        "pmo": {"1": {"samples": []}},
        "htr": {
            "A": {"samples": []},
            "unknown": {"samples": [{"t": 100.0, "counter": 500.0}]},
        },
    }

    coord.handle_ws_samples("dev", updates, lease_seconds=60.0)
    assert coord._ws_deadline == pytest.approx(120.0)
    assert coord.update_interval == timedelta(seconds=300)

    fake_now = 100.0
    updates = {
        "acm": {"B": {"samples": [{"t": 200.0, "counter": 0.0}]}},
        "htr": {
            "missing": {"samples": [{"t": 0.0, "counter": 0.0}]},
            "A": {"samples": [{"t": 3600.0, "counter": 2000.0}]},
        },
    }

    coord.handle_ws_samples("dev", updates, lease_seconds=120.0)

    energy_bucket = coord.data["dev"]["nodes_by_type"]["htr"].get("energy", {})
    assert energy_bucket.get("A") == pytest.approx(2.0)
    assert coord._ws_deadline == pytest.approx(280.0)

    fake_now = 200.0
    coord.handle_ws_samples(
        "dev",
        {"htr": {"A": {"samples": [{"t": 3500.0, "counter": 1000.0}]}}},
        lease_seconds=0,
    )
    assert coord.data["dev"]["nodes_by_type"]["htr"].get("power", {}).get("A") is None
    assert coord._ws_deadline is None

    fake_now = 300.0
    coord.handle_ws_samples(
        "dev",
        {"htr": {"A": {"samples": [{"t": 7100.0, "counter": 4000.0}]}}},
    )

    updated = coord.data["dev"]["nodes_by_type"]["htr"]
    assert updated.get("energy", {}).get("A") == pytest.approx(4.0)
    assert updated.get("power", {}).get("A") == pytest.approx(3000.0)

    coord._last.pop(("htr", "A"), None)
    fake_now = 500.0
    coord.handle_ws_samples(
        "dev",
        {"htr": {"A": {"samples": [{"t": 8200.0, "counter": 5000.0}]}}},
    )
    assert coord._last[("htr", "A")] == (8200.0, 5.0)


def test_handle_ws_samples_skips_empty_points() -> None:
    """Ensure empty websocket samples do not update coordinator state."""

    hass = HomeAssistant()
    client = types.SimpleNamespace()
    coord = EnergyStateCoordinator(hass, client, "dev", {"htr": ["A"]})
    coord.data = {
        "dev": {
            "nodes_by_type": {"htr": {"addrs": ["A"], "energy": {}, "power": {}}},
            "htr": {"addrs": ["A"], "energy": {}, "power": {}},
        }
    }

    coord.handle_ws_samples(
        "dev",
        {"htr": {"A": [{"samples": []}, {"samples": []}]}},
    )

    assert coord._last == {}

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
        client.get_node_samples = AsyncMock(
            return_value=[{"t": 1000, "counter": "1.0"}]
        )

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
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
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

    inventory = inventory_builder("dev", {})
    coord = StateCoordinator(
        hass,
        client,
        30,
        "dev",
        {"name": "Device"},
        inventory.payload,
        inventory=inventory,
    )

    coord.update_nodes(nodes)

    assert nodes in calls
    inventory = coord._inventory
    assert isinstance(inventory, coord_module.Inventory)
    assert inventory.nodes == tuple(built_nodes)


def test_state_coordinator_update_nodes_uses_provided_inventory(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    hass = HomeAssistant()
    client = types.SimpleNamespace()
    nodes = {"nodes": [{"addr": "A", "type": "htr"}]}
    provided_inventory = [Node(name="Heater", addr="A", node_type="htr")]

    inventory = inventory_builder("dev", {})
    coord = StateCoordinator(
        hass,
        client,
        30,
        "dev",
        {"name": "Device"},
        inventory.payload,
        inventory=inventory,
    )

    coord.update_nodes(nodes, provided_inventory)

    inventory = coord._inventory
    assert isinstance(inventory, coord_module.Inventory)
    assert inventory.payload == nodes
    assert inventory.nodes[0] is provided_inventory[0]


def test_energy_state_coordinator_update_addresses_filters_duplicates() -> None:
    hass = HomeAssistant()
    client = types.SimpleNamespace()
    coord = EnergyStateCoordinator(hass, client, "dev", ["orig"])  # type: ignore[arg-type]

    coord.update_addresses(["A", " ", "B", "A", "B ", ""])

    expected_map, _ = normalize_heater_addresses(["A", " ", "B", "A", "B ", ""])
    filtered_expected = {
        key: list(value) for key, value in expected_map.items() if value
    }
    assert coord._addresses_by_type == filtered_expected
    expected_flat = [addr for addrs in filtered_expected.values() for addr in addrs]
    assert coord._addrs == expected_flat


def test_energy_state_coordinator_update_addresses_ignores_invalid_types() -> None:
    hass = HomeAssistant()
    client = types.SimpleNamespace()
    coord = EnergyStateCoordinator(hass, client, "dev", [])  # type: ignore[arg-type]

    coord.update_addresses(
        {" ": ["skip"], "htr": ["A"], "acm": ["", "B"], "foo": ["X"]}
    )

    expected_map, _ = normalize_heater_addresses(
        {" ": ["skip"], "htr": ["A"], "acm": ["", "B"], "foo": ["X"]}
    )
    filtered_expected = {
        key: list(value) for key, value in expected_map.items() if value
    }
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
    filtered_expected = {
        key: list(value) for key, value in expected_map.items() if value
    }
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

    monkeypatch.setattr(coord_module, "_normalize_heater_payload", fake_normalize)

    coord = EnergyStateCoordinator(hass, client, "dev", [])  # type: ignore[arg-type]
    assert calls == [[]]

    coord.update_addresses(["ignored"])

    assert calls[-1] == ["ignored"]
    assert coord._addresses_by_type == {"htr": ["A"]}
    assert coord._compat_aliases == {"htr": "htr"}


def test_normalize_heater_payload_handles_none() -> None:
    mapping, aliases = coord_module._normalize_heater_payload(None)

    assert mapping == {}
    assert aliases == {"htr": "htr"}


def test_normalize_heater_payload_aliases() -> None:
    mapping, aliases = coord_module._normalize_heater_payload(
        {"heater": ["1"], "acm": ["2"]}
    )

    assert mapping == {"htr": ["1"], "acm": ["2"]}
    assert aliases["heater"] == "htr"
    assert aliases["acm"] == "acm"


def test_normalize_heater_payload_accepts_string() -> None:
    """String payloads should be coerced into heater address lists."""

    mapping, aliases = coord_module._normalize_heater_payload(" 5 ")

    assert mapping == {"htr": ["5"]}
    assert aliases == {"htr": "htr"}


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
        client.get_node_samples = AsyncMock(
            return_value=[{"t": 1000, "counter": "1.0"}]
        )

        hass = HomeAssistant()
        coord = EnergyStateCoordinator(hass, client, "1", ["A"])  # type: ignore[arg-type]

        await coord.async_refresh()
        assert coord.data["1"]["htr"]["energy"]["A"] == pytest.approx(0.001)

        client.get_node_samples = AsyncMock(
            return_value=[{"t": 2000, "counter": "2.0"}]
        )

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
