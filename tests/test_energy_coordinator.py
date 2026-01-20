from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
import types
from typing import Any, Callable, Iterable, Mapping
from unittest.mock import AsyncMock

import pytest

from conftest import _install_stubs
from conftest import build_device_metadata_payload

_install_stubs()

from aiohttp import ClientError
from custom_components.termoweb import coordinator as coord_module
from custom_components.termoweb.backend.rest_client import (
    BackendAuthError,
    BackendRateLimitError,
)
from custom_components.termoweb.const import (
    HTR_ENERGY_UPDATE_INTERVAL,
    signal_ws_data,
)
from custom_components.termoweb.domain import state_to_dict
from custom_components.termoweb.domain.energy import (
    EnergyNodeMetrics,
    EnergySnapshot,
    build_empty_snapshot,
    coerce_snapshot,
)
from custom_components.termoweb.domain.state import DomainStateStore
from custom_components.termoweb.domain.view import DomainStateView
from custom_components.termoweb.domain.ids import (
    NodeId as DomainNodeId,
    NodeType as DomainNodeType,
)
from custom_components.termoweb.inventory import (
    AccumulatorNode,
    HeaterNode,
    Inventory,
    Node,
    build_node_inventory,
    normalize_heater_addresses,
    normalize_power_monitor_addresses,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_connect, dispatcher_send
from homeassistant.helpers.update_coordinator import UpdateFailed

EnergyStateCoordinator = coord_module.EnergyStateCoordinator
StateCoordinator = coord_module.StateCoordinator


def _state_payload(
    coord: coord_module.StateCoordinator, node_type: str, addr: str
) -> dict[str, Any] | None:
    """Return the legacy payload stored in the domain state view."""

    view = getattr(coord, "domain_view", None)
    if view is None:
        return None
    state = view.get_heater_state(node_type, addr)
    return state_to_dict(state) if state is not None else None


def _energy_metric(
    coord: EnergyStateCoordinator, node_type: str, addr: str
) -> float | None:
    """Return the cached energy metric for ``node_type``/``addr``."""

    snapshot = coerce_snapshot(coord.data)
    assert snapshot is not None
    metrics = snapshot.metrics_for_type(node_type)
    metric = metrics.get(addr)
    return None if metric is None else metric.energy_kwh


def _power_metric(
    coord: EnergyStateCoordinator, node_type: str, addr: str
) -> float | None:
    """Return the cached power metric for ``node_type``/``addr``."""

    snapshot = coerce_snapshot(coord.data)
    assert snapshot is not None
    metrics = snapshot.metrics_for_type(node_type)
    metric = metrics.get(addr)
    return None if metric is None else metric.power_w


def _inventory_from_nodes(dev_id: str, payload: Mapping[str, Any]) -> Inventory:
    """Return an Inventory built from ``payload``."""

    return Inventory(dev_id, list(build_node_inventory(payload)))


def _state_coordinator_from_nodes(
    hass: HomeAssistant,
    client: Any,
    base_interval: int,
    dev_id: str,
    device: coord_module.DeviceMetadata | None,
    nodes: Mapping[str, Any],
) -> StateCoordinator:
    """Build a ``StateCoordinator`` with inventory derived from ``nodes``."""

    inventory = _inventory_from_nodes(dev_id, nodes)
    metadata = device or build_device_metadata_payload(dev_id)
    return StateCoordinator(
        hass,
        client,
        base_interval,
        dev_id,
        metadata,
        nodes,
        inventory=inventory,
    )


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
        build_device_metadata_payload("dev", name="Device"),
        nodes=None,
        inventory=container,
    )

    assert coord._inventory is container
    assert coord._inventory.nodes == tuple(nodes_list)

    coord.update_nodes(inventory=container)
    assert coord._inventory is container


def test_domain_view_energy_metrics_prune() -> None:
    """Energy view reads should honor the store inventory."""

    node_id_allowed = DomainNodeId(DomainNodeType.HEATER, "A")
    node_id_blocked = DomainNodeId(DomainNodeType.HEATER, "B")
    store = DomainStateStore([node_id_allowed])
    snapshot = EnergySnapshot(
        dev_id="dev",
        metrics={
            node_id_allowed: EnergyNodeMetrics(
                energy_kwh=1.25,
                power_w=250.0,
                source="rest",
                ts=123.0,
            ),
            node_id_blocked: EnergyNodeMetrics(
                energy_kwh=9.0,
                power_w=900.0,
                source="ws",
                ts=456.0,
            ),
        },
        updated_at=500.0,
        ws_deadline=None,
    )

    assert store.set_energy_snapshot(snapshot) is True

    view = DomainStateView("dev", store)
    metric = view.get_energy_metric(DomainNodeType.HEATER, "A")
    assert metric is not None
    assert metric.energy_kwh == pytest.approx(1.25)
    assert view.get_energy_metric(DomainNodeType.HEATER, "B") is None
    assert view.get_energy_metrics_for_type(DomainNodeType.HEATER) == {"A": metric}


def test_power_calculation(
    monkeypatch: pytest.MonkeyPatch,
    inventory_from_map: Callable[
        [Mapping[str, Iterable[str]] | None, str], coord_module.Inventory
    ],
) -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_node_samples = AsyncMock(
            side_effect=[
                [{"t": 1000, "counter": "1.0"}],
                [{"t": 1900, "counter": "1.5"}],
            ]
        )

        hass = HomeAssistant()
        inventory = inventory_from_map({"htr": ["A"]}, dev_id="1")
        coord = EnergyStateCoordinator(hass, client, "1", inventory)

        fake_time = 1000.0

        def _fake_time() -> float:
            return fake_time

        monkeypatch.setattr(coord_module.time, "time", _fake_time)
        monkeypatch.setattr(coord_module, "time_mod", _fake_time)

        await coord.async_refresh()
        assert _energy_metric(coord, "htr", "A") == pytest.approx(0.001)
        assert _power_metric(coord, "htr", "A") is None

        fake_time = 1900.0
        await coord.async_refresh()
        assert _energy_metric(coord, "htr", "A") == pytest.approx(0.0015)
        power = _power_metric(coord, "htr", "A")
        assert power == pytest.approx(2.0, rel=1e-3)

    asyncio.run(_run())


def test_energy_coordinator_writes_snapshot_to_state_view(
    monkeypatch: pytest.MonkeyPatch,
    inventory_from_map: Callable[
        [Mapping[str, Iterable[str]] | None, str], coord_module.Inventory
    ],
) -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_node_samples = AsyncMock(return_value=[{"t": 1000, "counter": 1000}])
        client.get_node_settings = AsyncMock()

        hass = HomeAssistant()
        inventory = inventory_from_map({"htr": ["A"]}, dev_id="dev")
        coordinator = StateCoordinator(
            hass,
            client,
            30,
            "dev",
            build_device_metadata_payload("dev"),
            nodes=None,
            inventory=inventory,
        )
        energy = EnergyStateCoordinator(
            hass,
            client,
            "dev",
            inventory,
            state_coordinator=coordinator,
        )

        def _fake_time() -> float:
            return 1000.0

        monkeypatch.setattr(coord_module.time, "time", _fake_time)
        monkeypatch.setattr(coord_module, "time_mod", _fake_time)

        await energy.async_refresh()

        snapshot = coordinator.domain_view.get_energy_snapshot()
        assert snapshot is not None
        metric = coordinator.domain_view.get_energy_metric("htr", "A")
        assert metric is not None
        assert metric.energy_kwh == pytest.approx(1.0)

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
        node_list = list(build_node_inventory(nodes))
        inventory = Inventory("dev", node_list)
        coord = StateCoordinator(
            hass,
            client,
            30,
            "dev",
            build_device_metadata_payload("dev", name="Device"),
            nodes=None,
            inventory=inventory,
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
        assert _state_payload(coord, "htr", "A") == {"mode": "auto"}
        assert _state_payload(coord, "acm", "B") == {"mode": "auto"}
        assert dev["inventory"] is inventory
        assert dev["inventory"].addresses_by_type["htr"] == ["A"]
        assert dev["inventory"].addresses_by_type["acm"] == ["B"]
        assert "nodes" not in dev
        assert "nodes_by_type" not in dev
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
        inventory = _inventory_from_nodes("dev", nodes)
        coord = StateCoordinator(
            hass,
            client,
            30,
            "dev",
            build_device_metadata_payload("dev", name="Device"),
            nodes,
            inventory=inventory,
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
        assert _state_payload(coord, "htr", "A") == {"mode": "auto"}
        assert _state_payload(coord, "htr", "C") == {"mode": "eco"}
        assert _state_payload(coord, "acm", "B") == {"mode": "charge"}
        assert dev["inventory"] is inventory
        assert dev["inventory"].addresses_by_type["htr"] == ["A", "C"]
        assert dev["inventory"].addresses_by_type["acm"] == ["B"]
        assert "nodes" not in dev

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
        inventory = _inventory_from_nodes("dev", nodes)
        coord = StateCoordinator(
            hass,
            client,
            30,
            "dev",
            build_device_metadata_payload("dev", name="Device"),
            nodes,
            inventory=inventory,
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
        assert _state_payload(coord, "htr", "B") == {"mode": "auto"}

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
        inventory = _inventory_from_nodes("dev", nodes)
        coord = StateCoordinator(
            hass,
            client,
            30,
            "dev",
            build_device_metadata_payload("dev", name=" Device "),
            nodes,
            inventory=inventory,
        )

        updates: list[dict[str, dict[str, Any]]] = []

        def _set_updated_data(self, data: dict[str, dict[str, Any]]) -> None:
            updates.append(data)
            self.data = data

        coord.async_set_updated_data = types.MethodType(  # type: ignore[attr-defined]
            _set_updated_data,
            coord,
        )

        coord.data = {
            "dev": {
                "settings": {"htr": {"A": {"mode": "manual"}}},
                "inventory": inventory,
            }
        }
        await coord.async_refresh_heater("")
        client.get_node_settings.assert_not_called()
        assert updates == []

        coord.data = {
            "dev": {
                "settings": {"htr": {}},
                "inventory": inventory,
            }
        }
        await coord.async_refresh_heater("A")
        client.get_node_settings.assert_called_once_with("dev", ("htr", "A"))
        assert updates == []

    asyncio.run(_run())


def test_register_pending_setting_normalizes_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = types.SimpleNamespace(get_node_settings=AsyncMock())
    hass = HomeAssistant()
    nodes = {"nodes": [{"addr": "1", "type": "htr"}]}
    coord = _state_coordinator_from_nodes(
        hass,
        client,
        30,
        "dev",
        build_device_metadata_payload("dev", name="Device"),
        nodes,
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
    nodes = {"nodes": [{"addr": "1", "type": "htr"}]}
    coord = _state_coordinator_from_nodes(
        hass,
        client,
        30,
        "dev",
        build_device_metadata_payload("dev", name="Device"),
        nodes,
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
    nodes = {"nodes": [{"addr": "2", "type": "acm"}]}
    coord = _state_coordinator_from_nodes(
        hass,
        client,
        30,
        "dev",
        build_device_metadata_payload("dev", name="Device"),
        nodes,
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
    nodes = {"nodes": [{"addr": "3", "type": "htr"}]}
    coord = _state_coordinator_from_nodes(
        hass,
        client,
        30,
        "dev",
        build_device_metadata_payload("dev", name="Device"),
        nodes,
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
    nodes = {"nodes": [{"addr": "4", "type": "htr"}]}
    coord = _state_coordinator_from_nodes(
        hass,
        client,
        30,
        "dev",
        build_device_metadata_payload("dev", name="Device"),
        nodes,
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
        inventory = _inventory_from_nodes("dev", nodes)
        coord = _state_coordinator_from_nodes(
            hass,
            client,
            15,
            "dev",
            build_device_metadata_payload("dev", name=" Device "),
            nodes,
        )
        coord.update_gateway_connection(
            status="connected",
            connected=True,
            last_event_at=None,
            healthy_since=None,
            healthy_minutes=None,
            last_payload_at=None,
            last_heartbeat_at=None,
            payload_stale=None,
            payload_stale_after=None,
            idle_restart_pending=None,
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
        assert dev["model"] is None
        assert "nodes" not in dev
        assert dev["connected"] is True
        assert _state_payload(coord, "htr", "A") == {"mode": "auto"}
        assert isinstance(dev.get("inventory"), coord_module.Inventory)
        assert dev["inventory"].addresses_by_type["htr"] == ["A", "B"]

        await coord.async_refresh_heater("B")
        assert client.get_node_settings.await_args_list[-1].args == (
            "dev",
            ("htr", "B"),
        )
        assert len(updates) == 2
        second = updates[-1]["dev"]
        assert _state_payload(coord, "htr", "A") == {"mode": "auto"}
        assert _state_payload(coord, "htr", "B") == {"mode": "eco"}
        assert isinstance(second.get("inventory"), coord_module.Inventory)
        assert second["inventory"].addresses_by_type["htr"] == ["A", "B"]
        assert second["inventory"].addresses_by_type["acm"] == ["C"]
        assert _state_payload(coord, "acm", "C") is None

    asyncio.run(_run())


def test_refresh_heater_handles_tuple_and_acm() -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_node_settings = AsyncMock(return_value={"mode": "auto"})

        hass = HomeAssistant()
        nodes_payload = {"nodes": [{"addr": "3", "type": "acm"}]}
        inventory_container = coord_module.Inventory(
            "dev",
            [AccumulatorNode(name="Acc", addr="3")],
        )
        coord = StateCoordinator(
            hass,
            client,
            30,
            "dev",
            build_device_metadata_payload("dev", name="Device"),
            nodes=None,
            inventory=inventory_container,
        )
        store = coord._state_store or coord._ensure_state_store(inventory_container)
        assert store is not None
        store.apply_full_snapshot("acm", "3", {"prev": True})

        updates: list[dict[str, dict[str, Any]]] = []

        def _set_updated_data(self, data: dict[str, dict[str, Any]]) -> None:
            updates.append(data)
            self.data = data

        coord.async_set_updated_data = types.MethodType(  # type: ignore[attr-defined]
            _set_updated_data,
            coord,
        )

        await coord.async_refresh_heater(("acm", "3"))

        client.get_node_settings.assert_awaited_once()
        assert updates, "Expected coordinator data to be updated"
        latest = updates[-1]["dev"]
        assert latest["inventory"] is inventory_container
        addrs = latest["inventory"].addresses_by_type["acm"]
        assert addrs == ["3"]
        assert _state_payload(coord, "acm", "3") == {"mode": "auto"}

    asyncio.run(_run())


def test_async_refresh_heater_adds_missing_type() -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_node_settings = AsyncMock(return_value={"mode": "eco"})

        hass = HomeAssistant()
        nodes = {
            "nodes": [
                {"addr": "A", "type": "htr"},
                {"addr": "B", "type": "acm"},
            ]
        }
        inventory = coord_module.Inventory(
            "dev",
            [
                HeaterNode(name="Heater", addr="A"),
                AccumulatorNode(name="Accumulator", addr="B"),
            ],
        )
        coord = StateCoordinator(
            hass,
            client,
            30,
            "dev",
            build_device_metadata_payload("dev", name="Device"),
            nodes=None,
            inventory=inventory,
        )

        store = coord._state_store or coord._ensure_state_store(inventory)
        assert store is not None
        store.apply_full_snapshot("htr", "A", {"mode": "manual"})

        await coord.async_refresh_heater(("acm", "B"))

        dev_data = coord.data["dev"]
        assert dev_data["inventory"] is inventory
        assert "B" in dev_data["inventory"].addresses_by_type["acm"]
        assert _state_payload(coord, "acm", "B") == {"mode": "eco"}

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
        coord = _state_coordinator_from_nodes(
            hass,
            client,
            45,
            "dev",
            {"name": " Device "},
            nodes,
        )
        coord.update_gateway_connection(
            status="connected",
            connected=True,
            last_event_at=None,
            healthy_since=None,
            healthy_minutes=None,
            last_payload_at=None,
            last_heartbeat_at=None,
            payload_stale=None,
            payload_stale_after=None,
            idle_restart_pending=None,
        )
        inventory = coord._ensure_inventory()

        updates: list[dict[str, dict[str, Any]]] = []

        def _set_updated_data(self, data: dict[str, dict[str, Any]]) -> None:
            updates.append(data)
            self.data = data

        coord.async_set_updated_data = types.MethodType(  # type: ignore[attr-defined]
            _set_updated_data,
            coord,
        )

        store = coord._state_store or coord._ensure_state_store(inventory)
        assert store is not None
        coord.data = coord._device_record()  # type: ignore[attr-defined]

        await coord.async_refresh_heater("A")

        assert updates, "Expected async_set_updated_data to be called"
        result = updates[-1]["dev"]
        client.get_node_settings.assert_called_once_with("dev", ("htr", "A"))
        assert result["name"] == "Device"
        assert result["model"] is None
        assert "nodes" not in result
        assert result["connected"] is True
        assert _state_payload(coord, "htr", "A") == {"mode": "heat"}
        assert result["inventory"] is inventory
        assert result["inventory"].addresses_by_type["acm"] == ["B"]

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
        coord = _state_coordinator_from_nodes(
            hass,
            client,
            30,
            "dev",
            build_device_metadata_payload("dev", name="Device"),
            nodes,
        )
        inventory = coord._ensure_inventory()

        updates: list[dict[str, dict[str, Any]]] = []

        def _set_updated_data(self, data: dict[str, dict[str, Any]]) -> None:
            updates.append(data)
            self.data = data

        coord.async_set_updated_data = types.MethodType(  # type: ignore[attr-defined]
            _set_updated_data,
            coord,
        )

        coord.data = {
            "dev": {
                "settings": {"htr": {}},
                "inventory": inventory,
            }
        }
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


def test_state_coordinator_async_update_data_reuses_previous() -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_node_settings = AsyncMock(return_value={"mode": "eco"})

        hass = HomeAssistant()
        nodes = {
            "nodes": [{"type": "acm", "addr": "7"}, {"type": "htr", "addr": "legacy"}]
        }
        inventory_nodes = list(build_node_inventory(nodes))
        inventory = Inventory("dev", inventory_nodes)
        coord = StateCoordinator(
            hass,
            client,
            30,
            "dev",
            build_device_metadata_payload("dev", name=" Device "),
            nodes=None,
            inventory=inventory,
        )

        coord.update_nodes(nodes, inventory=inventory)
        coord.data = {
            "dev": {
                "inventory": inventory,
            }
        }
        store = coord._state_store or coord._ensure_state_store(inventory)
        assert store is not None
        store.apply_full_snapshot("acm", "7", {"prev": True})
        store.apply_full_snapshot("htr", "legacy", {"mode": "auto"})

        result = await coord._async_update_data()

        assert client.get_node_settings.await_count == 2
        dev_data = result["dev"]
        assert _state_payload(coord, "acm", "7") == {"mode": "eco"}
        assert _state_payload(coord, "htr", "legacy") == {"mode": "eco"}
        assert dev_data["inventory"] is inventory
        assert dev_data["inventory"].addresses_by_type.get("htr") == ["legacy"]

    asyncio.run(_run())


def test_async_refresh_heater_updates_cache() -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_node_settings = AsyncMock(return_value={"mode": "heat"})

        hass = HomeAssistant()
        nodes = {"nodes": [{"type": "htr", "addr": "A"}]}
        inventory_nodes = list(build_node_inventory(nodes))
        inventory = Inventory("dev", inventory_nodes)
        coord = StateCoordinator(
            hass,
            client,
            30,
            "dev",
            {"name": " Device "},
            nodes=None,
            inventory=inventory,
        )

        coord.update_nodes(nodes, inventory=inventory)

        await coord.async_refresh_heater("A")

        dev_data = coord.data["dev"]
        assert _state_payload(coord, "htr", "A") == {"mode": "heat"}
        assert dev_data["inventory"] is inventory
        assert dev_data["inventory"].addresses_by_type["htr"] == ["A"]

    asyncio.run(_run())


def test_async_update_data_skips_non_dict_sections() -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_node_settings = AsyncMock(return_value={"mode": "heat"})

        hass = HomeAssistant()
        nodes = {"nodes": [{"addr": "B", "type": "acm"}]}
        coord = _state_coordinator_from_nodes(
            hass,
            client,
            30,
            "dev",
            build_device_metadata_payload("dev", name="Device"),
            nodes,
        )

        inventory = coord._ensure_inventory()
        store = coord._state_store or coord._ensure_state_store(inventory)
        assert store is not None
        store.apply_full_snapshot("acm", "B", {"mode": "auto"})
        coord.data = coord._device_record()  # type: ignore[attr-defined]

        result = await coord._async_update_data()

        dev_data = result["dev"]
        assert _state_payload(coord, "acm", "B") == {"mode": "heat"}
        assert isinstance(dev_data.get("inventory"), coord_module.Inventory)
        assert dev_data["inventory"].addresses_by_type["acm"] == ["B"]
        assert client.get_node_settings.await_count == 1

    asyncio.run(_run())


def test_counter_reset(
    monkeypatch: pytest.MonkeyPatch,
    inventory_from_map: Callable[
        [Mapping[str, Iterable[str]] | None, str], coord_module.Inventory
    ],
) -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_node_samples = AsyncMock(
            side_effect=[
                [{"t": 1000, "counter": "5.0"}],
                [{"t": 1900, "counter": "1.0"}],
            ]
        )

        hass = HomeAssistant()
        inventory = inventory_from_map({"htr": ["A"]}, dev_id="1")
        coord = EnergyStateCoordinator(hass, client, "1", inventory)

        fake_time = 1000.0

        def _fake_time() -> float:
            return fake_time

        monkeypatch.setattr(coord_module.time, "time", _fake_time)
        monkeypatch.setattr(coord_module, "time_mod", _fake_time)

        await coord.async_refresh()
        fake_time = 1900.0
        await coord.async_refresh()

        assert _energy_metric(coord, "htr", "A") == pytest.approx(0.001)
        assert _power_metric(coord, "htr", "A") is None

    asyncio.run(_run())


def test_energy_processing_consistent_between_poll_and_ws(
    monkeypatch: pytest.MonkeyPatch,
    inventory_from_map: Callable[
        [Mapping[str, Iterable[str]] | None, str], coord_module.Inventory
    ],
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
        inventory = inventory_from_map({"htr": ["A"]})
        poll_coord = EnergyStateCoordinator(hass, poll_client, "dev", inventory)

        await poll_coord.async_refresh()
        await poll_coord.async_refresh()

        poll_energy = _energy_metric(poll_coord, "htr", "A")
        poll_power = _power_metric(poll_coord, "htr", "A")

        ws_client = types.SimpleNamespace()
        ws_client.get_node_samples = AsyncMock(
            return_value=[{"t": 1000.0, "counter": 1200.0}]
        )

        ws_coord = EnergyStateCoordinator(hass, ws_client, "dev", inventory)

        await ws_coord.async_refresh()

        ws_coord.handle_ws_samples(
            "dev",
            {"htr": {"A": {"samples": [{"t": 1600.0, "counter": 2400.0}]}}},
        )

        assert _energy_metric(ws_coord, "htr", "A") == pytest.approx(poll_energy)
        assert _power_metric(ws_coord, "htr", "A") == pytest.approx(poll_power)
        assert ws_coord._last[("htr", "A")] == poll_coord._last[("htr", "A")]

    asyncio.run(_run())


def test_merge_samples_for_window_updates_energy(
    inventory_from_map: Callable[
        [Mapping[str, Iterable[str]] | None, str], coord_module.Inventory
    ],
) -> None:
    """Normalised hourly samples should merge into coordinator caches."""

    async def _run() -> None:
        hass = HomeAssistant()
        client = types.SimpleNamespace()
        client.get_node_samples = AsyncMock(return_value=[])
        inventory = inventory_from_map({"htr": ["A"], "pmo": ["M"]})
        coord = EnergyStateCoordinator(hass, client, "dev", inventory)
        await coord.async_refresh()

        start = datetime(2023, 3, 27, 7, 0, tzinfo=UTC)
        await coord.merge_samples_for_window(
            "dev",
            {
                ("htr", "A"): [
                    {"ts": start, "energy_wh": 1_200.0},
                    {"ts": start + timedelta(hours=1), "energy_wh": 2_400.0},
                ],
                ("pmo", "M"): [
                    {"ts": start + timedelta(minutes=30), "energy_wh": 500.0},
                ],
            },
        )

        assert _energy_metric(coord, "htr", "A") == pytest.approx(2.4)
        assert _power_metric(coord, "htr", "A") == pytest.approx(1_200.0)
        assert _energy_metric(coord, "pmo", "M") == pytest.approx(0.5)
        last_point = coord._last[("htr", "A")]
        assert last_point[0] == pytest.approx((start + timedelta(hours=1)).timestamp())
        assert last_point[1] == pytest.approx(2.4)

    asyncio.run(_run())


def test_energy_samples_missing_fields(
    inventory_from_map: Callable[
        [Mapping[str, Iterable[str]] | None, str], coord_module.Inventory
    ],
) -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_node_samples = AsyncMock(return_value=[{"t": 1000, "counter": None}])

        hass = HomeAssistant()
        inventory = inventory_from_map({"htr": ["A"]})
        coord = EnergyStateCoordinator(hass, client, "dev", inventory)

        await coord.async_refresh()
        snapshot = coerce_snapshot(coord.data)
        assert snapshot is not None
        assert snapshot.metrics_for_type("htr") == {}

    asyncio.run(_run())


def test_energy_samples_invalid_strings(
    inventory_from_map: Callable[
        [Mapping[str, Iterable[str]] | None, str], coord_module.Inventory
    ],
) -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_node_samples = AsyncMock(
            return_value=[{"t": " ", "counter": "garbage"}]
        )

        hass = HomeAssistant()
        inventory = inventory_from_map({"htr": ["A"]})
        coord = EnergyStateCoordinator(hass, client, "dev", inventory)

        await coord.async_refresh()
        snapshot = coerce_snapshot(coord.data)
        assert snapshot is not None
        assert snapshot.metrics_for_type("htr") == {}

    asyncio.run(_run())


def test_energy_coordinator_alias_creates_canonical_bucket(
    inventory_from_map: Callable[
        [Mapping[str, Iterable[str]] | None, str], coord_module.Inventory
    ],
) -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_node_samples = AsyncMock(return_value=[])

        hass = HomeAssistant()
        inventory = inventory_from_map({"pmo": ["01"]}, dev_id="dev")
        coord = EnergyStateCoordinator(hass, client, "dev", inventory)

        result = await coord._async_update_data()

        snapshot = coerce_snapshot(result)
        assert snapshot is not None
        assert snapshot.metrics_for_type("pmo") == {}

    asyncio.run(_run())


def test_update_interval_constant(
    inventory_from_map: Callable[
        [Mapping[str, Iterable[str]] | None, str], coord_module.Inventory
    ],
) -> None:
    hass = HomeAssistant()
    client = types.SimpleNamespace()
    inventory = inventory_from_map({"htr": ["A"]}, dev_id="1")
    coord = EnergyStateCoordinator(hass, client, "1", inventory)
    assert coord.update_interval == HTR_ENERGY_UPDATE_INTERVAL


def test_ws_samples_update_defers_polling(
    monkeypatch: pytest.MonkeyPatch,
    inventory_from_map: Callable[
        [Mapping[str, Iterable[str]] | None, str], coord_module.Inventory
    ],
) -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_node_samples = AsyncMock(
            return_value=[{"t": 0.0, "counter": 1000.0}]
        )

        hass = HomeAssistant()
        inventory = inventory_from_map({"htr": ["A"]})
        coord = EnergyStateCoordinator(hass, client, "dev", inventory)

        fake_time = 0.0

        def _fake_time() -> float:
            return fake_time

        monkeypatch.setattr(coord_module.time, "time", _fake_time)
        monkeypatch.setattr(coord_module, "time_mod", _fake_time)

        await coord.async_refresh()

        assert _energy_metric(coord, "htr", "A") == pytest.approx(1.0)
        assert _power_metric(coord, "htr", "A") is None

        client.get_node_samples.reset_mock()
        client.get_node_samples.return_value = [{"t": 7200.0, "counter": 3000.0}]

        fake_time = 3600.0
        coord.handle_ws_samples(
            "dev",
            {"htr": {" A ": {"samples": [{"t": 3600.0, "counter": 2000.0}]}}},
            lease_seconds=300.0,
        )

        assert _energy_metric(coord, "htr", "A") == pytest.approx(2.0)
        assert _power_metric(coord, "htr", "A") == pytest.approx(1000.0)
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


def test_should_skip_poll_conditions(
    monkeypatch: pytest.MonkeyPatch,
    inventory_from_map: Callable[
        [Mapping[str, Iterable[str]] | None, str], coord_module.Inventory
    ],
) -> None:
    """Exercise websocket-driven polling skip decisions."""

    hass = HomeAssistant()
    client = types.SimpleNamespace()
    inventory = inventory_from_map({"htr": ["A"]})
    coord = EnergyStateCoordinator(hass, client, "dev", inventory)

    assert coord._should_skip_poll() is False

    coord.data = None
    coord._ws_deadline = None
    assert coord._should_skip_poll() is False

    coord.data = build_empty_snapshot("dev")
    coord._ws_deadline = 10.0
    monkeypatch.setattr(coord_module, "time_mod", lambda: 15.0)
    assert coord._should_skip_poll() is False

    coord.data = build_empty_snapshot("dev")
    coord._ws_deadline = 20.0
    monkeypatch.setattr(coord_module, "time_mod", lambda: 5.0)
    assert coord._should_skip_poll() is True


def test_async_update_data_uses_cached_samples(
    monkeypatch: pytest.MonkeyPatch,
    inventory_from_map: Callable[
        [Mapping[str, Iterable[str]] | None, str], coord_module.Inventory
    ],
) -> None:
    """Ensure websocket freshness skips API polling and returns cached data."""

    async def _run() -> None:
        hass = HomeAssistant()
        client = types.SimpleNamespace()
        inventory = inventory_from_map({"htr": ["A"]})
        coord = EnergyStateCoordinator(hass, client, "dev", inventory)

        node_id = DomainNodeId(DomainNodeType.HEATER, "A")
        cached_metrics = EnergyNodeMetrics(
            energy_kwh=1.0,
            power_w=None,
            source="ws",
            ts=0.0,
        )
        coord.data = EnergySnapshot(
            dev_id="dev",
            metrics={node_id: cached_metrics},
            updated_at=0.0,
            ws_deadline=100.0,
        )
        coord._ws_deadline = 100.0

        monkeypatch.setattr(coord_module, "time_mod", lambda: 50.0)

        cached = await coord._async_update_data()
        assert cached == coord.data

        coord.data = ["bad"]  # type: ignore[assignment]
        monkeypatch.setattr(coord, "_should_skip_poll", lambda: True)
        assert await coord._async_update_data() == build_empty_snapshot(
            "dev", ws_deadline=coord._ws_deadline
        )

    asyncio.run(_run())


def test_ws_margin_seconds_bounds(
    inventory_from_map: Callable[
        [Mapping[str, Iterable[str]] | None, str], coord_module.Inventory
    ],
) -> None:
    """Verify the websocket margin respects defaults and upper bounds."""

    hass = HomeAssistant()
    client = types.SimpleNamespace()
    inventory = inventory_from_map({"htr": ["A"]})
    coord = EnergyStateCoordinator(hass, client, "dev", inventory)

    coord._ws_lease = -1
    assert coord._ws_margin_seconds() == coord._ws_margin_default

    coord._ws_lease = 10_000.0
    assert coord._ws_margin_seconds() == 600.0


def test_handle_ws_samples_skips_empty_points(
    inventory_from_map: Callable[
        [Mapping[str, Iterable[str]] | None, str], coord_module.Inventory
    ],
) -> None:
    """Ensure empty websocket samples do not update coordinator state."""

    hass = HomeAssistant()
    client = types.SimpleNamespace()
    inventory = inventory_from_map({"htr": ["A"]})
    coord = EnergyStateCoordinator(hass, client, "dev", inventory)
    coord.data = build_empty_snapshot("dev")

    coord.handle_ws_samples(
        "dev",
        {"htr": {"A": [{"samples": []}, {"samples": []}]}},
    )

    assert coord._last == {}


def test_pmo_samples_scale_and_power(
    inventory_from_map: Callable[
        [Mapping[str, Iterable[str]] | None, str], coord_module.Inventory
    ],
) -> None:
    """Power monitor counters should convert from watt-seconds to kWh."""

    hass = HomeAssistant()
    client = types.SimpleNamespace()
    inventory = inventory_from_map({"pmo": ["M"]})
    coord = EnergyStateCoordinator(hass, client, "dev", inventory)
    coord.data = build_empty_snapshot("dev")
    energy_bucket: dict[str, float] = {}
    power_bucket: dict[str, float] = {}
    coord._last = {("pmo", "M"): (1000.0, 1.0)}

    coord._process_energy_sample(
        "pmo",
        "M",
        4600.0,
        7_200_000.0,
        energy_bucket,
        power_bucket,
    )

    assert energy_bucket["M"] == pytest.approx(2.0)
    assert power_bucket["M"] == pytest.approx(1000.0)


def test_heater_energy_samples_empty_on_api_error(
    inventory_from_map: Callable[
        [Mapping[str, Iterable[str]] | None, str], coord_module.Inventory
    ],
) -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_node_samples = AsyncMock(side_effect=ClientError("fail"))

        hass = HomeAssistant()
        inventory = inventory_from_map({"htr": ["A"]}, dev_id="1")
        coord = EnergyStateCoordinator(
            hass,
            client,
            "1",
            inventory,
        )

        await coord.async_refresh()
        snapshot = coerce_snapshot(coord.data)
        assert snapshot is not None
        assert snapshot.metrics_for_type("htr") == {}
        assert coord._last == {}

    asyncio.run(_run())


def test_heater_energy_client_error_update_failed(
    monkeypatch: pytest.MonkeyPatch,
    inventory_from_map: Callable[
        [Mapping[str, Iterable[str]] | None, str], coord_module.Inventory
    ],
) -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_node_samples = AsyncMock(
            return_value=[{"t": 1000, "counter": "1.0"}]
        )

        hass = HomeAssistant()
        inventory = inventory_from_map({"htr": ["A"]}, dev_id="1")
        coord = EnergyStateCoordinator(
            hass,
            client,
            "1",
            inventory,
        )

        def _raise_client_error(_value: Any) -> float:
            raise ClientError("bad")

        monkeypatch.setattr(coord_module, "float_or_none", _raise_client_error)

        with pytest.raises(UpdateFailed, match="API error: bad"):
            await coord.async_refresh()

    asyncio.run(_run())


def test_energy_coordinator_handles_rate_limit_per_node(
    inventory_from_map: Callable[
        [Mapping[str, Iterable[str]] | None, str], coord_module.Inventory
    ],
) -> None:
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

        inventory = inventory_from_map({"htr": ["A"], "acm": ["B"]}, dev_id="dev")
        coord = EnergyStateCoordinator(
            hass,
            client,
            "dev",
            inventory,
        )

        await coord.async_refresh()

        assert _energy_metric(coord, "htr", "A") == pytest.approx(0.5)
        assert _energy_metric(coord, "acm", "B") is None
        assert ("acm", "B") not in coord._last
        assert ("htr", "A") in coord._last

    asyncio.run(_run())


def test_state_coordinator_update_nodes_uses_provided_inventory(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], coord_module.Inventory
    ],
) -> None:
    hass = HomeAssistant()
    client = types.SimpleNamespace()
    nodes = {"nodes": [{"addr": "A", "type": "htr"}]}
    provided_nodes = [Node(name="Heater", addr="A", node_type="htr")]
    provided_inventory = coord_module.Inventory("dev", provided_nodes)

    inventory = inventory_builder("dev", {})
    coord = StateCoordinator(
        hass,
        client,
        30,
        "dev",
        build_device_metadata_payload("dev", name="Device"),
        nodes=None,
        inventory=inventory,
    )

    coord.update_nodes(nodes, inventory=provided_inventory)

    inventory = coord._inventory
    assert inventory is provided_inventory
    assert inventory.nodes[0] is provided_nodes[0]


def test_energy_state_coordinator_requires_inventory(
    inventory_from_map: Callable[
        [Mapping[str, Iterable[str]] | None, str], coord_module.Inventory
    ],
) -> None:
    hass = HomeAssistant()
    client = types.SimpleNamespace()
    inventory = inventory_from_map({"htr": ["A"], "acm": ["B"], "pmo": ["M"]})
    coord = EnergyStateCoordinator(hass, client, "dev", inventory)

    resolved = coord._resolve_inventory()
    targets = list(coord._iter_energy_targets(resolved))
    assert targets == [("htr", "A"), ("acm", "B"), ("pmo", "M")]
    assert resolved.sample_alias_map(
        include_types=coord_module.ENERGY_NODE_TYPES,
        restrict_to=coord_module.ENERGY_NODE_TYPES,
    ) == {
        "htr": "htr",
        "acm": "acm",
        "pmo": "pmo",
        "power_monitor": "pmo",
        "power_monitors": "pmo",
    }

    with pytest.raises(TypeError):
        coord.update_addresses(None)

    with pytest.raises(TypeError):
        coord.update_addresses(object())  # type: ignore[arg-type]


def test_energy_state_coordinator_rejects_missing_inventory() -> None:
    hass = HomeAssistant()
    client = types.SimpleNamespace()

    with pytest.raises(TypeError):
        EnergyStateCoordinator(hass, client, "dev", None)  # type: ignore[arg-type]


def test_coordinator_rate_limit_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        async def _raise_rate_limit(*_args: Any, **_kwargs: Any) -> Any:
            raise BackendRateLimitError("429")

        client = types.SimpleNamespace()
        client.get_node_settings = AsyncMock(side_effect=_raise_rate_limit)

        hass = HomeAssistant()
        nodes = {"nodes": [{"addr": "A", "type": "htr"}, {"addr": "B", "type": "htr"}]}
        coord = _state_coordinator_from_nodes(
            hass,
            client,
            30,
            "1",
            {},
            nodes,
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
        coord = _state_coordinator_from_nodes(
            hass,
            client,
            30,
            "1",
            {},
            nodes,
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


def test_ws_driven_refresh(
    monkeypatch: pytest.MonkeyPatch,
    inventory_from_map: Callable[
        [Mapping[str, Iterable[str]] | None, str], coord_module.Inventory
    ],
) -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_node_samples = AsyncMock(
            return_value=[{"t": 1000, "counter": "1.0"}]
        )

        hass = HomeAssistant()
        inventory = inventory_from_map({"htr": ["A"]}, dev_id="1")
        coord = EnergyStateCoordinator(hass, client, "1", inventory)

        await coord.async_refresh()
        assert _energy_metric(coord, "htr", "A") == pytest.approx(0.001)

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

        assert _energy_metric(coord, "htr", "A") == pytest.approx(0.002)

    asyncio.run(_run())


def test_energy_poll_preserves_cached_samples(
    inventory_from_map: Callable[
        [Mapping[str, Iterable[str]] | None, str], coord_module.Inventory
    ],
) -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_node_samples = AsyncMock(
            side_effect=[
                [{"t": 1_000, "counter": 1_000}],
                [{"t": 1_600, "counter": 2_000}],
                [],
            ]
        )

        hass = HomeAssistant()
        inventory = inventory_from_map({"htr": ["A"]})
        coord = EnergyStateCoordinator(
            hass,
            client,
            "dev",
            inventory,
        )

        await coord.async_refresh()
        await coord.async_refresh()

        energy_after_second = _energy_metric(coord, "htr", "A")
        power_after_second = _power_metric(coord, "htr", "A")

        assert energy_after_second == pytest.approx(2.0)
        assert power_after_second == pytest.approx(6_000.0)

        await coord.async_refresh()

        assert _energy_metric(coord, "htr", "A") == pytest.approx(energy_after_second)
        assert _power_metric(coord, "htr", "A") == pytest.approx(power_after_second)

    asyncio.run(_run())


def test_coordinator_timeout() -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_node_settings = AsyncMock(side_effect=TimeoutError)

        hass = HomeAssistant()
        nodes = {"nodes": [{"addr": "A", "type": "htr"}]}
        coord = _state_coordinator_from_nodes(
            hass,
            client,
            30,
            "1",
            {},
            nodes,
        )

        with pytest.raises(UpdateFailed, match="API timeout"):
            await coord.async_refresh()

    asyncio.run(_run())


def test_heater_energy_timeout(
    inventory_from_map: Callable[
        [Mapping[str, Iterable[str]] | None, str], coord_module.Inventory
    ],
) -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_node_samples = AsyncMock(side_effect=asyncio.TimeoutError)

        hass = HomeAssistant()
        inventory = inventory_from_map({"htr": ["A"]}, dev_id="1")
        coord = EnergyStateCoordinator(
            hass,
            client,
            "1",
            inventory,
        )

        with pytest.raises(UpdateFailed, match="API timeout"):
            await coord.async_refresh()

    asyncio.run(_run())
