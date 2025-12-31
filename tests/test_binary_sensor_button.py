# ruff: noqa: D100, D101, D102, D103, D105, D107, INP001
from __future__ import annotations

import asyncio
import logging
import types
from typing import Any, Callable, Iterable, Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conftest import FakeCoordinator

from homeassistant.components.button import ButtonEntity
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_send

import custom_components.termoweb.binary_sensor as binary_sensor_module
import custom_components.termoweb.button as button_module
import custom_components.termoweb.heater as heater_module
from custom_components.termoweb import identifiers as identifiers_module
from custom_components.termoweb.const import DOMAIN, signal_ws_status
from custom_components.termoweb.inventory import (
    AccumulatorNode,
    HeaterNode,
    Inventory,
    InventoryNodeMetadata,
)
from custom_components.termoweb.utils import build_gateway_device_info

GatewayOnlineBinarySensor = binary_sensor_module.GatewayOnlineBinarySensor
async_setup_binary_sensor_entry = binary_sensor_module.async_setup_entry
StateRefreshButton = button_module.StateRefreshButton
async_setup_button_entry = button_module.async_setup_entry
AccumulatorBoostButton = button_module.AccumulatorBoostButton
AccumulatorBoostCancelButton = button_module.AccumulatorBoostCancelButton
AccumulatorBoostContext = button_module.AccumulatorBoostContext


def test_binary_sensor_setup_and_dispatch(
    heater_hass_data,
) -> None:
    async def _run() -> None:
        hass = HomeAssistant()
        entry = types.SimpleNamespace(entry_id="entry-1")
        dev_id = "device-123"

        inventory = Inventory(dev_id, {"nodes": []}, [])

        coordinator = FakeCoordinator(
            hass,
            dev_id=dev_id,
            inventory=inventory,
            data={
                dev_id: {
                    "name": "Living Room",  # attributes
                    "connected": True,
                    "model": "TW-GW",
                }
            },
        )

        heater_hass_data(
            hass,
            entry.entry_id,
            dev_id,
            coordinator,
            ws_state={
                dev_id: {
                    "status": "healthy",
                    "last_event_at": "2024-05-01T12:00:00Z",
                    "healthy_minutes": 42,
                }
            },
            extra={"version": "2.1.0"},
            inventory=inventory,
        )

        added: list = []

        def _add_entities(entities):
            added.extend(entities)

        guard_coordinator = FakeCoordinator(None, data={})
        guard_entity = GatewayOnlineBinarySensor(
            guard_coordinator,
            "guard-entry",
            "guard-device",
        )
        await guard_entity.async_added_to_hass()
        assert not guard_entity._gateway_dispatcher.is_connected  # pylint: disable=protected-access

        await async_setup_binary_sensor_entry(hass, entry, _add_entities)

        assert len(added) == 1
        entity = added[0]
        assert isinstance(entity, GatewayOnlineBinarySensor)

        entity.hass = hass
        with patch.object(
            entity._gateway_dispatcher,
            "subscribe",
            wraps=entity._gateway_dispatcher.subscribe,
        ) as mock_subscribe:
            await entity.async_added_to_hass()

        mock_subscribe.assert_called_once()
        _, call_signal, call_handler = mock_subscribe.call_args[0]
        assert call_signal == signal_ws_status(entry.entry_id)
        assert getattr(call_handler, "__self__", None) is entity
        assert getattr(call_handler, "__func__", None) is getattr(
            entity._handle_gateway_dispatcher, "__func__", None
        )

        assert entity.is_on is True
        assert entity._gateway_dispatcher.is_connected  # pylint: disable=protected-access

        info = entity.device_info
        expected_info = build_gateway_device_info(hass, entry.entry_id, dev_id)
        assert info == expected_info

        attrs = entity.extra_state_attributes
        assert attrs == {
            "dev_id": dev_id,
            "name": "Living Room",
            "connected": True,
            "model": "TW-GW",
            "ws_status": "healthy",
            "ws_last_event_at": "2024-05-01T12:00:00Z",
            "ws_healthy_minutes": 42,
        }

        entity.schedule_update_ha_state = MagicMock()
        async_dispatcher_send(
            hass, signal_ws_status(entry.entry_id), {"dev_id": "other"}
        )
        entity.schedule_update_ha_state.assert_not_called()
        async_dispatcher_send(
            hass, signal_ws_status(entry.entry_id), {"dev_id": dev_id}
        )
        entity.schedule_update_ha_state.assert_called_once_with()

        await entity.async_will_remove_from_hass()
        assert not entity._gateway_dispatcher.is_connected  # pylint: disable=protected-access

    asyncio.run(_run())


def test_binary_sensor_setup_requires_inventory(heater_hass_data) -> None:
    async def _run() -> None:
        hass = HomeAssistant()
        entry = types.SimpleNamespace(entry_id="entry-missing")
        dev_id = "device-missing"
        coordinator = FakeCoordinator(hass, dev_id=dev_id, inventory=None, data={})

        heater_hass_data(
            hass,
            entry.entry_id,
            dev_id,
            coordinator,
        )

        with pytest.raises(ValueError):
            await async_setup_binary_sensor_entry(hass, entry, lambda _: None)

    asyncio.run(_run())


def test_iter_boostable_inventory_nodes_uses_inventory_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = Inventory("dev", {"nodes": []}, [])

    metadata = [
        InventoryNodeMetadata(
            node_type="acm",
            addr="01",
            name="Accumulator 01",
            node=types.SimpleNamespace(supports_boost=lambda: False),
        ),
        InventoryNodeMetadata(
            node_type="htr",
            addr=" 2 ",
            name="Heater 2",
            node=types.SimpleNamespace(supports_boost=lambda: True),
        ),
        InventoryNodeMetadata(
            node_type="",
            addr="3",
            name="Invalid",
            node=types.SimpleNamespace(supports_boost=lambda: True),
        ),
    ]

    def _fake_iter(
        self: Inventory,
        *,
        node_types: Iterable[str] | None = None,
        default_name_simple: Callable[[str], str] | None = None,
    ) -> Iterator[InventoryNodeMetadata]:
        assert self is inventory
        yield from metadata

    monkeypatch.setattr(Inventory, "iter_nodes_metadata", _fake_iter)

    results = list(binary_sensor_module._iter_boostable_inventory_nodes(inventory))

    assert results == [("htr", "2", "Heater 2")]


def test_refresh_button_device_info_and_press(heater_hass_data) -> None:
    async def _run() -> None:
        hass = HomeAssistant()
        entry = types.SimpleNamespace(entry_id="entry-button")
        dev_id = "device-123"
        coordinator = types.SimpleNamespace(
            hass=hass,
            async_request_refresh=AsyncMock(),
        )

        inventory = Inventory(dev_id, {"nodes": []}, [])

        heater_hass_data(
            hass,
            entry.entry_id,
            dev_id,
            coordinator,
            inventory=inventory,
        )

        added: list = []
        seen_ids: set[str] = set()
        call_sizes: list[int] = []

        def _add_entities(entities):
            call_sizes.append(len(entities))
            for entity in entities:
                uid = getattr(entity, "unique_id", None)
                if uid is None:
                    uid = getattr(entity, "_attr_unique_id", None)
                if uid in seen_ids:
                    continue
                seen_ids.add(str(uid))
                entity.hass = hass
                added.append(entity)

        await async_setup_button_entry(hass, entry, _add_entities)
        assert call_sizes == [1]
        assert len(added) == 1

        button_entity = added[0]
        assert isinstance(button_entity, StateRefreshButton)

        await async_setup_button_entry(hass, entry, _add_entities)
        assert call_sizes == [1, 1]
        assert len(added) == 1
        assert len(seen_ids) == 1

        info = button_entity.device_info
        expected_info = build_gateway_device_info(
            hass,
            entry.entry_id,
            dev_id,
        )
        assert info == expected_info

        await button_entity.async_press()
        coordinator.async_request_refresh.assert_awaited_once()


def test_accumulator_boost_button_triggers_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        hass = HomeAssistant()
        entry_id = "entry-trigger"
        dev_id = "device-trigger"
        coordinator = types.SimpleNamespace(hass=hass, data={})
        hass.services = types.SimpleNamespace(async_call=AsyncMock())

        monkeypatch.setattr(
            button_module,
            "resolve_boost_runtime_minutes",
            lambda *_: 180,
        )
        monkeypatch.setattr(
            button_module,
            "resolve_boost_temperature",
            lambda *_args, **_kwargs: 22.5,
        )

        context = _make_boost_context(entry_id, dev_id, addr="2", name="Living Room")
        button = AccumulatorBoostButton(
            coordinator,
            context,
            _metadata_for("start"),
        )
        button.hass = hass

        await button.async_press()

        hass.services.async_call.assert_awaited_once_with(
            DOMAIN,
            button_module._SERVICE_REQUEST_ACCUMULATOR_BOOST,
            {
                "entry_id": entry_id,
                "dev_id": dev_id,
                "node_type": "acm",
                "addr": "2",
                "minutes": 180,
                "temperature": 22.5,
            },
            blocking=True,
        )

    asyncio.run(_run())


def test_accumulator_boost_cancel_button_tracks_availability() -> None:
    async def _run() -> None:
        hass = HomeAssistant()
        entry_id = "entry-cancel"
        dev_id = "device-cancel"
        addr = "5"

        coordinator = FakeCoordinator(
            hass,
            dev_id=dev_id,
            data={
                dev_id: {
                    "settings": {
                        "acm": {
                            addr: {"boost_active": True},
                        }
                    }
                }
            },
        )

        context = _make_boost_context(entry_id, dev_id, addr=addr, name="Hallway")
        button = AccumulatorBoostCancelButton(
            coordinator,
            context,
            _metadata_for("cancel"),
        )
        button.hass = hass
        button.async_write_ha_state = MagicMock()

        await button.async_added_to_hass()

        assert button.available is True

        coordinator.data[dev_id]["settings"]["acm"][addr]["boost_active"] = False
        for listener in list(getattr(coordinator, "listeners", [])):
            listener()

        assert button.available is False
        button.async_write_ha_state.assert_called()

    asyncio.run(_run())


def test_accumulator_boost_button_ignores_press_without_hass() -> None:
    async def _run() -> None:
        class AsyncCallStub:
            def __init__(self) -> None:
                self.called = False

            async def __call__(self, *_args, **_kwargs) -> None:
                self.called = True
                raise AssertionError("async_call should not be awaited without hass")

        async_call = AsyncCallStub()
        coordinator = types.SimpleNamespace(
            hass=types.SimpleNamespace(
                services=types.SimpleNamespace(async_call=async_call)
            ),
            data={},
        )

        context = _make_boost_context(
            "entry-guard",
            "device-guard",
            addr="8",
            name="Hallway",
        )
        button = AccumulatorBoostButton(
            coordinator,
            context,
            _metadata_for("start"),
        )

        button.hass = None

        await button.async_press()

        assert async_call.called is False

    asyncio.run(_run())


def test_accumulator_boost_button_handles_missing_hass() -> None:
    async def _run() -> None:
        hass = HomeAssistant()
        coordinator = types.SimpleNamespace(hass=hass, data={})
        hass.services = types.SimpleNamespace(async_call=AsyncMock())

        context = _make_boost_context(
            "entry-no-hass",
            "device-no-hass",
            addr="8",
            name="Kitchen",
        )
        button = AccumulatorBoostButton(
            coordinator,
            context,
            _metadata_for("start"),
        )
        button.hass = None

        await button.async_press()

        hass.services.async_call.assert_not_called()

    asyncio.run(_run())


def test_accumulator_boost_button_logs_service_errors(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        caplog.set_level(logging.ERROR)
        hass = HomeAssistant()
        entry_id = "entry-errors"
        dev_id = "device-errors"
        coordinator = types.SimpleNamespace(hass=hass, data={})
        hass.services = types.SimpleNamespace(async_call=AsyncMock())

        monkeypatch.setattr(
            "homeassistant.helpers.translation.async_get_exception_message",
            lambda *args, **kwargs: "service_not_found",
            raising=False,
        )

        monkeypatch.setattr(
            button_module,
            "resolve_boost_runtime_minutes",
            lambda *_: 120,
        )
        monkeypatch.setattr(
            button_module,
            "resolve_boost_temperature",
            lambda *_args, **_kwargs: 25.0,
        )

        context = _make_boost_context(entry_id, dev_id, addr="10", name="Office")
        button = AccumulatorBoostButton(
            coordinator,
            context,
            _metadata_for("start"),
        )
        button.hass = hass

        hass.services.async_call.side_effect = button_module.ServiceNotFound(
            "termoweb", "boost"
        )
        await button.async_press()
        assert "Boost helper service unavailable" in caplog.text

        hass.services.async_call.reset_mock()
        hass.services.async_call.side_effect = button_module.HomeAssistantError("boom")
        await button.async_press()
        assert "Boost helper service failed" in caplog.text

    asyncio.run(_run())


def test_state_refresh_button_direct_press_and_info() -> None:
    async def _run() -> None:
        hass = HomeAssistant()
        coordinator = types.SimpleNamespace(
            hass=hass,
            async_request_refresh=AsyncMock(),
        )

        button = StateRefreshButton(coordinator, "entry-direct", "device-direct")
        button.hass = hass

        info = button.device_info
        expected = build_gateway_device_info(hass, "entry-direct", "device-direct")
        assert info == expected

        await button.async_press()
        coordinator.async_request_refresh.assert_awaited_once()

    asyncio.run(_run())


def test_iter_accumulator_contexts_uses_inventory_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry_id = "entry-meta"
    dev_id = "device-meta"
    canonical = AccumulatorNode(name="Accumulator A", addr="1")
    inventory = Inventory(
        dev_id, {"nodes": []}, [canonical, HeaterNode(name="Heater", addr="2")]
    )

    metadata = [
        InventoryNodeMetadata(
            node_type="acm",
            addr="1",
            name="Accumulator A",
            node=canonical,
        ),
        InventoryNodeMetadata(
            node_type="acm",
            addr="3",
            name="Accumulator B",
            node=types.SimpleNamespace(addr="3", type="acm"),
        ),
        InventoryNodeMetadata(
            node_type="htr",
            addr="2",
            name="Heater",
            node=HeaterNode(name="Heater", addr="2"),
        ),
    ]

    def _fake_iter(
        self: Inventory,
        *,
        node_types: Iterable[str] | None = None,
        default_name_simple: Callable[[str], str] | None = None,
    ) -> Iterator[InventoryNodeMetadata]:
        assert self is inventory
        yield from metadata

    monkeypatch.setattr(Inventory, "iter_nodes_metadata", _fake_iter)

    contexts = list(button_module._iter_accumulator_contexts(entry_id, inventory))

    assert len(contexts) == 1
    context = contexts[0]
    assert context.entry_id == entry_id
    assert context.inventory is inventory
    assert context.node is canonical


def _make_boost_context(
    entry_id: str,
    dev_id: str,
    *,
    addr: str = "2",
    name: str = "Living Room",
) -> AccumulatorBoostContext:
    inventory = Inventory(
        dev_id, {"nodes": []}, [AccumulatorNode(name=name, addr=addr)]
    )
    nodes = inventory.nodes_by_type.get("acm", ())
    assert nodes, "inventory must expose at least one accumulator"
    node = nodes[0]
    assert isinstance(node, AccumulatorNode)
    return AccumulatorBoostContext.from_inventory(entry_id, inventory, node)


def _metadata_for(action: str) -> heater_module.BoostButtonMetadata:
    for metadata in heater_module.BOOST_BUTTON_METADATA:
        if metadata.action == action:
            return metadata
    raise AssertionError(f"metadata for action={action!r} not found")
