# ruff: noqa: D100, D101, D102, D103, D105, D107, INP001
from __future__ import annotations

import asyncio
import logging
import types
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
from custom_components.termoweb.inventory import Inventory
from custom_components.termoweb.utils import build_gateway_device_info

GatewayOnlineBinarySensor = binary_sensor_module.GatewayOnlineBinarySensor
async_setup_binary_sensor_entry = binary_sensor_module.async_setup_entry
StateRefreshButton = button_module.StateRefreshButton
async_setup_button_entry = button_module.async_setup_entry
AccumulatorBoostButton = button_module.AccumulatorBoostButton
AccumulatorBoostCancelButton = button_module.AccumulatorBoostCancelButton


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
                    "raw": {"model": "TW-GW"},
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
        assert (
            not guard_entity._gateway_dispatcher.is_connected
        )  # pylint: disable=protected-access

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
            "ws_status": "healthy",
            "ws_last_event_at": "2024-05-01T12:00:00Z",
            "ws_healthy_minutes": 42,
            "raw": {"model": "TW-GW"},
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
        assert (
            not entity._gateway_dispatcher.is_connected
        )  # pylint: disable=protected-access

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


def test_button_setup_adds_accumulator_entities(
    monkeypatch: pytest.MonkeyPatch,
    heater_hass_data,
    heater_node_factory,
) -> None:
    async def _run() -> None:
        hass = HomeAssistant()
        entry = types.SimpleNamespace(entry_id="entry-boost")
        dev_id = "device-boost"
        coordinator = types.SimpleNamespace(hass=hass, data={})

        entry_data = heater_hass_data(
            hass,
            entry.entry_id,
            dev_id,
            coordinator,
        )

        acm_node = heater_node_factory("5", node_type="acm")
        acm_skip = heater_node_factory("6", node_type="acm")
        htr_node = heater_node_factory("3", node_type="htr", supports_boost=False)

        inventory = Inventory(
            dev_id,
            {"nodes": []},
            [acm_node, acm_skip, htr_node],
        )
        entry_data["inventory"] = inventory
        entry_data["node_inventory"] = list(inventory.nodes)

        calls: list[str | None] = []

        def fake_iter_boostable(
            nodes_by_type,
            resolve_name,
            *,
            node_types=None,
            accumulators_only=False,
        ):
            assert accumulators_only is True
            assert node_types is None
            for node in nodes_by_type.get("acm", []):
                addr = getattr(node, "addr", None)
                calls.append(addr)
                if addr == acm_node.addr:
                    yield "acm", node, addr, resolve_name("acm", addr)

        monkeypatch.setattr(
            button_module,
            "iter_boostable_heater_nodes",
            fake_iter_boostable,
        )

        custom_metadata = (
            heater_module.BoostButtonMetadata(
                15,
                "15",
                "Quick boost",
                "mdi:flash",
            ),
            heater_module.BoostButtonMetadata(
                None,
                "stop",
                "Stop boost",
                "mdi:flash-off",
            ),
        )

        def fake_iter():
            yield from custom_metadata

        monkeypatch.setattr(
            button_module,
            "iter_boost_button_metadata",
            fake_iter,
        )

        def _fail_prepare(*_args, **_kwargs):
            raise AssertionError("prepare_heater_platform_data should not run")

        monkeypatch.setattr(
            heater_module,
            "prepare_heater_platform_data",
            _fail_prepare,
        )

        added: list = []

        def _add_entities(entities):
            for entity in entities:
                entity.hass = hass
            added.extend(entities)

        await async_setup_button_entry(hass, entry, _add_entities)

        assert len(added) == 1 + len(custom_metadata)
        assert calls == [acm_node.addr, acm_skip.addr]
        assert isinstance(added[0], StateRefreshButton)

        boost_entities = added[1:]
        assert all(isinstance(entity, ButtonEntity) for entity in boost_entities)
        names = [getattr(entity, "_attr_name", None) for entity in boost_entities]
        icons = [
            getattr(entity, "icon", getattr(entity, "_attr_icon", None))
            for entity in boost_entities
        ]
        unique_ids = [
            getattr(entity, "unique_id", getattr(entity, "_attr_unique_id", None))
            for entity in boost_entities
        ]
        expected_names = [item.label for item in custom_metadata]
        expected_icons = [item.icon for item in custom_metadata]
        expected_unique_ids = [
            "{}_{}".format(
                identifiers_module.build_heater_entity_unique_id(
                    dev_id,
                    "acm",
                    acm_node.addr,
                    ":boost",
                ),
                item.unique_suffix,
            )
            for item in custom_metadata
        ]
        assert names == expected_names
        assert icons == expected_icons
        assert unique_ids == expected_unique_ids

    asyncio.run(_run())


def test_button_setup_falls_back_to_prepare_heater_platform_data(
    monkeypatch: pytest.MonkeyPatch,
    heater_hass_data,
    heater_node_factory,
) -> None:
    async def _run() -> None:
        hass = HomeAssistant()
        entry = types.SimpleNamespace(entry_id="entry-fallback")
        dev_id = "device-fallback"
        coordinator = types.SimpleNamespace(hass=hass, data={})

        entry_data = heater_hass_data(
            hass,
            entry.entry_id,
            dev_id,
            coordinator,
        )

        entry_data["inventory"] = {"nodes": []}

        fallback_node = heater_node_factory("7", node_type="acm")
        nodes_by_type = {"acm": [fallback_node]}

        def _resolve_name(node_type: str, addr: str) -> str:
            assert node_type == "acm"
            assert addr == fallback_node.addr
            return f"Accumulator {addr}"

        mock_prepare = MagicMock(
            return_value=((), nodes_by_type, {"acm": [fallback_node.addr]}, _resolve_name)
        )
        monkeypatch.setattr(
            heater_module,
            "prepare_heater_platform_data",
            mock_prepare,
        )

        def _fake_iter(
            nodes_by_type_arg,
            resolve_name,
            *,
            node_types=None,
            accumulators_only=False,
        ):
            assert nodes_by_type_arg is nodes_by_type
            assert resolve_name is _resolve_name
            assert accumulators_only is True
            assert node_types is None
            yield "acm", fallback_node, fallback_node.addr, resolve_name("acm", fallback_node.addr)

        monkeypatch.setattr(
            button_module,
            "iter_boostable_heater_nodes",
            _fake_iter,
        )

        metadata = (
            heater_module.BoostButtonMetadata(
                15,
                "15",
                "Quick boost",
                "mdi:flash",
            ),
        )

        monkeypatch.setattr(
            button_module,
            "iter_boost_button_metadata",
            lambda: iter(metadata),
        )

        added: list = []

        def _add_entities(entities):
            for entity in entities:
                entity.hass = hass
            added.extend(entities)

        await async_setup_button_entry(hass, entry, _add_entities)

        assert mock_prepare.call_count == 1
        call_args, call_kwargs = mock_prepare.call_args
        assert call_args and call_args[0] == entry_data
        assert set(call_kwargs) == {"default_name_simple"}
        assert callable(call_kwargs["default_name_simple"])

        assert len(added) == 1 + len(metadata)
        assert isinstance(added[0], StateRefreshButton)

        boost_entity = added[1]
        assert isinstance(boost_entity, ButtonEntity)
        assert boost_entity.unique_id.endswith("_15")

    asyncio.run(_run())


def test_accumulator_boost_button_triggers_service() -> None:
    async def _run() -> None:
        hass = HomeAssistant()
        entry_id = "entry-trigger"
        dev_id = "device-trigger"
        coordinator = types.SimpleNamespace(hass=hass, data={})
        hass.services = types.SimpleNamespace(async_call=AsyncMock())

        button = AccumulatorBoostButton(
            coordinator,
            entry_id,
            dev_id,
            "2",
            "Living Room",
            "uid-boost-60",
            minutes=60,
            node_type="acm",
        )
        button.hass = hass

        assert button.translation_placeholders == {"minutes": "60"}

        await button.async_press()

        hass.services.async_call.assert_awaited_once_with(
            DOMAIN,
            button_module._SERVICE_REQUEST_ACCUMULATOR_BOOST,
            {
                "entry_id": entry_id,
                "dev_id": dev_id,
                "node_type": "acm",
                "addr": "2",
                "minutes": 60,
            },
            blocking=True,
        )

    asyncio.run(_run())


def test_accumulator_boost_cancel_button_triggers_service_without_minutes() -> None:
    async def _run() -> None:
        hass = HomeAssistant()
        entry_id = "entry-cancel"
        dev_id = "device-cancel"
        coordinator = types.SimpleNamespace(hass=hass, data={})
        hass.services = types.SimpleNamespace(async_call=AsyncMock())

        button = AccumulatorBoostCancelButton(
            coordinator,
            entry_id,
            dev_id,
            "4",
            "Bedroom",
            "uid-cancel",
            node_type="acm",
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
                "addr": "4",
            },
            blocking=True,
        )

    asyncio.run(_run())


def test_accumulator_boost_button_handles_missing_hass() -> None:
    async def _run() -> None:
        hass = HomeAssistant()
        coordinator = types.SimpleNamespace(hass=hass, data={})
        hass.services = types.SimpleNamespace(async_call=AsyncMock())

        button = AccumulatorBoostButton(
            coordinator,
            "entry-no-hass",
            "device-no-hass",
            "8",
            "Kitchen",
            "uid-no-hass",
            minutes=30,
            node_type="acm",
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

        button = AccumulatorBoostButton(
            coordinator,
            entry_id,
            dev_id,
            "10",
            "Office",
            "uid-errors",
            minutes=120,
            node_type="acm",
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


def test_binary_sensor_setup_adds_boost_entities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        hass = HomeAssistant()
        entry = types.SimpleNamespace(entry_id="entry-boost-binary")
        dev_id = "device-boost"
        coordinator = types.SimpleNamespace(hass=hass, data={})

        hass.data = {
            DOMAIN: {entry.entry_id: {"coordinator": coordinator, "dev_id": dev_id}}
        }

        boost_node = types.SimpleNamespace(type="acm", addr="4", name="Boost")
        skip_node = types.SimpleNamespace(type="acm", addr="5", name="Skip")
        inventory = Inventory(
            dev_id,
            {"nodes": []},
            [boost_node, skip_node],
        )
        record = hass.data[DOMAIN][entry.entry_id]
        record["inventory"] = inventory
        record["node_inventory"] = list(inventory.nodes)

        calls: list[str | None] = []

        def fake_iter_boostable(
            nodes_by_type,
            resolve_name,
            *,
            node_types=None,
            accumulators_only=False,
        ):
            assert accumulators_only is False
            assert node_types is None
            for node in nodes_by_type.get("acm", []):
                addr = getattr(node, "addr", None)
                calls.append(addr)
                if addr == boost_node.addr:
                    yield "acm", node, addr, resolve_name("acm", addr)

        monkeypatch.setattr(
            binary_sensor_module,
            "iter_boostable_heater_nodes",
            fake_iter_boostable,
        )

        def _fail_prepare(*_args, **_kwargs):
            raise AssertionError("prepare_heater_platform_data should not run")

        monkeypatch.setattr(
            heater_module,
            "prepare_heater_platform_data",
            _fail_prepare,
        )

        added: list = []

        def _add_entities(entities: list) -> None:
            added.extend(entities)

        await async_setup_binary_sensor_entry(hass, entry, _add_entities)

        assert len(added) == 2
        assert calls == [boost_node.addr, skip_node.addr]
        gateway, boost = added
        assert isinstance(gateway, GatewayOnlineBinarySensor)
        assert isinstance(boost, binary_sensor_module.HeaterBoostActiveBinarySensor)
        assert (
            boost._attr_name == f"{boost_node.name} Boost Active"
        )  # pylint: disable=protected-access

    asyncio.run(_run())
