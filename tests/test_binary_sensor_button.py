# ruff: noqa: D100, D101, D102, D103, D105, D107, INP001
from __future__ import annotations

import asyncio
import logging
import types
from typing import Any, Callable, Iterable, Iterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from conftest import FakeCoordinator

from homeassistant.components.button import ButtonEntity
from homeassistant.core import HomeAssistant

import custom_components.termoweb.binary_sensor as binary_sensor_module
import custom_components.termoweb.button as button_module
from custom_components.termoweb.entities import button as entities_button_module
import custom_components.termoweb.heater as heater_module
from custom_components.termoweb import identifiers as identifiers_module
from custom_components.termoweb.const import DOMAIN
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
DisplayFlashButton = button_module.DisplayFlashButton


def test_binary_sensor_setup_and_dispatch(
    heater_hass_data,
) -> None:
    async def _run() -> None:
        hass = HomeAssistant()
        entry = types.SimpleNamespace(entry_id="entry-1")
        dev_id = "device-123"

        inventory = Inventory(dev_id, [])

        coordinator = FakeCoordinator(
            hass,
            dev_id=dev_id,
            inventory=inventory,
            dev={
                "name": "Living Room",
                "model": "TW-GW",
            },
        )

        coordinator.update_gateway_connection(
            status="healthy",
            connected=True,
            last_event_at=171.0,
            healthy_since=111.0,
            healthy_minutes=42.0,
            last_payload_at=170.0,
            last_heartbeat_at=169.0,
            payload_stale=False,
            payload_stale_after=120.0,
            idle_restart_pending=False,
        )

        heater_hass_data(
            hass,
            entry.entry_id,
            dev_id,
            coordinator,
            extra={"version": "2.1.0"},
            inventory=inventory,
        )

        added: list = []

        def _add_entities(entities):
            added.extend(entities)

        await async_setup_binary_sensor_entry(hass, entry, _add_entities)

        assert len(added) == 1
        entity = added[0]
        assert isinstance(entity, GatewayOnlineBinarySensor)

        entity.hass = hass
        await entity.async_added_to_hass()

        assert entity.is_on is True
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
            "ws_last_event_at": 171.0,
            "ws_healthy_minutes": 42.0,
        }

        await entity.async_will_remove_from_hass()

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
    inventory = Inventory("dev", [])

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

        inventory = Inventory(dev_id, [])

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
            entities_button_module,
            "resolve_boost_runtime_minutes",
            lambda *_: 180,
        )
        monkeypatch.setattr(
            button_module,
            "resolve_boost_temperature",
            lambda *_args, **_kwargs: 22.5,
        )
        monkeypatch.setattr(
            entities_button_module,
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

        context = _make_boost_context(entry_id, dev_id, addr=addr, name="Hallway")
        coordinator = FakeCoordinator(
            hass,
            dev_id=dev_id,
            data={
                dev_id: {
                    "settings": {"acm": {addr: {"boost_active": True}}},
                }
            },
            inventory=context.inventory,
        )
        button = AccumulatorBoostCancelButton(
            coordinator,
            context,
            _metadata_for("cancel"),
        )
        button.hass = hass
        button.async_write_ha_state = MagicMock()

        await button.async_added_to_hass()

        assert button.available is True

        assert coordinator.apply_entity_patch(
            "acm", addr, lambda cur: setattr(cur, "boost_active", False)
        )
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
    inventory = Inventory(dev_id, [canonical, HeaterNode(name="Heater", addr="2")])

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


def test_button_setup_adds_flash_display_buttons_for_all_brands(
    heater_hass_data,
) -> None:
    async def _run() -> None:
        hass = HomeAssistant()
        entry = types.SimpleNamespace(entry_id="entry-flash-setup")
        dev_id = "device-flash-setup"
        inventory = Inventory(
            dev_id,
            [
                HeaterNode(name="Heater", addr="1"),
                AccumulatorNode(name="Accumulator", addr="2"),
            ],
        )
        coordinator = FakeCoordinator(hass, dev_id=dev_id, inventory=inventory, data={})
        backend = types.SimpleNamespace(set_node_display_select=AsyncMock())

        for brand in ("termoweb", "ducaheat", "tevolve"):
            heater_hass_data(
                hass,
                entry.entry_id,
                dev_id,
                coordinator,
                inventory=inventory,
                extra={"brand": brand, "backend": backend},
            )

            added: list[ButtonEntity] = []

            def _add_entities(entities: list[ButtonEntity]) -> None:
                added.extend(entities)

            await async_setup_button_entry(hass, entry, _add_entities)

            flash_buttons = [
                entity for entity in added if isinstance(entity, DisplayFlashButton)
            ]
            assert len(flash_buttons) == 2

    asyncio.run(_run())


def test_flash_display_button_calls_select_endpoint(heater_hass_data) -> None:
    async def _run() -> None:
        hass = HomeAssistant()
        entry = types.SimpleNamespace(entry_id="entry-flash-press")
        dev_id = "device-flash-press"
        coordinator = types.SimpleNamespace(hass=hass, _inventory=Inventory(dev_id, []))
        backend = types.SimpleNamespace(set_node_display_select=AsyncMock())

        heater_hass_data(
            hass,
            entry.entry_id,
            dev_id,
            coordinator,
            extra={"brand": "termoweb", "backend": backend},
            inventory=Inventory(dev_id, [HeaterNode(name="Heater", addr="7")]),
        )

        context = button_module.DisplayFlashContext(
            entry_id=entry.entry_id,
            dev_id=dev_id,
            node_type="htr",
            addr="7",
            name="Heater",
        )
        button = DisplayFlashButton(coordinator, context)
        button.hass = hass

        await button.async_press()

        backend.set_node_display_select.assert_awaited_once_with(
            dev_id,
            ("htr", "7"),
            select=True,
        )

    asyncio.run(_run())


def test_flash_display_button_keeps_home_assistant_context_attribute() -> None:
    coordinator = types.SimpleNamespace(hass=None, _inventory=Inventory("dev", []))
    context = button_module.DisplayFlashContext(
        entry_id="entry",
        dev_id="dev",
        node_type="htr",
        addr="1",
        name="Heater",
    )

    button = DisplayFlashButton(coordinator, context)

    assert not hasattr(button, "_context")
    assert button._flash_context is context


def _make_boost_context(
    entry_id: str,
    dev_id: str,
    *,
    addr: str = "2",
    name: str = "Living Room",
) -> AccumulatorBoostContext:
    inventory = Inventory(dev_id, [AccumulatorNode(name=name, addr=addr)])
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


# ---------------------------------------------------------------------------
# New tests targeting uncovered lines
# ---------------------------------------------------------------------------


def test_accumulator_boost_button_aborts_when_no_stored_minutes(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """AccumulatorBoostButton.async_press should abort when minutes is None."""

    async def _run() -> None:
        caplog.set_level(logging.ERROR)
        hass = HomeAssistant()
        entry_id = "entry-no-min"
        dev_id = "device-no-min"
        coordinator = types.SimpleNamespace(hass=hass, data={})
        hass.services = types.SimpleNamespace(async_call=AsyncMock())

        monkeypatch.setattr(
            button_module, "resolve_boost_runtime_minutes", lambda *_: None,
        )
        monkeypatch.setattr(
            entities_button_module, "resolve_boost_runtime_minutes", lambda *_: None,
        )

        context = _make_boost_context(entry_id, dev_id, addr="3", name="Hallway")
        button = AccumulatorBoostButton(
            coordinator, context, _metadata_for("start"),
        )
        button.hass = hass

        caplog.clear()
        await button.async_press()
        assert "Boost start requires a stored duration" in caplog.text
        hass.services.async_call.assert_not_called()

    asyncio.run(_run())


def test_accumulator_boost_button_aborts_when_no_stored_temperature(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """AccumulatorBoostButton.async_press should abort when temperature is None."""

    async def _run() -> None:
        caplog.set_level(logging.ERROR)
        hass = HomeAssistant()
        entry_id = "entry-no-temp"
        dev_id = "device-no-temp"
        coordinator = types.SimpleNamespace(hass=hass, data={})
        hass.services = types.SimpleNamespace(async_call=AsyncMock())

        monkeypatch.setattr(
            button_module, "resolve_boost_runtime_minutes", lambda *_: 120,
        )
        monkeypatch.setattr(
            entities_button_module, "resolve_boost_runtime_minutes", lambda *_: 120,
        )
        monkeypatch.setattr(
            button_module, "resolve_boost_temperature",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            entities_button_module, "resolve_boost_temperature",
            lambda *_args, **_kwargs: None,
        )

        context = _make_boost_context(entry_id, dev_id, addr="4", name="Kitchen")
        button = AccumulatorBoostButton(
            coordinator, context, _metadata_for("start"),
        )
        button.hass = hass

        caplog.clear()
        await button.async_press()
        assert "Boost start requires a stored temperature" in caplog.text
        hass.services.async_call.assert_not_called()

    asyncio.run(_run())


def test_accumulator_boost_cancel_button_calls_cancel_service() -> None:
    """AccumulatorBoostCancelButton.async_press should call the cancel service."""

    async def _run() -> None:
        hass = HomeAssistant()
        entry_id = "entry-cancel-press"
        dev_id = "device-cancel-press"
        addr = "6"
        hass.services = types.SimpleNamespace(async_call=AsyncMock())

        context = _make_boost_context(entry_id, dev_id, addr=addr, name="Office")
        coordinator = FakeCoordinator(
            hass,
            dev_id=dev_id,
            data={
                dev_id: {
                    "settings": {"acm": {addr: {"boost_active": True}}},
                }
            },
            inventory=context.inventory,
        )
        button = AccumulatorBoostCancelButton(
            coordinator, context, _metadata_for("cancel"),
        )
        button.hass = hass
        button.async_write_ha_state = MagicMock()

        await button.async_press()

        hass.services.async_call.assert_awaited_once_with(
            DOMAIN,
            entities_button_module._SERVICE_CANCEL_ACCUMULATOR_BOOST,
            {
                "entry_id": entry_id,
                "dev_id": dev_id,
                "node_type": "acm",
                "addr": addr,
            },
            blocking=True,
        )

    asyncio.run(_run())


def test_accumulator_boost_cancel_button_handles_missing_hass() -> None:
    """Cancel button should silently return when hass is None."""

    async def _run() -> None:
        context = _make_boost_context(
            "entry-guard", "device-guard", addr="7", name="Hallway",
        )
        coordinator = types.SimpleNamespace(hass=None, data={})
        button = AccumulatorBoostCancelButton(
            coordinator, context, _metadata_for("cancel"),
        )
        button.hass = None

        # Should not raise
        await button.async_press()

    asyncio.run(_run())


def test_accumulator_boost_cancel_button_logs_service_not_found(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancel button should log ServiceNotFound errors."""

    async def _run() -> None:
        caplog.set_level(logging.ERROR)
        hass = HomeAssistant()
        coordinator = types.SimpleNamespace(hass=hass, data={})
        hass.services = types.SimpleNamespace(async_call=AsyncMock(
            side_effect=button_module.ServiceNotFound("termoweb", "cancel_boost"),
        ))

        monkeypatch.setattr(
            "homeassistant.helpers.translation.async_get_exception_message",
            lambda *args, **kwargs: "service_not_found",
            raising=False,
        )

        context = _make_boost_context("entry-err", "device-err", addr="8", name="Room")
        button = AccumulatorBoostCancelButton(
            coordinator, context, _metadata_for("cancel"),
        )
        button.hass = hass

        caplog.clear()
        await button.async_press()
        assert "Boost cancel service unavailable" in caplog.text

    asyncio.run(_run())


def test_flash_display_button_device_info_varies_by_node_type() -> None:
    """DisplayFlashButton.device_info should report correct model per node type."""

    coordinator = types.SimpleNamespace(hass=None, _inventory=Inventory("dev", []))

    # Heater node
    htr_context = entities_button_module.DisplayFlashContext(
        entry_id="e", dev_id="dev", node_type="htr", addr="1", name="Heater 1",
    )
    htr_button = DisplayFlashButton(coordinator, htr_context)
    assert htr_button.device_info["model"] == "Heater"

    # Accumulator node
    acm_context = entities_button_module.DisplayFlashContext(
        entry_id="e", dev_id="dev", node_type="acm", addr="2", name="Acc 2",
    )
    acm_button = DisplayFlashButton(coordinator, acm_context)
    assert acm_button.device_info["model"] == "Accumulator"

    # Thermostat node (default fallback)
    thm_context = entities_button_module.DisplayFlashContext(
        entry_id="e", dev_id="dev", node_type="thm", addr="3", name="Thm 3",
    )
    thm_button = DisplayFlashButton(coordinator, thm_context)
    assert thm_button.device_info["model"] == "Thermostat"


def test_flash_display_button_handles_press_error(
    heater_hass_data,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """DisplayFlashButton.async_press should raise HomeAssistantError on failure."""

    async def _run() -> None:
        from homeassistant.exceptions import HomeAssistantError

        hass = HomeAssistant()
        entry = types.SimpleNamespace(entry_id="entry-flash-err")
        dev_id = "device-flash-err"
        backend = types.SimpleNamespace(
            set_node_display_select=AsyncMock(side_effect=RuntimeError("network")),
        )
        coordinator = types.SimpleNamespace(
            hass=hass, _inventory=Inventory(dev_id, []),
        )

        heater_hass_data(
            hass,
            entry.entry_id,
            dev_id,
            coordinator,
            extra={"brand": "termoweb", "backend": backend},
            inventory=Inventory(dev_id, [HeaterNode(name="Heater", addr="7")]),
        )

        context = entities_button_module.DisplayFlashContext(
            entry_id=entry.entry_id,
            dev_id=dev_id,
            node_type="htr",
            addr="7",
            name="Heater",
        )
        button = DisplayFlashButton(coordinator, context)
        button.hass = hass

        caplog.set_level(logging.ERROR)
        with pytest.raises(HomeAssistantError, match="Unable to flash"):
            await button.async_press()
        assert "Display flash failed" in caplog.text

    asyncio.run(_run())


def test_flash_display_button_skips_press_without_hass() -> None:
    """DisplayFlashButton.async_press should do nothing when hass is None."""

    async def _run() -> None:
        coordinator = types.SimpleNamespace(hass=None, _inventory=Inventory("dev", []))
        context = entities_button_module.DisplayFlashContext(
            entry_id="e", dev_id="dev", node_type="htr", addr="1", name="H",
        )
        button = DisplayFlashButton(coordinator, context)
        button.hass = None

        # Should not raise
        await button.async_press()

    asyncio.run(_run())


def test_flash_display_button_available_depends_on_inventory() -> None:
    """DisplayFlashButton.available should reflect inventory membership."""

    inventory = Inventory("dev", [HeaterNode(name="Heater", addr="1")])
    coordinator = types.SimpleNamespace(hass=None, _inventory=inventory)

    context = entities_button_module.DisplayFlashContext(
        entry_id="e", dev_id="dev", node_type="htr", addr="1", name="Heater",
    )
    button = DisplayFlashButton(coordinator, context)
    assert button.available is True

    # Missing node -> not available
    context_missing = entities_button_module.DisplayFlashContext(
        entry_id="e", dev_id="dev", node_type="htr", addr="99", name="Missing",
    )
    button_missing = DisplayFlashButton(coordinator, context_missing)
    assert button_missing.available is False

    # No inventory attr -> not available
    coordinator_empty = types.SimpleNamespace(hass=None)
    button_no_inv = DisplayFlashButton(coordinator_empty, context)
    assert button_no_inv.available is False


def test_build_boost_button_raises_for_unknown_action() -> None:
    """_build_boost_button should raise ValueError for unsupported actions."""

    coordinator = types.SimpleNamespace(hass=None, data={})
    context = _make_boost_context("entry", "dev", addr="1", name="Acc")
    bad_metadata = heater_module.BoostButtonMetadata(
        minutes=60,
        unique_suffix="unknown",
        label="Unknown",
        icon="mdi:help",
        action="unknown",
    )

    with pytest.raises(ValueError, match="Unsupported boost button action"):
        entities_button_module._build_boost_button(bad_metadata, coordinator, context)


def test_accumulator_boost_base_device_info() -> None:
    """AccumulatorBoostButtonBase.device_info should expose accumulator metadata."""

    coordinator = types.SimpleNamespace(hass=None, data={})
    context = _make_boost_context("entry-info", "dev-info", addr="5", name="Storage")
    button = AccumulatorBoostButton(
        coordinator, context, _metadata_for("start"),
    )

    info = button.device_info
    assert info["model"] == "Accumulator"
    assert info["manufacturer"] == "TermoWeb"
    assert ("termoweb", "dev-info", "5") in info["identifiers"]
    assert info["name"] == "Storage"


def test_accumulator_boost_base_coordinator_state() -> None:
    """AccumulatorBoostButtonBase._coordinator_state should resolve domain state."""

    addr = "3"
    context = _make_boost_context("entry-cs", "dev-cs", addr=addr, name="Hall")
    coordinator = FakeCoordinator(
        HomeAssistant(),
        dev_id="dev-cs",
        data={
            "dev-cs": {
                "settings": {"acm": {addr: {"boost_active": False}}},
            }
        },
        inventory=context.inventory,
    )
    button = AccumulatorBoostButton(
        coordinator, context, _metadata_for("start"),
    )

    state = button._coordinator_state()
    assert state is not None
    assert button._coordinator_boost_active() is False


def test_accumulator_boost_base_service_minutes_default_is_none() -> None:
    """AccumulatorBoostButtonBase._service_minutes should default to None."""

    coordinator = types.SimpleNamespace(hass=None, data={})
    context = _make_boost_context("entry-min", "dev-min", addr="1", name="Acc")

    # Using the base class indirectly via AccumulatorBoostCancelButton
    # which does not override _service_minutes
    button = AccumulatorBoostCancelButton(
        coordinator, context, _metadata_for("cancel"),
    )
    assert button._service_minutes is None
