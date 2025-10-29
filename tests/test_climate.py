from __future__ import annotations

import asyncio
import datetime as dt
import logging
from collections import deque
from collections.abc import Coroutine
import types
from typing import Any, Callable, Deque, Iterable, Mapping, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from conftest import FakeCoordinator, _install_stubs, build_coordinator_device_state

import custom_components.termoweb.inventory as inventory_module
from custom_components.termoweb.inventory import Inventory

_install_stubs()

from custom_components.termoweb import climate as climate_module
from custom_components.termoweb.heater import DEFAULT_BOOST_DURATION
from custom_components.termoweb.backend.ducaheat import DucaheatRESTClient
from custom_components.termoweb.const import (
    BRAND_DUCAHEAT,
    BRAND_TERMOWEB,
    DOMAIN,
    signal_ws_data,
)
from custom_components.termoweb.inventory import HeaterNode, Inventory, build_node_inventory
from homeassistant.components.climate import HVACAction, HVACMode
from homeassistant.const import ATTR_TEMPERATURE
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import entity_platform as entity_platform_module
from homeassistant.helpers.entity_platform import EntityPlatform
from homeassistant.helpers import dispatcher as dispatcher_module
from homeassistant.util import dt as dt_util

HeaterClimateEntity = climate_module.HeaterClimateEntity
async_setup_entry = climate_module.async_setup_entry


@pytest.fixture
def climate_inventory(
    inventory_builder: Callable[[str, Mapping[str, Any] | None, Iterable[Any] | None], Inventory]
) -> Callable[[str, Mapping[str, Any]], Inventory]:
    """Return helper that constructs inventory containers for climate tests."""

    def _factory(dev_id: str, raw_nodes: Mapping[str, Any]) -> Inventory:
        return inventory_builder(dev_id, raw_nodes, build_node_inventory(raw_nodes))

    return _factory


def _reset_environment() -> None:
    _install_stubs()
    entity_platform_module._set_current_platform(EntityPlatform())
    dispatcher_module._dispatch_map = {}
    dt_util.NOW = dt.datetime(2024, 1, 1, 0, 0, tzinfo=dt.timezone.utc)
    FakeCoordinator.instances.clear()


def _make_coordinator(
    hass: HomeAssistant,
    dev_id: str,
    record: dict[str, Any],
    *,
    client: Any | None = None,
    inventory: Any | None = None,
) -> FakeCoordinator:
    base_record = dict(record)

    raw_nodes = base_record.get("inventory_payload")
    if raw_nodes is None:
        inventory_container = base_record.get("inventory")
        if isinstance(inventory_container, Inventory):
            raw_nodes = getattr(inventory_container, "payload", None)
    nodes_payload = raw_nodes if isinstance(raw_nodes, Mapping) else None

    raw_settings: dict[str, dict[str, Any]] = {}
    raw_addresses: dict[str, Iterable[Any]] = {}
    section_extras: dict[str, dict[str, Any]] = {}

    def _merge_settings(node_type: str, bucket: Mapping[str, Any] | None) -> None:
        if not isinstance(bucket, Mapping):
            return
        target = raw_settings.setdefault(node_type, {})
        for addr, data in bucket.items():
            target.setdefault(addr, data)

    def _merge_addresses(node_type: str, addrs: Iterable[Any] | None) -> None:
        if addrs is None or isinstance(addrs, (str, bytes)):
            return
        existing = list(raw_addresses.get(node_type, ()))
        existing.extend(addrs)
        raw_addresses[node_type] = existing

    def _merge_section(node_type: str, section: Mapping[str, Any] | None) -> None:
        if not isinstance(section, Mapping):
            return
        extras = {
            key: value
            for key, value in section.items()
            if key not in {"settings", "addrs"}
        }
        _merge_settings(node_type, section.get("settings"))
        addrs_value = section.get("addrs")
        if isinstance(addrs_value, Iterable) and not isinstance(addrs_value, (str, bytes)):
            _merge_addresses(node_type, addrs_value)
        if extras:
            section_extras.setdefault(node_type, {}).update(extras)

    settings_section = base_record.get("settings")
    if isinstance(settings_section, Mapping):
        for node_type, bucket in settings_section.items():
            _merge_settings(str(node_type), bucket if isinstance(bucket, Mapping) else None)

    addresses_section = base_record.get("addresses_by_type")
    inventory_obj = base_record.get("inventory")
    if isinstance(inventory_obj, Inventory):
        addresses_section = inventory_obj.addresses_by_type
    if isinstance(addresses_section, Mapping):
        for node_type, addrs in addresses_section.items():
            _merge_addresses(str(node_type), addrs if isinstance(addrs, Iterable) else None)

    nodes_by_type = base_record.get("nodes_by_type")
    if isinstance(nodes_by_type, Mapping):
        for node_type, section in nodes_by_type.items():
            _merge_section(str(node_type), section if isinstance(section, Mapping) else None)

    for candidate_type in ("htr", "acm", "heater"):
        _merge_section(candidate_type, base_record.get(candidate_type))

    remaining_keys = {
        key: value
        for key, value in base_record.items()
        if key
        not in {
            "nodes",
            "nodes_by_type",
            "settings",
            "addresses_by_type",
            "htr",
            "acm",
            "heater",
        }
    }

    rebuilt_record = build_coordinator_device_state(
        nodes=nodes_payload,
        settings=raw_settings or None,
        addresses=raw_addresses or None,
        sections=section_extras or None,
        extra=remaining_keys or None,
    )

    normalised = FakeCoordinator._normalise_device_record(rebuilt_record)

    effective_inventory = inventory
    if not isinstance(effective_inventory, Inventory) and nodes_payload is not None:
        node_list = list(build_node_inventory(nodes_payload))
        effective_inventory = Inventory(dev_id, nodes_payload, node_list)

    return FakeCoordinator(
        hass,
        client=client,
        dev_id=dev_id,
        dev=normalised,
        nodes=normalised.get("inventory_payload", {}),
        inventory=effective_inventory,
        data={dev_id: normalised},
    )


# -------------------- Helpers for tests --------------------


def test_termoweb_heater_is_heater_node() -> None:
    _reset_environment()
    hass = HomeAssistant()
    dev_id = "dev"
    coordinator_record = build_coordinator_device_state(
        nodes={},
        settings={"htr": {}},
    )
    coordinator = _make_coordinator(hass, dev_id, coordinator_record)

    heater = HeaterClimateEntity(
        coordinator,
        "entry",
        "dev",
        "1",
        " Living Room ",
    )

    assert isinstance(heater, HeaterNode)
    assert heater.type == "htr"
    assert heater.addr == "1"
    assert heater.name == "Living Room"


def test_heater_climate_entity_normalizes_node_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_environment()
    hass = HomeAssistant()
    dev_id = "dev-acm"
    coordinator_record = build_coordinator_device_state(
        nodes={},
        settings={"htr": {}},
    )
    coordinator = _make_coordinator(hass, dev_id, coordinator_record)

    calls: list[tuple[object, dict[str, Any]]] = []

    original_normalize = climate_module.normalize_node_type

    def _record_normalize(value, **kwargs):
        calls.append((value, kwargs))
        return original_normalize(value, **kwargs)

    monkeypatch.setattr(climate_module, "normalize_node_type", _record_normalize)

    heater = HeaterClimateEntity(
        coordinator,
        "entry",
        dev_id,
        "1",
        "Heater",
        node_type=" ACM ",
    )

    assert heater.type == "acm"
    assert getattr(heater, "_node_type", "") == "acm"
    assert heater._attr_unique_id == f"{DOMAIN}:{dev_id}:acm:{heater._addr}"
    assert calls[0][0] in {"htr", None}
    assert calls[1][0] == " ACM "


def test_async_setup_entry_creates_entities(
    climate_inventory: Callable[[str, Mapping[str, Any]], Inventory],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        _reset_environment()
        hass = HomeAssistant()
        entry_id = "entry"
        dev_id = "dev1"
        nodes = {
            "nodes": [
                {"type": "htr", "addr": "A1", "name": " Living Room "},
                {"type": "HTR", "addr": "B2"},
                {"type": "acm", "addr": "C3", "name": " Basement Accumulator "},
                {"type": "other", "addr": "X"},
            ]
        }
        inventory = climate_inventory(dev_id, nodes)
        coordinator_record = build_coordinator_device_state(
            nodes=nodes,
            settings={
                "htr": {"A1": {}, "B2": {}},
                "acm": {"C3": {"units": "C"}},
            },
            extra={"version": "3.1.4"},
        )
        coordinator = _make_coordinator(
            hass,
            dev_id,
            coordinator_record,
            client=AsyncMock(),
            inventory=inventory,
        )

        hass.data = {
            DOMAIN: {
                entry_id: {
                    "coordinator": coordinator,
                    "dev_id": dev_id,
                    "client": AsyncMock(),
                    "nodes": nodes,
                    "inventory": inventory,
                    "version": "3.1.4",
                    "brand": BRAND_TERMOWEB,
                }
            }
        }

        added: list[HeaterClimateEntity] = []

        def _async_add_entities(entities: list[HeaterClimateEntity]) -> None:
            added.extend(entities)

        platform = EntityPlatform()
        entity_platform_module._set_current_platform(platform)

        entry = types.SimpleNamespace(entry_id=entry_id)
        calls: list[Mapping[str, Any] | None] = []

        original_resolver = Inventory.require_from_context

        def _record_inventory(*args: Any, **kwargs: Any) -> Inventory:
            calls.append(kwargs.get("container"))
            return original_resolver(*args, **kwargs)

        monkeypatch.setattr(
            inventory_module.Inventory,
            "require_from_context",
            staticmethod(_record_inventory),
        )
        await async_setup_entry(hass, entry, _async_add_entities)

        assert len(added) == 3
        assert calls and calls[0] is hass.data[DOMAIN][entry_id]
        entities_by_addr = {entity._addr: entity for entity in added}
        assert set(entities_by_addr) == {"A1", "B2", "C3"}
        assert isinstance(entities_by_addr["A1"], HeaterClimateEntity)
        assert isinstance(entities_by_addr["B2"], HeaterClimateEntity)
        acc = entities_by_addr["C3"]
        assert isinstance(acc, climate_module.AccumulatorClimateEntity)
        assert acc.available
        names = {entity._addr: entity._attr_name for entity in added}
        assert names["A1"] == "Living Room"
        assert names["B2"] == "Heater B2"
        assert names["C3"] == "Basement Accumulator"

        registered = [name for name, _, _ in platform.registered]
        assert registered == [
            "set_schedule",
            "set_preset_temperatures",
            "set_acm_preset",
            "start_boost",
            "cancel_boost",
        ]

        for entity in added:
            info = entity.device_info
            assert info["identifiers"] == {(DOMAIN, dev_id, entity._addr)}
            assert info["manufacturer"] == "TermoWeb"
            expected_model = "Accumulator"
            if getattr(entity, "_node_type", "htr") != "acm":
                expected_model = "Heater"
            assert info["model"] == expected_model
            assert info["via_device"] == (DOMAIN, dev_id)

        schedule_name, _, schedule_handler = platform.registered[0]
        preset_name, _, preset_handler = platform.registered[1]
        assert schedule_name == "set_schedule"
        assert preset_name == "set_preset_temperatures"

        schedule_prog = [0] * 168
        first = entities_by_addr["A1"]
        first.async_set_schedule = AsyncMock()
        await schedule_handler(first, ServiceCall({"prog": schedule_prog}))
        first.async_set_schedule.assert_awaited_once_with(schedule_prog)

        first.async_set_preset_temperatures = AsyncMock()
        await preset_handler(first, ServiceCall({"ptemp": [18.0, 19.0, 20.0]}))
        first.async_set_preset_temperatures.assert_awaited_once_with(
            ptemp=[18.0, 19.0, 20.0]
        )

        second = entities_by_addr["B2"]
        second.async_set_preset_temperatures = AsyncMock()
        await preset_handler(
            second,
            ServiceCall({"cold": 15.0, "night": 18.0, "day": 20.0}),
        )
        second.async_set_preset_temperatures.assert_awaited_once_with(
            cold=15.0, night=18.0, day=20.0
        )

        acm_entity = entities_by_addr["C3"]
        _, _, acm_preset_handler = platform.registered[2]
        _, _, start_boost_handler = platform.registered[3]
        _, _, cancel_boost_handler = platform.registered[4]

        acm_entity.async_set_acm_preset = AsyncMock()
        await acm_preset_handler(
            acm_entity,
            ServiceCall({"minutes": 75, "temperature": 22.5}),
        )
        acm_entity.async_set_acm_preset.assert_awaited_once_with(
            minutes=75,
            temperature=22.5,
        )

        acm_entity.async_start_boost = AsyncMock()
        await start_boost_handler(acm_entity, ServiceCall({"minutes": 30}))
        acm_entity.async_start_boost.assert_awaited_once_with(minutes=30)
        acm_entity.async_start_boost.reset_mock()
        await start_boost_handler(acm_entity, ServiceCall({}))
        acm_entity.async_start_boost.assert_awaited_once_with(minutes=None)

        acm_entity.async_cancel_boost = AsyncMock()
        await cancel_boost_handler(acm_entity, ServiceCall({}))
        acm_entity.async_cancel_boost.assert_awaited_once()

        heater_entity = entities_by_addr["A1"]
        heater_entity.async_set_acm_preset = AsyncMock()
        await acm_preset_handler(heater_entity, ServiceCall({}))
        heater_entity.async_set_acm_preset.assert_not_called()

        heater_entity.async_start_boost = AsyncMock()
        await start_boost_handler(heater_entity, ServiceCall({}))
        heater_entity.async_start_boost.assert_not_called()

        heater_entity.async_cancel_boost = AsyncMock()
        await cancel_boost_handler(heater_entity, ServiceCall({}))
        heater_entity.async_cancel_boost.assert_not_called()

    asyncio.run(_run())


def test_accumulator_preferred_boost_defaults_without_hass() -> None:
    """Ensure accumulators fall back to the default boost duration offline."""

    _reset_environment()
    hass = HomeAssistant()
    dev_id = "dev-acc"
    record = build_coordinator_device_state(nodes={}, settings={"htr": {}})
    coordinator = _make_coordinator(hass, dev_id, record)

    entity = climate_module.AccumulatorClimateEntity(
        coordinator,
        "entry-acc",
        dev_id,
        "01",
        "Accumulator",
        node_type="acm",
    )

    entity.hass = None
    assert entity._preferred_boost_minutes() == DEFAULT_BOOST_DURATION


def test_thermostat_climate_entity_maps_settings(
    climate_inventory: Callable[[str, Mapping[str, Any]], Inventory]
) -> None:
    _reset_environment()
    hass = HomeAssistant()
    dev_id = "dev-thm"
    raw_nodes = {"nodes": [{"type": "thm", "addr": "T1"}]}
    inventory = climate_inventory(dev_id, raw_nodes)

    prog = [0, 0, 0, 1, 1, 1] * 28
    payload = {
        "mode": "manual",
        "state": "on",
        "stemp": "21.5",
        "mtemp": "20.3",
        "units": "C",
        "ptemp": ["16.0", "19.0", "20.0"],
        "prog": prog,
        "batt_level": 5,
    }

    coordinator_record = build_coordinator_device_state(
        nodes=raw_nodes,
        settings={"thm": {"T1": payload}},
    )
    coordinator = _make_coordinator(
        hass,
        dev_id,
        coordinator_record,
        client=AsyncMock(),
        inventory=inventory,
    )

    entity = HeaterClimateEntity(
        coordinator,
        "entry-thm",
        dev_id,
        "T1",
        "Thermostat T1",
        node_type="thm",
        inventory=inventory,
    )

    assert entity.hvac_mode == HVACMode.HEAT
    assert entity.hvac_action == HVACAction.HEATING
    assert entity.current_temperature == pytest.approx(20.3)
    assert entity.target_temperature == pytest.approx(21.5)
    assert entity.extra_state_attributes["prog"] == prog


def test_async_setup_entry_default_names_and_invalid_nodes(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    climate_inventory: Callable[[str, Mapping[str, Any]], Inventory],
) -> None:
    async def _run() -> None:
        _reset_environment()
        hass = HomeAssistant()
        entry_id = "entry-default"
        dev_id = "dev-default"
        raw_nodes = {
            "nodes": [
                {"type": "htr", "addr": "1"},
                {"type": "acm", "addr": "2"},
                {"type": "thm", "addr": "T1"},
                {"type": "pmo", "addr": "P1"},
                {"type": "  ", "addr": "extra"},
                {"type": "htr", "addr": " "},
            ]
        }
        inventory = climate_inventory(dev_id, raw_nodes)

        coordinator_record = build_coordinator_device_state(
            nodes={},
            settings={"htr": {}, "thm": {}},
        )
        coordinator = _make_coordinator(
            hass,
            dev_id,
            coordinator_record,
            client=AsyncMock(),
            inventory=inventory,
        )

        hass.data = {
            DOMAIN: {
                entry_id: {
                    "coordinator": coordinator,
                    "dev_id": dev_id,
                    "client": AsyncMock(),
                    "nodes": raw_nodes,
                    "inventory": inventory,
                }
            }
        }

        added: list[HeaterClimateEntity] = []

        def _add_entities(entities: list[HeaterClimateEntity]) -> None:
            added.extend(entities)

        entity_platform_module._set_current_platform(EntityPlatform())

        calls: list[tuple[str, dict[str, Any]]] = []
        original_helper = climate_module.log_skipped_nodes

        def _mock_helper(
            platform_name: str,
            inventory_or_details: Any,
            *,
            logger: logging.Logger | None = None,
            skipped_types: Iterable[str] = ("pmo",),
        ) -> None:
            calls.append((platform_name, inventory_or_details))
            original_helper(
                platform_name,
                inventory_or_details,
                logger=logger or climate_module._LOGGER,
                skipped_types=skipped_types,
            )

        monkeypatch.setattr(climate_module, "log_skipped_nodes", _mock_helper)

        entry = types.SimpleNamespace(entry_id=entry_id)
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger=climate_module._LOGGER.name):
            await async_setup_entry(hass, entry, _add_entities)

        names = sorted(entity._attr_name for entity in added)
        assert names == ["Accumulator 2", "Heater 1", "Thermostat T1"]
        assert all(entity._addr in {"1", "2", "T1"} for entity in added)

        assert calls and calls[0][0] == "climate"
        logged_details = calls[0][1]
        if isinstance(logged_details, tuple):
            logged_nodes = logged_details[0]
        elif hasattr(logged_details, "nodes_by_type"):
            logged_nodes = logged_details.nodes_by_type
        else:
            logged_nodes = {}
        assert "pmo" in logged_nodes
        messages = [record.getMessage() for record in caplog.records]
        assert any(
            "Skipping TermoWeb pmo nodes for climate platform: P1" in message
            for message in messages
        )

    asyncio.run(_run())


def test_async_setup_entry_skips_blank_addresses(
    climate_inventory: Callable[[str, Mapping[str, Any]], Inventory]
) -> None:
    async def _run() -> None:
        _reset_environment()
        hass = HomeAssistant()
        entry_id = "entry-skip"
        dev_id = "dev-skip"
        raw_nodes = {
            "nodes": [
                {"type": "htr", "addr": "  "},
                {"type": "htr", "addr": "7"},
            ]
        }
        inventory = climate_inventory(dev_id, raw_nodes)
        coordinator_data = {"nodes": raw_nodes, "htr": {"settings": {}}}
        coordinator = _make_coordinator(
            hass,
            dev_id,
            coordinator_data,
            client=AsyncMock(),
            inventory=inventory,
        )

        hass.data = {
            DOMAIN: {
                entry_id: {
                    "coordinator": coordinator,
                    "dev_id": dev_id,
                    "client": AsyncMock(),
                    "nodes": raw_nodes,
                    "inventory": inventory,
                }
            }
        }

        added: list[HeaterClimateEntity] = []

        def _add_entities(entities: list[HeaterClimateEntity]) -> None:
            added.extend(entities)

        entry = types.SimpleNamespace(entry_id=entry_id)
        await async_setup_entry(hass, entry, _add_entities)

        assert len(added) == 1
        assert added[0]._attr_unique_id.endswith(":htr:7:climate")

    asyncio.run(_run())


def test_async_setup_entry_creates_accumulator_entity(
    climate_inventory: Callable[[str, Mapping[str, Any]], Inventory]
) -> None:
    async def _run() -> None:
        _reset_environment()
        hass = HomeAssistant()
        entry_id = "entry-acm"
        dev_id = "dev-acm"
        nodes = {"nodes": [{"type": "acm", "addr": "7", "name": "Store"}]}
        inventory = climate_inventory(dev_id, nodes)
        settings = {
            "mode": "boost",
            "state": "idle",
            "mtemp": "19.0",
            "stemp": "21.0",
            "ptemp": ["18.0", "19.0", "20.0"],
            "prog": [0, 1, 2] * 56,
            "units": "C",
        }
        coordinator_data = {
            dev_id: {
                "nodes": nodes,
                "nodes_by_type": {
                    "acm": {"addrs": ["7"], "settings": {"7": dict(settings)}}
                },
                "htr": {"settings": {}},
            }
        }
        coordinator = _make_coordinator(
            hass,
            dev_id,
            coordinator_data[dev_id],
            client=AsyncMock(),
            inventory=inventory,
        )

        client = AsyncMock()
        client.set_node_settings = AsyncMock()

        hass.data = {
            DOMAIN: {
                entry_id: {
                    "coordinator": coordinator,
                    "dev_id": dev_id,
                    "client": client,
                    "nodes": nodes,
                    "inventory": inventory,
                    "brand": BRAND_DUCAHEAT,
                }
            }
        }

        added: list[climate_module.HeaterClimateEntity] = []

        def _async_add_entities(
            entities: list[climate_module.HeaterClimateEntity],
        ) -> None:
            added.extend(entities)

        entry = types.SimpleNamespace(entry_id=entry_id)
        await async_setup_entry(hass, entry, _async_add_entities)

        assert len(added) == 1
        acc = added[0]
        assert isinstance(acc, climate_module.AccumulatorClimateEntity)
        assert acc._attr_unique_id == f"{DOMAIN}:{dev_id}:acm:7:climate"
        assert acc.available
        assert acc.device_info["model"] == "Accumulator"
        assert acc._attr_hvac_modes == [HVACMode.OFF, HVACMode.AUTO]
        assert "boost" not in {
            getattr(mode, "value", str(mode)).lower() for mode in acc._attr_hvac_modes
        }
        assert acc.preset_modes == ["none", "boost"]
        assert "boost" in acc.preset_modes
        assert acc.hvac_mode == HVACMode.AUTO
        assert acc.preset_mode == "boost"

        prog = [0, 1, 2] * 56
        await acc.async_set_schedule(list(prog))
        call = client.set_node_settings.await_args
        assert call.args == (dev_id, ("acm", "7"))
        assert call.kwargs["prog"] == list(prog)
        assert call.kwargs["units"] == "C"
        client.set_node_settings.reset_mock()

        hass.data[DOMAIN][entry_id]["brand"] = BRAND_TERMOWEB

        await acc.async_set_schedule(list(prog))
        call = client.set_node_settings.await_args
        assert call.args == (dev_id, ("acm", "7"))
        assert call.kwargs["prog"] == list(prog)
        assert call.kwargs["units"] == "C"
        client.set_node_settings.reset_mock()

        await acc.async_set_preset_temperatures(ptemp=[18.5, 19.5, 20.5])
        call = client.set_node_settings.await_args
        assert call.kwargs["ptemp"] == [18.5, 19.5, 20.5]
        assert call.kwargs["units"] == "C"
        assert client.set_node_settings.await_count == 1

    asyncio.run(_run())


def test_async_setup_entry_uses_inventory_node_for_boost_detection(
    climate_inventory: Callable[[str, Mapping[str, Any]], Inventory],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        _reset_environment()
        hass = HomeAssistant()
        entry_id = "entry-boost"
        dev_id = "dev-boost"
        nodes = {"nodes": [{"type": "htr", "addr": "1"}]}
        inventory = climate_inventory(dev_id, nodes)

        coordinator = _make_coordinator(
            hass,
            dev_id,
            {"nodes": nodes, "htr": {"settings": {"1": {}}}},
            client=AsyncMock(),
            inventory=inventory,
        )

        record: dict[str, Any] = {
            "coordinator": coordinator,
            "dev_id": dev_id,
            "client": AsyncMock(),
            "nodes": nodes,
            "inventory": inventory,
        }
        hass.data = {DOMAIN: {entry_id: record}}

        node = types.SimpleNamespace(addr="1", type="htr")
        iter_calls: list[Any] = []

        def _iter_metadata(self: climate_module.HeaterPlatformDetails):
            iter_calls.append(node)
            yield ("htr", node, "1", "Boost Heater")

        boost_calls: list[Any] = []

        def _supports_boost(candidate: Any) -> bool:
            boost_calls.append(candidate)
            return True

        monkeypatch.setattr(
            climate_module.HeaterPlatformDetails,
            "iter_metadata",
            _iter_metadata,
        )
        monkeypatch.setattr(climate_module, "supports_boost", _supports_boost)

        added: list[climate_module.HeaterClimateEntity] = []

        def _async_add_entities(
            entities: list[climate_module.HeaterClimateEntity],
        ) -> None:
            added.extend(entities)

        entry = types.SimpleNamespace(entry_id=entry_id)
        await async_setup_entry(hass, entry, _async_add_entities)

        assert iter_calls == [node]
        assert boost_calls == [node]
        assert len(added) == 1
        assert isinstance(added[0], climate_module.AccumulatorClimateEntity)

    asyncio.run(_run())


def test_async_setup_entry_prefers_inventory_node_type(
    climate_inventory: Callable[[str, Mapping[str, Any]], Inventory],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prefer inventory node metadata when classifying heater entities."""
    async def _run() -> None:
        _reset_environment()
        hass = HomeAssistant()
        entry_id = "entry-node-type"
        dev_id = "dev-node-type"
        nodes = {"nodes": [{"type": "htr", "addr": "2"}]}
        inventory = climate_inventory(dev_id, nodes)

        coordinator = _make_coordinator(
            hass,
            dev_id,
            {"nodes": nodes, "htr": {"settings": {"2": {}}}},
            client=AsyncMock(),
            inventory=inventory,
        )

        hass.data = {
            DOMAIN: {
                entry_id: {
                    "coordinator": coordinator,
                    "dev_id": dev_id,
                    "client": AsyncMock(),
                    "nodes": nodes,
                    "inventory": inventory,
                }
            }
        }

        node = types.SimpleNamespace(addr="2", type="acm")

        def _iter_metadata(self: climate_module.HeaterPlatformDetails):
            yield ("htr", node, "2", "Accumulator from Node")

        monkeypatch.setattr(
            climate_module.HeaterPlatformDetails,
            "iter_metadata",
            _iter_metadata,
        )

        def _supports_boost(_: Any) -> bool:
            raise AssertionError("supports_boost should not run when node type is acm")

        monkeypatch.setattr(climate_module, "supports_boost", _supports_boost)

        added: list[climate_module.HeaterClimateEntity] = []

        def _async_add_entities(
            entities: list[climate_module.HeaterClimateEntity],
        ) -> None:
            added.extend(entities)

        entry = types.SimpleNamespace(entry_id=entry_id)
        await async_setup_entry(hass, entry, _async_add_entities)

        assert len(added) == 1
        entity = added[0]
        assert isinstance(entity, climate_module.AccumulatorClimateEntity)
        assert entity._node_type == "acm"
        assert entity._attr_unique_id.endswith(":acm:2:climate")

    asyncio.run(_run())


def test_settings_maps_include_inventory_aliases(
    climate_inventory: Callable[[str, Mapping[str, Any]], Inventory]
) -> None:
    """Ensure optimistic updates touch all alias mappings for an address."""

    _reset_environment()
    hass = HomeAssistant()
    entry_id = "entry-alias"
    dev_id = "dev-alias"
    addr = "A1"
    nodes = {"nodes": [{"type": "htr", "addr": addr}, {"type": "acm", "addr": addr}]}
    inventory = climate_inventory(dev_id, nodes)

    record = build_coordinator_device_state(
        nodes=nodes,
        settings={
            "htr": {addr: {"mode": "auto"}},
            "acm": {addr: {"mode": "auto"}},
        },
        sections={
            "htr": {"settings": {addr: {"mode": "auto"}}},
            "acm": {"settings": {addr: {"mode": "auto"}}},
        },
    )

    coordinator = _make_coordinator(
        hass,
        dev_id,
        record,
        inventory=inventory,
    )

    entity = HeaterClimateEntity(
        coordinator,
        entry_id,
        dev_id,
        addr,
        "Alias Heater",
        node_type="htr",
        inventory=inventory,
    )
    entity.hass = hass
    entity.async_write_ha_state = MagicMock()

    entity._optimistic_update(lambda payload: payload.__setitem__("mode", "manual"))

    device_state = coordinator.data[dev_id]
    assert device_state["settings"]["htr"][addr]["mode"] == "manual"
    assert device_state["settings"]["acm"][addr]["mode"] == "manual"
    assert device_state["htr"]["settings"][addr]["mode"] == "manual"
    assert (
        device_state["nodes_by_type"]["acm"]["settings"][addr]["mode"] == "manual"
    )


def test_accumulator_hvac_mode_reporting() -> None:
    """Ensure accumulator HVAC mode normalisation covers all branches."""

    _reset_environment()
    hass = HomeAssistant()
    entry_id = "entry-acm-hvac"
    dev_id = "dev-acm-hvac"
    addr = "7"
    settings: dict[str, Any] = {"mode": "off", "units": "C"}
    coordinator = _make_coordinator(
        hass,
        dev_id,
        {
            "nodes": {},
            "nodes_by_type": {"acm": {"settings": {addr: settings}}},
            "htr": {"settings": {}},
        },
    )
    hass.data = {
        DOMAIN: {
            entry_id: {
                "coordinator": coordinator,
                "dev_id": dev_id,
                "client": AsyncMock(),
                "brand": BRAND_TERMOWEB,
            }
        }
    }

    entity = climate_module.AccumulatorClimateEntity(
        coordinator,
        entry_id,
        dev_id,
        addr,
        "Accumulator",
        node_type="acm",
    )
    entity.hass = hass

    assert entity._default_mode_for_setpoint() is None
    assert entity._requires_setpoint_with_mode(HVACMode.AUTO) is False
    assert entity._allows_setpoint_in_mode(HVACMode.AUTO) is True
    assert entity.hvac_mode == HVACMode.OFF
    settings["mode"] = "auto"
    assert entity.hvac_mode == HVACMode.AUTO
    settings["mode"] = "boost"
    assert entity.hvac_mode == HVACMode.AUTO
    assert entity.preset_mode == "boost"
    settings["mode"] = "manual"
    assert entity.hvac_mode == HVACMode.HEAT
    settings["mode"] = "unexpected"
    assert entity.hvac_mode == HVACMode.HEAT
    assert entity.preset_mode == "none"






def _make_accumulator_for_validation() -> climate_module.AccumulatorClimateEntity:
    hass = HomeAssistant()
    entry_id = "entry-acm-validate"
    dev_id = "dev-acm-validate"
    addr = "9"
    record = {
        "nodes": {},
        "nodes_by_type": {"acm": {"settings": {addr: {"mode": "auto"}}}},
        "htr": {"settings": {}},
    }
    coordinator = _make_coordinator(hass, dev_id, record)
    entity = climate_module.AccumulatorClimateEntity(
        coordinator,
        entry_id,
        dev_id,
        addr,
        "Accumulator",
        node_type="acm",
    )
    entity.hass = hass
    return entity


def _patch_boost_minutes(
    monkeypatch: pytest.MonkeyPatch, return_value: int | None
) -> list[Any]:
    """Patch boost coercion helper and collect input arguments."""

    calls: list[Any] = []

    def _fake(value: Any) -> int | None:
        calls.append(value)
        return return_value

    monkeypatch.setattr(climate_module, "coerce_boost_minutes", _fake)
    return calls


def test_accumulator_validate_boost_minutes_accepts_valid_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Boost validation should return the coerced duration for valid input."""

    _reset_environment()
    entity = _make_accumulator_for_validation()
    calls = _patch_boost_minutes(monkeypatch, 60)

    result = entity._validate_boost_minutes(60)

    assert result == 60
    assert calls == [60]


def test_accumulator_validate_boost_minutes_handles_coercion_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Boost validation should log and return None when coercion fails."""

    _reset_environment()
    entity = _make_accumulator_for_validation()
    calls = _patch_boost_minutes(monkeypatch, None)
    caplog.set_level(logging.ERROR)

    result = entity._validate_boost_minutes("bad")

    assert result is None
    assert calls == ["bad"]
    assert any("Invalid boost minutes" in record.message for record in caplog.records)


def test_accumulator_validate_boost_minutes_rejects_out_of_bounds(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Boost validation should reject durations outside the supported range."""

    _reset_environment()
    entity = _make_accumulator_for_validation()
    calls = _patch_boost_minutes(monkeypatch, 130)
    caplog.set_level(logging.ERROR)

    result = entity._validate_boost_minutes(130)

    assert result is None
    assert calls == [130]
    assert any(
        "Boost duration must be between 60 and 600" in record.message
        for record in caplog.records
    )


def test_accumulator_extra_state_attributes_handles_resolver_fallbacks() -> None:
    """Edge cases in boost metadata should handle resolver failures gracefully."""

    _reset_environment()
    hass = HomeAssistant()
    entry_id = "entry-acm-resolver"
    dev_id = "dev-acm-resolver"
    addr = "7"

    class RaiseOnStr:
        def __str__(self) -> str:
            raise RuntimeError("boom")

    settings = {
        "mode": "Boost",
        "units": "C",
        "prog": [0] * 168,
        "boost_active": None,
        "boost": RaiseOnStr(),
        "boost_end_day": 12,
        "boost_end_min": 90,
        "boost_end": {"day": 14, "minute": 150},
        "boost_remaining": True,
    }

    coordinator = _make_coordinator(
        hass,
        dev_id,
        {
            "nodes": {},
            "nodes_by_type": {"acm": {"settings": {addr: settings}}},
            "htr": {"settings": {}},
        },
    )

    class FlakyResolver:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(
            self, day: Any, minute: Any
        ) -> tuple[dt.datetime | None, int | None]:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("resolver failure")
            return (
                dt.datetime(2024, 1, 1, 3, 0, tzinfo=dt.timezone.utc),
                None,
            )

    coordinator.resolve_boost_end = FlakyResolver()  # type: ignore[assignment]

    hass.data = {
        DOMAIN: {
            entry_id: {
                "coordinator": coordinator,
                "dev_id": dev_id,
                "client": AsyncMock(),
                "brand": BRAND_TERMOWEB,
            }
        }
    }

    entity = climate_module.AccumulatorClimateEntity(
        coordinator,
        entry_id,
        dev_id,
        addr,
        "Accumulator",
        node_type="acm",
    )
    entity.hass = hass

    original_now = dt_util.NOW
    try:
        dt_util.NOW = dt.datetime(2024, 1, 1, 0, 0, tzinfo=dt.timezone.utc)
        attrs = entity.extra_state_attributes
    finally:
        dt_util.NOW = original_now

    assert attrs["boost_active"] is True
    assert attrs["boost_minutes_remaining"] is None
    assert attrs["boost_end"] == "2024-01-01T03:00:00+00:00"
    assert attrs["boost_end_label"] is None


def test_accumulator_extra_state_attributes_varied_inputs() -> None:
    """Accumulator boost metadata should normalise a range of inputs."""

    _reset_environment()
    hass = HomeAssistant()
    entry_id = "entry-acm-variants"
    dev_id = "dev-acm-variants"
    addr = "8"
    settings = {
        "mode": "auto",
        "units": "C",
        "prog": [0] * 168,
        "boost_active": 1,
        "boost_end_day": 2,
        "boost_end_min": 60,
        "boost_end": None,
        "boost_remaining": None,
    }

    coordinator = _make_coordinator(
        hass,
        dev_id,
        {
            "nodes": {},
            "nodes_by_type": {"acm": {"settings": {addr: settings}}},
            "htr": {"settings": {}},
        },
    )

    hass.data = {
        DOMAIN: {
            entry_id: {
                "coordinator": coordinator,
                "dev_id": dev_id,
                "client": AsyncMock(),
                "brand": BRAND_TERMOWEB,
            }
        }
    }

    entity = climate_module.AccumulatorClimateEntity(
        coordinator,
        entry_id,
        dev_id,
        addr,
        "Accumulator",
        node_type="acm",
    )
    entity.hass = hass

    class BrokenDateTime(dt.datetime):
        def isoformat(self, *args: Any, **kwargs: Any) -> str:
            raise RuntimeError("bad isoformat")

    def _resolver(day: Any, minute: Any) -> tuple[dt.datetime, int | None]:
        return (
            BrokenDateTime(2024, 1, 1, 1, 0, tzinfo=dt.timezone.utc),
            None,
        )

    coordinator.resolve_boost_end = _resolver  # type: ignore[assignment]

    original_now = dt_util.NOW
    try:
        dt_util.NOW = dt.datetime(2024, 1, 1, 0, 0, tzinfo=dt.timezone.utc)
        attrs = entity.extra_state_attributes
    finally:
        dt_util.NOW = original_now

    assert attrs["boost_active"] is True
    assert attrs["boost_minutes_remaining"] == 60
    assert attrs["boost_end"] == "2024-01-01T01:00:00+00:00"
    assert attrs["boost_end_label"] is None

    class RaisingResolver:
        def __call__(self, day: Any, minute: Any) -> tuple[dt.datetime | None, int | None]:
            raise ValueError("resolver error")

    settings["boost_active"] = " OFF "
    settings["boost_end_day"] = None
    settings["boost_end_min"] = None
    settings["boost_end"] = {"day": 10, "minute": 15}
    settings["boost_remaining"] = ""
    coordinator.resolve_boost_end = RaisingResolver()  # type: ignore[assignment]

    attrs = entity.extra_state_attributes
    assert attrs["boost_active"] is False
    assert attrs["boost_end_label"] == "Never"
    assert attrs["boost_minutes_remaining"] is None
    assert attrs["boost_end"] is None
    assert attrs["boost_end_label"] == "Never"

    settings["boost_active"] = " maybe "
    settings["boost"] = None
    settings["mode"] = "auto"
    settings["boost_end"] = None
    settings["boost_remaining"] = None
    coordinator.resolve_boost_end = None  # type: ignore[assignment]

    attrs = entity.extra_state_attributes
    assert attrs["boost_active"] is False

    settings["boost_active"] = 0
    settings["boost_end"] = None
    settings["boost_remaining"] = 7.5
    coordinator.resolve_boost_end = None  # type: ignore[assignment]

    attrs = entity.extra_state_attributes
    assert attrs["boost_active"] is False
    assert attrs["boost_minutes_remaining"] == 7
    assert attrs["boost_end"] == "2024-01-01T00:07:00+00:00"
    assert attrs["boost_end_label"] is None

def test_accumulator_submit_settings_brand_switch() -> None:
    """Verify accumulator writes use Ducaheat client when the brand matches."""

    async def _run() -> None:
        _reset_environment()
        hass = HomeAssistant()
        entry_id = "entry-acm-submit"
        dev_id = "dev-acm-submit"
        addr = "11"
        coordinator = _make_coordinator(
            hass,
            dev_id,
            {
                "nodes": {},
                "nodes_by_type": {"acm": {"settings": {addr: {"mode": "auto"}}}},
                "htr": {"settings": {}},
            },
        )
        hass.data = {
            DOMAIN: {
                entry_id: {
                    "coordinator": coordinator,
                    "dev_id": dev_id,
                    "client": AsyncMock(),
                    "brand": BRAND_DUCAHEAT,
                }
            }
        }

        entity = climate_module.AccumulatorClimateEntity(
            coordinator,
            entry_id,
            dev_id,
            addr,
            "Accumulator",
            node_type="acm",
        )
        entity.hass = hass

        ducaheat_client = AsyncMock(spec=DucaheatRESTClient)
        await entity._async_submit_settings(
            ducaheat_client,
            mode="auto",
            stemp=21.0,
            prog=None,
            ptemp=None,
            units="C",
        )
        call = ducaheat_client.set_node_settings.await_args
        assert call.args == (dev_id, ("acm", addr))
        assert call.kwargs["mode"] == "auto"

        hass.data[DOMAIN][entry_id]["brand"] = BRAND_TERMOWEB

        generic_client = AsyncMock()
        generic_client.set_node_settings = AsyncMock()
        await entity._async_submit_settings(
            generic_client,
            mode="manual",
            stemp=19.0,
            prog=[0] * 168,
            ptemp=[18.0, 19.0, 20.0],
            units="C",
        )
        call = generic_client.set_node_settings.await_args
        assert call.args == (dev_id, ("acm", addr))
        assert call.kwargs["ptemp"] == [18.0, 19.0, 20.0]

    asyncio.run(_run())


def test_accumulator_submit_settings_handles_boost_state_error() -> None:
    async def _run() -> None:
        _reset_environment()
        hass = HomeAssistant()
        entry_id = "entry-acm-cancel"
        dev_id = "dev-acm-cancel"
        addr = "6"
        settings = {"boost_active": True, "mode": "auto", "units": "C", "prog": [0] * 168}

        coordinator = _make_coordinator(
            hass,
            dev_id,
            {
                "nodes": {},
                "nodes_by_type": {"acm": {"settings": {addr: settings}}},
                "htr": {"settings": {}},
            },
        )

        entity = climate_module.AccumulatorClimateEntity(
            coordinator,
            entry_id,
            dev_id,
            addr,
            "Accumulator",
            node_type="acm",
        )
        entity.hass = hass

        hass.data = {
            DOMAIN: {
                entry_id: {
                    "coordinator": coordinator,
                    "dev_id": dev_id,
                    "client": AsyncMock(),
                    "brand": BRAND_DUCAHEAT,
                }
            }
        }

        entity.heater_settings = MagicMock(return_value={"boost_active": True})

        def _boom() -> Any:
            raise RuntimeError("boom")

        entity.boost_state = MagicMock(side_effect=_boom)  # type: ignore[assignment]

        client = types.SimpleNamespace(set_node_settings=AsyncMock())
        await entity._async_submit_settings(
            client,
            mode="auto",
            stemp=None,
            prog=None,
            ptemp=None,
            units="C",
        )

        call = client.set_node_settings.await_args
        assert call.kwargs["cancel_boost"] is True

    asyncio.run(_run())


def test_accumulator_submit_settings_legacy_mode_detection() -> None:
    async def _run() -> None:
        _reset_environment()
        hass = HomeAssistant()
        entry_id = "entry-acm-legacy"
        dev_id = "dev-acm-legacy"
        addr = "7"
        settings = {"mode": "auto", "units": "C", "prog": [0] * 168}

        coordinator = _make_coordinator(
            hass,
            dev_id,
            {
                "nodes": {},
                "nodes_by_type": {"acm": {"settings": {addr: settings}}},
                "htr": {"settings": {}},
            },
        )

        entity = climate_module.AccumulatorClimateEntity(
            coordinator,
            entry_id,
            dev_id,
            addr,
            "Accumulator",
            node_type="acm",
        )
        entity.hass = hass

        hass.data = {
            DOMAIN: {
                entry_id: {
                    "coordinator": coordinator,
                    "dev_id": dev_id,
                    "client": AsyncMock(),
                    "brand": BRAND_DUCAHEAT,
                }
            }
        }

        entity.boost_state = MagicMock(return_value=types.SimpleNamespace(active=None))  # type: ignore[assignment]
        entity.heater_settings = MagicMock(
            return_value={"boost_active": "maybe", "boost": "no", "mode": " Boost "}
        )

        client = types.SimpleNamespace(set_node_settings=AsyncMock())
        await entity._async_submit_settings(
            client,
            mode="auto",
            stemp=None,
            prog=None,
            ptemp=None,
            units="C",
        )

        call = client.set_node_settings.await_args
        assert call.kwargs["cancel_boost"] is True

    asyncio.run(_run())


def test_accumulator_submit_settings_legacy_boost_flag() -> None:
    async def _run() -> None:
        _reset_environment()
        hass = HomeAssistant()
        entry_id = "entry-acm-legacy-flag"
        dev_id = "dev-acm-legacy-flag"
        addr = "8"
        settings = {"mode": "auto", "units": "C", "prog": [0] * 168}

        coordinator = _make_coordinator(
            hass,
            dev_id,
            {
                "nodes": {},
                "nodes_by_type": {"acm": {"settings": {addr: settings}}},
                "htr": {"settings": {}},
            },
        )

        entity = climate_module.AccumulatorClimateEntity(
            coordinator,
            entry_id,
            dev_id,
            addr,
            "Accumulator",
            node_type="acm",
        )
        entity.hass = hass

        hass.data = {
            DOMAIN: {
                entry_id: {
                    "coordinator": coordinator,
                    "dev_id": dev_id,
                    "client": AsyncMock(),
                    "brand": BRAND_DUCAHEAT,
                }
            }
        }

        entity.boost_state = MagicMock(return_value=types.SimpleNamespace(active=None))  # type: ignore[assignment]
        entity.heater_settings = MagicMock(
            return_value={"boost_active": None, "boost": True, "mode": "auto"}
        )

        client = types.SimpleNamespace(set_node_settings=AsyncMock())
        await entity._async_submit_settings(
            client,
            mode="auto",
            stemp=None,
            prog=None,
            ptemp=None,
            units="C",
        )

        call = client.set_node_settings.await_args
        assert call.kwargs["cancel_boost"] is True

    asyncio.run(_run())


def test_async_write_settings_without_client_returns_false() -> None:
    async def _run() -> None:
        _reset_environment()
        hass = HomeAssistant()
        dev_id = "dev-missing-client"
        coordinator = _make_coordinator(
            hass,
            dev_id,
            {"htr": {"settings": {}}, "nodes": {}},
        )
        hass.data = {
            DOMAIN: {
                "entry": {
                    "coordinator": coordinator,
                    "dev_id": dev_id,
                    "client": None,
                }
            }
        }

        heater = HeaterClimateEntity(coordinator, "entry", dev_id, "1", "Heater 1")
        heater.hass = hass

        success = await heater._async_write_settings(
            log_context="test", mode="auto", stemp=20.0
        )
        assert success is False

    asyncio.run(_run())


def test_commit_write_runs_optimistic_and_fallback() -> None:
    async def _run() -> None:
        _reset_environment()
        hass = HomeAssistant()
        entry_id = "entry"
        dev_id = "dev"
        addr = "1"
        record = {"htr": {"settings": {addr: {}}, "addrs": [addr]}, "nodes": {}}
        coordinator = _make_coordinator(
            hass,
            dev_id,
            record,
            client=AsyncMock(),
        )

        heater = HeaterClimateEntity(coordinator, entry_id, dev_id, addr, "Heater")

        async_write = AsyncMock(return_value=True)
        optimistic = MagicMock()
        fallback = MagicMock()
        apply_fn = MagicMock()

        heater._async_write_settings = async_write
        heater._optimistic_update = optimistic
        heater._schedule_refresh_fallback = fallback

        await heater._commit_write(
            log_context="Test write",
            write_kwargs={"prog": [0, 1, 2]},
            apply_fn=apply_fn,
            success_details={"detail": "value"},
        )

        async_write.assert_awaited_once_with(log_context="Test write", prog=[0, 1, 2])
        optimistic.assert_called_once_with(apply_fn)
        fallback.assert_called_once()

        async_write.reset_mock()
        optimistic.reset_mock()
        fallback.reset_mock()
        async_write.return_value = False

        await heater._commit_write(
            log_context="Test write",
            write_kwargs={"ptemp": [1.0, 2.0, 3.0]},
            apply_fn=apply_fn,
        )

        async_write.assert_awaited_once_with(
            log_context="Test write", ptemp=[1.0, 2.0, 3.0]
        )
        optimistic.assert_not_called()
        fallback.assert_not_called()

    asyncio.run(_run())


def test_async_setup_entry_without_inventory_skips_entities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        _reset_environment()
        hass = HomeAssistant()
        entry = types.SimpleNamespace(entry_id="entry-missing")
        dev_id = "dev-missing"
        nodes = {
            "nodes": [
                {"type": "htr", "addr": "11", "name": " First "},
                {"type": "HTR", "addr": "22"},
            ]
        }

        coordinator = _make_coordinator(
            hass,
            dev_id,
            {"nodes": nodes, "htr": {"settings": {"11": {}, "22": {}}}},
            client=AsyncMock(),
        )

        record: dict[str, Any] = {
            "coordinator": coordinator,
            "dev_id": dev_id,
            "client": AsyncMock(),
            "nodes": nodes,
        }
        hass.data = {DOMAIN: {entry.entry_id: record}}

        added: list[HeaterClimateEntity] = []

        def _async_add_entities(entities: list[HeaterClimateEntity]) -> None:
            added.extend(entities)

        calls: list[Mapping[str, Any] | None] = []

        def _missing_inventory(*args: Any, **kwargs: Any) -> Inventory:
            calls.append(kwargs.get("container"))
            raise LookupError("missing inventory")

        monkeypatch.setattr(
            inventory_module.Inventory,
            "require_from_context",
            staticmethod(_missing_inventory),
        )

        with pytest.raises(ValueError):
            await async_setup_entry(hass, entry, _async_add_entities)

        assert added == []
        record_after = hass.data[DOMAIN][entry.entry_id]
        assert "inventory" not in record_after
        assert "node_inventory" not in record_after
        assert calls and calls[0] is record

    asyncio.run(_run())


def test_async_setup_entry_reuses_coordinator_inventory(
    climate_inventory: Callable[[str, Mapping[str, Any]], Inventory],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        _reset_environment()
        hass = HomeAssistant()
        entry = types.SimpleNamespace(entry_id="entry-coord")
        dev_id = "dev-coord"
        raw_nodes = {"nodes": [{"type": "htr", "addr": "5"}]}
        inventory = climate_inventory(dev_id, raw_nodes)

        coordinator = _make_coordinator(
            hass,
            dev_id,
            {"nodes": raw_nodes, "htr": {"settings": {"5": {}}}},
            client=AsyncMock(),
            inventory=inventory,
        )

        record: dict[str, Any] = {
            "coordinator": coordinator,
            "dev_id": dev_id,
            "client": AsyncMock(),
            "nodes": raw_nodes,
            "inventory": inventory,
        }
        hass.data = {DOMAIN: {entry.entry_id: record}}

        added: list[HeaterClimateEntity] = []

        def _async_add_entities(entities: list[HeaterClimateEntity]) -> None:
            added.extend(entities)

        calls: list[Mapping[str, Any] | None] = []

        original_resolver = Inventory.require_from_context

        def _reuse_inventory(*args: Any, **kwargs: Any) -> Inventory:
            calls.append(kwargs.get("container"))
            return original_resolver(*args, **kwargs)

        monkeypatch.setattr(
            inventory_module.Inventory,
            "require_from_context",
            staticmethod(_reuse_inventory),
        )
        await async_setup_entry(hass, entry, _async_add_entities)

        assert len(added) == 1
        assert hass.data[DOMAIN][entry.entry_id]["inventory"] is inventory
        assert calls and calls[0] is record

    asyncio.run(_run())


def test_refresh_fallback_skips_when_hass_inactive(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _run() -> None:
        _reset_environment()

        hass = HomeAssistant()
        hass.is_stopping = True
        hass.is_running = True
        entry_id = "entry"
        dev_id = "dev"
        addr = "A"
        coordinator = _make_coordinator(
            hass,
            dev_id,
            {"nodes": {"nodes": []}, "htr": {"settings": {addr: {}}}},
            client=AsyncMock(),
        )

        hass.data = {
            DOMAIN: {
                entry_id: {
                    "client": AsyncMock(),
                    "coordinator": coordinator,
                    "dev_id": dev_id,
                    "version": "1",
                    "ws_state": {},
                }
            }
        }

        heater = HeaterClimateEntity(coordinator, entry_id, dev_id, addr, "Heater")
        await heater.async_added_to_hass()

        async def fast_sleep(_delay: float) -> None:
            return None

        monkeypatch.setattr(climate_module.asyncio, "sleep", fast_sleep)

        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            heater._schedule_refresh_fallback()
            task = heater._refresh_fallback
            assert task is not None
            await task
        coordinator.async_refresh_heater.assert_not_awaited()
        assert heater._refresh_fallback is None
        assert "hass stopping" in caplog.text

        hass.is_stopping = False
        hass.is_running = False
        coordinator.async_refresh_heater.reset_mock()

        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            heater._schedule_refresh_fallback()
            task = heater._refresh_fallback
            assert task is not None
            await task
        coordinator.async_refresh_heater.assert_not_awaited()
        assert heater._refresh_fallback is None
        assert "hass not running" in caplog.text

    asyncio.run(_run())


def test_heater_additional_cancelled_edges(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _run() -> None:
        _reset_environment()
        hass = HomeAssistant()
        entry_id = "entry"
        dev_id = "dev"
        addr = "A"
        base_prog: list[int] = [0, 1, 2] * 56
        settings = {
            "mode": "manual",
            "state": "heating",
            "mtemp": "19.0",
            "stemp": "21.0",
            "ptemp": ["18.0", "19.0", "20.0"],
            "prog": list(base_prog),
            "units": "C",
        }
        coordinator = _make_coordinator(
            hass,
            dev_id,
            {"nodes": {"nodes": []}, "htr": {"settings": {addr: settings}}},
            client=AsyncMock(),
        )
        client = AsyncMock()
        client.set_node_settings = AsyncMock()

        hass.data = {
            DOMAIN: {
                entry_id: {
                    "client": client,
                    "coordinator": coordinator,
                    "dev_id": dev_id,
                    "ws_state": {},
                    "version": "1",
                }
            }
        }

        heater = HeaterClimateEntity(coordinator, entry_id, dev_id, addr, "Heater")
        await heater.async_added_to_hass()

        class SentinelCancelled(Exception):
            pass

        monkeypatch.setattr(climate_module.asyncio, "CancelledError", SentinelCancelled)

        async def fast_sleep(_delay: float) -> None:
            return None

        monkeypatch.setattr(climate_module.asyncio, "sleep", fast_sleep)

        orig_float = climate_module.float_or_none

        def raising_float(_value: Any) -> float | None:
            raise SentinelCancelled()

        climate_module.float_or_none = raising_float
        with pytest.raises(SentinelCancelled):
            _ = heater.extra_state_attributes
        climate_module.float_or_none = orig_float

        prog = list(base_prog)
        orig_write = heater.async_write_ha_state

        def raising_write() -> None:
            raise SentinelCancelled()

        heater.async_write_ha_state = raising_write
        with pytest.raises(SentinelCancelled):
            await heater.async_set_schedule(prog)
        heater.async_write_ha_state = orig_write

        caplog.clear()
        with caplog.at_level(logging.ERROR):
            await heater.async_set_preset_temperatures()
        assert "Preset temperatures require" in caplog.text

        heater.async_write_ha_state = raising_write
        with pytest.raises(SentinelCancelled):
            await heater.async_set_preset_temperatures(ptemp=[18.0, 19.0, 20.0])
        heater.async_write_ha_state = orig_write

        client.set_node_settings.side_effect = SentinelCancelled()
        heater._pending_mode = HVACMode.AUTO
        heater._pending_stemp = 20.5
        with pytest.raises(SentinelCancelled):
            await heater._write_after_debounce()
        client.set_node_settings.side_effect = None

        class BadFloat:
            def __float__(self) -> float:
                raise SentinelCancelled()

        heater._pending_mode = HVACMode.HEAT
        heater._pending_stemp = BadFloat()
        with pytest.raises(SentinelCancelled):
            await heater._write_after_debounce()

        coordinator.async_refresh_heater = AsyncMock(side_effect=SentinelCancelled())
        heater._refresh_fallback = None
        heater._schedule_refresh_fallback()
        assert heater._refresh_fallback is not None
        with pytest.raises(SentinelCancelled):
            await heater._refresh_fallback

    asyncio.run(_run())




def test_write_after_debounce_registers_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        _reset_environment()
        hass = HomeAssistant()
        entry_id = "entry"
        dev_id = "dev"
        addr = "1"
        nodes = {"nodes": [{"type": "htr", "addr": addr}]}
        record = {
            "nodes": nodes,
            "htr": {"settings": {addr: {}}, "addrs": [addr]},
            "nodes_by_type": {
                "htr": {"settings": {addr: {}}, "addrs": [addr]},
            },
        }
        coordinator = _make_coordinator(
            hass,
            dev_id,
            record,
            client=AsyncMock(),
        )
        heater = HeaterClimateEntity(coordinator, entry_id, dev_id, addr, "Heater")

        async def fast_sleep(_delay: float) -> None:
            return None

        monkeypatch.setattr(climate_module.asyncio, "sleep", fast_sleep)

        heater._pending_mode = HVACMode.HEAT
        heater._pending_stemp = 21.5
        heater._async_write_settings = AsyncMock(return_value=True)

        await heater._write_after_debounce()

        key = ("htr", addr)
        assert key in coordinator.pending_settings
        pending = coordinator.pending_settings[key]
        assert pending["mode"] == "manual"
        assert pending["stemp"] == pytest.approx(21.5)

    asyncio.run(_run())


def test_heater_write_paths_and_errors(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def _run() -> None:
        _reset_environment()
        from homeassistant.const import ATTR_TEMPERATURE

        hass = HomeAssistant()
        entry_id = "entry"
        dev_id = "dev1"
        addr = "A1"
        base_prog: list[int] = [0, 1, 2] * 56
        settings = {
            "mode": "manual",
            "state": "heating",
            "mtemp": "19.5",
            "stemp": "21.0",
            "ptemp": ["15.0", "18.0", "21.0"],
            "prog": list(base_prog),
            "units": "C",
            "max_power": 1200,
        }
        coordinator_data = {
            dev_id: {
                "nodes": {"nodes": []},
                "htr": {"settings": {addr: settings}},
                "version": "5.0.0",
            }
        }

        coordinator = _make_coordinator(
            hass,
            dev_id,
            coordinator_data[dev_id],
            client=AsyncMock(),
        )
        client = AsyncMock()
        hass.data = {
            DOMAIN: {
                entry_id: {
                    "client": client,
                    "coordinator": coordinator,
                    "dev_id": dev_id,
                    "version": "5.0.0",
                    "ws_state": {
                        dev_id: {"status": "disconnected", "last_event_at": None}
                    },
                }
            }
        }

        heater = HeaterClimateEntity(coordinator, entry_id, dev_id, addr, "Heater")
        await heater.async_added_to_hass()

        fallback_waiters: Deque[asyncio.Future[None]] = deque()
        write_waiters: Deque[asyncio.Future[None]] = deque()
        write_block = False
        real_sleep = asyncio.sleep

        async def fake_sleep(delay: float) -> None:
            if delay == climate_module._WRITE_DEBOUNCE:
                if write_block:
                    loop = asyncio.get_running_loop()
                    fut: asyncio.Future[None] = loop.create_future()
                    write_waiters.append(fut)
                    await fut
                    return None
                return None
            if delay == climate_module._WS_ECHO_FALLBACK_REFRESH:
                loop = asyncio.get_running_loop()
                fut: asyncio.Future[None] = loop.create_future()
                fallback_waiters.append(fut)
                await fut
                return None
            await real_sleep(delay)
            return None

        real_create_task = asyncio.create_task
        created_tasks: list[asyncio.Task[Any]] = []

        def track_create_task(
            coro: Coroutine[Any, Any, Any], *, name: str | None = None
        ) -> asyncio.Task[Any]:
            task = real_create_task(coro, name=name)
            created_tasks.append(task)
            return task

        async def _pop_waiter() -> asyncio.Future[None]:
            for _ in range(10):
                if fallback_waiters:
                    return fallback_waiters.popleft()
                await real_sleep(0)
            raise AssertionError("fallback waiter not created")

        async def _pop_write_waiter() -> asyncio.Future[None]:
            for _ in range(10):
                if write_waiters:
                    return write_waiters.popleft()
                await real_sleep(0)
            raise AssertionError("write waiter not created")

        async def _complete_fallback_once() -> None:
            waiter = await _pop_waiter()
            task = heater._refresh_fallback
            assert task is not None
            assert coordinator.async_refresh_heater.await_count == 0
            waiter.set_result(None)
            await task
            coordinator.async_refresh_heater.assert_awaited_once_with(("htr", addr))
            coordinator.async_refresh_heater.reset_mock()
            assert heater._refresh_fallback is None

        class RaisingMapping:
            def __init__(self, real: dict[str, Any]) -> None:
                self._real = real
                self._calls = 0

            def get(self, *args: Any, **kwargs: Any) -> Any:
                self._calls += 1
                if self._calls >= 2:
                    raise RuntimeError("boom mapping")
                return self._real.get(*args, **kwargs)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)
        monkeypatch.setattr(asyncio, "create_task", track_create_task)

        caplog.set_level(logging.DEBUG)

        # -------------------- async_set_schedule (valid) --------------------
        await heater.async_set_schedule(list(base_prog))
        call = client.set_node_settings.await_args
        assert call.args == (dev_id, ("htr", addr))
        assert call.kwargs["prog"] == list(base_prog)
        assert call.kwargs["units"] == "C"

        settings_after = coordinator.data[dev_id]["htr"]["settings"][addr]
        assert settings_after["prog"] == list(base_prog)

        assert heater._refresh_fallback is not None
        await _complete_fallback_once()

        client.set_node_settings.reset_mock()

        # -------------------- async_set_schedule (invalid length/value) -----
        caplog.clear()
        await heater.async_set_schedule([0, 1])
        assert client.set_node_settings.await_count == 0
        assert "Invalid prog length" in caplog.text
        assert not fallback_waiters

        caplog.clear()
        client.set_node_settings.reset_mock()
        bad_prog = list(base_prog)
        bad_prog[5] = 7
        await heater.async_set_schedule(bad_prog)
        assert client.set_node_settings.await_count == 0
        assert "Invalid prog for type" in caplog.text
        assert not fallback_waiters

        # -------------------- async_set_schedule (API error) ----------------
        caplog.clear()
        client.set_node_settings.reset_mock()
        client.set_node_settings.side_effect = RuntimeError("boom schedule")
        prev_prog = list(settings_after["prog"])
        prev_fallback = heater._refresh_fallback
        await heater.async_set_schedule(list(base_prog))
        assert client.set_node_settings.await_count == 1
        assert settings_after["prog"] == prev_prog
        assert heater._refresh_fallback is prev_fallback
        assert not fallback_waiters
        assert "Schedule write failed" in caplog.text
        client.set_node_settings.side_effect = None
        client.set_node_settings.reset_mock()

        # -------------------- async_set_schedule (optimistic failure) -------
        caplog.clear()
        client.set_node_settings.reset_mock()
        old_data = coordinator.data
        coordinator.data = RaisingMapping(old_data)
        await heater.async_set_schedule(list(base_prog))
        assert client.set_node_settings.await_count == 1
        assert settings_after["prog"] == prev_prog
        assert (
            "Optimistic update failed" in caplog.text
            or "Failed to resolve device record" in caplog.text
            or "missing immutable inventory cache" in caplog.text
        )
        waiter = await _pop_waiter()
        task = heater._refresh_fallback
        assert task is not None
        coordinator.data = old_data
        waiter.set_result(None)
        await task
        coordinator.async_refresh_heater.assert_awaited_once_with(("htr", addr))
        coordinator.async_refresh_heater.reset_mock()
        assert heater._refresh_fallback is None
        client.set_node_settings.reset_mock()

        # -------------------- async_set_preset_temperatures (valid forms) ---
        caplog.clear()
        preset_payload = [18.5, 19.5, 20.5]
        await heater.async_set_preset_temperatures(ptemp=preset_payload)
        call = client.set_node_settings.await_args
        assert call.args == (dev_id, ("htr", addr))
        assert call.kwargs["ptemp"] == preset_payload
        assert call.kwargs["units"] == "C"
        assert settings_after["ptemp"] == ["18.5", "19.5", "20.5"]
        await _complete_fallback_once()
        client.set_node_settings.reset_mock()

        caplog.clear()
        await heater.async_set_preset_temperatures(cold=16.5, night=17.5, day=18.5)
        call = client.set_node_settings.await_args
        assert call.kwargs["ptemp"] == [16.5, 17.5, 18.5]
        assert settings_after["ptemp"] == ["16.5", "17.5", "18.5"]
        await _complete_fallback_once()
        client.set_node_settings.reset_mock()

        # -------------------- async_set_preset_temperatures (invalid) -------
        caplog.clear()
        await heater.async_set_preset_temperatures(ptemp=[18.0, 19.0])
        assert client.set_node_settings.await_count == 0
        assert "Invalid ptemp length" in caplog.text
        assert not fallback_waiters

        caplog.clear()
        client.set_node_settings.reset_mock()
        await heater.async_set_preset_temperatures(ptemp=["bad", "bad", "bad"])
        assert client.set_node_settings.await_count == 0
        assert "Invalid ptemp values" in caplog.text
        assert not fallback_waiters

        # -------------------- async_set_preset_temperatures (API error) -----
        caplog.clear()
        client.set_node_settings.reset_mock()
        client.set_node_settings.side_effect = RuntimeError("boom preset")
        prev_ptemp = list(settings_after["ptemp"])
        prev_fallback = heater._refresh_fallback
        await heater.async_set_preset_temperatures(ptemp=[19.1, 20.1, 21.1])
        assert client.set_node_settings.await_count == 1
        assert settings_after["ptemp"] == prev_ptemp
        assert heater._refresh_fallback is prev_fallback
        assert not fallback_waiters
        assert "Preset write failed" in caplog.text
        client.set_node_settings.side_effect = None
        client.set_node_settings.reset_mock()

        # -------------------- async_set_preset_temperatures (optimistic failure) -
        caplog.clear()
        client.set_node_settings.reset_mock()
        old_data = coordinator.data
        coordinator.data = RaisingMapping(old_data)
        await heater.async_set_preset_temperatures(ptemp=[19.2, 20.2, 21.2])
        assert client.set_node_settings.await_count == 1
        assert settings_after["ptemp"] == prev_ptemp
        assert (
            "Optimistic update failed" in caplog.text
            or "Failed to resolve device record" in caplog.text
            or "missing immutable inventory cache" in caplog.text
        )
        waiter = await _pop_waiter()
        task = heater._refresh_fallback
        assert task is not None
        coordinator.data = old_data
        waiter.set_result(None)
        await task
        coordinator.async_refresh_heater.assert_awaited_once_with(("htr", addr))
        coordinator.async_refresh_heater.reset_mock()
        assert heater._refresh_fallback is None
        client.set_node_settings.reset_mock()

        # -------------------- async_set_temperature (valid + clamps) -------
        client.set_node_settings.reset_mock()
        await heater.async_set_temperature(**{ATTR_TEMPERATURE: 35.6})
        assert heater._write_task is not None
        await heater._write_task
        call = client.set_node_settings.await_args
        assert call.args == (dev_id, ("htr", addr))
        assert call.kwargs["mode"] == "manual"
        assert call.kwargs["stemp"] == pytest.approx(30.0)
        assert call.kwargs["units"] == "C"
        assert settings_after["stemp"] == "30.0"
        await _complete_fallback_once()
        client.set_node_settings.reset_mock()

        await heater.async_set_temperature(**{ATTR_TEMPERATURE: 2.0})
        assert heater._write_task is not None
        await heater._write_task
        call = client.set_node_settings.await_args
        assert call.kwargs["stemp"] == pytest.approx(5.0)
        assert settings_after["stemp"] == "5.0"
        await _complete_fallback_once()
        client.set_node_settings.reset_mock()

        caplog.clear()
        await heater.async_set_temperature(**{ATTR_TEMPERATURE: "bad"})
        assert client.set_node_settings.await_count == 0
        assert (heater._write_task is None) or heater._write_task.done()
        assert "Invalid temperature payload" in caplog.text
        assert not fallback_waiters

        # -------------------- async_set_hvac_mode --------------------------
        client.set_node_settings.reset_mock()
        await heater.async_set_hvac_mode(HVACMode.AUTO)
        assert heater._write_task is not None
        await heater._write_task
        call = client.set_node_settings.await_args
        assert call.kwargs["mode"] == "auto"
        assert call.kwargs["stemp"] is None
        assert settings_after["mode"] == "auto"
        assert settings_after["stemp"] == "5.0"
        await _complete_fallback_once()
        client.set_node_settings.reset_mock()

        await heater.async_set_hvac_mode(HVACMode.OFF)
        assert heater._write_task is not None
        await heater._write_task
        call = client.set_node_settings.await_args
        assert call.kwargs["mode"] == "off"
        assert settings_after["mode"] == "off"
        assert settings_after["stemp"] == "5.0"
        await _complete_fallback_once()
        client.set_node_settings.reset_mock()

        await heater.async_set_hvac_mode(HVACMode.HEAT)
        assert heater._write_task is not None
        await heater._write_task
        call = client.set_node_settings.await_args
        assert call.kwargs["mode"] == "manual"
        assert call.kwargs["stemp"] == pytest.approx(5.0)
        assert settings_after["mode"] == "manual"
        assert settings_after["stemp"] == "5.0"
        await _complete_fallback_once()
        client.set_node_settings.reset_mock()

        caplog.clear()
        await heater.async_set_hvac_mode(cast(HVACMode, "eco"))
        assert client.set_node_settings.await_count == 0
        assert "Unsupported hvac_mode" in caplog.text
        assert not fallback_waiters

        # -------------------- _ensure_write_task and debounce -------------
        client.set_node_settings.reset_mock()
        write_block = True
        pre_fallback = heater._refresh_fallback
        heater._pending_mode = None
        heater._pending_stemp = None
        await heater._ensure_write_task()
        first_task = heater._write_task
        assert first_task is not None
        await heater._ensure_write_task()
        assert heater._write_task is first_task
        write_waiter = await _pop_write_waiter()
        write_block = False
        write_waiter.set_result(None)
        await first_task
        assert client.set_node_settings.await_count == 0
        assert heater._refresh_fallback is pre_fallback
        assert not write_waiters

        # -------------------- _write_after_debounce error path -------------
        caplog.clear()
        client.set_node_settings.reset_mock()
        client.set_node_settings.side_effect = RuntimeError("write boom")
        heater._pending_mode = HVACMode.AUTO
        heater._pending_stemp = None
        heater._refresh_fallback = None
        await heater._ensure_write_task()
        assert heater._write_task is not None
        await heater._write_task
        assert "Mode/setpoint write failed" in caplog.text
        assert heater._refresh_fallback is None
        assert not fallback_waiters
        client.set_node_settings.side_effect = None
        client.set_node_settings.reset_mock()

        # -------------------- _schedule_refresh_fallback behaviour --------
        heater._schedule_refresh_fallback()
        task_a = heater._refresh_fallback
        waiter_a = await _pop_waiter()
        heater._schedule_refresh_fallback()
        task_b = heater._refresh_fallback
        waiter_b = await _pop_waiter()

        assert task_a is not None and task_b is not None and task_a is not task_b
        with pytest.raises(asyncio.CancelledError):
            await task_a
        if not waiter_a.done():
            waiter_a.cancel()

        assert coordinator.async_refresh_heater.await_count == 0

        waiter_b.set_result(None)
        await task_b
        coordinator.async_refresh_heater.assert_awaited_once_with(("htr", addr))
        coordinator.async_refresh_heater.reset_mock()

        caplog.clear()
        coordinator.async_refresh_heater.side_effect = RuntimeError("refresh boom")
        heater._schedule_refresh_fallback()
        waiter_err = await _pop_waiter()
        task_err = heater._refresh_fallback
        assert task_err is not None
        waiter_err.set_result(None)
        await task_err
        assert "Refresh fallback failed" in caplog.text
        coordinator.async_refresh_heater.side_effect = None
        coordinator.async_refresh_heater.reset_mock()
        assert not fallback_waiters

        # -------------------- WS healthy suppresses fallback --------------
        hass.data[DOMAIN][entry_id]["ws_state"][dev_id] = {
            "status": "healthy",
            "last_event_at": 0,
            "last_payload_at": climate_module.time.time(),
            "idle_restart_pending": False,
        }
        client.set_node_settings.reset_mock()
        await heater.async_set_temperature(**{ATTR_TEMPERATURE: 22.5})
        assert heater._write_task is not None
        await heater._write_task
        assert client.set_node_settings.await_count == 1
        assert heater._refresh_fallback is None
        assert not fallback_waiters
        client.set_node_settings.reset_mock()

        # -------------------- WS healthy but stale payload triggers fallback ----
        hass.data[DOMAIN][entry_id]["ws_state"][dev_id] = {
            "status": "healthy",
            "last_event_at": 0,
            "last_payload_at": climate_module.time.time() - 60,
            "idle_restart_pending": False,
        }
        await heater.async_set_temperature(**{ATTR_TEMPERATURE: 21.5})
        assert heater._write_task is not None
        await heater._write_task
        assert client.set_node_settings.await_count == 1
        assert heater._refresh_fallback is not None
        await _complete_fallback_once()
        client.set_node_settings.reset_mock()

        # -------------------- WS status missing triggers fallback ---------
        hass.data[DOMAIN][entry_id]["ws_state"].pop(dev_id, None)
        await heater.async_set_temperature(**{ATTR_TEMPERATURE: 24.5})
        assert heater._write_task is not None
        await heater._write_task
        assert heater._refresh_fallback is not None
        await _complete_fallback_once()
        client.set_node_settings.reset_mock()

        # -------------------- WS down restores fallback -------------------
        hass.data[DOMAIN][entry_id]["ws_state"][dev_id] = {
            "status": "disconnected",
            "last_event_at": None,
        }
        await heater.async_set_temperature(**{ATTR_TEMPERATURE: 23.5})
        assert heater._write_task is not None
        await heater._write_task
        assert heater._refresh_fallback is not None
        await _complete_fallback_once()
        client.set_node_settings.reset_mock()

        assert created_tasks, "Expected background tasks to be created"

    asyncio.run(_run())


def test_heater_cancellation_and_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        _reset_environment()
        hass = HomeAssistant()
        entry_id = "entry"
        dev_id = "dev"
        addr = "A"
        base_prog: list[int] = [0, 1, 2] * 56
        settings = {
            "mode": "manual",
            "state": "heating",
            "mtemp": "19.0",
            "stemp": "21.0",
            "ptemp": ["18.0", "19.0", "20.0"],
            "prog": list(base_prog),
            "units": "C",
        }
        coordinator = _make_coordinator(
            hass,
            dev_id,
            {"nodes": {"nodes": []}, "htr": {"settings": {addr: settings}}},
            client=AsyncMock(),
        )
        client = AsyncMock()
        client.set_node_settings = AsyncMock()

        hass.data = {
            DOMAIN: {
                entry_id: {
                    "client": client,
                    "coordinator": coordinator,
                    "dev_id": dev_id,
                    "version": "1",
                    "ws_state": {},
                }
            }
        }

        heater = HeaterClimateEntity(coordinator, entry_id, dev_id, addr, "Heater")
        await heater.async_added_to_hass()
        orig_cancelled = climate_module.asyncio.CancelledError

        class CancelList(list):
            def __getitem__(self, idx):
                raise ValueError("cancel slot")

        monkeypatch.setattr(climate_module.asyncio, "CancelledError", ValueError)
        settings["prog"] = CancelList(list(base_prog))
        with pytest.raises(ValueError):
            heater._current_prog_slot(settings)
        settings["prog"] = list(base_prog)

        class BadPTList(list):
            def __getitem__(self, idx):
                raise RuntimeError("bad ptemp")

        settings["ptemp"] = BadPTList(["18", "19", "20"])
        attrs = heater.extra_state_attributes
        assert "program_setpoint" not in attrs
        settings["ptemp"] = ["18.0", "19.0", "20.0"]

        class CancelInt(int):
            def __int__(self) -> int:
                raise ValueError("cancel prog")

        prog_cancel = list(base_prog)
        prog_cancel[0] = CancelInt(0)
        with pytest.raises(ValueError):
            await heater.async_set_schedule(prog_cancel)

        client.set_node_settings.reset_mock()
        client.set_node_settings.side_effect = ValueError("api cancel")
        with pytest.raises(ValueError):
            await heater.async_set_schedule(list(base_prog))
        client.set_node_settings.side_effect = None

        class CancelMapping(dict):
            def __init__(self, real: dict[str, Any]) -> None:
                super().__init__(real)

            def get(self, *_args: Any, **_kwargs: Any) -> Any:
                raise ValueError("optimistic cancel")

        original_data = coordinator.data
        coordinator.data = CancelMapping(original_data)
        await heater.async_set_schedule(list(base_prog))
        coordinator.data = original_data

        monkeypatch.setattr(climate_module.asyncio, "CancelledError", KeyError)
        with pytest.raises(KeyError):
            await heater.async_set_preset_temperatures()

        monkeypatch.setattr(climate_module.asyncio, "CancelledError", ValueError)

        class CancelFloat:
            def __float__(self) -> float:
                raise ValueError("cancel float")

        with pytest.raises(ValueError):
            await heater.async_set_preset_temperatures(
                ptemp=[CancelFloat(), 19.0, 20.0]
            )

        client.set_node_settings.reset_mock()
        client.set_node_settings.side_effect = ValueError("preset cancel")
        with pytest.raises(ValueError):
            await heater.async_set_preset_temperatures(ptemp=[18.0, 19.0, 20.0])
        client.set_node_settings.side_effect = None

        coordinator.data = CancelMapping(original_data)
        await heater.async_set_preset_temperatures(ptemp=[18.0, 19.0, 20.0])
        coordinator.data = original_data

        monkeypatch.setattr(climate_module.asyncio, "CancelledError", orig_cancelled)

        async def fast_sleep(_delay: float) -> None:
            return None

        monkeypatch.setattr(climate_module.asyncio, "sleep", fast_sleep)

        client.set_node_settings.reset_mock()
        heater._pending_mode = None
        heater._pending_stemp = 22.0
        await heater._write_after_debounce()
        call = client.set_node_settings.await_args
        assert call.kwargs["mode"] == "manual"
        assert call.kwargs["stemp"] == 22.0

        client.set_node_settings.reset_mock()
        heater._pending_mode = HVACMode.HEAT
        heater._pending_stemp = None
        await heater._write_after_debounce()
        call = client.set_node_settings.await_args
        assert call.kwargs["mode"] == "manual"
        assert call.kwargs["stemp"] == 22.0

        client.set_node_settings.reset_mock()
        client.set_node_settings.side_effect = ValueError("write cancel")
        heater._pending_mode = HVACMode.AUTO
        heater._pending_stemp = 19.5
        await heater._write_after_debounce()
        call = client.set_node_settings.await_args
        assert call.kwargs["mode"] == "manual"
        assert call.kwargs["stemp"] == 19.5
        client.set_node_settings.side_effect = None

        class BadFloat:
            def __float__(self) -> float:
                raise RuntimeError("bad float")

        heater._pending_mode = HVACMode.HEAT
        heater._pending_stemp = BadFloat()
        await heater._write_after_debounce()

        writer = MagicMock(side_effect=ValueError("optimistic fail"))
        monkeypatch.setattr(heater, "async_write_ha_state", writer)
        heater._pending_mode = HVACMode.AUTO
        heater._pending_stemp = 20.0
        await heater._write_after_debounce()
        assert writer.call_count == 1

        async def failing_refresh() -> None:
            raise ValueError("fallback cancel")

        coordinator.async_refresh_heater = AsyncMock(side_effect=failing_refresh)
        heater._schedule_refresh_fallback()
        task = heater._refresh_fallback
        assert task is not None
        await task
        coordinator.async_refresh_heater.assert_awaited_once_with(("htr", addr))
        assert heater._refresh_fallback is None

    asyncio.run(_run())




def test_heater_cancelled_paths_propagate(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _run() -> None:
        _reset_environment()
        from homeassistant.components.climate import HVACMode

        hass = HomeAssistant()
        entry_id = "entry"
        dev_id = "dev"
        addr = "A"
        base_prog: list[int] = [0, 1, 2] * 56
        settings = {
            "mode": "manual",
            "state": "heating",
            "mtemp": "19.0",
            "stemp": "21.0",
            "ptemp": ["18.0", "19.0", "20.0"],
            "prog": list(base_prog),
            "units": "C",
        }
        coordinator = _make_coordinator(
            hass,
            dev_id,
            {"nodes": {"nodes": []}, "htr": {"settings": {addr: settings}}},
            client=AsyncMock(),
        )
        client = AsyncMock()
        client.set_node_settings = AsyncMock()

        hass.data = {
            DOMAIN: {
                entry_id: {
                    "client": client,
                    "coordinator": coordinator,
                    "dev_id": dev_id,
                    "ws_state": {},
                    "version": "1",
                }
            }
        }

        heater = HeaterClimateEntity(coordinator, entry_id, dev_id, addr, "Heater")
        await heater.async_added_to_hass()

        orig_float = climate_module.float_or_none

        def raising_float(_value: Any) -> float | None:
            raise asyncio.CancelledError()

        monkeypatch.setattr(climate_module, "float_or_none", raising_float)
        with pytest.raises(asyncio.CancelledError):
            _ = heater.extra_state_attributes
        monkeypatch.setattr(climate_module, "float_or_none", orig_float)

        class CancelInt(int):
            def __int__(self) -> int:
                raise asyncio.CancelledError()

        prog_cancel = list(base_prog)
        prog_cancel[0] = CancelInt(0)
        with pytest.raises(asyncio.CancelledError):
            await heater.async_set_schedule(prog_cancel)

        caplog.clear()
        with caplog.at_level(logging.ERROR):
            await heater.async_set_preset_temperatures()
        assert "Preset temperatures require" in caplog.text

        class CancelSettings(dict):
            def get(self, *_args: Any, **_kwargs: Any) -> Any:
                raise asyncio.CancelledError()

        original_settings = coordinator.data[dev_id]["htr"]["settings"]
        coordinator.data[dev_id]["htr"]["settings"] = CancelSettings(original_settings)
        with pytest.raises(asyncio.CancelledError):
            await heater.async_set_preset_temperatures(ptemp=[18.0, 19.0, 20.0])
        coordinator.data[dev_id]["htr"]["settings"] = original_settings

        client.set_node_settings.side_effect = asyncio.CancelledError()
        heater._pending_mode = HVACMode.AUTO
        heater._pending_stemp = 21.0
        with pytest.raises(asyncio.CancelledError):
            await heater._write_after_debounce()

        client.set_node_settings.side_effect = None

        class CancelFloat:
            def __float__(self) -> float:
                raise asyncio.CancelledError()

        heater._pending_mode = HVACMode.HEAT
        heater._pending_stemp = CancelFloat()
        with pytest.raises(asyncio.CancelledError):
            await heater._write_after_debounce()

        class CancelMapping(dict):
            def get(self, *_args: Any, **_kwargs: Any) -> Any:
                raise asyncio.CancelledError()

        original_data = coordinator.data
        coordinator.data = CancelMapping(original_data)
        heater._pending_mode = HVACMode.AUTO
        heater._pending_stemp = 20.0
        with pytest.raises(asyncio.CancelledError):
            await heater._write_after_debounce()
        coordinator.data = original_data

        async def fast_sleep(_delay: float) -> None:
            return None

        monkeypatch.setattr(climate_module.asyncio, "sleep", fast_sleep)
        coordinator.async_refresh_heater = AsyncMock(
            side_effect=asyncio.CancelledError()
        )
        heater._schedule_refresh_fallback()
        task = heater._refresh_fallback
        assert task is not None
        with pytest.raises(asyncio.CancelledError):
            await task
        assert heater._refresh_fallback is None

    asyncio.run(_run())
