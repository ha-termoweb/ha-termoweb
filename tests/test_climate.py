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

from conftest import (
    FakeCoordinator,
    _install_stubs,
    build_coordinator_device_state,
    build_entry_runtime,
)

import custom_components.termoweb.inventory as inventory_module
from custom_components.termoweb.domain import state_to_dict
from custom_components.termoweb.inventory import Inventory

_install_stubs()

from custom_components.termoweb import climate as climate_module
from custom_components.termoweb.entities import climate as entities_climate_module
from custom_components.termoweb.heater import DEFAULT_BOOST_DURATION
from custom_components.termoweb.const import (
    BRAND_DUCAHEAT,
    BRAND_TERMOWEB,
    DOMAIN,
)
from custom_components.termoweb.inventory import (
    HeaterNode,
    Inventory,
    build_node_inventory,
)
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
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], Inventory
    ],
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


def _patch_climate_attr(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: Any,
    *,
    raising: bool | None = None,
) -> None:
    """Patch a climate module attribute across shim + entity modules."""

    if raising is None:
        monkeypatch.setattr(climate_module, name, value)
        monkeypatch.setattr(entities_climate_module, name, value)
    else:
        monkeypatch.setattr(climate_module, name, value, raising=raising)
        monkeypatch.setattr(entities_climate_module, name, value, raising=raising)


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
        if isinstance(addrs_value, Iterable) and not isinstance(
            addrs_value, (str, bytes)
        ):
            _merge_addresses(node_type, addrs_value)
        if extras:
            section_extras.setdefault(node_type, {}).update(extras)

    settings_section = base_record.get("settings")
    if isinstance(settings_section, Mapping):
        for node_type, bucket in settings_section.items():
            _merge_settings(
                str(node_type), bucket if isinstance(bucket, Mapping) else None
            )

    addresses_section = base_record.get("addresses_by_type")
    inventory_obj = base_record.get("inventory")
    if isinstance(inventory_obj, Inventory):
        addresses_section = inventory_obj.addresses_by_type
    if isinstance(addresses_section, Mapping):
        for node_type, addrs in addresses_section.items():
            _merge_addresses(
                str(node_type), addrs if isinstance(addrs, Iterable) else None
            )

    nodes_by_type = base_record.get("nodes_by_type")
    if isinstance(nodes_by_type, Mapping):
        for node_type, section in nodes_by_type.items():
            _merge_section(
                str(node_type), section if isinstance(section, Mapping) else None
            )

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
        effective_inventory = Inventory(dev_id, node_list)

    return FakeCoordinator(
        hass,
        client=client,
        dev_id=dev_id,
        dev=normalised,
        nodes=None,
        inventory=effective_inventory,
        data={dev_id: normalised},
    )


def _attach_runtime(
    hass: HomeAssistant,
    entry_id: str,
    dev_id: str,
    *,
    coordinator: Any,
    client: Any | None = None,
    inventory: Inventory | None = None,
    version: str | None = None,
    brand: str | None = None,
    ws_state: Mapping[str, Any] | None = None,
) -> "EntryRuntime":
    """Attach an ``EntryRuntime`` to ``hass.data`` for climate tests."""

    runtime = build_entry_runtime(
        hass=hass,
        entry_id=entry_id,
        dev_id=dev_id,
        coordinator=coordinator,
        client=client,
        inventory=inventory,
        version=version or "",
        brand=brand or "",
    )
    if ws_state is not None:
        runtime.ws_state = dict(ws_state)
    return runtime


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

    _patch_climate_attr(monkeypatch, "normalize_node_type", _record_normalize)

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

        _attach_runtime(
            hass,
            entry_id,
            dev_id,
            coordinator=coordinator,
            client=AsyncMock(),
            inventory=inventory,
            version="3.1.4",
            brand=BRAND_TERMOWEB,
        )

        added: list[HeaterClimateEntity] = []

        def _async_add_entities(entities: list[HeaterClimateEntity]) -> None:
            added.extend(entities)

        platform = EntityPlatform()
        entity_platform_module._set_current_platform(platform)

        entry = types.SimpleNamespace(entry_id=entry_id)
        await async_setup_entry(hass, entry, _async_add_entities)

        assert len(added) == 3
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

        # Unnamed nodes get a translatable default device name; named ones don't.
        default_info = entities_by_addr["B2"].device_info
        assert default_info["name"] == "Heater B2"
        assert default_info["translation_key"] == "heater"
        assert default_info["translation_placeholders"] == {"addr": "B2"}
        assert "translation_key" not in entities_by_addr["A1"].device_info

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
    climate_inventory: Callable[[str, Mapping[str, Any]], Inventory],
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

        _attach_runtime(
            hass,
            entry_id,
            dev_id,
            coordinator=coordinator,
            client=AsyncMock(),
            inventory=inventory,
        )

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

        _patch_climate_attr(monkeypatch, "log_skipped_nodes", _mock_helper)

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
    climate_inventory: Callable[[str, Mapping[str, Any]], Inventory],
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

        _attach_runtime(
            hass,
            entry_id,
            dev_id,
            coordinator=coordinator,
            client=AsyncMock(),
            inventory=inventory,
        )

        added: list[HeaterClimateEntity] = []

        def _add_entities(entities: list[HeaterClimateEntity]) -> None:
            added.extend(entities)

        entry = types.SimpleNamespace(entry_id=entry_id)
        await async_setup_entry(hass, entry, _add_entities)

        assert len(added) == 1
        assert added[0]._attr_unique_id.endswith(":htr:7:climate")

    asyncio.run(_run())


def test_async_setup_entry_creates_accumulator_entity(
    climate_inventory: Callable[[str, Mapping[str, Any]], Inventory],
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

        runtime = _attach_runtime(
            hass,
            entry_id,
            dev_id,
            coordinator=coordinator,
            client=client,
            inventory=inventory,
            brand=BRAND_DUCAHEAT,
        )

        added: list[climate_module.HeaterClimateEntity] = []

        def _async_add_entities(
            entities: list[climate_module.HeaterClimateEntity],
        ) -> None:
            added.extend(entities)

        entry = types.SimpleNamespace(entry_id=entry_id)
        await async_setup_entry(hass, entry, _async_add_entities)
        backend = runtime.backend
        backend.set_node_settings.reset_mock()

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
        call = backend.set_node_settings.await_args
        assert call.args == (dev_id, ("acm", "7"))
        assert call.kwargs["prog"] == list(prog)
        assert call.kwargs["units"] == "C"
        backend.set_node_settings.reset_mock()

        runtime.brand = BRAND_TERMOWEB

        await acc.async_set_schedule(list(prog))
        call = backend.set_node_settings.await_args
        assert call.args == (dev_id, ("acm", "7"))
        assert call.kwargs["prog"] == list(prog)
        assert call.kwargs["units"] == "C"
        backend.set_node_settings.reset_mock()

        await acc.async_set_preset_temperatures(ptemp=[18.5, 19.5, 20.5])
        call = backend.set_node_settings.await_args
        assert call.kwargs["ptemp"] == [18.5, 19.5, 20.5]
        assert call.kwargs["units"] == "C"
        assert backend.set_node_settings.await_count == 1

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

        _attach_runtime(
            hass,
            entry_id,
            dev_id,
            coordinator=coordinator,
            client=AsyncMock(),
            inventory=inventory,
        )

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
        _patch_climate_attr(monkeypatch, "supports_boost", _supports_boost)

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

        _attach_runtime(
            hass,
            entry_id,
            dev_id,
            coordinator=coordinator,
            client=AsyncMock(),
            inventory=inventory,
        )

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

        _patch_climate_attr(monkeypatch, "supports_boost", _supports_boost)

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
    climate_inventory: Callable[[str, Mapping[str, Any]], Inventory],
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

    entity._optimistic_update(lambda payload: setattr(payload, "mode", "manual"))

    device_state = coordinator.data[dev_id]
    assert device_state["settings"]["htr"][addr]["mode"] == "manual"
    assert device_state["settings"]["acm"][addr]["mode"] == "manual"
    assert device_state["htr"]["settings"][addr]["mode"] == "manual"
    assert device_state["nodes_by_type"]["acm"]["settings"][addr]["mode"] == "manual"


def test_accumulator_hvac_mode_reporting() -> None:
    """Ensure accumulator HVAC mode normalisation covers all branches."""

    _reset_environment()
    hass = HomeAssistant()
    entry_id = "entry-acm-hvac"
    dev_id = "dev-acm-hvac"
    addr = "7"
    inventory_payload = {"nodes": [{"type": "acm", "addr": addr}]}
    inventory_list = list(inventory_module.build_node_inventory(inventory_payload))
    inventory = Inventory(dev_id, inventory_list)
    settings: dict[str, Any] = {"mode": "off", "units": "C"}
    coordinator = _make_coordinator(
        hass,
        dev_id,
        {
            "nodes": {},
            "nodes_by_type": {"acm": {"settings": {addr: settings}}},
            "htr": {"settings": {}},
        },
        inventory=inventory,
    )
    _attach_runtime(
        hass,
        entry_id,
        dev_id,
        coordinator=coordinator,
        client=AsyncMock(),
        brand=BRAND_TERMOWEB,
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

    assert entity._default_mode_for_setpoint() is None
    assert entity._requires_setpoint_with_mode(HVACMode.AUTO) is False
    assert entity._allows_setpoint_in_mode(HVACMode.AUTO) is True
    assert entity.hvac_mode == HVACMode.OFF
    settings["mode"] = "auto"
    assert coordinator.apply_entity_patch(
        "acm", addr, lambda cur: setattr(cur, "mode", "auto")
    )
    assert entity.hvac_mode == HVACMode.AUTO
    assert coordinator.apply_entity_patch(
        "acm", addr, lambda cur: setattr(cur, "mode", "boost")
    )
    assert entity.hvac_mode == HVACMode.AUTO
    assert entity.preset_mode == "boost"


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

    _patch_climate_attr(monkeypatch, "coerce_boost_minutes", _fake)
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
        "boost_end_day": 12,
        "boost_end_min": 90,
        "boost_end_datetime": dt.datetime(2024, 1, 1, 3, 0, tzinfo=dt.timezone.utc),
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

    _attach_runtime(
        hass,
        entry_id,
        dev_id,
        coordinator=coordinator,
        client=AsyncMock(),
        brand=BRAND_TERMOWEB,
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

    original_now = dt_util.NOW
    try:
        dt_util.NOW = dt.datetime(2024, 1, 1, 0, 0, tzinfo=dt.timezone.utc)
        attrs = entity.extra_state_attributes
    finally:
        dt_util.NOW = original_now

    assert attrs["boost_active"] is True
    assert attrs["boost_minutes_remaining"] == 180
    assert attrs["boost_end"] == "2024-01-01T03:00:00+00:00"
    assert attrs["boost_end_label"] is None


def test_accumulator_extra_state_attributes_varied_inputs() -> None:
    """Accumulator boost metadata should normalise a range of inputs."""

    _reset_environment()
    hass = HomeAssistant()
    entry_id = "entry-acm-variants"
    dev_id = "dev-acm-variants"
    addr = "8"
    inventory_payload = {"nodes": [{"type": "acm", "addr": addr}]}
    inventory_list = list(inventory_module.build_node_inventory(inventory_payload))
    inventory = Inventory(dev_id, inventory_list)
    settings = {
        "mode": "auto",
        "units": "C",
        "prog": [0] * 168,
        "boost_active": 1,
        "boost_end_day": 2,
        "boost_end_min": 60,
        "boost_remaining": None,
        "charging": False,
        "current_charge_per": 67,
        "target_charge_per": 90,
    }

    coordinator = _make_coordinator(
        hass,
        dev_id,
        {
            "nodes": {},
            "nodes_by_type": {"acm": {"settings": {addr: settings}}},
            "htr": {"settings": {}},
        },
        inventory=inventory,
    )

    _attach_runtime(
        hass,
        entry_id,
        dev_id,
        coordinator=coordinator,
        client=AsyncMock(),
        brand=BRAND_TERMOWEB,
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
    assert attrs["charging"] is False
    assert attrs["current_charge_per"] == 67
    assert attrs["target_charge_per"] == 90

    class RaisingResolver:
        def __call__(
            self, day: Any, minute: Any
        ) -> tuple[dt.datetime | None, int | None]:
            raise ValueError("resolver error")

    settings["boost_active"] = " OFF "
    settings["boost_end_day"] = None
    settings["boost_end_min"] = None
    settings["boost_remaining"] = ""
    coordinator.resolve_boost_end = RaisingResolver()  # type: ignore[assignment]

    def _apply_first(cur: Any) -> None:
        cur.boost_active = settings["boost_active"]
        cur.boost_end_day = settings["boost_end_day"]
        cur.boost_end_min = settings["boost_end_min"]
        cur.boost_remaining = settings["boost_remaining"]

    assert coordinator.apply_entity_patch(
        "acm",
        addr,
        _apply_first,
    )

    attrs = entity.extra_state_attributes
    assert attrs["boost_active"] is False
    assert attrs["boost_end_label"] == "Never"
    assert attrs["boost_minutes_remaining"] is None
    assert attrs["boost_end"] is None
    assert attrs["boost_end_label"] == "Never"

    settings["boost_active"] = " maybe "
    settings["mode"] = "auto"
    settings["boost_remaining"] = None
    coordinator.resolve_boost_end = None  # type: ignore[assignment]

    def _apply_second(cur: Any) -> None:
        cur.boost_active = settings["boost_active"]
        cur.mode = settings["mode"]
        cur.boost_remaining = settings["boost_remaining"]

    assert coordinator.apply_entity_patch(
        "acm",
        addr,
        _apply_second,
    )

    attrs = entity.extra_state_attributes
    assert attrs["boost_active"] is False

    settings["boost_active"] = 0
    settings["boost_remaining"] = 7.5
    coordinator.resolve_boost_end = None  # type: ignore[assignment]

    def _apply_third(cur: Any) -> None:
        cur.boost_active = settings["boost_active"]
        cur.boost_remaining = settings["boost_remaining"]

    assert coordinator.apply_entity_patch(
        "acm",
        addr,
        _apply_third,
    )

    attrs = entity.extra_state_attributes
    assert attrs["boost_active"] is False
    assert attrs["boost_minutes_remaining"] == 7
    assert attrs["boost_end"] == "2024-01-01T00:07:00+00:00"
    assert attrs["boost_end_label"] is None


def test_accumulator_submit_settings_brand_switch() -> None:
    """Verify accumulator writes route through the backend helper."""

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
        runtime = _attach_runtime(
            hass,
            entry_id,
            dev_id,
            coordinator=coordinator,
            client=AsyncMock(),
            brand=BRAND_DUCAHEAT,
        )
        backend = runtime.backend
        runtime.backend = backend

        entity = climate_module.AccumulatorClimateEntity(
            coordinator,
            entry_id,
            dev_id,
            addr,
            "Accumulator",
            node_type="acm",
        )
        entity.hass = hass

        await entity._async_submit_settings(
            backend,
            mode="auto",
            stemp=21.0,
            prog=None,
            ptemp=None,
            units="C",
        )
        call = backend.set_node_settings.await_args
        assert call.args == (dev_id, ("acm", addr))
        assert call.kwargs["mode"] == "auto"

        runtime.brand = BRAND_TERMOWEB

        await entity._async_submit_settings(
            backend,
            mode="manual",
            stemp=19.0,
            prog=[0] * 168,
            ptemp=[18.0, 19.0, 20.0],
            units="C",
        )
        call = backend.set_node_settings.await_args
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
        settings = {
            "boost_active": True,
            "mode": "auto",
            "units": "C",
            "prog": [0] * 168,
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

        entity = climate_module.AccumulatorClimateEntity(
            coordinator,
            entry_id,
            dev_id,
            addr,
            "Accumulator",
            node_type="acm",
        )
        entity.hass = hass

        runtime = _attach_runtime(
            hass,
            entry_id,
            dev_id,
            coordinator=coordinator,
            client=AsyncMock(),
            brand=BRAND_DUCAHEAT,
        )
        backend = runtime.backend
        runtime.backend = backend

        entity.accumulator_state = MagicMock(
            return_value=types.SimpleNamespace(boost_active=True, mode="auto")
        )

        def _boom() -> Any:
            raise RuntimeError("boom")

        entity.boost_state = MagicMock(side_effect=_boom)  # type: ignore[assignment]

        await entity._async_submit_settings(
            backend,
            mode="auto",
            stemp=None,
            prog=None,
            ptemp=None,
            units="C",
        )

        call = backend.set_node_settings.await_args
        boost_context = call.kwargs["boost_context"]
        assert boost_context.active is None
        assert boost_context.mode == "auto"

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

        backend = AsyncMock()
        runtime = _attach_runtime(
            hass,
            entry_id,
            dev_id,
            coordinator=coordinator,
            client=AsyncMock(),
            brand=BRAND_DUCAHEAT,
        )
        runtime.backend = backend

        entity.boost_state = MagicMock(return_value=types.SimpleNamespace(active=None))  # type: ignore[assignment]
        entity.accumulator_state = MagicMock(
            return_value=types.SimpleNamespace(
                boost_active="maybe", boost="no", mode=" Boost "
            )
        )

        await entity._async_submit_settings(
            backend,
            mode="auto",
            stemp=None,
            prog=None,
            ptemp=None,
            units="C",
        )

        call = backend.set_node_settings.await_args
        boost_context = call.kwargs["boost_context"]
        assert boost_context.active is None
        assert boost_context.mode == " Boost "

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
        heater._refresh_fallback = fallback

        await heater._commit_write(
            log_context="Test write",
            write_kwargs={"prog": [0, 1, 2]},
            apply_fn=apply_fn,
            success_details={"detail": "value"},
        )

        async_write.assert_awaited_once_with(log_context="Test write", prog=[0, 1, 2])
        optimistic.assert_called_once_with(apply_fn)
        fallback.schedule.assert_called_once()

    asyncio.run(_run())


def test_async_setup_entry_without_inventory_skips_entities(
    runtime_factory: Callable[..., Any],
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

        runtime = runtime_factory(
            hass=hass,
            entry_id=entry.entry_id,
            dev_id=dev_id,
            coordinator=coordinator,
            client=AsyncMock(),
        )
        runtime.inventory = "invalid"

        added: list[HeaterClimateEntity] = []

        def _async_add_entities(entities: list[HeaterClimateEntity]) -> None:
            added.extend(entities)

        with pytest.raises(TypeError):
            await async_setup_entry(hass, entry, _async_add_entities)

        assert added == []
        runtime_after = hass.data[DOMAIN][entry.entry_id]
        assert runtime_after.inventory == "invalid"

    asyncio.run(_run())


def test_async_setup_entry_reuses_coordinator_inventory(
    climate_inventory: Callable[[str, Mapping[str, Any]], Inventory],
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
        runtime = _attach_runtime(
            hass,
            entry.entry_id,
            dev_id,
            coordinator=coordinator,
            client=AsyncMock(),
            inventory=inventory,
        )

        added: list[HeaterClimateEntity] = []

        def _async_add_entities(entities: list[HeaterClimateEntity]) -> None:
            added.extend(entities)

        await async_setup_entry(hass, entry, _async_add_entities)

        assert len(added) == 1
        assert runtime.inventory is inventory

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


def test_heater_setpoint_uses_modified_auto_mode() -> None:
    async def _run() -> None:
        _reset_environment()

        hass = HomeAssistant()
        entry_id = "entry-auto"
        dev_id = "dev-auto"
        addr = "1"
        coordinator_client = AsyncMock()
        settings = {
            "mode": "auto",
            "state": "idle",
            "mtemp": "20.0",
            "stemp": "21.0",
            "ptemp": ["16.0", "18.0", "20.0"],
            "prog": [0] * 168,
            "units": "C",
        }
        inventory = Inventory(
            dev_id,
            list(build_node_inventory({"nodes": [{"type": "htr", "addr": addr}]})),
        )
        coordinator = _make_coordinator(
            hass,
            dev_id,
            {
                "nodes": {},
                "nodes_by_type": {"htr": {"settings": {addr: settings}}},
                "htr": {"settings": {addr: settings}},
            },
            client=coordinator_client,
            inventory=inventory,
        )
        runtime = _attach_runtime(
            hass,
            entry_id,
            dev_id,
            coordinator=coordinator,
            client=coordinator_client,
            inventory=inventory,
        )
        backend = runtime.backend
        heater = HeaterClimateEntity(coordinator, entry_id, dev_id, addr, "Heater")
        heater.hass = hass

        async def fast_sleep(_delay: float) -> None:
            return None

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(climate_module.asyncio, "sleep", fast_sleep)
        try:
            assert heater._default_mode_for_setpoint() == "modified_auto"

            await heater.async_set_temperature(**{ATTR_TEMPERATURE: 22.5})
            assert heater._write_task is not None
            await heater._write_task
            call = backend.set_node_settings.await_args
            assert call.kwargs["mode"] == "modified_auto"
            assert call.kwargs["stemp"] == pytest.approx(22.5)
            pending = coordinator.pending_settings[("htr", addr)]
            assert pending["mode"] == "modified_auto"
            assert pending["stemp"] == pytest.approx(22.5)

            backend.set_node_settings.reset_mock()
            heater._pending_mode = None
            heater._pending_stemp = 23.0
            await heater._write_after_debounce()
            call = backend.set_node_settings.await_args
            assert call.kwargs["mode"] == "modified_auto"
            assert call.kwargs["stemp"] == pytest.approx(23.0)

            await heater.async_set_hvac_mode(HVACMode.HEAT)
            assert heater._write_task is not None
            await heater._write_task
            call = backend.set_node_settings.await_args
            assert call.kwargs["mode"] == "manual"
        finally:
            monkeypatch.undo()

    asyncio.run(_run())


def test_heater_mode_mapping_for_modified_auto() -> None:
    """Map modified_auto backend state to AUTO + temporary_override."""

    _reset_environment()
    hass = HomeAssistant()
    dev_id = "dev-mod-auto"
    addr = "4"
    settings = {
        "mode": "modified_auto",
        "state": "on",
        "mtemp": "20.1",
        "stemp": "21.0",
        "ptemp": ["16.0", "18.0", "20.0"],
        "prog": [0] * 168,
        "units": "C",
    }
    inventory = Inventory(
        dev_id,
        list(build_node_inventory({"nodes": [{"type": "htr", "addr": addr}]})),
    )
    coordinator = _make_coordinator(
        hass,
        dev_id,
        {
            "nodes": {},
            "nodes_by_type": {"htr": {"settings": {addr: settings}}},
            "htr": {"settings": {addr: settings}},
        },
        client=AsyncMock(),
        inventory=inventory,
    )

    entity = HeaterClimateEntity(coordinator, "entry-mod-auto", dev_id, addr, "Heater")

    assert entity.hvac_mode == HVACMode.AUTO
    assert entity.preset_mode == "temporary_override"


# ---------------------------------------------------------------------------
# New tests targeting uncovered lines
# ---------------------------------------------------------------------------


def test_heater_icon_reflects_off_and_idle_states(
    climate_inventory: Callable[[str, Mapping[str, Any]], Inventory],
) -> None:
    """Icon property should return distinct icons for OFF, HEATING, and IDLE."""

    _reset_environment()
    hass = HomeAssistant()
    dev_id = "dev-icon"
    addr = "1"
    raw_nodes = {"nodes": [{"type": "htr", "addr": addr}]}
    inventory = climate_inventory(dev_id, raw_nodes)

    # OFF mode -> radiator-off
    settings_off = {"mode": "off", "state": "off", "units": "C"}
    coordinator = _make_coordinator(
        hass,
        dev_id,
        {
            "nodes": raw_nodes,
            "nodes_by_type": {"htr": {"settings": {addr: settings_off}}},
            "htr": {"settings": {addr: settings_off}},
        },
        inventory=inventory,
    )
    entity = HeaterClimateEntity(
        coordinator, "entry-icon", dev_id, addr, "Heater",
        node_type="htr", inventory=inventory,
    )
    assert entity.icon == "mdi:radiator-off"

    # IDLE state (heater on but idle)
    settings_idle = {"mode": "auto", "state": "idle", "units": "C"}
    coordinator2 = _make_coordinator(
        hass,
        dev_id,
        {
            "nodes": raw_nodes,
            "nodes_by_type": {"htr": {"settings": {addr: settings_idle}}},
            "htr": {"settings": {addr: settings_idle}},
        },
        inventory=inventory,
    )
    entity2 = HeaterClimateEntity(
        coordinator2, "entry-icon2", dev_id, addr, "Heater",
        node_type="htr", inventory=inventory,
    )
    assert entity2.icon == "mdi:radiator-disabled"

    # HEATING state
    settings_heat = {"mode": "manual", "state": "on", "units": "C"}
    coordinator3 = _make_coordinator(
        hass,
        dev_id,
        {
            "nodes": raw_nodes,
            "nodes_by_type": {"htr": {"settings": {addr: settings_heat}}},
            "htr": {"settings": {addr: settings_heat}},
        },
        inventory=inventory,
    )
    entity3 = HeaterClimateEntity(
        coordinator3, "entry-icon3", dev_id, addr, "Heater",
        node_type="htr", inventory=inventory,
    )
    assert entity3.icon == "mdi:radiator"


def test_heater_async_will_remove_clears_entity_id() -> None:
    """Removing from HA should clear entity ID and cancel refresh fallback."""

    async def _run() -> None:
        _reset_environment()
        hass = HomeAssistant()
        dev_id = "dev-remove"
        addr = "1"
        coordinator = _make_coordinator(
            hass,
            dev_id,
            {"htr": {"settings": {addr: {}}}, "nodes": {}},
        )
        _attach_runtime(
            hass, "entry-remove", dev_id,
            coordinator=coordinator, client=AsyncMock(),
        )

        entity = HeaterClimateEntity(
            coordinator, "entry-remove", dev_id, addr, "Heater",
        )
        entity.hass = hass

        entity._refresh_fallback = MagicMock()

        await entity.async_will_remove_from_hass()
        entity._refresh_fallback.cancel.assert_called_once()

    asyncio.run(_run())


def test_shared_inventory_falls_back_to_underscore_attr() -> None:
    """_shared_inventory should check both 'inventory' and '_inventory' attrs."""

    _reset_environment()
    hass = HomeAssistant()
    dev_id = "dev-inv"
    addr = "1"
    raw_nodes = {"nodes": [{"type": "htr", "addr": addr}]}
    node_list = list(inventory_module.build_node_inventory(raw_nodes))
    inventory = Inventory(dev_id, node_list)

    coordinator = _make_coordinator(
        hass, dev_id,
        {"htr": {"settings": {addr: {}}}, "nodes": {}},
        inventory=inventory,
    )
    entity = HeaterClimateEntity(coordinator, "entry-inv", dev_id, addr, "Heater")

    # When coordinator has 'inventory' attribute
    assert entity._shared_inventory() is inventory

    # When coordinator only has '_inventory'
    del coordinator.inventory
    coordinator._inventory = inventory
    assert entity._shared_inventory() is inventory

    # When coordinator has neither
    del coordinator._inventory
    assert entity._shared_inventory() is None


def test_current_prog_slot_returns_none_for_short_prog() -> None:
    """_current_prog_slot should return None when prog is too short."""

    _reset_environment()
    hass = HomeAssistant()
    dev_id = "dev-slot"
    coordinator = _make_coordinator(
        hass, dev_id, {"htr": {"settings": {}}, "nodes": {}},
    )
    entity = HeaterClimateEntity(coordinator, "entry-slot", dev_id, "1", "Heater")

    # Short prog list
    state = types.SimpleNamespace(prog=[0, 1, 2])
    assert entity._current_prog_slot(state) is None

    # None prog
    state2 = types.SimpleNamespace(prog=None)
    assert entity._current_prog_slot(state2) is None

    # Valid prog
    full_prog = [2] * 168
    state3 = types.SimpleNamespace(prog=full_prog)
    slot = entity._current_prog_slot(state3)
    assert slot == 2

    # Prog with bad value at index
    bad_prog = ["not_int"] * 168
    state4 = types.SimpleNamespace(prog=bad_prog)
    result = entity._current_prog_slot(state4)
    # "not_int" cannot be converted to int, should return None
    assert result is None


def test_accumulator_set_preset_mode_toggles_boost(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Accumulator preset_mode transitions should start/cancel boost."""

    async def _run() -> None:
        _reset_environment()
        hass = HomeAssistant()
        dev_id = "dev-acm-preset"
        addr = "5"

        settings = {"mode": "auto", "units": "C", "stemp": "21.0", "boost_temp": "22.0"}
        coordinator = _make_coordinator(
            hass, dev_id,
            {
                "nodes": {},
                "nodes_by_type": {"acm": {"settings": {addr: settings}}},
                "htr": {"settings": {}},
            },
        )
        _attach_runtime(
            hass, "entry-acm-preset", dev_id,
            coordinator=coordinator, client=AsyncMock(),
        )

        entity = climate_module.AccumulatorClimateEntity(
            coordinator, "entry-acm-preset", dev_id, addr, "Accumulator",
            node_type="acm",
        )
        entity.hass = hass

        # Mock the boost methods
        entity.async_start_boost = AsyncMock()
        entity.async_cancel_boost = AsyncMock()

        # Setting same preset does nothing
        assert entity.preset_mode == "none"
        await entity.async_set_preset_mode("none")
        entity.async_start_boost.assert_not_called()
        entity.async_cancel_boost.assert_not_called()

        # Setting boost starts boost
        await entity.async_set_preset_mode("boost")
        entity.async_start_boost.assert_awaited_once()

    asyncio.run(_run())


def test_accumulator_set_preset_mode_cancels_boost() -> None:
    """Setting preset_mode to 'none' from 'boost' should cancel boost."""

    async def _run() -> None:
        _reset_environment()
        hass = HomeAssistant()
        dev_id = "dev-acm-cancel-preset"
        addr = "4"

        settings = {"mode": "boost", "units": "C", "stemp": "21.0", "boost_active": True}
        coordinator = _make_coordinator(
            hass, dev_id,
            {
                "nodes": {},
                "nodes_by_type": {"acm": {"settings": {addr: settings}}},
                "htr": {"settings": {}},
            },
        )
        client = AsyncMock()
        client.set_acm_boost_state = AsyncMock()
        runtime = _attach_runtime(
            hass, "entry-acm-cancel-preset", dev_id,
            coordinator=coordinator, client=client,
        )
        runtime.backend = client

        entity = climate_module.AccumulatorClimateEntity(
            coordinator, "entry-acm-cancel-preset", dev_id, addr, "Accumulator",
            node_type="acm",
        )
        entity.hass = hass
        entity.async_write_ha_state = MagicMock()

        assert entity.preset_mode == "boost"

        # Set a resume mode to exercise that branch
        entity._boost_resume_mode = HVACMode.AUTO

        # Cancel boost by setting preset to "none"
        await entity.async_set_preset_mode("none")

        # Should have called cancel boost
        client.set_acm_boost_state.assert_awaited_once()
        call = client.set_acm_boost_state.await_args
        assert call.kwargs["boost"] is False

        # Resume mode should be cleared
        assert entity._boost_resume_mode is None

    asyncio.run(_run())


def test_accumulator_set_acm_preset_validates_and_writes(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """async_set_acm_preset should validate inputs and call backend."""

    async def _run() -> None:
        _reset_environment()
        hass = HomeAssistant()
        dev_id = "dev-acm-preset-write"
        addr = "6"

        settings = {"mode": "auto", "units": "C"}
        coordinator = _make_coordinator(
            hass, dev_id,
            {
                "nodes": {},
                "nodes_by_type": {"acm": {"settings": {addr: settings}}},
                "htr": {"settings": {}},
            },
        )
        client = AsyncMock()
        client.set_acm_extra_options = AsyncMock()
        runtime = _attach_runtime(
            hass, "entry-acm-preset-write", dev_id,
            coordinator=coordinator, client=client,
        )
        runtime.backend = client

        entity = climate_module.AccumulatorClimateEntity(
            coordinator, "entry-acm-preset-write", dev_id, addr, "Accumulator",
            node_type="acm",
        )
        entity.hass = hass
        entity.async_write_ha_state = MagicMock()

        # Valid minutes and temperature
        await entity.async_set_acm_preset(minutes=60, temperature=22.5)
        client.set_acm_extra_options.assert_awaited_once()
        call = client.set_acm_extra_options.await_args
        assert call.kwargs["boost_time"] == 60
        assert call.kwargs["boost_temp"] == pytest.approx(22.5)

    asyncio.run(_run())


def test_accumulator_start_boost_calls_backend(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """async_start_boost should validate and delegate to the backend."""

    async def _run() -> None:
        _reset_environment()
        hass = HomeAssistant()
        dev_id = "dev-acm-start"
        addr = "8"

        settings = {
            "mode": "auto", "units": "C", "stemp": "21.0",
            "boost_temp": "23.0", "boost_active": False,
        }
        coordinator = _make_coordinator(
            hass, dev_id,
            {
                "nodes": {},
                "nodes_by_type": {"acm": {"settings": {addr: settings}}},
                "htr": {"settings": {}},
            },
        )
        client = AsyncMock()
        client.set_acm_boost_state = AsyncMock()
        runtime = _attach_runtime(
            hass, "entry-acm-start", dev_id,
            coordinator=coordinator, client=client,
        )
        runtime.backend = client

        entity = climate_module.AccumulatorClimateEntity(
            coordinator, "entry-acm-start", dev_id, addr, "Accumulator",
            node_type="acm",
        )
        entity.hass = hass
        entity.async_write_ha_state = MagicMock()

        # Start boost with explicit minutes
        await entity.async_start_boost(minutes=60)
        client.set_acm_boost_state.assert_awaited_once()
        call = client.set_acm_boost_state.await_args
        assert call.kwargs["boost"] is True
        assert call.kwargs["boost_time"] == 60
        assert call.kwargs["stemp"] == pytest.approx(23.0)

        # Start boost with default minutes (no explicit value)
        client.set_acm_boost_state.reset_mock()
        await entity.async_start_boost()
        # Should use _preferred_boost_minutes() as fallback
        assert client.set_acm_boost_state.await_count == 1

    asyncio.run(_run())


def test_accumulator_cancel_boost_calls_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """async_cancel_boost should call backend and reset optimistic state."""

    async def _run() -> None:
        _reset_environment()
        hass = HomeAssistant()
        dev_id = "dev-acm-cancel-boost"
        addr = "9"

        settings = {
            "mode": "boost", "units": "C", "boost_active": True,
            "boost_remaining": 30, "boost_end_day": 5, "boost_end_min": 90,
        }
        coordinator = _make_coordinator(
            hass, dev_id,
            {
                "nodes": {},
                "nodes_by_type": {"acm": {"settings": {addr: settings}}},
                "htr": {"settings": {}},
            },
        )
        client = AsyncMock()
        client.set_acm_boost_state = AsyncMock()
        runtime = _attach_runtime(
            hass, "entry-acm-cancel-boost", dev_id,
            coordinator=coordinator, client=client,
        )
        runtime.backend = client

        entity = climate_module.AccumulatorClimateEntity(
            coordinator, "entry-acm-cancel-boost", dev_id, addr, "Accumulator",
            node_type="acm",
        )
        entity.hass = hass
        entity.async_write_ha_state = MagicMock()

        assert entity.preset_mode == "boost"

        await entity.async_cancel_boost()

        client.set_acm_boost_state.assert_awaited_once()
        call = client.set_acm_boost_state.await_args
        assert call.kwargs["boost"] is False

    asyncio.run(_run())


def test_accumulator_extra_state_attributes_charging_non_bool() -> None:
    """Accumulator attributes should coerce non-bool charging values."""

    _reset_environment()
    hass = HomeAssistant()
    dev_id = "dev-acm-charge"
    addr = "10"

    settings = {
        "mode": "auto", "units": "C", "prog": [0] * 168,
        "charging": 1,  # non-bool truthy value
        "current_charge_per": 55.5,
        "target_charge_per": 80,
    }

    coordinator = _make_coordinator(
        hass, dev_id,
        {
            "nodes": {},
            "nodes_by_type": {"acm": {"settings": {addr: settings}}},
            "htr": {"settings": {}},
        },
    )
    _attach_runtime(
        hass, "entry-acm-charge", dev_id,
        coordinator=coordinator, client=AsyncMock(),
        brand="termoweb",
    )

    entity = climate_module.AccumulatorClimateEntity(
        coordinator, "entry-acm-charge", dev_id, addr, "Accumulator",
        node_type="acm",
    )
    entity.hass = hass

    attrs = entity.extra_state_attributes
    # Non-bool charging should be coerced to bool
    assert attrs["charging"] is True
    assert attrs["current_charge_per"] == 55
    assert attrs["target_charge_per"] == 80


def test_accumulator_async_submit_settings_non_bool_boost_active() -> None:
    """_async_submit_settings should handle non-bool boost_active values."""

    async def _run() -> None:
        _reset_environment()
        hass = HomeAssistant()
        entry_id = "entry-acm-nbool"
        dev_id = "dev-acm-nbool"
        addr = "12"
        settings = {"mode": "auto", "units": "C", "boost_active": "maybe"}

        coordinator = _make_coordinator(
            hass, dev_id,
            {
                "nodes": {},
                "nodes_by_type": {"acm": {"settings": {addr: settings}}},
                "htr": {"settings": {}},
            },
        )
        runtime = _attach_runtime(
            hass, entry_id, dev_id,
            coordinator=coordinator, client=AsyncMock(),
        )
        backend = runtime.backend

        entity = climate_module.AccumulatorClimateEntity(
            coordinator, entry_id, dev_id, addr, "Accumulator", node_type="acm",
        )
        entity.hass = hass

        await entity._async_submit_settings(
            backend, mode="auto", stemp=None, prog=None, ptemp=None, units="C",
        )
        call = backend.set_node_settings.await_args
        boost_context = call.kwargs["boost_context"]
        # non-bool boost_active should make boost_flag None
        assert boost_context.mode == "auto"

    asyncio.run(_run())
