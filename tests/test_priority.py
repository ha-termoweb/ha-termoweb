# ruff: noqa: D100,D101,D102,D103,INP001,E402
"""Tests for the priority field across domain state, codecs, number entity, and planner."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from unittest.mock import AsyncMock

import pytest

from conftest import FakeCoordinator, _install_stubs, build_coordinator_device_state

_install_stubs()

from custom_components.termoweb.codecs.ducaheat_codec import (
    encode_priority_command,
    infer_status_endpoint,
)
from custom_components.termoweb.codecs.ducaheat_models import PriorityWritePayload
from custom_components.termoweb.codecs.ducaheat_read_models import (
    DucaheatSegmentedSettings,
    DucaheatSetupSegment,
)
from custom_components.termoweb.codecs.termoweb_models import HeaterSettingsPayload
from custom_components.termoweb.domain.commands import (
    BaseCommand,
    SetPriority,
)
from custom_components.termoweb.domain.ids import NodeId, NodeType
from custom_components.termoweb.domain.state import (
    HeaterState,
    _populate_heater_state,
    canonicalize_settings_payload,
)
from custom_components.termoweb.entities.number import HeaterPriorityNumber
from custom_components.termoweb.inventory import Inventory, build_node_inventory
from custom_components.termoweb.planner.ducaheat_planner import (
    PlannedHttpCall,
    _build_write_call,
    plan_command,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import EntityCategory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def heater_inventory(inventory_builder) -> Inventory:
    """Return helper inventory for heater priority tests."""
    dev_id = "dev-htr"
    raw_nodes = {"nodes": [{"type": "htr", "addr": "H1"}]}
    node_list = build_node_inventory(raw_nodes)
    return inventory_builder(dev_id, raw_nodes, node_list)


def _make_heater_coordinator(
    inventory: Inventory,
    *,
    payload: Mapping[str, Mapping[str, Any]] | None = None,
) -> FakeCoordinator:
    """Construct a fake coordinator with heater settings."""
    hass = HomeAssistant()
    dev_id = "dev-htr"
    nodes = {"nodes": [{"type": "htr", "addr": "H1"}]}
    settings = payload or {"htr": {"H1": {}}}
    record = build_coordinator_device_state(nodes=nodes, settings=settings)
    return FakeCoordinator(
        hass,
        dev_id=dev_id,
        dev=record,
        nodes=nodes,
        inventory=inventory,
        data={dev_id: record},
    )


# ---------------------------------------------------------------------------
# 1. _populate_heater_state tests
# ---------------------------------------------------------------------------


class TestPopulateHeaterStatePriority:
    """Tests for priority extraction in _populate_heater_state."""

    def test_integer_priority_zero(self):
        state = _populate_heater_state(HeaterState(), {"priority": 0})
        assert state.priority == 0

    def test_integer_priority_15(self):
        state = _populate_heater_state(HeaterState(), {"priority": 15})
        assert state.priority == 15

    def test_integer_priority_30(self):
        state = _populate_heater_state(HeaterState(), {"priority": 30})
        assert state.priority == 30

    def test_string_wrapped_zero(self):
        state = _populate_heater_state(HeaterState(), {"priority": "0"})
        assert state.priority == 0

    def test_string_wrapped_15(self):
        state = _populate_heater_state(HeaterState(), {"priority": "15"})
        assert state.priority == 15

    def test_non_numeric_string_returns_none(self):
        state = _populate_heater_state(HeaterState(), {"priority": "medium"})
        assert state.priority is None

    def test_absent_priority_remains_none(self):
        state = _populate_heater_state(HeaterState(), {"mode": "on"})
        assert state.priority is None

    def test_none_priority_sets_none(self):
        state = _populate_heater_state(HeaterState(), {"priority": None})
        assert state.priority is None

    def test_float_priority_coerced_to_int(self):
        state = _populate_heater_state(HeaterState(), {"priority": 5.7})
        assert state.priority == 5


# ---------------------------------------------------------------------------
# 2. HeaterSettingsPayload (TermoWeb codec) tests
# ---------------------------------------------------------------------------


class TestTermoWebHeaterSettingsPayloadPriority:
    """Tests for priority field on HeaterSettingsPayload."""

    def test_integer_priority_preserved(self):
        payload = HeaterSettingsPayload.model_validate({"priority": 10})
        assert payload.priority == 10

    def test_none_priority_preserved(self):
        payload = HeaterSettingsPayload.model_validate({})
        assert payload.priority is None

    def test_priority_in_model_dump(self):
        payload = HeaterSettingsPayload.model_validate({"priority": 5})
        dumped = payload.model_dump(exclude_none=True)
        assert dumped["priority"] == 5


# ---------------------------------------------------------------------------
# 3. DucaheatSetupSegment tests
# ---------------------------------------------------------------------------


class TestDucaheatSetupSegmentPriority:
    """Tests for priority coercion in DucaheatSetupSegment."""

    def test_integer_priority(self):
        seg = DucaheatSetupSegment.model_validate({"priority": 10})
        assert seg.priority == 10

    def test_string_priority_coerced_to_int(self):
        seg = DucaheatSetupSegment.model_validate({"priority": "15"})
        assert seg.priority == 15

    def test_non_numeric_string_coerced_to_none(self):
        seg = DucaheatSetupSegment.model_validate({"priority": "medium"})
        assert seg.priority is None

    def test_none_priority(self):
        seg = DucaheatSetupSegment.model_validate({"priority": None})
        assert seg.priority is None

    def test_absent_priority(self):
        seg = DucaheatSetupSegment.model_validate({})
        assert seg.priority is None


# ---------------------------------------------------------------------------
# 4. DucaheatSegmentedSettings.to_flat_dict tests
# ---------------------------------------------------------------------------


class TestDucaheatSegmentedSettingsToFlatDict:
    """Tests that to_flat_dict surfaces priority from setup segment."""

    def test_priority_included_for_heater(self):
        settings = DucaheatSegmentedSettings.model_validate(
            {
                "status": {"mode": "on"},
                "setup": {"priority": 5},
            }
        )
        flat = settings.to_flat_dict(accumulator=False)
        assert flat["priority"] == 5

    def test_priority_included_for_accumulator(self):
        settings = DucaheatSegmentedSettings.model_validate(
            {
                "status": {"mode": "on"},
                "setup": {"priority": 20},
            }
        )
        flat = settings.to_flat_dict(accumulator=True)
        assert flat["priority"] == 20

    def test_priority_absent_when_setup_has_none(self):
        settings = DucaheatSegmentedSettings.model_validate(
            {
                "status": {"mode": "on"},
                "setup": {},
            }
        )
        flat = settings.to_flat_dict(accumulator=False)
        assert "priority" not in flat

    def test_priority_absent_when_no_setup(self):
        settings = DucaheatSegmentedSettings.model_validate(
            {
                "status": {"mode": "on"},
            }
        )
        flat = settings.to_flat_dict(accumulator=False)
        assert "priority" not in flat

    def test_string_priority_coerced_in_flat_dict(self):
        settings = DucaheatSegmentedSettings.model_validate(
            {
                "status": {"mode": "on"},
                "setup": {"priority": "10"},
            }
        )
        flat = settings.to_flat_dict(accumulator=False)
        assert flat["priority"] == 10


# ---------------------------------------------------------------------------
# 5. canonicalize_settings_payload tests
# ---------------------------------------------------------------------------


class TestCanonicalizeSettingsPayloadPriority:
    """Tests that canonicalize_settings_payload passes through priority."""

    def test_priority_passes_through(self):
        result = canonicalize_settings_payload({"priority": 5})
        assert result["priority"] == 5

    def test_priority_absent_when_not_in_payload(self):
        result = canonicalize_settings_payload({"mode": "on"})
        assert "priority" not in result


# ---------------------------------------------------------------------------
# 6. SetPriority command tests
# ---------------------------------------------------------------------------


class TestSetPriorityCommand:
    """Tests for SetPriority command creation."""

    def test_create_set_priority(self):
        cmd = SetPriority(priority=10)
        assert cmd.priority == 10

    def test_set_priority_is_base_command(self):
        cmd = SetPriority(priority=5)
        assert isinstance(cmd, BaseCommand)

    def test_set_priority_zero(self):
        cmd = SetPriority(priority=0)
        assert cmd.priority == 0

    def test_set_priority_max(self):
        cmd = SetPriority(priority=30)
        assert cmd.priority == 30


# ---------------------------------------------------------------------------
# 7. encode_priority_command tests
# ---------------------------------------------------------------------------


class TestEncodePriorityCommand:
    """Tests for encode_priority_command in ducaheat_codec."""

    def test_encode_priority_10(self):
        cmd = SetPriority(priority=10)
        result = encode_priority_command(cmd)
        assert result == {"priority": 10}

    def test_encode_priority_zero(self):
        cmd = SetPriority(priority=0)
        result = encode_priority_command(cmd)
        assert result == {"priority": 0}

    def test_encode_priority_30(self):
        cmd = SetPriority(priority=30)
        result = encode_priority_command(cmd)
        assert result == {"priority": 30}


# ---------------------------------------------------------------------------
# 8. PriorityWritePayload validation tests
# ---------------------------------------------------------------------------


class TestPriorityWritePayload:
    """Tests for PriorityWritePayload Pydantic model validation."""

    def test_valid_priority_zero(self):
        payload = PriorityWritePayload.model_validate({"priority": 0})
        assert payload.priority == 0

    def test_valid_priority_30(self):
        payload = PriorityWritePayload.model_validate({"priority": 30})
        assert payload.priority == 30

    def test_valid_priority_15(self):
        payload = PriorityWritePayload.model_validate({"priority": 15})
        assert payload.priority == 15

    def test_rejects_negative_priority(self):
        with pytest.raises(Exception):
            PriorityWritePayload.model_validate({"priority": -1})

    def test_rejects_priority_above_30(self):
        with pytest.raises(Exception):
            PriorityWritePayload.model_validate({"priority": 31})

    def test_model_dump(self):
        payload = PriorityWritePayload.model_validate({"priority": 12})
        assert payload.model_dump() == {"priority": 12}


# ---------------------------------------------------------------------------
# 9. infer_status_endpoint with SetPriority
# ---------------------------------------------------------------------------


class TestInferStatusEndpointPriority:
    """Tests that infer_status_endpoint routes SetPriority to setup."""

    def test_set_priority_routes_to_setup(self):
        cmd = SetPriority(priority=5)
        assert infer_status_endpoint(NodeType.HEATER, cmd) == "setup"

    def test_set_priority_routes_to_setup_for_accumulator(self):
        cmd = SetPriority(priority=10)
        assert infer_status_endpoint(NodeType.ACCUMULATOR, cmd) == "setup"


# ---------------------------------------------------------------------------
# 10. _build_write_call with SetPriority
# ---------------------------------------------------------------------------


class TestBuildWriteCallPriority:
    """Tests for _build_write_call routing SetPriority to the setup endpoint."""

    def test_set_priority_produces_setup_path(self):
        node_id = NodeId(NodeType.HEATER, "H1")
        cmd = SetPriority(priority=7)
        call = _build_write_call(
            base_path="/api/v2/devs/dev123/htr/H1",
            node_id=node_id,
            command=cmd,
            units=None,
        )
        assert isinstance(call, PlannedHttpCall)
        assert call.method == "POST"
        assert call.path == "/api/v2/devs/dev123/htr/H1/setup"
        assert call.json == {"priority": 7}

    def test_plan_command_returns_single_call(self):
        node_id = NodeId(NodeType.HEATER, "H1")
        cmd = SetPriority(priority=15)
        calls = plan_command("dev123", node_id, cmd)
        assert len(calls) == 1
        assert calls[0].path.endswith("/setup")
        assert calls[0].json == {"priority": 15}


# ---------------------------------------------------------------------------
# 11. HeaterPriorityNumber entity tests
# ---------------------------------------------------------------------------


class TestHeaterPriorityNumber:
    """Tests for the HeaterPriorityNumber entity."""

    def test_native_value_returns_int(self, heater_inventory):
        coordinator = _make_heater_coordinator(
            heater_inventory,
            payload={"htr": {"H1": {"priority": 15}}},
        )
        entity = HeaterPriorityNumber(
            coordinator,
            "entry-htr",
            "dev-htr",
            "H1",
            "uid-htr-priority",
            device_name="Heater H1",
            node_type="htr",
            inventory=heater_inventory,
        )
        assert entity.native_value == 15

    def test_native_value_returns_zero(self, heater_inventory):
        coordinator = _make_heater_coordinator(
            heater_inventory,
            payload={"htr": {"H1": {"priority": 0}}},
        )
        entity = HeaterPriorityNumber(
            coordinator,
            "entry-htr",
            "dev-htr",
            "H1",
            "uid-htr-priority-zero",
            device_name="Heater H1",
            node_type="htr",
            inventory=heater_inventory,
        )
        assert entity.native_value == 0

    def test_native_value_returns_none_when_absent(self, heater_inventory):
        coordinator = _make_heater_coordinator(
            heater_inventory,
            payload={"htr": {"H1": {"mode": "on"}}},
        )
        entity = HeaterPriorityNumber(
            coordinator,
            "entry-htr",
            "dev-htr",
            "H1",
            "uid-htr-priority-absent",
            device_name="Heater H1",
            node_type="htr",
            inventory=heater_inventory,
        )
        assert entity.native_value is None

    def test_native_value_returns_none_when_no_state(self, heater_inventory):
        coordinator = _make_heater_coordinator(
            heater_inventory,
            payload={"htr": {}},
        )
        entity = HeaterPriorityNumber(
            coordinator,
            "entry-htr",
            "dev-htr",
            "H1",
            "uid-htr-priority-no-state",
            device_name="Heater H1",
            node_type="htr",
            inventory=heater_inventory,
        )
        assert entity.native_value is None

    def test_translation_key(self, heater_inventory):
        coordinator = _make_heater_coordinator(heater_inventory)
        entity = HeaterPriorityNumber(
            coordinator,
            "entry-htr",
            "dev-htr",
            "H1",
            "uid-htr-priority-tkey",
            device_name="Heater H1",
            node_type="htr",
            inventory=heater_inventory,
        )
        assert entity._attr_translation_key == "heater_priority"

    def test_entity_category_is_config(self, heater_inventory):
        coordinator = _make_heater_coordinator(heater_inventory)
        entity = HeaterPriorityNumber(
            coordinator,
            "entry-htr",
            "dev-htr",
            "H1",
            "uid-htr-priority-cat",
            device_name="Heater H1",
            node_type="htr",
            inventory=heater_inventory,
        )
        assert entity._attr_entity_category == EntityCategory.CONFIG

    def test_icon_is_priority_high(self, heater_inventory):
        coordinator = _make_heater_coordinator(heater_inventory)
        entity = HeaterPriorityNumber(
            coordinator,
            "entry-htr",
            "dev-htr",
            "H1",
            "uid-htr-priority-icon",
            device_name="Heater H1",
            node_type="htr",
            inventory=heater_inventory,
        )
        assert entity._attr_icon == "mdi:priority-high"

    def test_min_max_step(self, heater_inventory):
        coordinator = _make_heater_coordinator(heater_inventory)
        entity = HeaterPriorityNumber(
            coordinator,
            "entry-htr",
            "dev-htr",
            "H1",
            "uid-htr-priority-range",
            device_name="Heater H1",
            node_type="htr",
            inventory=heater_inventory,
        )
        assert entity._attr_native_min_value == 0
        assert entity._attr_native_max_value == 30
        assert entity._attr_native_step == 1

    @pytest.mark.asyncio
    async def test_async_set_native_value_calls_backend(self, heater_inventory):
        coordinator = _make_heater_coordinator(
            heater_inventory,
            payload={"htr": {"H1": {"priority": 5}}},
        )
        entity = HeaterPriorityNumber(
            coordinator,
            "entry-htr",
            "dev-htr",
            "H1",
            "uid-htr-priority-write",
            device_name="Heater H1",
            node_type="htr",
            inventory=heater_inventory,
        )

        mock_backend = AsyncMock()
        mock_backend.set_node_priority = AsyncMock()

        from custom_components.termoweb.runtime import EntryRuntime

        runtime = EntryRuntime.__new__(EntryRuntime)
        runtime.backend = mock_backend

        from custom_components.termoweb.const import DOMAIN

        hass = coordinator.hass
        if not hasattr(hass, "data"):
            hass.data = {}
        if DOMAIN not in hass.data:
            hass.data[DOMAIN] = {}
        hass.data[DOMAIN]["entry-htr"] = runtime

        entity.hass = hass

        await entity.async_set_native_value(10.0)

        mock_backend.set_node_priority.assert_awaited_once_with(
            "dev-htr",
            ("htr", "H1"),
            priority=10,
        )
        coordinator.async_request_refresh.assert_awaited()

    @pytest.mark.asyncio
    async def test_async_set_native_value_rejects_out_of_range(self, heater_inventory):
        coordinator = _make_heater_coordinator(heater_inventory)
        entity = HeaterPriorityNumber(
            coordinator,
            "entry-htr",
            "dev-htr",
            "H1",
            "uid-htr-priority-reject",
            device_name="Heater H1",
            node_type="htr",
            inventory=heater_inventory,
        )

        with pytest.raises(ValueError, match="Priority must be 0-30"):
            await entity.async_set_native_value(31.0)

    @pytest.mark.asyncio
    async def test_async_set_native_value_rejects_negative(self, heater_inventory):
        coordinator = _make_heater_coordinator(heater_inventory)
        entity = HeaterPriorityNumber(
            coordinator,
            "entry-htr",
            "dev-htr",
            "H1",
            "uid-htr-priority-neg",
            device_name="Heater H1",
            node_type="htr",
            inventory=heater_inventory,
        )

        with pytest.raises(ValueError, match="Priority must be 0-30"):
            await entity.async_set_native_value(-1.0)
