from __future__ import annotations

import asyncio
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from conftest import FakeCoordinator, _install_stubs, build_coordinator_device_state

_install_stubs()

import custom_components.termoweb.select as select_module
from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.heater import (
    BOOST_DURATION_OPTIONS,
    DEFAULT_BOOST_DURATION,
    format_boost_duration_label,
    get_boost_runtime_minutes,
    set_boost_runtime_minutes,
)
from custom_components.termoweb.identifiers import build_heater_entity_unique_id
from homeassistant.core import HomeAssistant

AccumulatorBoostDurationSelect = select_module.AccumulatorBoostDurationSelect
async_setup_entry = select_module.async_setup_entry


def _make_select_entity() -> AccumulatorBoostDurationSelect:
    """Create a selector instance for direct method testing."""

    hass = HomeAssistant()
    coordinator = FakeCoordinator(hass, dev_id="dev-select-test")
    return AccumulatorBoostDurationSelect(
        coordinator,
        "entry-select-test",
        "dev-select-test",
        "01",
        "Accumulator 1",
        "test-uid",
        node_type="acm",
    )




@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (format_boost_duration_label(60), 60),
        (" 2 hours ", 120),
        ("060", 60),
        ("120.0", 120),
        (60, 60),
        (60.0, 60),
        ("600", 600),
    ],
)
def test_option_to_minutes_accepts_valid_values(
    value: object, expected: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure valid inputs resolve to supported boost durations."""

    calls: list[object] = []
    real_coerce = select_module.coerce_boost_minutes

    def fake_coerce(candidate: object) -> int | None:
        calls.append(candidate)
        return real_coerce(candidate)

    monkeypatch.setattr(select_module, "coerce_boost_minutes", fake_coerce)

    entity = _make_select_entity()
    assert entity._option_to_minutes(value) == expected
    if not (isinstance(value, str) and value.strip() in entity._OPTION_MAP):
        assert calls == [value]


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "invalid",
        0,
        -60,
        45,
        "45",
        "11 hours",
        True,
    ],
)
def test_option_to_minutes_rejects_invalid_values(value: object) -> None:
    """Ensure invalid values are rejected by the coercion helper."""

    entity = _make_select_entity()
    assert entity._option_to_minutes(value) is None




