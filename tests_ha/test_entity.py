"""Shared heater-entity behaviour: boost metadata and the boost entities."""

from __future__ import annotations

from collections.abc import Generator
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any

from freezegun.api import FrozenDateTimeFactory
from homeassistant.const import STATE_OFF, STATE_ON, STATE_UNKNOWN
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.termoweb.boost import coerce_boost_minutes
from custom_components.termoweb.entity import derive_boost_state

from .conftest import FakeCloud
from .fakes.entities import FakeNodes, setup_entry

NOW = datetime(2024, 1, 1, 12, 0, tzinfo=dt_util.UTC)
BOOST_ACTIVE = "binary_sensor.store_boost_active"
BOOST_MINUTES = "sensor.store_boost_minutes_remaining"
BOOST_END = "sensor.store_boost_end"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (True, None),
        (False, None),
        (0, None),
        (-5, None),
        ("   ", None),
        ("invalid", None),
        ("90", 90),
        (120.7, 120),
        (45, 45),
    ],
)
def test_coerce_boost_minutes(value: Any, expected: int | None) -> None:
    """Only positive numbers (or numeric strings) are boost durations."""
    assert coerce_boost_minutes(value) == expected


def _resolver(day: Any, minute: Any) -> tuple[datetime | None, int | None]:
    """Resolve a boost end like the coordinator: day 1 + minutes past midnight."""
    end = datetime(2024, 1, day, 0, 0, tzinfo=dt_util.UTC) + timedelta(minutes=minute)
    return end, int((end - NOW).total_seconds() // 60)


def _failing_resolver(day: Any, minute: Any) -> tuple[datetime | None, int | None]:
    """Fail like a coordinator without a device clock estimate."""
    raise ValueError("no device clock")


@pytest.mark.parametrize(
    ("settings", "resolver", "active", "minutes", "end", "label"),
    [
        pytest.param({}, None, False, None, None, "Never", id="nothing"),
        pytest.param({"mode": "boost"}, None, True, None, None, None, id="mode"),
        pytest.param(
            {"boost_active": "off", "mode": "boost"},
            None,
            False,
            None,
            None,
            "Never",
            id="flag-wins-over-mode",
        ),
        pytest.param(
            {"boost_active": True, "boost_remaining": 30},
            None,
            True,
            30,
            NOW + timedelta(minutes=30),
            None,
            id="remaining",
        ),
        pytest.param(
            {"boost_active": 1, "boost_end_day": 1, "boost_end_min": 14 * 60},
            _resolver,
            True,
            120,
            NOW + timedelta(hours=2),
            None,
            id="day-minute",
        ),
        pytest.param(
            {
                "boost_active": 1,
                "boost_end_day": 1,
                "boost_end_min": 0,
                "boost_remaining": 15,
            },
            _failing_resolver,
            True,
            15,
            NOW + timedelta(minutes=15),
            None,
            id="resolver-fails",
        ),
        pytest.param(
            {"boost_active": True, "boost_end_datetime": "2024-01-01T13:00:00+00:00"},
            None,
            True,
            60,
            NOW + timedelta(hours=1),
            None,
            id="iso-end",
        ),
        pytest.param(
            {"boost_active": False, "boost_end_datetime": "1970-01-01T00:00:00+00:00"},
            None,
            False,
            None,
            None,
            "Never",
            id="epoch-placeholder",
        ),
        pytest.param(
            {"boost_active": True, "boost_remaining": 0},
            None,
            True,
            None,
            None,
            None,
            id="zero-remaining",
        ),
    ],
)
def test_derive_boost_state(
    freezer: FrozenDateTimeFactory,
    settings: dict[str, Any],
    resolver: Any,
    active: bool,
    minutes: int | None,
    end: datetime | None,
    label: str | None,
) -> None:
    """Boost metadata is derived from the flag, remaining minutes or end fields."""
    freezer.move_to(NOW)
    coordinator = SimpleNamespace(resolve_boost_end=resolver)

    state = derive_boost_state(settings, coordinator)

    assert state.active is active
    assert state.minutes_remaining == minutes
    assert state.end_datetime == end
    assert state.end_iso == (end.isoformat() if end else None)
    assert state.end_label == label


@pytest.fixture
def store(cloud: FakeCloud) -> Generator[FakeNodes]:
    """Serve one accumulator, "Store" (addr 3), behind the fake cloud."""
    fake = FakeNodes(
        cloud,
        {
            ("acm", "3"): (
                "Store",
                {"mode": "auto", "state": "off", "stemp": "19.0", "units": "C"},
            )
        },
    )
    with fake.patched():
        yield fake


async def test_boost_entities_show_a_running_boost(
    hass: HomeAssistant,
    store: FakeNodes,
    config_entry: MockConfigEntry,
    freezer: FrozenDateTimeFactory,
) -> None:
    """The boost binary sensor and sensors report a boost with 30 minutes left."""
    freezer.move_to(NOW)
    store.settings[("acm", "3")].update(
        mode="boost", boost_active=True, boost_remaining=30
    )
    await setup_entry(hass, config_entry)

    active = hass.states.get(BOOST_ACTIVE)
    assert active.state == STATE_ON
    assert active.attributes["boost_minutes_remaining"] == 30
    assert hass.states.get(BOOST_MINUTES).state == "30"
    end = hass.states.get(BOOST_END)
    assert dt_util.parse_datetime(end.state) == NOW + timedelta(minutes=30)
    assert end.attributes["boost_end_label"] is None


async def test_boost_entities_without_a_boost(
    hass: HomeAssistant, store: FakeNodes, config_entry: MockConfigEntry
) -> None:
    """Without a boost the binary sensor is off and nothing is left to run."""
    await setup_entry(hass, config_entry)

    assert hass.states.get(BOOST_ACTIVE).state == STATE_OFF
    assert hass.states.get(BOOST_MINUTES).state == STATE_UNKNOWN
    # The end sensor's state is review finding B14 (text "Never" in a timestamp
    # sensor); only its label attribute is a settled contract.
    assert hass.states.get(BOOST_END).attributes["boost_end_label"] == "Never"
