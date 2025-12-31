from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from homeassistant.core import HomeAssistant

from custom_components.termoweb import coordinator as coord_module


@pytest.mark.asyncio
async def test_handle_instant_power_update_records_ws(
    inventory_builder,
) -> None:
    """Websocket updates should populate the instant power cache."""

    hass = HomeAssistant()
    inventory = inventory_builder("dev", {"nodes": []})
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device={},
        nodes=None,
        inventory=inventory,
    )

    coordinator.handle_instant_power_update(
        "dev",
        "htr",
        "01",
        47.0,
        timestamp=1_700_000_000.0,
    )

    entry = coordinator.instant_power_entry("htr", "01")
    assert entry is not None
    assert entry.watts == 47.0
    assert entry.timestamp == 1_700_000_000.0
    assert entry.source == "ws"

    overview = coordinator.instant_power_overview()
    assert overview["htr"]["01"]["source"] == "ws"


@pytest.mark.asyncio
async def test_handle_instant_power_update_rejects_invalid(
    inventory_builder,
) -> None:
    """Invalid updates should not modify the instant power cache."""

    hass = HomeAssistant()
    inventory = inventory_builder("dev", {"nodes": []})
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device={},
        nodes=None,
        inventory=inventory,
    )

    coordinator.handle_instant_power_update("other", "htr", "01", 12)
    coordinator.handle_instant_power_update("dev", "htr", "01", -5)
    coordinator.handle_instant_power_update("dev", "htr", "01", float("nan"))

    assert coordinator.instant_power_entry("htr", "01") is None


@pytest.mark.asyncio
async def test_rest_updates_respect_ws_priority(
    inventory_builder,
) -> None:
    """Websocket readings should not be overwritten by older REST data."""

    hass = HomeAssistant()
    inventory = inventory_builder("dev", {"nodes": []})
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device={},
        nodes=None,
        inventory=inventory,
    )

    coordinator.handle_instant_power_update(
        "dev",
        "htr",
        "01",
        10,
        timestamp=1_700_000_000.0,
    )

    assert not coordinator._record_instant_power(  # noqa: SLF001
        "htr",
        "01",
        20,
        timestamp=1_699_999_990.0,
        source="rest",
    )
    entry = coordinator.instant_power_entry("htr", "01")
    assert entry is not None
    assert entry.watts == 10
    assert entry.source == "ws"

    assert coordinator._record_instant_power(  # noqa: SLF001
        "htr",
        "01",
        25,
        timestamp=1_700_000_100.0,
        source="rest",
    )

    entry = coordinator.instant_power_entry("htr", "01")
    assert entry is not None
    assert entry.watts == 25
    assert entry.source == "rest"


@pytest.mark.asyncio
async def test_should_skip_rest_power(monkeypatch, inventory_builder) -> None:
    """Recent websocket data should suppress redundant REST polling."""

    hass = HomeAssistant()
    inventory = inventory_builder("dev", {"nodes": []})
    coordinator = coord_module.StateCoordinator(
        hass,
        client=AsyncMock(),
        base_interval=30,
        dev_id="dev",
        device={},
        nodes=None,
        inventory=inventory,
    )

    base_time = 1_700_000_000.0
    monkeypatch.setattr(coord_module.time, "time", lambda: base_time)

    coordinator.handle_instant_power_update(
        "dev",
        "htr",
        "01",
        30,
        timestamp=base_time,
    )

    interval = coordinator.update_interval.total_seconds()
    monkeypatch.setattr(
        coord_module.time,
        "time",
        lambda: base_time + interval / 2,
    )
    assert coordinator._should_skip_rest_power("htr", "01")  # noqa: SLF001

    monkeypatch.setattr(
        coord_module.time,
        "time",
        lambda: base_time + interval * 2,
    )
    assert not coordinator._should_skip_rest_power("htr", "01")  # noqa: SLF001
