"""Tests for energy service registration utilities."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from conftest import build_entry_runtime
from custom_components.termoweb import energy
from custom_components.termoweb.services.energy_history import (
    async_register_import_energy_history_service,
)


class _StubServices:
    """Stub Home Assistant service registry for testing."""

    def __init__(self, *, existing: bool = True) -> None:
        self._existing = existing
        self.registrations: list[tuple[str, str, Any]] = []

    def has_service(self, domain: str, service: str) -> bool:
        """Pretend the service is already registered."""

        return self._existing

    def async_register(
        self, domain: str, service: str, handler
    ) -> None:  # pragma: no cover - defensive
        """Record registrations for inspection."""

        self.registrations.append((domain, service, handler))


@pytest.mark.asyncio
async def test_async_register_import_energy_history_service_skips_registration() -> (
    None
):
    """The service registration is skipped when it already exists."""

    hass = SimpleNamespace(services=_StubServices())
    import_fn = AsyncMock()

    await async_register_import_energy_history_service(hass, import_fn)

    import_fn.assert_not_called()
    assert hass.services.registrations == []


@pytest.mark.asyncio
async def test_service_handler_forwards_filters(
    inventory_from_map,
) -> None:
    """Service handler should normalise filter inputs for the importer."""

    services = _StubServices(existing=False)
    hass = SimpleNamespace(
        services=services,
        data={energy.DOMAIN: {}},
    )
    entry = SimpleNamespace(entry_id="entry-filter")
    inventory = inventory_from_map({"htr": ["A"]}, dev_id="dev-filter")
    build_entry_runtime(
        hass=hass,
        entry_id=entry.entry_id,
        dev_id="dev-filter",
        inventory=inventory,
        config_entry=entry,
    )

    import_fn = AsyncMock()

    await async_register_import_energy_history_service(hass, import_fn)

    assert services.registrations
    _, _, handler = services.registrations[0]

    call = SimpleNamespace(
        data={
            "node_types": {"htr", "HTR"},
            "addresses": {"A", 1},
            "day_chunk_hours": "12",
            "max_history_retrieval": 5,
            "reset_progress": True,
        }
    )

    await handler(call)

    import_fn.assert_awaited_once()
    args, kwargs = import_fn.await_args
    assert args == (hass, entry)
    assert kwargs["reset_progress"] is True
    assert kwargs["max_days"] == 5
    assert isinstance(kwargs["node_types"], tuple)
    assert set(kwargs["node_types"]) == {"htr", "HTR"}
    assert isinstance(kwargs["addresses"], tuple)
    assert set(kwargs["addresses"]) == {"A", "1"}
    assert kwargs["day_chunk_hours"] == 12


@pytest.mark.asyncio
async def test_service_logs_rejected_filters(
    inventory_from_map, caplog: pytest.LogCaptureFixture
) -> None:
    """Service handler should log when the importer rejects its filters."""

    services = _StubServices(existing=False)
    hass = SimpleNamespace(
        services=services,
        data={energy.DOMAIN: {}},
    )
    entry = SimpleNamespace(entry_id="entry-invalid")
    inventory = inventory_from_map({"htr": ["A"]}, dev_id="dev-invalid")
    build_entry_runtime(
        hass=hass,
        entry_id=entry.entry_id,
        dev_id="dev-invalid",
        inventory=inventory,
        config_entry=entry,
    )

    import_fn = AsyncMock(side_effect=ValueError("bad filters"))

    await async_register_import_energy_history_service(hass, import_fn)

    assert services.registrations
    _, _, handler = services.registrations[0]

    call = SimpleNamespace(data={"node_types": ["invalid"]})

    with caplog.at_level(logging.ERROR):
        await handler(call)

    import_fn.assert_called_once()
    assert "import_energy_history task failed" in caplog.text
