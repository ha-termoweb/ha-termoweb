"""Tests for energy service registration utilities."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from custom_components.termoweb.energy import (
    async_register_import_energy_history_service,
)


class _StubServices:
    """Stub Home Assistant service registry for testing."""

    def __init__(self) -> None:
        self.registrations: list[tuple[str, str]] = []

    def has_service(self, domain: str, service: str) -> bool:
        """Pretend the service is already registered."""

        return True

    def async_register(self, domain: str, service: str, handler) -> None:  # pragma: no cover - defensive
        """Record registrations for inspection."""

        self.registrations.append((domain, service))


@pytest.mark.asyncio
async def test_async_register_import_energy_history_service_skips_registration() -> None:
    """The service registration is skipped when it already exists."""

    hass = SimpleNamespace(services=_StubServices())
    import_fn = AsyncMock()

    await async_register_import_energy_history_service(hass, import_fn)

    import_fn.assert_not_called()
    assert hass.services.registrations == []
