"""Tests for the GatewayDispatcherEntity mixin."""

from __future__ import annotations

import asyncio

import pytest

from custom_components.termoweb.entity import GatewayDispatcherEntity


class _BaseEntity:
    """Minimal stub mimicking Home Assistant's Entity base class."""

    def __init__(self) -> None:
        """Initialise the stub without an assigned hass instance."""

        self.hass = None

    async def async_added_to_hass(self) -> None:
        """Simulate Home Assistant's entity lifecycle hook."""

        return None


class _TestEntity(GatewayDispatcherEntity, _BaseEntity):
    """Concrete implementation of GatewayDispatcherEntity for testing."""

    def __init__(self) -> None:
        """Initialise the dispatcher mixin and base stub."""

        _BaseEntity.__init__(self)
        GatewayDispatcherEntity.__init__(self)

    @property
    def gateway_signal(self) -> str:
        """Return a fake dispatcher signal name."""

        return "termoweb-test-signal"

    def _handle_gateway_dispatcher(self, payload: dict[str, object]) -> None:
        """Stubbed dispatcher handler for interface compliance."""

        return None


def test_async_added_to_hass_returns_without_hass(monkeypatch: pytest.MonkeyPatch) -> None:
    """`async_added_to_hass` should exit early when hass is not initialised."""

    entity = _TestEntity()
    subscribe_called = False

    def _subscribe(*args: object, **kwargs: object) -> None:
        nonlocal subscribe_called
        subscribe_called = True
        raise AssertionError("subscribe should not be called when hass is missing")

    monkeypatch.setattr(entity._gateway_dispatcher, "subscribe", _subscribe)

    asyncio.run(entity.async_added_to_hass())

    assert not subscribe_called
