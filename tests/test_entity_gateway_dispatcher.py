"""Tests for the GatewayDispatcherEntity mixin."""

from __future__ import annotations

import asyncio

import pytest
from homeassistant.core import HomeAssistant

from custom_components.termoweb.entity import GatewayDispatcherEntity


class _BaseEntity:
    """Minimal stub mimicking Home Assistant's Entity base class."""

    def __init__(self) -> None:
        """Initialise the stub without an assigned hass instance."""

        self.hass = None

    async def async_added_to_hass(self) -> None:
        """Simulate Home Assistant's entity lifecycle hook."""

        return None

    async def async_will_remove_from_hass(self) -> None:
        """Stub the removal hook invoked prior to entity cleanup."""

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


def test_async_added_to_hass_returns_without_hass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_async_added_to_hass_subscribes_when_hass_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dispatcher helper should subscribe once hass is assigned."""

    hass = HomeAssistant()
    entity = _TestEntity()
    entity.hass = hass

    captured: dict[str, object] = {}

    def _subscribe(hass_arg: HomeAssistant, signal: str, handler: object) -> None:
        captured["hass"] = hass_arg
        captured["signal"] = signal
        captured["handler"] = handler

    monkeypatch.setattr(entity._gateway_dispatcher, "subscribe", _subscribe)

    asyncio.run(entity.async_added_to_hass())

    assert captured["hass"] is hass
    assert captured["signal"] == entity.gateway_signal
    handler = captured["handler"]
    assert getattr(handler, "__self__", None) is entity
    assert getattr(handler, "__func__", None) is entity.gateway_signal_handler.__func__


def test_async_will_remove_from_hass_unsubscribes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`async_will_remove_from_hass` should always unsubscribe the dispatcher."""

    hass = HomeAssistant()
    entity = _TestEntity()
    entity.hass = hass

    unsubscribe_called = False
    base_called = False

    async def _base_will_remove(self: _BaseEntity) -> None:
        nonlocal base_called
        base_called = True

    def _unsubscribe() -> None:
        nonlocal unsubscribe_called
        unsubscribe_called = True

    monkeypatch.setattr(_BaseEntity, "async_will_remove_from_hass", _base_will_remove)
    monkeypatch.setattr(entity._gateway_dispatcher, "unsubscribe", _unsubscribe)

    asyncio.run(entity.async_will_remove_from_hass())

    assert unsubscribe_called
    assert base_called
