"""Tests for the `gateway_signal_handler` property."""

from __future__ import annotations

from custom_components.termoweb.entity import GatewayDispatcherEntity


class _BaseEntity:
    """Minimal stub mimicking Home Assistant's Entity base class."""

    def __init__(self) -> None:
        """Initialise the stub without an assigned hass instance."""

        self.hass = None


class _TestEntity(GatewayDispatcherEntity, _BaseEntity):
    """Concrete implementation of GatewayDispatcherEntity for testing."""

    def __init__(self) -> None:
        """Initialise the dispatcher mixin and base stub."""

        _BaseEntity.__init__(self)
        GatewayDispatcherEntity.__init__(self)
        self.invocations: list[dict[str, object]] = []

    @property
    def gateway_signal(self) -> str:
        """Return a fake dispatcher signal name."""

        return "termoweb-test-signal"

    def _handle_gateway_dispatcher(self, payload: dict[str, object]) -> None:
        """Record the payload for assertion in tests."""

        self.invocations.append(payload)


def test_gateway_signal_handler_returns_bound_method() -> None:
    """`gateway_signal_handler` should expose the bound dispatcher handler."""

    entity = _TestEntity()
    handler = entity.gateway_signal_handler

    assert handler.__self__ is entity
    assert handler.__func__ is entity._handle_gateway_dispatcher.__func__

    payload: dict[str, object] = {"example": 1}
    handler(payload)
    assert entity.invocations == [payload]
