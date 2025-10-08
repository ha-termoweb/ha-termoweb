"""Entity mixins shared across TermoWeb platforms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from homeassistant.helpers.entity import Entity

from .heater import DispatcherSubscriptionHelper


class GatewaySignalHandler(Protocol):
    """Protocol describing dispatcher callbacks."""

    def __call__(self, payload: dict[str, Any]) -> None:
        """Invoke the callback with a dispatcher payload."""


class GatewayDispatcherEntity:
    """Mixin standardising dispatcher subscriptions for gateway entities."""

    _gateway_dispatcher: DispatcherSubscriptionHelper

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialise the dispatcher helper and base classes."""

        super().__init__(*args, **kwargs)
        self._gateway_dispatcher = DispatcherSubscriptionHelper(self)

    @property
    def gateway_signal(self) -> str:
        """Return the dispatcher signal name for this entity."""

        raise NotImplementedError  # pragma: no cover - abstract contract

    @property
    def gateway_signal_handler(self) -> GatewaySignalHandler:
        """Return the callback invoked for dispatcher updates."""

        return self._handle_gateway_dispatcher

    async def async_added_to_hass(self) -> None:
        """Subscribe to dispatcher updates when the entity is added."""

        await super().async_added_to_hass()
        if self.hass is None:
            return

        signal = self.gateway_signal
        if not signal:  # pragma: no cover - defensive guard
            return

        self._gateway_dispatcher.subscribe(
            self.hass,
            signal,
            self.gateway_signal_handler,
        )

    async def async_will_remove_from_hass(self) -> None:
        """Unsubscribe from dispatcher updates before removal."""

        self._gateway_dispatcher.unsubscribe()
        await super().async_will_remove_from_hass()

    def _handle_gateway_dispatcher(self, payload: dict[str, Any]) -> None:
        """Handle dispatcher payloads targeting the entity."""

        raise NotImplementedError  # pragma: no cover - abstract contract

