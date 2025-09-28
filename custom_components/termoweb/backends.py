"""Backend factory helpers for selecting websocket clients."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from homeassistant.core import HomeAssistant

from .const import BRAND_DUCAHEAT, BRAND_TERMOWEB


class BaseBackend:
    """Backend factory responsible for websocket client selection."""

    def __init__(
        self,
        hass: HomeAssistant,
        *,
        entry_id: str,
        api_client,
        coordinator,
        ws_client_factory: Callable[..., Any],
    ) -> None:
        """Store shared dependencies used by backend implementations."""
        self.hass = hass
        self.entry_id = entry_id
        self._api_client = api_client
        self._coordinator = coordinator
        self._ws_client_factory = ws_client_factory

    def create_ws_client(self, dev_id: str):
        """Return a websocket client for the given device id."""
        raise NotImplementedError


class TermowebBackend(BaseBackend):
    """Backend for the legacy TermoWeb deployment."""

    def create_ws_client(self, dev_id: str):
        """Create the legacy websocket client for TermoWeb deployments."""
        return self._ws_client_factory(
            self.hass,
            entry_id=self.entry_id,
            dev_id=dev_id,
            api_client=self._api_client,
            coordinator=self._coordinator,
        )


class DucaheatBackend(BaseBackend):
    """Backend for the Ducaheat/Tevolve deployment."""

    def create_ws_client(self, dev_id: str):
        """Create the Socket.IO v2 websocket client for Ducaheat deployments."""
        return self._ws_client_factory(
            self.hass,
            entry_id=self.entry_id,
            dev_id=dev_id,
            api_client=self._api_client,
            coordinator=self._coordinator,
        )


_BACKEND_MAP: dict[str, type[BaseBackend]] = {
    BRAND_TERMOWEB: TermowebBackend,
    BRAND_DUCAHEAT: DucaheatBackend,
}


def get_backend_for_brand(brand: str) -> type[BaseBackend]:
    """Return the backend implementation for the configured brand."""

    return _BACKEND_MAP.get(brand, TermowebBackend)
