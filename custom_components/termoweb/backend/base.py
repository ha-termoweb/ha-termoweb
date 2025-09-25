"""Backend abstractions for brand-specific behavior."""
from __future__ import annotations

from abc import ABC, abstractmethod
from asyncio import Task
from typing import Any, Protocol


class HttpClientProto(Protocol):
    """Protocol for the HTTP client used by TermoWeb entities."""

    async def list_devices(self) -> list[dict[str, Any]]:
        """Return the list of devices associated with the account."""

    async def get_nodes(self, dev_id: str) -> Any:
        """Return the node description for the given device."""

    async def get_htr_settings(self, dev_id: str, addr: str | int) -> Any:
        """Return heater settings for the specified node."""

    async def set_htr_settings(
        self,
        dev_id: str,
        addr: str | int,
        *,
        mode: str | None = None,
        stemp: float | None = None,
        prog: list[int] | None = None,
        ptemp: list[float] | None = None,
        units: str = "C",
    ) -> Any:
        """Update heater settings for the specified node."""

    async def get_htr_samples(
        self,
        dev_id: str,
        addr: str | int,
        start: float,
        stop: float,
    ) -> list[dict[str, str | int]]:
        """Return historical heater samples for the specified node."""


class WsClientProto(Protocol):
    """Protocol for websocket clients used by the integration."""

    def start(self) -> Task[Any]:
        """Start the websocket client."""

    async def stop(self) -> None:
        """Stop the websocket client."""


class Backend(ABC):
    """Base class for brand-specific integration backends."""

    def __init__(self, *, brand: str, client: HttpClientProto) -> None:
        """Initialize backend metadata."""

        self._brand = brand
        self._client = client

    @property
    def brand(self) -> str:
        """Return the configured brand."""

        return self._brand

    @property
    def client(self) -> HttpClientProto:
        """Return the HTTP client associated with this backend."""

        return self._client

    @abstractmethod
    def create_ws_client(
        self,
        hass: Any,
        entry_id: str,
        dev_id: str,
        coordinator: Any,
    ) -> WsClientProto:
        """Create a websocket client for the given device."""
