"""Compatibility wrapper exposing the Engine.IO/Socket.IO v2 client."""

from __future__ import annotations

from typing import Any

from homeassistant.helpers.dispatcher import async_dispatcher_send

from .const import API_BASE, DOMAIN, signal_ws_data, signal_ws_status
from .ws_client import HandshakeError, TermoWebSocketClient, WSStats

__all__ = [
    "API_BASE",
    "DOMAIN",
    "DucaheatWSClient",
    "HandshakeError",
    "WSStats",
    "async_dispatcher_send",
    "signal_ws_data",
    "signal_ws_status",
]


class DucaheatWSClient(TermoWebSocketClient):
    """Engine.IO/Socket.IO v2 client wrapper."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialise the compatibility client."""
        super().__init__(*args, protocol="engineio2", **kwargs)
