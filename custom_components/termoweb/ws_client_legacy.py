"""Compatibility wrapper exposing the legacy Socket.IO client."""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any

import aiohttp
from homeassistant.helpers.dispatcher import async_dispatcher_send

from .const import API_BASE, DOMAIN, WS_NAMESPACE, signal_ws_data, signal_ws_status
from .ws_client import HandshakeError, TermoWebSocketClient, WSStats

__all__ = [
    "API_BASE",
    "DOMAIN",
    "WS_NAMESPACE",
    "HandshakeError",
    "WSStats",
    "WebSocket09Client",
    "aiohttp",
    "async_dispatcher_send",
    "asyncio",
    "random",
    "signal_ws_data",
    "signal_ws_status",
    "time",
]


class WebSocket09Client(TermoWebSocketClient):
    """Legacy Socket.IO 0.9 client wrapper."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialise the compatibility client."""
        super().__init__(*args, protocol="socketio09", **kwargs)
