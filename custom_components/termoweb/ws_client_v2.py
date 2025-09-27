"""Compatibility shims exposing the unified websocket client for Ducaheat."""

from __future__ import annotations

import asyncio

from homeassistant.helpers.dispatcher import async_dispatcher_send

from .const import API_BASE, DOMAIN, signal_ws_data, signal_ws_status
from .ws_client import HandshakeError, TermoWebSocketClient, WSStats

DucaheatWSClient = TermoWebSocketClient

__all__ = [
    "API_BASE",
    "DOMAIN",
    "DucaheatWSClient",
    "HandshakeError",
    "WSStats",
    "async_dispatcher_send",
    "asyncio",
    "signal_ws_data",
    "signal_ws_status",
]
