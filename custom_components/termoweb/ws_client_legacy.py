"""Compatibility shims exposing the unified websocket client."""

from __future__ import annotations

import asyncio
import random
import time

import aiohttp
from homeassistant.helpers.dispatcher import async_dispatcher_send

from .const import API_BASE, DOMAIN, WS_NAMESPACE, signal_ws_data, signal_ws_status
from .ws_client import HandshakeError, TermoWebSocketClient, WSStats

WebSocket09Client = TermoWebSocketClient

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
