"""TermoWeb backend implementation."""
from __future__ import annotations

from typing import Any, cast

from ..inventory import Inventory
from .base import Backend, WsClientProto
from .ws_client import WebSocketClient

try:  # pragma: no cover - exercised via backend tests
    from custom_components.termoweb.backend.termoweb_ws import (
        TermoWebWSClient as _TermoWebWSClient,
    )
except ImportError:  # pragma: no cover - exercised via backend tests
    _TermoWebWSClient = cast(type[Any], WebSocketClient)

TermoWebWSClient = _TermoWebWSClient


class TermoWebBackend(Backend):
    """Backend for the TermoWeb brand."""

    def _resolve_ws_client_cls(self) -> type[Any]:
        """Return the websocket client class compatible with this backend."""

        if isinstance(TermoWebWSClient, type):
            return TermoWebWSClient
        return WebSocketClient

    def create_ws_client(
        self,
        hass: Any,
        entry_id: str,
        dev_id: str,
        coordinator: Any,
        *,
        inventory: Inventory | None = None,
    ) -> WsClientProto:
        """Instantiate the unified websocket client for TermoWeb."""

        ws_cls = self._resolve_ws_client_cls()
        kwargs: dict[str, Any] = {
            "entry_id": entry_id,
            "dev_id": dev_id,
            "api_client": self.client,
            "coordinator": coordinator,
            "inventory": inventory,
        }
        if issubclass(ws_cls, WebSocketClient):
            kwargs["protocol"] = "socketio09"
        return ws_cls(
            hass,
            **kwargs,
        )
