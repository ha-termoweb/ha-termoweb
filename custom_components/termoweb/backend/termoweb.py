"""TermoWeb backend implementation."""
from __future__ import annotations

from importlib import import_module
from typing import Any

from .base import Backend, WsClientProto


class TermoWebBackend(Backend):
    """Backend for the TermoWeb brand."""

    def _resolve_ws_client_cls(self) -> type[Any]:
        module = import_module("custom_components.termoweb.__init__")
        ws_cls = getattr(module, "WebSocket09Client", None)
        if ws_cls is None:
            legacy_module = import_module("custom_components.termoweb.ws_client_legacy")
            ws_cls = legacy_module.WebSocket09Client
        return ws_cls

    def create_ws_client(
        self,
        hass: Any,
        entry_id: str,
        dev_id: str,
        coordinator: Any,
    ) -> WsClientProto:
        """Instantiate the legacy websocket client used by TermoWeb."""

        ws_cls = self._resolve_ws_client_cls()
        return ws_cls(
            hass,
            entry_id=entry_id,
            dev_id=dev_id,
            api_client=self.client,
            coordinator=coordinator,
        )
