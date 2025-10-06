"""TermoWeb backend implementation."""
from __future__ import annotations

from importlib import import_module
import sys
from types import ModuleType
from typing import Any

from .ws_client import WebSocketClient

from .base import Backend, WsClientProto


class TermoWebBackend(Backend):
    """Backend for the TermoWeb brand."""

    def _resolve_ws_client_cls(self) -> type[Any]:
        """Return the websocket client class compatible with this backend."""

        saw_any_module = False
        saw_real_module = False
        for module_name in (
            "custom_components.termoweb.__init__",
            "custom_components.termoweb",
        ):
            module = sys.modules.get(module_name)
            if module is None:
                continue
            saw_any_module = True
            if isinstance(module, ModuleType):
                saw_real_module = True
            ws_cls = getattr(module, "TermoWebWSClient", None)
            if isinstance(ws_cls, type):
                return ws_cls
        if not saw_any_module:
            return WebSocketClient
        if not saw_real_module:
            return WebSocketClient
        try:
            ws_module = import_module("custom_components.termoweb.backend.termoweb_ws")
        except ImportError:
            return WebSocketClient
        ws_cls = getattr(ws_module, "TermoWebWSClient", None)
        if isinstance(ws_cls, type):
            return ws_cls
        return WebSocketClient

    def create_ws_client(
        self,
        hass: Any,
        entry_id: str,
        dev_id: str,
        coordinator: Any,
    ) -> WsClientProto:
        """Instantiate the unified websocket client for TermoWeb."""

        ws_cls = self._resolve_ws_client_cls()
        kwargs: dict[str, Any] = {
            "entry_id": entry_id,
            "dev_id": dev_id,
            "api_client": self.client,
            "coordinator": coordinator,
        }
        if issubclass(ws_cls, WebSocketClient):
            kwargs["protocol"] = "socketio09"
        return ws_cls(
            hass,
            **kwargs,
        )
