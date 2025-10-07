"""Backend package exports."""
from __future__ import annotations

from typing import Any

from .base import Backend, HttpClientProto, WsClientProto
from .factory import create_backend

__all__ = [
    "Backend",
    "DucaheatBackend",
    "DucaheatRESTClient",
    "TermoWebBackend",
    "HttpClientProto",
    "WsClientProto",
    "create_backend",
]


def __getattr__(name: str) -> Any:
    """Lazily import backend implementations to avoid circular imports."""

    if name in {"DucaheatBackend", "DucaheatRESTClient"}:
        from .ducaheat import DucaheatBackend, DucaheatRESTClient

        return {"DucaheatBackend": DucaheatBackend, "DucaheatRESTClient": DucaheatRESTClient}[name]
    if name == "TermoWebBackend":
        from .termoweb import TermoWebBackend

        return TermoWebBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
