"""Backend package exports."""
from __future__ import annotations

from typing import Any

from .base import Backend, HttpClientProto, WsClientProto
from .factory import create_backend

__all__ = [
    "Backend",
    "DucaheatBackend",
    "DucaheatRESTClient",
    "HttpClientProto",
    "TermoWebBackend",
    "WsClientProto",
    "create_backend",
]


def __getattr__(name: str) -> Any:
    """Lazily import backend implementations to avoid circular imports."""

    if name in {"DucaheatBackend", "DucaheatRESTClient"}:
        from .ducaheat import DucaheatBackend, DucaheatRESTClient

        mapping = {
            "DucaheatBackend": DucaheatBackend,
            "DucaheatRESTClient": DucaheatRESTClient,
        }
        value = mapping[name]
        globals()[name] = value
        return value
    if name == "TermoWebBackend":
        from .termoweb import TermoWebBackend

        globals()[name] = TermoWebBackend
        return TermoWebBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
