"""Backend factory."""
from __future__ import annotations

from .base import Backend, HttpClientProto
from .termoweb import TermoWebBackend


def create_backend(*, brand: str, client: HttpClientProto) -> Backend:
    """Create a backend for the given brand."""

    return TermoWebBackend(brand=brand, client=client)
