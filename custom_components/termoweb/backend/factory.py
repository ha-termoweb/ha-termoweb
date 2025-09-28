"""Backend factory."""

from __future__ import annotations

from ..const import BRAND_DUCAHEAT
from .base import Backend, HttpClientProto
from .ducaheat import DucaheatBackend
from .termoweb import TermoWebBackend


def create_backend(
    *, brand: str, client: HttpClientProto, ws_impl: str | None = None
) -> Backend:
    """Create a backend for the given brand."""

    _ = ws_impl  # reserved for future websocket selection overrides
    if brand == BRAND_DUCAHEAT:
        return DucaheatBackend(brand=brand, client=client)
    return TermoWebBackend(brand=brand, client=client)
