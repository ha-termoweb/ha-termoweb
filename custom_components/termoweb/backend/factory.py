"""Backend factory."""

from __future__ import annotations

from .base import Backend, HttpClientProto
from .termoweb import TermoWebBackend


def create_backend(
    *, brand: str, client: HttpClientProto, ws_impl: str | None = None
) -> Backend:
    """Create a backend for the given brand."""

    _ = ws_impl  # placeholder until Ducaheat WS is wired in
    return TermoWebBackend(brand=brand, client=client)
