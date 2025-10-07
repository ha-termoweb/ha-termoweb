"""Backend factory."""

from __future__ import annotations

from ..const import BRAND_DUCAHEAT
from .base import Backend, HttpClientProto


def create_backend(*, brand: str, client: HttpClientProto) -> Backend:
    """Create a backend for the given brand."""

    if brand == BRAND_DUCAHEAT:
        from .ducaheat import DucaheatBackend

        return DucaheatBackend(brand=brand, client=client)
    from .termoweb import TermoWebBackend

    return TermoWebBackend(brand=brand, client=client)
