"""Backend factory."""

from __future__ import annotations

from ..const import BRAND_DUCAHEAT
from .base import Backend, HttpClientProto
from .ducaheat import DucaheatBackend
from .termoweb import TermoWebBackend


def create_backend(*, brand: str, client: HttpClientProto) -> Backend:
    """Create a backend for the given brand."""

    if brand == BRAND_DUCAHEAT:
        return DucaheatBackend(brand=brand, client=client)
    return TermoWebBackend(brand=brand, client=client)
