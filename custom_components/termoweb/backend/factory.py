"""Backend factory."""

from __future__ import annotations

from custom_components.termoweb.const import uses_ducaheat_backend

from .base import Backend, HttpClientProto


def create_backend(*, brand: str, client: HttpClientProto) -> Backend:
    """Create a backend for the given brand."""

    if uses_ducaheat_backend(brand):
        from . import DucaheatBackend  # noqa: PLC0415

        return DucaheatBackend(brand=brand, client=client)

    from . import TermoWebBackend  # noqa: PLC0415

    return TermoWebBackend(brand=brand, client=client)
