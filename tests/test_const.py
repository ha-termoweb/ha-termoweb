"""Tests for the TermoWeb constants module."""

from __future__ import annotations

from custom_components.termoweb import const


def test_get_brand_socketio_path_known_brand() -> None:
    """Return the configured Socket.IO path for known brands."""

    assert const.get_brand_socketio_path(const.BRAND_DUCAHEAT) == "api/v2/socket_io"


def test_get_brand_socketio_path_unknown_brand() -> None:
    """Default to the generic Socket.IO path for unknown brands."""

    assert const.get_brand_socketio_path("unknown") == "socket.io"


def test_get_brand_socketio_path_tevolve() -> None:
    """Tevolve reuses the Ducaheat Socket.IO path."""

    assert const.get_brand_socketio_path(const.BRAND_TEVOLVE) == "api/v2/socket_io"


def test_uses_ducaheat_backend_aliases() -> None:
    """Brands mapped to Ducaheat share backend selection."""

    assert const.uses_ducaheat_backend(const.BRAND_DUCAHEAT) is True
    assert const.uses_ducaheat_backend(const.BRAND_TEVOLVE) is True
    assert const.uses_ducaheat_backend(const.BRAND_TERMOWEB) is False
