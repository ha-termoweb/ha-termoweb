"""Tests for the TermoWeb constants module."""

from __future__ import annotations

from custom_components.termoweb import const


def test_get_brand_socketio_path_known_brand() -> None:
    """Return the configured Socket.IO path for known brands."""

    assert const.get_brand_socketio_path(const.BRAND_DUCAHEAT) == "api/v2/socket_io"


def test_get_brand_socketio_path_unknown_brand() -> None:
    """Default to the generic Socket.IO path for unknown brands."""

    assert const.get_brand_socketio_path("unknown") == "socket.io"
