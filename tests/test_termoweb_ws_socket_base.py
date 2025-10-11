"""Tests for TermoWeb websocket base URL calculation."""

from types import SimpleNamespace

import pytest

from custom_components.termoweb.backend import termoweb_ws
from custom_components.termoweb.backend.termoweb_ws import TermoWebWSClient
from custom_components.termoweb.const import API_BASE


@pytest.mark.parametrize(
    ("api_base", "expected"),
    [
        ("https://example.com/api", "https://example.com/api"),
        ("https://example.com/api/", "https://example.com/api"),
        ("example.com", "https://example.com"),
        ("example.com/api", "https://example.com/api"),
    ],
)
def test_socket_base_normalises_urls(api_base: str, expected: str) -> None:
    """Ensure the websocket base is normalised for different input forms."""

    client = TermoWebWSClient.__new__(TermoWebWSClient)
    client._client = SimpleNamespace(api_base=api_base)  # type: ignore[attr-defined]

    assert client._socket_base() == expected


def test_socket_base_uses_default_api_base_when_blank() -> None:
    """Ensure a blank client api_base falls back to the integration default."""

    client = TermoWebWSClient.__new__(TermoWebWSClient)
    client._client = SimpleNamespace(api_base="")  # type: ignore[attr-defined]

    assert client._socket_base() == termoweb_ws.API_BASE == API_BASE
