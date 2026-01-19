"""Tests for the REST client program normalisation helper."""

from __future__ import annotations

from unittest.mock import AsyncMock

import aiohttp
import pytest

from custom_components.termoweb.backend.rest_client import RESTClient


@pytest.fixture
def rest_client() -> RESTClient:
    """Return a REST client instance with a mocked aiohttp session."""

    session = AsyncMock(spec=aiohttp.ClientSession)
    return RESTClient(session, "user@example.com", "secret")


def test_ensure_prog_accepts_int_convertible_values(rest_client: RESTClient) -> None:
    """_ensure_prog accepts iterable values that normalise to ints."""

    source = [str(i % 3) for i in range(168)]

    result = rest_client._ensure_prog(source)

    assert result == [int(value) for value in source]
    assert all(isinstance(value, int) for value in result)


def test_ensure_prog_rejects_wrong_length(rest_client: RESTClient) -> None:
    """_ensure_prog rejects lists that are not exactly 168 entries long."""

    with pytest.raises(ValueError, match="prog must be a list of 168 integers"):
        rest_client._ensure_prog([0] * 10)


def test_ensure_prog_rejects_out_of_range_values(rest_client: RESTClient) -> None:
    """_ensure_prog rejects values outside the 0/1/2 range."""

    invalid = [0] * 167 + [3]

    with pytest.raises(ValueError, match="prog values must be 0, 1, or 2"):
        rest_client._ensure_prog(invalid)
