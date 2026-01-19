"""Tests for RESTClient._ensure_temperature."""

from __future__ import annotations

import pytest

from custom_components.termoweb.backend.rest_client import RESTClient
from tests.test_api import FakeSession


@pytest.fixture
def rest_client() -> RESTClient:
    """Provide a RESTClient instance using the FakeSession helper."""

    return RESTClient(FakeSession(), "user@example.com", "secret")


@pytest.mark.parametrize(
    "value,expected",
    [
        (20, "20.0"),
        (21.37, "21.4"),
        ("19.2", "19.2"),
    ],
)
def test_ensure_temperature_formats_numeric_values(
    rest_client: RESTClient, value: object, expected: str
) -> None:
    """Ensure numeric values are formatted with a single decimal place."""

    assert rest_client._ensure_temperature(value) == expected


@pytest.mark.parametrize("value", [None, "warm", object()])
def test_ensure_temperature_rejects_invalid_values(
    rest_client: RESTClient, value: object
) -> None:
    """Ensure non-numeric values raise ValueError."""

    with pytest.raises(ValueError):
        rest_client._ensure_temperature(value)
