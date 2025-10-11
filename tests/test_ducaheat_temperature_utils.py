"""Unit tests for Ducaheat temperature utilities."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from custom_components.termoweb.backend.ducaheat import DucaheatRESTClient


@pytest.fixture()
def ducaheat_client() -> DucaheatRESTClient:
    """Create a minimal Ducaheat client for helper tests."""

    return DucaheatRESTClient(SimpleNamespace(), "user", "pass")


@pytest.mark.parametrize(
    "value, expected",
    [
        (21.567, "21.6"),
        ("19.2", "19.2"),
        ("19", "19.0"),
        (18, "18.0"),
    ],
)
def test_format_temp_accepts_numeric_values(
    ducaheat_client: DucaheatRESTClient, value: float | str, expected: str
) -> None:
    """_format_temp should normalise float-able values to one decimal string."""

    assert ducaheat_client._format_temp(value) == expected


@pytest.mark.parametrize("value", [None, "abc", object(), ""]) 
def test_format_temp_rejects_invalid_values(
    ducaheat_client: DucaheatRESTClient, value: object
) -> None:
    """_format_temp should raise ``ValueError`` for bad temperature input."""

    with pytest.raises(ValueError):
        ducaheat_client._format_temp(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value, expected",
    [
        ("c", "C"),
        ("f", "F"),
        (" C ", "C"),
        (None, "C"),
    ],
)
def test_ensure_units_uppercases_valid_values(
    ducaheat_client: DucaheatRESTClient, value: str | None, expected: str
) -> None:
    """_ensure_units should default to Celsius and uppercase valid codes."""

    assert ducaheat_client._ensure_units(value) == expected


@pytest.mark.parametrize("value", ["kelvin", "K", "x", 10])
def test_ensure_units_rejects_invalid_values(
    ducaheat_client: DucaheatRESTClient, value: object
) -> None:
    """_ensure_units should reject unsupported unit strings."""

    with pytest.raises(ValueError):
        ducaheat_client._ensure_units(value)  # type: ignore[arg-type]
