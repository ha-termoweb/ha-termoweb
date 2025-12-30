"""Tests for Ducaheat preset temperature serialisation."""

from __future__ import annotations

import pytest

from custom_components.termoweb.backend.ducaheat import DucaheatRESTClient


class _StubSession:
    """Minimal session stub for constructing the REST client."""


def _make_client() -> DucaheatRESTClient:
    """Create a Ducaheat REST client with placeholder credentials."""

    return DucaheatRESTClient(
        _StubSession(),
        "user",
        "pass",
        api_base="https://api.termoweb.fake",
    )


def test_serialise_prog_temps_formats_values() -> None:
    """Preset temperatures should be formatted to one decimal place."""

    client = _make_client()
    result = client._serialise_prog_temps([5, 15.26, 21])
    assert result == {"cold": "5.0", "night": "15.3", "day": "21.0"}


@pytest.mark.parametrize(
    "ptemp",
    (
        123,
        [10, 15],
        [10, "bad", 20],
    ),
)
def test_serialise_prog_temps_invalid_inputs(ptemp: object) -> None:
    """Invalid preset temperature inputs should raise ``ValueError``."""

    client = _make_client()
    with pytest.raises(ValueError):
        client._serialise_prog_temps(ptemp)  # type: ignore[arg-type]
