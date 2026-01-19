from __future__ import annotations

from unittest.mock import MagicMock

import aiohttp
import pytest

from custom_components.termoweb.backend.rest_client import RESTClient


@pytest.fixture()
def rest_client() -> RESTClient:
    """Return a REST client instance with a mocked session."""

    session = MagicMock(spec=aiohttp.ClientSession)
    return RESTClient(session, "user@example.com", "password")


def test_ensure_ptemp_formats_numeric_values(rest_client: RESTClient) -> None:
    """Ensure numeric preset temperatures are formatted as strings."""

    presets = [16, 18.5, "21"]

    result = rest_client._ensure_ptemp(presets)

    assert result == ["16.0", "18.5", "21.0"]


def test_ensure_ptemp_requires_list(rest_client: RESTClient) -> None:
    """Reject non-list inputs for preset temperatures."""

    with pytest.raises(ValueError, match="ptemp must be a list"):
        rest_client._ensure_ptemp((18, 19, 20))  # type: ignore[arg-type]


def test_ensure_ptemp_requires_three_values(rest_client: RESTClient) -> None:
    """Reject lists that do not contain exactly three values."""

    with pytest.raises(ValueError, match="ptemp must be a list"):
        rest_client._ensure_ptemp([18, 19])


def test_ensure_ptemp_validates_elements(rest_client: RESTClient) -> None:
    """Reject values that cannot be normalised by _ensure_temperature."""

    with pytest.raises(ValueError, match="ptemp contains non-numeric value: bad"):
        rest_client._ensure_ptemp([18, "bad", 22])


def test_ensure_ptemp_delegates_to_ensure_temperature(
    rest_client: RESTClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure _ensure_ptemp calls _ensure_temperature for each element."""

    calls: list[int] = []

    def fake_ensure_temperature(value: int) -> str:
        calls.append(value)
        return f"value-{value}"

    monkeypatch.setattr(rest_client, "_ensure_temperature", fake_ensure_temperature)

    result = rest_client._ensure_ptemp([1, 2, 3])

    assert calls == [1, 2, 3]
    assert result == ["value-1", "value-2", "value-3"]
