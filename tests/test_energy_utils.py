"""Tests for energy utilities."""

from custom_components.termoweb.energy import _iso_date


def test_iso_date_for_recent_timestamp() -> None:
    """_iso_date should convert timestamp to ISO date string."""

    assert _iso_date(1_700_000_000) == "2023-11-14"


def test_iso_date_for_unix_epoch() -> None:
    """_iso_date should convert zero to the Unix epoch date."""

    assert _iso_date(0) == "1970-01-01"
