"""Tests for the Ducaheat brand header defaults."""

from custom_components.termoweb.backend.ducaheat_ws import _brand_headers
from custom_components.termoweb.const import USER_AGENT


def test_brand_headers_default_user_agent() -> None:
    """User agent falls back to integration default while keeping required keys."""

    headers = _brand_headers("", "")
    expected_keys = {
        "User-Agent",
        "Accept-Language",
        "X-Requested-With",
        "Origin",
        "Referer",
        "Accept",
        "Accept-Encoding",
        "Cache-Control",
        "Pragma",
        "Connection",
    }

    assert headers["User-Agent"] == USER_AGENT
    assert expected_keys <= set(headers)
    assert headers["X-Requested-With"] == ""

    custom_agent = "Integration/1.0"
    assert _brand_headers(custom_agent, "")["User-Agent"] == custom_agent
