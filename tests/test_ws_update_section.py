"""Unit tests for ``resolve_ws_update_section``."""

from __future__ import annotations

import pytest

from custom_components.termoweb.backend.ws_client import resolve_ws_update_section


@pytest.mark.parametrize(
    "input_section, expected",
    [
        (None, (None, None)),
        ("status", ("status", None)),
        ("advanced_setup", ("advanced", "advanced_setup")),
        ("setup", ("settings", "setup")),
        ("custom", ("settings", "custom")),
    ],
)
def test_resolve_ws_update_section(
    input_section: str | None, expected: tuple[str | None, str | None]
) -> None:
    """Ensure websocket section names map to the expected tuples."""

    assert resolve_ws_update_section(input_section) == expected
