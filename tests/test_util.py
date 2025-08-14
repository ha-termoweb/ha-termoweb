from __future__ import annotations

import pytest

from custom_components.termoweb.util import float_or_none


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        ("abc", None),
        ("123", 123.0),
        (5, 5.0),
        ("   ", None),
    ],
)
def test_float_or_none(value, expected) -> None:
    assert float_or_none(value) == expected
