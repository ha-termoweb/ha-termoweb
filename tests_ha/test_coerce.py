"""Contract tests for the shared coercion helpers."""

from __future__ import annotations

from typing import Any

import pytest

from custom_components.termoweb.coerce import (
    as_bool,
    as_float,
    as_int,
    as_number,
    as_percentage,
)

_JUNK: list[Any] = [None, "", "   ", "abc", "nan", "inf", float("nan"), float("inf")]
_JUNK += [True, False, [1], {"a": 1}, object(), b"21"]


@pytest.mark.parametrize("value", _JUNK)
def test_numeric_helpers_reject_junk(value: Any) -> None:
    """Numeric helpers share one rejection set, including bools and non-finite."""

    assert as_number(value) is None
    assert as_float(value) is None
    assert as_int(value) is None
    assert as_percentage(value) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [(5, 5.0), (21.5, 21.5), ("123", 123.0), (" 7.2 ", 7.2), ("-3", -3.0)],
)
def test_as_float(value: Any, expected: float) -> None:
    """as_float returns finite floats from numbers and numeric strings."""

    result = as_float(value)
    assert result == expected
    assert type(result) is float


def test_as_number_keeps_int_and_float_types() -> None:
    """as_number keeps ints as int and turns strings into float."""

    assert as_number(1000) == 1000
    assert type(as_number(1000)) is int
    assert type(as_number(2.5)) is float
    assert as_number(" 21 ") == 21.0
    assert type(as_number("21")) is float


@pytest.mark.parametrize(
    ("value", "expected"),
    [(42, 42), ("7", 7), ("21.5", 21), (21.5, 21), (" 7.2 ", 7), (-1.9, -1)],
)
def test_as_int_truncates_toward_zero(value: Any, expected: int) -> None:
    """as_int is int(as_float(value)): '21.5' and 21.5 both give 21."""

    assert as_int(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(50, 50), (150, 100), (-10, 0), (75.9, 75), ("42.5", 42)],
)
def test_as_percentage_clamps(value: Any, expected: int) -> None:
    """as_percentage is as_int clamped to 0..100."""

    assert as_percentage(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        (1.0, True),
        (0.0, False),
        ("Yes", True),
        (" ON ", True),
        ("true", True),
        ("1", True),
        ("off", False),
        ("No", False),
        ("0", False),
        ("false", False),
        (2, None),
        (-1, None),
        (0.5, None),
        ("maybe", None),
        ("", None),
        (None, None),
        ([1], None),
    ],
)
def test_as_bool(value: Any, expected: bool | None) -> None:
    """as_bool only maps bools, 1/0 and the documented strings; 2 is None."""

    assert as_bool(value) is expected
