"""Tests for the shared wire-codec validators."""

from __future__ import annotations

from typing import Any

import pytest

from custom_components.termoweb.codecs.common import (
    format_temperature,
    safe_temperature,
    validate_prog,
    validate_ptemp,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(20, "20.0"), (21.37, "21.4"), ("19.2", "19.2"), (" 19 ", "19.0")],
)
def test_format_temperature(value: Any, expected: str) -> None:
    """Finite numbers and numeric strings format with one decimal."""

    assert format_temperature(value) == expected


@pytest.mark.parametrize(
    "value", [None, "", "warm", object(), True, float("nan"), "inf"]
)
def test_format_temperature_rejects(value: Any) -> None:
    """Non-numeric, bool and non-finite values raise (never 'nan' on the wire)."""

    with pytest.raises(ValueError, match="Invalid stemp value"):
        format_temperature(value, label="stemp")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (21.5, "21.5"),
        ("19", "19.0"),
        (" warm ", "warm"),
        ("", None),
        ("  ", None),
        ([1], None),
        (float("nan"), None),
    ],
)
def test_safe_temperature(value: Any, expected: str | None) -> None:
    """Inbound temperatures format when numeric, else stripped text or None."""

    assert safe_temperature(value) == expected


def test_validate_prog_accepts_integral_values() -> None:
    """Integral ints, floats and strings normalise to ints."""

    source: list[Any] = [str(i % 3) for i in range(166)] + [1.0, "2.0"]
    result = validate_prog(source)
    assert result[:3] == [0, 1, 2]
    assert result[-2:] == [1, 2]
    assert all(type(value) is int for value in result)


@pytest.mark.parametrize(
    ("prog", "match"),
    [
        ([0] * 10, "list of 168"),
        (tuple([0] * 168), "list of 168"),
        ([0] * 167 + [3], "must be 0, 1, or 2; got 3"),
        ([0] * 167 + ["x"], "non-integer value: 'x'"),
        ([0] * 167 + [1.5], "non-integer value: 1.5"),
        ([0] * 167 + [True], "non-integer value: True"),
        ([0] * 167 + [None], "non-integer value: None"),
    ],
)
def test_validate_prog_rejects(prog: Any, match: str) -> None:
    """Wrong shapes, out-of-range and non-integral slots raise ValueError."""

    with pytest.raises(ValueError, match=match):
        validate_prog(prog)


def test_validate_ptemp() -> None:
    """Three numeric presets format as one-decimal strings."""

    assert validate_ptemp([16, 18.5, "21"]) == ["16.0", "18.5", "21.0"]


@pytest.mark.parametrize(
    ("ptemp", "match"),
    [
        ((18, 19, 20), "ptemp must be a list"),
        ([18, 19], "ptemp must be a list"),
        ([18, "bad", 22], "ptemp contains non-numeric value: bad"),
    ],
)
def test_validate_ptemp_rejects(ptemp: Any, match: str) -> None:
    """Wrong shapes and non-numeric presets raise ValueError."""

    with pytest.raises(ValueError, match=match):
        validate_ptemp(ptemp)
