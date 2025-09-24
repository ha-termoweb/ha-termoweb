from __future__ import annotations

import pytest

import custom_components.termoweb.utils as utils
from custom_components.termoweb.utils import float_or_none


def test_extract_heater_addrs() -> None:
    extract = utils.extract_heater_addrs

    assert extract({}) == []

    nodes = {
        "nodes": [
            {"type": "htr", "addr": "A"},
            {"type": "foo", "addr": "B"},
            {"type": "HTR", "addr": 1},
        ]
    }

    assert extract(nodes) == ["A", "1"]


def test_extract_heater_addrs_deduplicates() -> None:
    extract = utils.extract_heater_addrs

    nodes = {"nodes": [{"type": "htr", "addr": "A"}, {"type": "HTR", "addr": "A"}]}

    assert extract(nodes) == ["A"]



@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        ("abc", None),
        ("123", 123.0),
        (5, 5.0),
        ("   ", None),
        (float("nan"), None),
        (float("inf"), None),
    ],
)
def test_float_or_none(value, expected) -> None:
    assert float_or_none(value) == expected


@pytest.mark.parametrize("value", ["nan", "inf"])
def test_float_or_none_non_finite_strings(value) -> None:
    assert float_or_none(value) is None


def test_get_brand_api_base_fallback() -> None:
    from custom_components.termoweb import const

    assert const.get_brand_api_base("unknown-brand") == const.API_BASE
