"""Tests for sensor energy normalisation helper."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from custom_components.termoweb.coordinator import EnergyStateCoordinator
from custom_components.termoweb.sensor import _normalise_energy_value


class _ScaleStub(SimpleNamespace):
    """Coordinator stub that exposes a custom energy scale."""


@pytest.mark.parametrize(
    ("coordinator", "raw", "expected"),
    [
        pytest.param(object(), True, None, id="bool_true"),
        pytest.param(object(), False, None, id="bool_false"),
        pytest.param(
            object.__new__(EnergyStateCoordinator),
            5,
            5.0,
            id="energy_coordinator_defaults_to_kwh",
        ),
        pytest.param(
            object(),
            "1200",
            1.2,
            id="integer_string_interpreted_as_wh",
        ),
        pytest.param(
            _ScaleStub(_termoweb_energy_scale=0.5),
            4,
            2.0,
            id="numeric_scale_attribute",
        ),
        pytest.param(
            _ScaleStub(_termoweb_energy_scale="2"),
            4,
            8.0,
            id="string_numeric_scale_attribute",
        ),
        pytest.param(
            _ScaleStub(_termoweb_energy_scale="wh"),
            500,
            0.5,
            id="textual_scale_wh",
        ),
        pytest.param(
            object(),
            "not-a-number",
            None,
            id="invalid_string_returns_none",
        ),
    ],
)
def test_normalise_energy_value(coordinator: object, raw: object, expected: float | None) -> None:
    """Ensure ``_normalise_energy_value`` handles diverse inputs."""

    assert _normalise_energy_value(coordinator, raw) == expected
