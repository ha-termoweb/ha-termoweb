"""Tests for cadence extraction utilities in the Ducaheat websocket client."""

from __future__ import annotations

import pytest

from custom_components.termoweb.backend import ducaheat_ws


class CadenceProbe(ducaheat_ws.DucaheatWSClient):
    """Lightweight subclass that exposes cadence helpers without full setup."""

    def __init__(self) -> None:
        """Initialise the probe without invoking the parent constructor."""
        # Intentionally skip parent initialisation to avoid heavy dependencies.


@pytest.fixture
def cadence_probe() -> CadenceProbe:
    """Return a cadence probe instance for testing the helper methods."""

    return CadenceProbe()


def test_normalise_cadence_value_filters_invalid_inputs(
    cadence_probe: CadenceProbe,
) -> None:
    """Ensure invalid cadence values are rejected."""

    invalid_values = [
        None,
        -1,
        0,
        "-5",
        float("inf"),
        float("-inf"),
        float("nan"),
        object(),
    ]
    for candidate in invalid_values:
        assert cadence_probe._normalise_cadence_value(candidate) is None


def test_normalise_cadence_value_accepts_positive_inputs(
    cadence_probe: CadenceProbe,
) -> None:
    """Ensure positive strings and floats are accepted."""

    assert cadence_probe._normalise_cadence_value("30") == pytest.approx(30.0)
    assert cadence_probe._normalise_cadence_value(12.5) == pytest.approx(12.5)


def test_extract_cadence_candidates_handles_nested_mappings(
    cadence_probe: CadenceProbe,
) -> None:
    """Validate that cadence hints are collected once per mapping."""

    shared: dict[str, object] = {"poll_seconds": "15"}
    shared["self"] = shared

    payload: dict[str, object] = {
        "lease_seconds": "5",
        "nested": {"cadence_seconds": "10"},
        "ignored": {"poll_seconds": "nan", "cadence_seconds": "-1"},
        "loop1": shared,
        "loop2": shared,
        "unrelated": {"cadence": "25"},
    }
    payload["nested"]["again"] = payload

    values = cadence_probe._extract_cadence_candidates(payload)

    assert sorted(values) == [5.0, 10.0, 15.0]
    assert len(values) == 3
