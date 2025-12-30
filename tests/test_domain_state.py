"""Unit tests for domain state models."""

from custom_components.termoweb.domain.state import (
    AccumulatorState,
    HeaterState,
)


def test_accumulator_inherits_heater_state() -> None:
    """AccumulatorState should inherit from HeaterState and add fields."""

    state = AccumulatorState()

    assert isinstance(state, HeaterState)
    assert state.charge_level is None
    assert state.boost_active is None
