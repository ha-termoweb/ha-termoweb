"""TermoWeb read models format temperatures with the shared inbound helper."""

from __future__ import annotations

from custom_components.termoweb.codecs.termoweb_models import (
    HeaterSettingsPayload,
    ThermostatSettingsPayload,
)


def test_heater_temperatures_use_safe_temperature() -> None:
    """Numbers format to one decimal; text is stripped; junk and nan become None."""

    payload = HeaterSettingsPayload.model_validate(
        {
            "stemp": 21,
            "mtemp": " warm ",
            "temp": float("nan"),
            "ptemp": ["5", "", 21.25],
        }
    )

    assert payload.stemp == "21.0"
    assert payload.mtemp == "warm"
    assert payload.temp is None
    assert payload.ptemp == ["5.0", None, "21.2"]


def test_thermostat_non_scalar_temperature_is_dropped() -> None:
    """A non-scalar temperature no longer passes through unchanged."""

    payload = ThermostatSettingsPayload.model_validate({"stemp": [1], "mtemp": None})

    assert payload.stemp is None
    assert payload.mtemp is None
