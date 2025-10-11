"""Tests for Ducaheat node normalisation helpers."""

from __future__ import annotations

from types import SimpleNamespace


def test_ducaheat_rest_normalise_ws_nodes_passthrough_scalars() -> None:
    """Non-mapping payloads should return the original object unchanged."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    scalar_payload = "scalar"
    list_payload = ["entry"]

    scalar_result = client.normalise_ws_nodes(scalar_payload)
    list_result = client.normalise_ws_nodes(list_payload)

    assert scalar_result is scalar_payload
    assert list_result is list_payload
    assert list_payload == ["entry"]


def test_ducaheat_rest_normalise_ws_nodes_preserve_non_settings_sections() -> None:
    """Only the settings section should be normalised within nested payloads."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    heater_prog = {str(day): [day] * 48 for day in range(7)}
    payload = {
        "htr": {
            "settings": {
                "01": {"prog": {"prog": heater_prog}, "mode": "auto"},
            },
            "status": {"01": {"temp": 21}},
            "alerts": ["unchanged"],
        }
    }

    result = client.normalise_ws_nodes(payload)

    # Settings bucket should be normalised into a flattened schedule.
    settings = result["htr"]["settings"]["01"]
    assert isinstance(settings["prog"], list)
    assert len(settings["prog"]) == 168

    # Non-settings sections should retain their original identities.
    assert result["htr"]["status"] is payload["htr"]["status"]
    assert result["htr"]["alerts"] is payload["htr"]["alerts"]

    # Original payload must remain untouched after normalisation.
    assert len(payload["htr"]["settings"]["01"]["prog"]["prog"]["1"]) == 48

from custom_components.termoweb.backend.ducaheat import DucaheatRESTClient


def test_ducaheat_rest_normalise_ws_nodes_mixed_settings_types() -> None:
    """Normalisation should coerce mapping payloads while preserving scalars."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    heater_prog = {str(day): [day] * 48 for day in range(7)}
    accumulator_prog = {str(day): [1] * 24 for day in range(7)}

    payload = {
        "htr": {
            "settings": {
                "01": {
                    "prog": {"prog": heater_prog},
                    "mode": "auto",
                },
                "02": ["unexpected"],
            },
            "status": [{"temp": 21}],
        },
        "acm": {
            "settings": {
                "03": {
                    "prog": {"days": accumulator_prog},
                    "mode": "charge",
                },
                "04": "raw",
            },
            "status": {"03": {"temp": 19}},
        },
        "pmo": ["unchanged"],
    }

    result = client.normalise_ws_nodes(payload)

    heater_settings = result["htr"]["settings"]["01"]
    assert isinstance(heater_settings, dict)
    assert len(heater_settings["prog"]) == 168
    # Spot check that day 1 (Monday) slots collapsed from the vendor's 48 entries.
    assert heater_settings["prog"][24:48] == [1] * 24

    accumulator_settings = result["acm"]["settings"]["03"]
    assert isinstance(accumulator_settings, dict)
    assert len(accumulator_settings["prog"]) == 168

    # Non-mapping settings entries should pass through untouched.
    assert result["htr"]["settings"]["02"] == ["unexpected"]
    assert result["acm"]["settings"]["04"] == "raw"

    # Non-settings sections retain their original typing.
    assert result["htr"]["status"] == [{"temp": 21}]
    assert result["pmo"] == ["unchanged"]

    # Original payload should remain unchanged for mapping coercions.
    assert len(payload["htr"]["settings"]["01"]["prog"]["prog"]["1"]) == 48
    assert len(payload["acm"]["settings"]["03"]["prog"]["days"]["1"]) == 24


def test_ducaheat_rest_normalise_ws_settings_boost_fields() -> None:
    """Websocket settings should expose boost end metadata consistently."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    payload = {
        "boost": False,
        "boost_end": {"day": 6, "minute": 15},
        "boost_end_day": 3,
    }

    result = client._normalise_ws_settings(payload)

    assert result["boost"] is False
    assert result["boost_end"] == {"day": 6, "minute": 15}
    # Direct fields should take precedence over derived values.
    assert result["boost_end_day"] == 3
    assert result["boost_end_min"] == 15
