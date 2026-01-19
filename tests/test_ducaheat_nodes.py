"""Tests for Ducaheat node normalisation helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from aiohttp import ClientResponseError

from custom_components.termoweb.codecs.ducaheat_codec import decode_settings
from custom_components.termoweb.domain.ids import NodeType
from custom_components.termoweb.backend.ducaheat import DucaheatRESTClient


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
    """Websocket settings should expose boost end metadata without raw blobs."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    payload = {
        "boost": False,
        "boost_end": {"day": 6, "minute": 15},
        "boost_end_day": 3,
    }

    result = client._normalise_ws_settings(payload)

    assert result["boost"] is False
    assert "boost_end" not in result
    assert result["boost_end_day"] == 3
    assert result["boost_end_min"] == 15


def test_ducaheat_decode_settings_drops_boost_end_mapping() -> None:
    """Decoded accumulator settings should not retain raw boost_end mappings."""

    payload = {
        "status": {
            "mode": "boost",
            "boost_end": {"day": 2, "minute": 45},
        },
        "setup": {"extra_options": {"boost_end_min": 90}},
    }

    decoded = decode_settings(payload, node_type=NodeType.ACCUMULATOR)

    assert "boost_end" not in decoded
    assert decoded["boost_end_day"] == 2
    assert decoded["boost_end_min"] == 45


def test_ducaheat_decode_settings_validates_aliases_with_models() -> None:
    """Pydantic validation should normalise accumulator payload aliases."""

    payload = {
        "status": {
            "set_temp": "22.5",
            "ambient": "19.3",
            "units": "c",
            "boost_end": {"day": "5", "minute": "75"},
            "current_charge_per": "110",
            "lock": "off",
        },
        "setup": {
            "extra_options": {
                "boost_time": "120",
                "boost_end_min": 30,
                "target_charge_per": "15.9",
            }
        },
        "prog": [0, 1, 2] * 56,
        "prog_temps": {"antifrost": "7", "eco": "17.0", "comfort": "21"},
    }

    decoded = decode_settings(payload, node_type=NodeType.ACCUMULATOR)

    assert decoded["stemp"] == "22.5"
    assert decoded["mtemp"] == "19.3"
    assert decoded["units"] == "C"
    assert decoded["boost_end_day"] == 5
    assert decoded["boost_end_min"] == 75
    assert decoded["boost_time"] == 120
    assert decoded["current_charge_per"] == 100
    assert decoded["target_charge_per"] == 15
    assert decoded["prog"] == [0, 1, 2] * 56
    assert decoded["ptemp"] == ["7.0", "17.0", "21.0"]
    assert "boost_end" not in decoded


def test_ducaheat_rest_normalise_settings_charge_fields() -> None:
    """Accumulator settings should expose normalised charge metadata."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    payload = {
        "status": {
            "mode": "auto",
            "charging": "1",
            "current_charge_per": "45.5",
            "target_charge_per": 120,
        },
        "setup": {
            "extra_options": {
                "current_charge_per": 10,
            }
        },
    }

    result = client._normalise_settings(payload, node_type="acm")

    assert result["charging"] is True
    assert result["current_charge_per"] == 45
    assert result["target_charge_per"] == 100


def test_ducaheat_decode_thermostat_aliases_and_limits() -> None:
    """Thermostat decoding should rely on Pydantic validation."""

    valid_day = {"values": [1] * 24}
    payload = {
        "mode": "Manual",
        "state": "On",
        "setpoint": "20.0",
        "room_temp": "19.5",
        "units": "f",
        "ptemp": {"cold": "5", "night": "10.0", "day": "21"},
        "prog": {
            "days": {
                day: valid_day
                for day in ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
            }
        },
        "batt_level": "6",
    }

    decoded = decode_settings(payload, node_type=NodeType.THERMOSTAT)

    assert decoded["mode"] == "manual"
    assert decoded["state"] == "on"
    assert decoded["stemp"] == pytest.approx(20.0)
    assert decoded["mtemp"] == pytest.approx(19.5)
    assert decoded["units"] == "F"
    assert decoded["ptemp"] == ["5.0", "10.0", "21.0"]
    assert decoded["prog"] == [1] * 168
    assert decoded["batt_level"] == 5


def test_ducaheat_rest_normalise_ws_status_merges_charge_fields() -> None:
    """Status updates should refresh accumulator charge metadata in settings."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    payload = {
        "acm": {
            "settings": {"01": {"mode": "charge", "current_charge_per": 5}},
            "status": {
                "01": {
                    "charging": "1",
                    "current_charge_per": "15.2",
                    "target_charge_per": 80,
                }
            },
        }
    }

    normalised = client.normalise_ws_nodes(payload)

    settings = normalised["acm"]["settings"]["01"]
    assert settings["charging"] is True
    assert settings["current_charge_per"] == 15
    assert settings["target_charge_per"] == 80


@pytest.mark.asyncio
async def test_ducaheat_get_node_settings_normalises_thm() -> None:
    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")
    sample_prog = [0, 0, 0, 1, 1, 1] * 28
    payload = {
        "mode": "Manual",
        "state": "On",
        "stemp": "21.5",
        "mtemp": "20.3",
        "units": "c",
        "ptemp": ["16.0", "19.0", "20.0"],
        "prog": sample_prog,
        "batt_level": "5",
    }

    with (
        patch.object(client, "authed_headers", AsyncMock(return_value={})),
        patch.object(
            client, "_request", AsyncMock(return_value=payload)
        ) as mock_request,
    ):
        result = await client.get_node_settings("dev", ("thm", "01"))

    mock_request.assert_awaited_once()
    method, path = mock_request.await_args.args[:2]
    assert method == "GET"
    assert path == "/api/v2/devs/dev/thm/01/settings"
    assert result["mode"] == "manual"
    assert result["state"] == "on"
    assert result["stemp"] == pytest.approx(21.5)
    assert result["mtemp"] == pytest.approx(20.3)
    assert result["ptemp"] == [16.0, 19.0, 20.0]
    assert result["prog"] == sample_prog
    assert result["batt_level"] == 5


@pytest.mark.asyncio
async def test_ducaheat_get_node_samples_skips_thm() -> None:
    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    with (
        patch.object(client, "authed_headers", AsyncMock()) as mock_headers,
        patch.object(client, "_request", AsyncMock()) as mock_request,
    ):
        result = await client.get_node_samples("dev", ("thm", "01"), 0, 10)

    mock_headers.assert_not_called()
    mock_request.assert_not_called()
    assert result == []


@pytest.mark.asyncio
async def test_ducaheat_set_node_settings_thm_fallback_post() -> None:
    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")
    response = {"status": "ok"}
    side_effect = [
        ClientResponseError(SimpleNamespace(real_url=None), (), status=405, message=""),
        response,
    ]

    with (
        patch.object(client, "authed_headers", AsyncMock(return_value={})),
        patch.object(
            client, "_request", AsyncMock(side_effect=side_effect)
        ) as mock_request,
    ):
        result = await client.set_node_settings(
            "dev",
            ("thm", "01"),
            mode="auto",
            stemp=21.0,
        )

    assert result is response
    first_call = mock_request.await_args_list[0]
    second_call = mock_request.await_args_list[1]
    assert first_call.args[:2] == ("PATCH", "/api/v2/devs/dev/thm/01/settings")
    assert second_call.args[:2] == ("POST", "/api/v2/devs/dev/thm/01/settings")
    payload = second_call.kwargs["json"]
    assert payload["mode"] == "auto"
    assert payload["stemp"] == "21.0"
