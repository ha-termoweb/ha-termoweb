"""Golden-payload tests for Ducaheat writes against docs/ducaheat_api.md."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from custom_components.termoweb.backend.ducaheat import DucaheatRESTClient
from custom_components.termoweb.codecs.ducaheat_codec import (
    decode_settings,
    encode_program_command,
    encode_units_command,
    extract_prog_days,
)
from custom_components.termoweb.codecs.ducaheat_read_models import (
    DucaheatExtraOptions,
    DucaheatSetupSegment,
    DucaheatStatusSegment,
)
from custom_components.termoweb.domain.commands import SetProgram, SetUnits
from custom_components.termoweb.domain.ids import NodeType


def _client(get_payload: Any = None) -> tuple[DucaheatRESTClient, list[tuple]]:
    """Return a Ducaheat client whose HTTP layer records every request."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")
    calls: list[tuple] = []

    async def fake_headers() -> dict[str, str]:
        return {"Authorization": "Bearer token"}

    async def fake_request(method: str, path: str, **kwargs: Any) -> Any:
        calls.append((method, path, kwargs.get("json")))
        return get_payload if method == "GET" else {}

    client.authed_headers = fake_headers  # type: ignore[method-assign]
    client._request = fake_request  # type: ignore[method-assign]
    return client, calls


@pytest.mark.asyncio
@pytest.mark.parametrize("node_type", ["htr", "acm"])
async def test_preset_write_uses_status_ice_eco_comf(node_type: str) -> None:
    """Presets are written via /status.

    docs/ducaheat_api.md "Change live status": "In the capture it is also used
    on htr to update preset temperatures":
    ``{"ice_temp":"5.0","eco_temp":"17.5","comf_temp":"20.5","units":"C"}``
    and "Program preset temperatures": "The app used /status with keys
    ice_temp, eco_temp, comf_temp".
    """

    client, calls = _client()

    await client.set_node_settings(
        "dev", (node_type, "1"), ptemp=[5.0, 17.5, 20.5], units="C"
    )

    assert calls == [
        (
            "POST",
            f"/api/v2/devs/dev/{node_type}/1/status",
            {"ice_temp": "5.0", "eco_temp": "17.5", "comf_temp": "20.5", "units": "C"},
        )
    ]


def test_preset_read_accepts_status_keys_and_zero() -> None:
    """GET status presets (ice/eco/comf) decode to ptemp; 0 is a real value."""

    decoded = decode_settings(
        {
            "status": {
                "mode": "auto",
                "ice_temp": "0.0",
                "eco_temp": "17.5",
                "comf_temp": "20.5",
            }
        },
        node_type=NodeType.HEATER,
    )

    assert decoded["ptemp"] == ["0.0", "17.5", "20.5"]
    assert "ice_temp" not in decoded


def test_preset_read_prefers_complete_status_over_prog_temps() -> None:
    """Status presets win; an incomplete status triple falls back to prog_temps."""

    prog_temps = {"antifrost": "7.0", "eco": "18.0", "comfort": "21.0"}
    full = decode_settings(
        {
            "status": {"ice_temp": "5", "eco_temp": "17", "comf_temp": "20"},
            "prog_temps": prog_temps,
        },
        node_type=NodeType.HEATER,
    )
    partial = decode_settings(
        {"status": {"ice_temp": "5"}, "prog_temps": prog_temps},
        node_type=NodeType.HEATER,
    )

    assert full["ptemp"] == ["5.0", "17.0", "20.0"]
    assert partial["ptemp"] == ["7.0", "18.0", "21.0"]


@pytest.mark.asyncio
async def test_prog_write_echoes_24_slot_get() -> None:
    """A 24-slot GET is written back with 24 hourly slots per day.

    docs/ducaheat_api.md "Weekly program": "Send the full program object
    echoed from GET. In this dump, htr days "0"..."6" each carry 24 integers
    (hourly)." Example: ``{"prog":{"0":[2,2,2,2,...,2],"1":[...],...}}``.
    """

    current = {"prog": {str(day): [0] * 24 for day in range(7)}}
    client, calls = _client(current)
    prog = [day % 3 for day in range(7) for _ in range(24)]

    await client.set_node_settings("dev", ("htr", "1"), prog=prog)

    assert calls == [
        ("GET", "/api/v2/devs/dev/htr/1", None),
        (
            "POST",
            "/api/v2/devs/dev/htr/1/prog",
            {"prog": {str(day): [day % 3] * 24 for day in range(7)}},
        ),
    ]


@pytest.mark.asyncio
async def test_prog_write_echoes_48_slot_get_preserving_half_hours() -> None:
    """A 48-slot GET is written back with 48 slots, keeping unchanged half-hours.

    docs/ducaheat_api.md "Validation invariants": "For weekly programs, echo
    the GET shape and write the whole object."
    """

    day0 = [1] * 48
    day0[0:2] = [0, 2]  # hour 0 reads as 2 (max); the user leaves it alone
    day0[10:12] = [2, 0]  # hour 5 reads as 2; the user changes it to 0
    current = {"prog": {"0": day0, **{str(d): [1] * 48 for d in range(1, 7)}}}
    client, calls = _client(current)
    prog = [1] * 168
    prog[0] = 2
    prog[5] = 0

    await client.set_node_settings("dev", ("acm", "2"), prog=prog)

    method, path, body = calls[-1]
    assert (method, path) == ("POST", "/api/v2/devs/dev/acm/2/prog")
    expected_day0 = [1] * 48
    expected_day0[0:2] = [0, 2]
    expected_day0[10:12] = [0, 0]
    assert body == {
        "prog": {"0": expected_day0, **{str(d): [1] * 48 for d in range(1, 7)}}
    }


def test_encode_program_defaults_to_documented_24_slots() -> None:
    """Without a usable GET shape the documented 24-slot form is written."""

    payload = encode_program_command(SetProgram([2] * 168), current={})

    assert payload == {"prog": {str(day): [2] * 24 for day in range(7)}}


def test_extract_prog_days_filters_invalid_entries() -> None:
    """Only "0".."6" days with 24/48 valid slots are echoed."""

    section = {
        "0": [0] * 48,
        "1": [0] * 10,
        "2": ["x"] * 24,
        "3": [5] * 24,
        "4": None,
        "5": [1] * 24,
    }

    assert extract_prog_days({"prog": section}) == {"0": [0] * 48, "5": [1] * 24}
    assert extract_prog_days(None) == {}


@pytest.mark.asyncio
async def test_thm_prog_write_is_single_level() -> None:
    """Thermostat prog is the day mapping itself, not ``{"prog": {"prog": ...}}``."""

    current = {"prog": {str(day): [0] * 24 for day in range(7)}}
    client, calls = _client(current)

    await client.set_node_settings("dev", ("thm", "3"), prog=[1] * 168)

    assert calls[0] == ("GET", "/api/v2/devs/dev/thm/3/settings", None)
    method, path, body = calls[1]
    assert (method, path) == ("PATCH", "/api/v2/devs/dev/thm/3/settings")
    assert body == {"prog": {str(day): [1] * 24 for day in range(7)}}


@pytest.mark.asyncio
@pytest.mark.parametrize("node_type", ["htr", "acm"])
async def test_no_units_write_unless_requested(node_type: str) -> None:
    """A call without explicit units must not write units (F11)."""

    client, calls = _client()

    assert await client.set_node_settings("dev", (node_type, "1")) == {}
    await client.set_node_settings("dev", (node_type, "1"), units="F")

    assert calls == [("POST", f"/api/v2/devs/dev/{node_type}/1/status", {"units": "F"})]


def test_encode_units_command_validates() -> None:
    """Units are validated before reaching the wire (F7)."""

    assert encode_units_command(SetUnits(" f ")) == {"units": "F"}
    with pytest.raises(ValueError):
        encode_units_command(SetUnits("unit:F"))


@pytest.mark.parametrize(
    "model", [DucaheatStatusSegment, DucaheatExtraOptions, DucaheatSetupSegment]
)
def test_boost_end_min_zero_is_midnight_not_missing(model: type) -> None:
    """boost_end_min == 0 (midnight) is kept over the nested mapping (F5)."""

    parsed = model.model_validate(
        {"boost_end_day": 0, "boost_end_min": 0, "boost_end": {"day": 9, "minute": 30}}
    )

    assert parsed.boost_end_day == 0
    assert parsed.boost_end_min == 0
