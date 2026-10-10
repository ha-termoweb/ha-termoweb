"""Codec helpers for the Ducaheat vendor interactions."""

from __future__ import annotations

from collections.abc import Mapping
import logging
from typing import Any

from custom_components.termoweb.domain import canonicalize_settings_payload
from custom_components.termoweb.domain.commands import (
    AccumulatorCommand,
    SetLock,
    SetMode,
    SetPresetTemps,
    SetPriority,
    SetProgram,
    SetSetpoint,
    SetUnits,
    StartBoost,
    StopBoost,
)
from custom_components.termoweb.domain.ids import NodeType

from .common import validate_prog, validate_units
from .ducaheat_models import (
    BoostPayload,
    LockWritePayload,
    ModeWritePayload,
    PriorityWritePayload,
    StatusWritePayload,
)
from .ducaheat_read_models import DucaheatSegmentedSettings, DucaheatThermostatSettings

_LOGGER = logging.getLogger(__name__)


def decode_settings(payload: Any, *, node_type: NodeType) -> dict[str, Any]:
    """Decode segmented settings payloads into canonical mappings."""

    if not isinstance(payload, dict):
        return {}

    if node_type is NodeType.THERMOSTAT:
        validated = DucaheatThermostatSettings.model_validate(payload)
        return canonicalize_settings_payload(
            validated.model_dump(exclude_none=True),
        )

    if node_type in {NodeType.HEATER, NodeType.ACCUMULATOR}:
        validated = DucaheatSegmentedSettings.model_validate(payload)
        flattened = validated.to_flat_dict(
            accumulator=node_type is NodeType.ACCUMULATOR
        )
        raw_keys = set(payload.keys())
        if isinstance(payload.get("status"), dict):
            raw_keys |= {f"status.{k}" for k in payload["status"]}
        if isinstance(payload.get("setup"), dict):
            raw_keys |= {f"setup.{k}" for k in payload["setup"]}
        decoded_keys = set(flattened.keys()) if flattened else set()
        _LOGGER.debug(
            "Ducaheat %s raw_keys=%s decoded_keys=%s",
            node_type.value,
            sorted(raw_keys),
            sorted(decoded_keys),
        )
        return canonicalize_settings_payload(flattened)

    return canonicalize_settings_payload(payload)


def encode_mode_command(command: SetMode) -> dict[str, Any]:
    """Encode a SetMode command for the mode endpoint."""

    return ModeWritePayload.model_validate(
        {"mode": command.mode, "boost_time": command.boost_time}
    ).model_dump(exclude_none=True)


def encode_setpoint_command(
    command: SetSetpoint,
    *,
    units: str | None = None,
    mode: str | None = None,
    boost_time: int | None = None,
) -> dict[str, Any]:
    """Encode a SetSetpoint command for the status endpoint."""

    payload: dict[str, Any] = {"stemp": command.setpoint}
    if units is not None:
        payload["units"] = units
    if mode is not None:
        payload["mode"] = mode
    if boost_time is not None:
        payload["boost_time"] = boost_time
    return StatusWritePayload.model_validate(payload).model_dump(exclude_none=True)


def encode_units_command(command: SetUnits) -> dict[str, Any]:
    """Encode a SetUnits command for the status endpoint."""

    return {"units": validate_units(command.units, trim=True)}


def encode_preset_temps_command(
    command: SetPresetTemps, *, units: str
) -> dict[str, Any]:
    """Encode preset temperatures as ``ice_temp``/``eco_temp``/``comf_temp`` for /status."""

    if len(command.presets) != 3:
        msg = "presets must contain [cold, night, day] values"
        raise ValueError(msg)

    ice, eco, comf = command.presets
    return StatusWritePayload.model_validate(
        {"ice_temp": ice, "eco_temp": eco, "comf_temp": comf, "units": units}
    ).model_dump(exclude_none=True)


def extract_prog_days(section: Any) -> dict[str, list[int]]:
    """Return the ``"0"``..``"6"`` day slot lists from a GET ``prog`` section."""

    if isinstance(section, Mapping) and isinstance(section.get("prog"), Mapping):
        section = section["prog"]
    if not isinstance(section, Mapping):
        return {}
    days: dict[str, list[int]] = {}
    for idx in range(7):
        slots = section.get(str(idx))
        if not isinstance(slots, list):
            continue
        try:
            values = [int(value) for value in slots]
        except (TypeError, ValueError):
            continue
        if len(values) in (24, 48) and all(value in (0, 1, 2) for value in values):
            days[str(idx)] = values
    return days


def encode_program_command(
    command: SetProgram, *, current: Mapping[str, list[int]] | None = None
) -> dict[str, Any]:
    """Encode a 168-slot program, echoing the slot resolution of ``current`` days.

    Without 48-slot days in ``current`` the documented 24 hourly slots per day
    are written. With 48-slot days, each half-hour pair is kept when its hourly
    value (``max`` of the pair, as shown on read) is unchanged.
    """

    validated = validate_prog(command.program)

    existing_days = current or {}
    half_hour = any(len(slots) == 48 for slots in existing_days.values())
    days: dict[str, list[int]] = {}
    for idx in range(7):
        hourly = validated[idx * 24 : (idx + 1) * 24]
        if not half_hour:
            days[str(idx)] = hourly
            continue
        existing = existing_days.get(str(idx))
        slots: list[int] = []
        for hour, value in enumerate(hourly):
            pair = existing[hour * 2 : hour * 2 + 2] if existing else []
            if len(pair) == 2 and max(pair) == value:
                slots.extend(pair)
            else:
                slots.extend([value, value])
        days[str(idx)] = slots

    return {"prog": days}


def encode_boost_command(command: AccumulatorCommand) -> dict[str, Any]:
    """Encode accumulator boost commands for the boost endpoint."""

    if not isinstance(command, (StartBoost, StopBoost)):  # pragma: no cover - defensive
        raise TypeError(f"Unsupported boost command: {type(command).__name__}")

    boost_flag = isinstance(command, StartBoost)
    payload = BoostPayload.model_validate(
        {
            "boost": boost_flag,
            "boost_time": command.boost_time,
            "stemp": command.stemp,
            "units": command.units,
        }
    )
    return payload.model_dump(exclude_none=True)


def encode_lock_command(command: SetLock) -> dict[str, Any]:
    """Encode a SetLock command for the lock endpoint."""

    return LockWritePayload.model_validate({"lock": command.lock}).model_dump()


def encode_priority_command(command: SetPriority) -> dict[str, Any]:
    """Encode a SetPriority command for the setup endpoint."""

    return PriorityWritePayload.model_validate(
        {"priority": command.priority}
    ).model_dump()
