"""Codec helpers for the Ducaheat vendor interactions."""

from __future__ import annotations

import logging
from typing import Any

from custom_components.termoweb.domain import canonicalize_settings_payload
from custom_components.termoweb.domain.commands import (
    AccumulatorCommand,
    BaseCommand,
    SetExtraOptions,
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

from .common import format_temperature, validate_units
from .ducaheat_models import (
    BoostPayload,
    ExtraOptionsPayload,
    LockWritePayload,
    ModeWritePayload,
    PriorityWritePayload,
    SetupPayload,
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
        if isinstance(payload, dict):
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
    boost: bool | None = None,
    boost_time: int | None = None,
) -> dict[str, Any]:
    """Encode a SetSetpoint command for the status endpoint."""

    payload: dict[str, Any] = {"stemp": command.setpoint}
    if units is not None:
        payload["units"] = units
    if mode is not None:
        payload["mode"] = mode
    if boost is not None:
        payload["boost"] = boost
    if boost_time is not None:
        payload["boost_time"] = boost_time
    return StatusWritePayload.model_validate(payload).model_dump(exclude_none=True)


def encode_units_command(command: SetUnits) -> dict[str, Any]:
    """Encode a SetUnits command for the status endpoint."""

    return {"units": command.units}


def encode_preset_temps_command(
    command: SetPresetTemps, *, units: str | None = None
) -> dict[str, Any]:
    """Encode preset temperature updates for the status endpoint."""

    if len(command.presets) != 3:
        msg = "presets must contain [cold, night, day] values"
        raise ValueError(msg)

    cold, night, day = command.presets
    payload: dict[str, Any] = {
        "cold": format_temperature(cold),
        "night": format_temperature(night),
        "day": format_temperature(day),
    }
    if units is not None:
        payload["units"] = validate_units(units, trim=True)
    return payload


def encode_program_command(command: SetProgram) -> dict[str, Any]:
    """Encode a SetProgram command for the prog endpoint."""

    if not isinstance(command.program, list) or len(command.program) != 168:
        msg = "prog must be a list of 168 integers (0, 1, or 2)"
        raise ValueError(msg)

    try:
        validated = [int(value) for value in command.program]
    except (TypeError, ValueError) as err:
        msg = "prog contains non-integer value"
        raise ValueError(msg) from err

    if any(value not in (0, 1, 2) for value in validated):
        msg = "prog values must be 0, 1, or 2"
        raise ValueError(msg)

    half_hour: dict[str, list[int]] = {}
    for idx in range(7):
        start = idx * 24
        hourly = validated[start : start + 24]
        slots: list[int] = []
        for value in hourly:
            slots.extend([value, value])
        half_hour[str(idx)] = slots

    return {"prog": half_hour}


def encode_extra_options_command(command: SetExtraOptions) -> dict[str, Any]:
    """Encode an extra options command for the setup endpoint."""

    payload = SetupPayload.model_validate(
        {
            "extra_options": ExtraOptionsPayload(
                boost_time=command.boost_time,
                boost_temp=command.boost_temp,
            )
        }
    )
    return payload.model_dump(exclude_none=True)


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


def infer_status_endpoint(node_type: NodeType, command: BaseCommand) -> str:
    """Return the segmented endpoint name for a status-like command."""

    _ = node_type  # reserved for future specialisation
    if isinstance(command, SetProgram):
        return "prog"
    if isinstance(command, SetMode):
        return "mode"
    if isinstance(command, (SetSetpoint, SetPresetTemps, SetUnits)):
        return "status"
    if isinstance(command, SetExtraOptions):
        return "setup"
    if isinstance(command, (StartBoost, StopBoost)):
        return "boost"
    if isinstance(command, SetLock):
        return "lock"
    if isinstance(command, SetPriority):
        return "setup"
    raise TypeError(f"Unsupported command type: {type(command).__name__}")
