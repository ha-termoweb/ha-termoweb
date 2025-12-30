"""Codec helpers for the Ducaheat vendor interactions."""

from __future__ import annotations

from typing import Any

from custom_components.termoweb.domain.commands import (
    AccumulatorCommand,
    BaseCommand,
    SetExtraOptions,
    SetMode,
    SetPresetTemps,
    SetProgram,
    SetSetpoint,
    SetUnits,
    StartBoost,
    StopBoost,
)
from custom_components.termoweb.domain.ids import NodeType

from .ducaheat_models import (
    BoostPayload,
    ExtraOptionsPayload,
    ModeWritePayload,
    ProgramWritePayload,
    SelectRequest,
    SetupPayload,
    StatusWritePayload,
)


def decode_payload(payload: Any) -> Any:
    """Decode a Ducaheat payload (placeholder)."""

    raise NotImplementedError


def decode_status(payload: Any) -> Any:
    """Decode a Ducaheat status payload into domain state."""

    raise NotImplementedError


def decode_samples(payload: Any) -> Any:
    """Decode Ducaheat samples payloads."""

    raise NotImplementedError


def decode_prog(payload: Any) -> Any:
    """Decode Ducaheat program payloads."""

    raise NotImplementedError


def encode_select_payload(select: bool) -> dict[str, Any]:
    """Encode a selection payload."""

    return SelectRequest.model_validate({"select": select}).model_dump()


def encode_mode_command(command: SetMode) -> dict[str, Any]:
    """Encode a SetMode command for the mode endpoint."""

    return ModeWritePayload.model_validate({"mode": command.mode}).model_dump()


def encode_setpoint_command(
    command: SetSetpoint, *, units: str | None = None
) -> dict[str, Any]:
    """Encode a SetSetpoint command for the status endpoint."""

    payload: dict[str, Any] = {"stemp": command.setpoint}
    if units is not None:
        payload["units"] = units
    return StatusWritePayload.model_validate(payload).model_dump(exclude_none=True)


def encode_units_command(command: SetUnits) -> dict[str, Any]:
    """Encode a SetUnits command for the status endpoint."""

    return StatusWritePayload.model_validate({"units": command.units}).model_dump(
        exclude_none=True
    )


def encode_preset_temps_command(
    command: SetPresetTemps, *, units: str | None = None
) -> dict[str, Any]:
    """Encode preset temperature updates for the status endpoint."""

    if len(command.presets) != 3:
        msg = "presets must contain [cold, night, day] values"
        raise ValueError(msg)

    cold, night, day = command.presets
    payload: dict[str, Any] = {
        "ice_temp": cold,
        "eco_temp": night,
        "comf_temp": day,
    }
    if units is not None:
        payload["units"] = units
    return StatusWritePayload.model_validate(payload).model_dump(exclude_none=True)


def encode_program_command(command: SetProgram) -> dict[str, Any]:
    """Encode a SetProgram command for the prog endpoint."""

    return ProgramWritePayload.model_validate({"prog": command.program}).model_dump()


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

    boost_flag: bool
    stemp_value: Any = None
    units_value: str | None = None
    minutes: Any = None
    if isinstance(command, StartBoost):
        boost_flag = True
        stemp_value = command.stemp
        units_value = command.units
        minutes = command.boost_time
    elif isinstance(command, StopBoost):
        boost_flag = False
        stemp_value = command.stemp
        units_value = command.units
        minutes = command.boost_time
    else:  # pragma: no cover - defensive
        raise TypeError(f"Unsupported boost command: {type(command).__name__}")

    payload = BoostPayload.model_validate(
        {
            "boost": boost_flag,
            "boost_time": minutes,
            "stemp": stemp_value,
            "units": units_value,
        }
    )
    return payload.model_dump(exclude_none=True)


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
    raise TypeError(f"Unsupported command type: {type(command).__name__}")
