"""Planner for Ducaheat segmented write sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from custom_components.termoweb.codecs.ducaheat_codec import (
    encode_boost_command,
    encode_extra_options_command,
    encode_lock_command,
    encode_mode_command,
    encode_preset_temps_command,
    encode_program_command,
    encode_setpoint_command,
    encode_units_command,
)
from custom_components.termoweb.codecs.ducaheat_models import StatusWritePayload
from custom_components.termoweb.domain.commands import (
    AccumulatorCommand,
    BaseCommand,
    SetExtraOptions,
    SetLock,
    SetMode,
    SetPresetTemps,
    SetProgram,
    SetSetpoint,
    SetUnits,
    StartBoost,
    StopBoost,
)
from custom_components.termoweb.domain.ids import NodeId, NodeType


@dataclass(slots=True)
class PlannedHttpCall:
    """HTTP call blueprint for segmented Ducaheat writes."""

    method: str
    path: str
    json: dict[str, Any] | None
    requires_token: bool = True


def plan_command(
    dev_id: str,
    node_id: NodeId,
    command: BaseCommand,
    *,
    units: str | None = None,
) -> list[PlannedHttpCall]:
    """Return the segmented write call plan for a single command."""

    base_path = f"/api/v2/devs/{dev_id}/{node_id.node_type.value}/{node_id.addr}"
    write_call = _build_write_call(
        base_path=base_path,
        node_id=node_id,
        command=command,
        units=units,
    )

    return [write_call]


def _build_write_call(
    *,
    base_path: str,
    node_id: NodeId,
    command: BaseCommand,
    units: str | None,
) -> PlannedHttpCall:
    """Build the primary write call for the supplied command."""

    if isinstance(command, SetMode):
        if node_id.node_type is NodeType.HEATER:
            payload = StatusWritePayload.model_validate(
                {"mode": command.mode, "boost_time": command.boost_time}
            ).model_dump(exclude_none=True)
            path = f"{base_path}/status"
        else:
            payload = encode_mode_command(command)
            path = f"{base_path}/mode"
    elif isinstance(command, SetSetpoint):
        payload = encode_setpoint_command(
            command,
            units=units,
            mode=command.mode,
            boost_time=command.boost_time,
        )
        path = f"{base_path}/status"
    elif isinstance(command, SetUnits):
        payload = encode_units_command(command)
        path = f"{base_path}/status"
    elif isinstance(command, SetPresetTemps):
        payload = encode_preset_temps_command(command, units=units)
        path = f"{base_path}/prog_temps"
    elif isinstance(command, SetProgram):
        payload = encode_program_command(command)
        path = f"{base_path}/prog"
    elif isinstance(command, SetExtraOptions):
        payload = encode_extra_options_command(command)
        path = f"{base_path}/setup"
    elif isinstance(command, SetLock):
        payload = encode_lock_command(command)
        path = f"{base_path}/lock"
    elif isinstance(command, (StartBoost, StopBoost)):
        _ensure_accumulator(node_id=node_id, command=command)
        payload = encode_boost_command(command)
        path = f"{base_path}/boost"
    else:
        msg = f"Unsupported command type: {type(command).__name__}"
        raise TypeError(msg)

    return PlannedHttpCall("POST", path, payload)


def _ensure_accumulator(*, node_id: NodeId, command: AccumulatorCommand) -> None:
    """Validate that accumulator-only commands target an accumulator."""

    if node_id.node_type is not NodeType.ACCUMULATOR:
        msg = (
            f"{type(command).__name__} commands are only supported for accumulators "
            f"(node_type={node_id.node_type.value})"
        )
        raise ValueError(msg)
