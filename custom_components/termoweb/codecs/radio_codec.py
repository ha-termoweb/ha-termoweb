"""Codec between the integration's canonical settings dict and radio records.

The radio wire is raw bytes; ``backend/radio/protocol.py`` turns them into
frozen dataclasses. This module maps those records onto the same canonical
settings dict the cloud codecs produce, and maps canonical write commands back
onto radio payloads. A key is only emitted when the record carries its value.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

from custom_components.termoweb.backend.radio import protocol
from custom_components.termoweb.backend.radio.dialect import DIALECT_A, Dialect
from custom_components.termoweb.codecs.common import format_temperature
from custom_components.termoweb.domain.commands import (
    BaseCommand,
    SetLock,
    SetMode,
    SetPresetTemps,
    SetProgram,
    SetSetpoint,
)

HOURS_PER_DAY = 24
PROGRAM_LEN = protocol.DAYS_PER_WEEK * HOURS_PER_DAY  # 168, Monday 00:00 first
RADIO_UNITS = "C"  # the radio carries every temperature in half degrees Celsius

# Radio mode byte -> canonical mode string (the cloud ``mode`` values).
MODE_FROM_RADIO: dict[int, str] = {
    protocol.MODE_AUTO: "auto",
    protocol.MODE_MANUAL: "manual",
    protocol.MODE_OVERRIDE: "modified_auto",
    protocol.MODE_OFF: "off",
}

# Canonical mode string -> radio mode byte for a plain ``B4 <mode>`` write.
MODE_TO_RADIO: dict[str, int] = {
    "auto": protocol.MODE_AUTO,
    "manual": protocol.MODE_MANUAL,
    "heat": protocol.MODE_MANUAL,
    "off": protocol.MODE_OFF,
}

OVERRIDE_MODE = "modified_auto"
_SETPOINT_MODES = frozenset({None, "manual", "heat", OVERRIDE_MODE})


def _temperature(value: float) -> str:
    """Return a temperature in the cloud codec's one-decimal string form."""

    return format_temperature(value)


def settings_from_status(record: protocol.StatusRecord) -> dict[str, Any]:
    """Return canonical settings carried by a decoded status record."""

    settings: dict[str, Any] = {
        "units": RADIO_UNITS,
        "ptemp": [
            _temperature(record.anti_frost_c),
            _temperature(record.eco_c),
            _temperature(record.comfort_c),
        ],
    }
    mode = MODE_FROM_RADIO.get(record.mode_code)
    if mode is not None:
        settings["mode"] = mode
    if record.room_temp_c is not None:
        settings["mtemp"] = _temperature(record.room_temp_c)
    if record.setpoint_c is not None:
        settings["stemp"] = _temperature(record.setpoint_c)
    if record.measured_power_w is not None:
        settings["max_power"] = record.measured_power_w
    if record.flags is not None:
        heating = bool(record.flags & protocol.FLAG_ACTIVE)
        settings["state"] = "on" if heating else "off"
        settings["lock"] = bool(record.flags & protocol.FLAG_LOCKED)
    return settings


def prog_from_program(record: protocol.ProgramRecord) -> list[int] | None:
    """Return the 168-slot Monday-first ``prog``, or None if any hour is unknown."""

    hourly = record.hourly_monday_first
    if any(slot is None for slot in hourly):
        return None
    return [int(slot) for slot in hourly]


def settings_from_power_request(request: protocol.PowerRequest) -> dict[str, Any]:
    """Return the canonical ``max_power`` carried by a heater power request, if any."""

    if request.measured_power_w is None:
        return {}
    return {"max_power": request.measured_power_w}


def settings_from_power_record(record: protocol.PowerRecord) -> dict[str, Any]:
    """Return heating ``state``, room temperature and active setpoint of a power record."""

    return {
        "state": "on" if record.heating else "off",
        "mtemp": _temperature(record.room_temp_c),
        "stemp": _temperature(record.setpoint_c),
    }


def validate_prog(prog: Sequence[Any]) -> list[int]:
    """Return ``prog`` as 168 slot codes 0/1/2, raising ValueError otherwise."""

    if isinstance(prog, (str, bytes)) or len(prog) != PROGRAM_LEN:
        raise ValueError(f"prog must be {PROGRAM_LEN} values of 0, 1 or 2")
    values: list[int] = []
    for value in prog:
        try:
            slot = int(value)
        except (TypeError, ValueError) as err:
            raise ValueError(f"prog contains non-integer value {value!r}") from err
        if slot not in protocol.SLOT_CODES:
            raise ValueError(f"prog values must be 0, 1 or 2; got {slot}")
        values.append(slot)
    return values


def _celsius(value: Any, label: str) -> float:
    """Return ``value`` as a finite float, raising ValueError otherwise."""

    try:
        number = float(value)
    except (TypeError, ValueError) as err:
        raise ValueError(f"invalid {label} {value!r}") from err
    if not math.isfinite(number):
        raise ValueError(f"invalid {label} {value!r}")
    return number


def _normalise_mode(mode: str | None) -> str | None:
    """Return a lower-case canonical mode string, or None."""

    return None if mode is None else str(mode).strip().lower()


def _mode_code(mode: str | None) -> int:
    """Return the radio mode byte for a canonical mode, raising ValueError otherwise."""

    mode = _normalise_mode(mode)
    if mode == OVERRIDE_MODE:
        raise ValueError("modified_auto needs a setpoint")
    code = MODE_TO_RADIO.get(mode or "")
    if code is None:
        raise ValueError(f"mode {mode!r} is not supported over radio")
    return code


def _presets(presets: Sequence[Any]) -> tuple[float, float, float]:
    """Return ``[cold, night, day]`` as three Celsius floats, raising ValueError."""

    if isinstance(presets, (str, bytes)) or len(presets) != 3:
        raise ValueError("ptemp must be three values [cold, night, day]")
    anti_frost, eco, comfort = (_celsius(value, "preset") for value in presets)
    return anti_frost, eco, comfort


def encode_preset_mode_write(
    status: protocol.StatusRecord,
    *,
    presets: Sequence[Any] | None = None,
    mode: str | None = None,
    setpoint: Any = None,
) -> bytes:
    """Return ``B6 <af> <eco> <comfort> <mode> [<setpoint>]``; gaps come from ``status``.

    A target temperature is the heater's own setpoint byte, written with mode
    manual, or with mode 03 (temporary override, which the heater ends at the
    next program change) for ``modified_auto``. The presets are not touched.
    """

    if presets is None:
        values = (status.anti_frost_c, status.eco_c, status.comfort_c)
    else:
        values = _presets(presets)
    target = None
    if setpoint is not None:
        normalised = _normalise_mode(mode)
        if normalised not in _SETPOINT_MODES:
            raise ValueError(f"a setpoint cannot be written together with mode {mode}")
        target = _celsius(setpoint, "setpoint")
        code = (
            protocol.MODE_OVERRIDE
            if normalised == OVERRIDE_MODE
            else protocol.MODE_MANUAL
        )
    elif mode is not None:
        code = _mode_code(mode)
    elif status.mode_code in protocol.PRESET_WRITE_MODES:
        code = status.mode_code
    else:
        raise ValueError(f"current mode {status.mode!r} cannot be re-sent with presets")
    return protocol.write_presets(*values, code, target)


def encode_command(command: BaseCommand, dialect: Dialect = DIALECT_A) -> bytes:
    """Return the radio payload for one canonical write command."""

    if isinstance(command, SetSetpoint):
        mode = _normalise_mode(command.mode)
        if mode not in _SETPOINT_MODES:
            raise ValueError(f"a setpoint cannot be written together with mode {mode}")
        celsius = _celsius(command.setpoint, "setpoint")
        if mode == OVERRIDE_MODE:
            return protocol.set_override(celsius)
        return protocol.set_setpoint(celsius)
    if isinstance(command, SetMode):
        return protocol.set_mode(_mode_code(command.mode))
    if isinstance(command, SetPresetTemps):
        return protocol.write_presets(*_presets(command.presets))
    if isinstance(command, SetProgram):
        prog = validate_prog(command.program)
        days = [
            prog[day * HOURS_PER_DAY : (day + 1) * HOURS_PER_DAY]
            for day in range(protocol.DAYS_PER_WEEK)
        ]
        return protocol.write_program(days, wire_slots=dialect.program_write_slots)
    if isinstance(command, SetLock):
        return protocol.set_toggle(protocol.TOGGLE_LOCK, bool(command.lock))
    raise TypeError(f"Unsupported radio command: {type(command).__name__}")


__all__ = [
    "MODE_FROM_RADIO",
    "MODE_TO_RADIO",
    "PROGRAM_LEN",
    "RADIO_UNITS",
    "encode_command",
    "encode_preset_mode_write",
    "prog_from_program",
    "settings_from_power_record",
    "settings_from_power_request",
    "settings_from_status",
    "validate_prog",
]
