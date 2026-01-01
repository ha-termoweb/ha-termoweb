"""Codec helpers for the Ducaheat vendor interactions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from custom_components.termoweb.domain import canonicalize_settings_payload
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

from .common import format_temperature, validate_units
from .ducaheat_models import (
    BoostPayload,
    ExtraOptionsPayload,
    ModeWritePayload,
    SelectRequest,
    SetupPayload,
    StatusWritePayload,
)


def decode_settings(payload: Any, *, node_type: NodeType) -> dict[str, Any]:
    """Decode segmented settings payloads into canonical mappings."""

    if node_type is NodeType.THERMOSTAT:
        return _decode_thm_settings(payload)
    if node_type in {NodeType.HEATER, NodeType.ACCUMULATOR}:
        return _decode_status_payload(payload, node_type=node_type)
    return {} if not isinstance(payload, dict) else canonicalize_settings_payload(payload)


def _decode_status_payload(
    payload: Any, *, node_type: NodeType
) -> dict[str, Any]:
    """Normalise heater/accumulator settings payloads."""

    if not isinstance(payload, dict):
        return {}

    status = payload.get("status")
    status_dict = status if isinstance(status, dict) else {}
    setup = payload.get("setup")
    setup_dict = setup if isinstance(setup, dict) else {}
    prog = payload.get("prog")
    prog_temps = payload.get("prog_temps")

    flattened: dict[str, Any] = {}

    mode = status_dict.get("mode")
    if isinstance(mode, str):
        flattened["mode"] = mode.lower()

    state = (
        status_dict.get("state")
        or status_dict.get("heating_state")
        or status_dict.get("output_state")
    )
    if isinstance(state, str):
        flattened["state"] = state

    units = status_dict.get("units")
    if isinstance(units, str):
        flattened["units"] = units.upper()

    stemp = status_dict.get("stemp")
    if stemp is None:
        stemp = (
            status_dict.get("set_temp")
            or status_dict.get("target")
            or status_dict.get("setpoint")
        )
    if stemp is not None:
        formatted = _safe_temperature(stemp)
        if formatted is not None:
            flattened["stemp"] = formatted

    mtemp = (
        status_dict.get("mtemp")
        or status_dict.get("temp")
        or status_dict.get("ambient")
        or status_dict.get("room_temp")
    )
    if mtemp is not None:
        formatted = _safe_temperature(mtemp)
        if formatted is not None:
            flattened["mtemp"] = formatted

    include_boost = node_type is NodeType.ACCUMULATOR
    for extra_key in (
        "boost_active",
        "boost_remaining",
        "lock",
        "lock_active",
        "max_power",
    ):
        if not include_boost and extra_key.startswith("boost"):
            continue
        if extra_key in status_dict:
            flattened[extra_key] = status_dict[extra_key]

    if include_boost:
        _merge_boost_metadata(flattened, status_dict)
        _merge_accumulator_charge_metadata(flattened, status_dict)

        extra = setup_dict.get("extra_options")
        if isinstance(extra, dict):
            if "boost_time" in extra:
                flattened["boost_time"] = extra["boost_time"]
            if "boost_temp" in extra:
                formatted = _safe_temperature(extra["boost_temp"])
                if formatted is not None:
                    flattened["boost_temp"] = formatted

            _merge_boost_metadata(flattened, extra, prefer_existing=True)
            _merge_accumulator_charge_metadata(flattened, extra, prefer_existing=True)

        _merge_boost_metadata(flattened, setup_dict, prefer_existing=True)
        _merge_accumulator_charge_metadata(flattened, setup_dict, prefer_existing=True)

        if "boost_temp" not in flattened:
            boost_temp = status_dict.get("boost_temp")
            if boost_temp is not None:
                formatted = _safe_temperature(boost_temp)
                if formatted is not None:
                    flattened["boost_temp"] = formatted

        if "boost_time" not in flattened:
            boost_time = status_dict.get("boost_time")
            if boost_time is not None:
                flattened["boost_time"] = boost_time

    prog_list = _normalise_prog(prog)
    if prog_list is not None:
        flattened["prog"] = prog_list

    ptemp_list = _normalise_prog_temps(prog_temps)
    if ptemp_list is not None:
        flattened["ptemp"] = ptemp_list

    return canonicalize_settings_payload(flattened)


def _decode_thm_settings(payload: Any) -> dict[str, Any]:
    """Return a normalised thermostat settings mapping."""

    if not isinstance(payload, dict):
        return {}

    def _to_float(value: Any) -> float | None:
        """Return ``value`` coerced to float when possible."""

        if value is None:
            return None
        try:
            candidate = float(value)
        except (TypeError, ValueError):
            try:
                candidate = float(str(value).strip())
            except (TypeError, ValueError):
                return None
        return candidate

    normalised: dict[str, Any] = {}

    mode = payload.get("mode")
    if isinstance(mode, str):
        normalised["mode"] = mode.lower()

    state = payload.get("state")
    if isinstance(state, str):
        normalised["state"] = state.lower()

    stemp = _to_float(payload.get("stemp"))
    if stemp is not None:
        normalised["stemp"] = stemp

    mtemp = _to_float(payload.get("mtemp"))
    if mtemp is not None:
        normalised["mtemp"] = mtemp

    units = payload.get("units")
    if isinstance(units, str):
        normalised["units"] = units.upper()

    ptemp_raw = payload.get("ptemp")
    if isinstance(ptemp_raw, Mapping):
        cleaned = _normalise_prog_temps(ptemp_raw)
        if cleaned is not None:
            normalised["ptemp"] = cleaned
    elif isinstance(ptemp_raw, list):
        cleaned_list: list[float] = []
        for value in ptemp_raw:
            converted = _to_float(value)
            if converted is None:
                break
            cleaned_list.append(converted)
        if len(cleaned_list) == len(ptemp_raw):
            normalised["ptemp"] = cleaned_list

    prog = _normalise_prog(payload.get("prog"))
    if prog is not None:
        normalised["prog"] = prog

    batt_level = payload.get("batt_level")
    try:
        batt_value = int(batt_level)
    except (TypeError, ValueError):
        batt_value = None
    if batt_value is not None:
        normalised["batt_level"] = max(0, min(5, batt_value))

    return canonicalize_settings_payload(normalised)


def _merge_boost_metadata(
    target: dict[str, Any],
    source: Mapping[str, Any] | None,
    *,
    prefer_existing: bool = False,
) -> None:
    """Copy boost metadata from ``source`` into ``target`` safely."""

    if not isinstance(source, Mapping):
        return

    def _assign(
        key: str,
        value: Any,
        *,
        prefer: bool | None = None,
        allow_none: bool = False,
    ) -> None:
        """Assign a metadata value while respecting preference rules."""

        if value is None and not allow_none:
            return

        prefer_flag = prefer_existing if prefer is None else prefer
        if prefer_flag and key in target and target[key] is not None:
            return
        if prefer_flag and key in target and target[key] is None and value is None:
            return

        target[key] = value

    for key in ("boost", "boost_end_day", "boost_end_min"):
        if key in source:
            _assign(key, source[key])

    if "boost_end" in source:
        boost_end = source["boost_end"]
        if isinstance(boost_end, Mapping):
            _assign("boost_end_day", boost_end.get("day"), prefer=True)
            _assign("boost_end_min", boost_end.get("minute"), prefer=True)


def _merge_accumulator_charge_metadata(
    target: dict[str, Any],
    source: Mapping[str, Any] | None,
    *,
    prefer_existing: bool = False,
) -> None:
    """Copy accumulator charge metadata from ``source`` into ``target``."""

    if not isinstance(source, Mapping):
        return

    def _should_assign(key: str) -> bool:
        if not prefer_existing:
            return True
        if key not in target:
            return True
        return target[key] is None

    charging_value = source.get("charging")
    if charging_value is not None and _should_assign("charging"):
        try:
            target["charging"] = bool(charging_value)
        except (TypeError, ValueError):
            target["charging"] = bool(charging_value)

    for key in ("current_charge_per", "target_charge_per"):
        if not _should_assign(key):
            continue
        coerced = source.get(key)
        try:
            if coerced is None:
                continue
            value = int(coerced)
        except (TypeError, ValueError):
            continue
        target[key] = max(0, min(100, value))


def _normalise_prog(data: Any) -> list[int] | None:
    """Convert vendor programme payloads into a 168-slot list."""

    if isinstance(data, list):
        try:
            values = [int(v) for v in data]
        except (TypeError, ValueError):
            return None
        if len(values) == 168:
            if all(v in (0, 1, 2) for v in values):
                return values
            return None

    if not isinstance(data, Mapping):
        return None

    days_section: dict[str, Any] | None = None
    if isinstance(data.get("days"), dict):
        days_section = data["days"]
    else:
        candidate = {
            k: v
            for k, v in data.items()
            if str(k) in {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}
            or str(k).isdigit()
        }
        if candidate:
            days_section = candidate

    if days_section is None and isinstance(data.get("prog"), dict):
        days_section = data["prog"]

    if not isinstance(days_section, Mapping):
        return None

    day_order = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")

    def _coerce_slots(entry: Any) -> list[int] | None:
        """Convert a day's slots into a 24-value list."""

        candidate = entry
        if isinstance(candidate, Mapping):
            slots_candidate = candidate.get("slots")
            values_candidate = candidate.get("values")
            if isinstance(slots_candidate, list):
                candidate = slots_candidate
            else:
                candidate = values_candidate

        if candidate is None:
            return None

        if not isinstance(candidate, list):
            return None

        try:
            values = [int(v) for v in candidate]
        except (TypeError, ValueError):
            return None

        if len(values) == 48:
            values = [max(values[i : i + 2]) for i in range(0, 48, 2)]

        if len(values) < 24:
            values = values + [0] * (24 - len(values))
        if len(values) > 24:
            values = values[:24]

        return values if len(values) == 24 else None

    values: list[int] = []
    for idx, day in enumerate(day_order):
        entry = days_section.get(day)
        if entry is None:
            entry = days_section.get(str(idx))
        if entry is None:
            entry = days_section.get(idx)

        if entry is None:
            day_values = [0] * 24
        else:
            day_values = _coerce_slots(entry)
            if day_values is None:
                return None
        values.extend(day_values)

    return values if len(values) == 168 else None


def _normalise_prog_temps(data: Any) -> list[str] | None:
    """Convert preset temperature payloads into stringified list."""

    if not isinstance(data, Mapping):
        return None
    antifrost = data.get("antifrost") or data.get("cold")
    eco = data.get("eco") or data.get("night")
    comfort = data.get("comfort") or data.get("day")
    temps = [antifrost, eco, comfort]
    formatted: list[str] = []
    for value in temps:
        if value is None:
            formatted.append("")
            continue
        safe = _safe_temperature(value)
        if safe is None:
            formatted.append(str(value))
        else:
            formatted.append(safe)
    return formatted


def _safe_temperature(value: Any) -> str | None:
    """Defensively format inbound temperature values."""

    if value is None:
        return None
    try:
        return format_temperature(value)
    except ValueError:
        if isinstance(value, str):
            cleaned = value.strip()
            return cleaned or None
        return None


def encode_select_payload(select: bool) -> dict[str, Any]:
    """Encode a selection payload."""

    return SelectRequest.model_validate({"select": select}).model_dump()


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
