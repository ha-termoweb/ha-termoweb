"""Codec helpers for TermoWeb vendor interactions."""

from __future__ import annotations

from collections.abc import Mapping
import logging
from typing import Any

from pydantic import BaseModel, ValidationError

from custom_components.termoweb.backend.sanitize import validate_boost_minutes
from custom_components.termoweb.codecs.common import format_temperature, validate_units
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

from .termoweb_models import (
    AcmBoostWritePayload,
    AcmExtraOptionsWritePayload,
    DevListResponse,
    DevSummary,
    HeaterSettingsPayload,
    NodeSettingsWritePayload,
    NodesResponse,
    PowerMonitorPayload,
    SamplesResponse,
    ThermostatSettingsPayload,
)

_LOGGER = logging.getLogger(__name__)


def _validate_prog(prog: list[int]) -> list[int]:
    """Validate a weekly program sequence."""

    if not isinstance(prog, list) or len(prog) != 168:
        raise ValueError("prog must be a list of 168 integers (0, 1, or 2)")
    normalised: list[int] = []
    for value in prog:
        try:
            ivalue = int(value)
        except (TypeError, ValueError) as err:
            raise ValueError(f"prog contains non-integer value: {value!r}") from err
        if ivalue not in (0, 1, 2):
            raise ValueError(f"prog values must be 0, 1, or 2; got {ivalue}")
        normalised.append(ivalue)
    return normalised


def _validate_ptemp(ptemp: list[float | str]) -> list[str]:
    """Validate preset temperatures and return formatted strings."""

    if not isinstance(ptemp, list) or len(ptemp) != 3:
        raise ValueError(
            "ptemp must be a list of three numeric values [cold, night, day]"
        )
    formatted: list[str] = []
    for value in ptemp:
        try:
            formatted.append(format_temperature(value, label="temperature"))
        except ValueError as err:
            raise ValueError(f"ptemp contains non-numeric value: {value}") from err
    return formatted


def _normalise_mode(mode: str) -> str:
    """Lower-case and normalise heater modes."""

    mode_str = str(mode).lower()
    if mode_str == "heat":
        return "manual"
    return mode_str


def build_settings_payload(
    node_type: str, commands: list[BaseCommand]
) -> dict[str, Any]:
    """Encode node setting commands into a TermoWeb payload."""

    _ = node_type  # reserved for future branching on node type
    mode: str | None = None
    stemp: str | None = None
    prog: list[int] | None = None
    ptemp: list[str] | None = None
    units: str | None = None

    for command in commands:
        if isinstance(command, SetMode):
            mode = _normalise_mode(command.mode)
        elif isinstance(command, SetSetpoint):
            stemp = format_temperature(command.setpoint, label="stemp")
        elif isinstance(command, SetProgram):
            prog = _validate_prog(command.program)
        elif isinstance(command, SetPresetTemps):
            ptemp = _validate_ptemp(command.presets)
        elif isinstance(command, SetUnits):
            units = validate_units(command.units)
        else:
            raise TypeError(f"Unsupported command type: {type(command).__name__}")

    payload: dict[str, Any] = {}
    if mode is not None:
        payload["mode"] = mode
    if stemp is not None:
        payload["stemp"] = stemp
    if prog is not None:
        payload["prog"] = prog
    if ptemp is not None:
        payload["ptemp"] = ptemp
    if units is not None:
        payload["units"] = units

    model = NodeSettingsWritePayload.model_validate(payload)
    return model.model_dump(exclude_none=True)


def build_extra_options_payload(command: SetExtraOptions) -> dict[str, Any]:
    """Encode accumulator extra options for TermoWeb."""

    extra: dict[str, Any] = {}
    minutes = validate_boost_minutes(command.boost_time)
    if minutes is not None:
        extra["boost_time"] = minutes
    if command.boost_temp is not None:
        try:
            extra["boost_temp"] = format_temperature(
                command.boost_temp, label="boost_temp"
            )
        except ValueError as err:
            raise ValueError(
                f"Invalid boost_temp value: {command.boost_temp!r}"
            ) from err
    if not extra:
        raise ValueError("boost_time or boost_temp must be provided")

    payload = AcmExtraOptionsWritePayload.model_validate({"extra_options": extra})
    return payload.model_dump(exclude_none=True)


def build_boost_payload(command: AccumulatorCommand) -> dict[str, Any]:
    """Encode accumulator boost commands for TermoWeb."""

    boost_flag: bool
    stemp_value: str | float | None
    units_value: str | None
    minutes: int | None
    if isinstance(command, StartBoost):
        boost_flag = True
        stemp_value = command.stemp
        units_value = command.units
        minutes = validate_boost_minutes(command.boost_time)
    elif isinstance(command, StopBoost):
        boost_flag = False
        stemp_value = command.stemp
        units_value = command.units
        minutes = validate_boost_minutes(command.boost_time)
    else:  # pragma: no cover - defensive guard
        raise TypeError(f"Unsupported boost command: {type(command).__name__}")

    payload: dict[str, Any] = {"boost": boost_flag}
    if minutes is not None:
        payload["boost_time"] = minutes
    if stemp_value is not None:
        try:
            payload["stemp"] = format_temperature(stemp_value, label="stemp")
        except ValueError as err:
            raise ValueError(f"Invalid stemp value: {stemp_value!r}") from err
    if units_value is not None:
        payload["units"] = validate_units(units_value, trim=True)

    model = AcmBoostWritePayload.model_validate(payload)
    return model.model_dump(exclude_none=True)


def decode_devs_payload(raw: Any) -> list[dict[str, Any]]:
    """Validate and normalise a device list payload."""

    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]

    if isinstance(raw, dict):
        for key in ("devs", "devices"):
            value = raw.get(key)
            if isinstance(value, list):
                filtered = [item for item in value if isinstance(item, dict)]
                try:
                    return [
                        DevSummary.model_validate(item).model_dump(exclude_none=True)
                        for item in filtered
                    ]
                except ValidationError:
                    return filtered

        try:
            model = DevListResponse.model_validate(raw)
        except ValidationError:
            return []

        for field in ("devs", "devices"):
            value = getattr(model, field)
            if isinstance(value, list):
                return [item.model_dump(exclude_none=True) for item in value]

    return []


def decode_nodes_payload(raw: Any) -> Any:
    """Validate and normalise a nodes payload without changing semantics."""

    if not isinstance(raw, (dict, list)):
        return raw

    try:
        model = NodesResponse.model_validate(raw)
    except ValidationError:
        return raw

    return model.model_dump(by_alias=True, exclude_none=True)


def decode_node_settings(node_type: str, raw: Any) -> dict[str, Any]:
    """Validate and normalise node settings while preserving canonical keys only."""

    if not isinstance(raw, Mapping):
        return {}

    model_cls = HeaterSettingsPayload
    if node_type == "thm":
        model_cls = ThermostatSettingsPayload
    elif node_type == "pmo":
        model_cls = PowerMonitorPayload

    try:
        model = model_cls.model_validate(raw)
    except ValidationError:
        return canonicalize_settings_payload(raw) if isinstance(raw, Mapping) else {}

    validated = model.model_dump(exclude_none=True)
    return canonicalize_settings_payload(validated)


def decode_samples(
    raw: Any,
    *,
    timestamp_divisor: float = 1.0,
    logger: logging.Logger | None = None,
) -> list[dict[str, str | int]]:
    """Normalise heater samples payloads into {"t", "counter"} lists."""

    log = logger or _LOGGER
    items: list[Any] | None = None
    if isinstance(raw, dict) and isinstance(raw.get("samples"), list):
        try:
            model = SamplesResponse.model_validate(raw)
            items = [
                sample.model_dump(exclude_none=True)
                if isinstance(sample, BaseModel)
                else sample
                for sample in model.samples or []
            ]
        except ValidationError:
            items = raw["samples"]
    elif isinstance(raw, list):
        items = [
            sample.model_dump(exclude_none=True)
            if isinstance(sample, BaseModel)
            else sample
            for sample in raw
        ]

    if items is None:
        log.debug(
            "Unexpected htr samples payload (%s); returning empty list",
            type(raw).__name__,
        )
        return []

    samples: list[dict[str, str | int]] = []
    for item in items:
        if not isinstance(item, dict):
            log.debug("Unexpected htr sample item: %r", item)
            continue
        timestamp: Any = item.get("t")
        if timestamp is None:
            timestamp = item.get("timestamp")
        if not isinstance(timestamp, (int, float)):
            log.debug("Unexpected htr sample shape: %s", item)
            log.debug("Unexpected htr sample timestamp: %r", timestamp)
            continue

        counter_value: Any = item.get("counter")
        counter_min: Any = item.get("counter_min")
        counter_max: Any = item.get("counter_max")
        if isinstance(counter_value, dict):
            counter_min = counter_value.get("min", counter_min)
            counter_max = counter_value.get("max", counter_max)
            if "value" in counter_value:
                counter_value = counter_value.get("value")
            elif "counter" in counter_value:
                counter_value = counter_value.get("counter")
        if counter_value is None:
            counter_value = item.get("value")
        if counter_value is None:
            counter_value = item.get("energy")
        if counter_value is None:
            log.debug("Unexpected htr sample shape: %s", item)
            log.debug("Unexpected htr sample counter: %r", item)
            continue

        sample: dict[str, str | int] = {
            "t": int(float(timestamp) / timestamp_divisor),
            "counter": str(counter_value),
        }
        if counter_min is not None:
            sample["counter_min"] = str(counter_min)
        if counter_max is not None:
            sample["counter_max"] = str(counter_max)
        samples.append(sample)
    return samples
