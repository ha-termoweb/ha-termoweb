"""Codec helpers for TermoWeb vendor interactions."""

from __future__ import annotations

from collections.abc import Mapping
import logging
from typing import Any

from pydantic import BaseModel, ValidationError

from custom_components.termoweb.boost import validate_boost_minutes
from custom_components.termoweb.codecs.common import (
    format_temperature,
    validate_prog,
    validate_ptemp,
    validate_units,
)
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
    DevSummary,
    HeaterSettingsPayload,
    NodeSettingsWritePayload,
    NodesResponse,
    PowerMonitorPayload,
    SamplesResponse,
    ThermostatSettingsPayload,
)

_LOGGER = logging.getLogger(__name__)


def _normalise_mode(mode: str) -> str:
    """Lower-case and normalise heater modes."""

    mode_str = str(mode).strip().lower()
    if mode_str == "heat":
        return "manual"
    return mode_str


def build_settings_payload(commands: list[BaseCommand]) -> dict[str, Any]:
    """Encode node setting commands into a TermoWeb payload."""

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
            prog = validate_prog(command.program)
        elif isinstance(command, SetPresetTemps):
            ptemp = validate_ptemp(command.presets)
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

    if not isinstance(command, (StartBoost, StopBoost)):  # pragma: no cover - defensive guard
        raise TypeError(f"Unsupported boost command: {type(command).__name__}")

    boost_flag = isinstance(command, StartBoost)
    minutes = validate_boost_minutes(command.boost_time)

    payload: dict[str, Any] = {"boost": boost_flag}
    if minutes is not None:
        payload["boost_time"] = minutes
    if command.stemp is not None:
        try:
            payload["stemp"] = format_temperature(command.stemp, label="stemp")
        except ValueError as err:
            raise ValueError(f"Invalid stemp value: {command.stemp!r}") from err
    if command.units is not None:
        payload["units"] = validate_units(command.units, trim=True)

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

    return []


def decode_nodes_payload(raw: Any) -> Any:
    """Validate and normalise a nodes payload without changing semantics."""

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
        result = canonicalize_settings_payload(raw)
        stage = "settings (fallback)"
    else:
        result = canonicalize_settings_payload(model.model_dump(exclude_none=True))
        stage = "settings"

    all_raw = set(raw.keys())
    if isinstance(raw.get("status"), Mapping):
        all_raw |= set(raw["status"].keys())
    dropped = all_raw - set(result.keys()) - {"status", "raw"}
    if dropped:
        _LOGGER.debug(
            "Undecoded fields in %s %s: %s", node_type, stage, sorted(dropped)
        )

    return result


_MILLISECOND_EPOCH_THRESHOLD = 1_000_000_000_000


def decode_samples(
    raw: Any,
    *,
    logger: logging.Logger | None = None,
) -> list[dict[str, str | int]]:
    """Normalise samples payloads into {"t", "counter"} lists with second timestamps."""

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
            counter_value = counter_value.get("value", counter_value.get("counter"))
        if counter_value is None:
            counter_value = item.get("value")
        if counter_value is None:
            counter_value = item.get("energy")
        if counter_value is None:
            log.debug("Unexpected htr sample shape: %s", item)
            log.debug("Unexpected htr sample counter: %r", item)
            continue

        seconds = float(timestamp)
        if abs(seconds) >= _MILLISECOND_EPOCH_THRESHOLD:
            seconds /= 1000.0
        sample: dict[str, str | int] = {
            "t": int(seconds),
            "counter": str(counter_value),
        }
        if counter_min is not None:
            sample["counter_min"] = str(counter_min)
        if counter_max is not None:
            sample["counter_max"] = str(counter_max)
        samples.append(sample)
    return samples
