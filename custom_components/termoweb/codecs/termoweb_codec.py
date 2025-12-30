"""Codec helpers for TermoWeb vendor interactions."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ValidationError

from .termoweb_models import (
    DevListResponse,
    DevSummary,
    HeaterSettingsPayload,
    NodesResponse,
    PowerMonitorPayload,
    SamplesResponse,
    ThermostatSettingsPayload,
)

_LOGGER = logging.getLogger(__name__)


def decode_payload(payload: Any) -> Any:
    """Decode a TermoWeb payload."""

    raise NotImplementedError


def encode_payload(data: Any) -> Any:
    """Encode data for TermoWeb payloads."""

    raise NotImplementedError


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
    """Validate and normalise node settings while preserving legacy keys."""

    if not isinstance(raw, dict):
        return raw if raw is not None else {}

    model_cls = HeaterSettingsPayload
    if node_type == "thm":
        model_cls = ThermostatSettingsPayload
    elif node_type == "pmo":
        model_cls = PowerMonitorPayload

    try:
        model = model_cls.model_validate(raw)
        validated = model.model_dump(exclude_none=True)
    except ValidationError:
        return dict(raw)

    if node_type in {"htr", "acm"}:
        for key in ("mode", "stemp", "mtemp", "temp", "ptemp", "prog"):
            if key in raw:
                validated.setdefault(key, raw.get(key))

    status_raw = raw.get("status")
    if node_type in {"htr", "acm", "thm"} and isinstance(status_raw, dict):
        status_validated = validated.setdefault("status", {})
        if isinstance(status_validated, dict):
            for key in ("stemp", "mtemp", "temp", "mode", "ptemp", "prog"):
                if key in status_raw:
                    status_validated.setdefault(key, status_raw.get(key))
    return validated


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
