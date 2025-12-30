"""Codec helpers for TermoWeb vendor interactions."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from .termoweb_models import DevListResponse, DevSummary, NodesResponse


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
