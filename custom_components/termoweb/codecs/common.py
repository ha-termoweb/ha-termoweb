"""Shared codec validation helpers."""

from __future__ import annotations

from typing import Any


def format_temperature(value: Any, *, label: str | None = None) -> str:
    """Format numeric temperatures to a one-decimal string."""

    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError) as err:
        field = "temperature" if label is None else label
        raise ValueError(f"Invalid {field} value: {value!r}") from err


def validate_units(units: str | None, *, trim: bool = False) -> str:
    """Validate and normalise temperature units."""

    raw = "" if units is None else str(units)
    unit_value = raw.strip().upper() if trim else raw.upper()
    if unit_value not in {"C", "F"}:
        raise ValueError(f"Invalid units: {units!r}")
    return unit_value


__all__ = ["format_temperature", "validate_units"]
