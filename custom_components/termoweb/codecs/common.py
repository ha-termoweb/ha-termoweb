"""Shared codec validation helpers."""

from __future__ import annotations

from typing import Any

from custom_components.termoweb.coerce import as_float, as_int

PROG_SLOTS = 168
PTEMP_SLOTS = 3


def format_temperature(value: Any, *, label: str | None = None) -> str:
    """Format a finite numeric temperature as a one-decimal string or raise."""

    number = as_float(value)
    if number is None:
        field = "temperature" if label is None else label
        raise ValueError(f"Invalid {field} value: {value!r}")
    return f"{number:.1f}"


def safe_temperature(value: Any) -> str | None:
    """Format an inbound temperature; keep non-numeric text stripped, else None."""

    try:
        return format_temperature(value)
    except ValueError:
        if isinstance(value, str):
            return value.strip() or None
        return None


def validate_units(units: str | None, *, trim: bool = False) -> str:
    """Validate and normalise temperature units."""

    raw = "" if units is None else str(units)
    unit_value = raw.strip().upper() if trim else raw.upper()
    if unit_value not in {"C", "F"}:
        raise ValueError(f"Invalid units: {units!r}")
    return unit_value


def validate_prog(prog: Any) -> list[int]:
    """Return a 168-slot weekly program of 0/1/2 integers or raise ValueError."""

    if not isinstance(prog, list) or len(prog) != PROG_SLOTS:
        raise ValueError("prog must be a list of 168 integers (0, 1, or 2)")
    normalised: list[int] = []
    for value in prog:
        ivalue = as_int(value)
        if ivalue is None or ivalue != as_float(value):
            raise ValueError(f"prog contains non-integer value: {value!r}")
        if ivalue not in (0, 1, 2):
            raise ValueError(f"prog values must be 0, 1, or 2; got {ivalue}")
        normalised.append(ivalue)
    return normalised


def validate_ptemp(ptemp: Any) -> list[str]:
    """Return three formatted preset temperatures [cold, night, day] or raise."""

    if not isinstance(ptemp, list) or len(ptemp) != PTEMP_SLOTS:
        raise ValueError(
            "ptemp must be a list of three numeric values [cold, night, day]"
        )
    formatted: list[str] = []
    for value in ptemp:
        try:
            formatted.append(format_temperature(value))
        except ValueError as err:
            raise ValueError(f"ptemp contains non-numeric value: {value}") from err
    return formatted


__all__ = [
    "format_temperature",
    "safe_temperature",
    "validate_prog",
    "validate_ptemp",
    "validate_units",
]
