"""Utility helpers for TermoWeb integration."""

from __future__ import annotations

from typing import Any


def float_or_none(value: Any) -> float | None:
    """Return value as ``float`` if possible, else ``None``.

    Converts integers, floats, and numeric strings to ``float`` while safely
    handling ``None`` and non-numeric inputs.
    """
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        string_val = str(value).strip()
        return float(string_val) if string_val else None
    except Exception:
        return None
