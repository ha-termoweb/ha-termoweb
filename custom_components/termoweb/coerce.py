"""Lenient value coercion shared by the domain, entities and read codecs.

One semantics per helper; every helper returns ``None`` instead of raising:

- ``as_float``: finite ``float`` from an int, float or numeric string
  (surrounding whitespace allowed). ``bool``, ``nan``/``inf``, blanks and
  anything else give ``None``.
- ``as_number``: like ``as_float`` but an ``int`` stays an ``int`` and a
  ``float`` stays a ``float``; strings become ``float``.
- ``as_int``: ``int(as_float(value))``, truncating toward zero, so
  ``"21.5"`` and ``21.5`` both give ``21``.
- ``as_bool``: ``bool`` as is; ``1``/``0`` (int or float); the strings
  true/false, 1/0, yes/no, on/off (case-insensitive). Any other value,
  including ``2``, gives ``None``.
- ``as_percentage``: ``as_int`` clamped to 0..100.
"""

from __future__ import annotations

import math
from typing import Any

_TRUE_STRINGS = frozenset({"true", "1", "yes", "on"})
_FALSE_STRINGS = frozenset({"false", "0", "no", "off"})


def as_number(value: Any) -> int | float | None:
    """Return a finite int/float (ints kept as int, strings as float) or None."""

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if not isinstance(value, str):
        return None
    try:
        number = float(value.strip())
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def as_float(value: Any) -> float | None:
    """Return ``value`` as a finite float, or None."""

    number = as_number(value)
    return None if number is None else float(number)


def as_int(value: Any) -> int | None:
    """Return ``value`` as an int truncated toward zero, or None."""

    number = as_number(value)
    return None if number is None else int(number)


def as_bool(value: Any) -> bool | None:
    """Return ``value`` as a bool from bool, 1/0 or common strings, or None."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE_STRINGS:
            return True
        if text in _FALSE_STRINGS:
            return False
    return None


def as_percentage(value: Any) -> int | None:
    """Return ``as_int(value)`` clamped to 0..100, or None."""

    number = as_int(value)
    return None if number is None else max(0, min(100, number))


__all__ = ["as_bool", "as_float", "as_int", "as_number", "as_percentage"]
