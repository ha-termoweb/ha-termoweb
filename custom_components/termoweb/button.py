"""Home Assistant platform shim for button entities."""

from __future__ import annotations

from .entities import button as _button
from .entities.button import *  # noqa: F403

_SERVICE_REQUEST_ACCUMULATOR_BOOST = _button._SERVICE_REQUEST_ACCUMULATOR_BOOST  # noqa: SLF001
_iter_accumulator_contexts = _button._iter_accumulator_contexts  # noqa: SLF001
