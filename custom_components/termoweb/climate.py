"""Home Assistant platform shim for climate entities."""

from __future__ import annotations

from .entities import climate as _climate
from .entities.climate import *  # noqa: F403

_LOGGER = _climate._LOGGER  # noqa: SLF001
_WRITE_DEBOUNCE = _climate._WRITE_DEBOUNCE  # noqa: SLF001
_WS_ECHO_FALLBACK_REFRESH = _climate._WS_ECHO_FALLBACK_REFRESH  # noqa: SLF001
