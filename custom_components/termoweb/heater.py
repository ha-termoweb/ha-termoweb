"""Home Assistant platform shim for heater entities."""

from __future__ import annotations

from .entities import heater as _heater
from .entities.heater import *  # noqa: F403

_boost_runtime_store = _heater._boost_runtime_store  # noqa: SLF001
