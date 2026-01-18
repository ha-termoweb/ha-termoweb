"""Home Assistant platform shim for sensor entities."""

from __future__ import annotations

from .entities import sensor as _sensor
from .entities.sensor import *  # noqa: F401,F403

_normalise_energy_value = _sensor._normalise_energy_value
