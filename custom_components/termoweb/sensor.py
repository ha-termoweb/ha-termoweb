"""Home Assistant platform shim for sensor entities."""

from __future__ import annotations

from .entities import sensor as _sensor
from .entities.sensor import *  # noqa: F403

_create_boost_sensors = _sensor._create_boost_sensors  # noqa: SLF001
_create_heater_sensors = _sensor._create_heater_sensors  # noqa: SLF001
_normalise_energy_value = _sensor._normalise_energy_value  # noqa: SLF001
