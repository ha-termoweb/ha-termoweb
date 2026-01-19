"""Home Assistant platform shim for binary_sensor entities."""

from __future__ import annotations

from .entities import binary_sensor as _binary_sensor
from .entities.binary_sensor import *  # noqa: F403

_iter_boostable_inventory_nodes = _binary_sensor._iter_boostable_inventory_nodes  # noqa: SLF001
