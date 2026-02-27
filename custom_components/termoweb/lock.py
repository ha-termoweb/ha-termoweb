"""Home Assistant platform shim for lock entities."""

from __future__ import annotations

from .entities import lock as _lock
from .entities.lock import *  # noqa: F403

_iter_lockable_inventory_nodes = _lock._iter_lockable_inventory_nodes  # noqa: SLF001
_build_settings_resolver = _lock._build_settings_resolver  # noqa: SLF001
