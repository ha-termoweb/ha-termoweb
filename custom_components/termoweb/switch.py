"""Home Assistant platform shim for switch entities."""

from __future__ import annotations

from .entities import switch as _switch
from .entities.switch import *  # noqa: F403

_iter_lockable_inventory_nodes = _switch._iter_lockable_inventory_nodes  # noqa: SLF001
_build_settings_resolver = _switch._build_settings_resolver  # noqa: SLF001
