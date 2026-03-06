"""Backward-compatible exports for legacy child lock switch imports."""

from __future__ import annotations

from . import lock as _lock

_iter_lockable_inventory_nodes = _lock._iter_lockable_inventory_nodes  # noqa: SLF001
build_settings_resolver = _lock.build_settings_resolver
