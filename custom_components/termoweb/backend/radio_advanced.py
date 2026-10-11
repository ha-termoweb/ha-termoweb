"""The advanced setup (``C4``) last written to each radio heater.

The heater cannot report these settings back, so the record a write sent is
the only source of the current values. It lives in the config entry options
(``radio_advanced``: ``{"<addr>": {<C4 field>: int, ...}}``) through the
``load``/``save`` callbacks, like the power manager's settings.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, fields
import logging
from typing import Any

from .radio.protocol import AdvancedSetup

_LOGGER = logging.getLogger(__name__)
_FIELDS = frozenset(field.name for field in fields(AdvancedSetup))


class AdvancedSetupStore:
    """Per-heater ``C4`` records, keyed by radio address."""

    def __init__(
        self,
        load: Callable[[], Mapping[str, Any] | None] | None = None,
        save: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Read through ``load`` on every use; ``save`` stores a changed mapping."""

        self._memory: dict[str, Any] = {}
        self._load = load or (lambda: self._memory)
        self._save = save or self._replace_memory

    def _replace_memory(self, records: dict[str, Any]) -> None:
        """Keep ``records`` in memory (no config entry)."""

        self._memory = records

    def _records(self) -> dict[str, Any]:
        """Return a copy of the stored mapping."""

        return dict(self._load() or {})

    def get(self, addr: int) -> AdvancedSetup | None:
        """Return the record last written to ``addr``, or None (never written)."""

        raw = self._records().get(str(addr))
        if not isinstance(raw, Mapping) or set(raw) != _FIELDS:
            return None
        try:
            return AdvancedSetup(**{key: int(value) for key, value in raw.items()})
        except TypeError, ValueError:
            _LOGGER.debug("Ignoring a bad stored advanced setup for heater %s", addr)
            return None

    def put(self, addr: int, setup: AdvancedSetup) -> None:
        """Store the record just written to ``addr``."""

        self._save({**self._records(), str(addr): asdict(setup)})

    def forget(self, addr: int) -> None:
        """Drop the record of ``addr`` (the heater was reset or replaced)."""

        records = self._records()
        if records.pop(str(addr), None) is not None:
            self._save(records)


__all__ = ["AdvancedSetupStore"]
