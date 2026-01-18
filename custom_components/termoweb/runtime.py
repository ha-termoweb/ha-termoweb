"""Runtime container helpers for TermoWeb config entries."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator, MutableMapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import DOMAIN
from .inventory import Inventory

if TYPE_CHECKING:
    from .backend import Backend, WsClientProto
    from .backend.base import HttpClientProto
    from .coordinator import EnergyStateCoordinator, StateCoordinator
    from .hourly_poller import HourlySamplesPoller


@dataclass(slots=True)
class EntryRuntime(MutableMapping[str, Any]):
    """Runtime container for a configured TermoWeb entry."""

    backend: "Backend"
    client: "HttpClientProto"
    coordinator: "StateCoordinator"
    energy_coordinator: "EnergyStateCoordinator"
    dev_id: str
    inventory: Inventory
    hourly_poller: "HourlySamplesPoller"
    config_entry: ConfigEntry
    base_poll_interval: int
    stretched: bool = False
    poll_suspended: bool = False
    poll_resume_unsub: Callable[[], None] | None = None
    ws_tasks: dict[str, asyncio.Task] = field(default_factory=dict)
    ws_clients: dict[str, "WsClientProto"] = field(default_factory=dict)
    ws_state: dict[str, Any] = field(default_factory=dict)
    ws_trackers: dict[str, Any] = field(default_factory=dict)
    version: str = ""
    brand: str = ""
    debug: bool = False
    boost_runtime: dict[str, dict[str, int]] = field(default_factory=dict)
    boost_temperature: dict[str, dict[str, float]] = field(default_factory=dict)
    climate_entities: dict[str, dict[str, str]] = field(default_factory=dict)
    diagnostics_task: asyncio.Task | None = None
    fallback_translations: dict[str, str] | None = None
    recalc_poll: Callable[[], None] | None = None
    start_ws: Callable[[str], Awaitable[None]] | None = None
    unsub_ws_status: Callable[[], None] | None = None
    _shutdown_complete: bool = False
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    def _field_keys(self) -> tuple[str, ...]:
        """Return the public runtime fields used for mapping access."""

        return tuple(
            key
            for key in self.__dataclass_fields__
            if key not in {"extra"} and not key.startswith("_")
        )

    def __getitem__(self, key: str) -> Any:
        """Return runtime attributes using mapping-style access."""

        if key in self.__dataclass_fields__:
            return getattr(self, key)
        return self.extra[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Store runtime attributes using mapping-style access."""

        if key in self.__dataclass_fields__:
            setattr(self, key, value)
            return
        self.extra[key] = value

    def __delitem__(self, key: str) -> None:
        """Delete runtime attributes using mapping-style access."""

        if key in self.__dataclass_fields__:
            raise KeyError(f"Cannot delete runtime field {key}")
        del self.extra[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over runtime keys exposed via mapping semantics."""

        return iter(self._field_keys() + tuple(self.extra))

    def __len__(self) -> int:
        """Return the number of keys exposed via mapping semantics."""

        return len(self._field_keys()) + len(self.extra)

    def get(self, key: str, default: Any = None) -> Any:
        """Return a runtime attribute or ``default``."""

        if key in self.__dataclass_fields__:
            return getattr(self, key)
        return self.extra.get(key, default)


def require_runtime(hass: HomeAssistant, entry_id: str) -> EntryRuntime:
    """Return the runtime container stored for ``entry_id``."""

    domain_data = hass.data.get(DOMAIN)
    if not isinstance(domain_data, dict):
        raise LookupError("TermoWeb runtime data is unavailable")
    runtime = domain_data.get(entry_id)
    if not isinstance(runtime, EntryRuntime):
        raise LookupError("TermoWeb runtime data is unavailable")
    return runtime


__all__ = ["EntryRuntime", "require_runtime"]
