"""Runtime container helpers for TermoWeb config entries."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
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


@dataclass(slots=True)
class EntryRuntime:
    """Runtime container for a configured TermoWeb entry."""

    backend: Backend
    client: HttpClientProto
    coordinator: StateCoordinator
    energy_coordinator: EnergyStateCoordinator
    dev_id: str
    inventory: Inventory
    config_entry: ConfigEntry
    base_poll_interval: int
    poll_suspended: bool = False
    poll_resume_unsub: Callable[[], None] | None = None
    ws_tasks: dict[str, asyncio.Task] = field(default_factory=dict)
    ws_clients: dict[str, WsClientProto] = field(default_factory=dict)
    ws_state: dict[str, Any] = field(default_factory=dict)
    ws_trackers: dict[str, Any] = field(default_factory=dict)
    version: str = ""
    brand: str = ""
    last_energy_import_summary: dict[str, Any] | None = None
    energy_import_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_radio_survey: dict[str, Any] | None = None
    last_radio_capture: dict[str, Any] | None = None
    recalc_poll: Callable[[], None] | None = None
    unsub_ws_status: Callable[[], None] | None = None
    _shutdown_complete: bool = False


type TermoWebConfigEntry = ConfigEntry[EntryRuntime]


def _live_runtime(entry: ConfigEntry | None) -> EntryRuntime | None:
    """Return the entry's runtime unless the entry is missing or was shut down."""

    runtime = getattr(entry, "runtime_data", None)
    if isinstance(runtime, EntryRuntime) and not runtime._shutdown_complete:  # noqa: SLF001
        return runtime
    return None


def _domain_entries(hass: HomeAssistant) -> list[ConfigEntry]:
    """Return the config entries of this integration."""

    return hass.config_entries.async_entries(DOMAIN)


def require_runtime(hass: HomeAssistant, entry_id: str) -> EntryRuntime:
    """Return the running runtime of ``entry_id``; raise LookupError otherwise."""

    for entry in _domain_entries(hass):
        if entry.entry_id == entry_id and (runtime := _live_runtime(entry)):
            return runtime
    raise LookupError("TermoWeb runtime data is unavailable")


def loaded_runtimes(hass: HomeAssistant) -> list[EntryRuntime]:
    """Return the runtimes of every running TermoWeb entry."""

    return [
        runtime
        for entry in _domain_entries(hass)
        if (runtime := _live_runtime(entry)) is not None
    ]


__all__ = ["EntryRuntime", "TermoWebConfigEntry", "loaded_runtimes", "require_runtime"]
