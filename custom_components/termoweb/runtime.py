"""Runtime container helpers for TermoWeb config entries."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

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
class EntryRuntime:
    """Runtime container for a configured TermoWeb entry."""

    backend: Backend
    client: HttpClientProto
    coordinator: StateCoordinator
    energy_coordinator: EnergyStateCoordinator
    dev_id: str
    inventory: Inventory
    hourly_poller: HourlySamplesPoller
    config_entry: ConfigEntry
    base_poll_interval: int
    stretched: bool = False
    poll_suspended: bool = False
    poll_resume_unsub: Callable[[], None] | None = None
    ws_tasks: dict[str, asyncio.Task] = field(default_factory=dict)
    ws_clients: dict[str, WsClientProto] = field(default_factory=dict)
    ws_state: dict[str, Any] = field(default_factory=dict)
    ws_trackers: dict[str, Any] = field(default_factory=dict)
    version: str = ""
    brand: str = ""
    debug: bool = False
    last_energy_import_summary: dict[str, Any] | None = None
    boost_runtime: dict[str, dict[str, int]] = field(default_factory=dict)
    boost_temperature: dict[str, dict[str, float]] = field(default_factory=dict)
    climate_entities: dict[str, dict[str, str]] = field(default_factory=dict)
    diagnostics_task: asyncio.Task | None = None
    fallback_translations: dict[str, str] | None = None
    recalc_poll: Callable[[], None] | None = None
    start_ws: Callable[[str], Awaitable[None]] | None = None
    unsub_ws_status: Callable[[], None] | None = None
    _shutdown_complete: bool = False


def require_runtime(hass: HomeAssistant, entry_id: str) -> EntryRuntime:
    """Return the runtime container stored for ``entry_id``."""

    hass_data = getattr(hass, "data", None)
    if not isinstance(hass_data, dict):
        raise LookupError("TermoWeb runtime data is unavailable")  # noqa: TRY004
    domain_data = hass_data.get(DOMAIN)
    if not isinstance(domain_data, dict):
        raise LookupError("TermoWeb runtime data is unavailable")  # noqa: TRY004
    runtime = domain_data.get(entry_id)
    if isinstance(runtime, EntryRuntime):
        return runtime
    if runtime is not None and all(
        hasattr(runtime, key)
        for key in (
            "backend",
            "client",
            "coordinator",
            "energy_coordinator",
            "dev_id",
            "inventory",
            "hourly_poller",
            "config_entry",
            "base_poll_interval",
            "version",
            "brand",
            "ws_tasks",
            "ws_clients",
            "ws_state",
            "ws_trackers",
        )
    ):
        return cast(EntryRuntime, runtime)
    raise LookupError("TermoWeb runtime data is unavailable")


__all__ = ["EntryRuntime", "require_runtime"]
