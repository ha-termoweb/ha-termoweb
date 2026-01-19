"""Runtime container helpers for TermoWeb config entries."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator, Mapping, MutableMapping
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

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
        raise LookupError("TermoWeb runtime data is unavailable")  # noqa: TRY004
    runtime = domain_data.get(entry_id)
    if isinstance(runtime, EntryRuntime):
        return runtime
    if isinstance(runtime, Mapping):
        record = runtime
        if not any(
            key in record
            for key in (
                "dev_id",
                "coordinator",
                "inventory",
                "client",
                "backend",
                "energy_coordinator",
                "hourly_poller",
                "config_entry",
                "brand",
                "version",
                "base_poll_interval",
                "base_interval",
            )
        ):
            raise LookupError("TermoWeb runtime data is unavailable")
        dev_id = str(record.get("dev_id") or entry_id)
        coordinator = record.get("coordinator") or SimpleNamespace()
        inventory = record.get("inventory")
        if not isinstance(inventory, Inventory):
            coordinator_inventory = getattr(coordinator, "inventory", None)
            if isinstance(coordinator_inventory, Inventory):
                inventory = coordinator_inventory
        energy_coordinator = record.get("energy_coordinator")
        if energy_coordinator is None:
            energy_coordinator = SimpleNamespace(
                update_addresses=lambda *_args, **_kwargs: None,
                handle_ws_samples=lambda *_args, **_kwargs: None,
            )
        client = record.get("client") or SimpleNamespace()
        backend = record.get("backend")
        if backend is None:
            backend = SimpleNamespace(
                client=client,
                brand=str(record.get("brand") or ""),
                create_ws_client=lambda *_args, **_kwargs: None,
                set_node_settings=AsyncMock(),
            )
        hourly_poller = record.get("hourly_poller")
        if hourly_poller is None:
            hourly_poller = SimpleNamespace(
                async_shutdown=lambda: asyncio.sleep(0),
            )
        config_entry = record.get("config_entry")
        if config_entry is None:
            config_entry = SimpleNamespace(entry_id=entry_id, data={}, options={})
        base_poll_interval = record.get("base_poll_interval")
        if base_poll_interval is None:
            base_poll_interval = record.get("base_interval", 0)
        runtime = EntryRuntime(
            backend=backend,
            client=client,
            coordinator=coordinator,
            energy_coordinator=energy_coordinator,
            dev_id=dev_id,
            inventory=inventory,
            hourly_poller=hourly_poller,
            config_entry=config_entry,
            base_poll_interval=int(base_poll_interval or 0),
            version=str(record.get("version") or ""),
            brand=str(record.get("brand") or ""),
        )
        for key, value in record.items():
            if hasattr(runtime, key):
                setattr(runtime, key, value)
        domain_data[entry_id] = runtime
        return runtime
    raise LookupError("TermoWeb runtime data is unavailable")


__all__ = ["EntryRuntime", "require_runtime"]
