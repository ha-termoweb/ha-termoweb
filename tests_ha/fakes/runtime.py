"""Build an ``EntryRuntime`` attached to a real config entry."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.termoweb.backend.factory import backend_capabilities
from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.inventory import Inventory
from custom_components.termoweb.runtime import EntryRuntime


def build_entry_runtime(
    *,
    hass: HomeAssistant | None = None,
    entry_id: str = "entry",
    dev_id: str = "dev",
    inventory: Inventory | None = None,
    coordinator: Any | None = None,
    energy_coordinator: Any | None = None,
    client: Any | None = None,
    backend: Any | None = None,
    config_entry: Any | None = None,
    brand: str = "termoweb",
    version: str = "0.0.0",
    base_poll_interval: int = 30,
) -> EntryRuntime:
    """Return an ``EntryRuntime``; with ``hass``, attach it to a config entry.

    Collaborators that are not given are lightweight doubles, so a test only
    builds the pieces it exercises.
    """
    if inventory is None:
        inventory = getattr(coordinator, "inventory", None)
    if not isinstance(inventory, Inventory):
        inventory = Inventory(dev_id, [])
    if coordinator is None:
        coordinator = SimpleNamespace(inventory=inventory, data={})
    if energy_coordinator is None:
        energy_coordinator = SimpleNamespace(
            update_addresses=MagicMock(), handle_ws_samples=MagicMock()
        )
    if client is None:
        client = SimpleNamespace()
    if backend is None:
        backend = SimpleNamespace(
            client=client,
            brand=brand,
            capabilities=backend_capabilities(brand),
            diagnostics=lambda _entry_data: None,
            create_ws_client=MagicMock(),
            set_node_settings=AsyncMock(),
            set_acm_boost_state=AsyncMock(),
        )
    if config_entry is None:
        config_entry = MockConfigEntry(domain=DOMAIN, entry_id=entry_id, data={})
        if hass is not None:
            config_entry.add_to_hass(hass)

    runtime = EntryRuntime(
        backend=backend,
        client=client,
        coordinator=coordinator,
        energy_coordinator=energy_coordinator,
        dev_id=dev_id,
        inventory=inventory,
        config_entry=config_entry,
        base_poll_interval=base_poll_interval,
        version=version,
        brand=brand,
    )
    config_entry.runtime_data = runtime
    return runtime
