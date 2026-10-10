"""Loaded radio entries for service and options-flow tests on real Home Assistant."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from types import SimpleNamespace
from typing import Any

from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.domain.ids import NodeId, NodeType
from custom_components.termoweb.domain.state import DomainStateStore, HeaterState
from custom_components.termoweb.domain.view import DomainStateView
from custom_components.termoweb.runtime import EntryRuntime
from tests_ha.fakes.runtime import build_entry_runtime


def heater_view(dev_id: str, states: Mapping[str, HeaterState]) -> DomainStateView:
    """Return a real domain view whose heaters ``states`` hold the given states."""
    store = DomainStateStore([NodeId(NodeType.HEATER, addr) for addr in states])
    for addr, state in states.items():
        store.replace_state(NodeType.HEATER, addr, state)
    return DomainStateView(dev_id, store)


def add_radio_entry(
    hass: HomeAssistant,
    *,
    client: Any,
    entry_id: str = "radio-entry",
    data: Mapping[str, Any] | None = None,
    options: Mapping[str, Any] | None = None,
    states: Mapping[str, HeaterState] | None = None,
    refresh: Callable[[Any], Awaitable[None]] | None = None,
    brand: str = "radio",
) -> EntryRuntime:
    """Add a radio config entry with a running runtime around ``client``."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        entry_id=entry_id,
        data=dict(data or {"brand": brand}),
        options=dict(options or {}),
    )
    entry.add_to_hass(hass)
    coordinator = SimpleNamespace(
        domain_view=heater_view("dev", states or {}),
        async_refresh_heater=refresh,
        data={},
    )
    return build_entry_runtime(
        hass=hass,
        entry_id=entry_id,
        client=client,
        config_entry=entry,
        coordinator=coordinator,
        brand=brand,
    )


def record_reloads(monkeypatch: Any, hass: HomeAssistant) -> list[str]:
    """Record scheduled entry reloads instead of setting the integration up."""
    reloads: list[str] = []
    monkeypatch.setattr(hass.config_entries, "async_schedule_reload", reloads.append)
    return reloads
