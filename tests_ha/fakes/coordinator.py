"""Build real state/energy coordinators around a scripted REST client."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from homeassistant.core import HomeAssistant
import pytest

from custom_components.termoweb import coordinator as coord_module
from custom_components.termoweb.backend.rest_client import RESTClient
from custom_components.termoweb.coordinator import (
    EnergyStateCoordinator,
    StateCoordinator,
)
from custom_components.termoweb.domain.state import state_to_dict
from custom_components.termoweb.inventory import Inventory, build_node_inventory

DEV_ID = "0123456789abcdef"


class Clock:
    """Fake wall and monotonic clock for the coordinator module."""

    def __init__(self, now: float = 0.0) -> None:
        """Start the clock at ``now`` seconds."""
        self.now = now

    def __call__(self) -> float:
        """Return the current fake time."""
        return self.now

    def install(self, monkeypatch: pytest.MonkeyPatch) -> Clock:
        """Drive ``time.time`` and ``time.monotonic`` as seen by the coordinator."""
        monkeypatch.setattr(coord_module, "time", SimpleNamespace(time=self))
        monkeypatch.setattr(coord_module, "time_mod", self)
        return self


def rest_client(**methods: Any) -> AsyncMock:
    """Return a REST client double; async methods are ``AsyncMock``s.

    ``get_power_limit`` answers ``None`` (no limit) unless overridden.
    """
    client = AsyncMock(spec=RESTClient)
    client.get_power_limit.return_value = None
    for name, value in methods.items():
        setattr(client, name, value)
    return client


def inventory(nodes: Mapping[str, Iterable[str]], dev_id: str = DEV_ID) -> Inventory:
    """Return an inventory with the given ``{node_type: [addr, ...]}`` nodes."""
    payload = {
        "nodes": [
            {"type": node_type, "addr": addr}
            for node_type, addrs in nodes.items()
            for addr in addrs
        ]
    }
    return Inventory(dev_id, list(build_node_inventory(payload)))


def state_coordinator(
    hass: HomeAssistant,
    client: Any,
    nodes: Mapping[str, Iterable[str]],
    *,
    device: Mapping[str, Any] | None = None,
    base_interval: int = 30,
    brand: str = "termoweb",
) -> StateCoordinator:
    """Return a real ``StateCoordinator`` for ``nodes`` on ``DEV_ID``."""
    return StateCoordinator(
        hass,
        client,
        base_interval,
        DEV_ID,
        device if device is not None else {"name": "Home"},
        inventory(nodes),
        brand=brand,
    )


def energy_coordinator(
    hass: HomeAssistant,
    client: Any,
    nodes: Mapping[str, Iterable[str]],
    *,
    state: StateCoordinator | None = None,
) -> EnergyStateCoordinator:
    """Return a real ``EnergyStateCoordinator`` for ``nodes`` on ``DEV_ID``."""
    return EnergyStateCoordinator(
        hass, client, DEV_ID, inventory(nodes), state_coordinator=state
    )


def node_state(coord: StateCoordinator, node_type: str, addr: str) -> dict | None:
    """Return the stored state of a node as a dict, or ``None``."""
    state = coord.domain_view.get_heater_state(node_type, addr)
    return None if state is None else state_to_dict(state)


def energy(coord: EnergyStateCoordinator, node_type: str, addr: str) -> float | None:
    """Return the published energy (kWh) of a node."""
    metric = coord.data.metrics_for_type(node_type).get(addr)
    return None if metric is None else metric.energy_kwh


def power(coord: EnergyStateCoordinator, node_type: str, addr: str) -> float | None:
    """Return the published derived power (W) of a node."""
    metric = coord.data.metrics_for_type(node_type).get(addr)
    return None if metric is None else metric.power_w
