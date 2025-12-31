"""Tests for websocket sample alias fallback behaviour."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from custom_components.termoweb.backend.ws_client import forward_ws_sample_updates
from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.inventory import Inventory


class CoordinatorStub:
    """Track websocket handler invocations for assertions."""

    def __init__(self) -> None:
        """Initialise the stub call recorder."""

        self.calls: list[tuple[str, dict[str, dict[str, Any]], float | None]] = []

    def handle_ws_samples(
        self,
        dev_id: str,
        updates: dict[str, dict[str, Any]],
        *,
        lease_seconds: float | None = None,
    ) -> None:
        """Record forwarded websocket sample payloads."""

        self.calls.append((dev_id, updates, lease_seconds))


def test_forward_ws_sample_updates_requires_energy_coordinator() -> None:
    """forward_ws_sample_updates should exit when the energy handler is absent."""

    entry_id = "entry"
    coordinator = CoordinatorStub()
    hass = SimpleNamespace(data={DOMAIN: {entry_id: {"coordinator": coordinator}}})

    forward_ws_sample_updates(
        hass,
        entry_id,
        "dev",
        {"heater": {"samples": {"1": {"temp": 23}}}},
    )

    assert coordinator.calls == []


def test_forward_ws_sample_updates_uses_alias_fallback_and_max_lease() -> None:
    """forward_ws_sample_updates should normalise aliases and pick the largest lease."""

    entry_id = "entry"
    hass = SimpleNamespace(data={DOMAIN: {entry_id: {}}})
    inventory = Inventory("dev", [])

    object.__setattr__(
        inventory,
        "_heater_sample_address_cache",
        ({"htr": ("1",)}, {"heater": "htr", "htr": "htr"}),
    )
    object.__setattr__(
        inventory,
        "_power_monitor_sample_address_cache",
        ({"pmo": ("7", "8")}, {"meter": "pmo", "pmo": "pmo"}),
    )

    coordinator = CoordinatorStub()
    hass.data[DOMAIN][entry_id] = {
        "inventory": inventory,
        "coordinator": SimpleNamespace(inventory=inventory),
        "energy_coordinator": coordinator,
    }

    forward_ws_sample_updates(
        hass,
        entry_id,
        "dev",
        {
            "heater": {
                "samples": {"1": {"temp": 23}},
                "lease_seconds": 30,
            },
            "pmo": {
                "samples": {"7": {"power": 180}},
                "lease_seconds": 45,
            },
            "power_monitor": {
                "samples": {"8": {"power": 200}},
                "lease_seconds": 120,
            },
        },
    )

    assert coordinator.calls == [
        (
            "dev",
            {
                "htr": {"1": {"temp": 23}},
                "pmo": {"7": {"power": 180}, "8": {"power": 200}},
            },
            120.0,
        )
    ]
