"""Tests for websocket sample alias merging."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from conftest import build_entry_runtime
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


def test_forward_ws_sample_updates_merges_inventory_aliases() -> None:
    """forward_ws_sample_updates should canonicalise inventory alias maps."""

    entry_id = "entry"
    hass = SimpleNamespace(data={DOMAIN: {}})
    inventory = Inventory("dev", [])

    object.__setattr__(
        inventory,
        "_heater_sample_address_cache",
        ({"htr": ("1",)}, {"heater": "htr", "htr": "htr"}),
    )
    object.__setattr__(
        inventory,
        "_power_monitor_sample_address_cache",
        ({"pmo": ("7",)}, {"meter": "pmo", "pmo": "pmo"}),
    )

    coordinator = CoordinatorStub()
    build_entry_runtime(
        hass=hass,
        entry_id=entry_id,
        dev_id="dev",
        inventory=inventory,
        coordinator=SimpleNamespace(inventory=inventory),
        energy_coordinator=coordinator,
    )

    forward_ws_sample_updates(
        hass,
        entry_id,
        "dev",
        {
            "heater": {"samples": {"1": {"temp": 23}}},
            "meter": {"samples": {"7": {"power": 200}}, "lease_seconds": 45},
            "mystery": {"samples": {"lease_seconds": 10}},
        },
    )

    assert coordinator.calls == [
        (
            "dev",
            {"htr": {"1": {"temp": 23}}, "pmo": {"7": {"power": 200}}},
            45.0,
        )
    ]
