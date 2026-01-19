"""Tests for websocket sample payload validation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from conftest import build_entry_runtime
from custom_components.termoweb.backend.ws_client import forward_ws_sample_updates
from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.inventory import Inventory, build_node_inventory


class EnergyCoordinatorStub:
    """Record forwarded websocket samples for assertions."""

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
        """Store forwarded payloads for later inspection."""

        self.calls.append((dev_id, updates, lease_seconds))


def test_forward_ws_sample_updates_ignores_non_mapping_sections() -> None:
    """forward_ws_sample_updates should ignore non-mapping sections."""

    entry_id = "entry"
    coordinator = EnergyCoordinatorStub()
    hass = SimpleNamespace(data={DOMAIN: {}})
    inventory = Inventory(
        "dev",
        build_node_inventory({"nodes": [{"type": "htr", "addr": "1"}]}),
    )
    runtime = build_entry_runtime(
        hass=hass,
        entry_id=entry_id,
        dev_id="dev",
        inventory=inventory,
    )
    runtime.energy_coordinator = coordinator

    forward_ws_sample_updates(
        hass,
        entry_id,
        "dev",
        {
            "heater": {
                "samples": {"1": {"temp": 23}},
                "lease_seconds": 15,
            },
            "scalar": "ignored",
            "number": 5,
            "none": None,
            "list": ["bad"],
        },
    )

    assert coordinator.calls == [
        (
            "dev",
            {"heater": {"1": {"temp": 23}}},
            15.0,
        )
    ]
