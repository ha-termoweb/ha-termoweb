"""Tests for websocket sample pseudo lease address handling."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from custom_components.termoweb.backend.ws_client import forward_ws_sample_updates
from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.inventory import Inventory, build_node_inventory


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


def test_forward_ws_sample_updates_omits_inner_lease_entry() -> None:
    """forward_ws_sample_updates should ignore pseudo lease sample addresses."""

    entry_id = "entry"
    coordinator = CoordinatorStub()
    hass = SimpleNamespace(
        data={
            DOMAIN: {
                entry_id: {
                    "energy_coordinator": coordinator,
                    "inventory": Inventory(
                        "dev",
                        {"nodes": [{"type": "htr", "addr": "1"}]},
                        build_node_inventory({"nodes": [{"type": "htr", "addr": "1"}]}),
                    ),
                }
            }
        }
    )

    forward_ws_sample_updates(
        hass,
        entry_id,
        "dev",
        {
            "htr": {
                "samples": {"lease_seconds": 15, "1": {"temp": 23}},
                "lease_seconds": 30,
            }
        },
    )

    assert coordinator.calls == [("dev", {"htr": {"1": {"temp": 23}}}, 30.0)]
