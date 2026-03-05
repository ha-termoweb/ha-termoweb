"""Tests for websocket sample pseudo lease address handling."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from conftest import CoordinatorStub, build_entry_runtime
from custom_components.termoweb.backend.ws_client import forward_ws_sample_updates
from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.inventory import Inventory, build_node_inventory


def test_forward_ws_sample_updates_omits_inner_lease_entry() -> None:
    """forward_ws_sample_updates should ignore pseudo lease sample addresses."""

    entry_id = "entry"
    coordinator = CoordinatorStub()
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
            "htr": {
                "samples": {"lease_seconds": 15, "1": {"temp": 23}},
                "lease_seconds": 30,
            }
        },
    )

    assert coordinator.calls == [("dev", {"htr": {"1": {"temp": 23}}}, 30.0)]
