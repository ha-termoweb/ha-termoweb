"""Additional coverage tests for boost helpers."""

from __future__ import annotations

import types
from typing import Any

import pytest

from custom_components.termoweb import boost as boost_module
from custom_components.termoweb.inventory import Inventory


def test_supports_boost_accepts_boolean_attribute() -> None:
    """A boolean ``supports_boost`` attribute should be returned verbatim."""

    node = types.SimpleNamespace(supports_boost=True)

    assert boost_module.supports_boost(node) is True


def test_supports_boost_handles_callable_failure(caplog: pytest.LogCaptureFixture) -> None:
    """Callable ``supports_boost`` errors should be logged and return ``False``."""

    class FailingNode:
        addr = "01"

        def supports_boost(self) -> bool:
            raise RuntimeError("boom")

    with caplog.at_level("DEBUG"):
        assert boost_module.supports_boost(FailingNode()) is False
    assert "Ignoring boost support probe failure" in caplog.text


def test_supports_boost_defaults_to_false_for_unknown_value() -> None:
    """Unsupported values should fall back to ``False``."""

    node = types.SimpleNamespace(supports_boost="maybe")

    assert boost_module.supports_boost(node) is False


def test_iter_inventory_heater_metadata_covers_branch_variants(
    monkeypatch: pytest.MonkeyPatch, inventory_builder
) -> None:
    """Inventory metadata iterator should cope with mixed node structures."""

    inventory = inventory_builder("dev", {})

    invalid_nodes = (
        types.SimpleNamespace(type=None, addr="1"),
        types.SimpleNamespace(type="acm", addr=" "),
    )
    object.__setattr__(inventory, "_heater_nodes_cache", tuple(invalid_nodes))

    htr_node = types.SimpleNamespace(addr="4", type="htr", supports_boost="no")
    acm_node = types.SimpleNamespace(
        addr="5", type="acm", supports_boost=lambda: "true"
    )
    pmo_node = types.SimpleNamespace(addr="6", type="pmo", supports_boost=None)

    nodes_by_type = {
        " ": [types.SimpleNamespace(addr="ignored")],
        "htr": {
            "bad": types.SimpleNamespace(addr=" ", type="htr"),
            "good": htr_node,
        },
        "acm": [acm_node],
        "pmo": pmo_node,
    }

    addresses_by_type = {
        " ": ["1"],
        "acm-empty": [],
        "htr": ["4"],
        "acm": ["", "5"],
        "thm": ["7"],
    }

    def _resolve_name(node_type: str, addr: str) -> str:
        return f"{node_type}:{addr}"

    def _fake_helper(
        inventory_value: Inventory, *, default_name_simple: Any
    ) -> tuple[dict[str, Any], dict[str, Any], Any]:
        assert inventory_value is inventory
        return nodes_by_type, addresses_by_type, _resolve_name

    monkeypatch.setattr(
        boost_module,
        "heater_platform_details_from_inventory",
        _fake_helper,
    )

    results = list(boost_module.iter_inventory_heater_metadata(inventory))

    assert [(item.node_type, item.addr) for item in results] == [
        ("htr", "4"),
        ("acm", "5"),
    ]
    assert results[0].name == "htr:4"
    assert results[0].supports_boost is False
    assert results[1].name == "acm:5"
    assert results[1].supports_boost is True
