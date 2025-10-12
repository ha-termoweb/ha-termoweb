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

    htr_node = types.SimpleNamespace(addr="4", type="htr", supports_boost="no")
    acm_node = types.SimpleNamespace(
        addr="5", type="acm", supports_boost=lambda: "true"
    )
    pmo_node = types.SimpleNamespace(addr="6", type="pmo", supports_boost=None)

    object.__setattr__(
        inventory,
        "_nodes_by_type_cache",
        {
            " ": (types.SimpleNamespace(addr="ignored"),),
            "htr": (
                types.SimpleNamespace(addr=" ", type="htr"),
                htr_node,
            ),
            "acm": (
                types.SimpleNamespace(addr=None, type="acm"),
                acm_node,
            ),
            "pmo": (pmo_node,),
        },
    )
    object.__setattr__(
        inventory,
        "_heater_address_map_cache",
        (
            {
                " ": ("1",),
                "acm-empty": (),
                "htr": ("4",),
                "acm": ("", "5"),
                "thm": ("7",),
            },
            {},
        ),
    )

    def _fake_resolve(
        self: Inventory,
        node_type: str,
        addr: str,
        *,
        default_factory: Any | None = None,
    ) -> str:
        return f"{node_type}:{addr}"

    monkeypatch.setattr(Inventory, "resolve_heater_name", _fake_resolve)

    results = list(boost_module.iter_inventory_heater_metadata(inventory))

    assert [
        (node_type, addr) for node_type, addr, _, _ in results
    ] == [
        ("htr", "4"),
        ("acm", "5"),
    ]
    assert results[0][2] == "htr:4"
    assert boost_module.supports_boost(results[0][3]) is False
    assert results[1][2] == "acm:5"
    assert boost_module.supports_boost(results[1][3]) is True
