"""Tests for merging accumulator capability payloads."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from custom_components.termoweb.backend.ducaheat import DucaheatRESTClient


def test_normalise_acm_capabilities_merges_nested_payloads() -> None:
    """Root, status, and setup capabilities should merge recursively."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    payload = {
        "capabilities": {
            "limit": {
                "max": 12,
                "nested": {"flag": True},
            },
            "mode": "auto",
        },
        "status": {
            "capabilities": {
                "limit": {
                    "min": 3,
                    "nested": {"flag": False, "status": "ok"},
                },
                "status_only": 42,
            }
        },
        "setup": {
            "capabilities": {
                "limit": {
                    "nested": {
                        "flag": False,
                        "setup": "done",
                        "depth": {"level": 3},
                    }
                },
                "setup_only": True,
            }
        },
    }

    original = deepcopy(payload)

    result = client._normalise_acm_capabilities(payload)

    assert result == {
        "limit": {
            "max": 12,
            "min": 3,
            "nested": {
                "flag": False,
                "status": "ok",
                "setup": "done",
                "depth": {"level": 3},
            },
        },
        "mode": "auto",
        "status_only": 42,
        "setup_only": True,
    }

    # Source payload should remain unchanged after merging.
    assert payload == original


def test_normalise_acm_capabilities_ignores_non_dict_containers() -> None:
    """Non-mapping capability containers should be skipped."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    payload = {
        "capabilities": {"root": True},
        "status": ["not", "a", "mapping"],
        "setup": None,
    }

    original = deepcopy(payload)

    result = client._normalise_acm_capabilities(payload)

    assert result == {"root": True}
    assert payload == original
