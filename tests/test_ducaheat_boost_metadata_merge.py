"""Tests for merging boost metadata payloads."""

from __future__ import annotations

from types import SimpleNamespace

from custom_components.termoweb.backend.ducaheat import DucaheatRESTClient


def test_merge_boost_metadata_prefers_existing_when_requested() -> None:
    """Prefer existing boost metadata entries when flagged."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    target: dict[str, object] = {
        "boost_active": True,
        "boost_end_day": 1,
        "boost_end_min": None,
    }
    source = {
        "boost": False,
        "boost_end_day": 4,
        "boost_end_min": 240,
        "boost_end": {"day": 4, "minute": 240},
    }

    client._merge_boost_metadata(target, source, prefer_existing=True)

    assert target == {
        "boost_active": True,
        "boost_end_day": 1,
        "boost_end_min": 240,
    }


def test_merge_boost_metadata_nested_mapping_respects_prefer_flags() -> None:
    """Derived boost end values should obey the explicit prefer flag."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    target: dict[str, object] = {
        "boost_end_day": 3,
        "boost_end_min": None,
    }
    nested = {"day": 5, "minute": 150}

    client._merge_boost_metadata(target, {"boost_end": nested})

    assert target["boost_end_day"] == 3
    assert target["boost_end_min"] == 150
