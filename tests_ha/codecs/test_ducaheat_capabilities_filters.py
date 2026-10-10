"""Tests for stripping capability payloads from decoded settings."""

from __future__ import annotations

from custom_components.termoweb.codecs.ducaheat_codec import decode_settings
from custom_components.termoweb.domain.ids import NodeType


def test_codec_decode_settings_discards_capabilities() -> None:
    """Codec decoding should emit only canonical state fields."""

    payload = {
        "status": {"mode": "auto"},
        "setup": {"capabilities": {"raw": True}},
        "capabilities": {"nested": {"payload": True}},
    }

    decoded = decode_settings(payload, node_type=NodeType.ACCUMULATOR)

    assert decoded == {"mode": "auto"}
