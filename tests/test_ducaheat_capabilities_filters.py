"""Tests for stripping capability payloads from decoded settings."""

from __future__ import annotations

from types import SimpleNamespace

from custom_components.termoweb.backend.ducaheat import DucaheatRESTClient
from custom_components.termoweb.codecs.ducaheat_codec import decode_settings
from custom_components.termoweb.domain.ids import NodeType


def test_backend_normalise_settings_strips_capabilities() -> None:
    """Backend normalisation should drop vendor capability blobs."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")
    payload = {
        "status": {
            "mode": "auto",
            "capabilities": {"nested": True},
            "boost_temp": "25.0",
        },
        "setup": {
            "capabilities": {"another": "value"},
            "extra_options": {"boost_time": 30},
        },
    }

    normalised = client._normalise_settings(payload, node_type="acm")

    assert normalised == {"mode": "auto", "boost_temp": "25.0", "boost_time": 30}


def test_codec_decode_settings_discards_capabilities() -> None:
    """Codec decoding should emit only canonical state fields."""

    payload = {
        "status": {"mode": "auto"},
        "setup": {"capabilities": {"raw": True}},
        "capabilities": {"nested": {"payload": True}},
    }

    decoded = decode_settings(payload, node_type=NodeType.ACCUMULATOR)

    assert decoded == {"mode": "auto"}
