"""Tests for Engine.IO / Socket.IO framing (backend/ducaheat_engineio.py)."""

from __future__ import annotations

import gzip

import pytest

from custom_components.termoweb.backend.ducaheat_engineio import (
    SocketIOParseError,
    decode_polling_packets,
    decode_socketio_event,
    encode_polling_packet,
    encode_socketio_event,
    rand_t,
)


def test_rand_t_token_format() -> None:
    """Polling cache-busters are ``P`` plus seven alphanumerics."""
    token = rand_t()
    assert token.startswith("P")
    assert len(token) == 8
    assert token[1:].isalnum()


def test_polling_packets_round_trip_plain_and_gzip() -> None:
    """Encoded packets carry an ASCII length; binary framing decodes, gzip too."""
    assert encode_polling_packet("40/message") == b"10:40/message"

    body = bytes([0, 1, 0, 0xFF]) + b"40/message"
    assert decode_polling_packets(body) == ["40/message"]
    assert decode_polling_packets(gzip.compress(body)) == ["40/message"]


@pytest.mark.parametrize(
    "body",
    [
        b"\x00\x00\x00",  # truncated
        b"\x00\x0a",  # invalid length digit
        b"\x00\xff\x00\x00",  # no length digits
        b"\x00\x02\xff\x00",  # length overruns the body
        b"\x1f\x8bbad",  # corrupt gzip
    ],
)
def test_decode_polling_packets_rejects_malformed_bodies(body: bytes) -> None:
    """Malformed polling bodies decode to nothing instead of raising."""
    assert decode_polling_packets(body) == []


def test_socketio_events_round_trip_with_and_without_namespace() -> None:
    """Encoded ``42`` frames decode back to ``(event, args)``."""
    frame = encode_socketio_event("/api/v2/socket_io", "subscribe", "/htr/1/status")
    assert frame == '42/api/v2/socket_io,["subscribe","/htr/1/status"]'
    assert decode_socketio_event(frame[2:]) == ("subscribe", ["/htr/1/status"])
    assert decode_socketio_event('["update",{"a":1}]') == ("update", [{"a": 1}])


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ("/api/v2/socket_io", "namespace"),
        ("/ns,not json", "json"),
        ("{}", "shape"),
        ("[]", "shape"),
    ],
)
def test_decode_socketio_event_reports_the_failure_reason(
    payload: str, reason: str
) -> None:
    """Undecodable payloads raise with a short reason for parse-error tracking."""
    with pytest.raises(SocketIOParseError) as err:
        decode_socketio_event(payload)
    assert err.value.reason == reason
