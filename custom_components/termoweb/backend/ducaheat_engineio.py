"""Engine.IO v3 / Socket.IO framing for the Ducaheat websocket."""

from __future__ import annotations

import contextlib
import gzip
import json
import random
import string
from typing import Any


class SocketIOParseError(ValueError):
    """Raised when a Socket.IO event payload cannot be decoded."""

    def __init__(self, reason: str) -> None:
        """Store the short failure ``reason`` (namespace, json or shape)."""
        super().__init__(reason)
        self.reason = reason


def rand_t() -> str:
    """Return a pseudo-random token string for polling requests."""

    return "P" + "".join(
        random.choice(string.ascii_letters + string.digits) for _ in range(7)
    )


def encode_polling_packet(pkt: str) -> bytes:
    """Encode Engine.IO polling payloads."""

    data = pkt.encode("utf-8")
    return f"{len(data)}:".encode("ascii") + data


def decode_polling_packets(body: bytes) -> list[str]:
    """Decode Engine.IO v3 binary polling frames."""

    buf = body
    if len(buf) >= 2 and buf[:2] == b"\x1f\x8b":
        with contextlib.suppress(Exception):
            buf = gzip.decompress(buf)
    out: list[str] = []
    i, n = 0, len(buf)
    while i < n:
        if i + 4 > n:
            break
        _typ = buf[i]
        i += 1
        length = 0
        had_digit = False
        while i < n and buf[i] != 0xFF:
            d = buf[i]
            i += 1
            if d > 9:
                return out
            had_digit = True
            length = length * 10 + d
        if not had_digit or i >= n or buf[i] != 0xFF:
            return out
        i += 1
        end = i + length
        if end > n:
            return out
        payload = buf[i:end]
        i = end
        try:
            pkt = payload.decode("utf-8", errors="ignore")
        except UnicodeDecodeError, ValueError:
            pkt = ""
        if pkt:
            out.append(pkt)
    return out


def decode_socketio_event(payload: str) -> tuple[str, list[Any]]:
    """Decode a Socket.IO ``42`` payload into ``(event, args)``."""

    content = payload
    if content.startswith("/"):
        _namespace, sep, content = content.partition(",")
        if sep != ",":
            raise SocketIOParseError("namespace")
    try:
        arr = json.loads(content)
    except json.JSONDecodeError as err:
        raise SocketIOParseError("json") from err
    if not isinstance(arr, list) or not arr:
        raise SocketIOParseError("shape")
    return arr[0], arr[1:]


def encode_socketio_event(namespace: str, event: str, *args: Any) -> str:
    """Return a namespaced Socket.IO ``42`` event frame."""

    payload = json.dumps([event, *args], separators=(",", ":"), default=str)
    return f"42{namespace}," + payload
