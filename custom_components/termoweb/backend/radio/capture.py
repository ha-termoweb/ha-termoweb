"""Record every frame and gateway line heard during a capture, and summarise it.

A capture is passive: it only adds listeners to a :class:`RadioLink` and never
transmits. Each frame becomes a JSON-able record (UTC time with milliseconds,
RSSI, dialect, network id, header fields, payload); gateway lines that are not
frames are kept as text. :func:`redact` masks network ids, identity/serial
bytes and on-air hex so a capture can be shared. This module has no Home
Assistant dependency. See ``docs/radio_protocol.md`` (section 10).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
import re
from typing import Any

from . import protocol
from .dialect import Dialect, Frame, decode, detect_dialect
from .link import ReceivedFrame
from .pairing import ANNOUNCE_LEN_A, ANNOUNCE_MARKER_A

KIND_DATA = "data"
KIND_ACK = "ack"
KIND_UNDECODABLE = "undecodable"
KIND_LINE = "line"
FRAME_KINDS = frozenset({KIND_DATA, KIND_ACK})
# Tags whose payload is not an opcode: pairing announcement, id assignment.
PAIRING_TAGS = frozenset({0x03, 0x04})
MASK_BYTE = "XX"
MAC_FIELD = re.compile(r"\bmac=\S+")
NET_FIELD = re.compile(r"\bnet=\S+")
HEX_RUN = re.compile(r"\b[0-9A-Fa-f]{12,}\b")  # on-air bytes in TX/ACK/RX lines


def _utcnow() -> datetime:
    """Return the current UTC time."""
    return datetime.now(UTC)


def timestamp(at: datetime) -> str:
    """Return ``at`` as an ISO 8601 UTC string with milliseconds."""
    return at.astimezone(UTC).isoformat(timespec="milliseconds")


def _hex(data: bytes | None) -> str | None:
    """Return upper-case hex, or None."""
    return None if data is None else data.hex().upper()


def opcode_key(frame: Frame) -> tuple[str, str | None]:
    """Return a data frame's histogram key and its known meaning (None: unknown)."""
    if frame.payload and frame.tag not in PAIRING_TAGS:
        opcode = frame.payload[0]
        return f"{opcode:02X}", protocol.OPCODE_NAMES.get(opcode)
    if frame.tag is None:
        return "tag --", None
    return f"tag {frame.tag:02X}", protocol.TAG_NAMES.get(frame.tag)


def frame_record(
    received: ReceivedFrame, dialect: Dialect, at: datetime
) -> dict[str, Any]:
    """Return one received frame as a capture record.

    A frame that fails in ``dialect`` (the link switched dialects while it was
    on air) is tried in every known dialect before it counts as undecodable.
    """
    frame = received.frame
    if not frame.ok:
        other = detect_dialect(frame.air)
        if other is not None:
            dialect, frame = other, decode(other, frame.air)
    record: dict[str, Any] = {
        "t": timestamp(at),
        "kind": KIND_UNDECODABLE,
        "dialect": None,
        "rssi": received.rssi_dbm,
        "lqi": received.lqi,
        "micros": received.micros,
        "air": _hex(frame.air),
    }
    if not frame.ok:
        record["len"] = len(frame.air)
        return record
    record.update(
        kind=KIND_ACK if frame.is_ack else KIND_DATA,
        dialect=dialect.name,
        net=_hex(frame.network_id),
        src=frame.src,
        dst=frame.dst,
        flags=frame.flags,
    )
    if not frame.is_ack:
        key, name = opcode_key(frame)
        record.update(
            path=_hex(frame.path),
            tag=frame.tag,
            payload=_hex(frame.payload),
            op=key,
            name=name,
        )
    return record


def line_record(line: str, at: datetime) -> dict[str, Any]:
    """Return one gateway line as a capture record; a gateway MAC is always masked."""
    return {
        "t": timestamp(at),
        "kind": KIND_LINE,
        "line": MAC_FIELD.sub("mac=XX", line),
    }


class FrameCapture:
    """Collect capture records from a link's frame and line listeners."""

    def __init__(
        self,
        dialect: Callable[[], Dialect],
        *,
        now: Callable[[], datetime] = _utcnow,
    ) -> None:
        """Store how to read the link's current dialect and the clock."""
        self._dialect = dialect
        self._now = now
        self.records: list[dict[str, Any]] = []

    def on_frame(self, received: ReceivedFrame) -> None:
        """Record a frame the link decoded."""
        self.records.append(frame_record(received, self._dialect(), self._now()))

    def on_line(self, line: str) -> None:
        """Record a gateway line that is not a decoded frame."""
        self.records.append(line_record(line, self._now()))


def _mask_identity(payload: str) -> str:
    """Mask the identity/serial bytes of a ``5B`` reply or dialect-A announcement."""
    data = bytes.fromhex(payload)
    if data[:1] == bytes([protocol.IDENTITY_MARKER]):
        keep = 2  # 5B and the form byte
    elif len(data) == ANNOUNCE_LEN_A and data[0] == ANNOUNCE_MARKER_A:
        keep = 1
    else:
        return payload
    return payload[: keep * 2] + MASK_BYTE * (len(data) - keep)


def _redact_line(line: str) -> str:
    """Mask network ids and on-air hex in a gateway line."""
    return HEX_RUN.sub("<hex>", NET_FIELD.sub("net=XXXX", line))


def redact(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return records safe to share: network ids become NET1, NET2...

    Identity and serial bytes are masked, on-air hex is dropped and gateway
    lines lose network ids and hex runs. Every other payload byte is kept.
    """
    aliases: dict[str, str] = {}
    redacted: list[dict[str, Any]] = []
    for original in records:
        record = dict(original)
        if record["kind"] == KIND_LINE:
            record["line"] = _redact_line(record["line"])
        else:
            record.pop("air", None)
            net = record.get("net")
            if net is not None:
                record["net"] = aliases.setdefault(net, f"NET{len(aliases) + 1}")
            if record.get("payload"):
                record["payload"] = _mask_identity(record["payload"])
        redacted.append(record)
    return redacted


def summarise(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return counts, networks, node ids and an opcode histogram with known names."""
    records = list(records)
    kinds = Counter(record["kind"] for record in records)
    frames = [record for record in records if record["kind"] in FRAME_KINDS]
    data = [record for record in frames if record["kind"] == KIND_DATA]
    histogram = Counter(record["op"] for record in data)
    names = {record["op"]: record["name"] for record in data}
    return {
        "frames": len(frames),
        "data_frames": kinds[KIND_DATA],
        "acks": kinds[KIND_ACK],
        "undecodable": kinds[KIND_UNDECODABLE],
        "lines": kinds[KIND_LINE],
        "first": records[0]["t"] if records else None,
        "last": records[-1]["t"] if records else None,
        "dialects": dict(sorted(Counter(r["dialect"] for r in frames).items())),
        "networks": sorted({r["net"] for r in frames if r["net"] is not None}),
        "nodes": sorted(
            {node for r in frames for node in (r["src"], r["dst"]) if node is not None}
        ),
        "opcodes": {
            key: {"count": count, "name": names[key]}
            for key, count in sorted(histogram.items())
        },
        "unknown_opcodes": sorted(key for key in histogram if names[key] is None),
    }


__all__ = [
    "FrameCapture",
    "frame_record",
    "line_record",
    "opcode_key",
    "redact",
    "summarise",
    "timestamp",
]
