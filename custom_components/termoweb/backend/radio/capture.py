"""Record every frame and gateway line heard during a capture, and summarise it.

A capture is passive: it only adds listeners to a :class:`RadioLink` and never
transmits. Each decoded frame becomes a JSON-able record (UTC time with
milliseconds, RSSI, dialect, network id, header fields, payload); gateway lines
that are not frames, and frames no known dialect decodes, are kept as raw text.
:func:`redact` masks network ids, identity/serial bytes and on-air hex so a
capture can be shared. :class:`FrameLog` keeps the last frames heard and sent,
masked when read, for diagnostics. This module has no Home Assistant
dependency. See ``docs/radio_protocol.md`` (section 10).
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
import re
from typing import Any

from . import protocol
from .dialect import Dialect, Frame, decode, detect_dialect
from .link import ReceivedFrame
from .pairing import ANNOUNCE_LEN_A, ANNOUNCE_MARKER_A, classify_announcement

KIND_DATA = "data"
KIND_ACK = "ack"
# Tags whose payload is not an opcode: pairing announcement, id assignment.
PAIRING_TAGS = frozenset({0x03, 0x04})
MASK_BYTE = "XX"
MAC_FIELD = re.compile(r"\bmac=\S+")
NET_FIELD = re.compile(r"\bnet=\S+")
HEX_RUN = re.compile(r"\b[0-9A-Fa-f]{12,}\b")  # on-air bytes in TX/ACK/RX lines
FRAME_LOG_SIZE = 200
# Frame log kinds; each is counted for every frame since the client started.
LOG_OWN = "own_network"  # data frames on this station's network
LOG_ACK = "ack"  # link-layer acks on this station's network
LOG_FOREIGN = "foreign_network"  # data frames or acks on another network id
LOG_PAIRING = "pairing_announcement"  # a heater in pairing mode, any network
LOG_TX = "tx"  # frames this station transmitted
LOG_UNDECODABLE = "undecodable"  # no known dialect verifies its length and CRC
LOG_KINDS = (LOG_OWN, LOG_ACK, LOG_FOREIGN, LOG_PAIRING, LOG_TX, LOG_UNDECODABLE)


def _utcnow() -> datetime:
    """Return the current UTC time."""
    return datetime.now(UTC)


def timestamp(at: datetime) -> str:
    """Return ``at`` as an ISO 8601 UTC string with milliseconds."""
    return at.astimezone(UTC).isoformat(timespec="milliseconds")


def _hex(data: bytes | None) -> str | None:
    """Return upper-case hex without spaces, or None."""
    return None if data is None else data.hex().upper()


def mask_mac(line: str) -> str:
    """Mask a gateway MAC (``mac=..``) in a firmware line."""
    return MAC_FIELD.sub("mac=XX", line)


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
) -> dict[str, Any] | None:
    """Return one received frame as a capture record, None when it is undecodable.

    A frame that fails in ``dialect`` (the link switched dialects while it was
    on air) is tried in every known dialect first.
    """
    frame = received.frame
    if not frame.ok:
        other = detect_dialect(frame.air)
        if other is None:
            return None
        dialect, frame = other, decode(other, frame.air)
    key, name = (None, None) if frame.is_ack else opcode_key(frame)
    data = not frame.is_ack
    return {
        "t": timestamp(at),
        "kind": KIND_DATA if data else KIND_ACK,
        "rssi": received.rssi_dbm,
        "lqi": received.lqi,
        "micros": received.micros,
        "dialect": dialect.name,
        "net": _hex(frame.network_id),
        "src": frame.src,
        "dst": frame.dst,
        "flags": frame.flags,
        "path": _hex(frame.path) if data else None,
        "tag": frame.tag if data else None,
        "payload": _hex(frame.payload) if data else None,
        "op": key,
        "name": name,
        "air": _hex(frame.air),
    }


def rx_line(received: ReceivedFrame) -> str:
    """Return an undecodable frame as the gateway's RX line (CRC bit 0)."""
    return (
        f"RX {received.micros} {received.rssi_dbm} {received.lqi} 0 "
        f"{_hex(received.frame.air)}"
    )


def raw_record(line: str, at: datetime) -> dict[str, Any]:
    """Return one raw firmware line as a record; a gateway MAC is always masked."""
    return {"t": timestamp(at), "line": mask_mac(line)}


class FrameCapture:
    """Collect decoded frames and raw lines from a link's listeners."""

    def __init__(
        self,
        dialect: Callable[[], Dialect],
        *,
        now: Callable[[], datetime] = _utcnow,
    ) -> None:
        """Store how to read the link's current dialect and the clock."""
        self._dialect = dialect
        self._now = now
        self.frames: list[dict[str, Any]] = []
        self.raw: list[dict[str, Any]] = []
        self.started: str | None = None
        self.ended: str | None = None

    def start(self) -> None:
        """Note when the capture window opened."""
        self.started = timestamp(self._now())

    def stop(self) -> None:
        """Note when the capture window closed."""
        self.ended = timestamp(self._now())

    def on_frame(self, received: ReceivedFrame) -> None:
        """Record a frame the link heard; undecodable ones go to ``raw``."""
        at = self._now()
        record = frame_record(received, self._dialect(), at)
        if record is None:
            self.raw.append(raw_record(rx_line(received), at))
        else:
            self.frames.append(record)

    def on_line(self, line: str) -> None:
        """Record a gateway line that is not a decoded frame."""
        self.raw.append(raw_record(line, self._now()))


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


def redact_line(line: str) -> str:
    """Mask the MAC, network ids and on-air hex in a gateway line."""
    return HEX_RUN.sub("<hex>", NET_FIELD.sub("net=XXXX", mask_mac(line)))


def redact(
    frames: Iterable[Mapping[str, Any]], raw: Iterable[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return frames and raw lines safe to share: network ids become NET1, NET2...

    Identity and serial bytes are masked, on-air hex is dropped and raw lines
    lose network ids and hex runs. Every other payload byte is kept.
    """
    aliases: dict[str, str] = {}
    redacted: list[dict[str, Any]] = []
    for original in frames:
        record = dict(original)
        record.pop("air", None)
        net = record.get("net")
        if net is not None:
            record["net"] = aliases.setdefault(net, f"NET{len(aliases) + 1}")
        if record.get("payload"):
            record["payload"] = _mask_identity(record["payload"])
        redacted.append(record)
    lines = [{**line, "line": redact_line(line["line"])} for line in raw]
    return redacted, lines


@dataclass(frozen=True, slots=True)
class _LoggedFrame:
    """One frame log entry: when, direction, kind, dialect and the frame."""

    at: datetime
    direction: str
    kind: str
    dialect: str
    frame: Frame
    own: bool  # on the station's network id at the time


class FrameLog:
    """Keep the last frames heard and sent plus a count per kind; no I/O."""

    def __init__(
        self, size: int = FRAME_LOG_SIZE, *, now: Callable[[], datetime] = _utcnow
    ) -> None:
        """Create an empty log that holds at most ``size`` frames."""
        self._entries: deque[_LoggedFrame] = deque(maxlen=size)
        self._counts: Counter[str] = Counter()
        self._now = now

    def on_rx(
        self, received: ReceivedFrame, dialect: Dialect, network_id: bytes
    ) -> None:
        """Log a received frame; one failing in ``dialect`` is tried in every dialect."""
        frame = received.frame
        if not frame.ok:
            other = detect_dialect(frame.air)
            if other is not None:
                dialect, frame = other, decode(other, frame.air)
        if not frame.ok:
            kind = LOG_UNDECODABLE
        elif classify_announcement(dialect, frame) is not None:
            kind = LOG_PAIRING
        elif frame.network_id != network_id:
            kind = LOG_FOREIGN
        else:
            kind = LOG_ACK if frame.is_ack else LOG_OWN
        self._add("rx", kind, dialect, frame, network_id)

    def on_tx(self, air: bytes, dialect: Dialect, network_id: bytes) -> None:
        """Log a frame this station transmitted."""
        self._add("tx", LOG_TX, dialect, decode(dialect, air), network_id)

    def _add(
        self, direction: str, kind: str, dialect: Dialect, frame: Frame, net: bytes
    ) -> None:
        """Append one entry and count its kind."""
        self._counts[kind] += 1
        self._entries.append(
            _LoggedFrame(
                self._now(),
                direction,
                kind,
                dialect.name,
                frame,
                frame.network_id == net,
            )
        )

    def snapshot(self) -> dict[str, Any]:
        """Return the counts and the logged frames, oldest first, safe to share.

        Network ids become ``own`` or ``NET1``, ``NET2``... in order of
        appearance, identity and serial bytes are masked and undecodable
        frames keep only their length.
        """
        aliases: dict[bytes | None, str] = {}
        return {
            "size": self._entries.maxlen,
            "counts": {kind: self._counts[kind] for kind in LOG_KINDS},
            "frames": [_log_record(entry, aliases) for entry in self._entries],
        }


def _log_record(
    entry: _LoggedFrame, aliases: dict[bytes | None, str]
) -> dict[str, Any]:
    """Return one frame log entry as a masked diagnostics record."""
    record: dict[str, Any] = {
        "t": timestamp(entry.at),
        "dir": entry.direction,
        "kind": entry.kind,
    }
    frame = entry.frame
    if entry.kind == LOG_UNDECODABLE:
        record["len"] = len(frame.air)
        return record
    net = frame.network_id
    alias = "own" if entry.own else aliases.setdefault(net, f"NET{len(aliases) + 1}")
    data = not frame.is_ack
    op, name = opcode_key(frame) if data else (None, None)
    payload = _hex(frame.payload) if data and frame.payload else None
    record.update(
        dialect=entry.dialect,
        net=alias,
        src=frame.src,
        dst=frame.dst,
        tag=frame.tag if data else None,
        op=op,
        name=name,
        payload=None if payload is None else _mask_identity(payload),
    )
    return record


def summarise(frames: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return frame and ack counts, networks, node ids and a named opcode histogram."""
    frames = list(frames)
    data = [record for record in frames if record["kind"] == KIND_DATA]
    histogram = Counter(record["op"] for record in data)
    names = {record["op"]: record["name"] for record in data}
    return {
        "frames": len(frames),
        "acks": len(frames) - len(data),
        "networks": sorted({r["net"] for r in frames if r["net"] is not None}),
        "nodes": sorted(
            {node for r in frames for node in (r["src"], r["dst"]) if node is not None}
        ),
        "opcodes": {
            key: {"count": count, "name": names[key]}
            for key, count in sorted(histogram.items())
        },
    }


__all__ = [
    "FrameCapture",
    "FrameLog",
    "frame_record",
    "mask_mac",
    "opcode_key",
    "raw_record",
    "redact",
    "redact_line",
    "rx_line",
    "summarise",
    "timestamp",
]
