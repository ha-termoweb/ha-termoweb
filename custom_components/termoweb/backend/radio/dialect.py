"""On-air framing for the two TermoWeb 869 MHz radio dialects.

Both dialects share one logical layout after byte 0:
``net(2) src dst flags path(5) tag payload crc(2)``; an ack stops after
``flags``. They differ in sync word, the length rule for byte 0, whitening,
CRC and network id: dialect A has a fixed network id, dialect B's is per
installation and must be supplied. See ``docs/radio_protocol.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

HEADER_LEN = 12  # logical bytes 0..11 before the payload
ACK_TOTAL_LEN = 8  # length byte, net(2), acker, sender, flags, crc(2)
ACK_FLAGS = 0x80
PATH_LEN = 5
PATH_SLICE = slice(6, 11)
CRC_LEN = 2


def crc16_ccitt_a(data: bytes) -> int:
    """Return dialect A's CRC-16/CCITT (poly 0x1021, init 0x1D0F, xorout 0xFFFF)."""
    crc = 0x1D0F
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = (
                ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
            )
    return crc ^ 0xFFFF


def crc16_modbus(data: bytes) -> int:
    """Return dialect B's CRC-16/MODBUS (reflected poly 0xA001, init 0xFFFF)."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


def keystream(length: int) -> bytes:
    """Return the first ``length`` bytes of dialect A's PN9 keystream, MSB-first."""
    reg, feedback, out = 0xFF, 1, bytearray()
    for _ in range(length):
        out.append(reg)
        for _ in range(8):
            previous = feedback
            feedback = ((reg ^ ((reg << 5) & 0xFF)) >> 7) & 1
            reg = ((reg << 1) | previous) & 0xFF
    return bytes(out)


def whiten(data: bytes) -> bytes:
    """XOR ``data`` with the PN9 keystream; scrambling and descrambling are identical."""
    return bytes(b ^ k for b, k in zip(data, keystream(len(data)), strict=True))


@dataclass(frozen=True)
class Dialect:
    """One on-air framing: sync, length rule, whitening, CRC and defaults."""

    name: str
    sync: bytes
    network_id: bytes | None  # None: per installation, the caller must supply it
    scrambled: bool
    length_offset: int  # total frame length = logical byte 0 + length_offset
    crc: Callable[[bytes], int] = field(repr=False)
    eb_clock_suffix: bytes
    firmware_mode: int  # argument of the gateway's ``Y`` command
    program_write_slots: int  # slots per day in a ``B2`` program write
    mode_in_preset_write: bool  # True: ``B6`` carries the mode; ``B4`` mode is ignored


DIALECT_A = Dialect(
    name="A",
    sync=bytes.fromhex("2DE5"),
    network_id=bytes.fromhex("1B30"),
    scrambled=True,
    length_offset=3,
    crc=crc16_ccitt_a,
    eb_clock_suffix=b"\x03",
    firmware_mode=0,
    program_write_slots=48,
    mode_in_preset_write=False,
)

DIALECT_B = Dialect(
    name="B",
    sync=bytes.fromhex("2DD4"),
    network_id=None,
    scrambled=False,
    length_offset=0,
    crc=crc16_modbus,
    eb_clock_suffix=b"",
    firmware_mode=1,
    program_write_slots=24,
    mode_in_preset_write=True,
)

DIALECTS: dict[str, Dialect] = {d.name: d for d in (DIALECT_A, DIALECT_B)}


@dataclass(frozen=True)
class Frame:
    """One decoded frame; header fields are None when the input is too short."""

    length_ok: bool
    crc_ok: bool
    network_id: bytes | None
    src: int | None
    dst: int | None
    flags: int | None
    path: bytes | None
    tag: int | None
    payload: bytes
    logical: bytes
    air: bytes

    @property
    def ok(self) -> bool:
        """Return True when both the length byte and the CRC verify."""
        return self.length_ok and self.crc_ok

    @property
    def is_ack(self) -> bool:
        """Return True for a valid 8-byte link-layer ack (flags 0x80, no payload)."""
        return (
            self.ok and len(self.logical) == ACK_TOTAL_LEN and self.flags == ACK_FLAGS
        )


def _to_air(dialect: Dialect, logical: bytes) -> bytes:
    """Apply the dialect's whitening (if any) to logical bytes."""
    return whiten(logical) if dialect.scrambled else bytes(logical)


def _to_logical(dialect: Dialect, air: bytes) -> bytes:
    """Remove the dialect's whitening (if any) from on-air bytes."""
    return whiten(air) if dialect.scrambled else bytes(air)


def encode(dialect: Dialect, logical: bytes) -> bytes:
    """Return on-air bytes for logical bytes 1..n-3 (net id through payload)."""
    body = bytes(logical)
    total = len(body) + 1 + CRC_LEN
    length_byte = total - dialect.length_offset
    if not 0 <= length_byte <= 0xFF:
        raise ValueError(f"frame of {total} bytes does not fit dialect {dialect.name}")
    head = bytes([length_byte]) + body
    crc = dialect.crc(head)
    return _to_air(dialect, head + crc.to_bytes(2, "big"))


def frame_total_length(dialect: Dialect, first_air_byte: int) -> int:
    """Return the total on-air length implied by a frame's first air byte."""
    logical0 = first_air_byte ^ 0xFF if dialect.scrambled else first_air_byte
    return (logical0 & 0xFF) + dialect.length_offset


def decode(dialect: Dialect, air: bytes) -> Frame:
    """Decode on-air bytes, verifying length and CRC; never raises on bad input."""
    air = bytes(air)
    logical = _to_logical(dialect, air)
    if not logical:
        return Frame(False, False, None, None, None, None, None, None, b"", b"", air)
    total = logical[0] + dialect.length_offset
    length_ok = total == len(air)
    body_end = max(total - CRC_LEN, 0)
    crc_ok = (
        total >= 1 + CRC_LEN
        and len(logical) >= total
        and dialect.crc(logical[:body_end])
        == int.from_bytes(logical[body_end:total], "big")
    )
    body = logical[:body_end]
    path = bytes(body[PATH_SLICE]) if len(body) >= PATH_SLICE.stop else None
    return Frame(
        length_ok=length_ok,
        crc_ok=crc_ok,
        network_id=bytes(body[1:3]) if len(body) >= 3 else None,
        src=body[3] if len(body) > 3 else None,
        dst=body[4] if len(body) > 4 else None,
        flags=body[5] if len(body) > 5 else None,
        path=path,
        tag=body[11] if len(body) > 11 else None,
        payload=bytes(body[HEADER_LEN:]),
        logical=logical,
        air=air,
    )


def build_frame(
    dialect: Dialect,
    src: int,
    dst: int,
    payload: bytes,
    *,
    flags: int = 0,
    tag: int = 0,
    path: Iterable[int] | None = None,
    network_id: bytes | None = None,
) -> bytes:
    """Return on-air bytes for a data frame; path defaults to (src, dst, 0, 0, 0)."""
    path_bytes = bytes((src, dst, 0, 0, 0) if path is None else path)
    if len(path_bytes) != PATH_LEN:
        raise ValueError("path must be exactly five bytes")
    net = _network_id(dialect, network_id)
    body = net + bytes([src, dst, flags]) + path_bytes + bytes([tag]) + bytes(payload)
    return encode(dialect, body)


def build_ack(
    dialect: Dialect, acker: int, sender: int, network_id: bytes | None = None
) -> bytes:
    """Return on-air bytes for the ack ``acker`` sends for a frame from ``sender``."""
    net = _network_id(dialect, network_id)
    return encode(dialect, net + bytes([acker, sender, ACK_FLAGS]))


def _network_id(dialect: Dialect, network_id: bytes | None) -> bytes:
    """Return a validated two-byte network id, defaulting to the dialect's own."""
    net = dialect.network_id if network_id is None else bytes(network_id)
    if net is None:
        raise ValueError(f"dialect {dialect.name} needs an explicit network_id")
    if len(net) != 2:
        raise ValueError("network_id must be exactly two bytes")
    return net


def detect_dialect(air: bytes) -> Dialect | None:
    """Return the first dialect under which ``air`` has a valid length and CRC."""
    for dialect in DIALECTS.values():
        if decode(dialect, air).ok:
            return dialect
    return None
