"""Tests for the radio dialect framing (dialect A and dialect B)."""

from __future__ import annotations

import pytest

from custom_components.termoweb.backend.radio import dialect as d
from custom_components.termoweb.backend.radio.dialect import (
    DIALECT_A,
    DIALECT_B,
    DIALECTS,
    build_ack,
    build_frame,
    decode,
    detect_dialect,
    encode,
    frame_total_length,
)

# Dialect-B captures from heater 06, re-encoded with the synthetic network
# id 12 34 (CRCs recomputed).
NET = bytes.fromhex("1234")
B_REGISTRATION = bytes.fromhex("0F1234060100060101010100507222")
B_ROUTE_PROBE = bytes.fromhex("0E1234060200060201010106F691")
B_ACK = bytes.fromhex("081234010680A0E4")

# Dialect-A worked frames from ha-termoweb-local's test_frame.py.
A_SETPOINT = bytes.fromhex("F19C8858B3A1CD20575E4B9CBAEBD9F6E5")
A_MODE_OFF = bytes.fromhex("F29C8858B3A1CD20575E4B9CBAEDF895")
A_E5_REPORT = bytes.fromhex(
    "E59C885DB6A1C825565F4A9C5850E47E05BCB4E185AE81C3E7BFE380F1"
)
A_ACK = bytes.fromhex("FA9C885DB621FBD1")
A_RELAY_PROBE = bytes.fromhex("F49C885BB6A1C826565F4A9AB57C")


def test_crc_vectors() -> None:
    """Both CRCs match their standard check values and the captured trailers."""
    assert d.crc16_modbus(b"123456789") == 0x4B37
    # CRC-16/CCITT with init 0x1D0F and no xorout is 0xE5CC ("AUG-CCITT").
    assert d.crc16_ccitt_a(b"123456789") == 0xE5CC ^ 0xFFFF
    assert d.crc16_modbus(B_REGISTRATION[:-2]) == 0x7222
    assert d.crc16_modbus(B_ACK[:-2]) == 0xA0E4


def test_keystream_prefix_and_whiten_round_trip() -> None:
    """The PN9 keystream starts FF 87 B8 59 and whitening is its own inverse."""
    assert d.keystream(8).hex() == "ff87b859b7a1cc24"
    assert d.whiten(d.whiten(b"hello world")) == b"hello world"
    assert d.keystream(0) == b""


def test_dialect_table() -> None:
    """Both dialects are registered with their distinguishing parameters."""
    assert DIALECTS == {"A": DIALECT_A, "B": DIALECT_B}
    assert DIALECT_A.sync == bytes.fromhex("2DE5")
    assert DIALECT_B.sync == bytes.fromhex("2DD4")
    assert (DIALECT_A.length_offset, DIALECT_B.length_offset) == (3, 0)
    assert (DIALECT_A.firmware_mode, DIALECT_B.firmware_mode) == (0, 1)
    assert (DIALECT_A.eb_clock_suffix, DIALECT_B.eb_clock_suffix) == (b"\x03", b"")
    assert DIALECT_A.network_id == bytes.fromhex("1B30")
    assert DIALECT_B.network_id is None
    assert "crc" not in repr(DIALECT_A)


@pytest.mark.parametrize(
    ("air", "src", "dst", "tag", "payload", "path"),
    [
        (B_REGISTRATION, 0x06, 0x01, 0x00, b"\x50", bytes([6, 1, 1, 1, 1])),
        (B_ROUTE_PROBE, 0x06, 0x02, 0x06, b"", bytes([6, 2, 1, 1, 1])),
    ],
)
def test_dialect_b_captures_decode_and_rebuild(
    air, src, dst, tag, payload, path
) -> None:
    """Captured dialect-B frames decode and rebuild byte for byte."""
    frame = decode(DIALECT_B, air)
    assert frame.ok
    assert frame.network_id == NET
    assert (frame.src, frame.dst, frame.flags, frame.tag) == (src, dst, 0, tag)
    assert frame.path == path
    assert frame.payload == payload
    assert frame.logical == air == frame.air
    assert not frame.is_ack
    assert (
        build_frame(DIALECT_B, src, dst, payload, tag=tag, path=path, network_id=NET)
        == air
    )
    assert encode(DIALECT_B, frame.logical[1:-2]) == air


def test_dialect_b_ack() -> None:
    """The captured station ack decodes as an ack and rebuilds exactly."""
    frame = decode(DIALECT_B, B_ACK)
    assert frame.ok and frame.is_ack
    assert (frame.src, frame.dst, frame.flags) == (0x01, 0x06, 0x80)
    assert frame.path is None and frame.tag is None and frame.payload == b""
    assert build_ack(DIALECT_B, 0x01, 0x06, NET) == B_ACK


def test_dialect_a_worked_frames() -> None:
    """Dialect-A builders reproduce the proven worked frames."""
    assert build_frame(DIALECT_A, 0x01, 0x04, bytes([0xB4, 0x02, 0x33])) == A_SETPOINT
    assert build_frame(DIALECT_A, 0x01, 0x04, bytes([0xB4, 0x04])) == A_MODE_OFF
    assert build_ack(DIALECT_A, 0x04, 0x01) == A_ACK
    relay = decode(DIALECT_A, A_RELAY_PROBE)
    assert relay.ok and relay.tag == 0x06 and relay.path == bytes([4, 2, 1, 1, 1])
    assert (
        build_frame(DIALECT_A, 0x02, 0x01, b"", tag=0x06, path=relay.path)
        == A_RELAY_PROBE
    )


def test_dialect_a_report_decodes() -> None:
    """A captured E5 report decodes with its 56 B9 payload."""
    frame = decode(DIALECT_A, A_E5_REPORT)
    assert frame.ok
    assert frame.network_id == bytes.fromhex("1B30")
    assert (frame.src, frame.dst) == (0x04, 0x01)
    assert frame.payload[:2] == bytes([0x56, 0xB9])
    assert len(frame.payload) == 15
    assert decode(DIALECT_A, A_ACK).is_ack


@pytest.mark.parametrize("dialect", [DIALECT_A, DIALECT_B])
def test_round_trip_with_options(dialect) -> None:
    """Flags, tag, path and network id survive a build/decode round trip."""
    air = build_frame(
        dialect,
        0x01,
        0x09,
        b"\x01\x02\x03",
        flags=0x80,
        tag=0x04,
        path=[1, 9, 2, 3, 0],
        network_id=b"\x00\x00",
    )
    frame = decode(dialect, air)
    assert frame.ok
    assert frame.network_id == b"\x00\x00"
    assert frame.flags == 0x80 and not frame.is_ack
    assert frame.tag == 0x04 and frame.path == bytes([1, 9, 2, 3, 0])
    assert frame.payload == b"\x01\x02\x03"
    assert frame_total_length(dialect, air[0]) == len(air)
    assert detect_dialect(air) is dialect


def test_frame_total_length() -> None:
    """The first air byte alone gives the total frame length."""
    assert frame_total_length(DIALECT_A, A_SETPOINT[0]) == len(A_SETPOINT)
    assert frame_total_length(DIALECT_A, A_ACK[0]) == 8
    assert frame_total_length(DIALECT_B, 0x0F) == 15
    assert frame_total_length(DIALECT_B, 0x00) == 0


def test_detect_dialect() -> None:
    """Captures are attributed to the right dialect; garbage to none."""
    assert detect_dialect(B_REGISTRATION) is DIALECT_B
    assert detect_dialect(B_ACK) is DIALECT_B
    assert detect_dialect(A_E5_REPORT) is DIALECT_A
    assert detect_dialect(A_ACK) is DIALECT_A
    assert detect_dialect(b"") is None
    assert detect_dialect(b"\x00\x01\x02\x03") is None


@pytest.mark.parametrize("dialect", [DIALECT_A, DIALECT_B])
@pytest.mark.parametrize(
    "air",
    [
        b"",
        b"\x0f",
        b"\x0f\xed",
        b"\x0f\x12\x34\x06",
        b"\xff" * 40,
        b"\x01\x02",
        b"\x00",
    ],
)
def test_decode_never_raises_on_garbage(dialect, air) -> None:
    """Short or garbage input decodes to a not-ok frame without raising."""
    frame = decode(dialect, air)
    assert not frame.ok
    assert frame.air == air
    assert isinstance(frame.payload, bytes)


def test_decode_truncated_and_corrupted() -> None:
    """Truncation breaks the length check; a flipped byte breaks the CRC."""
    truncated = decode(DIALECT_B, B_REGISTRATION[:-3])
    assert not truncated.length_ok and not truncated.crc_ok
    assert truncated.src == 0x06
    corrupted = bytearray(B_REGISTRATION)
    corrupted[12] ^= 0x01
    frame = decode(DIALECT_B, bytes(corrupted))
    assert frame.length_ok and not frame.crc_ok
    assert not frame.ok
    longer = decode(DIALECT_B, B_REGISTRATION + b"\x00")
    assert not longer.length_ok and longer.crc_ok
    assert longer.payload == b"\x50"
    # A dialect-B frame is not valid dialect A and vice versa.
    assert not decode(DIALECT_A, B_REGISTRATION).ok
    assert not decode(DIALECT_B, A_SETPOINT).ok


def test_builder_validation() -> None:
    """Bad paths, network ids and oversize frames raise ValueError."""
    with pytest.raises(ValueError, match="five bytes"):
        build_frame(DIALECT_B, 1, 6, b"", path=(1, 6), network_id=NET)
    with pytest.raises(ValueError, match="two bytes"):
        build_frame(DIALECT_B, 1, 6, b"", network_id=b"\x01")
    with pytest.raises(ValueError, match="two bytes"):
        build_ack(DIALECT_A, 1, 6, network_id=b"\x01\x02\x03")
    with pytest.raises(ValueError, match="explicit network_id"):
        build_frame(DIALECT_B, 1, 6, b"")
    with pytest.raises(ValueError, match="explicit network_id"):
        build_ack(DIALECT_B, 1, 6)
    with pytest.raises(ValueError, match="does not fit"):
        encode(DIALECT_B, bytes(253))
    with pytest.raises(ValueError, match="does not fit"):
        encode(DIALECT_A, bytes(256))
    assert len(encode(DIALECT_B, bytes(252))) == 255
    assert len(encode(DIALECT_A, bytes(255))) == 258
