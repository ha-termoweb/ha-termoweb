"""Tests for capture records, redaction and the capture summary."""

from __future__ import annotations

from tests_ha.fakes.radio_link import build_ack

from datetime import UTC, datetime, timedelta, timezone

from tests_ha.fakes.radio_gateway import NET

from custom_components.termoweb.backend.radio import capture, protocol as p
from custom_components.termoweb.backend.radio.dialect import (
    DIALECT_A,
    DIALECT_B,
    build_frame,
    decode,
    encode,
)
from custom_components.termoweb.backend.radio.link import ReceivedFrame

AT = datetime(2026, 1, 2, 3, 4, 5, 678901, tzinfo=UTC)
IDENTITY = bytes.fromhex("5B55010203040506070809" + b"X123456".hex())
ANNOUNCE_A = bytes([0x77]) + bytes(range(1, 13))


def rx(dialect, src, dst, payload=b"", *, tag=0, net=NET) -> ReceivedFrame:
    """Return a received data frame."""
    air = build_frame(
        dialect, src, dst, payload, tag=tag, path=(src, dst, 0, 0, 0), network_id=net
    )
    return ReceivedFrame(decode(dialect, air), -61.5, 9, 4321)


def ack(src=6, dst=1) -> ReceivedFrame:
    """Return a received dialect-B ack."""
    return ReceivedFrame(
        decode(DIALECT_B, build_ack(DIALECT_B, src, dst, NET)), -70.0, 3, 77
    )


def test_timestamp_is_utc_with_milliseconds() -> None:
    """Times are ISO 8601 in UTC with milliseconds, whatever the input zone."""
    assert capture.timestamp(AT) == "2026-01-02T03:04:05.678+00:00"
    local = AT.astimezone(timezone(timedelta(hours=2)))
    assert capture.timestamp(local) == "2026-01-02T03:04:05.678+00:00"


def test_data_frame_record() -> None:
    """A data frame keeps every header field, the payload and its opcode name."""
    record = capture.frame_record(rx(DIALECT_B, 1, 6, p.flash_display()), DIALECT_B, AT)
    assert record == {
        "t": "2026-01-02T03:04:05.678+00:00",
        "kind": "data",
        "rssi": -61.5,
        "lqi": 9,
        "micros": 4321,
        "dialect": "B",
        "net": "1234",
        "src": 1,
        "dst": 6,
        "flags": 0,
        "path": "0106000000",
        "tag": 0,
        "payload": "5E01",
        "op": "5E",
        "name": "flash display (identify)",
        "air": record["air"],
    }
    assert bytes.fromhex(record["air"])[1:3] == NET
    assert record["air"] == record["air"].upper() and " " not in record["air"]


def test_ack_wrong_dialect_and_undecodable() -> None:
    """Acks carry no payload; another dialect is re-decoded; junk is None."""
    record = capture.frame_record(ack(), DIALECT_B, AT)
    assert record["kind"] == "ack" and record["src"] == 6 and record["dst"] == 1
    assert record["payload"] is None and record["name"] is None
    assert record["path"] is None and record["tag"] is None and record["op"] is None

    # Heard while the link had already switched to dialect B.
    a_frame = build_frame(DIALECT_A, 1, 6, p.request_status())
    record = capture.frame_record(
        ReceivedFrame(decode(DIALECT_B, a_frame), -70.0, 1, 5), DIALECT_B, AT
    )
    assert record["kind"] == "data" and record["dialect"] == "A"
    assert record["net"] == "1B30" and record["op"] == "B8"

    junk = ReceivedFrame(decode(DIALECT_A, bytes.fromhex("0102030405")), -90.0, 0, 7)
    assert capture.frame_record(junk, DIALECT_A, AT) is None
    assert capture.rx_line(junk) == "RX 7 -90.0 0 0 0102030405"


def test_opcode_keys() -> None:
    """Tags carry the meaning of pairing frames and empty route probes."""
    probe = rx(DIALECT_B, 6, 0xFF, tag=p.ROUTE_PROBE_TAG).frame
    assert capture.opcode_key(probe) == ("tag 06", "route probe")
    announce = rx(DIALECT_B, 0xFF, 1, b"\x55", tag=0x03).frame
    assert capture.opcode_key(announce) == ("tag 03", "pairing announcement")
    assign = rx(DIALECT_B, 1, 0xFF, b"\x07", tag=0x04).frame
    assert capture.opcode_key(assign) == ("tag 04", "id assignment")
    unknown = rx(DIALECT_B, 6, 1, bytes.fromhex("E7 01")).frame
    assert capture.opcode_key(unknown) == ("E7", None)
    odd_tag = rx(DIALECT_B, 6, 1, tag=0x09).frame
    assert capture.opcode_key(odd_tag) == ("tag 09", None)
    # A valid frame too short to carry a tag (only net, src, dst, flags, path).
    short = decode(DIALECT_B, encode(DIALECT_B, NET + bytes([6, 1, 0, 1, 2])))
    assert short.ok and short.tag is None and not short.is_ack
    assert capture.opcode_key(short) == ("tag --", None)
    record = capture.frame_record(ReceivedFrame(short, None, None, None), DIALECT_B, AT)
    assert record["path"] is None and record["payload"] == ""


def test_frame_capture_collects_frames_and_raw_lines() -> None:
    """Frames, undecodable frames and lines land in frames/raw; MACs are masked."""
    times = iter([AT + timedelta(seconds=s) for s in range(5)])
    cap = capture.FrameCapture(lambda: DIALECT_B, now=lambda: next(times))
    cap.start()
    cap.on_frame(rx(DIALECT_B, 6, 1, p.request_status()))
    cap.on_frame(ReceivedFrame(decode(DIALECT_B, b"\x01\x02"), -90.0, 0, 7))
    cap.on_line("# Q termoweb_rx 3.7-esp32 id=01 mac=AA:BB:CC:00:11:22 net=1234")
    cap.stop()
    assert cap.started == "2026-01-02T03:04:05.678+00:00"
    assert cap.ended == "2026-01-02T03:04:09.678+00:00"
    assert [r["op"] for r in cap.frames] == ["B8"]
    assert cap.raw == [
        {"t": "2026-01-02T03:04:07.678+00:00", "line": "RX 7 -90.0 0 0 0102"},
        {
            "t": "2026-01-02T03:04:08.678+00:00",
            "line": "# Q termoweb_rx 3.7-esp32 id=01 mac=XX net=1234",
        },
    ]
    fresh = capture.FrameCapture(lambda: DIALECT_A)
    fresh.on_line("x")
    assert fresh.started is None and fresh.raw[0]["line"] == "x"


def test_redact_masks_networks_identity_and_hex_but_keeps_payload_bytes() -> None:
    """Network ids become NET1/NET2; serial bytes are masked; other bytes stay."""
    frames = [
        capture.frame_record(rx(DIALECT_B, 1, 6, p.flash_display()), DIALECT_B, AT),
        capture.frame_record(rx(DIALECT_B, 6, 1, IDENTITY), DIALECT_B, AT),
        capture.frame_record(
            rx(DIALECT_B, 9, 1, p.request_status(), net=bytes.fromhex("BEEF")),
            DIALECT_B,
            AT,
        ),
        capture.frame_record(
            rx(DIALECT_A, 0xFF, 1, ANNOUNCE_A, net=DIALECT_A.network_id),
            DIALECT_A,
            AT,
        ),
        capture.frame_record(ack(), DIALECT_B, AT),
        capture.frame_record(rx(DIALECT_B, 6, 1), DIALECT_B, AT),
    ]
    raw = [
        capture.raw_record("TX 100 14 0A1234010600010600000000 net=1234", AT),
        capture.raw_record("# Q x mac=AA:BB:CC:00:11:22", AT),
    ]
    red_frames, red_raw = capture.redact(frames, raw)
    assert frames[0]["net"] == "1234" and "air" in frames[0]  # input unchanged
    assert [r["net"] for r in red_frames] == [
        "NET1",
        "NET1",
        "NET2",
        "NET3",
        "NET1",
        "NET1",
    ]
    assert all("air" not in r for r in red_frames)
    assert red_frames[0]["payload"] == "5E01"
    assert red_frames[1]["payload"] == "5B55" + "XX" * 16
    assert red_frames[2]["payload"] == "B8"
    assert red_frames[3]["payload"] == "77" + "XX" * 12
    assert red_frames[4]["payload"] is None
    assert red_frames[5]["payload"] == ""
    assert red_raw[0] == {"t": raw[0]["t"], "line": "TX 100 14 <hex> net=XXXX"}
    assert red_raw[1]["line"] == "# Q x mac=XX"
    text = repr((red_frames, red_raw))
    for secret in ("1234", "BEEF", "1B30", b"X123456".hex().upper(), "AA:BB"):
        assert secret not in text


def test_summary_counts_networks_nodes_and_opcodes() -> None:
    """The summary has frame and ack counts and a histogram with names."""
    frames = [
        capture.frame_record(rx(DIALECT_A, 1, 6, p.flash_display()), DIALECT_A, AT),
        capture.frame_record(rx(DIALECT_A, 6, 1, bytes.fromhex("5F55")), DIALECT_A, AT),
        capture.frame_record(rx(DIALECT_A, 1, 6, p.flash_display()), DIALECT_A, AT),
        capture.frame_record(rx(DIALECT_B, 1, 7, bytes.fromhex("E701")), DIALECT_B, AT),
        capture.frame_record(ack(7, 1), DIALECT_B, AT),
    ]
    assert capture.summarise(frames) == {
        "frames": 5,
        "acks": 1,
        "networks": ["1234"],
        "nodes": [1, 6, 7],
        "opcodes": {
            "5E": {"count": 2, "name": "flash display (identify)"},
            "5F": {"count": 1, "name": "flash display reply"},
            "E7": {"count": 1, "name": None},
        },
    }
    assert capture.summarise(capture.redact(frames, [])[0])["networks"] == ["NET1"]
    assert capture.summarise([]) == {
        "frames": 0,
        "acks": 0,
        "networks": [],
        "nodes": [],
        "opcodes": {},
    }
