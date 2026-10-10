"""Tests for capture records, redaction and the capture summary."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

from radio_fakes import NET

from custom_components.termoweb.backend.radio import capture, protocol as p
from custom_components.termoweb.backend.radio.dialect import (
    DIALECT_A,
    DIALECT_B,
    build_ack,
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
        "dialect": "B",
        "rssi": -61.5,
        "lqi": 9,
        "micros": 4321,
        "air": record["air"],
        "net": "1234",
        "src": 1,
        "dst": 6,
        "flags": 0,
        "path": "0106000000",
        "tag": 0,
        "payload": "5E01",
        "op": "5E",
        "name": "flash display (identify)",
    }
    assert bytes.fromhex(record["air"])[1:3] == NET


def test_ack_and_wrong_dialect_and_undecodable_records() -> None:
    """Acks have no payload; a frame from the other dialect is re-decoded."""
    ack_air = build_ack(DIALECT_B, 6, 1, NET)
    ack = capture.frame_record(
        ReceivedFrame(decode(DIALECT_B, ack_air), None, None, None), DIALECT_B, AT
    )
    assert ack["kind"] == "ack" and ack["src"] == 6 and ack["dst"] == 1
    assert "payload" not in ack and "op" not in ack

    # Heard while the link had already switched to dialect B.
    a_frame = build_frame(DIALECT_A, 1, 6, p.request_status())
    record = capture.frame_record(
        ReceivedFrame(decode(DIALECT_B, a_frame), -70.0, 1, 5), DIALECT_B, AT
    )
    assert record["kind"] == "data" and record["dialect"] == "A"
    assert record["net"] == "1B30" and record["op"] == "B8"

    junk = bytes.fromhex("0102030405")
    bad = capture.frame_record(
        ReceivedFrame(decode(DIALECT_A, junk), -90.0, 0, 7), DIALECT_A, AT
    )
    assert bad["kind"] == "undecodable" and bad["dialect"] is None
    assert bad["air"] == "0102030405" and bad["len"] == 5


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


def test_frame_capture_collects_frames_and_lines() -> None:
    """The capture reads the link's dialect at each frame and masks MACs."""
    dialects = [DIALECT_B]
    times = iter([AT, AT + timedelta(seconds=1)])
    cap = capture.FrameCapture(lambda: dialects[0], now=lambda: next(times))
    cap.on_frame(rx(DIALECT_B, 6, 1, p.request_status()))
    cap.on_line("# Q termoweb_rx 3.7-esp32 id=01 mac=AA:BB:CC:00:11:22 net=1234")
    assert [r["kind"] for r in cap.records] == ["data", "line"]
    assert cap.records[1] == {
        "t": "2026-01-02T03:04:06.678+00:00",
        "kind": "line",
        "line": "# Q termoweb_rx 3.7-esp32 id=01 mac=XX net=1234",
    }
    assert capture.FrameCapture(lambda: DIALECT_A).on_line("x") is None


def test_redact_masks_networks_identity_and_hex_but_keeps_payload_bytes() -> None:
    """Network ids become NET1/NET2; serial bytes are masked; other bytes stay."""
    other_net = bytes.fromhex("BEEF")
    records = [
        capture.frame_record(rx(DIALECT_B, 1, 6, p.flash_display()), DIALECT_B, AT),
        capture.frame_record(rx(DIALECT_B, 6, 1, IDENTITY), DIALECT_B, AT),
        capture.frame_record(
            rx(DIALECT_B, 9, 1, p.request_status(), net=other_net), DIALECT_B, AT
        ),
        capture.frame_record(
            rx(DIALECT_A, 0xFF, 1, ANNOUNCE_A, net=DIALECT_A.network_id),
            DIALECT_A,
            AT,
        ),
        capture.frame_record(
            ReceivedFrame(decode(DIALECT_B, build_ack(DIALECT_B, 6, 1, NET)), 0, 0, 0),
            DIALECT_B,
            AT,
        ),
        capture.frame_record(
            ReceivedFrame(decode(DIALECT_A, b"\x01\x02"), None, None, None),
            DIALECT_A,
            AT,
        ),
        capture.line_record("TX 100 14 0A1234010600010600000000 net=1234", AT),
        capture.frame_record(rx(DIALECT_B, 6, 1), DIALECT_B, AT),
    ]
    redacted = capture.redact(records)
    assert records[0]["net"] == "1234"  # the input is not changed
    assert [r.get("net") for r in redacted] == [
        "NET1",
        "NET1",
        "NET2",
        "NET3",
        "NET1",
        None,
        None,
        "NET1",
    ]
    assert all("air" not in r for r in redacted)
    assert redacted[0]["payload"] == "5E01"
    assert redacted[1]["payload"] == "5B55" + "XX" * 16
    assert redacted[2]["payload"] == "B8"
    assert redacted[3]["payload"] == "77" + "XX" * 12
    assert redacted[5]["len"] == 2
    assert redacted[6]["line"] == "TX 100 14 <hex> net=XXXX"
    assert redacted[7]["payload"] == ""
    text = repr(redacted)
    assert "1234" not in text
    assert "X123456".encode().hex().upper() not in text
    assert "BEEF" not in text and "1B30" not in text


def test_summary_counts_networks_nodes_and_opcodes() -> None:
    """The summary has a histogram with names and lists unknown opcodes."""
    records = [
        capture.frame_record(rx(DIALECT_A, 1, 6, p.flash_display()), DIALECT_A, AT),
        capture.frame_record(rx(DIALECT_A, 6, 1, bytes.fromhex("5F55")), DIALECT_A, AT),
        capture.frame_record(rx(DIALECT_A, 1, 6, p.flash_display()), DIALECT_A, AT),
        capture.frame_record(rx(DIALECT_B, 1, 7, bytes.fromhex("E701")), DIALECT_B, AT),
        capture.frame_record(
            ReceivedFrame(decode(DIALECT_B, build_ack(DIALECT_B, 7, 1, NET)), 0, 0, 0),
            DIALECT_B,
            AT + timedelta(seconds=2),
        ),
        capture.frame_record(
            ReceivedFrame(decode(DIALECT_A, b"\x01"), None, None, None), DIALECT_A, AT
        ),
        capture.line_record("# survey off", AT + timedelta(seconds=3)),
    ]
    summary = capture.summarise(records)
    assert summary == {
        "frames": 5,
        "data_frames": 4,
        "acks": 1,
        "undecodable": 1,
        "lines": 1,
        "first": "2026-01-02T03:04:05.678+00:00",
        "last": "2026-01-02T03:04:08.678+00:00",
        "dialects": {"A": 3, "B": 2},
        "networks": ["1234"],
        "nodes": [1, 6, 7],
        "opcodes": {
            "5E": {"count": 2, "name": "flash display (identify)"},
            "5F": {"count": 1, "name": "flash display reply"},
            "E7": {"count": 1, "name": None},
        },
        "unknown_opcodes": ["E7"],
    }
    redacted = capture.summarise(capture.redact(records))
    assert redacted["networks"] == ["NET1"]

    empty = capture.summarise([])
    assert empty["frames"] == 0 and empty["first"] is None and empty["last"] is None
    assert empty["opcodes"] == {} and empty["networks"] == []
