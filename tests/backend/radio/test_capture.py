"""Tests for the recent-frame log kept for diagnostics."""

from __future__ import annotations

from datetime import UTC, datetime
import json

from custom_components.termoweb.backend.radio import protocol as p
from custom_components.termoweb.backend.radio.capture import FrameLog
from custom_components.termoweb.backend.radio.dialect import (
    DIALECT_A,
    DIALECT_B,
    build_frame,
    decode,
)
from custom_components.termoweb.backend.radio.link import ReceivedFrame
from tests.fakes.radio_link import (
    IDENTITY_SHORT,
    NET,
    STATUS_SHORT,
    received,
    received_ack,
)

AT = datetime(2026, 1, 2, 3, 4, 5, 678000, tzinfo=UTC)
OTHER = bytes.fromhex("5678")  # synthetic neighbouring network ids
THIRD = bytes.fromhex("9ABC")
IDENTITY_A = bytes(range(0x10, 0x1C))  # synthetic 12-byte dialect-A identity


def _log(size: int = 200) -> FrameLog:
    """Return a frame log with a fixed clock."""
    return FrameLog(size, now=lambda: AT)


def test_frames_are_classified_counted_and_masked() -> None:
    """Every kind is counted; ids are aliased and identity bytes masked."""
    log = _log()
    log.on_rx(received(6, STATUS_SHORT), DIALECT_B, NET)
    log.on_rx(received(6, IDENTITY_SHORT), DIALECT_B, NET)
    log.on_rx(received_ack(6), DIALECT_B, NET)
    log.on_rx(received(0xFF, b"\x55", tag=0x03, network_id=OTHER), DIALECT_B, NET)
    log.on_rx(received(7, STATUS_SHORT, network_id=OTHER), DIALECT_B, NET)
    log.on_rx(received(8, STATUS_SHORT, network_id=THIRD), DIALECT_B, NET)
    log.on_rx(
        ReceivedFrame(decode(DIALECT_B, b"\x05\x01"), None, None, None), DIALECT_B, NET
    )
    log.on_tx(
        build_frame(DIALECT_B, 1, 6, p.request_status(), network_id=NET),
        DIALECT_B,
        NET,
    )

    snapshot = log.snapshot()
    assert snapshot["size"] == 200
    assert snapshot["counts"] == {
        "own_network": 2,
        "ack": 1,
        "foreign_network": 2,
        "pairing_announcement": 1,
        "tx": 1,
        "undecodable": 1,
    }
    status, identity, ack, announce, foreign, third, bad, tx = snapshot["frames"]
    assert status == {
        "t": "2026-01-02T03:04:05.678+00:00",
        "dir": "rx",
        "kind": "own_network",
        "dialect": "B",
        "net": "own",
        "src": 6,
        "dst": 1,
        "tag": 0,
        "op": "B9",
        "name": p.OPCODE_NAMES[0xB9],
        "payload": STATUS_SHORT.hex().upper(),
    }
    assert identity["payload"] == "5B55" + "XX" * (len(IDENTITY_SHORT) - 2)
    assert identity["name"] == "identity reply"
    assert ack["kind"] == "ack" and ack["payload"] is None and ack["op"] is None
    assert announce["kind"] == "pairing_announcement"
    assert announce["net"] == "NET1" and announce["op"] == "tag 03"
    assert announce["name"] == "pairing announcement"
    assert (foreign["net"], third["net"]) == ("NET1", "NET2")
    assert bad == {
        "t": "2026-01-02T03:04:05.678+00:00",
        "dir": "rx",
        "kind": "undecodable",
        "len": 2,
    }
    assert tx["dir"] == "tx" and tx["kind"] == "tx" and tx["net"] == "own"
    assert tx["src"] == 1 and tx["dst"] == 6 and tx["name"] == "status request"

    text = json.dumps(snapshot)
    assert "5678" not in text and "9ABC" not in text and "1234" not in text
    assert IDENTITY_SHORT[-7:].hex().upper() not in text


def test_dialect_a_announcement_identity_is_masked() -> None:
    """A dialect-A announcement keeps only its marker; a re-decoded frame is logged."""
    log = _log()
    announce = received(
        9,
        bytes([0x77]) + IDENTITY_A,
        dialect=DIALECT_A,
        tag=0x03,
        network_id=bytes.fromhex("5678"),
    )
    # The link still decodes in dialect B: the log retries every dialect.
    as_b = ReceivedFrame(decode(DIALECT_B, announce.frame.air), -70.0, 1, 5)
    log.on_rx(as_b, DIALECT_B, NET)

    (record,) = log.snapshot()["frames"]
    assert record["kind"] == "pairing_announcement" and record["dialect"] == "A"
    assert record["net"] == "NET1"
    assert record["payload"] == "77" + "XX" * len(IDENTITY_A)
    assert IDENTITY_A.hex().upper() not in json.dumps(record)


def test_ring_is_bounded_but_counts_keep_growing() -> None:
    """Only the newest ``size`` frames are kept; the counters count every frame."""
    log = _log(size=3)
    for src in range(2, 7):
        log.on_rx(received(src, STATUS_SHORT), DIALECT_B, NET)

    snapshot = log.snapshot()
    assert [record["src"] for record in snapshot["frames"]] == [4, 5, 6]
    assert snapshot["counts"]["own_network"] == 5
    assert snapshot["size"] == 3


def test_empty_payload_is_logged_as_none() -> None:
    """A data frame without a payload logs no payload."""
    log = _log()
    log.on_rx(received(6, b""), DIALECT_B, NET)
    (record,) = log.snapshot()["frames"]
    assert record["payload"] is None and record["op"] == "tag 00"
