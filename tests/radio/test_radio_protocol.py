"""Tests for radio payload builders and decoders."""

from __future__ import annotations

from datetime import datetime

import pytest

from custom_components.termoweb.backend.radio import protocol as p
from custom_components.termoweb.backend.radio.dialect import (
    DIALECT_A,
    DIALECT_B,
    build_ack,
    build_frame,
    decode,
)

NET = bytes.fromhex("1234")  # synthetic dialect-B network id

# Dialect-B payloads captured from heater 06 (ops/esp32/README.md and logs).
B_SHORT_STATUS = bytes.fromhex("B921252A02")
B_IDENTITY = bytes.fromhex("5B55 010203040506070809 58313233343536".replace(" ", ""))
B_ENERGY = bytes.fromhex("BDDB002CF039000000")
B_HEATER_REPORT = bytes.fromhex("BEDB002CF039000000")
B_PROGRAM = bytes.fromhex("B155" + "6AAAAAAA9555" * 6 + "6AAAAAAA95")

# Dialect-A payloads from ha-termoweb-local's captures.
A_E5 = decode(
    DIALECT_A,
    bytes.fromhex("E59C885DB6A1C825565F4A9C5850E47E05BCB4E185AE81C3E7BFE380F1"),
).payload
A_E6 = decode(
    DIALECT_A, bytes.fromhex("E69C885BB6A1CE25565F4A9CB7E7C47F2EBE5432FE915DF1F89A3003")
).payload
A_E3 = decode(
    DIALECT_A,
    bytes.fromhex("E39C885DB6A1C825565F4A9C5850E47E05BAB4F587AD3BF1C286E341886B6A"),
).payload
A_E4 = bytes.fromhex("b90e24340200ec341daf00a01d0064a3")
A_ENERGY = decode(
    DIALECT_A, bytes.fromhex("EF9C885BB6A1CE25565F4A9CB3E9F2E8034AD8")
).payload
A_IDENTITY = bytes.fromhex("5b7701010104a1b2c3d4e5f6071829304a5b0a1a")


# --- builders ---------------------------------------------------------------


def test_setpoint_and_override() -> None:
    """Setpoint and override carry half degrees and check the 7-35 C range."""
    assert p.set_setpoint(25.5) == bytes([0xB4, 0x02, 0x33])
    assert p.set_setpoint(7.0) == bytes([0xB4, 0x02, 14])
    assert p.set_setpoint(35) == bytes([0xB4, 0x02, 70])
    assert p.set_override(21.0) == bytes([0xB4, 0x03, 42])
    for bad in (6.5, 35.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="outside"):
            p.set_setpoint(bad)
        with pytest.raises(ValueError, match="outside"):
            p.set_override(bad)


def test_set_mode() -> None:
    """Only auto, manual and off are settable with a bare mode write."""
    assert p.set_mode(p.MODE_AUTO) == b"\xb4\x01"
    assert p.set_mode(p.MODE_MANUAL) == b"\xb4\x02"
    assert p.set_mode(p.MODE_OFF) == b"\xb4\x04"
    for bad in (p.MODE_OVERRIDE, 0, 5):
        with pytest.raises(ValueError, match="mode"):
            p.set_mode(bad)


def test_write_presets() -> None:
    """Presets must be half-degree multiples, strictly increasing, in range."""
    assert p.write_presets(7.0, 18.0, 21.0) == bytes([0xB6, 14, 36, 42])
    with pytest.raises(ValueError, match="multiple"):
        p.write_presets(7.2, 18.0, 21.0)
    with pytest.raises(ValueError, match="multiple"):
        p.write_presets(7.0, float("nan"), 21.0)
    for values in (
        (18.0, 18.0, 21.0),
        (7.0, 21.0, 18.0),
        (6.5, 8.0, 9.0),
        (7.0, 8.0, 35.5),
    ):
        with pytest.raises(ValueError, match="presets must"):
            p.write_presets(*values)


def test_toggles_and_simple_payloads() -> None:
    """Toggles, confirmations, verdicts and requests are fixed byte strings."""
    assert p.set_toggle(p.TOGGLE_BOOST, True) == b"\xd2\x01"
    assert p.set_toggle(p.TOGGLE_LOCK, False) == b"\xba\x00"
    with pytest.raises(ValueError, match="toggle"):
        p.set_toggle(0xB4, True)
    assert p.confirm_report() == b"\x57\x55"
    assert p.power_verdict() == b"\xbf\x01"
    assert p.power_verdict(granted=False) == b"\xbf\x00"
    assert p.request_status() == b"\xb8"
    assert p.request_program() == b"\xb0"
    assert p.request_energy() == b"\xbc"
    assert p.request_identity() == b"\x5a"
    assert p.request_advanced() == b"\xda"


def test_sync_clock_per_dialect() -> None:
    """Dialect A ends the EB clock sync with 03; dialect B (8 bytes) does not."""
    when = datetime(2026, 10, 9, 16, 52, 9)  # a Friday
    assert p.sync_clock(when, False, DIALECT_A) == bytes(
        [0x52, 26, 10, 9, 5, 16, 52, 9, 0x03]
    )
    reg_b = p.sync_clock(when, True, DIALECT_B)
    assert reg_b == bytes([0x51, 26, 10, 9, 5, 16, 52, 9])
    assert len(reg_b) == 8
    sunday = datetime(2026, 9, 6, 17, 52, 14)
    assert p.sync_clock(sunday, False, DIALECT_B)[4] == 0


def test_write_program_rotates_monday_first_to_sunday_first() -> None:
    """Hourly Monday-first input is doubled to half hours and rotated to wire order."""
    week = [[0] * 24 for _ in range(7)]
    week[0][0] = 2  # Monday 00:00 comfort
    week[6][23] = 1  # Sunday 23:00 eco
    payload = p.write_program(week)
    assert payload[0] == 0xB2 and len(payload) == 85
    record = p.decode_program(bytes([0xB1]) + payload[1:])
    assert record is not None and record.resolution == 48
    # Wire day 0 is Sunday: its last hour is eco; wire day 1 (Monday) starts comfort.
    assert record.hourly[23] == 1
    assert record.hourly[24] == 2
    assert record.hourly_monday_first[0] == 2
    assert record.hourly_monday_first[-1] == 1
    assert record.hourly_only
    half_hourly = [[0] * 48 for _ in range(7)]
    half_hourly[0][1] = 2
    rec2 = p.decode_program(bytes([0xB1]) + p.write_program(half_hourly)[1:])
    assert rec2.slots_monday_first[:2] == (0, 2)
    assert rec2.hourly_monday_first[0] is None
    assert not rec2.hourly_only


def test_write_program_validation() -> None:
    """Wrong day counts, slot counts and slot values are refused."""
    with pytest.raises(ValueError, match="7 days"):
        p.write_program([[0] * 24] * 6)
    with pytest.raises(ValueError, match="24"):
        p.write_program([[0] * 23] * 7)
    with pytest.raises(ValueError, match="24"):
        p.write_program([[0] * 24] * 6 + [[0] * 48])
    with pytest.raises(ValueError, match="slot value"):
        p.write_program([[3] * 24] * 7)


def test_write_presets_with_mode() -> None:
    """Dialect B appends the mode to B6 (verified: B6 21 25 2A 04 sets off)."""
    assert p.write_presets(16.5, 18.5, 21.0, p.MODE_OFF) == bytes.fromhex("B621252A04")
    assert p.write_presets(16.5, 18.5, 21.0, p.MODE_MANUAL) == bytes.fromhex(
        "B621252A02"
    )
    with pytest.raises(ValueError, match="mode"):
        p.write_presets(16.5, 18.5, 21.0, p.MODE_OVERRIDE)


def test_write_program_hourly_wire_echoes_dialect_b_read() -> None:
    """A 24-slot write of the decoded B1 record is B2 + the same 42 bytes (B3 55)."""
    record = p.decode_program(B_PROGRAM)
    week = [
        record.hourly_monday_first[d * 24 : (d + 1) * 24]
        for d in range(p.DAYS_PER_WEEK)
    ]
    assert p.write_program(week, wire_slots=p.SLOTS_HOURLY) == b"\xb2" + B_PROGRAM[1:]
    half = [[slot for slot in day for _ in range(2)] for day in week]
    assert p.write_program(half, wire_slots=p.SLOTS_HOURLY) == b"\xb2" + B_PROGRAM[1:]
    half[0] = [2] + half[0][1:]
    with pytest.raises(ValueError, match="half-hour"):
        p.write_program(half, wire_slots=p.SLOTS_HOURLY)
    with pytest.raises(ValueError, match="wire_slots"):
        p.write_program(week, wire_slots=12)


def test_rotate_week_round_trip_and_validation() -> None:
    """Rotation to the wire and back is the identity; wrong sizes raise."""
    week = list(range(7 * 24))
    wire = p.rotate_week(week, 24, to_wire=True)
    assert wire[:24] == week[6 * 24 :]
    assert p.rotate_week(wire, 24, to_wire=False) == week
    with pytest.raises(ValueError, match="expected"):
        p.rotate_week(week[:-1], 24, to_wire=True)


def test_reply_predicate_and_verdict() -> None:
    """Replies match on opcode + 1 and optional payload lengths."""
    frame = decode(
        DIALECT_B, build_frame(DIALECT_B, 6, 1, B_SHORT_STATUS, network_id=NET)
    )
    assert p.reply_predicate(p.OP_STATUS)(frame)
    assert p.reply_predicate(p.OP_STATUS, 5, 14)(frame)
    assert not p.reply_predicate(p.OP_STATUS, 14)(frame)
    assert not p.reply_predicate(p.OP_PROGRAM_READ)(frame)
    empty = decode(DIALECT_B, build_ack(DIALECT_B, 6, 1, NET))
    assert not p.reply_predicate(p.OP_STATUS)(empty)
    assert p.reply_verdict(b"\x53\x55", 0x52) is True
    assert p.reply_verdict(b"\x53\x56", 0x52) is False
    assert p.reply_verdict(b"\x53\x57", 0x52) is None
    assert p.reply_verdict(b"\x54\x55", 0x52) is None
    assert p.reply_verdict(b"\x53", 0x52) is None


# --- status -------------------------------------------------------------------


def test_status_short_form() -> None:
    """The dialect-B short status carries presets and mode only."""
    record = p.decode_status(B_SHORT_STATUS)
    assert record is not None
    assert record.form == "short"
    assert (record.anti_frost_c, record.eco_c, record.comfort_c) == (16.5, 18.5, 21.0)
    assert record.mode_code == 2 and record.mode == "manual"
    for name in (
        "room_temp_c",
        "setpoint_c",
        "measured_power_w",
        "duty",
        "flags",
        "boost_end_day",
        "boost_end_min",
    ):
        assert getattr(record, name) is None
    assert record.raw == B_SHORT_STATUS


def test_status_e5_report() -> None:
    """A dialect-A E5 report decodes every field it carries."""
    record = p.decode_status(A_E5)
    assert record.form == "E5"
    assert (record.anti_frost_c, record.eco_c, record.comfort_c) == (7.0, 23.0, 23.5)
    assert record.mode_code == 2
    assert record.room_temp_c == 25.0
    assert record.setpoint_c == 25.5
    assert record.measured_power_w == 790.0
    assert record.duty == 0x32
    assert record.flags == p.FLAG_ACTIVE
    assert record.boost_end_day is None


def test_status_e6_e4_e3() -> None:
    """E6 drops the 56 marker; E4 and E3 add the boost tail."""
    e6 = p.decode_status(A_E6)
    assert e6.form == "E6" and e6.measured_power_w == 1846.5
    e4 = p.decode_status(A_E4)
    assert e4.form == "E4"
    assert (e4.anti_frost_c, e4.eco_c, e4.comfort_c) == (7.0, 18.0, 26.0)
    assert (e4.room_temp_c, e4.setpoint_c, e4.measured_power_w) == (23.6, 26.0, 759.9)
    assert e4.flags == p.FLAG_RUNBACK | p.FLAG_BOOST
    assert (e4.boost_end_day, e4.boost_end_min) == (6, 1187)
    e3 = p.decode_status(A_E3)
    assert e3.form == "E3" and e3.mode == "off"
    assert (e3.room_temp_c, e3.setpoint_c, e3.measured_power_w) == (23.8, 24.5, 752.6)
    assert (e3.boost_end_day, e3.boost_end_min) == (0, 1141)


def test_status_unknown_mode_name() -> None:
    """An unknown mode code keeps its number and has no name."""
    assert p.decode_status(bytes([0xB9, 14, 36, 42, 0x09])).mode is None


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"\xb9",
        bytes.fromhex("B921252A"),
        bytes.fromhex("BA21252A02"),  # wrong marker
        bytes([0x57]) + A_E6,  # 15 bytes without the 56 report marker
        bytes([0x56, 0xBA]) + A_E6[1:],  # report with the wrong record marker
        A_E6 + b"\x00",  # 15 bytes, no 56 marker
        bytes(30),
    ],
)
def test_status_rejects_garbage(payload) -> None:
    """Wrong lengths and wrong markers decode to None."""
    assert p.decode_status(payload) is None


# --- program ------------------------------------------------------------------


def test_program_dialect_b_capture() -> None:
    """The captured 43-byte C9 reply is an hourly program, the same every day."""
    assert len(B_PROGRAM) == 43
    record = p.decode_program(B_PROGRAM)
    assert record.resolution == 24
    assert len(record.slots) == len(record.hourly) == 168
    day = [1] * 5 + [2] * 16 + [1] * 3
    assert list(record.hourly) == day * 7
    assert record.raw == B_PROGRAM[1:]
    assert record.hourly_only
    assert record.slots_monday_first == record.slots


def test_program_report_and_unknown_codes() -> None:
    """A 56 B1 report decodes; code 3 and split hours become None."""
    raw = bytes([0xFF]) + bytes([0x5A]) + bytes(82)
    report = p.decode_program(bytes([0x56, 0xB1]) + raw)
    assert report.resolution == 48
    assert report.slots[:4] == (None, None, None, None)
    assert report.slots[4:8] == (1, 1, 2, 2)
    assert report.hourly[:4] == (None, None, 1, 2)


@pytest.mark.parametrize(
    "payload",
    [b"", b"\xb1", bytes([0xB2]) + bytes(42), bytes([0xB1]) + bytes(41), bytes(85)],
)
def test_program_rejects_garbage(payload) -> None:
    """Wrong lengths or opcodes decode to None."""
    assert p.decode_program(payload) is None


# --- identity -----------------------------------------------------------------


def test_identity_short_form() -> None:
    """The dialect-B 5B 55 identity keeps a 16-byte tail and its ASCII serial."""
    record = p.decode_identity(B_IDENTITY)
    assert record.form == "short"
    assert record.node_id is None
    assert len(record.tail) == 16
    assert record.ascii_serial == "X123456"
    no_ascii = p.decode_identity(bytes([0x5B, 0x55]) + bytes(16))
    assert no_ascii.ascii_serial is None


def test_identity_e0_form() -> None:
    """The E0 5B 77 identity carries a node id and a 12-byte tail."""
    record = p.decode_identity(A_IDENTITY)
    assert record.form == "E0"
    assert record.node_id == 0x04
    assert record.tail == bytes.fromhex("a1b2c3d4e5f6071829304a5b")
    assert record.ascii_serial is None


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"\x5b",
        bytes([0x5C, 0x55]) + bytes(16),
        bytes([0x5B, 0x77]) + bytes(16),  # E0 marker at the short length
        bytes([0x5B, 0x55]) + bytes(18),  # short marker at the E0 length
        bytes([0x5B, 0x55]) + bytes(10),
    ],
)
def test_identity_rejects_garbage(payload) -> None:
    """Anything but the two known identity forms decodes to None."""
    assert p.decode_identity(payload) is None


# --- energy -------------------------------------------------------------------


def test_energy_forms() -> None:
    """Dialect A carries a Wh counter; dialect B's BD reply is a power record."""
    a = p.decode_energy(A_ENERGY)
    assert a.energy_wh == 1620009
    assert a.measured_power_w is None
    b = p.decode_energy(B_ENERGY)
    assert b.energy_wh is None
    assert b.measured_power_w == 1150.4
    assert b.raw == B_ENERGY
    for bad in (b"", b"\xbd", bytes([0xBD]) + bytes(5), B_HEATER_REPORT):
        assert p.decode_energy(bad) is None


# --- power request ------------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "form", "watts"),
    [
        (B_HEATER_REPORT, "B", 1150.4),
        (bytes.fromhex("BEDA002CA9390B0100"), "B", 1143.3),
        (bytes.fromhex("BEDA002CC8390B0100"), "B", 1146.4),
        (bytes([0xBE, 0x1D, 0x66]), "A", 752.6),
    ],
)
def test_power_request_forms(payload, form, watts) -> None:
    """Both BE forms decode the measured full-load power and keep the raw bytes."""
    record = p.decode_power_request(payload)
    assert record.form == form
    assert record.measured_power_w == watts
    assert record.raw == payload
    assert p.classify_unsolicited(payload) is p.Unsolicited.POWER_REQUEST


@pytest.mark.parametrize(
    "payload", [b"", b"\xbe", bytes([0xBE, 0x01]), bytes([0xBE]) + bytes(5), B_ENERGY]
)
def test_power_request_rejects_garbage(payload) -> None:
    """Other lengths or opcodes are not power requests."""
    assert p.decode_power_request(payload) is None


# --- unsolicited --------------------------------------------------------------


def test_classify_unsolicited_payloads() -> None:
    """Bare payloads classify by their leading bytes and length."""
    u = p.Unsolicited
    assert p.classify_unsolicited(b"\x50") is u.REGISTRATION
    assert p.classify_unsolicited(A_E5) is u.REPORT
    assert p.classify_unsolicited(bytes([0x56, 0xB1]) + bytes(84)) is u.PROGRAM_REPORT
    assert p.classify_unsolicited(B_HEATER_REPORT) is u.POWER_REQUEST
    assert p.classify_unsolicited(bytes([0xBE, 0x1E, 0xDC])) is u.POWER_REQUEST
    assert p.classify_unsolicited(bytes([0xBE, 0x01])) is u.UNKNOWN
    assert p.classify_unsolicited(b"") is u.UNKNOWN
    assert p.classify_unsolicited(B_SHORT_STATUS) is u.UNKNOWN


def test_classify_unsolicited_frames() -> None:
    """Frames add the ack and route-probe cases payloads cannot express."""
    u = p.Unsolicited
    ack = decode(DIALECT_B, bytes.fromhex("081234010680A0E4"))
    probe = decode(DIALECT_B, bytes.fromhex("0E1234060200060201010106F691"))
    reg = decode(DIALECT_B, bytes.fromhex("0F1234060100060101010100507222"))
    plain = decode(DIALECT_B, build_frame(DIALECT_B, 6, 1, b"", network_id=NET))
    assert p.classify_unsolicited(ack) is u.ACK
    assert p.classify_unsolicited(probe) is u.ROUTE_PROBE
    assert p.classify_unsolicited(reg) is u.REGISTRATION
    assert p.classify_unsolicited(plain) is u.UNKNOWN
    assert str(u.REPORT) == "report"
