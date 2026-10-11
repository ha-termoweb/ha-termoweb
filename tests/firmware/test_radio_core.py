"""Host tests of firmware/radio_core: dialects, acks, parser and replies.

The core is compiled with the host gcc (see conftest.py) and driven through
a fake CC1101. Expected bytes come from the integration's own codec
(``backend/radio/dialect.py``), so the firmware is checked against the host
that decodes it, not against itself. Expected lines follow the ESP32
firmware's formats (esphome/termoweb_radio.h, 3.7-esp32).
"""

from __future__ import annotations

import random

import pytest

from custom_components.termoweb.backend.radio.dialect import (
    DIALECT_A,
    DIALECT_B,
    build_frame,
    crc16_ccitt_a,
    crc16_modbus,
    decode,
    encode,
    frame_total_length,
    keystream,
)
from custom_components.termoweb.backend.radio.link import (
    BANNER_PREFIX,
    parse_query_line,
    parse_rx_line,
    supports_survey,
)
from custom_components.termoweb.backend.radio.survey import parse_run_tokens

from .conftest import Core, nanocul_make_var

VERSION = nanocul_make_var("VERSION")
BANNER = (
    f"# termoweb_rx {VERSION} freq=869.525 rate=9.6k sync=2DE5 mode=dynamic tx=paC0"
)
QUERY = (
    f"# Q termoweb_rx {VERSION} freq=869.525 pa=C0 sync=2DE5 mode=dynamic "
    "autoack=off id=01 dialect=A net=1B30"
)
NET_B = bytes.fromhex("1234")  # synthetic network id
CC1101_PKTLEN, CC1101_PKTCTRL0 = 0x06, 0x08
CC1101_SYNC0, CC1101_FREQ2, CC1101_FREQ1, CC1101_FREQ0 = 0x05, 0x0D, 0x0E, 0x0F
CC1101_PATABLE, CC1101_SFRX, CC1101_SRX = 0x3E, 0x3A, 0x34
CC1101_PKTSTATUS = 0x38
RX_IDLE, RX_DONE = 0, 3


def ack_frame(dialect, net: bytes, acker: int, src: int) -> bytes:
    """Return the on-air ack the host codec expects for ``acker`` acking ``src``."""
    return encode(dialect, net + bytes([acker, src, 0x80]))


def c_ack(core: Core, dialect: int, net: int, acker: int, src: int) -> bytes:
    """Return the core's ack for these ids."""
    out = bytes(8)
    core.rc_build_ack(dialect, net, acker, src, out)
    return out


# --------------------------------------------------------------- protocol


def test_crcs_and_keystream_match_the_host_codec(core_lib: Core) -> None:
    """CRC-16/CCITT-1D0F, CRC-16/MODBUS and PN9 equal dialect.py's."""
    rng = random.Random(7)
    for length in (0, 1, 6, 13, 64, 253):
        data = bytes(rng.randrange(256) for _ in range(length))
        assert core_lib.rc_crc_ccitt(data, length) == crc16_ccitt_a(data)
        assert core_lib.rc_crc_modbus(data, length) == crc16_modbus(data)
    stream = keystream(255)
    assert stream[:8] == bytes.fromhex("FF87B859B7A1CC24")
    # rc_field() in dialect A is air ^ keystream at that index.
    for index in range(0, 255, 17):
        air = bytes(256)
        assert core_lib.rc_field(0, air, index) == stream[index]
        assert core_lib.rc_field(1, bytes([0x5A]) * 256, index) == 0x5A


def test_frame_total_follows_each_dialects_length_rule(core_lib: Core) -> None:
    """Byte 0 gives the total as in the host codec; outside [8, 255] is noise."""
    for byte0 in range(256):
        for dialect, code in ((DIALECT_A, 0), (DIALECT_B, 1)):
            total = frame_total_length(dialect, byte0)
            expected = total if 8 <= total <= 255 else 0
            assert core_lib.rc_frame_total(code, byte0) == expected


@pytest.mark.parametrize(("dialect", "code"), [(DIALECT_A, 0), (DIALECT_B, 1)])
def test_crc_verdict_matches_decode(core_lib: Core, dialect, code: int) -> None:
    """rc_frame_crc_ok agrees with decode().ok on valid, corrupted and foreign frames."""
    rng = random.Random(code)
    other = 1 - code
    for _ in range(300):
        payload = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 90)))
        air = build_frame(
            dialect, rng.randrange(256), rng.randrange(256), payload, network_id=NET_B
        )
        assert core_lib.rc_frame_crc_ok(code, air, len(air)) == 1
        assert core_lib.rc_frame_crc_ok(other, air, len(air)) == int(
            decode(DIALECT_B if other else DIALECT_A, air).ok
        )
        bad = bytearray(air)
        bad[rng.randrange(len(bad))] ^= 1 << rng.randrange(8)
        assert core_lib.rc_frame_crc_ok(code, bytes(bad), len(bad)) == int(
            decode(dialect, bytes(bad)).ok
        )
    assert core_lib.rc_frame_crc_ok(code, bytes(4), 4) == 0


def test_acks_match_the_host_codec_byte_for_byte(core_lib: Core) -> None:
    """Both dialects' acks equal encode(net + acker + sender + 0x80)."""
    for acker, src in ((0x01, 0x04), (0x01, 0x06), (0x7E, 0x22), (0xFF, 0x00)):
        assert c_ack(core_lib, 0, 0x1B30, acker, src) == ack_frame(
            DIALECT_A, DIALECT_A.network_id, acker, src
        )
        assert c_ack(core_lib, 1, 0x1234, acker, src) == ack_frame(
            DIALECT_B, NET_B, acker, src
        )
        for code, dialect in ((0, DIALECT_A), (1, DIALECT_B)):
            frame = decode(dialect, c_ack(core_lib, code, 0x1234, acker, src))
            assert frame.is_ack and (frame.src, frame.dst) == (acker, src)


def test_verify_ack_vectors(core_lib: Core) -> None:
    """termoweb_rx's verify_ack.py checks, plus the ESP32 boot self-test vectors."""
    # PROTOCOL.md 5.5 worked example: heater 04 acking gateway 01.
    assert c_ack(core_lib, 0, 0x1B30, 0x04, 0x01) == bytes.fromhex("FA9C885DB621FBD1")
    assert c_ack(core_lib, 0, 0x1B30, 0x01, 0x06) == bytes.fromhex("FA9C8858B12189B6")
    assert c_ack(core_lib, 1, 0x1234, 0x01, 0x06) == bytes.fromhex("081234010680A0E4")
    assert core_lib.rc_self_test() == 1


def test_freq_word_and_rssi(core_lib: Core) -> None:
    """The 32-bit FREQ word equals the ESP32's 64-bit rounding over the band."""
    for khz in [779000, 868000, 869525, 915000, 928000, *range(779000, 928001, 997)]:
        assert core_lib.rc_freq_word(khz) == (khz * 65536 + 13000) // 26000
    assert core_lib.rc_freq_word(869525) == 0x21717A
    assert core_lib.rc_rssi_dbm10(0x00) == -740
    assert core_lib.rc_rssi_dbm10(0x2C) == -520
    assert core_lib.rc_rssi_dbm10(0x80) == -1380


# ------------------------------------------------------------ boot, V, Q


def test_boot_prints_the_banner_and_arms_the_receiver(core_lib: Core) -> None:
    """Boot: banner (no mac= on a nanoCUL), chip set up, RX armed, edges unmasked."""
    assert core_lib.start() == [BANNER]
    assert BANNER.startswith(BANNER_PREFIX)
    assert core_lib.host_reg(CC1101_SYNC0) == 0xE5
    assert core_lib.host_reg(CC1101_PATABLE) == 0xC0
    regs = [core_lib.host_reg(r) for r in (CC1101_FREQ2, CC1101_FREQ1, CC1101_FREQ0)]
    assert regs == [0x21, 0x71, 0x7A]
    assert core_lib.host_reg(CC1101_PKTCTRL0) == 0x02
    assert core_lib.host_strobes(CC1101_SRX) == 1
    assert core_lib.host_masked() == 0


def test_mac_field_is_appended_when_the_port_has_one(core_lib: Core) -> None:
    """A port with a MAC (the ESP32) gets mac= appended to banner and Q."""
    assert core_lib.start(b"AA:BB:CC:DD:EE:FF") == [f"{BANNER} mac=AA:BB:CC:DD:EE:FF"]
    assert core_lib.send("Q") == [f"{QUERY} mac=AA:BB:CC:DD:EE:FF"]


def test_query_is_dialect_capable_for_the_integration(core: Core) -> None:
    """The integration reads the nanoCUL's Q line as dialect- and survey-capable."""
    assert core.send("V\nQ\n") == [BANNER, QUERY]
    info = parse_query_line(QUERY)
    assert info is not None
    assert info.version == VERSION
    assert info.dialect == "A"
    assert info.mac is None
    assert info.station_id == 1 and info.autoack is False
    assert supports_survey(info)


# --------------------------------------------------------------- commands


def test_session_commands(core: Core) -> None:
    """A, I, N, Y, X and their replies, reflected in Q."""
    assert core.send("A1\nI22\nNABCD\nY1\nX\n") == [
        "# autoack=on",
        "# id=22",
        "# net=ABCD",
        "# dialect=B sync=2DD4",
        "# raw=on",
    ]
    assert core.host_reg(CC1101_SYNC0) == 0xD4
    assert core.host_masked() == 0
    query_b = (
        f"# Q termoweb_rx {VERSION} freq=869.525 pa=C0 sync=2DD4 mode=dynamic "
        "autoack=on id=22 dialect=B net=ABCD"
    )
    assert core.send("Q") == [query_b]
    assert core.send("X\nA0\nY0\n") == [
        "# raw=off",
        "# autoack=off",
        "# dialect=A sync=2DE5",
    ]
    assert core.send("A2Y9\n") == ["# A? expected 0 or 1", "# Y? expected 0 or 1"]


def test_arguments_with_command_letters_stay_arguments(core: Core) -> None:
    """Hex letters inside an argument never run as commands (the old stick did)."""
    assert core.send("NAFDA\n") == ["# net=AFDA"]
    assert core.host_auto_ack() == 0 and core.host_freq_khz() == 869525
    assert core.send("IFA\n") == ["# id=FA"]
    assert core.send("Nd0a1\n") == ["# net=D0A1"]
    assert core.host_net() == 0xD0A1
    lines = core.send("TADFA\n")
    assert lines[0].startswith("TX ") and lines[0].endswith(" 2 ADFA")
    assert core.tx_frame() == bytes.fromhex("ADFA")
    # A non-hex byte ends the argument with an error and is consumed.
    assert core.send("N1Q34\n") == ["# N? expected 4 hex digits"]
    assert core.host_net() == 0xD0A1


@pytest.mark.parametrize(
    ("text", "reply"),
    [
        ("I1\n", "# I? expected 2 hex digits"),
        ("I\n", "# I? expected 2 hex digits"),
        ("I0102\n", "# I? expected 2 hex digits"),
        ("N123\n", "# N? expected 4 hex digits"),
        ("N123456\n", "# N? expected 4 hex digits"),
        ("T\n", "TXERR empty or bad hex"),
        ("T123\n", "TXERR empty or bad hex"),
        ("T12G4\n", "TXERR empty or bad hex"),
        ("F\n", "# F? expected kHz"),
        ("F778999\n", "# F? expected kHz"),
        ("F928001\n", "# F? expected kHz"),
        ("F12345678\n", "# F? expected kHz"),
        ("F86x\n", "# F? expected kHz"),
        ("R\n", "# R? expected seconds"),
        ("R12345\n", "# R? expected seconds"),
    ],
)
def test_bad_arguments(core: Core, text: str, reply: str) -> None:
    """Every malformed argument gets the ESP32's error line, once."""
    assert core.send(text) == [reply]
    assert core.host_pstate() == 0
    assert core.host_tx_count() == 0


def test_half_command_times_out_after_one_second(core: Core) -> None:
    """A command letter without its argument fails after 1 s of silence."""
    for text, reply in (
        ("A", "# A? expected 0 or 1"),
        ("Y", "# Y? expected 0 or 1"),
        ("I0", "# I? expected 2 hex digits"),
        ("N12", "# N? expected 4 hex digits"),
        ("T0102", "TXERR empty or bad hex"),
        ("F8695", "# F? expected kHz"),
        ("R1", "# R? expected seconds"),
    ):
        assert core.send(text) == []
        core.host_advance(1_000_000)
        core.rc_poll()
        assert core.lines() == []
        core.host_advance(1)
        core.rc_poll()
        assert core.lines() == [reply]
    core.rc_poll()
    assert core.lines() == []


def test_transmit_long_frame(core: Core) -> None:
    """T sends up to 255 bytes verbatim (FIFO refill past 64) and re-arms RX."""
    frame = bytes(range(99))
    lines = core.send(f"T{frame.hex()}\n")
    assert len(lines) == 1
    parts = lines[0].split()
    assert parts[0] == "TX" and parts[2] == "99" and parts[3] == frame.hex().upper()
    assert core.tx_frame() == frame
    assert core.host_reg(CC1101_PKTLEN) == 99
    assert core.host_reg(CC1101_PKTCTRL0) == 0x02
    assert core.host_masked() == 0 and core.host_rx_state() == RX_IDLE
    full = bytes(range(255))
    assert core.send(f"T{full.hex()}\n")[0].split()[2] == "255"
    assert core.tx_frame() == full
    assert core.send(f"T{full.hex()}00\n") == ["TXERR empty or bad hex"]


def test_transmit_errors(core: Core) -> None:
    """A stuck MARCSTATE and a FIFO underflow are reported, RX re-armed."""
    core.host_set_marcstate_tx(0x13)
    assert core.send("T0102\n") == ["TXERR marcstate=13"]
    core.host_set_marcstate_tx(0x16)
    assert core.send("T0102\n") == ["TXERR underflow"]
    assert core.host_masked() == 0


def test_frequency_command(core: Core) -> None:
    """F<kHz> retunes the chip and shows up in V and Q."""
    word = (868300 * 65536 + 13000) // 26000
    assert core.send("F868300\n") == [f"# freq=868.300 word={word:06X}"]
    regs = [core.host_reg(r) for r in (CC1101_FREQ2, CC1101_FREQ1, CC1101_FREQ0)]
    assert regs == [word >> 16, (word >> 8) & 0xFF, word & 0xFF]
    assert core.send("V")[0].startswith(f"# termoweb_rx {VERSION} freq=868.300 ")
    assert core.send("F869525\n") == ["# freq=869.525 word=21717A"]
    # A reset goes back to the power-on default.
    core.send("F868000\n")
    assert core.start() == [BANNER]


def test_diagnostic_dump(core: Core) -> None:
    """D reports chip state and the sync / rxreset counters."""
    core.host_set_status(CC1101_PKTSTATUS, 0x2A)
    assert core.send("D") == [
        "# marcstate=01 pktstatus=2A rxbytes=00 syncs=0 rxreset=0"
    ]
    core.rc_on_gdo0(1)
    assert core.send("D") == [
        "# marcstate=01 pktstatus=2A rxbytes=00 syncs=1 rxreset=0"
    ]
    assert core.host_lock_depth() == 0


def test_survey_commands(core: Core) -> None:
    """R starts the platform's survey (capped at 600 s); R0 ends it; TX waits."""
    assert core.send("R0\n") == []
    assert core.host_survey_calls() == 0
    assert core.send("R700\n") == ["# survey on secs=600 floor=-100.0 thr=-92.0"]
    assert core.host_last_survey_secs() == 600
    assert core.send("T01\n") == ["TXERR survey active"]
    assert core.host_tx_count() == 0
    assert core.send("R0\n") == ["# survey off bursts=0"]
    assert core.host_last_survey_secs() == 0
    assert core.host_reg(CC1101_PKTCTRL0) == 0x02 and core.host_masked() == 0


def test_survey_lines_parse_on_the_host(core: Core) -> None:
    """RAWB/RAW/RAWE lines have the ESP32's format and parse as runs."""
    core.rc_print_survey_on(30, -1005, -925)
    core.rc_print_rawb(3, -655)
    core.rc_print_raw_start(3)
    for level, micros in ((1, 104), (0, 208), (1, 32767)):
        core.rc_print_raw_run(level, micros)
    core.rc_print_eol()
    core.rc_print_rawe(3, 3)
    core.rc_print_survey_off(3)
    core.rc_print_overrun()
    lines = core.lines()
    assert lines == [
        "# survey on secs=30 floor=-100.5 thr=-92.5",
        "RAWB 3 -65.5",
        "RAW 3 +104 -208 +32767",
        "RAWE 3 3",
        "# survey off bursts=3",
        "# uart overrun",
    ]
    assert parse_run_tokens(lines[2].split()[2:]) == [(1, 104), (0, 208), (1, 32767)]


# ------------------------------------------------------------ receive path


def poll_a(src: int = 0x06, dst: int = 0x01, flags: int = 0) -> bytes:
    """Return a dialect A data frame."""
    return build_frame(DIALECT_A, src, dst, b"\x57\x55", flags=flags)


def test_rx_line_dialect_a(core: Core) -> None:
    """A received frame prints as the ESP32's RX line; length set from byte 0."""
    air = poll_a()
    core.host_rx_push(air[:4], 4)
    core.rc_on_gdo2()
    assert core.host_reg(CC1101_PKTLEN) == len(air)
    assert core.host_reg(CC1101_PKTCTRL0) == 0x00
    core.host_rx_push(air[4:] + b"\x2c\x95", len(air) - 4 + 2)
    core.rc_on_gdo0(5000)
    core.rc_service()
    line = f"RX 5000 -52.0 21 1 {air.hex().upper()}"
    assert core.lines() == [line]
    parsed = parse_rx_line(line)
    assert parsed is not None and parsed[0] == air and parsed[1] == -52.0
    assert decode(DIALECT_A, parsed[0]).ok
    assert core.host_frames_ok() == 1 and core.host_rx_state() == RX_IDLE
    assert core.host_tx_count() == 0  # auto-ack is off by default
    core.send("X")
    core.receive(air)
    assert core.lines()[0].endswith(f"{air.hex().upper()} 2C95")


def test_long_frame_is_received_in_chunks(core: Core) -> None:
    """A 99-byte frame (past the 64-byte FIFO) arrives whole."""
    air = build_frame(DIALECT_A, 6, 1, bytes(range(85)))
    assert len(air) == 99
    core.receive(air)
    assert core.lines() == [f"RX 5000 -52.0 21 1 {air.hex().upper()}"]


@pytest.mark.parametrize(
    ("dialect", "code", "net"), [(DIALECT_A, 0, b"\x1b\x30"), (DIALECT_B, 1, NET_B)]
)
def test_auto_ack(core: Core, dialect, code: int, net: bytes) -> None:
    """A CRC-valid data frame to our id is acked in the current dialect and net."""
    core.send(f"Y{code}\nN{net.hex()}\nI01\nA1\n")
    air = build_frame(dialect, 0x06, 0x01, b"\x57\x55", network_id=net)
    core.receive(air, edge_us=7000)
    lines = core.lines()
    expected = ack_frame(dialect, net, 0x01, 0x06)
    assert core.tx_frame() == expected
    assert lines[0].startswith("ACK ") and lines[0].endswith(expected.hex().upper())
    assert lines[1].startswith("RX 7000 ") and lines[1].endswith(
        f" 1 {air.hex().upper()}"
    )
    assert core.host_masked() == 0
    assert decode(dialect, bytes.fromhex(lines[0].split()[2])).is_ack


def test_auto_ack_skips_bad_crc_other_ids_and_acks(core: Core) -> None:
    """No ack for a CRC-failing frame, a frame for someone else, or an ack."""
    core.send("A1\n")
    bad = bytearray(poll_a())
    bad[-1] ^= 0x01
    core.receive(bytes(bad))
    assert core.lines() == [f"RX 5000 -52.0 21 0 {bytes(bad).hex().upper()}"]
    core.receive(poll_a(dst=0x02))
    core.receive(poll_a(flags=0x40))
    core.receive(ack_frame(DIALECT_A, b"\x1b\x30", 0x06, 0x01))
    assert core.host_tx_count() == 0
    assert all(" 1 " in line for line in core.lines())
    # A dialect B frame heard while in dialect A: wrong CRC, never acked.
    b_frame = build_frame(DIALECT_B, 0x06, 0x01, b"", network_id=NET_B)
    core.send("Y0\n")
    core.receive(b_frame)
    assert core.host_tx_count() == 0


def test_noise_and_overflow_restart_the_receiver(core: Core) -> None:
    """A length byte outside [8, 255] or a FIFO overflow drops the frame."""
    core.host_rx_push(bytes([0xFF ^ 2]) * 4, 4)  # dialect A total 5
    core.rc_on_gdo2()
    assert core.host_rx_state() == RX_IDLE
    sfrx = core.host_strobes(CC1101_SFRX)
    core.host_rx_push(poll_a()[:4], 4)
    core.host_set_rx_overflow(1)
    core.rc_on_gdo2()
    assert core.host_rx_state() == RX_IDLE
    assert core.host_strobes(CC1101_SFRX) > sfrx
    core.host_set_rx_overflow(0)
    core.rc_on_gdo0(1)  # packet-done edge with no frame: just re-arm
    core.rc_service()
    assert core.lines() == []


def test_liveness_guard_recovers_a_stuck_receiver(core: Core) -> None:
    """RX_DONE without a pending packet re-arms after 250 ms, with a report."""
    core.host_rx_push(poll_a(), len(poll_a()))
    core.rc_on_gdo2()
    core.rc_on_gdo0(1000)
    core.host_clear_packet_ready()
    core.rc_service()
    assert core.lines() == [] and core.host_rx_state() == RX_DONE
    core.host_advance(250_001)
    core.rc_service()
    assert core.lines() == ["# rxreset n=1"]
    assert core.host_rx_state() == RX_IDLE
    assert core.send("D")[0].endswith("rxreset=1")
