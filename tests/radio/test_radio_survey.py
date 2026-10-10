"""Tests for raw radio surveys: runs to bits, known dialects, unknown-dialect search."""

from __future__ import annotations

from fake_radio_link import build_ack

import asyncio
import json
import random

import pytest
from radio_fakes import NET, FakeGateway, FakeTime

from custom_components.termoweb.backend.radio import link as link_mod, survey as s
from custom_components.termoweb.backend.radio.dialect import (
    DIALECT_A,
    DIALECT_B,
    build_frame,
)
from custom_components.termoweb.backend.radio.link import RadioLink

BIT_US = 1e6 / 9600  # 104.17 us at 9.6 kbps
OTHER_NET = bytes.fromhex("5678")  # a second synthetic installation

# An invented "dialect C": its own sync, no whitening, CRC-16/ARC sent
# little-endian, and byte 0 = total length - 1.
C_SYNC = bytes.fromhex("2D3A")
ARC = next(spec for spec in s.CRC_SPECS if spec.name == "CRC-16/ARC")


def dialect_c_frame(src: int, payload: bytes) -> bytes:
    """Return dialect-C air bytes for a data frame on the synthetic network."""
    body = NET + bytes([src, 1, 0, src, 1, 0, 0, 0, 0]) + payload
    total = 1 + len(body) + 2
    head = bytes([total - 1]) + body
    return head + ARC.finish(ARC.register_after(head)[-1]).to_bytes(2, "little")


def to_bits(data: bytes) -> str:
    """Return bytes as a bit string, MSB first."""
    return "".join(f"{byte:08b}" for byte in data)


def capture(
    air: bytes,
    sync: bytes,
    seed: int,
    *,
    invert: bool = False,
    preamble_bytes: int = 4,
) -> s.RawBurst:
    """Return a raw burst: noise, idle, preamble, sync, frame, idle, noise."""
    rng = random.Random(seed)
    bits = "10101010" * preamble_bytes + to_bits(sync) + to_bits(air) + "1010"
    if invert:
        bits = bits.translate(str.maketrans("01", "10"))
    runs: list[tuple[int, int]] = []
    for _ in range(12):  # noise before the burst
        runs.append((len(runs) % 2, rng.randint(5, 400)))
    runs.append((0, 6000))  # idle
    start = 0
    for index in range(1, len(bits) + 1):
        if index == len(bits) or bits[index] != bits[start]:
            level = int(bits[start])
            count = index - start
            bias = 18 if level else -18  # the demodulator stretches highs
            runs.append((level, round(count * BIT_US + bias + rng.uniform(-8, 8))))
            start = index
    runs.append((0, 9000))  # idle after the frame
    for _ in range(6):
        runs.append((len(runs) % 2, rng.randint(5, 300)))
    return s.RawBurst(-80.0 - seed % 10, tuple(runs))


def b_frame(net: bytes = NET, payload: bytes = b"\x50") -> bytes:
    """Return a dialect-B registration-like frame from heater 6."""
    return build_frame(DIALECT_B, 6, 1, payload, network_id=net)


# --- runs, bits and preambles ------------------------------------------------------


def test_runs_merge_parse_and_bits() -> None:
    """Same-level runs merge, bad tokens are skipped, bits come out MSB first."""
    assert s.merge_runs([(1, 50), (1, 54), (0, 0), (0, 104)]) == [(1, 104), (0, 104)]
    assert s.parse_run_tokens(["+118", "-92", "x", "+", "-1a", "+7"]) == [
        (1, 118),
        (0, 92),
        (1, 7),
    ]
    assert s.bits_to_bytes("0010110111010100111") == bytes.fromhex("2DD4")
    assert s.bits_to_bytes("1" * 80, limit=2) == b"\xff\xff"
    assert s.find_preambles([(1, 104), (0, 104)]) == []


def test_preamble_timing_corrects_demodulator_bias() -> None:
    """The preamble gives the bit period and the high/low stretch to remove."""
    burst = capture(b_frame(), DIALECT_B.sync, seed=1)
    runs = s.merge_runs(burst.runs)
    preamble = s.find_preambles(runs)[0]
    assert 100 < preamble.bit_us < 108
    assert 12 < preamble.bias_us < 24


def test_crc_catalogue_matches_reference_values() -> None:
    """Each catalogue entry gives the standard check value for '123456789'."""
    check = {
        "CRC-16/MODBUS": 0x4B37,
        "CRC-16/CCITT-FALSE": 0x29B1,
        "CRC-16/AUG-CCITT": 0xE5CC,
        "CRC-16/XMODEM": 0x31C3,
        "CRC-16/GENIBUS": 0xD64E,
        "CRC-16/KERMIT": 0x2189,
        "CRC-16/X-25": 0x906E,
        "CRC-16/ARC": 0xBB3D,
        "CRC-16/USB": 0xB4C8,
        "CRC-16/MAXIM": 0x44C2,
        "CRC-16/CMS": 0xAEE7,
        "CRC-16/BUYPASS": 0xFEE8,
        "CRC-16/DNP": 0xEA82,
    }
    for spec in s.CRC_SPECS:
        if spec.name in check:
            crc = spec.finish(spec.register_after(b"123456789")[-1])
            assert crc == check[spec.name], spec.name


# --- known dialects -------------------------------------------------------------


def test_known_dialects_and_inverted_polarity() -> None:
    """Dialect A and B bursts decode; an inverted demodulator is noticed."""
    a = s.analyse_burst(
        capture(build_frame(DIALECT_A, 6, 1, b"\xb8"), DIALECT_A.sync, 2)
    )
    assert (a.dialect, a.sync, a.inverted) == ("A", "2DE5", False)
    assert a.frame is not None and a.frame[1:3] == DIALECT_A.network_id
    b = s.analyse_burst(capture(b_frame(), DIALECT_B.sync, 3, invert=True))
    assert (b.dialect, b.sync, b.inverted) == ("B", "2DD4", True)
    assert b.bits is None and b.fits == ()


def test_report_for_a_known_dialect_lists_networks() -> None:
    """Several installations heard: every network id, most frequent first."""
    bursts = [capture(b_frame(), DIALECT_B.sync, seed) for seed in (4, 5)]
    bursts.append(capture(b_frame(OTHER_NET), DIALECT_B.sync, 6))
    bursts.append(
        capture(build_ack(DIALECT_B, 6, 1, NET), DIALECT_B.sync, 7)
    )  # 8-byte acks are too short to need decoding, but decode anyway
    report = s.analyse(bursts)
    assert report.verdict == "known" and report.dialect == "B"
    assert report.network_ids == ("1234", "5678")
    assert report.confidence == 1.0
    data = report.as_dict()
    json.dumps(data)
    assert data["bursts"][0]["frame"].startswith("0F 12 34 06")
    assert data["contains_raw_bits"] is False


# --- unknown dialects -----------------------------------------------------------


def test_unknown_dialect_parameters_are_found() -> None:
    """Dialect C is unknown, but three bursts agree on its exact framing."""
    bursts = [
        capture(dialect_c_frame(6, b"\xbe\x01\x02"), C_SYNC, 10),
        capture(dialect_c_frame(7, b"\xb9\x21\x25\x2a\x02"), C_SYNC, 11),
        capture(dialect_c_frame(6, b"\x50"), C_SYNC, 12),
    ]
    report = s.analyse(bursts)
    assert report.verdict == "candidate"
    best = report.candidates[0]
    assert best.framing == s.Framing(
        sync="2D3A",
        inverted=False,
        scrambled=False,
        length_offset=1,
        crc="CRC-16/ARC",
        crc_from=0,
        crc_order="little",
    )
    assert best.frames == 3
    assert report.network_ids == ("1234",)
    assert report.confidence == 1.0
    assert 'sync=bytes.fromhex("2D3A")' in report.suggestion
    assert "crc=crc16_arc" in report.suggestion
    assert "little-endian CRC" in report.suggestion
    assert all(b.framing == best.framing and b.bits is None for b in report.bursts)
    assert report.bursts[0].frame == dialect_c_frame(6, b"\xbe\x01\x02")


def test_candidate_report_keeps_bursts_that_do_not_fit() -> None:
    """A noise burst next to dialect-C bursts stays undecoded with its bits."""
    rng = random.Random(14)
    bursts = [
        capture(dialect_c_frame(6, b"\x50"), C_SYNC, 15),
        capture(dialect_c_frame(7, b"\xbe\x01"), C_SYNC, 16),
        capture(bytes(rng.randrange(256) for _ in range(20)), b"\x55\x01", 17),
    ]
    report = s.analyse(bursts)
    assert report.verdict == "candidate"
    assert report.bursts[2].framing is None and report.bursts[2].bits
    assert report.confidence == pytest.approx(2 / 3)


def test_decoders_skip_false_sync_hits_and_short_streams() -> None:
    """A sync match that does not decode moves on; too-short streams are skipped."""
    sync = to_bits(DIALECT_B.sync)
    frame = to_bits(b_frame())
    junk = to_bits(bytes([0x40, 1, 2, 3, 4, 5, 6]))  # length byte beyond the data
    found = s._decode_known("1010" + sync + junk + sync + frame)  # noqa: SLF001
    assert found is not None and found[0] is DIALECT_B and found[3] == 0
    assert s._search("11" + "0" * 30, 1) == []  # noqa: SLF001


def test_suggestion_lists_codec_needs() -> None:
    """Framings the current codec cannot express say so in the suggestion."""
    framing = s.Framing("ABCD", True, True, 3, "CRC-16/X-25", 1, "big")
    text = s.Candidate(framing, 2).suggestion()
    assert "crc=crc16_x_25" in text and "scrambled=True" in text
    assert "CRC from byte 1" in text and "inverted demodulator polarity" in text
    plain = s.Candidate(s.Framing("ABCD", False, False, 0, "CRC-16/CMS", 0, "big"), 2)
    assert "codec needs" not in plain.suggestion()


def test_one_matching_burst_is_not_enough() -> None:
    """A single CRC match may be chance: no candidate without agreement."""
    report = s.analyse([capture(dialect_c_frame(6, b"\x50"), C_SYNC, 13)])
    assert report.verdict == "undecodable"
    assert report.candidates == () and report.suggestion is None
    assert report.bursts[0].bits and report.as_dict()["contains_raw_bits"] is True


def test_noise_and_silence() -> None:
    """Random bytes after a preamble are undecodable; no preamble is silent."""
    rng = random.Random(20)
    junk = [
        capture(bytes(rng.randrange(256) for _ in range(20)), b"\x12\x34", seed)
        for seed in (21, 22)
    ]
    report = s.analyse(junk)
    assert report.verdict == "undecodable"
    assert report.bursts[0].sync is not None and report.bursts[0].bits
    silent = s.analyse([s.RawBurst(None, ((1, 3000), (0, 50), (1, 900)))])
    assert silent.verdict == "silent" and silent.bursts[0].bit_us is None
    assert s.analyse([]).verdict == "silent"


# --- privacy --------------------------------------------------------------------


def test_redact_masks_network_ids_and_identity_payloads() -> None:
    """A shared report keeps no network id and no identity reply payload."""
    identity = b_frame(payload=bytes.fromhex("5B55010203040506070809") + b"X123456")
    report = s.analyse(
        [
            capture(identity, DIALECT_B.sync, 30),
            capture(b_frame(), DIALECT_B.sync, 31),
        ]
    )
    shared = s.redact(report)
    data = shared.as_dict()
    assert data["redacted"] is True and data["network_ids"] == ["XXXX"]
    first = bytes.fromhex(data["bursts"][0]["frame"])
    assert first[1:3] == b"\x00\x00" and first[12:] == b"\x5b"
    assert bytes.fromhex(data["bursts"][1]["frame"])[1:3] == b"\x00\x00"
    assert "1234" not in json.dumps(data) and "X123456" not in json.dumps(data)
    tiny = s.SurveyReport(
        (s.BurstResult(None, 4, 104.0, None, False, None, None, b"\x01", None),),
        "undecodable",
        None,
        (),
        (),
        None,
        0.0,
    )
    assert s.redact(tiny).bursts[0].frame == b"\x01"


# --- the gateway side ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_link_survey_collects_bursts_and_hides_raw_lines() -> None:
    """R<seconds> collects RAWB/RAW/RAWE bursts; RX listeners never see them."""
    gw = FakeGateway(DIALECT_B)
    ft = FakeTime()
    link = RadioLink(
        "radio.local",
        dialect=DIALECT_B,
        network_id=NET,
        clock=ft.clock,
        sleep=ft.sleep,
        open_connection=gw.open_connection,
    )
    await link.connect()
    heard: list[object] = []
    link.add_listener(heard.append)
    gw.survey_lines = [
        "# survey on secs=5",
        "RAWB 1 -61.5",
        "RAW 1 +118 -92 +104",
        "RAW 1 -104 junk",
        "RAWE 1 4",
        "RAWB 2 bad",  # unreadable RSSI
        "RAW 2 +500",  # its RAWE never arrives
        "RAW 3 -40",  # data without a RAWB
        "RAW x +1",  # malformed burst number
        "RAWE",
        "# survey off",
    ]
    bursts = await link.survey(5)
    assert gw.commands[-1] == "R5"
    assert bursts == [
        s.RawBurst(-61.5, ((1, 118), (0, 92), (1, 104), (0, 104))),
        s.RawBurst(None, ((1, 500),)),
        s.RawBurst(None, ((0, 40),)),
    ]
    assert heard == []
    gw.feed("RAW 9 +1")  # after the survey: ignored
    for _ in range(20):
        await asyncio.sleep(0)
    gw.survey_lines = ["RAWB 1 -70", "RAW 1 +50"]  # no "survey off": time out
    assert await link.survey(1) == [s.RawBurst(-70.0, ((1, 50),))]
    with pytest.raises(ValueError, match="survey length"):
        await link.survey(0)
    await link.close()


# --- weak signals ------------------------------------------------------------------


def weak_capture(
    air: bytes,
    sync: bytes,
    rng: random.Random,
    *,
    jitter: float = 0.15,
    flips: float = 0.0,
    flip_frame: bool = False,
    glitches: int = 0,
    preamble_bytes: int = 4,
) -> s.RawBurst:
    """Return a burst with edge jitter, glitches and bit flips (a weak signal)."""
    bits = list("10101010" * preamble_bytes + to_bits(sync) + to_bits(air) + "1010")
    region = len(bits) if flip_frame else 8 * preamble_bytes + 8 * len(sync)
    for index in range(region):
        if rng.random() < flips:
            bits[index] = "1" if bits[index] == "0" else "0"
    starts = [0] + [i for i in range(1, len(bits)) if bits[i] != bits[i - 1]]
    times = [0.0]
    times += [i * BIT_US + rng.uniform(-jitter, jitter) * BIT_US for i in starts[1:]]
    times.append(len(bits) * BIT_US)
    runs = [(n % 2, rng.randint(5, 400)) for n in range(12)] + [(0, 20000)]
    for k, start in enumerate(starts):
        level = int(bits[start])
        bias = 18 if level else -18
        runs.append((level, max(1, round(times[k + 1] - times[k] + bias))))
    for _ in range(glitches):
        index = rng.randrange(14, len(runs))
        level, micros = runs[index]
        if micros > 60:
            cut, glitch = rng.randint(20, micros - 30), rng.randint(5, 25)
            runs[index : index + 1] = [
                (level, cut),
                (1 - level, glitch),
                (level, micros - cut - glitch),
            ]
    runs += [(0, 20000)] + [(n % 2, rng.randint(5, 300)) for n in range(6)]
    return s.RawBurst(-92.0, tuple(runs))


def test_weak_signal_bursts_still_decode() -> None:
    """Jitter of +-15 %, glitches and preamble/sync bit errors: dialect B decodes."""
    rng = random.Random(40)
    air = b_frame()
    decoded = sum(
        s.analyse_burst(
            weak_capture(air, DIALECT_B.sync, rng, flips=0.03, glitches=3)
        ).dialect
        == "B"
        for _ in range(20)
    )
    assert decoded >= 18
    # Flips inside the frame break its CRC: only frames without one can decode,
    # which is why one CRC-valid frame is enough for a known dialect.
    report = s.analyse(
        [
            weak_capture(air, DIALECT_B.sync, rng, flips=0.01, flip_frame=True)
            for _ in range(10)
        ]
    )
    assert report.verdict == "known" and report.network_ids == ("1234",)


def test_sync_errors_are_tolerated_and_reported() -> None:
    """Up to two flipped sync bits still find the frame; three do not."""
    air = to_bits(b_frame())
    sync = to_bits(DIALECT_B.sync)
    two = (
        ("1" if sync[0] == "0" else "0")
        + sync[1:5]
        + ("1" if sync[5] == "0" else "0")
        + sync[6:]
    )
    found = s._decode_known("10101010" + two + air)  # noqa: SLF001
    assert found is not None and found[3] == 2
    three = two[:9] + ("1" if two[9] == "0" else "0") + two[10:]
    assert s._decode_known("10101010" + three + air) is None  # noqa: SLF001


def test_common_rates_are_tried_without_a_clean_preamble() -> None:
    """A one-byte preamble is too short to time from; 9.6 kbps is tried anyway."""
    burst = capture(b_frame(), DIALECT_B.sync, 41, preamble_bytes=1)
    assert s.find_preambles(s.merge_runs(burst.runs)) == []
    result = s.analyse_burst(burst)
    assert result.dialect == "B" and result.bit_us == pytest.approx(BIT_US, rel=0.03)


def test_slicer_helpers() -> None:
    """Glitches fold away, idle splits segments, preamble ends skip lone errors."""
    assert s.deglitch([(1, 100), (0, 10), (1, 100), (0, 200)], 30) == [
        (1, 210),
        (0, 200),
    ]
    assert s.deglitch([(0, 10), (1, 100)], 30) == [(0, 10), (1, 100)]
    runs = [(1, 104), (0, 104)] * 6 + [(1, 9000)] + [(0, 104), (1, 104)]
    assert s.slice_bits(runs, 104.0) == ["101010101010"]
    assert s.slice_bits([(1, 520), (0, 312), (1, 104)], 104.0) == ["111110001"]
    assert s.refine_period([(1, 90000)], 104.0) == 104.0
    bits = "1010101010" + "0" + "10101010" + "00101101"
    assert s.preamble_end(bits) == len(bits) - 8  # the lone error is skipped
    assert s.bit_preamble(bits) == (0, len(bits) - 8)
    assert s.bit_preamble("1100110011") is None
    assert s.preamble_end("10101010") == 8


def test_keep_runs_for_replay_and_redact_drops_them() -> None:
    """keep_runs puts the raw runs in the report; a shared report never has them."""
    burst = capture(b_frame(), DIALECT_B.sync, 42)
    data = s.analyse([burst], keep_runs=True).as_dict()
    assert data["contains_raw_runs"] is True
    first = data["bursts"][0]["raw_runs"].split()
    assert s.parse_run_tokens(first) == list(burst.runs)
    shared = s.redact(s.analyse([burst], keep_runs=True)).as_dict()
    assert (
        shared["contains_raw_runs"] is False and shared["bursts"][0]["raw_runs"] is None
    )
    assert s.analyse([burst]).as_dict()["bursts"][0]["raw_runs"] is None


def test_timing_preamble_without_bit_preamble(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runs that time like a preamble but slice to no preamble stay unexplained."""
    rng = random.Random(43)
    burst = capture(bytes(rng.randrange(256) for _ in range(20)), b"\x12\x34", 44)
    monkeypatch.setattr(s, "bit_preamble", lambda _bits: None)
    result = s.analyse_burst(burst)
    assert result.bit_us is not None and result.sync is None and result.bits is None


@pytest.mark.asyncio
async def test_nothing_is_transmitted_while_a_survey_runs() -> None:
    """The gateway is in raw capture during R<secs>: transmits wait for its end."""
    gw = FakeGateway(DIALECT_B)
    ft = FakeTime()
    parked: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    async def sleep(seconds: float) -> None:
        if seconds >= link_mod.SURVEY_GRACE_S:
            await parked  # the survey window: only "survey off" ends it
        else:
            await ft.sleep(seconds)

    link = RadioLink(
        "radio.local",
        dialect=DIALECT_B,
        network_id=NET,
        clock=ft.clock,
        sleep=sleep,
        open_connection=gw.open_connection,
    )
    await link.connect()
    survey = asyncio.ensure_future(link.survey(60))
    for _ in range(20):
        await asyncio.sleep(0)
    air = build_frame(DIALECT_B, 1, 6, b"\xbf\x01", network_id=NET)
    send = asyncio.ensure_future(link.send_frame(6, air, wait_ack=False))
    for _ in range(100):
        await asyncio.sleep(0)
    assert gw.commands[-1] == "R60"  # no T while the radio is in raw capture

    gw.feed("# survey off")
    await survey
    assert (await send).ok
    assert gw.commands[-1] == "T" + air.hex().upper()
    parked.cancel()
    await link.close()
