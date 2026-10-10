"""Analyse raw radio surveys to recognise known dialects and describe unknown ones.

The gateway's survey mode (``R<seconds>``) puts the CC1101 in asynchronous
serial mode and reports every RF burst as run lengths of the demodulator
output: ``(level, microseconds)``. That needs no sync word, length rule,
whitening or CRC, so it also captures heaters that speak a dialect this
package does not know yet. This module turns runs into bits, finds the
preamble and the sync word, decodes the known dialects, and otherwise
searches the usual framing choices for parameter sets whose CRC validates on
several bursts. The result is a JSON-able report a test user can submit.

Nothing here touches the radio or Home Assistant.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field, replace
from itertools import pairwise
from statistics import median
from typing import Any

from .dialect import DIALECTS, Dialect, decode, frame_total_length, whiten

MIN_PREAMBLE_RUNS = 16  # alternating runs (8 preamble bits per byte)
PAIR_TOLERANCE = 0.35  # a preamble pair may differ this much from the first
MAX_RUN_BITS = 64  # a longer run is idle or noise: the frame has ended
MAX_FRAME_BYTES = 80
SYNC_SEARCH = range(8, 41)  # frame start, in bits after the preamble break
LENGTH_OFFSETS = range(-4, 5)  # total frame length = byte 0 + offset
MIN_FRAME_BYTES = 5
MIN_AGREEING_BURSTS = 2  # a CRC match on one burst alone may be chance
RAW_BITS_LIMIT = 2000  # bits kept per undecoded burst in the report
IDENTITY_REPLY = 0x5B
PAYLOAD_START = 12  # logical layout shared by the known dialects


@dataclass(frozen=True)
class RawBurst:
    """One RF burst as demodulator run lengths: ``(level 1|0, microseconds)``."""

    rssi_dbm: float | None
    runs: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class CrcSpec:
    """A CRC-16 parameter set (Rocksoft model; refin == refout here)."""

    name: str
    poly: int
    init: int
    reflected: bool
    xorout: int
    table: tuple[int, ...] = field(default=(), repr=False, compare=False)

    def register_after(self, data: bytes) -> list[int]:
        """Return the register after each prefix of ``data`` (index = bytes fed)."""
        reg = _reflect16(self.init) if self.reflected else self.init
        out = [reg]
        table = self.table
        if self.reflected:
            for byte in data:
                reg = (reg >> 8) ^ table[(reg ^ byte) & 0xFF]
                out.append(reg)
        else:
            for byte in data:
                reg = ((reg << 8) & 0xFFFF) ^ table[((reg >> 8) ^ byte) & 0xFF]
                out.append(reg)
        return out

    def finish(self, register: int) -> int:
        """Return the CRC value for a register state."""
        return register ^ self.xorout

    def compute(self, data: bytes) -> int:
        """Return the CRC of ``data``."""
        return self.finish(self.register_after(data)[-1])


def _reflect16(value: int) -> int:
    """Return ``value`` with its 16 bits in reverse order."""
    return int(f"{value:016b}"[::-1], 2)


def _make_table(poly: int, reflected: bool) -> tuple[int, ...]:
    """Return the byte-wise lookup table for a CRC-16 polynomial."""
    table = []
    rpoly = _reflect16(poly)
    for byte in range(256):
        if reflected:
            reg = byte
            for _ in range(8):
                reg = (reg >> 1) ^ rpoly if reg & 1 else reg >> 1
        else:
            reg = byte << 8
            for _ in range(8):
                reg = (
                    ((reg << 1) ^ poly) & 0xFFFF
                    if reg & 0x8000
                    else (reg << 1) & 0xFFFF
                )
        table.append(reg)
    return tuple(table)


def _spec(name: str, poly: int, init: int, reflected: bool, xorout: int) -> CrcSpec:
    """Return a CRC spec with its lookup table built."""
    return CrcSpec(name, poly, init, reflected, xorout, _make_table(poly, reflected))


CRC_SPECS: tuple[CrcSpec, ...] = (
    _spec("CRC-16/TERMOWEB-A", 0x1021, 0x1D0F, False, 0xFFFF),  # dialect A
    _spec("CRC-16/MODBUS", 0x8005, 0xFFFF, True, 0x0000),  # dialect B
    _spec("CRC-16/CCITT-FALSE", 0x1021, 0xFFFF, False, 0x0000),
    _spec("CRC-16/AUG-CCITT", 0x1021, 0x1D0F, False, 0x0000),
    _spec("CRC-16/XMODEM", 0x1021, 0x0000, False, 0x0000),
    _spec("CRC-16/GENIBUS", 0x1021, 0xFFFF, False, 0xFFFF),
    _spec("CRC-16/KERMIT", 0x1021, 0x0000, True, 0x0000),
    _spec("CRC-16/X-25", 0x1021, 0xFFFF, True, 0xFFFF),
    _spec("CRC-16/ARC", 0x8005, 0x0000, True, 0x0000),
    _spec("CRC-16/USB", 0x8005, 0xFFFF, True, 0xFFFF),
    _spec("CRC-16/MAXIM", 0x8005, 0x0000, True, 0xFFFF),
    _spec("CRC-16/CMS", 0x8005, 0xFFFF, False, 0x0000),  # CC1101 hardware CRC
    _spec("CRC-16/BUYPASS", 0x8005, 0x0000, False, 0x0000),
    _spec("CRC-16/DNP", 0x3D65, 0x0000, True, 0xFFFF),
)


# --- runs to bits ---------------------------------------------------------------


def merge_runs(runs: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Join consecutive runs of the same level and drop empty ones."""
    merged: list[tuple[int, int]] = []
    for level, micros in runs:
        if micros <= 0:
            continue
        if merged and merged[-1][0] == level:
            merged[-1] = (level, merged[-1][1] + micros)
        else:
            merged.append((level, micros))
    return merged


@dataclass(frozen=True)
class Preamble:
    """The longest alternating stretch of a burst and the bit timing it implies."""

    start: int  # index of its first run
    end: int  # index after its last run
    bit_us: float
    bias_us: float  # half the high/low width difference, removed before rounding


def find_preambles(runs: Sequence[tuple[int, int]]) -> list[Preamble]:
    """Return every stretch of one-bit runs with steady pair sums, longest first."""
    found: list[tuple[int, int]] = []
    i = 0
    while i + 1 < len(runs):
        reference = runs[i][1] + runs[i + 1][1]
        j = i
        while (
            j + 1 < len(runs)
            and abs(runs[j][1] + runs[j + 1][1] - reference)
            <= PAIR_TOLERANCE * reference
            and max(runs[j][1], runs[j + 1][1]) < 0.85 * reference
        ):
            j += 1
        if j + 1 - i >= MIN_PREAMBLE_RUNS:
            found.append((i, j + 1))
        i = max(j, i + 1)
    found.sort(key=lambda window: window[0] - window[1])
    return [_preamble(runs, start, end) for start, end in found]


def find_preamble(runs: Sequence[tuple[int, int]]) -> Preamble | None:
    """Return the longest preamble-like stretch, or None."""
    preambles = find_preambles(runs)
    return preambles[0] if preambles else None


def _preamble(runs: Sequence[tuple[int, int]], start: int, end: int) -> Preamble:
    """Return the timing a preamble window implies."""
    window = runs[start:end]
    pairs = [a[1] + b[1] for a, b in pairwise(window)]
    highs = [m for level, m in window if level]
    lows = [m for level, m in window if not level]
    return Preamble(
        start=start,
        end=end,
        bit_us=median(pairs) / 2,
        bias_us=(median(highs) - median(lows)) / 2,
    )


def runs_to_bits(runs: Sequence[tuple[int, int]], preamble: Preamble) -> str:
    """Return the bits from the preamble on, stopping at the first idle gap.

    The preamble alone gives the bit period to about 2 %, too coarse for long
    runs of equal bits, so the period is refined once over the whole frame.
    """
    frame: list[tuple[int, float]] = []
    for level, micros in runs[preamble.start :]:
        corrected = micros - preamble.bias_us if level else micros + preamble.bias_us
        if corrected / preamble.bit_us > MAX_RUN_BITS + 0.5:
            break
        frame.append((level, corrected))
    counts = [round(corrected / preamble.bit_us) for _level, corrected in frame]
    total_bits = sum(counts)
    bit_us = (
        sum(corrected for _level, corrected in frame) / total_bits
        if total_bits
        else preamble.bit_us
    )
    return "".join(
        ("1" if level else "0") * round(corrected / bit_us)
        for level, corrected in frame
    )


def bits_to_bytes(bits: str, limit: int = MAX_FRAME_BYTES) -> bytes:
    """Pack whole bytes MSB first, at most ``limit`` of them."""
    count = min(len(bits) // 8, limit)
    return bytes(int(bits[8 * i : 8 * i + 8], 2) for i in range(count))


def _invert(bits: str) -> str:
    """Return ``bits`` with every bit flipped."""
    return bits.translate(str.maketrans("01", "10"))


def _sync_bits(sync: bytes) -> str:
    """Return a sync word as a bit string."""
    return "".join(f"{byte:08b}" for byte in sync)


def preamble_break(bits: str) -> int:
    """Return the index of the first bit equal to its predecessor (preamble end)."""
    for index in range(1, len(bits)):
        if bits[index] == bits[index - 1]:
            return index
    return len(bits)


# --- decoding -------------------------------------------------------------------


@dataclass(frozen=True)
class Framing:
    """Framing parameters that made a frame's CRC check."""

    sync: str  # hex of the 16 bits before the frame
    inverted: bool
    scrambled: bool
    length_offset: int  # total frame length = byte 0 + offset
    crc: str
    crc_from: int  # first byte the CRC covers
    crc_order: str  # "big" or "little"


@dataclass(frozen=True)
class BurstResult:
    """What one burst decoded to (or why it did not)."""

    rssi_dbm: float | None
    runs: int
    bit_us: float | None
    sync: str | None  # hex: known sync found, or the 16 bits after the preamble
    inverted: bool
    dialect: str | None  # a known dialect that decoded it
    framing: Framing | None  # parameters found by the search
    frame: bytes | None  # logical frame bytes, CRC included
    bits: str | None  # demodulated bits, kept when nothing decoded
    fits: tuple[Framing, ...] = ()  # every search match (before agreement)


def _decode_known(bits: str) -> tuple[Dialect, bool, bytes, int] | None:
    """Return (dialect, inverted, logical frame, sync end) for a known dialect."""
    for inverted in (False, True):
        stream = _invert(bits) if inverted else bits
        for dialect in DIALECTS.values():
            pattern = _sync_bits(dialect.sync)
            start = stream.find(pattern)
            while start >= 0:
                end = start + len(pattern)
                air = bits_to_bytes(stream[end:])
                if air:
                    total = frame_total_length(dialect, air[0])
                    if MIN_FRAME_BYTES <= total <= len(air):
                        frame = decode(dialect, air[:total])
                        if frame.ok:
                            return dialect, inverted, frame.logical[:total], end
                start = stream.find(pattern, start + 1)
    return None


def _search(bits: str) -> list[tuple[Framing, bytes]]:
    """Return every framing whose CRC validates on this bit stream."""
    fits: list[tuple[Framing, bytes]] = []
    brk = preamble_break(bits)
    for inverted in (False, True):
        stream = _invert(bits) if inverted else bits
        for offset in SYNC_SEARCH:
            start = brk + offset
            if start < 16:
                continue
            raw = bits_to_bytes(stream[start:])
            if len(raw) < MIN_FRAME_BYTES:
                continue
            sync = bits_to_bytes(stream[start - 16 : start]).hex().upper()
            for scrambled in (False, True):
                data = whiten(raw) if scrambled else raw
                for spec in CRC_SPECS:
                    for crc_from in (0, 1):
                        registers = spec.register_after(data[crc_from:])
                        for length_offset in LENGTH_OFFSETS:
                            total = data[0] + length_offset
                            body_end = total - 2
                            if total < MIN_FRAME_BYTES or total > len(data):
                                continue
                            value = spec.finish(registers[body_end - crc_from])
                            tail = data[body_end:total]
                            fits.extend(
                                (
                                    Framing(
                                        sync,
                                        inverted,
                                        scrambled,
                                        length_offset,
                                        spec.name,
                                        crc_from,
                                        order,
                                    ),
                                    bytes(data[:total]),
                                )
                                for order in ("big", "little")
                                if int.from_bytes(tail, order) == value  # type: ignore[arg-type]
                            )
    return fits


def analyse_burst(burst: RawBurst) -> BurstResult:
    """Decode one burst with the known dialects, else search its framing."""
    runs = merge_runs(burst.runs)
    preambles = find_preambles(runs)
    if not preambles:
        return BurstResult(
            burst.rssi_dbm, len(runs), None, None, False, None, None, None, None
        )
    for candidate in preambles:
        known = _decode_known(runs_to_bits(runs, candidate))
        if known is not None:
            dialect, inverted, frame, _end = known
            return BurstResult(
                burst.rssi_dbm,
                len(runs),
                candidate.bit_us,
                dialect.sync.hex().upper(),
                inverted,
                dialect.name,
                None,
                frame,
                None,
            )
    preamble = preambles[0]
    bits = runs_to_bits(runs, preamble)
    fits = _search(bits)
    brk = preamble_break(bits)
    plain_sync = bits_to_bytes(bits[brk - 1 : brk + 15]).hex().upper() or None
    return BurstResult(
        burst.rssi_dbm,
        len(runs),
        preamble.bit_us,
        plain_sync,
        False,
        None,
        None,
        None,
        bits[:RAW_BITS_LIMIT],
        tuple(framing for framing, _frame in fits),
    )


# --- report ---------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """A framing several bursts agree on: a new dialect in the making."""

    framing: Framing
    frames: int  # bursts whose CRC checks with it

    def suggestion(self) -> str:
        """Return a ``Dialect(...)`` definition to paste into dialect.py."""
        f = self.framing
        crc_fn = "crc16_" + f.crc.removeprefix("CRC-16/").lower().replace("-", "_")
        needs = []
        if f.crc_from:
            needs.append("CRC from byte 1")
        if f.crc_order == "little":
            needs.append("little-endian CRC")
        if f.inverted:
            needs.append("inverted demodulator polarity")
        note = f"  # codec needs: {', '.join(needs)}" if needs else ""
        return (
            "Dialect(\n"
            '    name="?",  # found by a radio survey\n'
            f'    sync=bytes.fromhex("{f.sync}"),\n'
            "    network_id=None,  # learned per installation, like dialect B\n"
            f"    scrambled={f.scrambled},\n"
            f"    length_offset={f.length_offset},\n"
            f"    crc={crc_fn},  # {f.crc}{note}\n"
            '    eb_clock_suffix=b"",  # unknown: test the clock sync\n'
            "    firmware_mode=2,  # new gateway mode needed for this sync\n"
            "    program_write_slots=24,  # unknown\n"
            "    mode_in_preset_write=True,  # unknown\n"
            "    ack_only_opcodes=frozenset(),  # unknown\n"
            ")"
        )


@dataclass(frozen=True)
class SurveyReport:
    """The outcome of a survey, ready to log, show or submit."""

    bursts: tuple[BurstResult, ...]
    verdict: str  # "silent", "known", "candidate" or "undecodable"
    dialect: str | None  # known dialect heard
    network_ids: tuple[str, ...]  # hex, most frequent first
    candidates: tuple[Candidate, ...]
    suggestion: str | None
    confidence: float  # share of bursts with a preamble that decoded
    redacted: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return the report as JSON-able data."""
        bursts = []
        for b in self.bursts:
            entry = {
                "rssi_dbm": b.rssi_dbm,
                "runs": b.runs,
                "bit_us": round(b.bit_us, 1) if b.bit_us else None,
                "sync": b.sync,
                "inverted": b.inverted,
                "dialect": b.dialect,
                "framing": asdict(b.framing) if b.framing else None,
                "frame": b.frame.hex(" ").upper() if b.frame else None,
                "bits": b.bits,
            }
            bursts.append(entry)
        return {
            "verdict": self.verdict,
            "dialect": self.dialect,
            "network_ids": list(self.network_ids),
            "candidates": [
                {**asdict(c.framing), "frames": c.frames} for c in self.candidates
            ],
            "suggestion": self.suggestion,
            "confidence": round(self.confidence, 2),
            "redacted": self.redacted,
            "contains_raw_bits": any(b.bits for b in self.bursts),
            "bursts": bursts,
        }


def _network_id(frame: bytes | None) -> str | None:
    """Return bytes 1-2 of a logical frame as hex (the shared layout's network id)."""
    return frame[1:3].hex().upper() if frame and len(frame) >= 3 else None


def analyse(bursts: Iterable[RawBurst]) -> SurveyReport:
    """Analyse a survey: known dialects first, then framings bursts agree on."""
    results = [analyse_burst(burst) for burst in bursts]
    heard = [r for r in results if r.bit_us is not None]
    if not heard:
        return SurveyReport(tuple(results), "silent", None, (), (), None, 0.0)
    known = [r for r in heard if r.dialect]
    if known:
        dialect = Counter(r.dialect for r in known).most_common(1)[0][0]
        nets = Counter(_network_id(r.frame) for r in known if _network_id(r.frame))
        return SurveyReport(
            tuple(results),
            "known",
            dialect,
            tuple(net for net, _ in nets.most_common()),
            (),
            None,
            len(known) / len(heard),
        )
    votes: Counter[Framing] = Counter()
    for result in heard:
        votes.update(set(result.fits))
    agreed = [
        Candidate(framing, count)
        for framing, count in votes.most_common()
        if count >= MIN_AGREEING_BURSTS
    ]
    if not agreed:
        return SurveyReport(tuple(results), "undecodable", None, (), (), None, 0.0)
    best = agreed[0].framing
    decoded = [_with_framing(r, best) for r in results]
    nets = Counter(_network_id(r.frame) for r in decoded if r.frame)
    return SurveyReport(
        tuple(decoded),
        "candidate",
        None,
        tuple(net for net, _ in nets.most_common() if net),
        tuple(agreed),
        agreed[0].suggestion(),
        agreed[0].frames / len(heard),
    )


def _with_framing(result: BurstResult, framing: Framing) -> BurstResult:
    """Return ``result`` decoded with ``framing`` when that framing fits it."""
    if framing not in result.fits or result.bits is None:
        return result
    frame = _frame_for(result.bits, framing)
    return replace(result, framing=framing, frame=frame, sync=framing.sync, bits=None)


def _frame_for(bits: str, framing: Framing) -> bytes | None:
    """Re-extract the logical frame bytes a framing found."""
    for fit, frame in _search(bits):
        if fit == framing:
            return frame
    return None  # pragma: no cover - a fit always reproduces


def redact(report: SurveyReport) -> SurveyReport:
    """Mask network ids and drop identity-reply payloads before sharing a report."""
    bursts = tuple(replace(b, frame=_mask(b.frame)) for b in report.bursts)
    return replace(
        report,
        bursts=bursts,
        network_ids=tuple("XXXX" for _ in report.network_ids),
        redacted=True,
    )


def _mask(frame: bytes | None) -> bytes | None:
    """Zero the network id bytes and cut an identity reply's payload."""
    if frame is None or len(frame) < 3:
        return frame
    masked = bytearray(frame)
    masked[1:3] = b"\x00\x00"
    if len(masked) > PAYLOAD_START and masked[PAYLOAD_START] == IDENTITY_REPLY:
        masked = masked[: PAYLOAD_START + 1]
    return bytes(masked)


def parse_run_tokens(tokens: Iterable[str]) -> list[tuple[int, int]]:
    """Parse ``+118 -92`` run tokens into ``(level, micros)``; skip bad tokens."""
    runs: list[tuple[int, int]] = []
    for token in tokens:
        if len(token) < 2 or token[0] not in "+-" or not token[1:].isdigit():
            continue
        runs.append((1 if token[0] == "+" else 0, int(token[1:])))
    return runs


__all__ = [
    "CRC_SPECS",
    "BurstResult",
    "Candidate",
    "CrcSpec",
    "Framing",
    "Preamble",
    "RawBurst",
    "SurveyReport",
    "analyse",
    "analyse_burst",
    "bits_to_bytes",
    "find_preamble",
    "find_preambles",
    "merge_runs",
    "parse_run_tokens",
    "preamble_break",
    "redact",
    "runs_to_bits",
]
