"""Application payloads carried inside radio frames: opcodes, builders, decoders.

Builders return payload bytes only; :func:`dialect.build_frame` wraps them.
Decoders return frozen dataclasses whose fields are None wherever the record
form does not carry a value. Nothing is guessed. See ``docs/radio_protocol.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
import enum
import math

from .dialect import DIALECT_B, Dialect, Frame

# --- opcodes -----------------------------------------------------------------

OP_WRITE = 0xB4  # B4 <sub> [half-degrees]
OP_PROGRAM_WRITE = 0xB2
OP_PRESET_WRITE = 0xB6
OP_STATUS = 0xB8
OP_PROGRAM_READ = 0xB0
OP_ENERGY = 0xBC
OP_IDENTITY = 0x5A
OP_ADVANCED_READ = 0xDA
OP_POWER_REQUEST = 0xBE
OP_POWER_VERDICT = 0xBF
OP_REGISTRATION = 0x50
OP_CLOCK_REGISTERING = 0x51
OP_CLOCK_STEADY = 0x52
OP_CONFIRM_REPORT = 0x57
OP_REPORT = 0x56  # leading marker of every unsolicited report
OP_FLASH_DISPLAY = 0x5E
OP_WRITE_PARAMETER = 0xC8  # C8 <two argument bytes>; reply C9 55

TOGGLE_BOOST = 0xD2
TOGGLE_RUNBACK = 0xD4
TOGGLE_EASY = 0xD6
TOGGLE_LOCK = 0xBA
TOGGLE_OPCODES = frozenset({TOGGLE_BOOST, TOGGLE_RUNBACK, TOGGLE_EASY, TOGGLE_LOCK})

SUB_SETPOINT = 0x02
SUB_OVERRIDE = 0x03

REPLY_ACCEPTED = 0x55
REPLY_REJECTED = 0x56

ROUTE_PROBE_TAG = 0x06

# What a payload's first byte means (radio_protocol.md section 4), for captures.
OPCODE_NAMES: dict[int, str] = {
    0x50: "registration",
    0x51: "clock sync (registering)",
    0x52: "clock sync",
    0x53: "clock sync reply",
    0x56: "report",
    0x57: "report confirmation",
    0x5A: "identity request",
    0x5B: "identity reply",
    0x5E: "flash display (identify)",
    0x5F: "flash display reply",
    0x77: "pairing announcement (dialect A)",
    0xB0: "program read",
    0xB1: "program reply",
    0xB2: "program write",
    0xB3: "program write reply",
    0xB4: "mode / setpoint write",
    0xB5: "mode / setpoint write reply",
    0xB6: "preset write",
    0xB7: "preset write reply",
    0xB8: "status request",
    0xB9: "status reply",
    0xBA: "keypad lock",
    0xBB: "keypad lock reply",
    0xBC: "energy / power record request",
    0xBD: "energy / power record reply",
    0xBE: "power request",
    0xBF: "power verdict",
    0xC2: "unknown request C2",
    0xC3: "reply to C2",
    0xC4: "advanced setup write",
    0xC5: "advanced setup write reply",
    0xC6: "unknown request C6",
    0xC7: "reply to C6",
    0xC8: "write parameter",
    0xC9: "write parameter reply",
    0xD0: "capability request",
    0xD2: "boost toggle",
    0xD3: "boost toggle reply",
    0xD4: "runback toggle",
    0xD5: "runback toggle reply",
    0xD6: "EASY toggle",
    0xD7: "EASY toggle reply",
    0xDA: "advanced record request",
    0xDB: "advanced record reply",
}
# Frames whose meaning is in the tag byte rather than the payload.
TAG_NAMES: dict[int, str] = {
    0x03: "pairing announcement",
    0x04: "id assignment",
    ROUTE_PROBE_TAG: "route probe",
}

# --- modes -------------------------------------------------------------------

MODE_AUTO = 0x01
MODE_MANUAL = 0x02
MODE_OVERRIDE = 0x03
MODE_OFF = 0x04

MODE_NAMES: dict[int, str] = {
    MODE_AUTO: "auto",
    MODE_MANUAL: "manual",
    MODE_OVERRIDE: "override",
    MODE_OFF: "off",
}

SETTABLE_MODES = frozenset({MODE_AUTO, MODE_MANUAL, MODE_OFF})
# Dialect B's B6 also takes 03, a temporary override the heater ends itself at
# the next program change; manual and override carry their own setpoint byte.
PRESET_WRITE_MODES = SETTABLE_MODES | {MODE_OVERRIDE}
SETPOINT_MODES = frozenset({MODE_MANUAL, MODE_OVERRIDE})

# --- status flag bits (E5/E6/E4 flag byte) -----------------------------------

FLAG_ACTIVE = 0x01
FLAG_LOCKED = 0x02
FLAG_PRESENCE = 0x04
FLAG_WINDOW_OPEN = 0x08
FLAG_TRUE_RADIANT_ACTIVE = 0x10
FLAG_BOOST = 0x20
FLAG_EASY = 0x40
FLAG_RUNBACK = 0x80

MIN_SETPOINT_C = 7.0
MAX_SETPOINT_C = 35.0

# --- program record ----------------------------------------------------------

DAYS_PER_WEEK = 7
SLOTS_PER_BYTE = 4
SLOTS_HOURLY = 24
SLOTS_HALF_HOURLY = 48
SLOT_CODES = (0, 1, 2)  # cold / night (eco) / day (comfort)
PROGRAM_HOURLY_PAYLOAD_LEN = 43  # B1 + 42 bytes (C9 class)
PROGRAM_HALF_HOURLY_PAYLOAD_LEN = 85  # B1 + 84 bytes (9F class, dialect-A B2 writes)
_PROGRAM_RESOLUTION = {
    PROGRAM_HOURLY_PAYLOAD_LEN: SLOTS_HOURLY,
    PROGRAM_HALF_HOURLY_PAYLOAD_LEN: SLOTS_HALF_HOURLY,
}

# --- status record forms -----------------------------------------------------

STATUS_SHORT_LEN = 5  # B9 af eco comfort mode
STATUS_E6_LEN = 14
STATUS_E4_LEN = 16  # E6 + 2-byte boost tail
STATUS_E5_LEN = 15  # 56 + E6
STATUS_E3_LEN = 17  # 56 + E4
STATUS_MARKER = OP_STATUS + 1  # B9

IDENTITY_E0_LEN = 20
IDENTITY_SHORT_LEN = 18
IDENTITY_MARKER = OP_IDENTITY + 1  # 5B
IDENTITY_E0_FORM = 0x77
IDENTITY_SHORT_FORM = 0x55

ENERGY_A_LEN = 5
ENERGY_B_LEN = 9
ENERGY_MARKER = OP_ENERGY + 1  # BD

POWER_REQUEST_A_LEN = 3
POWER_REQUEST_B_LEN = 9


# --- builders ----------------------------------------------------------------


def _half_degrees(celsius: float) -> int:
    """Return ``celsius`` in half degrees, the wire unit for temperatures."""
    return round(celsius * 2)


def _check_setpoint(celsius: float) -> None:
    """Raise ValueError unless ``celsius`` lies within the accepted setpoint range."""
    if not (math.isfinite(celsius) and MIN_SETPOINT_C <= celsius <= MAX_SETPOINT_C):
        raise ValueError(
            f"setpoint {celsius} C outside {MIN_SETPOINT_C}-{MAX_SETPOINT_C} C"
        )


def set_setpoint(celsius: float) -> bytes:
    """Return ``B4 02 <half-degrees>``: manual setpoint, 7-35 C."""
    _check_setpoint(celsius)
    return bytes([OP_WRITE, SUB_SETPOINT, _half_degrees(celsius)])


def set_mode(mode_code: int) -> bytes:
    """Return ``B4 <mode>`` for auto (01), manual (02) or off (04)."""
    if mode_code not in SETTABLE_MODES:
        raise ValueError(f"mode {mode_code!r} is not one of auto/manual/off")
    return bytes([OP_WRITE, mode_code])


def set_override(celsius: float) -> bytes:
    """Return ``B4 03 <half-degrees>``: temporary override setpoint, 7-35 C."""
    _check_setpoint(celsius)
    return bytes([OP_WRITE, SUB_OVERRIDE, _half_degrees(celsius)])


def write_presets(
    anti_frost_c: float,
    eco_c: float,
    comfort_c: float,
    mode_code: int | None = None,
    setpoint_c: float | None = None,
) -> bytes:
    """Return ``B6 <anti-frost> <eco> <comfort> [<mode> [<setpoint>]]``.

    Presets must be strictly increasing. Dialect B takes the manual or override
    target as a sixth byte (verified: the heater's active setpoint follows it).
    """
    values = (anti_frost_c, eco_c, comfort_c)
    for value in values:
        if not math.isfinite(value) or abs(value * 2 - round(value * 2)) > 1e-9:
            raise ValueError(f"preset {value} C is not a multiple of 0.5 C")
    if not MIN_SETPOINT_C <= anti_frost_c < eco_c < comfort_c <= MAX_SETPOINT_C:
        raise ValueError(
            "presets must satisfy "
            f"{MIN_SETPOINT_C} <= anti-frost < eco < comfort <= {MAX_SETPOINT_C}"
        )
    if mode_code is not None and mode_code not in PRESET_WRITE_MODES:
        raise ValueError(f"mode {mode_code!r} is not one of auto/manual/override/off")
    tail: tuple[int, ...] = () if mode_code is None else (mode_code,)
    if setpoint_c is not None:
        if mode_code not in SETPOINT_MODES:
            raise ValueError("a setpoint needs mode manual or override")
        _check_setpoint(setpoint_c)
        tail = (*tail, _half_degrees(setpoint_c))
    return bytes([OP_PRESET_WRITE, *(_half_degrees(v) for v in values), *tail])


def set_toggle(opcode: int, on: bool) -> bytes:
    """Return ``<opcode> 01|00`` for the boost/runback/EASY/lock toggles."""
    if opcode not in TOGGLE_OPCODES:
        raise ValueError(f"opcode {opcode!r} is not a known toggle")
    return bytes([opcode, 0x01 if on else 0x00])


def flash_display() -> bytes:
    """Return ``5E 01``: flash the heater's display to identify it."""
    return bytes([OP_FLASH_DISPLAY, 0x01])


def confirm_report() -> bytes:
    """Return ``57 55``, the confirmation sent after every ``56`` report."""
    return bytes([OP_CONFIRM_REPORT, REPLY_ACCEPTED])


def sync_clock(when: datetime, registering: bool, dialect: Dialect) -> bytes:
    """Return the EB clock sync ``5x YY MM DD DOW HH MM SS`` plus the dialect suffix."""
    prefix = OP_CLOCK_REGISTERING if registering else OP_CLOCK_STEADY
    fields = (
        when.year % 100,
        when.month,
        when.day,
        when.isoweekday() % 7,  # 0 = Sunday
        when.hour,
        when.minute,
        when.second,
    )
    return bytes([prefix, *fields]) + dialect.eb_clock_suffix


def write_program(
    week_slots: Sequence[Sequence[int]], wire_slots: int = SLOTS_HALF_HOURLY
) -> bytes:
    """Return ``B2`` + 84 (48-slot) or 42 (24-slot) bytes from 7 Monday-first days."""
    if len(week_slots) != DAYS_PER_WEEK:
        raise ValueError(f"expected {DAYS_PER_WEEK} days, got {len(week_slots)}")
    resolution = len(week_slots[0])
    if resolution not in (SLOTS_HOURLY, SLOTS_HALF_HOURLY) or any(
        len(day) != resolution for day in week_slots
    ):
        raise ValueError("each day needs 24 (hourly) or 48 (half-hourly) slots")
    if wire_slots not in (SLOTS_HOURLY, SLOTS_HALF_HOURLY):
        raise ValueError(f"wire_slots must be 24 or 48, got {wire_slots!r}")
    if resolution == SLOTS_HOURLY and wire_slots == SLOTS_HALF_HOURLY:
        week_slots = [[slot for slot in day for _ in range(2)] for day in week_slots]
    elif resolution == SLOTS_HALF_HOURLY and wire_slots == SLOTS_HOURLY:
        if any(day[0::2] != day[1::2] for day in week_slots):
            raise ValueError("half-hour changes cannot be sent in a 24-slot write")
        week_slots = [day[0::2] for day in week_slots]
    flat = [slot for day in week_slots for slot in day]
    wire = rotate_week(flat, wire_slots, to_wire=True)
    return bytes([OP_PROGRAM_WRITE]) + _pack_slots(wire)


def factory_reset(dialect: Dialect) -> bytes:
    """Return dialect B's factory reset ``C8 01 D0``; raise ValueError elsewhere.

    It wipes every setting, the clock, the program and the radio pairing.
    """
    if dialect is not DIALECT_B:
        raise ValueError(f"no known factory reset in dialect {dialect.name}")
    return bytes([OP_WRITE_PARAMETER, 0x01, 0xD0])


def power_verdict(granted: bool = True) -> bytes:
    """Return ``BF 01`` (granted) or ``BF 00``, the reply to a BE power request."""
    return bytes([OP_POWER_VERDICT, 0x01 if granted else 0x00])


def request_status() -> bytes:
    """Return the ``B8`` status request."""
    return bytes([OP_STATUS])


def request_program() -> bytes:
    """Return the ``B0`` program read request."""
    return bytes([OP_PROGRAM_READ])


def request_energy() -> bytes:
    """Return the ``BC`` energy counter request."""
    return bytes([OP_ENERGY])


def request_identity() -> bytes:
    """Return the ``5A`` identity request."""
    return bytes([OP_IDENTITY])


def request_advanced() -> bytes:
    """Return the ``DA`` advanced-setup record request."""
    return bytes([OP_ADVANCED_READ])


def reply_predicate(opcode: int, *payload_lens: int) -> Callable[[Frame], bool]:
    """Return a predicate matching a reply to ``opcode`` (first byte opcode + 1)."""
    expected = (opcode + 1) & 0xFF

    def _matches(frame: Frame) -> bool:
        """Return True when ``frame`` carries the expected reply payload."""
        payload = frame.payload
        return (
            len(payload) > 0
            and payload[0] == expected
            and (not payload_lens or len(payload) in payload_lens)
        )

    return _matches


def reply_verdict(payload: bytes, opcode: int) -> bool | None:
    """Return True/False for an ``<opcode+1> 55|56`` reply, else None."""
    if len(payload) != 2 or payload[0] != (opcode + 1) & 0xFF:
        return None
    if payload[1] == REPLY_ACCEPTED:
        return True
    if payload[1] == REPLY_REJECTED:
        return False
    return None


# --- status ------------------------------------------------------------------


@dataclass(frozen=True)
class StatusRecord:
    """A decoded status record; fields the form does not carry are None."""

    form: str  # "E5", "E3", "E6", "E4" or "short"
    mode_code: int
    anti_frost_c: float
    eco_c: float
    comfort_c: float
    room_temp_c: float | None
    setpoint_c: float | None
    measured_power_w: float | None
    duty: int | None
    flags: int | None
    boost_end_day: int | None  # 0 = Sunday
    boost_end_min: int | None  # minute of day
    raw: bytes

    @property
    def mode(self) -> str | None:
        """Return the mode name, or None for an unknown code."""
        return MODE_NAMES.get(self.mode_code)


_STATUS_FORMS = {
    STATUS_SHORT_LEN: ("short", False),
    STATUS_E6_LEN: ("E6", False),
    STATUS_E4_LEN: ("E4", False),
    STATUS_E5_LEN: ("E5", True),
    STATUS_E3_LEN: ("E3", True),
}


def decode_status(payload: bytes) -> StatusRecord | None:
    """Decode an E5/E3 report, E6/E4 status reply or short ``B9`` record."""
    payload = bytes(payload)
    form_info = _STATUS_FORMS.get(len(payload))
    if form_info is None:
        return None
    form, has_report_marker = form_info
    body = payload[1:] if has_report_marker else payload
    if (has_report_marker and payload[0] != OP_REPORT) or body[0] != STATUS_MARKER:
        return None
    full = len(body) >= STATUS_E6_LEN
    tail = len(body) == STATUS_E4_LEN
    raw_tail = int.from_bytes(body[-2:], "big") if tail else None
    return StatusRecord(
        form=form,
        mode_code=body[4],
        anti_frost_c=body[1] / 2,
        eco_c=body[2] / 2,
        comfort_c=body[3] / 2,
        room_temp_c=int.from_bytes(body[5:7], "big") / 10 if full else None,
        setpoint_c=body[7] / 2 if full else None,
        measured_power_w=int.from_bytes(body[8:10], "big") / 10 if full else None,
        duty=body[10] if full else None,
        flags=body[11] if full else None,
        boost_end_day=raw_tail >> 12 if raw_tail is not None else None,
        boost_end_min=raw_tail & 0x0FFF if raw_tail is not None else None,
        raw=payload,
    )


# --- program -----------------------------------------------------------------


@dataclass(frozen=True)
class ProgramRecord:
    """A weekly program at its own resolution, wire order (day 0 Sunday)."""

    resolution: int  # 24 or 48 slots a day
    slots: tuple[int | None, ...]
    hourly: tuple[int | None, ...]  # 168 values, None where unknown or split
    raw: bytes  # data bytes after the B1 opcode

    @property
    def slots_monday_first(self) -> tuple[int | None, ...]:
        """Return ``slots`` rotated to Monday-first day order."""
        return tuple(rotate_week(list(self.slots), self.resolution, to_wire=False))

    @property
    def hourly_monday_first(self) -> tuple[int | None, ...]:
        """Return ``hourly`` rotated to Monday-first day order."""
        return tuple(rotate_week(list(self.hourly), SLOTS_HOURLY, to_wire=False))

    @property
    def hourly_only(self) -> bool:
        """Return True when no hour is split into two different half hours."""
        if self.resolution == SLOTS_HOURLY:
            return True
        return all(
            a == b for a, b in zip(self.slots[0::2], self.slots[1::2], strict=True)
        )


def rotate_week[T](slots: Sequence[T], resolution: int, to_wire: bool) -> list[T]:
    """Rotate a flat week between Monday-first and the wire's Sunday-first order."""
    if len(slots) != resolution * DAYS_PER_WEEK:
        raise ValueError(
            f"expected {resolution * DAYS_PER_WEEK} slots, got {len(slots)}"
        )
    days = [slots[d * resolution : (d + 1) * resolution] for d in range(DAYS_PER_WEEK)]
    shift = 1 if to_wire else -1
    rotated: list[Sequence[T]] = [()] * DAYS_PER_WEEK
    for index, day in enumerate(days):
        rotated[(index + shift) % DAYS_PER_WEEK] = day
    return [slot for day in rotated for slot in day]


def _unpack_slots(raw: bytes) -> list[int | None]:
    """Expand 2-bit slot codes, MSB first; the unused code 3 becomes None."""
    slots: list[int | None] = []
    for byte in raw:
        for shift in (6, 4, 2, 0):
            code = (byte >> shift) & 0x3
            slots.append(code if code in SLOT_CODES else None)
    return slots


def _pack_slots(slots: Sequence[int]) -> bytes:
    """Pack slot codes four to a byte, MSB first; raise on an unknown code."""
    out = bytearray()
    for index in range(0, len(slots), SLOTS_PER_BYTE):
        byte = 0
        for offset, slot in enumerate(slots[index : index + SLOTS_PER_BYTE]):
            if slot not in SLOT_CODES:
                raise ValueError(f"unknown program slot value {slot!r}")
            byte |= slot << (6 - 2 * offset)
        out.append(byte)
    return bytes(out)


def decode_program(payload: bytes) -> ProgramRecord | None:
    """Decode a ``B1`` program reply (43 or 85 bytes) or a ``56 B1`` program report."""
    payload = bytes(payload)
    if payload[:2] == bytes([OP_REPORT, OP_PROGRAM_READ + 1]):
        payload = payload[1:]
    resolution = _PROGRAM_RESOLUTION.get(len(payload))
    if resolution is None or payload[0] != OP_PROGRAM_READ + 1:
        return None
    raw = payload[1:]
    slots = _unpack_slots(raw)
    if resolution == SLOTS_HOURLY:
        hourly = list(slots)
    else:
        hourly = [
            a if a == b else None for a, b in zip(slots[0::2], slots[1::2], strict=True)
        ]
    return ProgramRecord(
        resolution=resolution, slots=tuple(slots), hourly=tuple(hourly), raw=raw
    )


# --- identity ----------------------------------------------------------------


@dataclass(frozen=True)
class IdentityRecord:
    """A decoded ``5A`` identity reply in its E0 (``5B 77``) or short (``5B 55``) form."""

    form: str  # "E0" or "short"
    node_id: int | None  # only the E0 form carries one
    tail: bytes  # 12 bytes (E0) or 16 bytes (short)
    raw: bytes

    @property
    def ascii_serial(self) -> str | None:
        """Return the short form's trailing printable-ASCII run, or None."""
        if self.form != "short":
            return None
        end = len(self.tail)
        start = end
        while start > 0 and 0x20 < self.tail[start - 1] < 0x7F:
            start -= 1
        return self.tail[start:end].decode("ascii") if start < end else None


def decode_identity(payload: bytes) -> IdentityRecord | None:
    """Decode a ``5B 77`` (E0) or ``5B 55`` (short) identity reply."""
    payload = bytes(payload)
    if len(payload) < 2 or payload[0] != IDENTITY_MARKER:
        return None
    if len(payload) == IDENTITY_E0_LEN and payload[1] == IDENTITY_E0_FORM:
        return IdentityRecord("E0", payload[5], payload[6:18], payload)
    if len(payload) == IDENTITY_SHORT_LEN and payload[1] == IDENTITY_SHORT_FORM:
        return IdentityRecord("short", None, payload[2:], payload)
    return None


# --- energy ------------------------------------------------------------------


@dataclass(frozen=True)
class EnergyRecord:
    """A ``BD`` reply to ``BC``: a Wh counter (dialect A) or a power record (dialect B)."""

    energy_wh: int | None  # dialect A only; dialect B's power record has no counter
    raw: bytes


def decode_energy(payload: bytes) -> EnergyRecord | None:
    """Decode ``BD`` + u32 Wh (5 bytes) or the 9-byte dialect-B ``BD`` power record."""
    payload = bytes(payload)
    if not payload or payload[0] != ENERGY_MARKER:
        return None
    if len(payload) == ENERGY_A_LEN:
        return EnergyRecord(int.from_bytes(payload[1:5], "big"), payload)
    if len(payload) == ENERGY_B_LEN:
        return EnergyRecord(None, payload)
    return None


# --- power request -----------------------------------------------------------


@dataclass(frozen=True)
class PowerRequest:
    """A heater's ``BE`` power request, answered with ``BF 01`` (power_verdict)."""

    form: str  # "A" (3 bytes) or "B" (9 bytes)
    measured_power_w: float | None  # dialect A: full-load power; B: not carried
    raw: bytes  # dialect B's fields are not yet understood and stay here


def _deciwatts(data: bytes) -> float:
    """Return big-endian deciwatts as watts."""
    return int.from_bytes(data, "big") / 10


def decode_power_request(payload: bytes) -> PowerRequest | None:
    """Decode ``BE <hi> <lo>`` (dialect A) or the 9-byte ``BE`` record (dialect B)."""
    payload = bytes(payload)
    if not payload or payload[0] != OP_POWER_REQUEST:
        return None
    if len(payload) == POWER_REQUEST_A_LEN:
        return PowerRequest("A", _deciwatts(payload[1:3]), payload)
    if len(payload) == POWER_REQUEST_B_LEN:
        return PowerRequest("B", None, payload)
    return None


# --- dialect-B power record --------------------------------------------------

POWER_RECORD_HEATING = 0x01


@dataclass(frozen=True)
class PowerRecord:
    """Dialect B's 9-byte ``BE``/``BD`` record.

    ``<op> <room> 00 <setpoint> <v lo> <v hi> <duty> <heating> 00``: room
    temperature in tenths of a degree, the active setpoint in half degrees,
    mains voltage in 1/64 V, duty in percent, and 01 while heating (checked
    against a house meter and a reference thermometer).
    """

    heating: bool
    mains_voltage_v: float
    duty_pct: int
    room_temp_c: float  # byte 1, tenths of a degree
    setpoint_c: float  # byte 3, half degrees: the target the heater works to
    raw: bytes


def decode_power_record(payload: bytes) -> PowerRecord | None:
    """Decode a dialect-B ``BE`` or ``BD`` power record (9 bytes), else None."""
    payload = bytes(payload)
    if len(payload) != POWER_REQUEST_B_LEN or payload[0] not in (
        OP_POWER_REQUEST,
        ENERGY_MARKER,
    ):
        return None
    return PowerRecord(
        heating=payload[7] == POWER_RECORD_HEATING,
        mains_voltage_v=int.from_bytes(payload[4:6], "little") / 64,
        duty_pct=payload[6],
        room_temp_c=payload[1] / 10,
        setpoint_c=payload[3] / 2,
        raw=payload,
    )


# --- unsolicited -------------------------------------------------------------


class Unsolicited(enum.StrEnum):
    """What an unsolicited heater-to-station frame is."""

    REGISTRATION = "registration"
    REPORT = "report"
    PROGRAM_REPORT = "program_report"
    POWER_REQUEST = "power_request"
    ACK = "ack"
    ROUTE_PROBE = "route_probe"
    UNKNOWN = "unknown"


def classify_unsolicited(item: Frame | bytes) -> Unsolicited:
    """Classify a frame (or bare payload) a heater sent without being asked."""
    if isinstance(item, Frame):
        if item.is_ack:
            return Unsolicited.ACK
        if not item.payload and item.tag == ROUTE_PROBE_TAG:
            return Unsolicited.ROUTE_PROBE
        payload = item.payload
    else:
        payload = bytes(item)
    if payload == bytes([OP_REGISTRATION]):
        return Unsolicited.REGISTRATION
    if payload[:2] == bytes([OP_REPORT, OP_PROGRAM_READ + 1]):
        return Unsolicited.PROGRAM_REPORT
    if payload[:1] == bytes([OP_REPORT]):
        return Unsolicited.REPORT
    if decode_power_request(payload) is not None:
        return Unsolicited.POWER_REQUEST
    return Unsolicited.UNKNOWN
