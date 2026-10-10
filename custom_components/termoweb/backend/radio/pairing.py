"""Pair heaters to this station: hear their announcements, assign radio ids.

A heater in pairing mode (its owner pressed pairing on the panel) announces
itself from the broadcast id ``FF``, sweeping the destination across the id
range. The station answers with one id assignment: tag ``04``, payload the new
id, sent to ``FF`` on the station's network id. The heater adopts both the id
and the network id. See ``docs/radio_protocol.md`` (section 7).

- Dialect B announces with tag ``03`` and payload ``55``, without an identity,
  so heaters are paired one at a time: after an assignment, further sweeps are
  ignored for a few seconds and the new id must answer a status read. The
  heater only takes an assignment sent right after the sweep frame addressed
  to this station, so only that frame is answered.
- Dialect A announces with ``77`` + a 12-byte identity. Copies relayed by
  already-paired heaters are ignored: the relay sits on its own network and
  would drop an assignment sent on this station's network id.

This module has no Home Assistant dependency and transmits only while
:func:`pair_heaters` or :func:`pair_new_network` runs.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
import hashlib
import logging
import time

from . import protocol
from .dialect import DIALECT_A, DIALECT_B, Dialect, Frame, build_frame
from .link import RadioLink, ReceivedFrame, rotate_dialects

_LOGGER = logging.getLogger(__name__)

BROADCAST_ID = 0xFF
ASSIGNMENT_TAG = 0x04
ANNOUNCE_TAG_B = 0x03
ANNOUNCE_PAYLOAD_B = b"\x55"
ANNOUNCE_MARKER_A = 0x77
ANNOUNCE_LEN_A = 13  # 77 + 12-byte identity
MIN_NODE_ID = 0x02
MAX_NODE_ID = 0x41  # the stock gateway's node tables stop here
# 0000 and FFFF look like "no network"; 1B30 is the stock dialect-A network.
RESERVED_NETWORK_IDS = frozenset({b"\x00\x00", b"\xff\xff", b"\x1b\x30"})

PAIR_WINDOW_S = 300.0  # how long the owner has to press pairing on a heater
PAIR_DIALECT_WINDOW_S = 15.0  # per-dialect listening slice when the dialect is unknown
IDLE_STOP_S = 60.0  # after a pairing, stop this long after the last announcement
SETTLE_S = 5.0  # ignore further sweeps this long after an assignment
CONFIRM_TIMEOUT_S = 1.0
CONFIRM_ATTEMPTS = 3
# Airtime: one transmission per assignment, at most one assignment a second,
# and only for an announcement still inside its sweep dwell (120-200 ms).
ASSIGN_GAP_S = 1.0
ASSIGN_ACK_WAIT_S = 0.2
ANSWER_MAX_AGE_S = 0.1
POLL_STEP_S = 0.02
PAIRING_DIALECTS: tuple[Dialect, ...] = (DIALECT_B, DIALECT_A)

LinkFactory = Callable[..., RadioLink]
Sleep = Callable[[float], Awaitable[None]]


class PairingError(Exception):
    """Pairing could not continue; ``paired`` holds the heaters paired before."""

    def __init__(self, message: str, paired: Sequence[PairedHeater] = ()) -> None:
        """Store the message and the heaters already paired."""
        super().__init__(message)
        self.paired = list(paired)


class NoFreeAddressError(PairingError):
    """Every radio id in MIN_NODE_ID..MAX_NODE_ID is taken."""


@dataclass(frozen=True)
class Announcement:
    """A heater's pairing announcement: its dialect and (dialect A) identity."""

    dialect: Dialect
    identity: bytes | None  # dialect A: 12 bytes; dialect B carries none
    relayed: bool  # dialect A: a copy forwarded by an already-paired heater
    dst: int | None = None  # the swept destination of this frame


@dataclass(frozen=True)
class PairedHeater:
    """A heater that took its id and answered a status read on it."""

    node_id: int
    status: protocol.StatusRecord
    identity: bytes | None = None  # from the announcement or the 5A reply
    serial: str | None = None  # dialect B: ASCII serial from the 5A reply


def classify_announcement(dialect: Dialect, frame: Frame) -> Announcement | None:
    """Return the pairing announcement ``frame`` carries in ``dialect``, else None."""
    if not frame.ok or frame.is_ack:
        return None
    payload = frame.payload
    if dialect is DIALECT_B:
        if (
            frame.src == BROADCAST_ID
            and frame.tag == ANNOUNCE_TAG_B
            and payload == ANNOUNCE_PAYLOAD_B
        ):
            return Announcement(dialect, None, relayed=False, dst=frame.dst)
        return None
    if len(payload) != ANNOUNCE_LEN_A or payload[0] != ANNOUNCE_MARKER_A:
        return None
    relayed = frame.path is None or frame.src != frame.path[0]
    return Announcement(dialect, payload[1:], relayed=relayed, dst=frame.dst)


def free_node_id(used: Iterable[int]) -> int | None:
    """Return the lowest radio id in MIN_NODE_ID..MAX_NODE_ID not in ``used``."""
    taken = set(used)
    return next(
        (node for node in range(MIN_NODE_ID, MAX_NODE_ID + 1) if node not in taken),
        None,
    )


def build_assignment(
    dialect: Dialect, node_id: int, network_id: bytes, station_id: int = 1
) -> bytes:
    """Return the on-air id assignment: tag 04, payload ``node_id``, to ``FF``."""
    if not 0 < node_id < BROADCAST_ID or node_id == station_id:
        raise ValueError(f"cannot assign radio id {node_id}")
    return build_frame(
        dialect,
        station_id,
        BROADCAST_ID,
        bytes([node_id]),
        tag=ASSIGNMENT_TAG,
        path=(station_id, BROADCAST_ID, 0, 0, 0),
        network_id=network_id,
    )


def site_network_id(seed: str) -> bytes:
    """Return a stable two-byte network id from ``seed`` (SHA-256), never reserved."""
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    for index in range(0, len(digest) - 1, 2):
        candidate = digest[index : index + 2]
        if candidate not in RESERVED_NETWORK_IDS:
            return candidate
    raise ValueError("no usable network id in the digest")  # pragma: no cover


async def _read_identity(
    link: RadioLink, node_id: int
) -> tuple[bytes | None, str | None]:
    """Return ``(identity, serial)`` from the heater's ``5A`` reply, if it answers."""
    _acked, reply = await link.request(
        node_id,
        protocol.request_identity(),
        protocol.reply_predicate(protocol.OP_IDENTITY),
        CONFIRM_TIMEOUT_S,
    )
    record = None if reply is None else protocol.decode_identity(reply.payload)
    if record is None:
        return None, None
    return record.tail, record.ascii_serial


async def _confirm(
    link: RadioLink, node_id: int, identity: bytes | None
) -> PairedHeater | None:
    """Return the paired heater once ``node_id`` answers a status read, else None."""
    for _ in range(CONFIRM_ATTEMPTS):
        _acked, reply = await link.request(
            node_id,
            protocol.request_status(),
            protocol.reply_predicate(protocol.OP_STATUS),
            CONFIRM_TIMEOUT_S,
        )
        status = None if reply is None else protocol.decode_status(reply.payload)
        if status is not None:
            break
    else:
        return None
    read_identity, serial = await _read_identity(link, node_id)
    return PairedHeater(node_id, status, identity or read_identity, serial)


async def pair_heaters(
    link: RadioLink,
    *,
    window_s: float = PAIR_WINDOW_S,
    total_s: float | None = None,
    idle_stop_s: float | None = None,
    existing_ids: Iterable[int] = (),
    wanted_id: int | None = None,
    max_heaters: int | None = None,
    settle_s: float = SETTLE_S,
    clock: Callable[[], float] = time.monotonic,
    sleep: Sleep = asyncio.sleep,
) -> list[PairedHeater]:
    """Answer pairing announcements on a connected ``link`` and return the paired heaters.

    The link's dialect and network id are used. ``window_s`` is how long to
    wait for the first heater. With ``idle_stop_s``, every announcement keeps
    the window open that long (up to ``total_s``), and after a pairing the run
    stops ``idle_stop_s`` after the last announcement. ``wanted_id`` pairs
    exactly one heater to that id (a heater re-paired after a factory reset);
    otherwise each heater gets the lowest free id. A NoFreeAddressError carries
    the heaters paired before the ids ran out.

    Dialect B is answered only for the sweep frame addressed to this station.
    Each assignment is sent once (no blind retries), at most one a second,
    and only while the announcement it answers is fresh.
    """
    dialect = link.dialect
    if wanted_id is not None:
        if not 0 < wanted_id < BROADCAST_ID or wanted_id == link.station_id:
            raise ValueError(f"cannot assign radio id {wanted_id}")
        max_heaters = 1
    used = set(existing_ids)
    paired: list[PairedHeater] = []
    paired_identities: set[bytes] = set()
    identity_ids: dict[bytes, int] = {}
    last_sent: float | None = None
    heard: list[tuple[float, Announcement]] = []

    def _on_frame(received: ReceivedFrame) -> None:
        """Queue every pairing announcement with the time it was heard."""
        announcement = classify_announcement(dialect, received.frame)
        if announcement is not None:
            heard.append((clock(), announcement))

    start = clock()
    hard_end = start + (window_s if total_s is None else max(total_s, window_s))
    deadline = start + window_s
    ignore_until = start
    remove = link.add_listener(_on_frame)
    _LOGGER.info(
        "Pairing heaters in dialect %s for up to %.0f s", dialect.name, window_s
    )
    try:
        while clock() < deadline:
            if not heard:
                await sleep(POLL_STEP_S)
                continue
            heard_at, announcement = heard.pop(0)
            identity = announcement.identity
            if announcement.relayed:
                _LOGGER.debug("Ignoring a relayed pairing announcement")
                continue
            if heard_at < ignore_until or identity in paired_identities:
                continue
            if idle_stop_s is not None:
                deadline = max(deadline, min(hard_end, clock() + idle_stop_s))
            if dialect is DIALECT_B and announcement.dst != link.station_id:
                continue  # the heater only listens right after its sweep to us
            now = clock()
            if now - heard_at > ANSWER_MAX_AGE_S:
                continue  # the sweep has moved on
            if last_sent is not None and now - last_sent < ASSIGN_GAP_S:
                continue
            last_sent = now
            node_id = wanted_id
            if node_id is None and identity is not None:
                node_id = identity_ids.get(identity)
            if node_id is None:
                node_id = free_node_id(used | {link.station_id})
                if node_id is None:
                    raise NoFreeAddressError(
                        f"no free radio id between {MIN_NODE_ID} and {MAX_NODE_ID}",
                        paired,
                    )
            if identity is not None:
                identity_ids[identity] = node_id
            _LOGGER.debug("Assigning radio id %d", node_id)
            air = build_assignment(dialect, node_id, link.network_id, link.station_id)
            result = await link.send_frame(
                BROADCAST_ID, air, retries=1, retry_interval=ASSIGN_ACK_WAIT_S
            )
            if not result.ok:
                _LOGGER.debug("No ack for the assignment of radio id %d", node_id)
                continue
            ignore_until = clock() + settle_s
            heater = await _confirm(link, node_id, identity)
            if heater is None:
                _LOGGER.info("Heater took radio id %d but did not answer", node_id)
                continue
            paired.append(heater)
            used.add(node_id)
            if identity is not None:
                paired_identities.add(identity)
            _LOGGER.info("Paired a heater as radio id %d", node_id)
            if max_heaters is not None and len(paired) >= max_heaters:
                break
            if idle_stop_s is not None:
                deadline = min(hard_end, clock() + idle_stop_s)
    finally:
        remove()
    _LOGGER.info("Pairing finished: %d heater(s) paired", len(paired))
    return paired


async def pair_new_network(
    host: str,
    port: int,
    network_id: bytes,
    *,
    dialects: Sequence[Dialect] = PAIRING_DIALECTS,
    total_s: float = PAIR_WINDOW_S,
    window_s: float = PAIR_DIALECT_WINDOW_S,
    idle_stop_s: float = IDLE_STOP_S,
    link_factory: LinkFactory = RadioLink,
    clock: Callable[[], float] = time.monotonic,
    sleep: Sleep = asyncio.sleep,
) -> tuple[Dialect | None, list[PairedHeater]]:
    """Pair heaters into ``network_id``, alternating ``dialects`` until one answers.

    A dialect whose announcement is heard keeps its slice open (see
    :func:`pair_heaters`); the first dialect that pairs a heater ends the
    rotation. A dialect the gateway firmware cannot speak is dropped; if none
    is left, the UnsupportedDialectError is raised.
    """

    async def _pair(
        dialect: Dialect, left: float
    ) -> tuple[Dialect, list[PairedHeater]] | None:
        """Pair in ``dialect`` for one slice; None when no heater was paired."""
        link = link_factory(host, port, dialect, network_id=network_id)
        await link.connect()
        try:
            paired = await pair_heaters(
                link,
                window_s=min(window_s, left),
                total_s=left,
                idle_stop_s=idle_stop_s,
                clock=clock,
                sleep=sleep,
            )
        finally:
            await link.close()
        return (dialect, paired) if paired else None

    result = await rotate_dialects(dialects, _pair, total_s, clock)
    if result is None:
        _LOGGER.info("No heater was paired on %s:%s", host, port)
        return None, []
    return result


__all__ = [
    "PAIRING_DIALECTS",
    "PAIR_WINDOW_S",
    "Announcement",
    "NoFreeAddressError",
    "PairedHeater",
    "PairingError",
    "build_assignment",
    "classify_announcement",
    "free_node_id",
    "pair_heaters",
    "pair_new_network",
    "site_network_id",
]
