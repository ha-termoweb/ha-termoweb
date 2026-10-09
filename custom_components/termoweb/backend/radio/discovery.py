"""Find a heater network from its own traffic, then confirm heaters by address.

A dialect-B network id belongs to one installation and has no default, so the
only way to learn it is to listen: heaters send registrations, route probes and
power requests on their own. Listening never transmits (auto-ack is off).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
import logging
import time

from . import protocol
from .dialect import DIALECT_A, DIALECT_B, Dialect
from .link import DEFAULT_PORT, RadioLink, ReceivedFrame

_LOGGER = logging.getLogger(__name__)

LISTEN_PLACEHOLDER_NET = b"\x00\x00"  # never transmitted: listening sends nothing
DISCOVERY_DIALECTS: tuple[Dialect, ...] = (DIALECT_B, DIALECT_A)
LISTEN_WINDOW_S = 30.0
LISTEN_TOTAL_S = 180.0
LISTEN_GRACE_S = 5.0  # keep listening this long after the first frame
POLL_STEP_S = 0.5
SCAN_ADDRESSES: tuple[int, ...] = tuple(range(2, 33))
PROBE_GAP_S = 0.2

LinkFactory = Callable[..., RadioLink]
Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True)
class NetworkSighting:
    """A heater network heard on air: its dialect, network id and senders."""

    dialect: Dialect
    network_id: bytes
    sources: frozenset[int] = field(default_factory=frozenset)


async def listen_once(
    host: str,
    port: int,
    dialect: Dialect,
    window_s: float,
    *,
    station_id: int = 1,
    link_factory: LinkFactory = RadioLink,
    sleep: Sleep = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> NetworkSighting | None:
    """Listen in ``dialect`` for up to ``window_s``; return the first network heard."""

    link = link_factory(
        host,
        port,
        dialect,
        station_id=station_id,
        network_id=LISTEN_PLACEHOLDER_NET,
        auto_ack=False,
    )
    networks: dict[bytes, set[int]] = {}
    first_seen: float | None = None

    def _on_frame(received: ReceivedFrame) -> None:
        """Record the network id and sender of every valid data frame."""

        nonlocal first_seen
        frame = received.frame
        if not frame.ok or frame.is_ack or frame.network_id is None:
            return
        senders = networks.setdefault(frame.network_id, set())
        if frame.src is not None and frame.src != station_id:
            senders.add(frame.src)
        if first_seen is None:
            first_seen = clock()

    await link.connect()
    remove = link.add_listener(_on_frame)
    try:
        start = clock()
        while clock() - start < window_s:
            if first_seen is not None and clock() - first_seen >= LISTEN_GRACE_S:
                break
            await sleep(POLL_STEP_S)
    finally:
        remove()
        await link.close()
    if not networks:
        return None
    network_id = max(networks, key=lambda net: len(networks[net]))
    _LOGGER.info(
        "Heard dialect %s network with %d sender(s)",
        dialect.name,
        len(networks[network_id]),
    )
    return NetworkSighting(dialect, network_id, frozenset(networks[network_id]))


async def discover_network(
    host: str,
    port: int = DEFAULT_PORT,
    *,
    dialects: Sequence[Dialect] = DISCOVERY_DIALECTS,
    window_s: float = LISTEN_WINDOW_S,
    total_s: float = LISTEN_TOTAL_S,
    link_factory: LinkFactory = RadioLink,
    sleep: Sleep = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> NetworkSighting | None:
    """Alternate listening windows across ``dialects`` until a network is heard."""

    start = clock()
    while clock() - start < total_s:
        for dialect in dialects:
            sighting = await listen_once(
                host,
                port,
                dialect,
                window_s,
                link_factory=link_factory,
                sleep=sleep,
                clock=clock,
            )
            if sighting is not None:
                return sighting
            if clock() - start >= total_s:
                break
    _LOGGER.info("No heater traffic heard on %s:%s", host, port)
    return None


async def probe_heaters(
    host: str,
    port: int,
    dialect: Dialect,
    network_id: bytes,
    candidates: Iterable[int],
    *,
    link_factory: LinkFactory = RadioLink,
    sleep: Sleep = asyncio.sleep,
) -> dict[int, protocol.StatusRecord]:
    """Return the status of every candidate address that answers a ``B8`` request."""

    link = link_factory(host, port, dialect, network_id=network_id)
    found: dict[int, protocol.StatusRecord] = {}
    await link.connect()
    try:
        for addr in sorted(set(candidates)):
            if not 0 < addr < 0xFF or addr == link.station_id:
                continue
            reply = await link.request(
                addr,
                protocol.request_status(),
                protocol.reply_predicate(protocol.OP_STATUS),
            )
            record = None if reply is None else protocol.decode_status(reply.payload)
            if record is not None:
                found[addr] = record
            await sleep(PROBE_GAP_S)
    finally:
        await link.close()
    _LOGGER.info("Radio scan found heaters %s", sorted(found))
    return found


__all__ = [
    "DISCOVERY_DIALECTS",
    "SCAN_ADDRESSES",
    "NetworkSighting",
    "discover_network",
    "listen_once",
    "probe_heaters",
]
