"""Confirm heaters by address on a known network, plus shared listening constants.

Learning a network from its traffic happens in the background, in
``radio_discovery``; listening never transmits (auto-ack is off).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
import logging

from . import protocol
from .dialect import Dialect
from .link import RadioLink

_LOGGER = logging.getLogger(__name__)

LISTEN_PLACEHOLDER_NET = b"\x00\x00"  # never transmitted: listening sends nothing
LISTEN_WINDOW_S = 10.0  # longer than the gap between frames in a spell (< 8 s)
SCAN_ADDRESSES: tuple[int, ...] = tuple(range(2, 33))
PROBE_GAP_S = 0.2
SURVEY_S = 120  # raw survey after silent discovery: heaters repeat within minutes

LinkFactory = Callable[..., RadioLink]
Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True)
class NetworkSighting:
    """A heater network: its dialect and network id."""

    dialect: Dialect
    network_id: bytes


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
            _acked, reply = await link.request(
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
    "SCAN_ADDRESSES",
    "SURVEY_S",
    "NetworkSighting",
    "probe_heaters",
]
