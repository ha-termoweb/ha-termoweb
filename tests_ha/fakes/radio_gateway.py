"""In-memory ESP32 radio gateway and fake time for RadioLink tests.

Construct these inside a running event loop (the StreamReader needs one).
"""

from __future__ import annotations

from tests_ha.fakes.radio_link import build_ack

import asyncio
from collections import deque
from collections.abc import Callable

from custom_components.termoweb.backend.radio.dialect import (
    DIALECT_B,
    Dialect,
    build_frame,
    decode,
)

# Synthetic dialect-B network id used by every test vector.
NET = bytes.fromhex("1234")

BANNER = "# termoweb_rx 3.5 freq=869.525 rate=9.6k sync=2DD4 mode=dynamic tx=paC0"
Q_LINE = (
    "# Q termoweb_rx 3.6-esp32 freq=869.525 pa=C0 sync=2DD4 mode=dynamic autoack=on "
    "id=01 dialect=B net=1234"
)
# Stock AVR nanoCUL firmware: no runtime dialects, no MAC (dialect A only).
Q_LINE_NANOCUL = (
    "# Q termoweb_rx 3.5 freq=869.525 pa=C0 sync=2DE5 mode=dynamic autoack=on id=01"
)


def rx_line(
    air: bytes, *, micros: int = 5000, rssi: str = "-58.5", lqi: str = "0"
) -> str:
    """Return an RX line for raw on-air bytes."""
    return f"RX {micros} {rssi} {lqi} 1 {air.hex().upper()}"


class FakeTime:
    """Injectable clock and sleep; sleeping advances time and yields to the loop."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        for _ in range(20):
            await asyncio.sleep(0)


class FakeWriter:
    """StreamWriter stand-in that hands every write to the fake gateway."""

    def __init__(self, gateway: FakeGateway) -> None:
        self.gateway = gateway
        self.closed = False
        self.write_error: Exception | None = None
        self.wait_closed_error: Exception | None = None

    def write(self, data: bytes) -> None:
        if self.write_error is not None:
            raise self.write_error
        self.gateway.on_write(data)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True
        self.gateway.reader.feed_eof()  # like a socket: the reader sees EOF

    async def wait_closed(self) -> None:
        if self.wait_closed_error is not None:
            raise self.wait_closed_error


class FakeGateway:
    """Scripted gateway: answers Q/Y/N, confirms T lines and runs a responder."""

    def __init__(self, dialect: Dialect = DIALECT_B) -> None:
        self.dialect = dialect
        self.reader = asyncio.StreamReader()
        self.writer = FakeWriter(self)
        self.commands: list[str] = []
        self.banner: str | None = BANNER
        self.q_line: str | None = Q_LINE
        self.tx_overrides: deque[str] = deque()
        self.responder: Callable[[bytes], list[str]] | None = None
        self.opened: list[tuple[str, int]] = []
        self.open_error: Exception | None = None
        self.micros = 1000
        self.survey_lines: list[str] = []  # fed in answer to an R<seconds> command
        self.silent = False  # True: T lines get no TX confirmation (powered off)

    async def open_connection(self, host: str, port: int):
        self.opened.append((host, port))
        if self.open_error is not None:
            raise self.open_error
        if self.banner is not None:
            self.feed(self.banner)
        return self.reader, self.writer

    def feed(self, line: str) -> None:
        self.reader.feed_data((line + "\r\n").encode())

    def transmitted(self) -> list[bytes]:
        return [bytes.fromhex(c[1:]) for c in self.commands if c.startswith("T")]

    def on_write(self, data: bytes) -> None:
        for line in data.decode().splitlines():
            self.commands.append(line)
            if line == "Q" and self.q_line is not None:
                self.feed(self.q_line)
            elif line[:1] in ("Y", "N"):
                self.feed(f"# {line} ok")
            elif line[:1] == "R" and line[1:].isdigit():
                for survey_line in self.survey_lines:
                    self.feed(survey_line)
            elif line.startswith("T"):
                if self.silent:
                    continue
                if self.tx_overrides:
                    self.feed(self.tx_overrides.popleft())
                    continue
                self.micros += 100
                self.feed(f"TX {self.micros} {len(line[1:]) // 2} {line[1:]}")
                if self.responder is not None:
                    for reply in self.responder(bytes.fromhex(line[1:])):
                        self.feed(reply)


def heater(
    dialect: Dialect,
    node: int,
    replies: dict[int, bytes] | None = None,
    *,
    ack: bool = True,
    station: int = 1,
    network_id: bytes = NET,
) -> Callable[[bytes], list[str]]:
    """Return a responder that acks frames to ``node`` and answers opcodes."""

    def _respond(air: bytes) -> list[str]:
        frame = decode(dialect, air)
        if not frame.ok or frame.dst != node:
            return []
        lines = []
        if ack:
            lines.append(rx_line(build_ack(dialect, node, station, network_id)))
        reply = (replies or {}).get(frame.payload[0]) if frame.payload else None
        if reply is not None:
            lines.append(
                rx_line(
                    build_frame(
                        dialect,
                        node,
                        station,
                        reply,
                        path=(node, station, 1, 1, 1),
                        network_id=network_id,
                    )
                )
            )
        return lines

    return _respond
