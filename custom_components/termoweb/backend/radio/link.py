"""Asyncio client for the ESP32 radio gateway's TCP line protocol (port 2323).

The gateway prints one line per event (``RX``, ``TX``, ``TXERR``, ``ACK`` or a
``# ...`` status line) and accepts one-letter commands (``T<hex>``, ``I<id>``,
``A1``, ``Y<mode>``, ``N<net>``, ``Q``). RX lines carry raw on-air bytes, which
this client decodes with its configured :class:`Dialect`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import contextlib
from dataclasses import dataclass
import logging
import time
from typing import Any

from .dialect import Dialect, Frame, build_frame, decode
from .survey import RawBurst, parse_run_tokens

_LOGGER = logging.getLogger(__name__)

DEFAULT_PORT = 2323
BANNER_PREFIX = "# termoweb_rx"
QUERY_PREFIX = "# Q "
TXERR_BAD_HEX = "TXERR empty or bad hex"

CONNECT_TIMEOUT_S = 5.0
TX_CONFIRM_TIMEOUT_S = 3.0
# A gateway that loses power leaves a half-open socket the OS may only notice
# after ~15 minutes; this many missed TX confirmations in a row drop it sooner.
MAX_MISSED_TX_CONFIRMS = 3
TXERR_RETRY_WAIT_S = 0.05
MIN_COMMAND_GAP_S = 0.04
RETRY_COUNT = 3
RETRY_INTERVAL_S = 0.16
DEFAULT_REPLY_TIMEOUT_S = 0.3
SURVEY_OFF = "# survey off"
SURVEY_GRACE_S = 10.0  # extra wait for the gateway's "survey off" line
MAX_SURVEY_S = 3600

OpenConnection = Callable[..., Awaitable[tuple[Any, Any]]]
Sleep = Callable[[float], Awaitable[Any]]


class RadioLinkError(Exception):
    """Raised when the gateway connection fails or refuses a command."""


class UnsupportedDialectError(RadioLinkError):
    """Raised when the gateway firmware cannot speak the requested dialect."""


@dataclass(frozen=True)
class GatewayInfo:
    """Fields parsed from the gateway's ``# Q ...`` status line."""

    version: str | None
    freq: str | None
    sync: str | None
    autoack: bool | None
    station_id: int | None
    mac: str | None
    raw: str
    dialect: str | None = None  # None: firmware without runtime dialects (dialect A)


@dataclass(frozen=True)
class ReceivedFrame:
    """One decoded RX frame plus the radio's own receive metadata."""

    frame: Frame
    rssi_dbm: float | None
    lqi: int | None
    micros: int | None


@dataclass(frozen=True)
class AckResult:
    """Outcome of :meth:`RadioLink.send_frame`."""

    ok: bool
    attempts: int  # T commands written, including a TXERR resend
    tx_micros: int | None
    ack: ReceivedFrame | None = None


def _int_or_none(text: str, base: int = 10) -> int | None:
    """Return ``text`` parsed as an int, or None when it is not one."""
    try:
        return int(text, base)
    except ValueError:
        return None


def _float_or_none(text: str) -> float | None:
    """Return ``text`` parsed as a float, or None when it is not one."""
    try:
        return float(text)
    except ValueError:
        return None


def parse_rx_line(
    line: str,
) -> tuple[bytes, float | None, int | None, int | None] | None:
    """Parse ``RX <micros> <rssi> <lqi> <crc> <HEX>`` into (air, rssi, lqi, micros)."""
    parts = line.split()
    if len(parts) < 6 or parts[0] != "RX":
        return None
    try:
        air = bytes.fromhex(parts[5])
    except ValueError:
        return None
    return air, _float_or_none(parts[2]), _int_or_none(parts[3]), _int_or_none(parts[1])


def parse_query_line(line: str) -> GatewayInfo | None:
    """Parse ``# Q termoweb_rx <ver> key=value ...`` into a GatewayInfo."""
    if not line.startswith(QUERY_PREFIX):
        return None
    tokens = line[len(QUERY_PREFIX) :].split()
    fields = dict(token.split("=", 1) for token in tokens if "=" in token)
    version = None
    if "termoweb_rx" in tokens:
        index = tokens.index("termoweb_rx") + 1
        if index < len(tokens) and "=" not in tokens[index]:
            version = tokens[index]
    autoack = {"on": True, "off": False}.get(fields.get("autoack", ""))
    station = fields.get("id")
    return GatewayInfo(
        version=version,
        freq=fields.get("freq"),
        sync=fields.get("sync"),
        autoack=autoack,
        station_id=_int_or_none(station, 16) if station is not None else None,
        mac=fields.get("mac"),
        raw=line,
        dialect=fields.get("dialect"),
    )


class RadioLink:
    """One TCP connection to the ESP32 radio gateway, serialising every transmit."""

    def __init__(
        self,
        host: str,
        port: int = DEFAULT_PORT,
        dialect: Dialect | None = None,
        *,
        station_id: int = 1,
        network_id: bytes | None = None,
        auto_ack: bool = True,
        clock: Callable[[], float] = time.monotonic,
        sleep: Sleep = asyncio.sleep,
        open_connection: OpenConnection = asyncio.open_connection,
        on_disconnect: Callable[[], None] | None = None,
    ) -> None:
        """Store connection parameters; nothing is opened until connect()."""
        if dialect is None:
            raise ValueError("dialect is required")
        if not 0 < station_id < 0xFF:
            raise ValueError("station_id must be 1..254")
        self._host = host
        self._port = port
        self._dialect = dialect
        self._station_id = station_id
        net = dialect.network_id if network_id is None else network_id
        if net is None:
            raise ValueError(f"dialect {dialect.name} needs an explicit network_id")
        self._network_id = bytes(net)
        self._auto_ack = auto_ack
        if len(self._network_id) != 2:
            raise ValueError("network_id must be exactly two bytes")
        self._clock = clock
        self._sleep = sleep
        self._open_connection = open_connection
        self._on_disconnect = on_disconnect
        self._survey: dict[int, tuple[float | None, list[tuple[int, int]]]] | None = (
            None
        )
        self._missed_confirms = 0
        self._reader: Any = None
        self._writer: Any = None
        self._read_task: asyncio.Task[None] | None = None
        self._closing = False
        self._established = False
        self._send_lock = asyncio.Lock()
        self._last_command_at: float | None = None
        self._listeners: list[Callable[[ReceivedFrame], None]] = []
        self._line_waiters: list[tuple[Callable[[str], bool], asyncio.Future[str]]] = []
        self._frame_waiters: list[
            tuple[Callable[[ReceivedFrame], bool], asyncio.Future[ReceivedFrame]]
        ] = []
        self.gateway_info: GatewayInfo | None = None

    @property
    def dialect(self) -> Dialect:
        """Return the dialect this link frames and decodes with."""
        return self._dialect

    @property
    def station_id(self) -> int:
        """Return this station's own radio id."""
        return self._station_id

    @property
    def network_id(self) -> bytes:
        """Return the two-byte network id used for frames and firmware auto-acks."""
        return self._network_id

    @property
    def connected(self) -> bool:
        """Return True while the socket is open and the reader is running."""
        return (
            self._writer is not None
            and self._read_task is not None
            and not self._read_task.done()
            and not self._closing
        )

    # --- lifecycle -----------------------------------------------------------

    async def connect(self) -> GatewayInfo:
        """Open the socket, configure the gateway and return its ``Q`` status."""
        if self.connected:
            raise RadioLinkError("already connected")
        _LOGGER.info("Connecting to radio gateway %s:%s", self._host, self._port)
        self._closing = False
        self._established = False
        try:
            self._reader, self._writer = await self._open_connection(
                self._host, self._port
            )
        except OSError as err:
            raise RadioLinkError(
                f"cannot connect to {self._host}:{self._port}"
            ) from err
        banner = self._add_line_waiter(lambda line: line.startswith(BANNER_PREFIX))
        self._read_task = asyncio.get_running_loop().create_task(self._read_loop())
        try:
            query_line = await self._handshake(banner)
        except BaseException:
            await self.close()
            raise
        info = parse_query_line(query_line)
        assert info is not None  # the waiter only matches Q lines
        if info.dialect is None and self._dialect.firmware_mode != 0:
            await self.close()
            raise UnsupportedDialectError(
                f"firmware does not support dialect {self._dialect.name}; "
                "update the nanoCUL firmware"
            )
        self.gateway_info = info
        self._established = True
        expected_sync = self._dialect.sync.hex().upper()
        if info.sync is not None and info.sync.upper() != expected_sync:
            _LOGGER.error(
                "Gateway reports sync %s but dialect %s needs %s",
                info.sync,
                self._dialect.name,
                expected_sync,
            )
        _LOGGER.info("Radio gateway connected: %s", info.raw)
        return info

    async def _handshake(self, banner: asyncio.Future[str]) -> str:
        """Await the banner, send the setup commands and return the ``Q`` line."""
        banner_line = await self._wait(banner, CONNECT_TIMEOUT_S)
        if banner_line is None:
            raise RadioLinkError("no termoweb_rx banner from the gateway")
        _LOGGER.debug("Gateway banner: %s", banner_line)
        for command in (
            f"I{self._station_id:02X}",
            "A1" if self._auto_ack else "A0",
            f"Y{self._dialect.firmware_mode}",
            f"N{self._network_id.hex().upper()}",
        ):
            await self._send_command(command)
        query = self._add_line_waiter(lambda line: line.startswith(QUERY_PREFIX))
        await self._send_command("Q")
        query_line = await self._wait(query, CONNECT_TIMEOUT_S)
        if query_line is None:
            raise RadioLinkError("no Q status from the gateway")
        return query_line

    async def close(self) -> None:
        """Close the socket and stop the reader; pending waits fail."""
        self._closing = True
        task, self._read_task = self._read_task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await self._close_writer()
        self._fail_waiters(RadioLinkError("connection closed"))
        _LOGGER.info("Radio gateway connection closed")

    async def _close_writer(self) -> None:
        """Close the stream writer, ignoring errors from an already dead socket."""
        writer, self._writer = self._writer, None
        if writer is None:
            return
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()

    # --- listeners -----------------------------------------------------------

    def add_listener(
        self, callback: Callable[[ReceivedFrame], None]
    ) -> Callable[[], None]:
        """Register ``callback`` for every decoded RX frame; return a remover."""
        self._listeners.append(callback)

        def _remove() -> None:
            """Unregister the callback; safe to call more than once."""
            with contextlib.suppress(ValueError):
                self._listeners.remove(callback)

        return _remove

    # --- transmit ------------------------------------------------------------

    async def send_frame(
        self,
        dst: int,
        air: bytes,
        *,
        wait_ack: bool = True,
        retries: int = RETRY_COUNT,
        retry_interval: float = RETRY_INTERVAL_S,
    ) -> AckResult:
        """Transmit ``air`` and wait for ``dst``'s ack, retrying up to ``retries`` times."""
        async with self._send_lock:
            attempts = 0
            tx_micros: int | None = None
            for _ in range(max(retries, 1)):
                ack = self._add_frame_waiter(
                    lambda rx: (
                        rx.frame.is_ack
                        and rx.frame.src == dst
                        and rx.frame.dst == self._station_id
                    )
                )
                try:
                    tx_micros, written = await self._transmit(air)
                    attempts += written
                    if not wait_ack:
                        return AckResult(True, attempts, tx_micros)
                    received = await self._wait(ack, retry_interval)
                finally:
                    self._drop_waiter(ack)
                if received is not None:
                    _LOGGER.debug("Ack from %02X after %d attempt(s)", dst, attempts)
                    return AckResult(True, attempts, tx_micros, received)
            _LOGGER.debug("No ack from %02X after %d attempt(s)", dst, attempts)
            return AckResult(False, attempts, tx_micros)

    async def request(
        self,
        dst: int,
        payload: bytes,
        predicate: Callable[[Frame], bool],
        timeout: float = DEFAULT_REPLY_TIMEOUT_S,
        *,
        retries: int = RETRY_COUNT,
    ) -> Frame | None:
        """Send ``payload`` to ``dst`` and return its first reply matching ``predicate``."""
        air = build_frame(
            self._dialect, self._station_id, dst, payload, network_id=self._network_id
        )
        reply = self._add_frame_waiter(
            lambda rx: (
                rx.frame.ok
                and not rx.frame.is_ack
                and rx.frame.src == dst
                and rx.frame.dst == self._station_id
                and predicate(rx.frame)
            )
        )
        try:
            result = await self.send_frame(dst, air, retries=retries)
            if not result.ok:
                return None
            received = await self._wait(reply, timeout)
        finally:
            self._drop_waiter(reply)
        return None if received is None else received.frame

    async def _transmit(self, air: bytes) -> tuple[int | None, int]:
        """Write ``T<hex>`` and await ``TX``; resend once on a bad-hex TXERR."""
        command = "T" + bytes(air).hex().upper()
        for written in (1, 2):
            confirm = self._add_line_waiter(
                lambda line: line.startswith(("TX ", "TXERR"))
            )
            try:
                await self._send_command(command)
            except RadioLinkError:
                self._drop_waiter(confirm)
                raise
            line = await self._wait(confirm, TX_CONFIRM_TIMEOUT_S)
            if line is None:
                await self._note_missed_confirm()
                raise RadioLinkError("no TX confirmation from the gateway")
            self._missed_confirms = 0
            if line.startswith("TX "):
                parts = line.split()
                return _int_or_none(parts[1]) if len(parts) > 1 else None, written
            if line != TXERR_BAD_HEX or written == 2:
                raise RadioLinkError(f"gateway transmit error: {line}")
            _LOGGER.debug("Gateway reported '%s'; resending once", line)
            await self._sleep(TXERR_RETRY_WAIT_S)
        raise AssertionError("unreachable")  # pragma: no cover

    async def _note_missed_confirm(self) -> None:
        """Count a missed TX confirmation; drop a gateway that stopped answering."""
        self._missed_confirms += 1
        if self._missed_confirms < MAX_MISSED_TX_CONFIRMS:
            return
        _LOGGER.error(
            "Radio gateway missed %d TX confirmations; dropping the connection",
            self._missed_confirms,
        )
        self._missed_confirms = 0
        await self._close_writer()  # the reader sees EOF and reports the loss

    async def _send_command(self, command: str) -> None:
        """Write one command line, keeping at least MIN_COMMAND_GAP_S between commands."""
        writer = self._writer
        if writer is None:
            raise RadioLinkError("not connected")
        if self._last_command_at is not None:
            remaining = MIN_COMMAND_GAP_S - (self._clock() - self._last_command_at)
            if remaining > 0:
                await self._sleep(remaining)
        _LOGGER.debug("Gateway <- %s", command)
        try:
            writer.write((command + "\n").encode("ascii"))
            await writer.drain()
        except (OSError, RuntimeError) as err:
            raise RadioLinkError(f"write to gateway failed: {err}") from err
        self._last_command_at = self._clock()

    # --- receive -------------------------------------------------------------

    async def _read_loop(self) -> None:
        """Read gateway lines until EOF or error, dispatching each one."""
        try:
            while True:
                try:
                    raw = await self._reader.readline()
                except ValueError:
                    _LOGGER.debug("Discarding over-long gateway line")
                    continue
                if not raw:
                    break
                line = raw.decode("ascii", errors="replace").strip()
                if line:
                    self._dispatch(line)
        except (OSError, asyncio.IncompleteReadError) as err:
            _LOGGER.error("Radio gateway connection lost: %s", err)
        if self._closing:
            return
        _LOGGER.info("Radio gateway closed the connection")
        self._fail_waiters(RadioLinkError("connection lost"))
        await self._close_writer()
        if self._established and self._on_disconnect is not None:
            try:
                self._on_disconnect()
            except Exception:
                _LOGGER.exception("Radio disconnect callback failed")

    async def survey(self, seconds: int) -> list[RawBurst]:
        """Run the gateway's raw survey (``R<seconds>``) and return every burst heard.

        The gateway switches its radio to raw capture, reports each RF burst as
        ``RAWB``/``RAW``/``RAWE`` lines and then returns to normal reception. A
        burst without its ``RAWE`` line (lost in transit) is still returned.
        """
        if not 1 <= seconds <= MAX_SURVEY_S:
            raise ValueError(f"survey length must be 1-{MAX_SURVEY_S} s")
        self._survey = {}
        done = self._add_line_waiter(lambda line: line.startswith(SURVEY_OFF))
        try:
            await self._send_command(f"R{int(seconds)}")
            if await self._wait(done, seconds + SURVEY_GRACE_S) is None:
                _LOGGER.debug("Radio survey ended without a 'survey off' line")
            collected = self._survey
        finally:
            self._survey = None
            self._drop_waiter(done)
        return [
            RawBurst(rssi, tuple(runs))
            for _number, (rssi, runs) in sorted(collected.items())
        ]

    def _collect_survey(self, line: str) -> None:
        """Add one ``RAWB``/``RAW``/``RAWE`` line to the running survey."""
        survey = self._survey
        parts = line.split()
        if survey is None or len(parts) < 2 or not parts[1].isdigit():
            return
        number = int(parts[1])
        if parts[0] == "RAWB":
            rssi = _float_or_none(parts[2]) if len(parts) > 2 else None
            survey[number] = (rssi, [])
        elif parts[0] == "RAW":
            survey.setdefault(number, (None, []))[1].extend(parse_run_tokens(parts[2:]))

    def _dispatch(self, line: str) -> None:
        """Route one gateway line to waiters and listeners."""
        _LOGGER.debug("Gateway -> %s", line)
        if line.startswith(("RAWB ", "RAW ", "RAWE ")):
            self._collect_survey(line)  # raw survey data never reaches RX handling
            return
        if line.startswith("RX "):
            parsed = parse_rx_line(line)
            if parsed is None:
                return
            air, rssi, lqi, micros = parsed
            received = ReceivedFrame(decode(self._dialect, air), rssi, lqi, micros)
            self._resolve(self._frame_waiters, received)
            for listener in list(self._listeners):
                try:
                    listener(received)
                except Exception:
                    _LOGGER.exception("Radio frame listener failed")
            return
        self._resolve(self._line_waiters, line)

    @staticmethod
    def _resolve[T](
        waiters: list[tuple[Callable[[T], bool], asyncio.Future[T]]], item: T
    ) -> None:
        """Complete every pending waiter whose predicate accepts ``item``."""
        for predicate, future in list(waiters):
            if not future.done() and predicate(item):
                future.set_result(item)

    # --- waiting -------------------------------------------------------------

    def _add_line_waiter(self, predicate: Callable[[str], bool]) -> asyncio.Future[str]:
        """Register a future for the next line matching ``predicate``."""
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._line_waiters.append((predicate, future))
        return future

    def _add_frame_waiter(
        self, predicate: Callable[[ReceivedFrame], bool]
    ) -> asyncio.Future[ReceivedFrame]:
        """Register a future for the next RX frame matching ``predicate``."""
        future: asyncio.Future[ReceivedFrame] = (
            asyncio.get_running_loop().create_future()
        )
        self._frame_waiters.append((predicate, future))
        return future

    def _drop_waiter(self, future: asyncio.Future[Any]) -> None:
        """Forget a waiter and cancel it if still pending."""
        for waiters in (self._line_waiters, self._frame_waiters):
            for entry in list(waiters):
                if entry[1] is future:
                    waiters.remove(entry)
        if not future.done():
            future.cancel()
        elif not future.cancelled():
            future.exception()  # mark a stored error as retrieved

    def _fail_waiters(self, error: Exception) -> None:
        """Fail and forget every pending waiter."""
        for waiters in (self._line_waiters, self._frame_waiters):
            pending = [future for _, future in waiters if not future.done()]
            waiters.clear()
            for future in pending:
                future.set_exception(error)
                future.exception()  # an owner may never await it

    async def _wait[T](self, future: asyncio.Future[T], timeout: float) -> T | None:
        """Return ``future``'s result, or None if the injected sleep finishes first."""
        try:
            if not future.done():
                timer = asyncio.ensure_future(self._sleep(max(timeout, 0.0)))
                try:
                    await asyncio.wait(
                        (future, timer), return_when=asyncio.FIRST_COMPLETED
                    )
                finally:
                    timer.cancel()
            if not future.done():
                return None
            return future.result()
        finally:
            self._drop_waiter(future)
