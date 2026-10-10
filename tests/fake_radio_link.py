"""In-memory RadioLink stand-in with scripted heater replies (no sockets)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from custom_components.termoweb.backend.radio.dialect import (
    DIALECT_B,
    Dialect,
    build_ack,
    build_frame,
    decode,
)
from custom_components.termoweb.backend.radio.link import (
    AckResult,
    GatewayInfo,
    RadioLinkError,
    ReceivedFrame,
)

HEATER = 6
STATION = 1
MAC = "AA:BB:CC:00:11:22"
NET = bytes.fromhex("1234")  # synthetic dialect-B network id

# Payloads captured from the dialect-B reference heater (node 06); the
# identity tail is synthetic.
STATUS_SHORT = bytes.fromhex("B921252A02")
IDENTITY_SHORT = bytes.fromhex("5B55010203040506070809" + b"X123456".hex())
PROGRAM_HOURLY = bytes.fromhex("B155" + "6AAAAAAA9555" * 6 + "6AAAAAAA95")
REGISTRATION = bytes.fromhex("50")
POWER_REQUEST = bytes.fromhex("BEDA002CA9390B0100")
POWER_RECORD_IDLE = bytes.fromhex("BDDC002C783A000000")  # BC reply: not heating
POWER_RECORD_HEATING = bytes.fromhex("BDDB002CE43A0C0100")  # granted, heating
CLOCK_ACCEPTED = bytes.fromhex("5355")


def gateway_info(
    mac: str | None = MAC, version: str | None = "3.6-esp32"
) -> GatewayInfo:
    """Return a GatewayInfo like the firmware's ``# Q`` line."""

    return GatewayInfo(version, "869.525", "2DD4", True, STATION, mac, "# Q ...")


def received(
    src: int,
    payload: bytes,
    *,
    dst: int = STATION,
    dialect: Dialect = DIALECT_B,
    tag: int = 0,
) -> ReceivedFrame:
    """Return a decoded heater frame as the link would deliver it."""

    air = build_frame(
        dialect,
        src,
        dst,
        payload,
        tag=tag,
        path=(src, dst, 1, 1, 1),
        network_id=dialect.network_id or NET,
    )
    return ReceivedFrame(decode(dialect, air), -60.0, 0, 1000)


def received_ack(src: int, dst: int = STATION) -> ReceivedFrame:
    """Return a link-layer ack frame from ``src``."""

    return ReceivedFrame(
        decode(DIALECT_B, build_ack(DIALECT_B, src, dst, NET)), -60.0, 0, 1
    )


class FakeRadioLink:
    """Scripted link: acks frames, answers opcodes, records what was sent."""

    def __init__(
        self,
        host: str = "radio.local",
        port: int = 2323,
        dialect: Dialect = DIALECT_B,
        *,
        station_id: int = STATION,
        network_id: bytes | None = None,
        on_disconnect: Callable[[], None] | None = None,
        **_kwargs: Any,
    ) -> None:
        self.host = host
        self.port = port
        self.dialect = dialect
        self.station_id = station_id
        self.network_id = dialect.network_id if network_id is None else network_id
        self.on_disconnect = on_disconnect
        self.connected = False
        self.connects = 0
        self.closes = 0
        self.connect_errors: list[Exception] = []
        self.info: GatewayInfo | None = gateway_info()
        self.gateway_info: GatewayInfo | None = None
        self.listeners: list[Callable[[ReceivedFrame], None]] = []
        self.line_listeners: list[Callable[[str], None]] = []
        self.dialects: list[str] = []
        self.dialect_error: Exception | None = None
        self.kwargs = _kwargs
        self.sent: list[tuple[int, bytes]] = []
        self.replies: dict[int, list[bytes]] = {}
        self.no_ack: set[int] = set()
        self.on_send: Callable[[int, bytes], None] | None = None
        self.surveys: list[int] = []
        self.survey_bursts: list[Any] = []
        self.network_ids: list[bytes] = []

    # --- scripting -----------------------------------------------------------

    def reply(self, opcode: int, *payloads: bytes) -> None:
        """Queue replies to ``opcode``; the last one repeats forever."""

        self.replies[opcode] = list(payloads)

    def deliver(self, frame: ReceivedFrame) -> None:
        """Hand a frame to every registered listener."""

        for listener in list(self.listeners):
            listener(frame)

    def deliver_line(self, line: str) -> None:
        """Hand a non-frame gateway line to every line listener."""

        for listener in list(self.line_listeners):
            listener(line)

    def drop(self) -> None:
        """Simulate the gateway closing an established connection."""

        self.connected = False
        if self.on_disconnect is not None:
            self.on_disconnect()

    def payloads(self) -> list[bytes]:
        """Return every payload sent, in order."""

        return [payload for _dst, payload in self.sent]

    # --- RadioLink API -------------------------------------------------------

    async def connect(self) -> GatewayInfo | None:
        self.connects += 1
        if self.connect_errors:
            raise self.connect_errors.pop(0)
        self.connected = True
        self.gateway_info = self.info
        return self.info

    async def close(self) -> None:
        self.closes += 1
        self.connected = False

    def add_listener(self, callback: Callable[[ReceivedFrame], None]):
        self.listeners.append(callback)

        def _remove() -> None:
            if callback in self.listeners:
                self.listeners.remove(callback)

        return _remove

    def add_line_listener(self, callback: Callable[[str], None]):
        self.line_listeners.append(callback)

        def _remove() -> None:
            if callback in self.line_listeners:
                self.line_listeners.remove(callback)

        return _remove

    async def set_dialect(self, dialect: Dialect) -> None:
        if self.dialect_error is not None:
            raise self.dialect_error
        self.dialect = dialect
        self.dialects.append(dialect.name)

    async def send_frame(self, dst: int, air: bytes, **_kwargs: Any) -> AckResult:
        if not self.connected:
            raise RadioLinkError("not connected")
        frame = decode(self.dialect, air)
        assert frame.ok and frame.src == self.station_id and frame.dst == dst
        self.sent.append((dst, frame.payload))
        if self.on_send is not None:
            self.on_send(dst, frame.payload)
        if dst in self.no_ack:
            return AckResult(False, 3, 100)
        queue = self.replies.get(frame.payload[0])
        if queue:
            payload = queue.pop(0) if len(queue) > 1 else queue[0]
            self.deliver(received(dst, payload, dialect=self.dialect))
        return AckResult(True, 1, 100, received_ack(dst))

    async def set_network_id(self, network_id: bytes) -> None:
        self.network_id = bytes(network_id)
        self.network_ids.append(self.network_id)

    async def survey(self, seconds: int) -> list[Any]:
        if not self.connected:
            raise RadioLinkError("not connected")
        self.surveys.append(seconds)
        return list(self.survey_bursts)
