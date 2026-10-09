"""HTTP-client-shaped adapter that serves the integration over the local radio.

``RadioClient`` implements :class:`HttpClientProto` on top of one shared
:class:`RadioLink` to the ESP32 + CC1101 radio gateway. Reads return the same
canonical settings dict the cloud codecs produce; writes are acknowledged and
verified radio exchanges. Cloud-only features raise :class:`RadioUnsupportedError`
or return "no data". See ``docs/radio_backend.md``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime
import logging
import time
from typing import Any

from homeassistant.util import dt as dt_util

from custom_components.termoweb.codecs.radio_codec import (
    prog_from_program,
    settings_from_power_record,
    settings_from_status,
)
from custom_components.termoweb.domain.commands import SetLock
from custom_components.termoweb.inventory import (
    NodeDescriptor,
    normalize_node_addr,
    normalize_node_type,
)
from custom_components.termoweb.planner.radio_planner import (
    needs_status,
    plan_commands,
    plan_settings,
    validate_commands,
)

from .radio import protocol
from .radio.dialect import DIALECTS, Dialect, Frame, build_frame
from .radio.link import DEFAULT_PORT, GatewayInfo, RadioLink, RadioLinkError
from .radio_power import EnergyEstimator, PowerManager

_LOGGER = logging.getLogger(__name__)

REPLY_TIMEOUT_S = 1.0  # wait for a processed reply after the heater's ack
PROGRAM_REFRESH_S = 3600.0  # re-read a heater's weekly program at most hourly
STATUS_REPLY_LENS = (
    protocol.STATUS_SHORT_LEN,
    protocol.STATUS_E6_LEN,
    protocol.STATUS_E4_LEN,
)
# Mode to give back to a heater the power manager switched off (override -> program).
RESTORE_MODES: dict[int, str] = {
    protocol.MODE_AUTO: "auto",
    protocol.MODE_MANUAL: "manual",
    protocol.MODE_OVERRIDE: "auto",
}
POWER_RECORD_REPLY_LENS = (protocol.POWER_REQUEST_B_LEN,)  # dialect-B BD record
PROGRAM_REPLY_LENS = (
    protocol.PROGRAM_HOURLY_PAYLOAD_LEN,
    protocol.PROGRAM_HALF_HOURLY_PAYLOAD_LEN,
)
VERDICT_LEN = 2  # ``<opcode+1> 55|56``
CLOCK_REPLY_REQUEST = protocol.OP_CLOCK_STEADY  # both 51 and 52 are answered ``53``
GATEWAY_NAME = "Radio gateway"
GATEWAY_MODEL = "ESP32 + CC1101 radio gateway"

LinkFactory = Callable[..., RadioLink]


class RadioError(Exception):
    """Base error for radio backend failures."""


class RadioCommandError(RadioError):
    """Raised when a heater does not ack, answer or accept a radio command."""


class RadioUnsupportedError(RadioError):
    """Raised for cloud features the radio gateway cannot provide."""

    def __init__(self, feature: str) -> None:
        """Build the standard "not supported over the radio gateway" message."""

        super().__init__(f"{feature} is not supported over the radio gateway")
        self.feature = feature


def _local_now() -> datetime:
    """Return Home Assistant's local time; the radio station is the clock master."""

    return dt_util.now()


def radio_addr(addr: Any) -> int:
    """Return a node address as a radio id 1-254 (decimal string in the inventory)."""

    try:
        value = int(str(addr).strip(), 10)
    except (TypeError, ValueError) as err:
        raise ValueError(f"invalid radio node address {addr!r}") from err
    if not 0 < value < 0xFF:
        raise ValueError(f"radio node address {addr!r} outside 1-254")
    return value


def dev_id_from_mac(mac: str | None) -> str | None:
    """Return a MAC as lowercase hex without separators, or None."""

    if not mac:
        return None
    digits = "".join(ch for ch in mac.lower() if ch in "0123456789abcdef")
    return digits if len(digits) == 12 else None


class RadioClient:
    """Serve the HttpClientProto methods over the ESP32 radio gateway."""

    reply_timeout: float = REPLY_TIMEOUT_S
    program_refresh_s: float = PROGRAM_REFRESH_S

    def __init__(
        self,
        host: str,
        port: int,
        dialect_name: str,
        nodes: Iterable[Mapping[str, Any]],
        *,
        network_id: bytes | None,
        station_id: int = 1,
        link_factory: LinkFactory = RadioLink,
        clock: Callable[[], float] = time.monotonic,
        power: PowerManager | None = None,
    ) -> None:
        """Store the gateway address, dialect and stored node list; nothing connects."""

        dialect = DIALECTS.get(str(dialect_name).strip().upper())
        if dialect is None:
            raise ValueError(f"unknown radio dialect {dialect_name!r}")
        if network_id is None and dialect.network_id is None:
            raise ValueError(f"dialect {dialect.name} needs an explicit network_id")
        self._network_id = network_id
        self._host = host
        self._port = int(port) if port else DEFAULT_PORT
        self._dialect = dialect
        self._station_id = station_id
        self._link_factory = link_factory
        self._clock = clock
        self._nodes = [dict(node) for node in nodes if isinstance(node, Mapping)]
        self._link: RadioLink | None = None
        self._connect_lock = asyncio.Lock()
        self._exchange_lock = asyncio.Lock()
        self._disconnect_callbacks: list[Callable[[], None]] = []
        self._programs: dict[int, tuple[float, list[int] | None]] = {}
        self._locks: dict[int, bool] = {}  # last lock written, for records without it
        self.power = power or PowerManager()
        self.energy = EnergyEstimator(self.power.rated_power)

    # --- connection ----------------------------------------------------------

    @property
    def dialect(self) -> Dialect:
        """Return the radio dialect this client frames with."""

        return self._dialect

    @property
    def link(self) -> RadioLink | None:
        """Return the shared link, or None before the first use."""

        return self._link

    @property
    def connected(self) -> bool:
        """Return True while the gateway connection is open."""

        return self._link is not None and self._link.connected

    @property
    def gateway_info(self) -> GatewayInfo | None:
        """Return the gateway's last ``Q`` status, if connected once."""

        return None if self._link is None else self._link.gateway_info

    async def async_connect(self) -> RadioLink:
        """Return the shared link, opening the gateway connection if needed."""

        async with self._connect_lock:
            link = self._link
            if link is None:
                link = self._link_factory(
                    self._host,
                    self._port,
                    self._dialect,
                    station_id=self._station_id,
                    network_id=self._network_id,
                    on_disconnect=self._handle_disconnect,
                )
                self._link = link
            if not link.connected:
                await link.connect()
            return link

    async def async_close(self) -> None:
        """Close the gateway connection if one is open."""

        if self._link is not None:
            await self._link.close()

    def add_disconnect_listener(
        self, callback: Callable[[], None]
    ) -> Callable[[], None]:
        """Call ``callback`` when an established gateway connection drops."""

        self._disconnect_callbacks.append(callback)

        def _remove() -> None:
            """Unregister the callback; safe to call more than once."""

            if callback in self._disconnect_callbacks:
                self._disconnect_callbacks.remove(callback)

        return _remove

    def _handle_disconnect(self) -> None:
        """Fan a link disconnect out to every registered listener."""

        _LOGGER.info("Radio gateway %s:%s disconnected", self._host, self._port)
        for callback in list(self._disconnect_callbacks):
            try:
                callback()
            except Exception:
                _LOGGER.exception("Radio disconnect listener failed")

    # --- radio exchanges -----------------------------------------------------

    async def _exchange(
        self,
        addr: int,
        payload: bytes,
        reply_lens: tuple[int, ...],
        *,
        request_opcode: int | None = None,
    ) -> Frame:
        """Send ``payload`` to ``addr`` and return its reply; raise on no ack or reply."""

        link = await self.async_connect()
        opcode = payload[0] if request_opcode is None else request_opcode
        matches = protocol.reply_predicate(opcode, *reply_lens)
        async with self._exchange_lock:
            future: asyncio.Future[Frame] = asyncio.get_running_loop().create_future()

            def _on_frame(received: Any) -> None:
                """Resolve the reply future with the first matching heater frame."""

                frame = received.frame
                if (
                    not future.done()
                    and frame.ok
                    and not frame.is_ack
                    and frame.src == addr
                    and frame.dst == link.station_id
                    and matches(frame)
                ):
                    future.set_result(frame)

            remove = link.add_listener(_on_frame)
            try:
                air = build_frame(
                    link.dialect,
                    link.station_id,
                    addr,
                    payload,
                    network_id=link.network_id,
                )
                _LOGGER.debug("Radio -> %02X: %s", addr, payload.hex(" ").upper())
                result = await link.send_frame(addr, air)
                if not result.ok:
                    raise RadioCommandError(
                        f"heater {addr} did not acknowledge command {opcode:02X}"
                    )
                try:
                    reply = await asyncio.wait_for(future, self.reply_timeout)
                except TimeoutError as err:
                    raise RadioCommandError(
                        f"heater {addr} acknowledged command {opcode:02X} "
                        "but sent no reply"
                    ) from err
            finally:
                remove()
                if not future.done():
                    future.cancel()
        _LOGGER.debug("Radio <- %02X: %s", addr, reply.payload.hex(" ").upper())
        return reply

    async def _write(
        self, addr: int, payload: bytes, *, request_opcode: int | None = None
    ) -> None:
        """Send a write and require the heater's ``<opcode+1> 55`` verdict."""

        opcode = payload[0] if request_opcode is None else request_opcode
        reply = await self._exchange(
            addr, payload, (VERDICT_LEN,), request_opcode=opcode
        )
        verdict = protocol.reply_verdict(reply.payload, opcode)
        if verdict is not True:
            outcome = "rejected" if verdict is False else "gave an unknown answer to"
            raise RadioCommandError(
                f"heater {addr} {outcome} command {payload.hex(' ').upper()}"
            )

    async def _command(self, addr: int, payload: bytes) -> None:
        """Send a write: ack-only in this dialect, else require ``<opcode+1> 55``."""

        if payload[0] in self._dialect.ack_only_opcodes:
            await self.async_send(addr, payload)
        else:
            await self._write(addr, payload)

    async def async_send(self, addr: int, payload: bytes) -> None:
        """Send a payload that has no reply (e.g. ``BF 01``, ``57 55``); require its ack."""

        link = await self.async_connect()
        air = build_frame(
            link.dialect, link.station_id, addr, payload, network_id=link.network_id
        )
        _LOGGER.debug("Radio -> %02X: %s", addr, payload.hex(" ").upper())
        result = await link.send_frame(addr, air)
        if not result.ok:
            raise RadioCommandError(
                f"heater {addr} did not acknowledge {payload.hex(' ').upper()}"
            )

    async def async_sync_clock(self, addr: int, *, registering: bool) -> None:
        """Send the EB clock sync (``51`` while registering, else ``52``); require ``53 55``."""

        payload = protocol.sync_clock(_local_now(), registering, self._dialect)
        await self._write(addr, payload, request_opcode=CLOCK_REPLY_REQUEST)

    def note_max_power(self, addr: int, watts: float) -> None:
        """Remember a heater's measured full-load power from its power requests."""

        self.power.note_reported_power(addr, watts)

    async def async_balance_power(self) -> None:
        """Switch heaters off or back on as the power manager plans."""

        to_off, to_restore = self.power.plan()
        for addr in to_off:
            try:
                status = await self._read_status(addr)
                if status.mode_code == protocol.MODE_OFF:
                    continue
                await self._write_settings(addr, mode="off")
            except (RadioLinkError, RadioCommandError) as err:
                _LOGGER.error(
                    "Power limit: could not switch heater %s off: %s", addr, err
                )
                continue
            self.power.mark_shed(addr, status.mode_code)
        for addr, mode_code in to_restore:
            mode = RESTORE_MODES.get(mode_code)
            try:
                if mode is not None:
                    await self._write_settings(addr, mode=mode)
            except (RadioLinkError, RadioCommandError) as err:
                _LOGGER.error("Power limit: could not restore heater %s: %s", addr, err)
                continue
            _LOGGER.info("Power limit: heater %s back to %s", addr, mode)
            self.power.clear_shed(addr)

    # --- node helpers --------------------------------------------------------

    @staticmethod
    def _resolve_node(node: NodeDescriptor) -> tuple[str, int]:
        """Return ``(node_type, radio id)`` for a node descriptor."""

        if isinstance(node, tuple) and len(node) == 2:
            node_type, addr = node
        else:
            node_type = getattr(node, "type", None)
            addr = getattr(node, "addr", None)
        normalized_type = normalize_node_type(node_type, use_default_when_falsey=True)
        normalized_addr = normalize_node_addr(addr, use_default_when_falsey=True)
        if not normalized_type or not normalized_addr:
            raise ValueError(f"Invalid node descriptor: {node!r}")
        return normalized_type, radio_addr(normalized_addr)

    def _node_type_for(self, addr: int) -> str | None:
        """Return the stored node type for radio id ``addr``."""

        for node in self._nodes:
            try:
                if radio_addr(node.get("addr")) == addr:
                    return normalize_node_type(node.get("type")) or None
            except ValueError:
                continue
        return None

    async def _read_status(self, addr: int) -> protocol.StatusRecord:
        """Return the heater's decoded ``B8`` status; raise on no ack or reply."""

        reply = await self._exchange(addr, protocol.request_status(), STATUS_REPLY_LENS)
        record = protocol.decode_status(reply.payload)
        assert record is not None  # the reply predicate only accepts B9 records
        return record

    async def read_power_record(self, addr: int) -> protocol.PowerRecord | None:
        """Return the heater's dialect-B power record (``BC`` → ``BD``), or None."""

        try:
            reply = await self._exchange(
                addr, protocol.request_energy(), POWER_RECORD_REPLY_LENS
            )
        except (RadioLinkError, RadioCommandError) as err:
            _LOGGER.debug("Power record read from heater %s failed: %s", addr, err)
            return None
        record = protocol.decode_power_record(reply.payload)
        if record is not None:
            self.power.note_heating(addr, record.heating)
            self.energy.observe(addr, record.heating, record.duty_pct, self._clock())
        return record

    async def _program(self, addr: int) -> list[int] | None:
        """Return the heater's ``prog``, re-reading it after ``program_refresh_s``."""

        cached = self._programs.get(addr)
        now = self._clock()
        if cached is not None and now - cached[0] < self.program_refresh_s:
            return cached[1]
        try:
            reply = await self._exchange(
                addr, protocol.request_program(), PROGRAM_REPLY_LENS
            )
        except (RadioLinkError, RadioCommandError) as err:
            _LOGGER.debug("Program read from heater %s failed: %s", addr, err)
            return None if cached is None else cached[1]
        record = protocol.decode_program(reply.payload)
        assert record is not None  # the reply predicate only accepts B1 records
        prog = prog_from_program(record)
        if prog is None:
            _LOGGER.debug("Heater %s program has no hourly form; prog omitted", addr)
        self._programs[addr] = (now, prog)
        return prog

    # --- HttpClientProto -----------------------------------------------------

    async def list_devices(self) -> list[dict[str, Any]]:
        """Return the radio gateway as the single device, keyed by its MAC."""

        info = (await self.async_connect()).gateway_info
        dev_id = dev_id_from_mac(None if info is None else info.mac)
        if info is None or dev_id is None:
            raise RadioError("the radio gateway did not report its MAC address")
        return [
            {
                "dev_id": dev_id,
                "name": GATEWAY_NAME,
                "model": f"{GATEWAY_MODEL} (dialect {self._dialect.name})",
                "serial_id": dev_id,
                "fw_version": info.version,
            }
        ]

    async def get_nodes(self, dev_id: str) -> dict[str, list[dict[str, Any]]]:
        """Return the stored heater list in the cloud ``{"nodes": [...]}`` shape."""

        return {"nodes": [dict(node) for node in self._nodes]}

    async def get_node_settings(
        self, dev_id: str, node: NodeDescriptor
    ) -> dict[str, Any] | None:
        """Return canonical settings from B8 status (+ B0 program), or None if silent."""

        _node_type, addr = self._resolve_node(node)
        try:
            record = await self._read_status(addr)
        except (RadioLinkError, RadioCommandError) as err:
            _LOGGER.debug("Status read from heater %s failed: %s", addr, err)
            return None
        settings = settings_from_status(record)
        prog = await self._program(addr)
        if prog is not None:
            settings["prog"] = prog
        if "state" not in settings:
            power = await self.read_power_record(addr)
            if power is not None:
                settings.update(settings_from_power_record(power))
        if "lock" not in settings and addr in self._locks:
            settings["lock"] = self._locks[addr]
        if "state" in settings:
            self.power.note_heating(addr, settings["state"] == "on")
        rated = self.power.rated_power(addr)
        if "max_power" not in settings and rated is not None:
            settings["max_power"] = rated
        settings["priority"] = self.power.priority(addr)
        return settings

    async def set_node_settings(
        self,
        dev_id: str,
        node: NodeDescriptor,
        *,
        mode: str | None = None,
        stemp: float | None = None,
        prog: list[int] | None = None,
        ptemp: list[float] | None = None,
        units: str = "C",
        boost_time: int | None = None,
        cancel_boost: bool = False,
    ) -> None:
        """Write presets, program and mode/setpoint, each verified by the heater."""

        if boost_time is not None or cancel_boost:
            raise RadioUnsupportedError("Boost through a settings write")
        _node_type, addr = self._resolve_node(node)
        await self._write_settings(
            addr, mode=mode, stemp=stemp, prog=prog, ptemp=ptemp, units=units
        )
        if mode is not None or stemp is not None:
            self.power.clear_shed(addr)  # the user's choice wins over a shed
        _LOGGER.info("Radio settings written to heater %s", addr)

    async def _write_settings(
        self,
        addr: int,
        *,
        mode: str | None = None,
        stemp: float | None = None,
        prog: list[int] | None = None,
        ptemp: list[float] | None = None,
        units: str = "C",
    ) -> None:
        """Plan, validate and send a settings write, each frame verified."""

        commands = plan_settings(
            mode=mode, stemp=stemp, prog=prog, ptemp=ptemp, units=units
        )
        status = None
        if needs_status(commands, self._dialect):
            validate_commands(commands, self._dialect)  # before anything goes on air
            status = await self._read_status(addr)
        for planned in plan_commands(commands, self._dialect, status):
            await self._write(addr, planned.payload)
            if planned.opcode == protocol.OP_PROGRAM_WRITE:
                self._programs.pop(addr, None)

    async def set_acm_boost_state(
        self,
        dev_id: str,
        addr: str | int,
        *,
        boost: bool,
        boost_time: int | None = None,
        stemp: float | None = None,
        units: str | None = None,
    ) -> None:
        """Toggle a heater's boost (``D2``); accumulator boost is not known on radio."""

        radio_id = radio_addr(addr)
        if self._node_type_for(radio_id) != "htr":
            raise RadioUnsupportedError("Accumulator boost")
        if boost_time is not None or stemp is not None:
            _LOGGER.debug(
                "Heater %s boost uses its own duration and temperature", radio_id
            )
        await self._write(radio_id, protocol.set_toggle(protocol.TOGGLE_BOOST, boost))

    async def set_node_lock(
        self, dev_id: str, node: NodeDescriptor, *, lock: bool
    ) -> None:
        """Toggle the heater's keypad lock (``BA``) and remember what was written."""

        _node_type, addr = self._resolve_node(node)
        (planned,) = plan_commands([SetLock(lock)])
        await self._command(addr, planned.payload)
        self._locks[addr] = lock

    async def set_node_display_select(
        self, dev_id: str, node: NodeDescriptor, *, select: bool
    ) -> None:
        """Flash the heater's display (``5E 01``); there is nothing to deselect."""

        if not select:
            return
        _node_type, addr = self._resolve_node(node)
        await self._command(addr, protocol.flash_display())

    async def set_node_priority(
        self, dev_id: str, node: NodeDescriptor, *, priority: int
    ) -> None:
        """Store a heater's priority for the local power manager."""

        _node_type, addr = self._resolve_node(node)
        self.power.set_priority(addr, priority)

    async def get_power_limit(self, dev_id: str) -> int | None:
        """Return the local power manager's installation limit in W (0 = no limit)."""

        return self.power.power_limit or 0

    async def set_power_limit(self, dev_id: str, *, power_limit: int) -> None:
        """Set the local power manager's installation limit (W); 0 removes it."""

        self.power.set_power_limit(int(power_limit))

    async def set_acm_extra_options(
        self,
        dev_id: str,
        addr: str | int,
        *,
        boost_time: int | None = None,
        boost_temp: float | None = None,
    ) -> None:
        """Raise: accumulator boost defaults are not known on radio."""

        raise RadioUnsupportedError("Accumulator boost defaults")

    async def get_node_samples(
        self,
        dev_id: str,
        node: NodeDescriptor,
        start: float,
        stop: float,
    ) -> list[dict[str, float]]:
        """Return the latest estimated Wh counter as one sample (no history)."""

        _node_type, addr = self._resolve_node(node)
        counter = self.energy.counter_wh(addr)
        if counter is None:
            return []
        return [{"t": time.time(), "counter": round(counter, 3)}]

    async def get_geo_data(self, dev_id: str) -> None:
        """Return None: the radio gateway has no account location."""

    async def get_rtc_time(self, dev_id: str) -> dict[str, int]:
        """Return local time in the cloud RTC shape; the station is the clock master."""

        now = _local_now()
        return {
            "y": now.year,
            "n": now.month,
            "d": now.day,
            "h": now.hour,
            "m": now.minute,
            "s": now.second,
        }


__all__ = [
    "RadioClient",
    "RadioCommandError",
    "RadioError",
    "RadioUnsupportedError",
    "dev_id_from_mac",
    "radio_addr",
]
