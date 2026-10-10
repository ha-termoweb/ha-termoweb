"""Listen-only radio entries: the no-transmit guarantee, captures and the monitor."""

from __future__ import annotations

import asyncio
import dataclasses
from datetime import datetime
import logging
from typing import Any

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
import pytest

from custom_components.termoweb.backend import radio_monitor
from custom_components.termoweb.backend.base import BackendCapabilities
from custom_components.termoweb.backend.factory import (
    backend_capabilities,
    create_backend,
    create_radio_client,
)
from custom_components.termoweb.backend.radio import protocol as p
from custom_components.termoweb.backend.radio.link import (
    RadioLinkError,
    TransmitBlockedError,
)
from custom_components.termoweb.backend.radio_client import (
    LISTEN_ONLY_STATION_ID,
    RadioClient,
)
from custom_components.termoweb.const import BRAND_RADIO_MONITOR, signal_radio_frames
from custom_components.termoweb.inventory import Inventory
from tests.fakes.radio_link import (
    HEATER,
    POWER_REQUEST,
    REGISTRATION,
    STATUS_SHORT,
    FakeRadioLink,
    gateway_info,
    received,
    received_ack,
)
from tests.fakes.runtime import build_entry_runtime

DEV_ID = "aabbcc001122"
PLACEHOLDER_NET = b"\x00\x00"
ESP32_INFO = dataclasses.replace(gateway_info(version="3.7-esp32"), dialect="A")


def make_client(
    *, listen_only: bool = True, dialect: str = "A", info: Any = None
) -> tuple[RadioClient, list[FakeRadioLink]]:
    """Return (client, links) over fake links; listen-only by default."""
    links: list[FakeRadioLink] = []

    def factory(host: str, port: int, link_dialect: Any, **kwargs: Any) -> Any:
        link = FakeRadioLink(host, port, link_dialect, **kwargs)
        link.reply(0xB8, STATUS_SHORT)
        if info is not None:
            link.info = info
        links.append(link)
        return link

    client = RadioClient(
        "radio.local",
        2323,
        dialect,
        [{"type": "htr", "addr": str(HEATER)}],
        network_id=PLACEHOLDER_NET if listen_only else bytes.fromhex("1234"),
        station_id=0xFE if listen_only else 1,
        link_factory=factory,
        listen_only=listen_only,
    )
    client.reply_timeout = 0.01
    return client, links


async def settle(rounds: int = 30) -> None:
    """Let background tasks run."""
    for _ in range(rounds):
        await asyncio.sleep(0)


class Sleeper:
    """Injected sleep that parks until the test releases it."""

    def __init__(self) -> None:
        """Start with no pending sleeps."""
        self.calls: list[float] = []
        self.pending: list[asyncio.Future[None]] = []

    async def __call__(self, seconds: float) -> None:
        """Park until released."""
        self.calls.append(seconds)
        future = asyncio.get_running_loop().create_future()
        self.pending.append(future)
        await future

    def release(self) -> None:
        """Wake every parked sleep."""
        pending, self.pending = self.pending, []
        for future in pending:
            if not future.done():
                future.set_result(None)


# --- the no-transmit guarantee ---------------------------------------------------


async def test_listen_only_client_refuses_every_transmit() -> None:
    """Every command path raises before a frame is built; nothing is sent."""
    client, links = make_client()
    assert client.listen_only
    await client.async_connect()
    link = links[0]
    assert link.kwargs == {"listen_only": True, "auto_ack": False}
    node = ("htr", str(HEATER))
    attempts = [
        client.set_node_settings(DEV_ID, node, mode="off"),
        client.set_node_settings(DEV_ID, node, ptemp=[7.0, 16.0, 21.0]),
        client.set_node_display_select(DEV_ID, node, select=True),
        client.set_node_lock(DEV_ID, node, lock=True),
        client.set_acm_boost_state(DEV_ID, str(HEATER), boost=True),
        client.async_send(HEATER, p.confirm_report()),
        client.async_sync_clock(HEATER, registering=True),
        client.async_read_identity(HEATER),
        client.async_pair(1.0),
        client.async_restore(HEATER, mode="manual"),
    ]
    for attempt in attempts:
        with pytest.raises(TransmitBlockedError, match="listen"):
            await attempt
    assert await client.get_node_settings(DEV_ID, node) is None
    assert await client.read_power_record(HEATER) is None
    await client.async_balance_power()  # no heaters to shed: no-op
    assert link.sent == []

    (device,) = await client.list_devices()
    assert device["model"].endswith("(listen only)")
    assert device["name"] == "Radio monitor"
    await client.async_close()


async def test_normal_client_link_is_not_listen_only() -> None:
    """A normal client asks for no listen-only link and can transmit."""
    client, links = make_client(listen_only=False, dialect="B")
    assert not client.listen_only
    await client.set_node_display_select(DEV_ID, ("htr", str(HEATER)), select=True)
    assert links[0].kwargs == {}
    assert links[0].payloads() == [p.flash_display()]
    (device,) = await client.list_devices()
    assert device["model"].endswith("(dialect B)")
    assert device["name"] == "Radio gateway"
    await client.async_close()


def test_factory_builds_listen_only_clients() -> None:
    """create_radio_client(listen_only=True) uses the silent station id 0xFE."""
    tcp = create_radio_client(
        "10.0.0.5", 2323, "A", [], PLACEHOLDER_NET, listen_only=True
    )
    stick = create_radio_client(
        "/dev/ttyUSB0",
        0,
        "A",
        [],
        PLACEHOLDER_NET,
        serial_url="/dev/ttyUSB0",
        device_id="nanocul-x",
        listen_only=True,
    )
    normal = create_radio_client("10.0.0.5", 2323, "A", [], None)
    for client in (tcp, stick):
        assert client.listen_only
        assert client._station_id == LISTEN_ONLY_STATION_ID  # noqa: SLF001
    assert not normal.listen_only
    assert normal._station_id == 1  # noqa: SLF001


# --- capture ------------------------------------------------------------------


async def test_capture_records_frames_and_lines_while_commands_wait() -> None:
    """Frames and lines heard during the window are recorded; then listeners go."""
    client, links = make_client(listen_only=False, dialect="B")
    await client.async_connect()
    link = links[0]
    locked: list[bool] = []

    async def window(seconds: float) -> None:
        locked.append(client._exchange_lock.locked())  # noqa: SLF001
        link.deliver(received(HEATER, POWER_REQUEST))
        link.deliver(received_ack(HEATER))
        link.deliver_line("# survey off")
        assert seconds == 30

    capture = await client.async_capture(30, sleep=window)
    assert locked == [True]
    assert [r["kind"] for r in capture.frames] == ["data", "ack"]
    assert capture.frames[0]["op"] == "BE" and capture.frames[0]["dialect"] == "B"
    assert [r["line"] for r in capture.raw] == ["# survey off"]
    assert capture.started is not None and capture.ended >= capture.started
    assert link.line_listeners == []
    assert len(link.listeners) == 0
    link.deliver(received(HEATER, POWER_REQUEST))
    assert len(capture.frames) == 2
    assert link.sent == []
    await client.async_close()


async def test_capture_removes_listeners_when_cancelled() -> None:
    """A cancelled capture still detaches from the link."""
    client, links = make_client()

    async def cancelled(_seconds: float) -> None:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await client.async_capture(10, sleep=cancelled)
    assert links[0].listeners == [] and links[0].line_listeners == []
    await client.async_close()


# --- RadioMonitor ----------------------------------------------------------------


class Coordinator:
    """Records gateway connection updates; deltas must never arrive."""

    def __init__(self) -> None:
        """Start with no connection updates."""
        self.connections: list[dict[str, Any]] = []

    def handle_ws_deltas(self, *_args: Any, **_kwargs: Any) -> None:
        """Fail: a listen-only entry pushes no heater state."""
        raise AssertionError("a listen-only entry pushes no heater state")

    def update_gateway_connection(self, **kwargs: Any) -> None:
        """Record a gateway connection update."""
        self.connections.append(kwargs)


def build_monitor(
    hass: HomeAssistant, info: Any = None
) -> tuple[radio_monitor.RadioMonitor, list[FakeRadioLink], Sleeper]:
    """Return (monitor, links, sleeper) for a listen-only entry on ``hass``."""
    client, links = make_client(info=info)
    inventory = Inventory(DEV_ID, [])
    build_entry_runtime(
        hass=hass,
        entry_id="entry",
        dev_id=DEV_ID,
        inventory=inventory,
        client=client,
        brand=BRAND_RADIO_MONITOR,
    )
    sleeper = Sleeper()
    monitor = radio_monitor.RadioMonitor(
        hass,
        entry_id="entry",
        dev_id=DEV_ID,
        client=client,
        coordinator=Coordinator(),
        inventory=inventory,
        sleep=sleeper,
    )
    return monitor, links, sleeper


async def test_monitor_alternates_dialects_on_esp32_and_never_answers(
    hass: HomeAssistant,
) -> None:
    """Each window switches A, B, A...; heater frames are counted, not answered."""
    monitor, links, sleeper = build_monitor(hass, ESP32_INFO)
    published: list[dict[str, Any]] = []

    @callback
    def _record(payload: dict[str, Any]) -> None:
        # A plain (non-callback) target would run in the executor, out of order.
        published.append(payload)

    unsub = async_dispatcher_connect(hass, signal_radio_frames("entry"), _record)
    monitor.start()
    await settle()
    link = links[0]
    assert link.dialects == ["A"]
    assert sleeper.calls == [radio_monitor.MONITOR_WINDOW_S]
    sleeper.release()
    await settle()
    sleeper.release()
    await settle()
    assert link.dialects == ["A", "B", "A"]

    for payload in (REGISTRATION, POWER_REQUEST, bytes([0x56]) + STATUS_SHORT):
        link.deliver(received(HEATER, payload, dst=1))
    link.deliver(received_ack(HEATER))
    await hass.async_block_till_done()
    assert monitor.frames == 4
    assert isinstance(monitor.last_frame_at, datetime)
    assert published[-1] == {
        "frames": 4,
        "last_frame": monitor.last_frame_at.isoformat(),
    }
    assert link.sent == []  # no clock sync, no BF 01, no 57 55

    await monitor.stop()
    unsub()
    assert link.closes == 1


async def test_monitor_on_stock_nanocul_stays_on_dialect_a(
    hass: HomeAssistant,
) -> None:
    """Firmware without runtime dialects is never asked to switch."""
    monitor, links, sleeper = build_monitor(hass)  # no dialect= field
    monitor.start()
    await settle()
    sleeper.release()
    await settle()
    assert links[0].dialects == []
    assert links[0].dialect.name == "A"
    await monitor.stop()


async def test_monitor_survives_a_failed_dialect_switch(
    hass: HomeAssistant, caplog: pytest.LogCaptureFixture
) -> None:
    """A failed dialect switch is logged at debug level; listening goes on."""
    monitor, links, sleeper = build_monitor(hass, ESP32_INFO)
    with caplog.at_level(logging.DEBUG):
        monitor.start()
        await settle()
        links[0].dialect_error = RadioLinkError("gone")
        sleeper.release()
        await settle()
    assert "could not switch dialect" in caplog.text
    assert links[0].dialects == ["A"]
    sleeper.release()
    await settle()
    assert len(sleeper.calls) == 3  # still listening, window after window
    await monitor.stop()


# --- RadioMonitorBackend --------------------------------------------------------------


async def test_monitor_backend_wiring(hass: HomeAssistant) -> None:
    """The monitor brand has no heater features and builds a RadioMonitor."""
    caps = backend_capabilities(BRAND_RADIO_MONITOR)
    assert caps == BackendCapabilities(
        frame_monitor=True, local_radio=True, site_device=False, web_portal=False
    )

    client, _links = make_client()
    backend = create_backend(brand=BRAND_RADIO_MONITOR, client=client)
    assert isinstance(backend, radio_monitor.RadioMonitorBackend)
    ws = backend.create_ws_client(
        hass, "entry", DEV_ID, Coordinator(), inventory=Inventory(DEV_ID, [])
    )
    assert isinstance(ws, radio_monitor.RadioMonitor)
    diagnostics = backend.diagnostics({"radio_type": "nanocul"})
    assert diagnostics["listen_only"] is True
    assert diagnostics["radio_type"] == "nanocul"
    assert (
        await backend.fetch_hourly_samples(DEV_ID, [], datetime.now(), datetime.now())
        == {}
    )

    normal, _ = make_client(listen_only=False, dialect="B")
    for wrong in (normal, object()):
        backend = create_backend(brand=BRAND_RADIO_MONITOR, client=wrong)
        with pytest.raises(TypeError, match="listen-only"):
            backend.create_ws_client(hass, "entry", DEV_ID, Coordinator())
