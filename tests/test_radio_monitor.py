"""Tests for listen-only radio entries: the no-transmit guarantee and the monitor."""

from __future__ import annotations

import asyncio
import dataclasses
from datetime import datetime
import importlib
import logging
from types import SimpleNamespace
from typing import Any

import pytest
from conftest import build_entry_runtime
from tests_ha.fakes.radio_link import (
    HEATER,
    POWER_REQUEST,
    REGISTRATION,
    STATUS_SHORT,
    FakeRadioLink,
    gateway_info,
    received,
    received_ack,
)

from custom_components.termoweb.backend.radio import protocol as p
from custom_components.termoweb.const import (
    BRAND_RADIO_MONITOR,
    get_brand_label,
    signal_radio_frames,
)
from custom_components.termoweb.inventory import Inventory

DEV_ID = "aabbcc001122"
PLACEHOLDER_NET = b"\x00\x00"


def _mod(name: str):
    """Import a backend module at test time (other suites reload these modules)."""

    return importlib.import_module(f"custom_components.termoweb.backend{name}")


def make_client(*, listen_only: bool = True, dialect: str = "A", info=None):
    """Return (client, links) over fake links; listen-only by default."""

    links: list[FakeRadioLink] = []

    def factory(host, port, link_dialect, **kwargs):
        link = FakeRadioLink(host, port, link_dialect, **kwargs)
        link.reply(0xB8, STATUS_SHORT)
        if info is not None:
            link.info = info
        links.append(link)
        return link

    client = _mod(".radio_client").RadioClient(
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
        self.calls: list[float] = []
        self.pending: list[asyncio.Future[None]] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        future = asyncio.get_running_loop().create_future()
        self.pending.append(future)
        await future

    def release(self) -> None:
        pending, self.pending = self.pending, []
        for future in pending:
            if not future.done():
                future.set_result(None)


# --- the no-transmit guarantee ---------------------------------------------------


@pytest.mark.asyncio
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
        with pytest.raises(_mod(".radio.link").TransmitBlockedError, match="listen"):
            await attempt
    assert await client.get_node_settings(DEV_ID, node) is None
    assert await client.read_power_record(HEATER) is None
    await client.async_balance_power()  # no heaters to shed: no-op
    assert link.sent == []

    (device,) = await client.list_devices()
    assert device["model"].endswith("(listen only)")
    assert device["name"] == "Radio monitor"


@pytest.mark.asyncio
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


def test_factory_builds_listen_only_clients() -> None:
    """create_radio_client(listen_only=True) uses the silent station id 0xFE."""

    factory = _mod(".factory")
    radio_client = _mod(".radio_client")
    tcp = factory.create_radio_client(
        "10.0.0.5", 2323, "A", [], PLACEHOLDER_NET, listen_only=True
    )
    stick = factory.create_radio_client(
        "/dev/ttyUSB0",
        0,
        "A",
        [],
        PLACEHOLDER_NET,
        serial_url="/dev/ttyUSB0",
        device_id="nanocul-x",
        listen_only=True,
    )
    normal = factory.create_radio_client("10.0.0.5", 2323, "A", [], None)
    for client in (tcp, stick):
        assert client.listen_only
        assert client._station_id == radio_client.LISTEN_ONLY_STATION_ID  # noqa: SLF001
    assert not normal.listen_only and normal._station_id == 1  # noqa: SLF001


# --- capture ------------------------------------------------------------------


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_capture_removes_listeners_when_cancelled() -> None:
    """A cancelled capture still detaches from the link."""

    client, links = make_client()

    async def cancelled(_seconds: float) -> None:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await client.async_capture(10, sleep=cancelled)
    assert links[0].listeners == [] and links[0].line_listeners == []


# --- RadioMonitor ----------------------------------------------------------------


class Coordinator:
    """Records gateway connection updates; deltas must never arrive."""

    def __init__(self) -> None:
        self.connections: list[dict[str, Any]] = []

    def handle_ws_deltas(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("a listen-only entry pushes no heater state")

    def update_gateway_connection(self, **kwargs: Any) -> None:
        self.connections.append(kwargs)


def build_monitor(info=None):
    """Return (monitor, client, links, sleeper, hass)."""

    client, links = make_client(info=info)
    hass = SimpleNamespace(data={})
    inventory = Inventory(DEV_ID, [])
    build_entry_runtime(
        hass=hass,
        entry_id="entry",
        dev_id=DEV_ID,
        inventory=inventory,
        brand=BRAND_RADIO_MONITOR,
    )
    sleeper = Sleeper()
    monitor = _mod(".radio_monitor").RadioMonitor(
        hass,
        entry_id="entry",
        dev_id=DEV_ID,
        client=client,
        coordinator=Coordinator(),
        inventory=inventory,
        sleep=sleeper,
    )
    return monitor, client, links, sleeper, hass


ESP32_INFO = dataclasses.replace(gateway_info(version="3.7-esp32"), dialect="A")


@pytest.mark.asyncio
async def test_monitor_alternates_dialects_on_esp32_and_never_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each window switches A, B, A...; heater frames are counted, not answered."""

    monitor, _client, links, sleeper, hass = build_monitor(ESP32_INFO)
    published: list[dict[str, Any]] = []

    def send(target: Any, signal: str, payload: dict[str, Any]) -> None:
        assert target is hass and signal == signal_radio_frames("entry")
        published.append(payload)

    monkeypatch.setattr(_mod(".radio_monitor"), "async_dispatcher_send", send)
    monitor.start()
    await settle()
    link = links[0]
    assert link.dialects == ["A"]
    assert sleeper.calls == [_mod(".radio_monitor").MONITOR_WINDOW_S]
    sleeper.release()
    await settle()
    sleeper.release()
    await settle()
    assert link.dialects == ["A", "B", "A"]

    for payload in (REGISTRATION, POWER_REQUEST, bytes([0x56]) + STATUS_SHORT):
        link.deliver(received(HEATER, payload, dst=1))
    link.deliver(received_ack(HEATER))
    await settle()
    assert monitor.frames == 4
    assert isinstance(monitor.last_frame_at, datetime)
    assert published[-1] == {
        "frames": 4,
        "last_frame": monitor.last_frame_at.isoformat(),
    }
    assert monitor._ws_health_tracker().status == "healthy"  # noqa: SLF001
    assert link.sent == []  # no clock sync, no BF 01, no 57 55

    await monitor.stop()
    assert link.closes == 1 and not (
        monitor._task is not None and not monitor._task.done()
    )


@pytest.mark.asyncio
async def test_monitor_on_stock_nanocul_stays_on_dialect_a() -> None:
    """Firmware without runtime dialects is never asked to switch."""

    monitor, _client, links, sleeper, _hass = build_monitor()  # no dialect= field
    await monitor._refresh_all()  # noqa: SLF001 - before connect: no link yet
    monitor.start()
    await settle()
    sleeper.release()
    await settle()
    assert links[0].dialects == []
    assert links[0].dialect.name == "A"
    await monitor.stop()


@pytest.mark.asyncio
async def test_monitor_survives_a_failed_dialect_switch(caplog) -> None:
    """A dialect switch that fails is logged at debug level; listening goes on."""

    monitor, _client, links, sleeper, _hass = build_monitor(ESP32_INFO)
    with caplog.at_level(logging.DEBUG):
        monitor.start()
        await settle()
        links[0].dialect_error = _mod(".radio.link").RadioLinkError("gone")
        sleeper.release()
        await settle()
    assert "could not switch dialect" in caplog.text
    assert links[0].dialects == ["A"]
    assert monitor._task is not None and not monitor._task.done()
    await monitor.stop()


# --- RadioMonitorBackend --------------------------------------------------------------


def test_monitor_backend_wiring() -> None:
    """The monitor brand has no heater features and builds a RadioMonitor."""

    factory = _mod(".factory")
    monitor_mod = _mod(".radio_monitor")
    caps = factory.backend_capabilities(BRAND_RADIO_MONITOR)
    assert caps == _mod(".base").BackendCapabilities(
        frame_monitor=True, local_radio=True, site_device=False, web_portal=False
    )
    assert (
        factory._backend_class(BRAND_RADIO_MONITOR) is monitor_mod.RadioMonitorBackend
    )  # noqa: SLF001

    client, _links = make_client()
    backend = factory.create_backend(brand=BRAND_RADIO_MONITOR, client=client)
    hass = SimpleNamespace(data={})
    ws = backend.create_ws_client(
        hass, "entry", DEV_ID, Coordinator(), inventory=Inventory(DEV_ID, [])
    )
    assert isinstance(ws, monitor_mod.RadioMonitor)
    diagnostics = backend.diagnostics({"radio_type": "nanocul"})
    assert diagnostics["listen_only"] is True
    assert diagnostics["radio_type"] == "nanocul"
    assert (
        asyncio.run(
            backend.fetch_hourly_samples(DEV_ID, [], datetime.now(), datetime.now())
        )
        == {}
    )

    normal, _ = make_client(listen_only=False, dialect="B")
    for wrong in (normal, object()):
        backend = factory.create_backend(brand=BRAND_RADIO_MONITOR, client=wrong)
        with pytest.raises(TypeError, match="listen-only"):
            backend.create_ws_client(hass, "entry", DEV_ID, Coordinator())


def test_monitor_brand_labels() -> None:
    """A listen-only entry shows as Radio and has no cloud portal link."""

    assert get_brand_label(BRAND_RADIO_MONITOR) == "Radio"
    capabilities = _mod(".factory").backend_capabilities(BRAND_RADIO_MONITOR)
    assert not capabilities.web_portal and not capabilities.site_device
