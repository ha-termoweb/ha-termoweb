"""Tests for nanoCUL support: serial transport, dialect skipping, device id."""

from __future__ import annotations

import sys
import types

import pytest
from radio.radio_fakes import NET, FakeGateway, FakeTime

from custom_components.termoweb.backend import factory
from custom_components.termoweb.backend.radio import DIALECT_A, DIALECT_B, discovery
from custom_components.termoweb.backend.radio.link import (
    GatewayInfo,
    RadioLink,
    UnsupportedDialectError,
)
from custom_components.termoweb.backend.radio.serial_link import (
    BAUDRATE,
    serial_device_id,
    serial_opener,
)
from custom_components.termoweb.backend.radio_client import (
    NANOCUL_MODEL,
    RadioClient,
    RadioError,
)


@pytest.mark.asyncio
async def test_serial_opener_passes_url_and_baudrate() -> None:
    """RadioLink's host/port are ignored; the opener opens the serial URL."""
    calls: list[dict] = []

    async def fake_open(**kwargs):
        calls.append(kwargs)
        return "reader", "writer"

    opener = serial_opener("/dev/ttyUSB0", open_serial=fake_open)
    assert await opener("label", 0) == ("reader", "writer")
    assert calls == [{"url": "/dev/ttyUSB0", "baudrate": BAUDRATE}]


@pytest.mark.asyncio
async def test_serial_opener_uses_pyserial_asyncio_fast(monkeypatch) -> None:
    """Without an injected opener the pyserial-asyncio-fast one is imported lazily."""
    calls: list[dict] = []

    async def open_serial_connection(**kwargs):
        calls.append(kwargs)
        return "r", "w"

    module = types.ModuleType("serial_asyncio_fast")
    module.open_serial_connection = open_serial_connection
    monkeypatch.setitem(sys.modules, "serial_asyncio_fast", module)
    assert await serial_opener("socket://gw:2323")("x", 0) == ("r", "w")
    assert calls == [{"url": "socket://gw:2323", "baudrate": 115200}]


def test_serial_device_id() -> None:
    """USB serial number first; otherwise a stable hash of the path."""
    assert serial_device_id("/dev/x", "A1-b2 C3") == "nanocul-a1b2c3"
    hashed = serial_device_id("/dev/ttyUSB0", "--")
    assert hashed.startswith("nanocul-") and len(hashed) == 20
    assert serial_device_id("/dev/ttyUSB0") == hashed
    assert serial_device_id("/dev/ttyUSB1") != hashed


@pytest.mark.asyncio
async def test_discovery_skips_dialects_the_firmware_lacks() -> None:
    """Stock nanoCUL firmware: dialect B windows are dropped, A keeps listening."""
    ft = FakeTime()
    opened: list[str] = []

    def factory_(host, port, dialect, **kwargs):
        gw = FakeGateway(dialect)
        gw.q_line = "# Q termoweb_rx 3.5 freq=869.525 sync=2DE5 autoack=off id=01"
        opened.append(dialect.name)
        return RadioLink(
            host,
            port,
            dialect,
            clock=ft.clock,
            sleep=ft.sleep,
            open_connection=gw.open_connection,
            **kwargs,
        )

    result = await discovery.discover_network(
        "/dev/ttyUSB0",
        0,
        window_s=10,
        total_s=40,
        link_factory=factory_,
        sleep=ft.sleep,
        clock=ft.clock,
    )
    assert result is None
    assert opened[:2] == ["B", "A"] and opened.count("B") == 1

    with pytest.raises(UnsupportedDialectError):
        await discovery.discover_network(
            "/dev/ttyUSB0",
            0,
            dialects=(DIALECT_B,),
            link_factory=factory_,
            sleep=ft.sleep,
            clock=ft.clock,
        )


class InfoLink:
    """Link stand-in for list_devices: connected, with a given Q status."""

    def __init__(self, info: GatewayInfo) -> None:
        self.gateway_info = info
        self.connected = True


@pytest.mark.asyncio
async def test_list_devices_falls_back_to_the_configured_device_id() -> None:
    """Firmware without a MAC uses the id stored at setup; neither is an error."""
    info = GatewayInfo("3.5", "869.525", "2DE5", True, 1, None, "")
    client = RadioClient(
        "/dev/ttyUSB0",
        0,
        "A",
        [],
        network_id=None,
        link_factory=lambda *a, **k: InfoLink(info),
        device_id="nanocul-a1b2c3",
        model=NANOCUL_MODEL,
    )
    (device,) = await client.list_devices()
    assert device["dev_id"] == "nanocul-a1b2c3"
    assert device["model"] == "nanoCUL USB stick (dialect A)"

    bare = RadioClient(
        "/dev/ttyUSB0",
        0,
        "A",
        [],
        network_id=None,
        link_factory=lambda *a, **k: InfoLink(info),
    )
    with pytest.raises(RadioError, match="MAC"):
        await bare.list_devices()


def test_factory_builds_a_serial_client() -> None:
    """A serial URL gets a RadioLink factory with the serial opener."""
    client = factory.create_radio_client(
        "/dev/ttyUSB0",
        0,
        "B",
        [],
        NET,
        serial_url="/dev/ttyUSB0",
        device_id="nanocul-x",
    )
    assert client._device_id == "nanocul-x"  # noqa: SLF001
    assert client._model == NANOCUL_MODEL  # noqa: SLF001
    link_factory = client._link_factory  # noqa: SLF001
    assert link_factory.func.__name__ == "RadioLink"
    assert "open_connection" in link_factory.keywords
    assert client.dialect.name == "B"
    tcp = factory.create_radio_client("gw", 2323, "A", [], None)
    assert tcp._link_factory.__name__ == "RadioLink"  # noqa: SLF001
