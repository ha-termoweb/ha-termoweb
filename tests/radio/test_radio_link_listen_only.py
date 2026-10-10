"""Tests for listen-only RadioLinks, line listeners and runtime dialect switches."""

from __future__ import annotations

import logging

import pytest
from radio_fakes import NET, Q_LINE_NANOCUL, FakeGateway, FakeTime, rx_line

from custom_components.termoweb.backend.radio import protocol as p
from custom_components.termoweb.backend.radio.dialect import (
    DIALECT_A,
    DIALECT_B,
    build_frame,
)
from custom_components.termoweb.backend.radio.link import (
    RadioLink,
    RadioLinkError,
    TransmitBlockedError,
    UnsupportedDialectError,
)


def listen_link(gw: FakeGateway, ft: FakeTime, **kwargs) -> RadioLink:
    """Return a listen-only link wired to the fake gateway."""
    kwargs.setdefault("network_id", b"\x00\x00")
    return RadioLink(
        "radio.local",
        dialect=gw.dialect,
        station_id=0xFE,
        auto_ack=False,
        listen_only=True,
        clock=ft.clock,
        sleep=ft.sleep,
        open_connection=gw.open_connection,
        **kwargs,
    )


def test_listen_only_link_cannot_auto_ack() -> None:
    """Asking for auto-acks on a listen-only link is a programming error."""
    with pytest.raises(ValueError, match="cannot auto-ack"):
        RadioLink("radio.local", dialect=DIALECT_A, listen_only=True)
    link = RadioLink("radio.local", dialect=DIALECT_A, auto_ack=False)
    assert link.listen_only is False


@pytest.mark.asyncio
async def test_listen_only_link_never_writes_a_transmit(caplog) -> None:
    """Connect, query and survey work; every transmit raises before the wire."""
    gw = FakeGateway(DIALECT_A)
    ft = FakeTime()
    link = listen_link(gw, ft)
    await link.connect()
    assert link.listen_only
    assert gw.commands == ["IFE", "A0", "Y0", "N0000", "Q"]

    air = build_frame(DIALECT_A, 0xFE, 6, p.flash_display(), network_id=NET)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(TransmitBlockedError, match="listen-only"):
            await link.send_frame(6, air)
        with pytest.raises(TransmitBlockedError):
            await link.send_frame(6, air, wait_ack=False)
        with pytest.raises(TransmitBlockedError):
            await link.request(6, p.request_status(), lambda frame: True)
        for command in ("A1", "T00", "X", "D", "F869525", "N12", "I1", "Y", ""):
            with pytest.raises(TransmitBlockedError):
                await link._send_command(command)  # noqa: SLF001
    assert "refused command T" in caplog.text
    assert gw.transmitted() == []
    assert not any(c.startswith(("T", "A1")) for c in gw.commands)

    await link.set_network_id(b"\x00\x00")
    await link.survey(1)
    assert gw.commands[-2:] == ["N0000", "R1"]
    assert gw.transmitted() == []
    await link.close()


@pytest.mark.asyncio
async def test_line_listeners_get_every_line_that_is_not_a_frame() -> None:
    """Status, TX and unparseable RX lines reach line listeners; frames do not."""
    gw = FakeGateway(DIALECT_A)
    ft = FakeTime()
    link = listen_link(gw, ft)
    lines: list[str] = []
    frames: list = []
    remove = link.add_line_listener(lines.append)
    link.add_listener(frames.append)

    def broken(_line: str) -> None:
        raise RuntimeError("boom")

    remove_broken = link.add_line_listener(broken)
    await link.connect()
    gw.feed("# uart overrun")
    gw.feed("RX 1 2 3 4 ZZ")
    gw.feed("RAWB 1 -80.0")
    gw.feed(rx_line(build_frame(DIALECT_A, 6, 1, p.request_status())))
    await ft.sleep(0)
    assert "# uart overrun" in lines and "RX 1 2 3 4 ZZ" in lines
    assert lines[0].startswith("# termoweb_rx") and any(
        line.startswith("# Q ") for line in lines
    )
    assert not any(line.startswith("RAWB") for line in lines)
    assert len(frames) == 1

    remove()
    remove()  # idempotent
    remove_broken()
    gw.feed("# later")
    await ft.sleep(0)
    assert "# later" not in lines
    await link.close()


@pytest.mark.asyncio
async def test_set_dialect_switches_esp32_firmware(caplog) -> None:
    """Y switches the dialect used to decode; dialect-A-only firmware refuses B."""
    gw = FakeGateway(DIALECT_B)
    ft = FakeTime()
    link = listen_link(gw, ft)
    with pytest.raises(UnsupportedDialectError):
        await link.set_dialect(DIALECT_B)  # not connected: firmware unknown
    await link.connect()
    await link.set_dialect(DIALECT_A)
    assert gw.commands[-1] == "Y0" and link.dialect is DIALECT_A
    await link.set_dialect(DIALECT_B)
    assert gw.commands[-1] == "Y1" and link.dialect is DIALECT_B
    await link.close()
    with pytest.raises(RadioLinkError, match="not connected"):
        await link.set_dialect(DIALECT_A)

    gw = FakeGateway(DIALECT_A)
    gw.q_line = Q_LINE_NANOCUL
    link = listen_link(gw, FakeTime())
    await link.connect()
    with pytest.raises(UnsupportedDialectError, match="dialect B"):
        await link.set_dialect(DIALECT_B)
    await link.set_dialect(DIALECT_A)  # dialect A is always there
    assert link.dialect is DIALECT_A
    await link.close()
