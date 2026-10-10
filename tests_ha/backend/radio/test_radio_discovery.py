"""Tests for learning a heater network from traffic and probing heaters."""

from __future__ import annotations

from tests_ha.fakes.radio_link import build_ack

import pytest
from tests_ha.fakes.radio_gateway import NET, FakeGateway, FakeTime, heater, rx_line

from custom_components.termoweb.backend.radio import discovery as d, protocol as p
from custom_components.termoweb.backend.radio.dialect import (
    DIALECT_A,
    DIALECT_B,
    build_frame,
)
from custom_components.termoweb.backend.radio.link import RadioLink

STATUS = bytes.fromhex("B921252A02")


def registration(dialect, src: int = 6, net: bytes = NET) -> str:
    """Return an RX line for a heater registration frame to station 01."""
    return rx_line(build_frame(dialect, src, 1, b"\x50", network_id=net))


class Rig:
    """One fake gateway per connection, shared fake time, scripted air traffic."""

    def __init__(self, traffic: dict[str, list[str]] | None = None) -> None:
        self.ft = FakeTime()
        self.gateways: list[FakeGateway] = []
        self.traffic = traffic or {}
        self.responder = None

    def link_factory(self, host, port, dialect, **kwargs) -> RadioLink:
        gw = FakeGateway(dialect)
        gw.responder = self.responder
        self.gateways.append(gw)
        return RadioLink(
            host,
            port,
            dialect,
            clock=self.ft.clock,
            sleep=self.ft.sleep,
            open_connection=gw.open_connection,
            **kwargs,
        )

    async def sleep(self, seconds: float) -> None:
        gw = self.gateways[-1]
        for line in self.traffic.pop(gw.dialect.name, []):
            gw.feed(line)
        await self.ft.sleep(seconds)


@pytest.mark.asyncio
async def test_listen_once_learns_network_without_transmitting() -> None:
    """A heard registration yields dialect, network id and sender; auto-ack is off."""
    other = rx_line(build_frame(DIALECT_B, 7, 1, b"\xbe\x00", network_id=NET))
    rig = Rig({"B": [registration(DIALECT_B), other]})
    sighting = await d.listen_once(
        "gw",
        2323,
        DIALECT_B,
        30,
        link_factory=rig.link_factory,
        sleep=rig.sleep,
        clock=rig.ft.clock,
    )
    assert sighting == d.NetworkSighting(DIALECT_B, NET, frozenset({6, 7}))
    gw = rig.gateways[0]
    assert "A0" in gw.commands and gw.transmitted() == []
    assert rig.ft.now < 30  # stopped after the grace period
    assert gw.writer.closed


@pytest.mark.asyncio
async def test_listen_once_ignores_noise_and_picks_busiest_network() -> None:
    """Bad CRCs, acks and our own station are skipped; the busiest network wins."""
    bad = bytearray(build_frame(DIALECT_B, 9, 1, b"\x50", network_id=NET))
    bad[-1] ^= 0xFF
    lines = [
        rx_line(bytes(bad)),
        rx_line(build_ack(DIALECT_B, 6, 1, NET)),
        rx_line(build_frame(DIALECT_B, 1, 6, b"\xb8", network_id=b"\xab\xcd")),
        registration(DIALECT_B, 3, b"\xab\xcd"),
        registration(DIALECT_B, 6),
        registration(DIALECT_B, 7),
    ]
    rig = Rig({"B": lines})
    sighting = await d.listen_once(
        "gw",
        2323,
        DIALECT_B,
        30,
        link_factory=rig.link_factory,
        sleep=rig.sleep,
        clock=rig.ft.clock,
    )
    assert sighting == d.NetworkSighting(DIALECT_B, NET, frozenset({6, 7}))


@pytest.mark.asyncio
async def test_listen_once_returns_none_on_silence() -> None:
    """No valid frame in the window returns None after the full window."""
    rig = Rig()
    assert (
        await d.listen_once(
            "gw",
            2323,
            DIALECT_A,
            3,
            link_factory=rig.link_factory,
            sleep=rig.sleep,
            clock=rig.ft.clock,
        )
        is None
    )
    assert rig.ft.now >= 3


@pytest.mark.asyncio
async def test_discover_network_alternates_dialects() -> None:
    """Dialect B is tried first; dialect A traffic is found on the next window."""
    rig = Rig({"A": [registration(DIALECT_A, 4, DIALECT_A.network_id)]})
    sighting = await d.discover_network(
        "gw",
        window_s=10,
        total_s=60,
        link_factory=rig.link_factory,
        sleep=rig.sleep,
        clock=rig.ft.clock,
    )
    assert sighting == d.NetworkSighting(
        DIALECT_A, DIALECT_A.network_id, frozenset({4})
    )
    assert [gw.dialect.name for gw in rig.gateways] == ["B", "A"]


@pytest.mark.asyncio
async def test_discover_network_gives_up_after_total() -> None:
    """Silence on every dialect returns None once the total time is spent."""
    rig = Rig()
    assert (
        await d.discover_network(
            "gw",
            window_s=10,
            total_s=25,
            link_factory=rig.link_factory,
            sleep=rig.sleep,
            clock=rig.ft.clock,
        )
        is None
    )
    names = [gw.dialect.name for gw in rig.gateways]
    assert names[:2] == ["B", "A"] and rig.ft.now >= 25


@pytest.mark.asyncio
async def test_probe_heaters_keeps_only_answering_addresses() -> None:
    """Heaters answering B8 are returned with their status; others are skipped."""
    rig = Rig()
    rig.responder = heater(DIALECT_B, 6, {p.OP_STATUS: STATUS})
    found = await d.probe_heaters(
        "gw",
        2323,
        DIALECT_B,
        NET,
        [6, 7, 1, 0, 255, 6],
        link_factory=rig.link_factory,
        sleep=rig.sleep,
    )
    assert list(found) == [6]
    assert found[6].comfort_c == 21.0
    gw = rig.gateways[0]
    assert "A1" in gw.commands
    assert {frame[4] for frame in gw.transmitted()} == {6, 7}
    assert gw.writer.closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("version", "expected"),
    [("3.7-esp32", 1), ("3.6-esp32", None)],
)
async def test_survey_network_needs_survey_firmware(version, expected) -> None:
    """A 3.7 gateway returns the bursts it heard; older firmware returns None."""
    rig = Rig()
    made: list[FakeGateway] = []

    def factory(host, port, dialect, **kwargs) -> RadioLink:
        link = rig.link_factory(host, port, dialect, **kwargs)
        gw = rig.gateways[-1]
        gw.q_line = gw.q_line.replace("3.6-esp32", version)
        gw.survey_lines = ["RAWB 1 -90.0", "RAW 1 +104 -104", "RAWE 1", "# survey off"]
        made.append(gw)
        return link

    bursts = await d.survey_network("gw", 2323, 30, link_factory=factory)
    gw = made[0]
    assert gw.dialect is DIALECT_A and "A0" in gw.commands
    if expected is None:
        assert bursts is None
        assert not any(c.startswith("R") for c in gw.commands)
    else:
        assert len(bursts) == expected and bursts[0].rssi_dbm == -90.0
        assert "R30" in gw.commands
    assert gw.writer.closed
