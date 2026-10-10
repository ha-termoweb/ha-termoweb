"""Tests for heater pairing: announcements, id assignment and confirmation."""

from __future__ import annotations

from tests_ha.fakes.radio_link import build_ack

import hashlib

import pytest
from tests_ha.fakes.radio_gateway import NET, FakeGateway, FakeTime, rx_line

from custom_components.termoweb.backend.radio import pairing as pr
from custom_components.termoweb.backend.radio.dialect import (
    DIALECT_A,
    DIALECT_B,
    build_frame,
    decode,
)
from custom_components.termoweb.backend.radio.link import (
    RadioLink,
    UnsupportedDialectError,
)

STATUS = bytes.fromhex("B921252A02")
IDENTITY_B = bytes.fromhex("5B55010203040506070809" + b"X123456".hex())
IDENTITY_A = bytes(range(0x10, 0x1C))  # synthetic 12-byte identity
IDENTITY_A_REPLY = bytes.fromhex("5B7701010104") + IDENTITY_A + bytes.fromhex("0A1A")


def announce_b(dst: int = 1) -> str:
    """Return a dialect-B pairing sweep frame from FF on net 0000."""
    return rx_line(
        build_frame(
            DIALECT_B,
            0xFF,
            dst,
            b"\x55",
            tag=0x03,
            path=(0xFF, dst, 1, 1, 1),
            network_id=b"\x00\x00",
        )
    )


def announce_a(identity: bytes = IDENTITY_A, *, src: int = 0xFF, dst: int = 1) -> str:
    """Return a dialect-A identity announcement; ``src`` != FF is a relayed copy."""
    return rx_line(
        build_frame(DIALECT_A, src, dst, b"\x77" + identity, path=(0xFF, dst, 1, 1, 1))
    )


class PairingHeater:
    """Responder for a heater in pairing mode: takes an id, then answers on it."""

    def __init__(
        self,
        dialect=DIALECT_B,
        *,
        replies: dict[int, bytes] | None = None,
        refuse_acks: int = 0,
    ) -> None:
        self.dialect = dialect
        self.node: int | None = None
        self.net: bytes | None = None
        self.assignments: list = []
        self.refuse_acks = refuse_acks
        self.replies = {0xB8: STATUS, 0x5A: IDENTITY_B} if replies is None else replies

    def __call__(self, air: bytes) -> list[str]:
        frame = decode(self.dialect, air)
        if not frame.ok:
            return []
        if frame.dst == 0xFF and frame.tag == pr.ASSIGNMENT_TAG:
            self.assignments.append(frame)
            if self.refuse_acks:
                self.refuse_acks -= 1
                return []
            self.node, self.net = frame.payload[0], frame.network_id
            return [rx_line(build_ack(self.dialect, 0xFF, 1, self.net))]
        if self.node is None or frame.dst != self.node:
            return []
        lines = [rx_line(build_ack(self.dialect, self.node, 1, self.net))]
        reply = self.replies.get(frame.payload[0])
        if reply is not None:
            lines.append(
                rx_line(
                    build_frame(
                        self.dialect,
                        self.node,
                        1,
                        reply,
                        path=(self.node, 1, 1, 1, 1),
                        network_id=self.net,
                    )
                )
            )
        return lines


_OPEN_LINKS: list[RadioLink] = []


@pytest.fixture(autouse=True)
async def _close_links():
    """Close every link a test opened, so no read loop outlives the test."""
    yield
    while _OPEN_LINKS:
        await _OPEN_LINKS.pop().close()


class Air:
    """Fake time plus scheduled air traffic, fed to the gateway while sleeping."""

    def __init__(self, schedule: list[tuple[float, str]] | None = None) -> None:
        self.ft = FakeTime()
        self.schedule = sorted(schedule or [], key=lambda item: item[0])
        self.gateways: list[FakeGateway] = []
        self.responders: dict[str, object] = {}

    def link_factory(self, host, port, dialect, **kwargs) -> RadioLink:
        gw = FakeGateway(dialect)
        gw.responder = self.responders.get(dialect.name)
        self.gateways.append(gw)
        link = RadioLink(
            host,
            port,
            dialect,
            clock=self.ft.clock,
            sleep=self.link_sleep,
            open_connection=gw.open_connection,
            **kwargs,
        )
        _OPEN_LINKS.append(link)
        return link

    async def link_sleep(self, seconds: float) -> None:
        """Link waits: replies are already queued, so time barely moves."""
        await self.ft.sleep(min(seconds, 0.01))

    async def sleep(self, seconds: float) -> None:
        """Pairing polls: feed the air traffic that is due, then advance time."""
        while self.schedule and self.schedule[0][0] <= self.ft.now:
            _when, line = self.schedule.pop(0)
            self.gateways[-1].feed(line)
        await self.ft.sleep(seconds)

    async def link(self, dialect=DIALECT_B, responder=None) -> RadioLink:
        self.responders[dialect.name] = responder
        link = self.link_factory("gw", 2323, dialect, network_id=NET)
        await link.connect()
        return link

    def assignments(self) -> list:
        return [
            frame
            for gw in self.gateways
            for frame in (decode(gw.dialect, air) for air in gw.transmitted())
            if frame.tag == pr.ASSIGNMENT_TAG
        ]

    async def pair(self, link: RadioLink, **kwargs) -> list[pr.PairedHeater]:
        return await pr.pair_heaters(
            link, clock=self.ft.clock, sleep=self.sleep, **kwargs
        )


# One dialect-B sweep pass as heard live: 120-160 ms per destination.
SWEEP_DSTS = (0x18, 0x19, 0x1C, 0x02, 0x03, 0x01, 0x04)
SWEEP_STEP_S = 0.14


def sweeps(
    start: float, count: int, step: float = 1.0, dsts=SWEEP_DSTS, line=announce_b
) -> list:
    """Return ``count`` sweep passes over ``dsts``, ``step`` seconds apart."""
    return [
        (start + i * step + k * SWEEP_STEP_S, line(dst=dst))
        for i in range(count)
        for k, dst in enumerate(dsts)
    ]


# --- pure helpers ------------------------------------------------------------


def test_classify_announcements() -> None:
    b = decode(DIALECT_B, bytes.fromhex(announce_b(4).split()[-1]))
    assert pr.classify_announcement(DIALECT_B, b) == pr.Announcement(
        DIALECT_B, None, relayed=False, dst=4
    )
    direct = decode(DIALECT_A, bytes.fromhex(announce_a().split()[-1]))
    assert pr.classify_announcement(DIALECT_A, direct) == pr.Announcement(
        DIALECT_A, IDENTITY_A, relayed=False, dst=1
    )
    relayed = decode(DIALECT_A, bytes.fromhex(announce_a(src=4).split()[-1]))
    assert pr.classify_announcement(DIALECT_A, relayed).relayed is True


@pytest.mark.parametrize(
    ("dialect", "air"),
    [
        (DIALECT_B, build_ack(DIALECT_B, 0xFF, 1, NET)),
        (DIALECT_B, build_frame(DIALECT_B, 6, 1, b"\x55", tag=3, network_id=NET)),
        (DIALECT_B, build_frame(DIALECT_B, 0xFF, 1, b"\x55", tag=0, network_id=NET)),
        (DIALECT_B, build_frame(DIALECT_B, 0xFF, 1, b"\x56", tag=3, network_id=NET)),
        (DIALECT_B, build_frame(DIALECT_A, 0xFF, 1, b"\x55", tag=3)),  # wrong dialect
        (DIALECT_A, build_frame(DIALECT_A, 0xFF, 1, b"\x77" + bytes(11))),
        (DIALECT_A, build_frame(DIALECT_A, 0xFF, 1, b"\x78" + bytes(12))),
    ],
)
def test_classify_rejects_other_frames(dialect, air) -> None:
    assert pr.classify_announcement(dialect, decode(dialect, air)) is None


def test_free_node_id_and_assignment_frame() -> None:
    assert pr.free_node_id([]) == 2
    assert pr.free_node_id([2, 3, 5]) == 4
    assert pr.free_node_id(range(2, 0x42)) is None
    vector = bytes.fromhex("0F123401FF0001FF0000000406D057")  # radio_protocol.md
    assert pr.build_assignment(DIALECT_B, 6, NET) == vector
    frame = decode(DIALECT_B, pr.build_assignment(DIALECT_B, 6, NET))
    assert frame.ok and frame.network_id == NET
    assert (frame.src, frame.dst, frame.flags, frame.tag) == (1, 0xFF, 0, 4)
    assert frame.path == bytes.fromhex("01FF000000") and frame.payload == b"\x06"
    a_frame = decode(DIALECT_A, pr.build_assignment(DIALECT_A, 9, NET))
    assert a_frame.ok and a_frame.network_id == NET and a_frame.payload == b"\x09"
    for bad in (0, 1, 0xFF):
        with pytest.raises(ValueError, match="cannot assign"):
            pr.build_assignment(DIALECT_B, bad, NET)


def test_site_network_id_is_stable_and_never_reserved(monkeypatch) -> None:
    first = pr.site_network_id("instance:gateway")
    assert first == pr.site_network_id("instance:gateway")
    assert first == hashlib.sha256(b"instance:gateway").digest()[:2]
    assert first != pr.site_network_id("instance:other")

    class Digest:
        def digest(self) -> bytes:
            return b"\x00\x00\xff\xff\x1b\x30\x12\x34" + bytes(24)

    monkeypatch.setattr(pr.hashlib, "sha256", lambda _data: Digest())
    assert pr.site_network_id("x") == b"\x12\x34"


# --- pair_heaters --------------------------------------------------------------


@pytest.mark.asyncio
async def test_dialect_b_heater_is_paired_confirmed_and_identified() -> None:
    air = Air(sweeps(1.0, 3))
    heater = PairingHeater()
    link = await air.link(responder=heater)

    paired = await air.pair(link, window_s=20, existing_ids={2, 3})

    assert [h.node_id for h in paired] == [4]
    assert paired[0].status.raw == STATUS
    assert paired[0].serial == "X123456" and paired[0].identity == IDENTITY_B[2:]
    assert heater.net == NET  # the heater adopted the station's network id
    (assignment,) = air.assignments()  # later sweeps fall inside the settle time
    assert assignment.payload == b"\x04"
    assert air.ft.now >= 20  # no idle stop: the whole window is used


@pytest.mark.asyncio
async def test_wanted_id_repairs_to_the_same_address() -> None:
    air = Air(sweeps(0.5, 2))
    link = await air.link(responder=PairingHeater(replies={0xB8: STATUS}))
    paired = await air.pair(link, window_s=30, existing_ids={6}, wanted_id=6)
    assert [h.node_id for h in paired] == [6]
    assert paired[0].identity is None and paired[0].serial is None
    assert air.ft.now < 2  # one heater wanted: done at once


@pytest.mark.asyncio
async def test_wanted_id_is_validated() -> None:
    link = await Air().link()
    for bad in (0, 1, 255):
        with pytest.raises(ValueError, match="cannot assign"):
            await pr.pair_heaters(link, wanted_id=bad)


@pytest.mark.asyncio
async def test_no_announcement_returns_nothing_and_sends_nothing() -> None:
    air = Air()
    link = await air.link(responder=PairingHeater())
    assert await air.pair(link, window_s=3) == []
    assert air.ft.now >= 3
    assert air.assignments() == []
    assert link._listeners == []  # noqa: SLF001 - the listener is removed


@pytest.mark.asyncio
async def test_unacked_assignment_is_retried_on_the_next_sweep() -> None:
    air = Air(sweeps(0.5, 4, step=2.0))
    heater = PairingHeater(refuse_acks=1)  # the first pass at dst 01 is missed
    link = await air.link(responder=heater)
    paired = await air.pair(link, window_s=20, max_heaters=1)
    assert [h.node_id for h in paired] == [2]
    assert len(heater.assignments) == 2  # one per pass, no blind link retries


@pytest.mark.asyncio
async def test_heater_that_never_answers_is_not_reported() -> None:
    air = Air(sweeps(0.5, 2))
    link = await air.link(responder=PairingHeater(replies={}))
    assert await air.pair(link, window_s=10) == []
    assert len(air.assignments()) == 1  # the second sweep fell in the settle time


@pytest.mark.asyncio
async def test_dialect_a_direct_announcement_keeps_the_identity() -> None:
    lines = [(0.5, announce_a()), (0.55, announce_a()), (0.6, announce_a(src=4))]
    air = Air(lines + [(9.0, announce_a())])  # announces again after pairing
    heater = PairingHeater(DIALECT_A, replies={0xB8: STATUS, 0x5A: IDENTITY_A_REPLY})
    link = await air.link(DIALECT_A, heater)
    paired = await air.pair(link, window_s=15)
    assert [h.node_id for h in paired] == [2]
    assert paired[0].identity == IDENTITY_A and paired[0].serial is None
    assert len(heater.assignments) == 1  # duplicate, relayed and repeat ignored


@pytest.mark.asyncio
async def test_dialect_a_same_identity_gets_the_same_id_again() -> None:
    lines = [(0.5, announce_a()), (6.0, announce_a()), (6.1, announce_a(dst=2))]
    air = Air(lines)
    heater = PairingHeater(DIALECT_A, replies={}, refuse_acks=1)  # first try lost
    link = await air.link(DIALECT_A, heater)
    assert await air.pair(link, window_s=10) == []
    assert [f.payload for f in heater.assignments] == [b"\x02"] * 2


@pytest.mark.asyncio
async def test_only_relayed_announcements_are_never_answered() -> None:
    air = Air([(0.5, announce_a(src=4)), (1.5, announce_a(src=5))])
    link = await air.link(DIALECT_A, PairingHeater(DIALECT_A))
    assert await air.pair(link, window_s=5) == []
    assert air.assignments() == []


@pytest.mark.asyncio
async def test_announcements_of_another_dialect_are_ignored() -> None:
    air = Air([(0.5, announce_a())])
    link = await air.link(DIALECT_B, PairingHeater())
    assert await air.pair(link, window_s=3) == []
    assert air.assignments() == []


@pytest.mark.asyncio
async def test_id_exhaustion_raises_with_the_heaters_paired_so_far() -> None:
    air = Air(sweeps(0.5, 1) + sweeps(10.0, 1))
    link = await air.link(responder=PairingHeater())
    with pytest.raises(pr.NoFreeAddressError) as err:
        await air.pair(link, window_s=30, existing_ids=range(2, 0x41))
    assert [h.node_id for h in err.value.paired] == [0x41]
    assert link._listeners == []  # noqa: SLF001


@pytest.mark.asyncio
async def test_idle_stop_ends_the_run_after_the_last_pairing() -> None:
    first = PairingHeater()
    air = Air(sweeps(1.0, 1) + sweeps(30.0, 1))
    link = await air.link(responder=first)

    def _second_heater(air_bytes: bytes) -> list[str]:
        if first.node is None or decode(DIALECT_B, air_bytes).payload != b"\x03":
            return first(air_bytes)
        first.node = None  # the next sweep is a second heater
        return first(air_bytes)

    air.gateways[-1].responder = _second_heater
    paired = await air.pair(link, window_s=200, total_s=200, idle_stop_s=60)
    assert [h.node_id for h in paired] == [2, 3]
    assert 90 <= air.ft.now < 95  # 60 s after the second pairing


@pytest.mark.asyncio
async def test_announcements_keep_the_window_open() -> None:
    air = Air(sweeps(1.0, 4, step=8.0))  # heard but never acked
    link = await air.link(responder=PairingHeater(refuse_acks=99))
    assert await air.pair(link, window_s=5, total_s=100, idle_stop_s=10) == []
    assert 34 <= air.ft.now < 36  # 10 s after the last sweep at 25 s


# --- pair_new_network --------------------------------------------------------


@pytest.mark.asyncio
async def test_pair_new_network_alternates_until_a_dialect_pairs() -> None:
    air = Air([(20.0, announce_a())])  # heard during the second (dialect A) slice
    air.responders = {
        "B": PairingHeater(),
        "A": PairingHeater(DIALECT_A, replies={0xB8: STATUS}),
    }
    dialect, paired = await pr.pair_new_network(
        "gw",
        2323,
        NET,
        link_factory=air.link_factory,
        clock=air.ft.clock,
        sleep=air.sleep,
    )
    assert dialect is DIALECT_A and [h.node_id for h in paired] == [2]
    assert [gw.dialect.name for gw in air.gateways] == ["B", "A"]
    assert all(f"N{NET.hex().upper()}" in gw.commands for gw in air.gateways)
    assert all(gw.writer.closed for gw in air.gateways)


@pytest.mark.asyncio
async def test_pair_new_network_gives_up_after_total_time() -> None:
    air = Air()
    dialect, paired = await pr.pair_new_network(
        "gw",
        2323,
        NET,
        total_s=40,
        link_factory=air.link_factory,
        clock=air.ft.clock,
        sleep=air.sleep,
    )
    assert (dialect, paired) == (None, [])
    assert [gw.dialect.name for gw in air.gateways] == ["B", "A", "B"]
    assert air.ft.now >= 40


@pytest.mark.asyncio
async def test_pair_new_network_drops_dialects_the_firmware_lacks() -> None:
    air = Air()

    def factory(host, port, dialect, **kwargs):
        link = air.link_factory(host, port, dialect, **kwargs)
        if dialect is DIALECT_B:
            air.gateways[-1].q_line = "# Q termoweb_rx 3.5 sync=2DE5 id=01"
        return link

    dialect, paired = await pr.pair_new_network(
        "gw",
        2323,
        NET,
        total_s=20,
        link_factory=factory,
        clock=air.ft.clock,
        sleep=air.sleep,
    )
    assert (dialect, paired) == (None, [])
    assert [gw.dialect.name for gw in air.gateways] == ["B", "A", "A"]

    with pytest.raises(UnsupportedDialectError):
        await pr.pair_new_network(
            "gw", 2323, NET, dialects=(DIALECT_B,), link_factory=factory
        )


@pytest.mark.asyncio
async def test_dialect_a_airtime_is_one_assignment_a_second() -> None:
    lines = [(0.5, announce_a()), (0.6, announce_a(dst=2)), (1.0, announce_a(dst=3))]
    air = Air([*lines, (1.6, announce_a(dst=4))])
    heater = PairingHeater(DIALECT_A, refuse_acks=99)
    link = await air.link(DIALECT_A, heater)
    assert await air.pair(link, window_s=3) == []
    assert len(heater.assignments) == 2  # 0.5 s and 1.6 s, one transmission each


@pytest.mark.asyncio
async def test_dialect_b_sweeps_to_other_destinations_are_never_answered() -> None:
    """Fair use: frames swept at other ids get no transmission at all."""
    others = tuple(dst for dst in SWEEP_DSTS if dst != 1)
    air = Air(sweeps(0.5, 10, dsts=others))
    link = await air.link(responder=PairingHeater())
    sent_before = len(air.gateways[-1].transmitted())
    assert await air.pair(link, window_s=12) == []
    assert len(air.gateways[-1].transmitted()) == sent_before


@pytest.mark.asyncio
async def test_dialect_b_assignment_follows_the_sweep_frame_to_us() -> None:
    """Only the dst=01 frame of a pass is answered, once, and the heater acks it."""
    air = Air(sweeps(0.5, 1))
    heater = PairingHeater()
    sent_at: list[float] = []

    def _timed(air_bytes: bytes) -> list[str]:
        if decode(DIALECT_B, air_bytes).tag == pr.ASSIGNMENT_TAG:
            sent_at.append(air.ft.now)
        return heater(air_bytes)

    link = await air.link(responder=_timed)
    paired = await air.pair(link, window_s=5, max_heaters=1)
    assert [h.node_id for h in paired] == [2]
    (assignment,) = heater.assignments
    assert assignment.payload == b"\x02" and heater.node == 2
    t_dst01 = 0.5 + SWEEP_DSTS.index(1) * SWEEP_STEP_S
    assert t_dst01 <= sent_at[0] < t_dst01 + SWEEP_STEP_S  # inside our dwell


@pytest.mark.asyncio
async def test_stale_announcements_are_not_answered(monkeypatch) -> None:
    monkeypatch.setattr(pr, "ANSWER_MAX_AGE_S", -1.0)  # every sweep has moved on
    air = Air(sweeps(0.5, 3))
    heater = PairingHeater()
    link = await air.link(responder=heater)
    assert await air.pair(link, window_s=5) == []
    assert heater.assignments == []
