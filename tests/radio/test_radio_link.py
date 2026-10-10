"""Tests for the asyncio ESP32 radio gateway client."""

from __future__ import annotations

import asyncio
import logging

import pytest
from radio_fakes import (
    BANNER,
    NET,
    Q_LINE_NANOCUL,
    FakeGateway,
    FakeTime,
    heater,
    rx_line,
)

from custom_components.termoweb.backend.radio import link as link_mod, protocol as p
from custom_components.termoweb.backend.radio.dialect import (
    DIALECT_A,
    DIALECT_B,
    build_ack,
    build_frame,
)
from custom_components.termoweb.backend.radio.link import (
    GatewayInfo,
    RadioLink,
    RadioLinkError,
    UnsupportedDialectError,
    parse_query_line,
    parse_rx_line,
    supports_survey,
)

HEATER = 0x06


def make_link(gw: FakeGateway, ft: FakeTime, **kwargs) -> RadioLink:
    """Return a RadioLink wired to the fake gateway and fake time."""
    kwargs.setdefault("dialect", gw.dialect)
    if kwargs["dialect"].network_id is None:
        kwargs.setdefault("network_id", NET)
    return RadioLink(
        "radio.local",
        clock=ft.clock,
        sleep=ft.sleep,
        open_connection=gw.open_connection,
        **kwargs,
    )


async def connected(dialect=DIALECT_B, **kwargs):
    """Return (gateway, time, link) after a successful connect."""
    gw = FakeGateway(dialect)
    ft = FakeTime()
    link = make_link(gw, ft, **kwargs)
    await link.connect()
    return gw, ft, link


# --- parsing helpers -----------------------------------------------------------


def test_parse_rx_line_tolerates_garbage() -> None:
    """RX lines parse to air bytes and metadata; anything else is None."""
    assert parse_rx_line("RX 12 -58.5 7 1 0F1234") == (
        bytes.fromhex("0F1234"),
        -58.5,
        7,
        12,
    )
    assert parse_rx_line("RX x y z 0 08 0102") == (bytes([8]), None, None, None)
    for bad in (
        "",
        "RX",
        "RX 1 2 3 4",
        "TX 1 2 3 4 0F",
        "RX 1 2 3 4 ZZ",
        "RX 1 2 3 4 0F0",
    ):
        assert parse_rx_line(bad) is None


def test_parse_query_line_variants() -> None:
    """The Q line parses with or without mac=, and tolerates odd fields."""
    info = parse_query_line(
        "# Q termoweb_rx 3.6 freq=869.525 pa=C0 sync=2DD4 mode=dynamic "
        "autoack=off id=0A mac=AA:BB:CC:DD:EE:FF"
    )
    assert info.version == "3.6"
    assert info.freq == "869.525"
    assert info.sync == "2DD4"
    assert info.autoack is False
    assert info.station_id == 0x0A
    assert info.mac == "AA:BB:CC:DD:EE:FF"
    odd = parse_query_line("# Q termoweb_rx autoack=maybe id=zz")
    assert odd.version is None and odd.autoack is None and odd.station_id is None
    bare = parse_query_line("# Q other")
    assert bare.version is None and bare.sync is None and bare.mac is None
    assert parse_query_line("# Q termoweb_rx").version is None
    assert parse_query_line("# termoweb_rx 3.5") is None


def test_constructor_validation() -> None:
    """Missing dialect, bad station ids and bad network ids are refused."""
    with pytest.raises(ValueError, match="dialect"):
        RadioLink("h")
    for bad in (0, 255):
        with pytest.raises(ValueError, match="station_id"):
            RadioLink("h", dialect=DIALECT_B, station_id=bad)
    with pytest.raises(ValueError, match="network_id"):
        RadioLink("h", dialect=DIALECT_B, network_id=b"\x01")
    with pytest.raises(ValueError, match="explicit network_id"):
        RadioLink("h", dialect=DIALECT_B)
    link = RadioLink("h", dialect=DIALECT_A)
    assert link.network_id == DIALECT_A.network_id
    assert link.station_id == 1
    assert link.dialect is DIALECT_A
    assert not link.connected


# --- connect ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connect_handshake_dialect_b() -> None:
    """Connect reads the banner, configures the gateway and parses Q."""
    gw, ft, link = await connected()
    assert gw.opened == [("radio.local", 2323)]
    assert gw.commands == ["I01", "A1", "Y1", "N1234", "Q"]
    assert link.connected
    info = link.gateway_info
    assert info.version == "3.6-esp32"
    assert info.dialect == "B"
    assert info.sync == "2DD4"
    assert info.autoack is True
    assert info.station_id == 1
    assert info.mac is None
    # Commands after the first wait out the minimum gap.
    assert ft.sleeps.count(link_mod.MIN_COMMAND_GAP_S) == 4
    await link.close()
    assert not link.connected
    assert gw.writer.closed
    await link.close()  # idempotent


@pytest.mark.asyncio
async def test_connect_dialect_a_custom_ids(caplog) -> None:
    """Dialect A selects firmware mode 0; a sync mismatch is logged, not fatal."""
    gw = FakeGateway(DIALECT_A)
    ft = FakeTime()
    link = make_link(gw, ft, station_id=0x0A, network_id=b"\x12\x34", port=4000)
    with caplog.at_level(logging.ERROR):
        await link.connect()
    assert gw.opened == [("radio.local", 4000)]
    assert gw.commands == ["I0A", "A1", "Y0", "N1234", "Q"]
    assert "needs 2DE5" in caplog.text
    with pytest.raises(RadioLinkError, match="already connected"):
        await link.connect()
    await link.close()


@pytest.mark.asyncio
async def test_connect_q_without_sync() -> None:
    """A Q line without sync= is accepted without a mismatch check."""
    gw = FakeGateway()
    gw.q_line = "# Q termoweb_rx 3.6-esp32 autoack=on id=01 dialect=B"
    link = make_link(gw, FakeTime())
    info = await link.connect()
    assert info.sync is None
    await link.close()


@pytest.mark.asyncio
async def test_stock_nanocul_firmware_is_dialect_a_only() -> None:
    """A Q line without dialect= means firmware with dialect A compiled in."""
    gw = FakeGateway()
    gw.q_line = Q_LINE_NANOCUL
    with pytest.raises(UnsupportedDialectError, match="does not support dialect B"):
        await make_link(gw, FakeTime()).connect()
    assert gw.writer.closed

    gw = FakeGateway(DIALECT_A)
    gw.q_line = Q_LINE_NANOCUL
    link = make_link(gw, FakeTime())
    info = await link.connect()
    assert info.dialect is None and info.mac is None and info.sync == "2DE5"
    await link.close()


@pytest.mark.asyncio
async def test_connect_failures() -> None:
    """Socket errors, a missing banner and a missing Q reply all raise."""
    gw = FakeGateway()
    gw.open_error = OSError("refused")
    with pytest.raises(RadioLinkError, match="cannot connect"):
        await make_link(gw, FakeTime()).connect()

    gw = FakeGateway()
    gw.banner = "hello"
    link = make_link(gw, FakeTime())
    with pytest.raises(RadioLinkError, match="banner"):
        await link.connect()
    assert not link.connected and gw.writer.closed

    gw = FakeGateway()
    gw.q_line = None
    link = make_link(gw, FakeTime())
    with pytest.raises(RadioLinkError, match="Q status"):
        await link.connect()
    assert not link.connected


@pytest.mark.asyncio
async def test_connect_times_out_and_bad_urls_are_link_errors(monkeypatch) -> None:
    """A host that never answers times out; a bad serial URL is a link error."""
    monkeypatch.setattr(link_mod, "OPEN_TIMEOUT_S", 0.01)
    ft = FakeTime()
    never: asyncio.Future = asyncio.get_running_loop().create_future()

    async def hang(host: str, port: int):
        return await never

    link = RadioLink(
        "radio.local",
        dialect=DIALECT_A,
        clock=ft.clock,
        sleep=ft.sleep,
        open_connection=hang,
    )
    with pytest.raises(RadioLinkError, match="timed out"):
        await link.connect()
    assert not link.connected

    async def bad_url(host: str, port: int):
        raise ValueError("invalid URL, protocol 'sockt' not known")

    link = RadioLink("radio.local", dialect=DIALECT_A, open_connection=bad_url)
    with pytest.raises(RadioLinkError, match="cannot connect"):
        await link.connect()


@pytest.mark.asyncio
async def test_reset_banner_mid_session_drops_and_reconnect_restores_dialect(
    caplog,
) -> None:
    """A stick that resets prints a new banner: the link drops and reconnects."""
    gateways = [FakeGateway(), FakeGateway()]
    opened: list[FakeGateway] = []

    async def open_next(host: str, port: int):
        gw = gateways.pop(0)
        opened.append(gw)
        return await gw.open_connection(host, port)

    ft = FakeTime()
    calls: list[int] = []
    link = RadioLink(
        "radio.local",
        dialect=DIALECT_B,
        network_id=NET,
        clock=ft.clock,
        sleep=ft.sleep,
        open_connection=open_next,
        on_disconnect=lambda: calls.append(1),
    )
    await link.connect()
    caplog.set_level(logging.INFO)
    opened[0].feed(BANNER)  # the stick reset: back to dialect A, default network
    for _ in range(20):
        await asyncio.sleep(0)
    assert not link.connected
    assert calls == [1]
    assert "restarted" in caplog.text

    await link.connect()
    assert link.connected
    assert opened[1].commands[:4] == ["I01", "A1", "Y1", "N1234"]
    await link.close()


@pytest.mark.asyncio
async def test_connect_eof_does_not_call_disconnect_callback() -> None:
    """A connection dropped during the handshake raises without on_disconnect."""
    calls = []
    gw = FakeGateway()
    gw.banner = None
    gw.reader.feed_eof()
    link = make_link(gw, FakeTime(), on_disconnect=lambda: calls.append(1))
    with pytest.raises(RadioLinkError, match="connection lost"):
        await link.connect()
    assert calls == []


# --- send_frame ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_frame_with_ack() -> None:
    """A frame acked by the heater returns ok after one attempt."""
    gw, ft, link = await connected()
    gw.responder = heater(DIALECT_B, HEATER)
    air = build_frame(DIALECT_B, 1, HEATER, p.set_setpoint(21.0), network_id=NET)
    result = await link.send_frame(HEATER, air)
    assert result.ok and result.attempts == 1
    assert result.tx_micros == 1100
    assert result.ack.frame.is_ack and result.ack.frame.src == HEATER
    assert result.ack.rssi_dbm == -58.5 and result.ack.lqi == 0
    assert gw.transmitted() == [air]
    await link.close()


@pytest.mark.asyncio
async def test_send_frame_retries_then_gives_up() -> None:
    """With no ack, the frame is sent ``retries`` times, retry_interval apart."""
    gw, ft, link = await connected()
    air = build_frame(DIALECT_B, 1, HEATER, p.request_status(), network_id=NET)
    result = await link.send_frame(HEATER, air, retries=3, retry_interval=0.2)
    assert not result.ok and result.attempts == 3 and result.ack is None
    assert ft.sleeps.count(0.2) == 3
    assert gw.transmitted() == [air] * 3
    single = await link.send_frame(HEATER, air, retries=0)
    assert single.attempts == 1 and not single.ok
    await link.close()


@pytest.mark.asyncio
async def test_send_frame_ack_on_second_attempt_ignores_other_acks() -> None:
    """Acks from other nodes or to other stations do not count."""
    gw, ft, link = await connected()
    calls = []

    def responder(air: bytes) -> list[str]:
        calls.append(air)
        if len(calls) == 1:
            return [
                rx_line(build_ack(DIALECT_B, 0x07, 1, NET)),  # wrong node
                rx_line(build_ack(DIALECT_B, HEATER, 2, NET)),  # to another station
            ]
        return [rx_line(build_ack(DIALECT_B, HEATER, 1, NET))]

    gw.responder = responder
    result = await link.send_frame(
        HEATER, build_frame(DIALECT_B, 1, HEATER, b"\xb8", network_id=NET)
    )
    assert result.ok and result.attempts == 2
    await link.close()


@pytest.mark.asyncio
async def test_frames_from_another_network_are_ignored() -> None:
    """A neighbour's ack or reply with the same ids does not count as ours."""
    gw, ft, link = await connected()
    foreign = bytes.fromhex("5678")  # synthetic neighbouring network id
    gw.responder = heater(DIALECT_B, HEATER, network_id=foreign)
    air = build_frame(DIALECT_B, 1, HEATER, b"\xb8", network_id=NET)
    assert not (await link.send_frame(HEATER, air, retries=1)).ok

    own_ack = rx_line(build_ack(DIALECT_B, HEATER, 1, NET))
    foreign_reply = rx_line(
        build_frame(DIALECT_B, HEATER, 1, b"\xb9\x21\x25\x2a\x02", network_id=foreign)
    )
    gw.responder = lambda air: [own_ack, foreign_reply]
    reply = await link.request(
        HEATER, p.request_status(), p.reply_predicate(p.OP_STATUS)
    )
    assert reply is None
    await link.close()


@pytest.mark.asyncio
async def test_send_frame_without_ack_wait() -> None:
    """wait_ack=False returns as soon as the gateway confirms TX."""
    gw, ft, link = await connected()
    result = await link.send_frame(HEATER, b"\x01\x02", wait_ack=False)
    assert result.ok and result.attempts == 1 and result.ack is None
    await link.close()


@pytest.mark.asyncio
async def test_txerr_bad_hex_is_resent_once() -> None:
    """A garbled T line is resent once after a short pause."""
    gw, ft, link = await connected()
    gw.responder = heater(DIALECT_B, HEATER)
    gw.tx_overrides.append(link_mod.TXERR_BAD_HEX)
    air = build_frame(DIALECT_B, 1, HEATER, b"\xb8", network_id=NET)
    result = await link.send_frame(HEATER, air)
    assert result.ok and result.attempts == 2
    assert link_mod.TXERR_RETRY_WAIT_S in ft.sleeps
    assert len(gw.transmitted()) == 2
    await link.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ([link_mod.TXERR_BAD_HEX, link_mod.TXERR_BAD_HEX], "empty or bad hex"),
        (["TXERR underflow"], "underflow"),
        (["# unrelated"], "no TX confirmation"),
    ],
)
async def test_transmit_errors_raise(overrides, match) -> None:
    """Repeated bad-hex, other TXERRs and a missing TX line raise RadioLinkError."""
    gw, ft, link = await connected()
    gw.tx_overrides.extend(overrides)
    with pytest.raises(RadioLinkError, match=match):
        await link.send_frame(HEATER, b"\x01")
    await link.close()


@pytest.mark.asyncio
async def test_tx_line_with_bad_micros() -> None:
    """A TX line whose micros field is not a number still confirms the send."""
    gw, ft, link = await connected()
    gw.tx_overrides.extend(["TX soon"])
    result = await link.send_frame(HEATER, b"\x01", wait_ack=False)
    assert result.ok and result.tx_micros is None
    await link.close()


@pytest.mark.asyncio
async def test_no_gap_sleep_when_idle_long_enough() -> None:
    """No gap sleep is needed when the last command is old enough."""
    gw, ft, link = await connected()
    ft.now += 1.0
    before = len(ft.sleeps)
    await link.send_frame(HEATER, b"\x01", wait_ack=False)
    assert link_mod.MIN_COMMAND_GAP_S not in ft.sleeps[before:]
    await link.close()


@pytest.mark.asyncio
async def test_concurrent_sends_are_serialised() -> None:
    """Two concurrent sends never interleave their T lines."""
    gw, ft, link = await connected()
    gw.responder = heater(DIALECT_B, HEATER)
    a = build_frame(DIALECT_B, 1, HEATER, b"\xb8", network_id=NET)
    b = build_frame(DIALECT_B, 1, HEATER, b"\xb0", network_id=NET)
    results = await asyncio.gather(
        link.send_frame(HEATER, a), link.send_frame(HEATER, b)
    )
    assert all(r.ok for r in results)
    assert gw.transmitted() == [a, b]
    await link.close()


@pytest.mark.asyncio
async def test_write_errors_and_not_connected() -> None:
    """Writes before connect or on a broken socket raise RadioLinkError."""
    gw = FakeGateway()
    link = make_link(gw, FakeTime())
    with pytest.raises(RadioLinkError, match="not connected"):
        await link.send_frame(HEATER, b"\x01")
    await link.connect()
    gw.writer.write_error = ConnectionResetError("gone")
    with pytest.raises(RadioLinkError, match="write to gateway failed"):
        await link.send_frame(HEATER, b"\x01")
    gw.writer.wait_closed_error = ConnectionResetError("gone")
    await link.close()


# --- request ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_returns_matching_reply() -> None:
    """A status request returns the heater's B9 reply frame."""
    gw, ft, link = await connected()
    gw.responder = heater(DIALECT_B, HEATER, {p.OP_STATUS: bytes.fromhex("B921252A02")})
    frame = await link.request(
        HEATER, p.request_status(), p.reply_predicate(p.OP_STATUS)
    )
    assert frame is not None and frame.src == HEATER
    assert p.decode_status(frame.payload).comfort_c == 21.0
    assert gw.transmitted() == [
        build_frame(DIALECT_B, 1, HEATER, b"\xb8", network_id=NET)
    ]
    await link.close()


@pytest.mark.asyncio
async def test_request_reply_before_ack_and_filtering() -> None:
    """Replies are matched even before the ack; others are filtered out."""
    gw, ft, link = await connected()
    reply = build_frame(
        DIALECT_B, HEATER, 1, bytes.fromhex("B155") + bytes(41), network_id=NET
    )
    gw.responder = lambda air: [
        rx_line(
            build_frame(DIALECT_B, 0x07, 1, b"\xb1" + bytes(42), network_id=NET)
        ),  # other node
        rx_line(
            build_frame(DIALECT_B, HEATER, 1, b"\xb9\x01", network_id=NET)
        ),  # wrong opcode
        rx_line(reply),
        rx_line(build_ack(DIALECT_B, HEATER, 1, NET)),
    ]
    frame = await link.request(
        HEATER, p.request_program(), p.reply_predicate(p.OP_PROGRAM_READ, 43)
    )
    assert frame.air == reply
    await link.close()


@pytest.mark.asyncio
async def test_request_without_ack_or_reply() -> None:
    """No ack gives None without waiting for a reply; no reply times out to None."""
    gw, ft, link = await connected()
    assert await link.request(HEATER, b"\xb8", lambda f: True, retries=1) is None
    gw.responder = heater(DIALECT_B, HEATER)
    assert await link.request(HEATER, b"\xb8", lambda f: True, timeout=0.5) is None
    assert 0.5 in ft.sleeps
    await link.close()


# --- listeners and inbound lines ------------------------------------------------


@pytest.mark.asyncio
async def test_listener_receives_unsolicited_frames(caplog) -> None:
    """Every RX frame reaches listeners; a failing listener is logged."""
    gw, ft, link = await connected()
    got, other = [], []

    def broken(_rx) -> None:
        raise RuntimeError("boom")

    remove = link.add_listener(got.append)
    link.add_listener(broken)
    link.add_listener(other.append)
    registration = bytes.fromhex("0F1234060100060101010100507222")
    with caplog.at_level(logging.ERROR):
        gw.feed(rx_line(registration, micros=777, rssi="-61.0", lqi="12"))
        gw.feed("RX 1 2 3 1 ZZ")  # bad hex: dropped
        gw.feed(rx_line(b"\x00\x01"))  # garbage frame: delivered, not ok
        await ft.sleep(0)
    assert "listener failed" in caplog.text
    assert len(got) == 2 and len(other) == 2
    rx = got[0]
    assert rx.frame.ok and rx.micros == 777 and rx.rssi_dbm == -61.0 and rx.lqi == 12
    assert p.classify_unsolicited(rx.frame) is p.Unsolicited.REGISTRATION
    assert not got[1].frame.ok
    remove()
    remove()
    gw.feed(rx_line(registration))
    await ft.sleep(0)
    assert len(got) == 2 and len(other) == 3
    await link.close()


@pytest.mark.asyncio
async def test_reader_survives_noise_and_long_lines() -> None:
    """Blank lines, unknown text, ACK lines and over-long lines are skipped."""
    gw = FakeGateway()
    gw.reader = asyncio.StreamReader(limit=120)
    ft = FakeTime()
    link = make_link(gw, ft)
    await link.connect()
    gw.reader.feed_data(
        b"\r\n\xff\xfe garbage\r\nACK 1 081234\r\n" + b"X" * 400 + b"\r\n"
    )
    await ft.sleep(0)
    assert link.connected
    gw.responder = heater(DIALECT_B, HEATER)
    air = build_frame(DIALECT_B, 1, HEATER, b"\xb8", network_id=NET)
    assert (await link.send_frame(HEATER, air)).ok
    await link.close()


@pytest.mark.asyncio
async def test_eof_while_closing_skips_disconnect_handling() -> None:
    """An EOF that races close() does not fire on_disconnect."""
    calls = []
    gw, ft, link = await connected(on_disconnect=lambda: calls.append(1))
    link._closing = True
    gw.reader.feed_eof()
    for _ in range(5):
        await asyncio.sleep(0)
    assert calls == []
    assert not gw.writer.closed
    await link.close()


# --- disconnect ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disconnect_calls_callback_and_fails_pending() -> None:
    """EOF from the gateway fails pending sends and fires on_disconnect."""
    calls = []
    gw, ft, link = await connected(on_disconnect=lambda: calls.append("gone"))
    gw.tx_overrides.append("# busy")  # no TX line, so the send stays pending
    task = asyncio.ensure_future(link.send_frame(HEATER, b"\x01"))
    await asyncio.sleep(0)
    gw.reader.feed_eof()
    with pytest.raises(RadioLinkError, match="connection lost"):
        await task
    assert calls == ["gone"]
    assert not link.connected
    assert gw.writer.closed
    await link.close()


@pytest.mark.asyncio
async def test_read_error_and_failing_disconnect_callback(caplog) -> None:
    """A socket error is logged, and a failing disconnect callback is contained."""

    def broken() -> None:
        raise RuntimeError("callback boom")

    gw, ft, link = await connected(on_disconnect=broken)
    with caplog.at_level(logging.INFO):
        gw.reader.set_exception(ConnectionResetError("reset"))
        for _ in range(5):
            await asyncio.sleep(0)
    assert "connection lost" in caplog.text
    assert "disconnect callback failed" in caplog.text
    assert not link.connected
    await link.close()


@pytest.mark.asyncio
async def test_close_fails_pending_request() -> None:
    """Closing the link fails a request still waiting for its reply."""
    gw, ft, link = await connected()
    gw.responder = heater(DIALECT_B, HEATER)

    async def slow_sleep(seconds: float) -> None:
        await asyncio.Event().wait()

    link._sleep = slow_sleep  # the reply wait never times out on its own
    task = asyncio.ensure_future(link.request(HEATER, b"\xb8", lambda f: True))
    for _ in range(10):
        await asyncio.sleep(0)
    await link.close()
    with pytest.raises(RadioLinkError, match="connection closed"):
        await task


@pytest.mark.asyncio
async def test_silent_gateway_is_dropped_after_missed_confirmations(caplog) -> None:
    """A gateway that stops confirming transmits is treated as lost (reconnect)."""
    lost: list[bool] = []
    gw, ft, link = await connected(on_disconnect=lambda: lost.append(True))
    air = build_frame(DIALECT_B, 1, HEATER, b"\xb8", network_id=NET)
    gw.responder = heater(DIALECT_B, HEATER)
    assert (await link.send_frame(HEATER, air)).ok
    gw.silent = True
    with caplog.at_level(logging.ERROR):
        for _ in range(link_mod.MAX_MISSED_TX_CONFIRMS - 1):
            with pytest.raises(RadioLinkError, match="no TX confirmation"):
                await link.send_frame(HEATER, air)
        assert link.connected and lost == []
        with pytest.raises(RadioLinkError, match="no TX confirmation"):
            await link.send_frame(HEATER, air)
        for _ in range(20):
            await asyncio.sleep(0)
    assert lost == [True] and not link.connected
    assert "missed 3 TX confirmations" in caplog.text


@pytest.mark.asyncio
async def test_a_confirmation_resets_the_missed_count() -> None:
    """Occasional missed confirmations do not drop a working gateway."""
    lost: list[bool] = []
    gw, ft, link = await connected(on_disconnect=lambda: lost.append(True))
    air = build_frame(DIALECT_B, 1, HEATER, b"\xb8", network_id=NET)
    for _ in range(3):
        gw.silent = True
        for _ in range(link_mod.MAX_MISSED_TX_CONFIRMS - 1):
            with pytest.raises(RadioLinkError):
                await link.send_frame(HEATER, air)
        gw.silent = False
        await link.send_frame(HEATER, air, wait_ack=False)
    assert lost == [] and link.connected
    await link.close()


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("3.7-esp32", True),
        ("3.10-esp32", True),
        ("4.0-esp32", True),
        ("3.6-esp32", False),
        ("3.7", False),
        ("x.y-esp32", False),
        (None, False),
    ],
)
def test_supports_survey_gates_on_esp32_firmware(version, expected) -> None:
    """Only ESP32 firmware 3.7 or newer has the raw survey."""
    info = GatewayInfo(version, None, None, None, None, None, "# Q")
    assert supports_survey(info) is expected
    assert supports_survey(None) is False


@pytest.mark.asyncio
async def test_set_network_id_sends_n_and_frames_use_the_new_id() -> None:
    """``N<net>`` reaches the gateway and later frames carry the new id."""
    gw, _ft, link = await connected()
    await link.set_network_id(bytes.fromhex("ABCD"))
    assert gw.commands[-1] == "NABCD"
    assert link.network_id == bytes.fromhex("ABCD")
    with pytest.raises(ValueError, match="two bytes"):
        await link.set_network_id(b"\x01")
    assert link.network_id == bytes.fromhex("ABCD")
