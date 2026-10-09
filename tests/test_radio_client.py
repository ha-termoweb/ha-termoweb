"""Tests for RadioClient, the HttpClientProto served over the radio gateway."""

from __future__ import annotations

from datetime import datetime
import logging
from types import SimpleNamespace

import pytest
from fake_radio_link import (
    CLOCK_ACCEPTED,
    HEATER,
    NET,
    IDENTITY_SHORT,
    POWER_RECORD_IDLE,
    PROGRAM_HOURLY,
    STATUS_SHORT,
    FakeRadioLink,
    gateway_info,
)

from custom_components.termoweb.backend import radio_client as rc
from custom_components.termoweb.backend.radio.dialect import DIALECT_A, DIALECT_B
from custom_components.termoweb.backend.radio.link import RadioLinkError
from custom_components.termoweb.backend.radio_client import (
    RadioClient,
    RadioCommandError,
    RadioError,
    RadioUnsupportedError,
    radio_addr,
)
from custom_components.termoweb.inventory import build_node_inventory

NODES = [
    {"type": "htr", "addr": "6", "name": "Living room"},
    {"type": "acm", "addr": "7", "name": "Hall"},
    {"type": "htr", "addr": "bogus"},
]
DAY = [1] * 5 + [2] * 16 + [1] * 3
WHEN = datetime(2026, 10, 9, 16, 52, 9)  # a Friday


class Clock:
    """Settable monotonic clock."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def make_client(dialect: str = "B", **kwargs):
    """Return (client, links, clock) with a fake link factory."""

    links: list[FakeRadioLink] = []
    clock = Clock()

    def factory(host, port, dialect, **kw):
        link = FakeRadioLink(host, port, dialect, **kw)
        link.reply(0xB8, STATUS_SHORT)
        link.reply(0xB0, PROGRAM_HOURLY)
        link.reply(0xBC, POWER_RECORD_IDLE)
        for opcode in (0xB2, 0xB4, 0xB6, 0xBA, 0xD2):
            link.reply(opcode, bytes([opcode + 1, 0x55]))
        link.reply(0x51, CLOCK_ACCEPTED)
        link.reply(0x52, CLOCK_ACCEPTED)
        links.append(link)
        return link

    kwargs.setdefault("network_id", NET if dialect.upper() == "B" else None)
    client = RadioClient(
        "radio.local", 2323, dialect, NODES, link_factory=factory, clock=clock, **kwargs
    )
    client.reply_timeout = 0.01
    return client, links, clock


@pytest.fixture(autouse=True)
def _fixed_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the station clock used for clock syncs and RTC reads."""

    monkeypatch.setattr(rc, "_local_now", lambda: WHEN)


# --- construction and connection ---------------------------------------------


def test_constructor_validates_dialect_and_defaults_port() -> None:
    """Dialect names are case-insensitive; unknown ones fail fast."""

    client = RadioClient("h", 0, " b ", [], network_id=NET)
    assert client.dialect is DIALECT_B
    assert client._port == 2323  # noqa: SLF001
    assert client.link is None and not client.connected
    assert client.gateway_info is None
    with pytest.raises(ValueError, match="unknown radio dialect"):
        RadioClient("h", 2323, "C", [], network_id=NET)
    with pytest.raises(ValueError, match="explicit network_id"):
        RadioClient("h", 2323, "B", [], network_id=None)
    assert RadioClient("h", 2323, "A", [], network_id=None).dialect is DIALECT_A


@pytest.mark.asyncio
async def test_connects_lazily_once_and_reconnects_after_a_drop() -> None:
    """The link is created on first use, reused, and reopened when closed."""

    client, links, _ = make_client()
    assert links == []

    link = await client.async_connect()
    assert await client.async_connect() is link
    assert link.connects == 1 and client.connected
    assert (link.host, link.port, link.station_id) == ("radio.local", 2323, 1)
    assert link.network_id == NET
    assert client.gateway_info == gateway_info()

    link.drop()
    await client.async_connect()
    assert len(links) == 1 and link.connects == 2

    await client.async_close()
    assert link.closes == 1


@pytest.mark.asyncio
async def test_close_before_connect_is_a_no_op() -> None:
    """Closing an unused client does nothing."""

    client, links, _ = make_client()
    await client.async_close()
    assert links == []


@pytest.mark.asyncio
async def test_disconnect_listeners(caplog: pytest.LogCaptureFixture) -> None:
    """Link disconnects fan out; a failing callback is logged, removal is idempotent."""

    client, _, _ = make_client()
    link = await client.async_connect()
    calls: list[str] = []

    def boom() -> None:
        raise RuntimeError("boom")

    remove = client.add_disconnect_listener(lambda: calls.append("a"))
    client.add_disconnect_listener(boom)
    with caplog.at_level(logging.ERROR):
        link.drop()
    assert calls == ["a"]
    assert "Radio disconnect listener failed" in caplog.text

    remove()
    remove()
    link.drop()
    assert calls == ["a"]


# --- devices and nodes ---------------------------------------------------------


@pytest.mark.asyncio
async def test_list_devices_uses_the_gateway_mac() -> None:
    """The gateway's MAC, lowercase hex without colons, is the dev_id."""

    client, _, _ = make_client()
    devices = await client.list_devices()

    assert devices == [
        {
            "dev_id": "aabbcc001122",
            "name": "Radio gateway",
            "model": "ESP32 + CC1101 radio gateway (dialect B)",
            "serial_id": "aabbcc001122",
            "fw_version": "3.6-esp32",
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("mac", [None, "", "aa:bb", "zz:zz:zz:zz:zz:zz"])
async def test_list_devices_requires_a_mac(mac) -> None:
    """Without a usable MAC there is no stable dev_id, so setup must fail."""

    client, _, _ = make_client()
    link = await client.async_connect()
    link.info = gateway_info(mac=mac)
    link.gateway_info = link.info
    with pytest.raises(RadioError, match="MAC"):
        await client.list_devices()


@pytest.mark.asyncio
async def test_list_devices_without_gateway_info() -> None:
    """A link that never parsed a Q line also has no dev_id."""

    client, _, _ = make_client()
    link = await client.async_connect()
    link.gateway_info = None
    with pytest.raises(RadioError, match="MAC"):
        await client.list_devices()


@pytest.mark.asyncio
async def test_list_devices_propagates_connect_failure() -> None:
    """A gateway that cannot be reached raises RadioLinkError for setup to handle."""

    client, _, _ = make_client()
    link = await client.async_connect()
    link.drop()
    link.connect_errors.append(RadioLinkError("cannot connect"))
    with pytest.raises(RadioLinkError):
        await client.list_devices()


@pytest.mark.asyncio
async def test_get_nodes_returns_stored_nodes_in_cloud_shape() -> None:
    """Stored nodes come back as copies that build_node_inventory accepts."""

    client, links, _ = make_client()
    nodes = await client.get_nodes("aabbcc001122")

    assert nodes == {"nodes": NODES}
    nodes["nodes"][0]["name"] = "changed"
    assert NODES[0]["name"] == "Living room"
    inventory = build_node_inventory(await client.get_nodes("x"))
    assert [(n.type, n.addr) for n in inventory][:2] == [("htr", "6"), ("acm", "7")]
    assert links == []  # no radio traffic needed


# --- reads ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_node_settings_maps_status_and_program() -> None:
    """B8 + B0 replies become the canonical settings dict."""

    client, links, _ = make_client()
    settings = await client.get_node_settings("dev", ("htr", "6"))

    assert settings == {
        "units": "C",
        "ptemp": ["16.5", "18.5", "21.0"],
        "mode": "manual",
        "prog": DAY * 7,
        "stemp": "21.0",  # manual: the heater heats to its comfort preset
        "state": "off",  # from the BC power record: not heating
    }
    assert links[0].sent == [(HEATER, b"\xb8"), (HEATER, b"\xb0"), (HEATER, b"\xbc")]


@pytest.mark.asyncio
async def test_program_is_cached_until_refresh_window_expires() -> None:
    """B0 is only re-sent after program_refresh_s."""

    client, links, clock = make_client()
    await client.get_node_settings("dev", ("htr", "6"))
    clock.now += client.program_refresh_s - 1
    second = await client.get_node_settings("dev", ("htr", "6"))
    assert second["prog"] == DAY * 7
    assert links[0].payloads().count(b"\xb0") == 1

    clock.now += 2
    await client.get_node_settings("dev", ("htr", "6"))
    assert links[0].payloads().count(b"\xb0") == 2


@pytest.mark.asyncio
async def test_program_read_failure_keeps_the_last_known_program() -> None:
    """A failed B0 keeps the previous program, or omits prog when none is known."""

    client, links, clock = make_client()
    await client.async_connect()
    links[0].replies.pop(0xB0)

    first = await client.get_node_settings("dev", ("htr", "6"))
    assert "prog" not in first
    assert links[0].payloads().count(b"\xb0") == 1

    links[0].reply(0xB0, PROGRAM_HOURLY)
    await client.get_node_settings("dev", ("htr", "6"))
    clock.now += client.program_refresh_s + 1
    links[0].replies.pop(0xB0)
    third = await client.get_node_settings("dev", ("htr", "6"))
    assert third["prog"] == DAY * 7


@pytest.mark.asyncio
async def test_split_half_hour_program_is_cached_as_unknown() -> None:
    """A 48-slot program with a split hour omits prog and is not re-read each poll."""

    from custom_components.termoweb.backend.radio import protocol as p

    slots = [2] * (48 * 7)
    slots[1] = 0
    client, links, _ = make_client()
    await client.async_connect()
    links[0].reply(0xB0, b"\xb1" + p._pack_slots(slots))  # noqa: SLF001

    assert "prog" not in await client.get_node_settings("dev", ("htr", "6"))
    assert "prog" not in await client.get_node_settings("dev", ("htr", "6"))
    assert links[0].payloads().count(b"\xb0") == 1


@pytest.mark.asyncio
async def test_status_read_failures_return_none() -> None:
    """No ack, no matching reply or a dead gateway all mean "no data"."""

    client, links, _ = make_client()
    link = await client.async_connect()

    link.no_ack.add(HEATER)
    assert await client.get_node_settings("dev", ("htr", "6")) is None
    link.no_ack.clear()

    link.reply(0xB8, IDENTITY_SHORT)  # a reply, but not a status record
    assert await client.get_node_settings("dev", ("htr", "6")) is None
    assert link.listeners == []  # the reply listener is always removed

    link.drop()
    link.connect_errors.append(RadioLinkError("cannot connect"))
    assert await client.get_node_settings("dev", ("htr", "6")) is None
    assert len(links) == 1


@pytest.mark.asyncio
async def test_reply_from_another_node_is_ignored() -> None:
    """Only a frame from the addressed heater to this station resolves a read."""

    from fake_radio_link import received

    client, _, _ = make_client()
    link = await client.async_connect()
    link.replies.pop(0xB8)

    def inject(dst: int, payload: bytes) -> None:
        if payload == b"\xb8":
            link.deliver(received(7, STATUS_SHORT))  # wrong heater
            link.deliver(received(HEATER, STATUS_SHORT, dst=2))  # other station
            link.deliver(received(HEATER, STATUS_SHORT))
            link.deliver(received(HEATER, STATUS_SHORT))  # duplicate is ignored

    link.on_send = inject
    settings = await client.get_node_settings("dev", ("htr", "6"))
    assert settings["mode"] == "manual"


@pytest.mark.asyncio
async def test_noted_max_power_fills_in_when_status_lacks_it() -> None:
    """Power from BE requests is reported until a status record carries its own."""

    client, _, _ = make_client()
    client.note_max_power(HEATER, 1143.3)

    settings = await client.get_node_settings("dev", ("htr", "6"))
    assert settings["max_power"] == 1143.3


@pytest.mark.asyncio
async def test_node_descriptor_forms() -> None:
    """Tuples and node objects work; invalid descriptors raise ValueError."""

    client, _, _ = make_client()
    node = SimpleNamespace(type="htr", addr="6")
    assert (await client.get_node_settings("dev", node))["mode"] == "manual"
    with pytest.raises(ValueError, match="Invalid node descriptor"):
        await client.get_node_settings("dev", ("", ""))
    with pytest.raises(ValueError, match="outside"):
        await client.get_node_settings("dev", ("htr", "300"))


@pytest.mark.parametrize("bad", ["abc", None, "0", "255", "-1"])
def test_radio_addr_rejects_out_of_range(bad) -> None:
    """Radio ids are decimal 1-254."""

    with pytest.raises(ValueError):
        radio_addr(bad)
    assert radio_addr(" 06 ") == 6


# --- writes --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_set_node_settings_sends_verified_frames_in_order() -> None:
    """Presets, program and manual setpoint each get an accepted verdict."""

    client, links, clock = make_client("A")
    await client.get_node_settings("dev", ("htr", "6"))  # fills the program cache
    links[0].sent.clear()

    result = await client.set_node_settings(
        "dev",
        ("htr", "6"),
        mode="manual",
        stemp=21.0,
        prog=DAY * 7,
        ptemp=[7.0, 16.5, 21.0],
    )

    assert result is None
    assert [p[0] for p in links[0].payloads()] == [0xB6, 0xB2, 0xB4]
    assert links[0].payloads()[-1] == bytes.fromhex("B4022A")
    await client.get_node_settings("dev", ("htr", "6"))
    assert links[0].payloads()[-2:] == [b"\xb0", b"\xbc"]  # program re-read


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "stemp", "payload"),
    [
        ("auto", None, "B401"),
        ("off", None, "B404"),
        ("manual", None, "B402"),
        ("modified_auto", 22.0, "B4032C"),
    ],
)
async def test_mode_writes(mode, stemp, payload) -> None:
    """Mode strings from climate.py map to B4 writes in dialect A."""

    client, links, _ = make_client("A")
    await client.set_node_settings("dev", ("htr", "6"), mode=mode, stemp=stemp)
    assert links[0].payloads() == [bytes.fromhex(payload)]


@pytest.mark.asyncio
async def test_write_failures_raise_clear_errors() -> None:
    """Missing ack, missing reply, rejection and odd verdicts all raise."""

    client, links, _ = make_client("A")
    link = await client.async_connect()

    link.no_ack.add(HEATER)
    with pytest.raises(RadioCommandError, match="did not acknowledge command B4"):
        await client.set_node_settings("dev", ("htr", "6"), mode="auto")
    link.no_ack.clear()

    link.replies.pop(0xB4)
    with pytest.raises(RadioCommandError, match="sent no reply"):
        await client.set_node_settings("dev", ("htr", "6"), mode="auto")

    link.reply(0xB4, b"\xb5\x56")
    with pytest.raises(RadioCommandError, match="rejected command B4 01"):
        await client.set_node_settings("dev", ("htr", "6"), mode="auto")

    link.reply(0xB4, b"\xb5\x99")
    with pytest.raises(RadioCommandError, match="unknown answer"):
        await client.set_node_settings("dev", ("htr", "6"), mode="auto")


@pytest.mark.asyncio
async def test_dialect_b_settings_write_merges_presets_and_mode() -> None:
    """Dialect B re-reads status, then sends one B6 with presets + mode and a 24-slot B2."""

    client, links, _ = make_client()
    await client.set_node_settings(
        "dev", ("htr", "6"), mode="off", prog=DAY * 7, ptemp=[7.0, 16.5, 21.0]
    )
    payloads = links[0].payloads()
    assert payloads[:2] == [b"\xb8", bytes.fromhex("B60E212A04")]
    assert payloads[2][0] == 0xB2 and len(payloads[2]) == 43
    assert len(payloads) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "b6"),
    [
        ({"mode": "auto"}, "B621252A01"),  # presets kept from status B9 21 25 2A 02
        ({"ptemp": [7.0, 16.5, 21.0]}, "B60E212A02"),  # mode kept from status
    ],
)
async def test_dialect_b_keeps_unchanged_fields_from_status(kwargs, b6) -> None:
    """A mode-only or preset-only write re-sends the other fields as read."""

    client, links, _ = make_client()
    await client.set_node_settings("dev", ("htr", "6"), **kwargs)
    assert links[0].payloads() == [b"\xb8", bytes.fromhex(b6)]


@pytest.mark.asyncio
async def test_dialect_b_override_write_is_unsupported() -> None:
    """Dialect B has no known temporary-override write."""

    client, links, _ = make_client()
    with pytest.raises(RadioUnsupportedError, match="Temporary override on dialect B"):
        await client.set_node_settings(
            "dev", ("htr", "6"), mode="modified_auto", stemp=21.0
        )
    assert links == [] or links[0].sent == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [None, "manual", "heat"])
async def test_dialect_b_setpoint_is_written_as_comfort(mode) -> None:
    """A manual target becomes B6 with comfort = target and mode manual."""

    client, links, _ = make_client()
    await client.set_node_settings("dev", ("htr", "6"), mode=mode, stemp=23.0)
    assert links[0].payloads() == [b"\xb8", bytes.fromhex("B621252E02")]


@pytest.mark.asyncio
async def test_dialect_b_setpoint_must_stay_above_eco() -> None:
    """Comfort must stay above eco, so a target at or below eco is refused."""

    client, links, _ = make_client()
    with pytest.raises(ValueError, match="presets must"):
        await client.set_node_settings("dev", ("htr", "6"), stemp=18.5)
    assert bytes.fromhex("B6") not in [p[:1] for p in links[0].payloads()]


@pytest.mark.asyncio
async def test_invalid_writes_send_nothing() -> None:
    """Validation happens before the first frame goes on air."""

    client, links, _ = make_client()
    await client.async_connect()
    with pytest.raises(ValueError):
        await client.set_node_settings(
            "dev", ("htr", "6"), ptemp=[7.0, 16.0, 21.0], prog=[0] * 10
        )
    with pytest.raises(ValueError, match="Celsius"):
        await client.set_node_settings("dev", ("htr", "6"), mode="auto", units="F")
    assert links[0].sent == []


@pytest.mark.asyncio
async def test_boost_through_settings_is_unsupported() -> None:
    """The cloud's boost_time / cancel_boost settings fields have no radio form."""

    client, _, _ = make_client()
    with pytest.raises(RadioUnsupportedError, match="not supported over the radio"):
        await client.set_node_settings("dev", ("htr", "6"), boost_time=60)
    with pytest.raises(RadioUnsupportedError):
        await client.set_node_settings("dev", ("htr", "6"), cancel_boost=True)


@pytest.mark.asyncio
async def test_lock_toggle() -> None:
    """set_node_lock sends BA 01 / BA 00."""

    client, links, _ = make_client()
    await client.set_node_lock("dev", ("htr", "6"), lock=True)
    await client.set_node_lock("dev", ("htr", "6"), lock=False)
    assert links[0].payloads() == [b"\xba\x01", b"\xba\x00"]


@pytest.mark.asyncio
async def test_heater_boost_toggle_and_accumulator_boost_unsupported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """D2 toggles a heater's boost; accumulators are not supported yet."""

    client, links, _ = make_client()
    await client.set_acm_boost_state("dev", "6", boost=True)
    with caplog.at_level(logging.DEBUG, logger=rc.__name__):
        await client.set_acm_boost_state("dev", 6, boost=False, boost_time=30)
    assert links[0].payloads() == [b"\xd2\x01", b"\xd2\x00"]
    assert "own duration" in caplog.text

    with pytest.raises(RadioUnsupportedError, match="Accumulator boost"):
        await client.set_acm_boost_state("dev", "7", boost=True)
    with pytest.raises(RadioUnsupportedError):
        await client.set_acm_boost_state("dev", "9", boost=True)


@pytest.mark.asyncio
async def test_unsupported_features() -> None:
    """Cloud-only features raise or report "no data" per call site."""

    client, links, _ = make_client()
    for call in (
        client.set_node_display_select("dev", ("htr", "6"), select=True),
        client.set_node_priority("dev", ("htr", "6"), priority=3),
        client.set_power_limit("dev", power_limit=3000),
        client.set_acm_extra_options("dev", "7", boost_time=60),
    ):
        with pytest.raises(RadioUnsupportedError, match="not supported over the radio"):
            await call
    assert await client.get_power_limit("dev") is None
    assert await client.get_node_samples("dev", ("htr", "6"), 0, 1) == []
    assert await client.get_geo_data("dev") is None
    assert links == []


@pytest.mark.asyncio
async def test_get_rtc_time_is_local_time_in_cloud_shape() -> None:
    """The coordinator reads y/n/d/h/m/s."""

    client, links, _ = make_client()
    assert await client.get_rtc_time("dev") == {
        "y": 2026,
        "n": 10,
        "d": 9,
        "h": 16,
        "m": 52,
        "s": 9,
    }
    assert links == []


def test_local_now_is_timezone_aware(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real station clock is Home Assistant's local time."""

    monkeypatch.undo()
    assert rc._local_now().tzinfo is not None  # noqa: SLF001


# --- station helpers -----------------------------------------------------------


@pytest.mark.asyncio
async def test_clock_sync_uses_the_dialect_suffix() -> None:
    """Dialect B sends the 8-byte form; dialect A appends 03. Both expect 53 55."""

    client, links, _ = make_client("B")
    await client.async_sync_clock(HEATER, registering=True)
    await client.async_sync_clock(HEATER, registering=False)
    assert links[0].payloads() == [
        bytes.fromhex("511A0A0905103409"),
        bytes.fromhex("521A0A0905103409"),
    ]

    client_a, links_a, _ = make_client("A")
    await client_a.async_sync_clock(HEATER, registering=False)
    assert client_a.dialect is DIALECT_A
    assert links_a[0].payloads() == [bytes.fromhex("521A0A090510340903")]

    links[0].reply(0x52, b"\x53\x56")
    with pytest.raises(RadioCommandError, match="rejected"):
        await client.async_sync_clock(HEATER, registering=False)


@pytest.mark.asyncio
async def test_async_send_requires_an_ack() -> None:
    """Fire-and-forget payloads still need the heater's link-layer ack."""

    client, links, _ = make_client()
    await client.async_send(HEATER, b"\xbf\x01")
    assert links[0].payloads() == [b"\xbf\x01"]
    links[0].no_ack.add(HEATER)
    with pytest.raises(RadioCommandError, match="did not acknowledge BF 01"):
        await client.async_send(HEATER, b"\xbf\x01")


@pytest.mark.asyncio
async def test_power_record_read_failure_leaves_state_out() -> None:
    """A heater that does not answer BC simply has no heating state this poll."""

    client, links, _ = make_client()
    link = await client.async_connect()
    link.replies.pop(0xBC)
    settings = await client.get_node_settings("dev", ("htr", "6"))
    assert "state" not in settings
    assert await client.read_power_record(HEATER) is None
