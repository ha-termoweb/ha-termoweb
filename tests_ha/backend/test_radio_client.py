"""Tests for RadioClient, the HttpClientProto served over the radio gateway."""

from __future__ import annotations

import asyncio
from datetime import datetime
import logging
from types import SimpleNamespace

import pytest
from tests_ha.fakes.radio_link import (
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
FOREIGN_NET = bytes.fromhex("5678")  # synthetic neighbouring network id
STATUS_E6 = bytes.fromhex("B921252A0300CC2C2CF0100300FF")
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
        "state": "off",  # from the BC power record: not heating
        "stemp": "22.0",  # power record byte 3: the heater's active setpoint
        "mtemp": "22.0",  # power record byte 1
        "priority": 0,  # local power manager default
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

    from tests_ha.fakes.radio_link import received

    client, _, _ = make_client()
    link = await client.async_connect()
    link.replies.pop(0xB8)

    def inject(dst: int, payload: bytes) -> None:
        if payload == b"\xb8":
            link.deliver(received(7, STATUS_SHORT))  # wrong heater
            link.deliver(received(HEATER, STATUS_SHORT, dst=2))  # other station
            # Same ids on a neighbouring network: not our heater.
            link.deliver(received(HEATER, STATUS_E6, network_id=FOREIGN_NET))
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
@pytest.mark.parametrize("mode", [None, "manual", "heat"])
async def test_dialect_b_setpoint_is_its_own_b6_byte(mode) -> None:
    """A manual target is B6 <presets> 02 <setpoint>; the presets are untouched."""

    client, links, _ = make_client()
    await client.set_node_settings("dev", ("htr", "6"), mode=mode, stemp=23.0)
    assert links[0].payloads() == [b"\xb8", bytes.fromhex("B621252A022E")]


@pytest.mark.asyncio
async def test_dialect_b_temporary_override() -> None:
    """modified_auto + target: B6 <presets> 03 <setpoint> (the heater ends it)."""

    client, links, _ = make_client()
    await client.set_node_settings(
        "dev", ("htr", "6"), mode="modified_auto", stemp=24.0
    )
    assert links[0].payloads() == [b"\xb8", bytes.fromhex("B621252A0330")]
    with pytest.raises(ValueError, match="outside"):
        await client.set_node_settings("dev", ("htr", "6"), stemp=40.0)


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
async def test_dialect_b_lock_is_ack_only_and_remembered() -> None:
    """A dialect-B heater acks BA without a reply; the written state is reported."""

    client, links, _ = make_client()
    link = await client.async_connect()
    link.replies.pop(0xBA)  # dialect B sends no BB record
    assert "lock" not in await client.get_node_settings("dev", ("htr", "6"))
    await client.set_node_lock("dev", ("htr", "6"), lock=True)
    assert (await client.get_node_settings("dev", ("htr", "6")))["lock"] is True


@pytest.mark.asyncio
async def test_dialect_a_lock_needs_its_reply() -> None:
    """Dialect A still requires BB 55; nothing is remembered when it fails."""

    client, links, _ = make_client("A")
    link = await client.async_connect()
    link.replies.pop(0xBA)
    with pytest.raises(RadioCommandError, match="sent no reply"):
        await client.set_node_lock("dev", ("htr", "6"), lock=True)
    assert "lock" not in await client.get_node_settings("dev", ("htr", "6"))


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
@pytest.mark.parametrize(("dialect", "reply"), [("A", True), ("B", False)])
async def test_flash_display(dialect, reply) -> None:
    """5E 01 needs 5F 55 in dialect A; dialect B only acks it. Deselect is a no-op."""

    client, links, _ = make_client(dialect)
    link = await client.async_connect()
    link.reply(0x5E, b"\x5f\x55")
    await client.set_node_display_select("dev", ("htr", "6"), select=False)
    assert link.sent == []
    await client.set_node_display_select("dev", ("htr", "6"), select=True)
    assert link.payloads() == [b"\x5e\x01"]
    link.replies.pop(0x5E)
    if reply:
        with pytest.raises(RadioCommandError, match="sent no reply"):
            await client.set_node_display_select("dev", ("htr", "6"), select=True)
    else:
        await client.set_node_display_select("dev", ("htr", "6"), select=True)


@pytest.mark.asyncio
async def test_power_limit_switches_a_heater_off_and_back() -> None:
    """Over the limit the heater is switched off (B6 .. 04); without it, restored."""

    client, links, _ = make_client()
    link = await client.async_connect()
    assert await client.get_power_limit("dev") == 0  # no limit; entity stays usable
    await client.set_power_limit("dev", power_limit=1000)
    await client.set_node_priority("dev", ("htr", "6"), priority=3)
    assert await client.get_power_limit("dev") == 1000
    client.note_max_power(HEATER, 1500.0)
    link.reply(0xBC, bytes.fromhex("BDDB002CE43A0C0100"))  # heating
    await client.read_power_record(HEATER)
    link.sent.clear()

    await client.async_balance_power()
    assert link.payloads() == [b"\xb8", b"\xb8", bytes.fromhex("B621252A04")]
    assert client.power.shed() == {HEATER: 2}  # manual, to restore later
    settings = await client.get_node_settings("dev", ("htr", "6"))
    assert settings["priority"] == 3 and settings["max_power"] == 1500.0

    link.sent.clear()
    await client.set_power_limit("dev", power_limit=0)
    await client.async_balance_power()
    assert link.payloads() == [b"\xb8", bytes.fromhex("B621252A02")]
    assert client.power.shed() == {}


@pytest.mark.asyncio
async def test_power_limit_edge_cases(caplog) -> None:
    """Already off: nothing to do; radio errors are logged; the user's mode wins."""

    client, links, _ = make_client()
    link = await client.async_connect()
    await client.set_power_limit("dev", power_limit=1000)
    client.note_max_power(HEATER, 1500.0)
    client.power.note_heating(HEATER, True)
    link.reply(0xB8, bytes.fromhex("B921252A04"))  # the heater is already off
    await client.async_balance_power()
    assert client.power.shed() == {}

    link.replies.pop(0xB8)  # status read fails
    with caplog.at_level(logging.ERROR):
        await client.async_balance_power()
    assert "could not switch heater 6 off" in caplog.text

    client.power.mark_shed(HEATER, 1)
    await client.set_power_limit("dev", power_limit=0)
    with caplog.at_level(logging.ERROR):
        await client.async_balance_power()  # restore fails: stays shed
    assert "could not restore heater 6" in caplog.text
    assert client.power.shed() == {HEATER: 1}

    link.reply(0xB8, STATUS_SHORT)
    client.power.mark_shed(HEATER, 4)  # unknown restore mode: just forget it
    await client.async_balance_power()
    assert client.power.shed() == {}

    client.power.mark_shed(HEATER, 2)
    await client.set_node_settings("dev", ("htr", "6"), mode="auto")
    assert client.power.shed() == {}  # the user's own mode change wins


@pytest.mark.asyncio
async def test_samples_report_the_estimated_energy_counter() -> None:
    """With a heater's power known, each power-record read feeds the Wh estimate."""

    client, links, clock = make_client()
    assert await client.get_node_samples("dev", ("htr", "6"), 0, 1) == []
    client.note_max_power(HEATER, 1500.0)
    link = await client.async_connect()
    link.reply(0xBC, bytes.fromhex("BDDB002CE43A0C0100"))  # heating, duty 12 %
    await client.read_power_record(HEATER)
    clock.now += 600
    await client.read_power_record(HEATER)
    (sample,) = await client.get_node_samples("dev", ("htr", "6"), 0, 1)
    assert sample["counter"] == pytest.approx(1500 * 0.12 * 600 / 3600, abs=1e-3)


@pytest.mark.asyncio
async def test_unsupported_features() -> None:
    """Cloud-only features raise or report "no data" per call site."""

    client, links, _ = make_client()
    for call in (client.set_acm_extra_options("dev", "7", boost_time=60),):
        with pytest.raises(RadioUnsupportedError, match="not supported over the radio"):
            await call
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


# --- raw survey -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_survey_needs_survey_firmware_and_returns_bursts() -> None:
    """Firmware 3.7-esp32 surveys under the exchange lock; older firmware refuses."""

    client, links, _ = make_client()
    await client.async_connect()
    links[0].info = links[0].gateway_info = gateway_info(version="3.7-esp32")
    links[0].survey_bursts = ["burst"]
    assert await client.async_survey(30) == ["burst"]
    assert links[0].surveys == [30]

    links[0].gateway_info = gateway_info(version="3.6-esp32")
    with pytest.raises(RadioError, match="3.6-esp32 has no raw survey"):
        await client.async_survey(30)
    links[0].gateway_info = None
    with pytest.raises(RadioError, match="unknown has no raw survey"):
        await client.async_survey(30)
    assert links[0].surveys == [30]


# --- pairing, factory reset, restore --------------------------------------------


@pytest.mark.asyncio
async def test_pair_runs_under_the_exchange_lock_and_skips_stored_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pairing holds heater commands back and never hands out a stored id."""

    client, links, _ = make_client()
    client._programs[8] = (0.0, None)  # noqa: SLF001
    client._locks[8] = True  # noqa: SLF001
    seen: dict = {}

    async def fake_pair(link, **kwargs):
        seen.update(kwargs, link=link, locked=client._exchange_lock.locked())  # noqa: SLF001
        return [SimpleNamespace(node_id=8)]

    monkeypatch.setattr(rc, "pair_heaters", fake_pair)
    paired = await client.async_pair(120, wanted_id=8, max_heaters=1, idle_stop_s=5)

    assert [h.node_id for h in paired] == [8]
    assert seen["link"] is links[0] and seen["locked"] is True
    assert seen["existing_ids"] == {6, 7}  # "bogus" is skipped
    assert (seen["window_s"], seen["wanted_id"], seen["max_heaters"]) == (120, 8, 1)
    assert seen["idle_stop_s"] == 5
    assert 8 not in client._programs and 8 not in client._locks  # noqa: SLF001


@pytest.mark.asyncio
async def test_factory_reset_sends_c8_and_needs_c9_55() -> None:
    """Dialect B resets with ``C8 01 D0``; a rejection raises."""

    client, links, _ = make_client()
    await client.async_connect()
    links[0].reply(0xC8, bytes.fromhex("C955"), bytes.fromhex("C956"))
    client._locks[HEATER] = True  # noqa: SLF001
    await client.async_factory_reset(HEATER)
    assert links[0].sent == [(HEATER, bytes.fromhex("C801D0"))]
    assert HEATER not in client._locks  # noqa: SLF001
    with pytest.raises(RadioCommandError, match="rejected"):
        await client.async_factory_reset(HEATER)


@pytest.mark.asyncio
async def test_factory_reset_is_unsupported_in_dialect_a() -> None:
    client, links, _ = make_client("A")
    with pytest.raises(RadioUnsupportedError, match="Factory reset in dialect A"):
        await client.async_factory_reset(HEATER)
    assert links == []  # nothing connected, nothing sent


@pytest.mark.asyncio
async def test_restore_syncs_the_clock_then_writes_settings() -> None:
    """Clock (ack only), then the B6 presets+mode write and the B2 program."""

    client, links, _ = make_client()
    await client.async_connect()
    links[0].replies.pop(0x51)  # right after pairing the heater does not answer 53
    await client.async_restore(
        HEATER, mode="manual", stemp=21.5, ptemp=[7.0, 17.0, 20.0], prog=[1] * 168
    )
    payloads = links[0].payloads()
    assert payloads[0] == bytes.fromhex("511A0A0905103409")
    assert payloads[1] == bytes([0xB8])
    assert payloads[2] == bytes.fromhex("B60E222802" + "2B")
    assert payloads[3][0] == 0xB2 and len(payloads[3]) == 43


@pytest.mark.asyncio
async def test_manual_target_is_remembered_from_reads_and_writes() -> None:
    """A manual read or write keeps the target; an override write does not."""

    client, _links, _ = make_client()
    assert client.manual_setpoint(HEATER) is None
    await client.get_node_settings("dev", ("htr", str(HEATER)))  # mode 02, BD 2C
    assert client.manual_setpoint(HEATER) == 22.0
    await client.set_node_settings("dev", ("htr", str(HEATER)), stemp=20.5)
    assert client.manual_setpoint(HEATER) == 20.5
    await client.set_node_settings(
        "dev", ("htr", str(HEATER)), mode="modified_auto", stemp=24.0
    )
    assert client.manual_setpoint(HEATER) == 20.5


@pytest.mark.asyncio
async def test_restore_writes_the_manual_target_before_another_mode() -> None:
    """``B6 .. 02 <target>`` first, then ``B6 .. <mode>`` and the program."""

    client, links, _ = make_client()
    await client.async_connect()
    await client.async_restore(
        HEATER,
        mode="auto",
        ptemp=[16.5, 18.5, 21.0],
        prog=[1] * 168,
        manual_stemp=20.0,
    )
    payloads = links[0].payloads()
    assert payloads[1:] == [
        bytes([0xB8]),
        bytes.fromhex("B621252A0228"),
        bytes([0xB8]),
        bytes.fromhex("B621252A01"),
        payloads[5],
    ]
    assert payloads[5][0] == 0xB2

    links[0].sent.clear()
    await client.async_restore(HEATER, mode="manual", stemp=21.0, manual_stemp=20.0)
    assert links[0].payloads()[1:] == [bytes([0xB8]), bytes.fromhex("B621252A022A")]


@pytest.mark.asyncio
async def test_network_id_switch_and_identity_read() -> None:
    """The client moves its link to another network and reads 5A identities."""

    client, links, _ = make_client()
    await client.async_set_network_id(bytes.fromhex("ABCD"))
    assert links[0].network_ids == [bytes.fromhex("ABCD")]
    assert client._network_id == bytes.fromhex("ABCD")  # noqa: SLF001
    links[0].reply(0x5A, IDENTITY_SHORT, bytes.fromhex("5B56") + bytes(16))
    assert await client.async_read_identity(HEATER) == IDENTITY_SHORT[2:]
    with pytest.raises(RadioCommandError, match="unknown identity reply"):
        await client.async_read_identity(HEATER)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [
        bytes.fromhex("B90C252A02"),  # anti-frost 6.0 C: below the 7 C floor
        bytes.fromhex("B9212A2A02"),  # eco equals comfort
    ],
)
async def test_power_limit_skips_a_heater_whose_presets_cannot_be_written(
    caplog, status: bytes
) -> None:
    """Dialect B re-sends the presets with the mode; invalid ones skip the heater."""

    client, links, _ = make_client()
    link = await client.async_connect()
    await client.set_power_limit("dev", power_limit=1000)
    client.note_max_power(HEATER, 1500.0)
    client.power.note_heating(HEATER, True)
    link.reply(0xB8, status)
    with caplog.at_level(logging.ERROR):
        await client.async_balance_power()
    assert "could not switch heater 6 off" in caplog.text
    assert client.power.shed() == {}  # never switched off, so nothing to restore
    assert all(payload[0] != 0xB6 for payload in link.payloads())

    caplog.clear()
    client.power.mark_shed(HEATER, 2)
    await client.set_power_limit("dev", power_limit=0)
    with caplog.at_level(logging.ERROR):
        await client.async_balance_power()
    assert "could not restore heater 6" in caplog.text
    assert client.power.shed() == {HEATER: 2}


@pytest.mark.asyncio
async def test_concurrent_dialect_b_writes_do_not_revert_each_other() -> None:
    """B6 carries presets and mode together: each read-modify-write is atomic."""

    client, links, _ = make_client()
    link = await client.async_connect()
    status = bytearray(STATUS_SHORT)  # af 16.5, eco 18.5, comfort 21.0, manual

    def heater_state(dst: int, payload: bytes) -> None:
        if payload[0] == 0xB6:  # the heater applies presets and mode
            status[1:4] = payload[1:4]
            if len(payload) > 4:
                status[4] = payload[4]
            link.reply(0xB8, bytes(status))

    link.on_send = heater_state
    link.reply(0xB8, bytes(status))
    send = link.send_frame

    async def slow_send(dst, air, **kwargs):
        await asyncio.sleep(0)  # airtime: let the other writer run
        return await send(dst, air, **kwargs)

    link.send_frame = slow_send
    await asyncio.gather(
        client.set_node_settings("dev", ("htr", "6"), ptemp=[7.0, 16.5, 22.0]),
        client._write_settings(HEATER, mode="off"),  # noqa: SLF001 - power limit
    )

    assert status[1:5] == bytes([14, 33, 44, 4])  # new comfort 22 C and mode off
