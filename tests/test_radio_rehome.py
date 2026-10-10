# ruff: noqa: D103,INP001,E402
"""Tests for moving a radio entry's heaters onto the site network."""

from __future__ import annotations

import asyncio
import hashlib
from types import SimpleNamespace
from typing import Any

import pytest
from conftest import _install_stubs, build_entry_runtime

_install_stubs()

from custom_components.termoweb import config_flow, radio_pairing as rp, radio_rehome
from custom_components.termoweb.backend.radio.dialect import DIALECT_A, DIALECT_B
from custom_components.termoweb.backend.radio.link import RadioLinkError
from custom_components.termoweb.backend.radio.pairing import (
    NoFreeAddressError,
    PairedHeater,
)
from custom_components.termoweb.backend.radio_client import (
    RadioClient,
    RadioCommandError,
)
from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.domain.state import HeaterState
from custom_components.termoweb.services import radio_pairing as service
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
from homeassistant.exceptions import HomeAssistantError

OLD_NET = "1234"  # synthetic legacy network id
SITE_NET = hashlib.sha256(b"test-instance-id:dev").digest()[:2]
ID_6 = bytes(range(16))
ID_7 = bytes(range(1, 17))
STATE = HeaterState(mode="auto", stemp="18.5", ptemp=["7", "17", "20"])
SNAPSHOT = {"mode": "auto", "ptemp": [7.0, 17.0, 20.0], "manual_stemp": 20.0}


class FakeClient:
    """Records every rehome step; scripted identities, resets and pairing."""

    def __init__(self, dialect=DIALECT_B) -> None:
        self.dialect = dialect
        self.calls: list[tuple] = []
        self.identities: dict[int, Any] = {6: ID_6, 7: ID_7}
        self.reset_errors: dict[int, Exception] = {}
        self.paired: Any = []
        self.restore_error: Exception | None = None

    def manual_setpoint(self, addr: int) -> float | None:
        return 20.0

    async def async_read_identity(self, addr: int) -> bytes:
        self.calls.append(("identity", addr))
        value = self.identities.get(addr)
        if isinstance(value, Exception) or value is None:
            raise value or RadioCommandError(f"heater {addr} silent")
        return value

    async def async_factory_reset(self, addr: int) -> None:
        self.calls.append(("reset", addr))
        if addr in self.reset_errors:
            raise self.reset_errors[addr]

    async def async_set_network_id(self, net: bytes) -> None:
        self.calls.append(("net", net))

    async def async_pair(self, window_s, **kwargs):
        self.calls.append(("pair", window_s, kwargs))
        if isinstance(self.paired, Exception):
            raise self.paired
        return self.paired

    async def async_restore(self, addr, **kwargs):
        self.calls.append(("restore", addr, kwargs))
        if self.restore_error is not None:
            raise self.restore_error


class Rig:
    """A loaded radio entry with one or two heaters and a FakeClient."""

    def __init__(self, addrs=(6,), dialect=DIALECT_B) -> None:
        self.hass = HomeAssistant()
        nodes = [{"type": "htr", "addr": str(a), "name": f"Room {a}"} for a in addrs]
        nodes.append({"type": "htr", "addr": "bogus"})
        self.entry = ConfigEntry(
            "radio-entry",
            data={
                "brand": "radio",
                "dialect": "B",
                "network_id": OLD_NET,
                "nodes": nodes,
            },
            options={"debug": True},
        )
        self.hass.config_entries.add_entry(self.entry)
        self.client = FakeClient(dialect)
        view = SimpleNamespace(get_heater_state=lambda _t, addr: STATE)
        self.runtime = build_entry_runtime(
            hass=self.hass,
            entry_id=self.entry.entry_id,
            client=self.client,
            config_entry=self.entry,
            coordinator=SimpleNamespace(domain_view=view, data={}),
            brand="radio",
        )

    async def rehome(self) -> dict[str, Any]:
        return await radio_rehome.async_rehome(self.hass, self.runtime, 120)

    def steps(self) -> list[str]:
        return [call[0] for call in self.client.calls]


def _paired(node_id: int, identity: bytes | None = None) -> PairedHeater:
    return PairedHeater(node_id, SimpleNamespace(), identity)


# --- async_rehome ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_heater_moves_restores_and_keeps_its_name() -> None:
    rig = Rig()
    rig.client.paired = [_paired(2)]  # no identity needed for one heater

    result = await rig.rehome()

    assert result == {
        "moved": [{"from": 6, "to": 2, "restored": True}],
        "unidentified": [],
        "waiting": [],
    }
    assert rig.steps() == ["identity", "reset", "net", "pair", "restore"]
    assert rig.client.calls[2] == ("net", SITE_NET)
    assert rig.client.calls[3] == ("pair", 120, {"max_heaters": 1, "idle_stop_s": 60.0})
    assert rig.client.calls[4] == ("restore", 2, SNAPSHOT)
    assert rig.entry.data["network_id"] == SITE_NET.hex().upper()
    assert rig.entry.data["nodes"] == [
        {"type": "htr", "addr": "bogus"},
        {"type": "htr", "addr": "2", "name": "Room 6"},
    ]
    assert rig.entry.options["radio_restore"] == {}
    assert rig.hass.config_entries.scheduled_reloads == ["radio-entry"]
    summary = radio_rehome.rehome_summary(result)
    assert "Heater 6 is now heater 2. Its settings were restored." in summary


@pytest.mark.asyncio
async def test_two_heaters_are_matched_by_identity() -> None:
    rig = Rig((6, 7))
    rig.client.identities[9] = ID_6  # heater 9 had no identity after pairing
    rig.client.paired = NoFreeAddressError("x", [_paired(8, ID_7), _paired(9)])

    result = await rig.rehome()

    assert result["moved"] == [
        {"from": 7, "to": 8, "restored": True},
        {"from": 6, "to": 9, "restored": True},
    ]
    assert result["waiting"] == []
    names = {n["addr"]: n.get("name") for n in rig.entry.data["nodes"]}
    assert names == {"bogus": None, "8": "Room 7", "9": "Room 6"}


@pytest.mark.asyncio
async def test_unrecognised_and_unpaired_heaters_are_reported() -> None:
    rig = Rig((6, 7))
    rig.client.paired = [_paired(8, b"other")]  # identity matches neither heater
    rig.client.identities[8] = RadioLinkError("closed")  # not read again (has one)

    result = await rig.rehome()

    assert result == {"moved": [], "unidentified": [8], "waiting": [6, 7]}
    addrs = [n["addr"] for n in rig.entry.data["nodes"]]
    assert addrs == ["6", "7", "bogus", "8"]  # unpaired heaters keep their nodes
    assert set(rig.entry.options["radio_restore"]) == {"6", "7"}
    summary = radio_rehome.rehome_summary(result)
    assert "Heater 8 was paired, but its old heater was not recognised" in summary
    assert "heater number 7 to pair it" in summary
    assert "may be one of the heaters that were not recognised" in summary


@pytest.mark.asyncio
async def test_identity_read_again_can_fail() -> None:
    rig = Rig((6, 7))
    rig.client.paired = [_paired(8)]
    rig.client.identities[8] = RadioLinkError("closed")
    result = await rig.rehome()
    assert result["unidentified"] == [8]


@pytest.mark.asyncio
async def test_nothing_paired_still_moves_and_says_so() -> None:
    rig = Rig()
    rig.client.paired = RadioLinkError("closed")
    result = await rig.rehome()
    assert result == {"moved": [], "unidentified": [], "waiting": [6]}
    assert rig.entry.data["network_id"] == SITE_NET.hex().upper()
    assert rp.saved_snapshot(rig.entry, 6) == SNAPSHOT  # for a later Radio pair
    summary = radio_rehome.rehome_summary(result)
    assert summary.splitlines()[1] == "No heater was paired in time."


@pytest.mark.asyncio
async def test_failed_restore_keeps_the_settings_under_the_new_number() -> None:
    rig = Rig()
    rig.client.paired = [_paired(2)]
    rig.client.restore_error = RadioCommandError("rejected")
    result = await rig.rehome()
    assert result["moved"] == [{"from": 6, "to": 2, "restored": False}]
    assert rig.entry.options["radio_restore"] == {"2": SNAPSHOT}
    assert "could not be restored" in radio_rehome.rehome_summary(result)


@pytest.mark.asyncio
async def test_heater_without_known_settings_is_moved_without_restore() -> None:
    rig = Rig()
    rig.runtime.coordinator.domain_view = SimpleNamespace(
        get_heater_state=lambda _t, _a: None
    )
    rig.client.paired = [_paired(2)]
    result = await rig.rehome()
    assert result["moved"] == [{"from": 6, "to": 2, "restored": False}]
    assert "restore" not in rig.steps()
    assert rig.entry.options.get("radio_restore", {}) == {}


@pytest.mark.asyncio
async def test_silent_heater_stops_the_move_before_any_change() -> None:
    rig = Rig((6, 7))
    rig.client.identities[7] = None
    with pytest.raises(radio_rehome.RehomeError, match="Heater 7 does not answer"):
        await rig.rehome()
    assert rig.steps() == ["identity", "identity"]
    assert rig.entry.data["network_id"] == OLD_NET


@pytest.mark.asyncio
async def test_reset_failure_keeps_the_old_network() -> None:
    rig = Rig((6, 7))
    rig.client.reset_errors[7] = RadioCommandError("no ack")
    with pytest.raises(radio_rehome.RehomeError) as err:
        await rig.rehome()
    assert "Heater 7 could not be reset" in str(err.value)
    assert "Heaters 6 are already reset" in str(err.value)
    assert "net" not in rig.steps()
    assert rig.entry.data["network_id"] == OLD_NET
    assert rp.saved_snapshot(rig.entry, 6) == SNAPSHOT

    rig = Rig()
    rig.client.reset_errors[6] = RadioCommandError("no ack")
    with pytest.raises(radio_rehome.RehomeError) as err:
        await rig.rehome()
    assert "already reset" not in str(err.value)


@pytest.mark.asyncio
async def test_refusals(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(radio_rehome.RehomeError, match="Only dialect-B"):
        await Rig(dialect=DIALECT_A).rehome()
    rig = Rig(())
    with pytest.raises(radio_rehome.RehomeError, match="no heaters to move"):
        await rig.rehome()

    async def same_net(_hass, _gateway):
        return bytes.fromhex(OLD_NET)

    monkeypatch.setattr(radio_rehome, "async_site_network_id", same_net)
    with pytest.raises(radio_rehome.RehomeError, match="already use"):
        await Rig().rehome()


# --- service ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rehome_service() -> None:
    rig = Rig()
    rig.client.paired = [_paired(2)]
    real = RadioClient("10.0.0.5", 2323, "B", [], network_id=bytes.fromhex(OLD_NET))
    for name in (
        "manual_setpoint",
        "async_read_identity",
        "async_factory_reset",
        "async_set_network_id",
        "async_pair",
        "async_restore",
    ):
        setattr(real, name, getattr(rig.client, name))
    rig.runtime.client = real
    await service.async_register_radio_pairing_services(rig.hass)
    key = (DOMAIN, service.SERVICE_RADIO_REHOME)
    assert rig.hass.services.supports_response[key] is SupportsResponse.OPTIONAL
    schema = rig.hass.services.schemas[key]
    assert schema({"entry_id": "x"}) == {"entry_id": "x", "timeout": 300}
    with pytest.raises((ValueError, KeyError)):
        schema({"entry_id": "x", "timeout": 10})
    handler = rig.hass.services.get(*key)

    result = await handler(ServiceCall({"entry_id": "radio-entry", "timeout": 60}))
    assert result["moved"] == [{"from": 6, "to": 2, "restored": True}]
    assert result["summary"].startswith("Your heaters now use")
    assert rig.client.calls[3][1] == 60

    with pytest.raises(HomeAssistantError, match="already use"):
        await handler(ServiceCall({"entry_id": "radio-entry"}))  # now on SITE_NET


# --- options flow -----------------------------------------------------------------


def _flow(rig: Rig) -> config_flow.TermoWebOptionsFlow:
    flow = config_flow.TermoWebOptionsFlow(rig.entry)
    flow.hass = rig.hass
    return flow


async def _run(flow: config_flow.TermoWebOptionsFlow) -> Any:
    form = await flow.async_step_rehome()
    assert form["step_id"] == "rehome"
    assert form["description_placeholders"]["heaters"].startswith("Room 6, ")
    first = await flow.async_step_rehome({})
    if first["type"] != "progress":
        return first
    assert (first["step_id"], first["progress_action"]) == ("rehome_run", "rehome")
    assert (await flow.async_step_rehome_run())["type"] == "progress"
    await asyncio.wait([first["progress_task"]])
    done = await flow.async_step_rehome_run()
    assert done == {"type": "progress_done", "step_id": "rehome_done"}
    return await flow.async_step_rehome_done()


@pytest.mark.asyncio
async def test_options_flow_moves_and_shows_the_summary() -> None:
    rig = Rig()
    rig.client.paired = [_paired(2)]
    flow = _flow(rig)
    result = await _run(flow)
    assert result["step_id"] == "rehome_done"
    assert "Heater 6 is now heater 2" in result["description_placeholders"]["summary"]
    closed = await flow.async_step_rehome_done({})
    assert closed["type"] == "create_entry"
    assert closed["data"]["debug"] is True


@pytest.mark.asyncio
async def test_options_flow_shows_why_the_move_stopped() -> None:
    rig = Rig()
    rig.client.reset_errors[6] = RadioCommandError("no ack")
    result = await _run(_flow(rig))
    assert "could not be reset" in result["description_placeholders"]["summary"]


@pytest.mark.asyncio
async def test_options_flow_reports_unexpected_errors() -> None:
    rig = Rig()

    async def broken(*_args):
        raise RuntimeError("boom")

    rig.client.async_factory_reset = broken
    result = await _run(_flow(rig))
    assert "Unexpected error" in result["description_placeholders"]["summary"]


@pytest.mark.asyncio
async def test_options_flow_needs_a_loaded_entry() -> None:
    rig = Rig()
    rig.hass.data.clear()
    assert await _run(_flow(rig)) == {"type": "abort", "reason": "not_loaded"}
