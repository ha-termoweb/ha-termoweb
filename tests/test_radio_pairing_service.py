# ruff: noqa: D103,INP001,E402
"""Tests for the radio_factory_reset and radio_pair services and their helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from conftest import _install_stubs, build_entry_runtime

_install_stubs()

from custom_components.termoweb import radio_pairing as rp
from custom_components.termoweb.backend.radio.link import RadioLinkError
from custom_components.termoweb.backend.radio.pairing import (
    NoFreeAddressError,
    PairedHeater,
)
from custom_components.termoweb.backend.radio_client import (
    RadioClient,
    RadioCommandError,
    RadioUnsupportedError,
)
from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.domain.state import HeaterState
from custom_components.termoweb.services import radio_pairing as service
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError

NET = bytes.fromhex("1234")  # synthetic network id
ENTRY_ID = "radio-entry"
PROG = [0] * 7 + [2] * 14 + [1] * 3
STATE = HeaterState(
    mode="manual", stemp="21.5", ptemp=["7.0", "17.0", "20.0"], prog=PROG * 7
)
SNAPSHOT = {
    "mode": "manual",
    "stemp": 21.5,
    "ptemp": [7.0, 17.0, 20.0],
    "prog": PROG * 7,
}


class Rig:
    """A loaded radio entry whose client records the pairing calls."""

    def __init__(self, dialect: str = "B") -> None:
        self.hass = HomeAssistant()
        self.entry = ConfigEntry(
            ENTRY_ID,
            data={
                "brand": "radio",
                "nodes": [
                    {"type": "htr", "addr": "6", "name": "Heater 6"},
                    {"type": "htr", "addr": "bogus"},
                ],
            },
        )
        self.hass.config_entries.add_entry(self.entry)
        self.client = RadioClient("10.0.0.5", 2323, dialect, [], network_id=NET)
        self.states: dict[str, Any] = {"6": STATE}
        self.refreshed: list[Any] = []
        self.calls: list[tuple[str, Any]] = []
        self.paired: list[PairedHeater] | Exception = [_paired(6)]
        self.reset_error: Exception | None = None
        self.restore_error: Exception | None = None

        async def refresh(node):
            self.refreshed.append(node)

        view = SimpleNamespace(
            get_heater_state=lambda node_type, addr: self.states.get(addr)
        )
        coordinator = SimpleNamespace(
            domain_view=view, async_refresh_heater=refresh, data={}
        )
        build_entry_runtime(
            hass=self.hass,
            entry_id=ENTRY_ID,
            client=self.client,
            config_entry=self.entry,
            coordinator=coordinator,
            brand="radio",
        )

        async def pair(window_s, **kwargs):
            self.calls.append(("pair", (window_s, kwargs)))
            if isinstance(self.paired, Exception):
                raise self.paired
            return self.paired

        async def reset(addr):
            self.calls.append(("reset", addr))
            if self.reset_error is not None:
                raise self.reset_error

        async def restore(addr, **kwargs):
            self.calls.append(("restore", (addr, kwargs)))
            if self.restore_error is not None:
                raise self.restore_error

        self.client.async_pair = pair
        self.client.async_factory_reset = reset
        self.client.async_restore = restore

    async def handler(self, name: str):
        await service.async_register_radio_pairing_services(self.hass)
        return self.hass.services.get(DOMAIN, name)


def _paired(node_id: int) -> PairedHeater:
    return PairedHeater(node_id, SimpleNamespace())


# --- helpers ------------------------------------------------------------------


def test_settings_snapshot_keeps_only_restorable_values() -> None:
    assert rp.settings_snapshot(None) is None
    assert rp.settings_snapshot(HeaterState()) is None
    assert rp.settings_snapshot(STATE) == SNAPSHOT
    override = HeaterState(mode="modified_auto", stemp="24", ptemp=["7", "x", "9"])
    assert rp.settings_snapshot(override) == {"mode": "auto"}
    odd = HeaterState(mode="eco", stemp=None, ptemp=["7", "8"], prog=[0] * 24)
    assert rp.settings_snapshot(odd) is None
    assert rp.settings_snapshot(HeaterState(mode="heat")) == {"mode": "manual"}


def test_store_snapshot_and_add_nodes() -> None:
    rig = Rig()
    hass, entry = rig.hass, rig.entry
    rp.store_snapshot(hass, entry, 6, None)  # nothing saved: no update
    assert hass.config_entries.updated_entries == []
    rp.store_snapshot(hass, entry, 6, {"mode": "off"})
    assert rp.saved_snapshot(entry, 6) == {"mode": "off"}
    assert rp.saved_snapshot(entry, 7) is None
    rp.store_snapshot(hass, entry, 6, None)
    assert entry.options["radio_restore"] == {}

    assert rp.add_nodes(hass, entry, [6]) is False
    assert rp.add_nodes(hass, entry, [9, 3, 9]) is True
    assert entry.data["nodes"][-2:] == [rp.radio_node(3), rp.radio_node(9)]
    assert hass.config_entries.scheduled_reloads == [ENTRY_ID]


# --- registration and validation ----------------------------------------------


@pytest.mark.asyncio
async def test_registers_once_with_schemas() -> None:
    rig = Rig()
    handler = await rig.handler(service.SERVICE_RADIO_PAIR)
    await service.async_register_radio_pairing_services(rig.hass)
    assert rig.hass.services.get(DOMAIN, service.SERVICE_RADIO_PAIR) is handler
    services = rig.hass.services
    for name in (service.SERVICE_RADIO_PAIR, service.SERVICE_RADIO_FACTORY_RESET):
        assert services.supports_response[(DOMAIN, name)] is SupportsResponse.OPTIONAL
    pair_schema = services.schemas[(DOMAIN, service.SERVICE_RADIO_PAIR)]
    assert pair_schema({"entry_id": "x"}) == {
        "entry_id": "x",
        "restore": True,
        "timeout": 300,
    }
    assert pair_schema({"entry_id": "x", "heater": "6"})["heater"] == 6
    reset_schema = services.schemas[(DOMAIN, service.SERVICE_RADIO_FACTORY_RESET)]
    assert reset_schema({"entry_id": "x", "heater": 6}) == {
        "entry_id": "x",
        "heater": 6,
    }
    for bad in (
        {"entry_id": "x", "timeout": 29},
        {"entry_id": "x", "timeout": 601},
        {"entry_id": "x", "heater": 0},
        {"entry_id": "x", "heater": 255},
    ):
        with pytest.raises((ValueError, KeyError)):  # vol.Invalid in real HA
            pair_schema(bad)
    with pytest.raises((ValueError, KeyError)):
        reset_schema({"entry_id": "x"})


@pytest.mark.asyncio
async def test_services_validate_the_entry_and_heater() -> None:
    rig = Rig()
    reset = await rig.handler(service.SERVICE_RADIO_FACTORY_RESET)
    pair = await rig.handler(service.SERVICE_RADIO_PAIR)
    with pytest.raises(ServiceValidationError, match="No loaded TermoWeb entry"):
        await pair(ServiceCall({"entry_id": "missing"}))
    build_entry_runtime(hass=rig.hass, entry_id="cloud", client=object())
    with pytest.raises(ServiceValidationError, match="does not use a radio"):
        await reset(ServiceCall({"entry_id": "cloud", "heater": 6}))
    with pytest.raises(ServiceValidationError, match="Heater 7 is not part"):
        await reset(ServiceCall({"entry_id": ENTRY_ID, "heater": 7}))
    with pytest.raises(ServiceValidationError, match="Heater 7 is not part"):
        await pair(ServiceCall({"entry_id": ENTRY_ID, "heater": 7}))
    assert rig.calls == []


# --- radio_factory_reset --------------------------------------------------------


@pytest.mark.asyncio
async def test_factory_reset_saves_the_settings_first() -> None:
    rig = Rig()
    reset = await rig.handler(service.SERVICE_RADIO_FACTORY_RESET)
    result = await reset(ServiceCall({"entry_id": ENTRY_ID, "heater": 6}))
    assert result == {"heater": 6, "settings_saved": True}
    assert rig.calls == [("reset", 6)]
    assert rp.saved_snapshot(rig.entry, 6) == SNAPSHOT

    rig.states.clear()  # Home Assistant knows nothing about the heater
    rig.entry.options = {}
    result = await reset(ServiceCall({"entry_id": ENTRY_ID, "heater": 6}))
    assert result == {"heater": 6, "settings_saved": False}
    assert rp.saved_snapshot(rig.entry, 6) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "raised", "match"),
    [
        (
            RadioUnsupportedError("Factory reset in dialect A"),
            ServiceValidationError,
            "dialect A",
        ),
        (
            RadioCommandError("heater 6 rejected"),
            HomeAssistantError,
            "Factory reset failed",
        ),
        (RadioLinkError("closed"), HomeAssistantError, "Factory reset failed"),
    ],
)
async def test_factory_reset_errors_save_nothing(error, raised, match) -> None:
    rig = Rig()
    rig.reset_error = error
    reset = await rig.handler(service.SERVICE_RADIO_FACTORY_RESET)
    with pytest.raises(raised, match=match):
        await reset(ServiceCall({"entry_id": ENTRY_ID, "heater": 6}))
    assert rp.saved_snapshot(rig.entry, 6) is None


# --- radio_pair -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_repair_restores_the_settings_saved_before_the_reset() -> None:
    rig = Rig()
    saved = {"mode": "off", "ptemp": [5.0, 16.0, 19.0]}
    rp.store_snapshot(rig.hass, rig.entry, 6, saved)
    rig.states["6"] = HeaterState(mode="off", ptemp=["5.0", "17.0", "19.0"])
    pair = await rig.handler(service.SERVICE_RADIO_PAIR)

    result = await pair(ServiceCall({"entry_id": ENTRY_ID, "heater": 6, "timeout": 60}))

    assert result == {"heater": 6, "added": False, "restored": True}
    assert rig.calls == [
        ("pair", (60, {"wanted_id": 6, "max_heaters": 1})),
        ("restore", (6, saved)),
    ]
    assert rp.saved_snapshot(rig.entry, 6) is None
    assert rig.refreshed == [("htr", "6")]


@pytest.mark.asyncio
async def test_repair_without_a_saved_snapshot_uses_the_current_state() -> None:
    rig = Rig()
    pair = await rig.handler(service.SERVICE_RADIO_PAIR)
    result = await pair(ServiceCall({"entry_id": ENTRY_ID, "heater": 6}))
    assert result["restored"] is True
    assert rig.calls[1] == ("restore", (6, SNAPSHOT))


@pytest.mark.asyncio
async def test_repair_can_skip_the_restore() -> None:
    rig = Rig()
    pair = await rig.handler(service.SERVICE_RADIO_PAIR)
    call = ServiceCall({"entry_id": ENTRY_ID, "heater": 6, "restore": False})
    assert (await pair(call))["restored"] is False
    rig.states.clear()
    assert (await pair(ServiceCall({"entry_id": ENTRY_ID, "heater": 6})))[
        "restored"
    ] is False
    assert [name for name, _ in rig.calls] == ["pair", "pair"]


@pytest.mark.asyncio
async def test_restore_failure_is_reported_and_keeps_the_snapshot() -> None:
    rig = Rig()
    rig.restore_error = RadioCommandError("heater 6 rejected")
    rp.store_snapshot(rig.hass, rig.entry, 6, {"mode": "off"})
    pair = await rig.handler(service.SERVICE_RADIO_PAIR)
    with pytest.raises(HomeAssistantError, match="is paired, but restoring"):
        await pair(ServiceCall({"entry_id": ENTRY_ID, "heater": 6}))
    assert rp.saved_snapshot(rig.entry, 6) == {"mode": "off"}


@pytest.mark.asyncio
async def test_pairing_a_new_heater_adds_it_and_reloads() -> None:
    rig = Rig()
    rig.paired = [_paired(2)]
    pair = await rig.handler(service.SERVICE_RADIO_PAIR)
    result = await pair(ServiceCall({"entry_id": ENTRY_ID}))
    assert result == {"heater": 2, "added": True, "restored": False}
    assert rig.calls == [("pair", (300, {"wanted_id": None, "max_heaters": 1}))]
    assert rig.entry.data["nodes"][-1] == rp.radio_node(2)
    assert rig.hass.config_entries.scheduled_reloads == [ENTRY_ID]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("paired", "match"),
    [
        ([], "No heater was paired within 300 s"),
        (NoFreeAddressError("no free radio id"), "Radio pairing failed: no free"),
        (RadioLinkError("closed"), "Radio pairing failed: closed"),
    ],
)
async def test_pairing_failures(paired, match) -> None:
    rig = Rig()
    rig.paired = paired
    pair = await rig.handler(service.SERVICE_RADIO_PAIR)
    with pytest.raises(HomeAssistantError, match=match):
        await pair(ServiceCall({"entry_id": ENTRY_ID}))
    assert rig.hass.config_entries.scheduled_reloads == []
