# ruff: noqa: D103,INP001,E402
"""Tests for pairing new radio heaters from the config flow and options flow."""

from __future__ import annotations

import asyncio
import hashlib
from types import SimpleNamespace
from typing import Any

import pytest
from conftest import _install_stubs, build_entry_runtime

_install_stubs()

from custom_components.termoweb import config_flow, radio_pairing
from custom_components.termoweb.backend.radio import (
    DIALECT_A,
    DIALECT_B,
    RadioLinkError,
)
from custom_components.termoweb.backend.radio.pairing import (
    PAIRING_DIALECTS,
    NoFreeAddressError,
    PairedHeater,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

NET = bytes.fromhex("1234")  # synthetic network id
DEV_ID = "0a0b0c0d0e0f"
SITE_NET = hashlib.sha256(f"test-instance-id:{DEV_ID}".encode()).digest()[:2]
FORM = {"host": "10.0.0.5", "port": 2323, "dialect": "auto", "network_id": ""}


def _paired(*ids: int) -> list[PairedHeater]:
    return [PairedHeater(node, SimpleNamespace()) for node in ids]


def _flow(hass: HomeAssistant) -> config_flow.TermoWebConfigFlow:
    flow = config_flow.TermoWebConfigFlow()
    flow.hass = hass
    flow.context = {}
    return flow


@pytest.fixture
def pairing(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch the gateway probe and pair_new_network; record calls."""
    state: dict[str, Any] = {"result": (DIALECT_B, _paired(2, 3)), "calls": []}

    async def fake_probe(host: str, port: int) -> str:
        return DEV_ID

    async def fake_pair(host, port, network_id, **kwargs):
        state["calls"].append((host, port, network_id, kwargs))
        result = state["result"]
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(config_flow, "probe_gateway", fake_probe)
    monkeypatch.setattr(config_flow, "pair_new_network", fake_pair)
    return state


async def _pair(flow: config_flow.TermoWebConfigFlow, menu: Any) -> Any:
    """Choose pairing from the menu, start it and return the finish result."""
    assert menu["type"] == "menu" and menu["step_id"] == "radio_method"
    form = await flow.async_step_radio_pair()
    assert form["type"] == "form" and form["step_id"] == "radio_pair"
    first = await flow.async_step_radio_pair({})
    assert first["type"] == "progress"
    assert first["step_id"] == "radio_pair_run"
    assert first["progress_action"] == "radio_pair"
    await asyncio.wait([first["progress_task"]])
    done = await flow.async_step_radio_pair_run()
    assert done == {"type": "progress_done", "step_id": "radio_finish"}
    return await flow.async_step_radio_finish()


# --- pair_radio ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_pair_radio_picks_the_dialects_to_listen_for(pairing) -> None:
    sighting, heaters = await config_flow.pair_radio("gw", 1, "auto", NET)
    assert sighting.dialect is DIALECT_B and sighting.network_id == NET
    assert sighting.sources == frozenset({2, 3}) and sorted(heaters) == [2, 3]
    await config_flow.pair_radio("gw", 1, "auto", NET, dialect_capable=False)
    await config_flow.pair_radio("gw", 1, "B", NET)
    dialects = [call[3]["dialects"] for call in pairing["calls"]]
    assert dialects == [PAIRING_DIALECTS, (DIALECT_A,), (DIALECT_B,)]


@pytest.mark.asyncio
async def test_pair_radio_without_a_heater_is_an_error(pairing) -> None:
    pairing["result"] = (None, [])
    with pytest.raises(config_flow.RadioSetupError, match="no_heaters_paired"):
        await config_flow.pair_radio("gw", 1, "auto", NET)


@pytest.mark.asyncio
async def test_site_network_id_hashes_instance_and_gateway() -> None:
    assert await radio_pairing.async_site_network_id(HomeAssistant(), DEV_ID) == (
        SITE_NET
    )


# --- config flow ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_pairing_creates_an_entry_on_the_site_network(pairing) -> None:
    flow = _flow(HomeAssistant())
    result = await _pair(flow, await flow.async_step_radio(dict(FORM)))

    assert result["type"] == "create_entry"
    assert result["data"]["dialect"] == "B"
    assert result["data"]["network_id"] == SITE_NET.hex().upper()
    assert result["data"]["nodes"] == [
        {"type": "htr", "addr": "2", "name": "Heater 2"},
        {"type": "htr", "addr": "3", "name": "Heater 3"},
    ]
    host, port, network_id, kwargs = pairing["calls"][0]
    assert (host, port, network_id) == ("10.0.0.5", 2323, SITE_NET)
    assert kwargs["dialects"] == PAIRING_DIALECTS


@pytest.mark.asyncio
async def test_pairing_uses_a_network_id_the_user_entered(pairing) -> None:
    pairing["result"] = (DIALECT_A, _paired(2))
    flow = _flow(HomeAssistant())
    menu = await flow.async_step_radio({**FORM, "network_id": "1234"})
    result = await _pair(flow, menu)
    assert result["data"]["network_id"] == "1234"
    assert result["data"]["dialect"] == "A"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "reason"),
    [
        ((None, []), "no_heaters_paired"),
        (RadioLinkError("closed"), "cannot_connect_radio"),
        (RuntimeError("boom"), "unknown"),
    ],
)
async def test_pairing_errors_return_to_the_pairing_form(
    pairing, error, reason
) -> None:
    pairing["result"] = error
    flow = _flow(HomeAssistant())
    result = await _pair(flow, await flow.async_step_radio(dict(FORM)))
    assert result["type"] == "form" and result["step_id"] == "radio_pair"
    assert result["errors"] == {"base": reason}


@pytest.mark.asyncio
async def test_pairing_progress_while_running(pairing, monkeypatch) -> None:
    release = asyncio.Event()

    async def slow(*_args, **_kwargs):
        await release.wait()
        return DIALECT_B, _paired(2)

    monkeypatch.setattr(config_flow, "pair_new_network", slow)
    flow = _flow(HomeAssistant())
    await flow.async_step_radio(dict(FORM))
    first = await flow.async_step_radio_pair({})
    again = await flow.async_step_radio_pair_run()
    assert again["type"] == "progress"
    assert again["progress_task"] is first["progress_task"]
    release.set()
    await asyncio.wait([first["progress_task"]])
    assert (await flow.async_step_radio_pair_run())["type"] == "progress_done"


@pytest.mark.asyncio
async def test_nanocul_pairing_listens_for_dialect_a_only(pairing, monkeypatch) -> None:
    async def fake_probe_nanocul(device, usb_serial=None):
        return "usb-stick", False

    monkeypatch.setattr(config_flow, "probe_nanocul", fake_probe_nanocul)
    pairing["result"] = (DIALECT_A, _paired(2))
    flow = _flow(HomeAssistant())
    menu = await flow.async_step_nanocul_manual(
        {"device": "/dev/ttyUSB0", "dialect": "auto", "network_id": ""}
    )
    result = await _pair(flow, menu)
    assert result["data"]["radio_type"] == "nanocul"
    _host, port, network_id, kwargs = pairing["calls"][0]
    assert port == 0 and kwargs["dialects"] == (DIALECT_A,)
    seed = b"test-instance-id:usb-stick"
    assert network_id == hashlib.sha256(seed).digest()[:2]


# --- options flow ---------------------------------------------------------------


class Options:
    """A loaded radio entry and its options flow; the client's pairing is scripted."""

    def __init__(self, loaded: bool = True) -> None:
        self.hass = HomeAssistant()
        self.entry = ConfigEntry(
            "radio-entry",
            data={
                "brand": "radio",
                "network_id": "1234",
                "nodes": [{"type": "htr", "addr": "6", "name": "Heater 6"}],
            },
            options={"debug": True},
        )
        self.hass.config_entries.add_entry(self.entry)
        self.result: Any = _paired(2)
        self.calls: list[Any] = []

        async def pair(window_s, **kwargs):
            self.calls.append((window_s, kwargs))
            if isinstance(self.result, Exception):
                raise self.result
            return self.result

        if loaded:
            build_entry_runtime(
                hass=self.hass,
                entry_id=self.entry.entry_id,
                client=SimpleNamespace(async_pair=pair),
                config_entry=self.entry,
                brand="radio",
            )
        self.flow = config_flow.TermoWebOptionsFlow(self.entry)
        self.flow.hass = self.hass

    async def run(self) -> Any:
        menu = await self.flow.async_step_init()
        assert menu["menu_options"] == ["settings", "pair_heaters", "rehome"]
        form = await self.flow.async_step_pair_heaters()
        assert form["step_id"] == "pair_heaters" and form["errors"] == {}
        first = await self.flow.async_step_pair_heaters({})
        if first["type"] != "progress":
            return first
        assert first["step_id"] == "pair_run"
        assert first["progress_action"] == "pair_heaters"
        assert (await self.flow.async_step_pair_run())["type"] == "progress"
        await asyncio.wait([first["progress_task"]])
        done = await self.flow.async_step_pair_run()
        assert done == {"type": "progress_done", "step_id": "pair_done"}
        return await self.flow.async_step_pair_done()


@pytest.mark.asyncio
async def test_options_pairing_adds_heaters_and_reloads() -> None:
    rig = Options()
    result = await rig.run()
    assert result == {"type": "create_entry", "title": "", "data": {"debug": True}}
    assert rig.calls == [(300.0, {"idle_stop_s": 60.0})]
    assert rig.entry.data["nodes"][-1] == {
        "type": "htr",
        "addr": "2",
        "name": "Heater 2",
    }
    assert rig.hass.config_entries.scheduled_reloads == ["radio-entry"]


@pytest.mark.asyncio
async def test_options_pairing_keeps_heaters_paired_before_ids_ran_out() -> None:
    rig = Options()
    rig.result = NoFreeAddressError("no free radio id", _paired(0x41))
    result = await rig.run()
    assert result["type"] == "create_entry"
    assert rig.entry.data["nodes"][-1]["addr"] == "65"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "reason"),
    [
        ([], "no_heaters_paired"),
        (NoFreeAddressError("no free radio id"), "no_free_address"),
        (RadioLinkError("closed"), "cannot_connect_radio"),
        (RuntimeError("boom"), "unknown"),
    ],
)
async def test_options_pairing_errors_show_the_form_again(error, reason) -> None:
    rig = Options()
    rig.result = error
    result = await rig.run()
    assert result["type"] == "form" and result["step_id"] == "pair_heaters"
    assert result["errors"] == {"base": reason}
    assert rig.hass.config_entries.scheduled_reloads == []


@pytest.mark.asyncio
async def test_options_pairing_needs_a_loaded_entry() -> None:
    rig = Options(loaded=False)
    assert await rig.run() == {"type": "abort", "reason": "not_loaded"}
