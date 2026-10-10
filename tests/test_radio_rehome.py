"""Moving a radio entry's heaters onto the site network: helper, service, options."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.exceptions import HomeAssistantError
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.termoweb import radio_pairing as rp, radio_rehome
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
from tests.fakes.radio_setup import add_radio_entry, record_reloads

OLD_NET = "1234"  # synthetic legacy network id
ENTRY_ID = "radio-entry"
ID_6 = bytes(range(16))
ID_7 = bytes(range(1, 17))
STATE = HeaterState(mode="auto", stemp="18.5", ptemp=["7", "17", "20"])
SNAPSHOT = {"mode": "auto", "ptemp": [7.0, 17.0, 20.0], "manual_stemp": 20.0}


class Radio(RadioClient):
    """A real RadioClient whose radio exchanges are scripted and recorded."""

    def __init__(self, dialect: str = "B") -> None:
        """Start with two answering heaters and nothing paired."""
        super().__init__(
            "10.0.0.5", 2323, dialect, [], network_id=bytes.fromhex(OLD_NET)
        )
        self.calls: list[tuple] = []
        self.identities: dict[int, Any] = {6: ID_6, 7: ID_7}
        self.reset_errors: dict[int, Exception] = {}
        self.paired: Any = []
        self.restore_error: Exception | None = None

    def manual_setpoint(self, addr: int) -> float | None:
        """Every heater's last manual target is 20 °C."""
        return 20.0

    async def async_read_identity(self, addr: int) -> bytes:
        """Return the scripted identity, or raise for a silent heater."""
        self.calls.append(("identity", addr))
        value = self.identities.get(addr)
        if isinstance(value, Exception) or value is None:
            raise value or RadioCommandError(f"heater {addr} silent")
        return value

    async def async_factory_reset(self, addr: int) -> None:
        """Record the reset; raise the scripted error for ``addr``."""
        self.calls.append(("reset", addr))
        if addr in self.reset_errors:
            raise self.reset_errors[addr]

    async def async_set_network_id(self, network_id: bytes) -> None:
        """Record the network move."""
        self.calls.append(("net", network_id))

    async def async_pair(self, window_s: float, **kwargs: Any) -> Any:
        """Return (or raise) the scripted pairing result."""
        self.calls.append(("pair", window_s, kwargs))
        if isinstance(self.paired, Exception):
            raise self.paired
        return self.paired

    async def async_restore(self, addr: int, **kwargs: Any) -> None:
        """Record the restore; raise the scripted error."""
        self.calls.append(("restore", addr, kwargs))
        if self.restore_error is not None:
            raise self.restore_error


class Rig:
    """A loaded radio entry with heaters ``addrs`` and a scripted client."""

    def __init__(
        self,
        hass: HomeAssistant,
        monkeypatch: pytest.MonkeyPatch,
        addrs: tuple[int, ...] = (6,),
        dialect: str = "B",
        state: HeaterState | None = STATE,
    ) -> None:
        """Add the entry and record reloads."""
        self.hass = hass
        nodes = [{"type": "htr", "addr": str(a), "name": f"Room {a}"} for a in addrs]
        nodes.append({"type": "htr", "addr": "bogus"})
        self.client = Radio(dialect)
        states = {str(a): state for a in addrs} if state is not None else {}
        self.runtime = add_radio_entry(
            hass,
            client=self.client,
            entry_id=ENTRY_ID,
            data={
                "brand": "radio",
                "dialect": "B",
                "network_id": OLD_NET,
                "nodes": nodes,
            },
            options={"energy_history_imported": True},
            states=states,
        )
        self.entry = self.runtime.config_entry
        self.reloads = record_reloads(monkeypatch, hass)

    async def rehome(self) -> dict[str, Any]:
        """Run the move with a 120 s pairing window."""
        return await radio_rehome.async_rehome(self.hass, self.runtime, 120)

    def steps(self) -> list[str]:
        """Return the radio steps taken, in order."""
        return [call[0] for call in self.client.calls]


def _paired(node_id: int, identity: bytes | None = None) -> PairedHeater:
    return PairedHeater(node_id, SimpleNamespace(), identity)


@pytest.fixture
def rig(hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch) -> Rig:
    """Return a loaded radio entry with heater 6."""
    return Rig(hass, monkeypatch)


async def _site_net(hass: HomeAssistant) -> bytes:
    """Return this installation's network id for the entry's gateway."""
    return await rp.async_site_network_id(hass, "dev")


# --- async_rehome ----------------------------------------------------------------


async def test_single_heater_moves_restores_and_keeps_its_name(rig: Rig) -> None:
    """One heater: reset, move the gateway, pair, restore, rename the node."""
    rig.client.paired = [_paired(2)]  # no identity needed for one heater
    site_net = await _site_net(rig.hass)

    result = await rig.rehome()

    assert result == {
        "moved": [{"from": 6, "to": 2, "restored": True}],
        "unidentified": [],
        "waiting": [],
    }
    assert rig.client.calls == [
        ("identity", 6),
        ("reset", 6),
        ("net", site_net),
        ("pair", 120, {"max_heaters": 1, "idle_stop_s": 60.0}),
        ("restore", 2, SNAPSHOT),
    ]
    assert rig.entry.data["network_id"] == site_net.hex().upper()
    assert rig.entry.data["nodes"] == [
        {"type": "htr", "addr": "bogus"},
        {"type": "htr", "addr": "2", "name": "Room 6"},
    ]
    assert rig.entry.options["radio_restore"] == {}
    assert rig.entry.options["energy_history_imported"] is True
    assert rig.reloads == [ENTRY_ID]
    summary = radio_rehome.rehome_summary(result)
    assert "Heater 6 is now heater 2. Its settings were restored." in summary


async def test_two_heaters_are_matched_by_identity(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Paired heaters are matched to their old numbers by identity."""
    rig = Rig(hass, monkeypatch, (6, 7))
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


async def test_unrecognised_and_unpaired_heaters_are_reported(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unknown identities are added as new heaters; unpaired ones keep their node."""
    rig = Rig(hass, monkeypatch, (6, 7))
    rig.client.paired = [_paired(8, b"other")]  # identity matches neither heater
    rig.client.identities[8] = RadioLinkError("closed")  # not read again (has one)

    result = await rig.rehome()

    assert result == {"moved": [], "unidentified": [8], "waiting": [6, 7]}
    addrs = [n["addr"] for n in rig.entry.data["nodes"]]
    assert addrs == ["6", "7", "bogus", "8"]
    assert set(rig.entry.options["radio_restore"]) == {"6", "7"}
    summary = radio_rehome.rehome_summary(result)
    assert "Heater 8 was paired, but its old heater was not recognised" in summary
    assert "heater number 7 to pair it" in summary
    assert "may be one of the heaters that were not recognised" in summary


async def test_identity_read_again_can_fail(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A paired heater whose identity cannot be read is unidentified."""
    rig = Rig(hass, monkeypatch, (6, 7))
    rig.client.paired = [_paired(8)]
    rig.client.identities[8] = RadioLinkError("closed")
    assert (await rig.rehome())["unidentified"] == [8]


async def test_nothing_paired_still_moves_and_says_so(rig: Rig) -> None:
    """The gateway moves even when no heater pairs; settings stay saved."""
    rig.client.paired = RadioLinkError("closed")
    result = await rig.rehome()
    assert result == {"moved": [], "unidentified": [], "waiting": [6]}
    assert rig.entry.data["network_id"] == (await _site_net(rig.hass)).hex().upper()
    assert rp.saved_snapshot(rig.entry, 6) == SNAPSHOT  # for a later Radio pair
    summary = radio_rehome.rehome_summary(result)
    assert summary.splitlines()[1] == "No heater was paired in time."


async def test_failed_restore_keeps_the_settings_under_the_new_number(
    rig: Rig,
) -> None:
    """A failed restore keeps the snapshot under the heater's new number."""
    rig.client.paired = [_paired(2)]
    rig.client.restore_error = RadioCommandError("rejected")
    result = await rig.rehome()
    assert result["moved"] == [{"from": 6, "to": 2, "restored": False}]
    assert rig.entry.options["radio_restore"] == {"2": SNAPSHOT}
    assert "could not be restored" in radio_rehome.rehome_summary(result)


async def test_heater_without_known_settings_is_moved_without_restore(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No known settings: the heater moves but nothing is restored."""
    rig = Rig(hass, monkeypatch, state=None)
    rig.client.paired = [_paired(2)]
    result = await rig.rehome()
    assert result["moved"] == [{"from": 6, "to": 2, "restored": False}]
    assert "restore" not in rig.steps()
    assert rig.entry.options.get("radio_restore", {}) == {}


async def test_silent_heater_stops_the_move_before_any_change(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every heater must answer before anything is reset."""
    rig = Rig(hass, monkeypatch, (6, 7))
    rig.client.identities[7] = None
    with pytest.raises(radio_rehome.RehomeError, match="Heater 7 does not answer"):
        await rig.rehome()
    assert rig.steps() == ["identity", "identity"]
    assert rig.entry.data["network_id"] == OLD_NET


async def test_reset_failure_keeps_the_old_network(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed reset stops before the network move; all settings stay saved."""
    rig = Rig(hass, monkeypatch, (6, 7))
    rig.client.reset_errors[7] = RadioCommandError("no ack")
    with pytest.raises(radio_rehome.RehomeError) as err:
        await rig.rehome()
    assert "Heater 7 could not be reset" in str(err.value)
    assert "If heater 7 did reset, its settings are saved" in str(err.value)
    assert "Heaters 6 are already reset" in str(err.value)
    assert "net" not in rig.steps()
    assert rig.entry.data["network_id"] == OLD_NET
    assert rp.saved_snapshot(rig.entry, 6) == SNAPSHOT
    # Heater 7 may have reset before its verdict was lost: its settings are kept.
    assert rp.saved_snapshot(rig.entry, 7) == SNAPSHOT


async def test_first_reset_failure_mentions_no_reset_heaters(rig: Rig) -> None:
    """When the first reset fails no heater is reported as already reset."""
    rig.client.reset_errors[6] = RadioCommandError("no ack")
    with pytest.raises(radio_rehome.RehomeError) as err:
        await rig.rehome()
    assert "already reset" not in str(err.value)


async def test_refuses_dialect_a(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dialect-A heaters cannot be reset over the radio."""
    with pytest.raises(radio_rehome.RehomeError, match="Only dialect-B"):
        await Rig(hass, monkeypatch, dialect="A").rehome()


async def test_refuses_an_entry_without_heaters(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """There is nothing to move without heaters."""
    with pytest.raises(radio_rehome.RehomeError, match="no heaters to move"):
        await Rig(hass, monkeypatch, ()).rehome()


async def test_refuses_an_entry_already_on_the_site_network(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An entry already on its own network is not moved again."""

    async def same_net(_hass: HomeAssistant, _gateway: str) -> bytes:
        return bytes.fromhex(OLD_NET)

    monkeypatch.setattr(radio_rehome, "async_site_network_id", same_net)
    with pytest.raises(radio_rehome.RehomeError, match="already use"):
        await rig.rehome()


# --- service ---------------------------------------------------------------------


async def test_rehome_service(rig: Rig) -> None:
    """The service runs the move and returns a summary; a second run is refused."""
    rig.client.paired = [_paired(2)]
    await service.async_register_radio_pairing_services(rig.hass)

    result = await rig.hass.services.async_call(
        DOMAIN,
        service.SERVICE_RADIO_REHOME,
        {"entry_id": ENTRY_ID, "timeout": 60},
        blocking=True,
        return_response=True,
    )
    assert result["moved"] == [{"from": 6, "to": 2, "restored": True}]
    assert result["summary"].startswith("Your heaters now use")
    assert rig.client.calls[3][1] == 60

    with pytest.raises(HomeAssistantError, match="already use"):
        await rig.hass.services.async_call(
            DOMAIN,
            service.SERVICE_RADIO_REHOME,
            {"entry_id": ENTRY_ID},  # now on the site network
            blocking=True,
            return_response=True,
        )


# --- options flow -----------------------------------------------------------------


async def _run_options(hass: HomeAssistant, entry_id: str) -> Any:
    """Open the options, pick rehome, submit and return the step after progress."""
    menu = await hass.config_entries.options.async_init(entry_id)
    assert menu["type"] is FlowResultType.MENU
    assert menu["menu_options"] == ["settings", "pair_heaters", "rehome"]
    form = await hass.config_entries.options.async_configure(
        menu["flow_id"], {"next_step_id": "rehome"}
    )
    assert form["step_id"] == "rehome"
    assert form["description_placeholders"]["heaters"].startswith("Room 6, ")
    result = await hass.config_entries.options.async_configure(form["flow_id"], {})
    if result["type"] is not FlowResultType.SHOW_PROGRESS:
        return result
    assert (result["step_id"], result["progress_action"]) == ("rehome_run", "rehome")
    await hass.async_block_till_done()
    return await hass.config_entries.options.async_configure(result["flow_id"])


async def test_options_flow_moves_and_shows_the_summary(rig: Rig) -> None:
    """The summary is shown; confirming closes the flow and keeps the options."""
    rig.client.paired = [_paired(2)]
    result = await _run_options(rig.hass, ENTRY_ID)
    assert result["step_id"] == "rehome_done"
    assert "Heater 6 is now heater 2" in result["description_placeholders"]["summary"]
    closed = await rig.hass.config_entries.options.async_configure(
        result["flow_id"], {}
    )
    assert closed["type"] is FlowResultType.CREATE_ENTRY
    assert closed["data"]["energy_history_imported"] is True


async def test_options_flow_shows_why_the_move_stopped(rig: Rig) -> None:
    """A refused move is explained in the summary."""
    rig.client.reset_errors[6] = RadioCommandError("no ack")
    result = await _run_options(rig.hass, ENTRY_ID)
    assert "could not be reset" in result["description_placeholders"]["summary"]


async def test_options_flow_shows_an_immediate_refusal(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A move refused at once (dialect A) still shows why instead of closing."""
    rig = Rig(hass, monkeypatch, dialect="A")
    result = await _run_options(rig.hass, ENTRY_ID)
    assert result["step_id"] == "rehome_done"
    assert "Only dialect-B" in result["description_placeholders"]["summary"]
    assert rig.client.calls == []


async def test_options_flow_reports_unexpected_errors(rig: Rig) -> None:
    """An unexpected error is summarised instead of crashing the flow."""
    rig.client.reset_errors[6] = RuntimeError("boom")
    result = await _run_options(rig.hass, ENTRY_ID)
    assert "Unexpected error" in result["description_placeholders"]["summary"]


async def test_options_flow_needs_a_loaded_entry(hass: HomeAssistant) -> None:
    """Without a running radio the move cannot start."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            "brand": "radio",
            "nodes": [{"type": "htr", "addr": "6", "name": "Room 6"}, {"addr": "x"}],
        },
    )
    entry.add_to_hass(hass)
    result = await _run_options(hass, entry.entry_id)
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "not_loaded"


async def test_options_flow_shows_progress_while_moving(rig: Rig) -> None:
    """A move that is still running shows progress, then the summary."""
    release = asyncio.Event()
    real_pair = rig.client.async_pair

    async def slow(window_s: float, **kwargs: Any) -> Any:
        await release.wait()
        return await real_pair(window_s, **kwargs)

    rig.client.paired = [_paired(2)]
    rig.client.async_pair = slow
    menu = await rig.hass.config_entries.options.async_init(ENTRY_ID)
    form = await rig.hass.config_entries.options.async_configure(
        menu["flow_id"], {"next_step_id": "rehome"}
    )
    result = await rig.hass.config_entries.options.async_configure(form["flow_id"], {})
    assert result["type"] is FlowResultType.SHOW_PROGRESS
    assert result["progress_action"] == "rehome"
    release.set()
    await rig.hass.async_block_till_done()
    result = await rig.hass.config_entries.options.async_configure(result["flow_id"])
    assert result["step_id"] == "rehome_done"
    assert "Heater 6 is now heater 2" in result["description_placeholders"]["summary"]
