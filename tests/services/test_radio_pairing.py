"""The radio_factory_reset and radio_pair services, called through Home Assistant."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from homeassistant.core import HomeAssistant, SupportsResponse
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
import pytest
import voluptuous as vol

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
from tests.fakes.radio_setup import add_radio_entry, record_reloads
from tests.fakes.runtime import build_entry_runtime

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
NODES = [{"type": "htr", "addr": "6", "name": "Heater 6"}, {"type": "htr", "addr": "x"}]


class Rig:
    """A loaded radio entry whose client records pairing, reset and restore."""

    def __init__(
        self,
        hass: HomeAssistant,
        monkeypatch: pytest.MonkeyPatch,
        state: HeaterState | None = STATE,
    ) -> None:
        """Add the entry, wire the scripted client and register the services."""
        self.hass = hass
        self.client = RadioClient("10.0.0.5", 2323, "B", [], network_id=NET)
        self.refreshed: list[Any] = []
        self.calls: list[tuple[str, Any]] = []
        self.paired: list[PairedHeater] | Exception = [_paired(6)]
        self.reset_error: Exception | None = None
        self.restore_error: Exception | None = None
        self.client.async_pair = self._pair
        self.client.async_factory_reset = self._reset
        self.client.async_restore = self._restore
        self.runtime = add_radio_entry(
            hass,
            client=self.client,
            entry_id=ENTRY_ID,
            data={"brand": "radio", "nodes": NODES},
            states={"6": state} if state is not None else {},
            refresh=self._refresh,
        )
        self.entry = self.runtime.config_entry
        self.reloads = record_reloads(monkeypatch, hass)

    async def _refresh(self, node: Any) -> None:
        self.refreshed.append(node)

    async def _pair(self, window_s: float, **kwargs: Any) -> Any:
        self.calls.append(("pair", (window_s, kwargs)))
        if isinstance(self.paired, Exception):
            raise self.paired
        return self.paired

    async def _reset(self, addr: int) -> None:
        self.calls.append(("reset", addr))
        if self.reset_error is not None:
            raise self.reset_error

    async def _restore(self, addr: int, **kwargs: Any) -> None:
        self.calls.append(("restore", (addr, kwargs)))
        if self.restore_error is not None:
            raise self.restore_error

    async def call(self, name: str, **data: Any) -> Any:
        """Call a radio service through Home Assistant and return its response."""
        await service.async_register_radio_pairing_services(self.hass)
        return await self.hass.services.async_call(
            DOMAIN, name, data, blocking=True, return_response=True
        )


def _paired(node_id: int) -> PairedHeater:
    return PairedHeater(node_id, SimpleNamespace())


@pytest.fixture
def rig(hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch) -> Rig:
    """Return a loaded radio entry with heater 6 in manual mode."""
    return Rig(hass, monkeypatch)


# --- helpers ------------------------------------------------------------------


def test_settings_snapshot_keeps_only_restorable_values() -> None:
    """Only a complete, restorable mode/setpoint/program set is kept."""
    assert rp.settings_snapshot(None) is None
    assert rp.settings_snapshot(HeaterState()) is None
    assert rp.settings_snapshot(STATE) == SNAPSHOT
    override = HeaterState(mode="modified_auto", stemp="24", ptemp=["7", "x", "9"])
    assert rp.settings_snapshot(override) == {"mode": "auto"}
    odd = HeaterState(mode="eco", stemp=None, ptemp=["7", "8"], prog=[0] * 24)
    assert rp.settings_snapshot(odd) is None
    assert rp.settings_snapshot(HeaterState(mode="heat")) == {"mode": "manual"}


async def test_store_snapshot_and_add_nodes(rig: Rig) -> None:
    """Snapshots live in the options; new heaters are appended and the entry reloads."""
    hass, entry = rig.hass, rig.entry
    rp.store_snapshot(hass, entry, 6, None)  # nothing saved: options untouched
    assert "radio_restore" not in entry.options
    rp.store_snapshot(hass, entry, 6, {"mode": "off"})
    assert rp.saved_snapshot(entry, 6) == {"mode": "off"}
    assert rp.saved_snapshot(entry, 7) is None
    rp.store_snapshot(hass, entry, 6, None)
    assert entry.options["radio_restore"] == {}

    assert rp.add_nodes(hass, entry, [6]) is False
    assert rig.reloads == []
    assert rp.add_nodes(hass, entry, [9, 3, 9]) is True
    assert entry.data["nodes"][-2:] == [rp.radio_node(3), rp.radio_node(9)]
    assert rig.reloads == [ENTRY_ID]


# --- registration and validation ----------------------------------------------


async def test_registers_once_with_optional_responses(rig: Rig) -> None:
    """Registration is idempotent and every radio service may return a response."""
    await service.async_register_radio_pairing_services(rig.hass)
    await service.async_register_radio_pairing_services(rig.hass)
    for name in (
        service.SERVICE_RADIO_PAIR,
        service.SERVICE_RADIO_FACTORY_RESET,
        service.SERVICE_RADIO_REHOME,
    ):
        assert (
            rig.hass.services.supports_response(DOMAIN, name)
            is SupportsResponse.OPTIONAL
        )


@pytest.mark.parametrize(
    ("name", "data"),
    [
        (service.SERVICE_RADIO_PAIR, {"timeout": 29}),
        (service.SERVICE_RADIO_PAIR, {"timeout": 601}),
        (service.SERVICE_RADIO_PAIR, {"heater": 0}),
        (service.SERVICE_RADIO_PAIR, {"heater": 255}),
        (service.SERVICE_RADIO_FACTORY_RESET, {}),
    ],
)
async def test_schemas_reject_bad_values(rig: Rig, name: str, data: dict) -> None:
    """Out-of-range heaters and timeouts never reach the radio."""
    with pytest.raises(vol.Invalid):
        await rig.call(name, entry_id=ENTRY_ID, **data)
    assert rig.calls == []


async def test_services_validate_the_entry_and_heater(rig: Rig) -> None:
    """Unknown entries, cloud entries and unknown heaters are user errors."""
    with pytest.raises(ServiceValidationError, match="No loaded TermoWeb entry"):
        await rig.call(service.SERVICE_RADIO_PAIR, entry_id="missing")
    build_entry_runtime(hass=rig.hass, entry_id="cloud", client=object())
    with pytest.raises(ServiceValidationError, match="does not use a radio"):
        await rig.call(service.SERVICE_RADIO_FACTORY_RESET, entry_id="cloud", heater=6)
    with pytest.raises(ServiceValidationError, match="Heater 7 is not part"):
        await rig.call(service.SERVICE_RADIO_FACTORY_RESET, entry_id=ENTRY_ID, heater=7)
    with pytest.raises(ServiceValidationError, match="Heater 7 is not part"):
        await rig.call(service.SERVICE_RADIO_PAIR, entry_id=ENTRY_ID, heater="7")
    assert rig.calls == []


async def test_services_refuse_listen_only_entries(rig: Rig) -> None:
    """A listen-only entry never transmits, so it can neither pair nor rehome."""
    listen = RadioClient(
        "10.0.0.5", 2323, "A", [], network_id=b"\x00\x00", listen_only=True
    )
    build_entry_runtime(hass=rig.hass, entry_id="listen", client=listen)
    with pytest.raises(ServiceValidationError, match="listen-only"):
        await rig.call(service.SERVICE_RADIO_PAIR, entry_id="listen")
    with pytest.raises(ServiceValidationError, match="listen-only"):
        await rig.call(service.SERVICE_RADIO_REHOME, entry_id="listen")


# --- radio_factory_reset --------------------------------------------------------


async def test_factory_reset_saves_the_settings_first(rig: Rig) -> None:
    """The heater's settings are saved before the reset goes on air."""
    result = await rig.call(
        service.SERVICE_RADIO_FACTORY_RESET, entry_id=ENTRY_ID, heater=6
    )
    assert result == {"heater": 6, "settings_saved": True}
    assert rig.calls == [("reset", 6)]
    assert rp.saved_snapshot(rig.entry, 6) == SNAPSHOT


async def test_factory_reset_of_unknown_settings_saves_nothing(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Home Assistant knows nothing about the heater: the reset still runs."""
    rig = Rig(hass, monkeypatch, state=None)
    result = await rig.call(
        service.SERVICE_RADIO_FACTORY_RESET, entry_id=ENTRY_ID, heater=6
    )
    assert result == {"heater": 6, "settings_saved": False}
    assert rig.calls == [("reset", 6)]
    assert rp.saved_snapshot(rig.entry, 6) is None


async def test_unsupported_factory_reset_keeps_the_previous_snapshot(rig: Rig) -> None:
    """Nothing went on air: an earlier saved snapshot (or none) stays as it was."""
    rig.reset_error = RadioUnsupportedError("Factory reset in dialect A")
    with pytest.raises(ServiceValidationError, match="dialect A"):
        await rig.call(service.SERVICE_RADIO_FACTORY_RESET, entry_id=ENTRY_ID, heater=6)
    assert rp.saved_snapshot(rig.entry, 6) is None

    rp.store_snapshot(rig.hass, rig.entry, 6, {"mode": "off"})
    with pytest.raises(ServiceValidationError, match="dialect A"):
        await rig.call(service.SERVICE_RADIO_FACTORY_RESET, entry_id=ENTRY_ID, heater=6)
    assert rp.saved_snapshot(rig.entry, 6) == {"mode": "off"}


@pytest.mark.parametrize(
    "error",
    [
        RadioCommandError("heater 6 acknowledged command C8 but sent no reply"),
        RadioLinkError("connection lost"),
    ],
)
async def test_failed_factory_reset_still_keeps_the_settings(
    rig: Rig, error: Exception
) -> None:
    """The heater may be reset although its verdict was lost: its settings stay saved."""
    rig.reset_error = error
    with pytest.raises(HomeAssistantError, match="Factory reset failed"):
        await rig.call(service.SERVICE_RADIO_FACTORY_RESET, entry_id=ENTRY_ID, heater=6)
    assert rp.saved_snapshot(rig.entry, 6) == SNAPSHOT


# --- radio_pair -----------------------------------------------------------------


async def test_repair_restores_the_settings_saved_before_the_reset(rig: Rig) -> None:
    """A saved snapshot wins over the current state and is consumed by the restore."""
    saved = {"mode": "off", "ptemp": [5.0, 16.0, 19.0]}
    rp.store_snapshot(rig.hass, rig.entry, 6, saved)

    result = await rig.call(
        service.SERVICE_RADIO_PAIR, entry_id=ENTRY_ID, heater=6, timeout=60
    )

    assert result == {"heater": 6, "added": False, "restored": True}
    assert rig.calls == [
        ("pair", (60, {"wanted_id": 6, "max_heaters": 1})),
        ("restore", (6, saved)),
    ]
    assert rp.saved_snapshot(rig.entry, 6) is None
    assert rig.refreshed == [("htr", "6")]


async def test_repair_without_a_saved_snapshot_uses_the_current_state(
    rig: Rig,
) -> None:
    """Without a saved snapshot the heater's current settings are restored."""
    result = await rig.call(service.SERVICE_RADIO_PAIR, entry_id=ENTRY_ID, heater=6)
    assert result["restored"] is True
    assert rig.calls == [
        ("pair", (300, {"wanted_id": 6, "max_heaters": 1})),
        ("restore", (6, SNAPSHOT)),
    ]


async def test_repair_can_skip_the_restore(rig: Rig) -> None:
    """restore=false pairs only."""
    result = await rig.call(
        service.SERVICE_RADIO_PAIR, entry_id=ENTRY_ID, heater=6, restore=False
    )
    assert result["restored"] is False
    assert [name for name, _ in rig.calls] == ["pair"]


async def test_repair_without_known_settings_does_not_restore(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing to restore: the heater is paired and refreshed only."""
    rig = Rig(hass, monkeypatch, state=None)
    result = await rig.call(service.SERVICE_RADIO_PAIR, entry_id=ENTRY_ID, heater=6)
    assert result == {"heater": 6, "added": False, "restored": False}
    assert [name for name, _ in rig.calls] == ["pair"]


async def test_restore_failure_is_reported_and_keeps_the_snapshot(rig: Rig) -> None:
    """A failed restore keeps the snapshot for another try."""
    rig.restore_error = RadioCommandError("heater 6 rejected")
    rp.store_snapshot(rig.hass, rig.entry, 6, {"mode": "off"})
    with pytest.raises(HomeAssistantError, match="is paired, but restoring"):
        await rig.call(service.SERVICE_RADIO_PAIR, entry_id=ENTRY_ID, heater=6)
    assert rp.saved_snapshot(rig.entry, 6) == {"mode": "off"}


async def test_pairing_a_new_heater_adds_it_and_reloads(rig: Rig) -> None:
    """Without a heater number the paired heater is added to the entry."""
    rig.paired = [_paired(2)]
    result = await rig.call(service.SERVICE_RADIO_PAIR, entry_id=ENTRY_ID)
    assert result == {"heater": 2, "added": True, "restored": False}
    assert rig.calls == [("pair", (300, {"wanted_id": None, "max_heaters": 1}))]
    assert rig.entry.data["nodes"][-1] == rp.radio_node(2)
    assert rig.reloads == [ENTRY_ID]


@pytest.mark.parametrize(
    ("paired", "match"),
    [
        ([], "No heater was paired within 300 s"),
        (NoFreeAddressError("no free radio id"), "Radio pairing failed: no free"),
        (RadioLinkError("closed"), "Radio pairing failed: closed"),
    ],
)
async def test_pairing_failures(rig: Rig, paired: Any, match: str) -> None:
    """Pairing failures are reported and change nothing."""
    rig.paired = paired
    with pytest.raises(HomeAssistantError, match=match):
        await rig.call(service.SERVICE_RADIO_PAIR, entry_id=ENTRY_ID)
    assert rig.reloads == []
    assert rig.entry.data["nodes"] == NODES


async def test_manual_target_is_saved_and_restored_outside_manual_mode(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A heater in program mode still gets its last manual target back."""
    rig = Rig(
        hass,
        monkeypatch,
        state=HeaterState(mode="auto", stemp="18.5", ptemp=["7", "17", "20"]),
    )
    rig.client._manual_setpoints[6] = 20.0  # noqa: SLF001 - radio-only memory
    await rig.call(service.SERVICE_RADIO_FACTORY_RESET, entry_id=ENTRY_ID, heater=6)
    saved = {"mode": "auto", "ptemp": [7.0, 17.0, 20.0], "manual_stemp": 20.0}
    assert rp.saved_snapshot(rig.entry, 6) == saved

    await rig.call(service.SERVICE_RADIO_PAIR, entry_id=ENTRY_ID, heater=6)
    assert rig.calls[-1] == ("restore", (6, saved))
