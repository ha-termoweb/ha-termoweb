"""The radio_factory_reset and radio_rehome services, called through Home Assistant."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from homeassistant.core import HomeAssistant, SupportsResponse
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
import pytest
import voluptuous as vol

from custom_components.termoweb import radio_pairing as rp
from custom_components.termoweb.backend.radio.link import RadioLinkError
from custom_components.termoweb.backend.radio.pairing import PairedHeater
from custom_components.termoweb.backend.radio_client import (
    RadioClient,
    RadioCommandError,
    RadioUnsupportedError,
)
from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.domain.state import HeaterState
from custom_components.termoweb.services import radio_pairing as service
from tests.fakes.radio_setup import add_radio_entry
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
    """A loaded radio entry whose client records factory resets."""

    def __init__(
        self,
        hass: HomeAssistant,
        monkeypatch: pytest.MonkeyPatch,
        state: HeaterState | None = STATE,
    ) -> None:
        """Add the entry, wire the scripted client and register the services."""
        self.hass = hass
        self.client = RadioClient("10.0.0.5", 2323, "B", [], network_id=NET)
        self.calls: list[tuple[str, Any]] = []
        self.reset_error: Exception | None = None
        self.client.async_factory_reset = self._reset
        self.runtime = add_radio_entry(
            hass,
            client=self.client,
            entry_id=ENTRY_ID,
            data={"brand": "radio", "nodes": NODES},
            states={"6": state} if state is not None else {},
        )
        self.entry = self.runtime.config_entry

    async def _reset(self, addr: int) -> None:
        self.calls.append(("reset", addr))
        if self.reset_error is not None:
            raise self.reset_error

    async def call(self, name: str, **data: Any) -> Any:
        """Call a radio service through Home Assistant and return its response."""
        await service.async_register_radio_pairing_services(self.hass)
        return await self.hass.services.async_call(
            DOMAIN, name, data, blocking=True, return_response=True
        )


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


async def test_store_snapshot(rig: Rig) -> None:
    """Snapshots live in the options; dropping a missing one changes nothing."""
    hass, entry = rig.hass, rig.entry
    rp.store_snapshot(hass, entry, 6, None)  # nothing saved: options untouched
    assert "radio_restore" not in entry.options
    rp.store_snapshot(hass, entry, 6, {"mode": "off"})
    assert rp.saved_snapshot(entry, 6) == {"mode": "off"}
    assert rp.saved_snapshot(entry, 7) is None
    rp.store_snapshot(hass, entry, 6, None)
    assert entry.options["radio_restore"] == {}


def test_merge_heaters_keeps_a_known_serial_at_its_address() -> None:
    """A heater paired again at its stored address changes nothing."""
    nodes = [{"type": "htr", "addr": "6", "name": "Lounge", "serial": "S000006"}]
    heater = PairedHeater(6, SimpleNamespace(), serial="S000006")
    assert rp.merge_heaters(nodes, [heater]) == (nodes, [], 0)
    unnamed = PairedHeater(6, SimpleNamespace())  # no serial, address already stored
    assert rp.merge_heaters(nodes, [unnamed]) == (nodes, [], 0)


def test_radio_node_default_names() -> None:
    """Default names use the serial; a custom or serial name is kept."""
    assert rp.radio_node(2) == {"type": "htr", "addr": "2", "name": "Heater 2"}
    assert rp.radio_node(2, "1a2b3c4d")["name"] == "Heater 1a2b3c4d"
    assert rp.with_serial({"type": "htr", "addr": "2"}, "S1")["name"] == "Heater S1"
    custom = {"type": "htr", "addr": "2", "name": "Hall"}
    assert rp.with_serial(custom, "S1")["name"] == "Hall"
    assert rp.default_node_name("thm", "S1") == "Thermostat S1"


# --- registration and validation ----------------------------------------------


async def test_registers_once_with_optional_responses(rig: Rig) -> None:
    """Registration is idempotent and every radio service may return a response."""
    await service.async_register_radio_pairing_services(rig.hass)
    await service.async_register_radio_pairing_services(rig.hass)
    # Pairing is automatic (the pairing window); there is no pairing service.
    assert not rig.hass.services.has_service(DOMAIN, "radio_pair")
    for name in (
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
        (service.SERVICE_RADIO_REHOME, {"timeout": 29}),
        (service.SERVICE_RADIO_REHOME, {"timeout": 601}),
        (service.SERVICE_RADIO_FACTORY_RESET, {"heater": 0}),
        (service.SERVICE_RADIO_FACTORY_RESET, {"heater": 255}),
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
        await rig.call(service.SERVICE_RADIO_REHOME, entry_id="missing")
    build_entry_runtime(hass=rig.hass, entry_id="cloud", client=object())
    with pytest.raises(ServiceValidationError, match="does not use a radio"):
        await rig.call(service.SERVICE_RADIO_FACTORY_RESET, entry_id="cloud", heater=6)
    with pytest.raises(ServiceValidationError, match="Heater 7 is not part"):
        await rig.call(service.SERVICE_RADIO_FACTORY_RESET, entry_id=ENTRY_ID, heater=7)
    assert rig.calls == []


async def test_services_refuse_listen_only_entries(rig: Rig) -> None:
    """A listen-only entry never transmits, so it can neither reset nor rehome."""
    listen = RadioClient(
        "10.0.0.5", 2323, "A", [], network_id=b"\x00\x00", listen_only=True
    )
    build_entry_runtime(hass=rig.hass, entry_id="listen", client=listen)
    with pytest.raises(ServiceValidationError, match="listen-only"):
        await rig.call(service.SERVICE_RADIO_FACTORY_RESET, entry_id="listen", heater=6)
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


async def test_unsupported_reset_of_unknown_settings_saves_nothing(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused reset of a heater without known settings leaves the options alone."""
    rig = Rig(hass, monkeypatch, state=None)
    rig.reset_error = RadioUnsupportedError("Factory reset in dialect A")
    with pytest.raises(ServiceValidationError, match="dialect A"):
        await rig.call(service.SERVICE_RADIO_FACTORY_RESET, entry_id=ENTRY_ID, heater=6)
    assert "radio_restore" not in rig.entry.options


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


async def test_manual_target_is_saved_outside_manual_mode(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A heater in program mode also keeps its last manual target for the restore."""
    rig = Rig(
        hass,
        monkeypatch,
        state=HeaterState(mode="auto", stemp="18.5", ptemp=["7", "17", "20"]),
    )
    rig.client._manual_setpoints[6] = 20.0
    await rig.call(service.SERVICE_RADIO_FACTORY_RESET, entry_id=ENTRY_ID, heater=6)
    saved = {"mode": "auto", "ptemp": [7.0, 17.0, 20.0], "manual_stemp": 20.0}
    assert rp.saved_snapshot(rig.entry, 6) == saved
