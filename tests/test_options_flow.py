"""Tests for the radio entry options flow."""

from __future__ import annotations

import asyncio
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.termoweb.backend.radio import RadioLinkError
from custom_components.termoweb.backend.radio.pairing import (
    NoFreeAddressError,
    PairedHeater,
)
from custom_components.termoweb.const import BRAND_RADIO, CONF_BRAND, DOMAIN
from tests.fakes.radio_setup import record_reloads
from tests.fakes.runtime import build_entry_runtime

# Synthetic identifiers only.
NODES = [{"type": "htr", "addr": "6", "name": "Heater 6"}]


def _paired(node_id: int) -> PairedHeater:
    """Return a heater paired as ``node_id``."""
    return PairedHeater(node_id, None)


# --- options flow ------------------------------------------------------------------


def _options_entry(
    hass: HomeAssistant, nodes: list[dict[str, Any]] | None = None, **options: Any
) -> MockConfigEntry:
    """Add a radio entry with ``options``."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_BRAND: BRAND_RADIO,
            "network_id": "1234",
            "nodes": NODES if nodes is None else nodes,
        },
        options=options,
    )
    entry.add_to_hass(hass)
    return entry


async def _options_step(hass: HomeAssistant, entry: MockConfigEntry, step: str) -> Any:
    """Open the options menu and pick ``step``."""
    menu = await hass.config_entries.options.async_init(entry.entry_id)
    assert menu["menu_options"] == ["pair_heaters", "rehome"]
    return await hass.config_entries.options.async_configure(
        menu["flow_id"], {"next_step_id": step}
    )


async def test_options_menu_has_no_heater_power_settings(
    hass: HomeAssistant,
) -> None:
    """Heater power is read from the heaters, so the menu has no settings step."""
    entry = _options_entry(hass, radio_power={"power_limit": 2000})
    menu = await hass.config_entries.options.async_init(entry.entry_id)
    assert menu["type"] is FlowResultType.MENU
    assert menu["menu_options"] == ["pair_heaters", "rehome"]


class PairingClient:
    """The loaded entry's radio client; its pairing result is scripted."""

    def __init__(self) -> None:
        """Pair heater 2 by default."""
        self.result: Any = [_paired(2)]
        self.calls: list[Any] = []

    async def async_pair(self, window_s: float, **kwargs: Any) -> Any:
        """Return (or raise) the scripted pairing result."""
        self.calls.append((window_s, kwargs))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


async def _options_pair(hass: HomeAssistant, entry: MockConfigEntry) -> Any:
    """Run the options pairing steps and return the final result."""
    form = await _options_step(hass, entry, "pair_heaters")
    assert form["step_id"] == "pair_heaters" and form["errors"] == {}
    result = await hass.config_entries.options.async_configure(form["flow_id"], {})
    if result["type"] is not FlowResultType.SHOW_PROGRESS:
        return result
    await hass.async_block_till_done()
    return await hass.config_entries.options.async_configure(result["flow_id"])


@pytest.fixture
def pairing_entry(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> tuple[MockConfigEntry, PairingClient, list[str]]:
    """A loaded radio entry with a scripted pairing client; reloads recorded."""
    client = PairingClient()
    entry = _options_entry(hass, energy_history_imported=True)
    build_entry_runtime(
        hass=hass,
        entry_id=entry.entry_id,
        client=client,
        config_entry=entry,
        brand=BRAND_RADIO,
    )
    return entry, client, record_reloads(monkeypatch, hass)


async def test_options_pairing_adds_heaters_and_reloads(
    hass: HomeAssistant, pairing_entry: tuple
) -> None:
    """Paired heaters are added to the entry, which then reloads."""
    entry, client, reloads = pairing_entry
    result = await _options_pair(hass, entry)
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert dict(entry.options) == {"energy_history_imported": True}
    assert client.calls == [(300.0, {"idle_stop_s": 60.0})]
    assert entry.data["nodes"][-1] == {"type": "htr", "addr": "2", "name": "Heater 2"}
    assert reloads == [entry.entry_id]


async def test_options_pairing_keeps_heaters_paired_before_ids_ran_out(
    hass: HomeAssistant, pairing_entry: tuple
) -> None:
    """Heaters paired before the radio ids ran out are still added."""
    entry, client, _reloads = pairing_entry
    client.result = NoFreeAddressError("no free radio id", [_paired(0x41)])
    result = await _options_pair(hass, entry)
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert entry.data["nodes"][-1]["addr"] == "65"


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        ([], "no_heaters_paired"),
        (NoFreeAddressError("no free radio id"), "no_free_address"),
        (RadioLinkError("closed"), "cannot_connect_radio"),
        (RuntimeError("boom"), "unknown"),
    ],
)
async def test_options_pairing_errors_show_the_form_again(
    hass: HomeAssistant, pairing_entry: tuple, error: Any, reason: str
) -> None:
    """A failed run shows the pairing form with the reason; nothing reloads."""
    entry, client, reloads = pairing_entry
    client.result = error
    result = await _options_pair(hass, entry)
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "pair_heaters"
    assert result["errors"] == {"base": reason}
    assert reloads == []


async def test_options_pairing_needs_a_loaded_entry(hass: HomeAssistant) -> None:
    """Without a running radio, pairing cannot start."""
    entry = _options_entry(hass)
    result = await _options_pair(hass, entry)
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "not_loaded"


async def test_options_pairing_shows_progress_until_the_run_ends(
    hass: HomeAssistant, pairing_entry: tuple
) -> None:
    """Options pairing shows progress while the gateway listens."""
    entry, client, reloads = pairing_entry
    release = asyncio.Event()

    async def slow(window_s: float, **kwargs: Any) -> Any:
        await release.wait()
        return [_paired(2)]

    client.async_pair = slow
    form = await _options_step(hass, entry, "pair_heaters")
    result = await hass.config_entries.options.async_configure(form["flow_id"], {})
    assert result["type"] is FlowResultType.SHOW_PROGRESS
    assert result["progress_action"] == "pair_heaters"
    release.set()
    await hass.async_block_till_done()
    result = await hass.config_entries.options.async_configure(result["flow_id"])
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert reloads == [entry.entry_id]
