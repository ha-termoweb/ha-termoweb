"""Home Assistant side of radio pairing: settings snapshots and new nodes.

A factory reset wipes a heater's settings. Before the reset, the heater's last
known settings are saved in the config entry options, so a later re-pairing
can give them back (``termoweb.radio_pair``). See ``docs/radio_protocol.md``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import logging
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import instance_id

from .backend.radio.pairing import site_network_id
from .const import CONF_NODES, CONF_RADIO_RESTORE

_LOGGER = logging.getLogger(__name__)

PROGRAM_HOURS = 168
RESTORE_MODES = {
    "auto": "auto",
    "modified_auto": "auto",  # a temporary override is not worth restoring
    "manual": "manual",
    "heat": "manual",
    "off": "off",
}


def radio_node(addr: int) -> dict[str, str]:
    """Return the stored node entry for a heater with radio id ``addr``."""

    return {"type": "htr", "addr": str(addr), "name": f"Heater {addr}"}


async def async_site_network_id(hass: HomeAssistant, gateway_id: str) -> bytes:
    """Return this installation's network id for a gateway: hash of instance + gateway."""

    return site_network_id(f"{await instance_id.async_get(hass)}:{gateway_id}")


def _float_or_none(value: Any) -> float | None:
    """Return ``value`` as a float, or None when it is not a number."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def settings_snapshot(state: Any) -> dict[str, Any] | None:
    """Return the restorable settings of a heater state, or None if it has none."""

    if state is None:
        return None
    snapshot: dict[str, Any] = {}
    mode = RESTORE_MODES.get(str(getattr(state, "mode", None)))
    if mode is not None:
        snapshot["mode"] = mode
        stemp = _float_or_none(getattr(state, "stemp", None))
        if mode == "manual" and stemp is not None:
            snapshot["stemp"] = stemp
    ptemp = getattr(state, "ptemp", None)
    if isinstance(ptemp, (list, tuple)) and len(ptemp) == 3:
        values = [_float_or_none(value) for value in ptemp]
        if all(value is not None for value in values):
            snapshot["ptemp"] = values
    prog = getattr(state, "prog", None)
    if isinstance(prog, (list, tuple)) and len(prog) == PROGRAM_HOURS:
        snapshot["prog"] = [int(slot) for slot in prog]
    return snapshot or None


def heater_snapshot(runtime: Any, addr: int) -> dict[str, Any] | None:
    """Return the restorable settings Home Assistant last saw for heater ``addr``."""

    view = runtime.coordinator.domain_view
    return settings_snapshot(view.get_heater_state("htr", str(addr)))


def saved_snapshot(entry: ConfigEntry, addr: int) -> dict[str, Any] | None:
    """Return the settings saved for ``addr`` before its factory reset, if any."""

    saved = (entry.options.get(CONF_RADIO_RESTORE) or {}).get(str(addr))
    return dict(saved) if isinstance(saved, Mapping) else None


def store_snapshot(
    hass: HomeAssistant, entry: ConfigEntry, addr: int, snapshot: dict[str, Any] | None
) -> None:
    """Save (or with None, drop) the settings to restore for heater ``addr``."""

    saved = dict(entry.options.get(CONF_RADIO_RESTORE) or {})
    if snapshot is None:
        if str(addr) not in saved:
            return
        saved.pop(str(addr))
    else:
        saved[str(addr)] = snapshot
    hass.config_entries.async_update_entry(
        entry, options={**entry.options, CONF_RADIO_RESTORE: saved}
    )


def add_nodes(hass: HomeAssistant, entry: ConfigEntry, addrs: Iterable[int]) -> bool:
    """Add heaters to the entry's node list and reload it; False if none were new."""

    nodes = [dict(node) for node in entry.data.get(CONF_NODES, [])]
    known = {str(node.get("addr")) for node in nodes}
    new = [radio_node(addr) for addr in sorted(set(addrs)) if str(addr) not in known]
    if not new:
        return False
    hass.config_entries.async_update_entry(
        entry, data={**entry.data, CONF_NODES: nodes + new}
    )
    _LOGGER.info("Added %d paired heater(s); reloading the entry", len(new))
    # The inventory is fixed for the entry's lifetime: a reload picks the nodes up.
    hass.config_entries.async_schedule_reload(entry.entry_id)
    return True


__all__ = [
    "add_nodes",
    "async_site_network_id",
    "heater_snapshot",
    "radio_node",
    "saved_snapshot",
    "settings_snapshot",
    "store_snapshot",
]
