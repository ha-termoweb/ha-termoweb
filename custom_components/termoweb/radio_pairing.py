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

from .backend.radio.pairing import PairedHeater, site_network_id
from .backend.radio_client import RadioUnsupportedError
from .backend.sanitize import mask_identifier
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


# Default device name per radio node type; a thermostat would be "Thermostat <serial>".
_DEFAULT_NAMES = {"htr": "Heater", "thm": "Thermostat"}


def default_node_name(node_type: str, token: Any) -> str:
    """Return a radio node's default name: "Heater <serial>" (or its address)."""

    return f"{_DEFAULT_NAMES.get(node_type, 'Heater')} {token}"


def radio_node(addr: int, serial: str | None = None) -> dict[str, str]:
    """Return the stored node entry for a heater with radio id ``addr``."""

    node = {"type": "htr", "addr": str(addr), "name": default_node_name("htr", addr)}
    return with_serial(node, serial) if serial else node


def with_serial(node: Mapping[str, Any], serial: str) -> dict[str, Any]:
    """Return ``node`` carrying ``serial``; an address-based default name follows it."""

    updated = {**node, "serial": serial}
    node_type = str(node.get("type") or "htr")
    if not node.get("name") or node["name"] == default_node_name(
        node_type, node.get("addr")
    ):
        updated["name"] = default_node_name(node_type, serial)
    return updated


async def async_site_network_id(hass: HomeAssistant, gateway_id: str) -> bytes:
    """Return this installation's network id for a gateway: hash of instance + gateway."""

    return site_network_id(f"{await instance_id.async_get(hass)}:{gateway_id}")


def _float_or_none(value: Any) -> float | None:
    """Return ``value`` as a float, or None when it is not a number."""

    try:
        return float(value)
    except TypeError, ValueError:
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


def with_manual_target(
    snapshot: dict[str, Any] | None, client: Any, addr: int
) -> dict[str, Any] | None:
    """Add the last manual target the client saw when the mode was not manual."""

    manual = client.manual_setpoint(addr)
    if snapshot is None or "stemp" in snapshot or manual is None:
        return snapshot
    return {**snapshot, "manual_stemp": manual}


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


async def async_reset_keeping_settings(
    hass: HomeAssistant, runtime: Any, addr: int
) -> bool:
    """Save heater ``addr``'s settings, then factory-reset it; True if any were saved.

    The settings are saved before the reset goes on air and kept when it fails:
    a heater can reset and still lose its verdict. Only a reset that was never
    sent (unsupported dialect) leaves the saved settings as they were.
    """

    entry, client = runtime.config_entry, runtime.client
    previous = saved_snapshot(entry, addr)
    snapshot = with_manual_target(heater_snapshot(runtime, addr), client, addr)
    if snapshot is not None:
        store_snapshot(hass, entry, addr, snapshot)
    try:
        await client.async_factory_reset(addr)
    except RadioUnsupportedError:
        if snapshot is not None:
            store_snapshot(hass, entry, addr, previous)
        raise
    return snapshot is not None


def merge_heaters(
    stored_nodes: Iterable[Mapping[str, Any]], heaters: Iterable[PairedHeater]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Return ``(nodes, added nodes, moved count)`` after merging ``heaters`` in.

    A heater whose serial is already stored moves that node to its new address,
    so its devices and entities (keyed by the serial) stay the same. A heater
    at an address that is already stored is not added again.
    """

    nodes = [dict(node) for node in stored_nodes]
    by_serial = {node["serial"]: node for node in nodes if node.get("serial")}
    known = {str(node.get("addr")) for node in nodes}
    added: list[dict[str, Any]] = []
    moved = 0
    for heater in sorted(heaters, key=lambda heater: heater.node_id):
        addr = str(heater.node_id)
        stored = by_serial.get(heater.serial) if heater.serial else None
        if stored is not None:
            if stored["addr"] != addr:
                _LOGGER.info(
                    "Heater %s moved from radio id %s to %s",
                    mask_identifier(heater.serial),
                    stored["addr"],
                    addr,
                )
                stored["addr"] = addr
                moved += 1
        elif addr not in known:
            added.append(radio_node(heater.node_id, heater.serial))
            nodes.append(added[-1])
        known.add(addr)
    return nodes, added, moved


def add_nodes(
    hass: HomeAssistant, entry: ConfigEntry, heaters: Iterable[PairedHeater]
) -> bool:
    """Add paired heaters to the entry's nodes and reload it; False if nothing changed."""

    nodes, added, moved = merge_heaters(entry.data.get(CONF_NODES, []), heaters)
    if not added and not moved:
        return False
    hass.config_entries.async_update_entry(
        entry, data={**entry.data, CONF_NODES: nodes}
    )
    _LOGGER.info(
        "Added %d and moved %d paired heater(s); reloading the entry",
        len(added),
        moved,
    )
    # The inventory is fixed for the entry's lifetime: a reload picks the nodes up.
    hass.config_entries.async_schedule_reload(entry.entry_id)
    return True


__all__ = [
    "add_nodes",
    "async_reset_keeping_settings",
    "async_site_network_id",
    "default_node_name",
    "heater_snapshot",
    "merge_heaters",
    "radio_node",
    "saved_snapshot",
    "settings_snapshot",
    "store_snapshot",
    "with_manual_target",
    "with_serial",
]
