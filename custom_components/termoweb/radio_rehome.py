"""Move a radio entry's heaters onto this installation's own network ("re-home").

An entry set up by discovery keeps the network id of the gateway the heaters
were sold with. Re-homing resets every heater (dialect B ``C8 01 D0``), moves
the station to the site network id, pairs the heaters again and restores
their settings. See ``docs/radio_protocol.md`` (section 9).
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

from homeassistant.core import HomeAssistant

from .backend.radio.dialect import DIALECT_B
from .backend.radio.link import RadioLinkError
from .backend.radio.pairing import IDLE_STOP_S, PairedHeater, PairingError
from .backend.radio_client import RadioError, radio_addr
from .const import CONF_NETWORK_ID, CONF_NODES
from .radio_pairing import (
    async_site_network_id,
    heater_snapshot,
    radio_node,
    saved_snapshot,
    store_snapshot,
    with_manual_target,
)

_LOGGER = logging.getLogger(__name__)


class RehomeError(Exception):
    """The move stopped; the message tells the user what state the heaters are in."""


def _addr(node: dict[str, Any]) -> int | None:
    """Return a stored node's radio id, or None for an invalid address."""

    try:
        return radio_addr(node.get("addr"))
    except ValueError:
        return None


async def _read_identities(client: Any, addrs: list[int]) -> dict[int, bytes]:
    """Return every heater's ``5A`` identity; raise RehomeError before any change."""

    identities = {}
    for addr in addrs:
        try:
            identities[addr] = await client.async_read_identity(addr)
        except (RadioError, RadioLinkError) as err:
            raise RehomeError(
                f"Heater {addr} does not answer, so nothing was changed. Move the "
                f"gateway closer to it and try again. ({err})"
            ) from err
    return identities


async def _reset_all(hass: HomeAssistant, runtime: Any, addrs: list[int]) -> None:
    """Save each heater's settings, then factory-reset it; stop at the first failure."""

    entry, client = runtime.config_entry, runtime.client
    done: list[int] = []
    for addr in addrs:
        snapshot = with_manual_target(heater_snapshot(runtime, addr), client, addr)
        try:
            await client.async_factory_reset(addr)
        except (RadioError, RadioLinkError) as err:
            waiting = (
                f" Heaters {', '.join(map(str, done))} are already reset and keep "
                "their saved settings: pair each one with the Radio pair action "
                "and its heater number."
                if done
                else ""
            )
            raise RehomeError(
                f"Heater {addr} could not be reset ({err}). The network was not "
                f"changed.{waiting}"
            ) from err
        if snapshot is not None:
            store_snapshot(hass, entry, addr, snapshot)
        done.append(addr)


async def _identified(client: Any, heater: PairedHeater) -> PairedHeater:
    """Return ``heater`` with its 5A identity, reading it again if pairing had none."""

    if heater.identity is not None:
        return heater
    try:
        identity = await client.async_read_identity(heater.node_id)
    except (RadioError, RadioLinkError):
        return heater
    return dataclasses.replace(heater, identity=identity)


def _match(
    addrs: list[int], identities: dict[int, bytes], paired: list[Any]
) -> dict[int, int]:
    """Return ``{new id: old id}``: direct for one heater, else by 5A identity."""

    if len(addrs) == 1 and len(paired) == 1:
        return {paired[0].node_id: addrs[0]}
    matches: dict[int, int] = {}
    for heater in paired:
        for old, identity in identities.items():
            if heater.identity == identity and old not in matches.values():
                matches[heater.node_id] = old
                break
    return matches


async def async_rehome(
    hass: HomeAssistant, runtime: Any, timeout_s: float
) -> dict[str, Any]:
    """Reset, re-pair and restore every heater on the site network; reload the entry.

    Nothing changes when a heater does not answer the identity read. Once all
    heaters are reset, the entry always moves to the site network: paired
    heaters get new ids (old ids are skipped, so heaters that were not paired
    keep their node and saved settings for a later Radio pair).
    """

    entry, client = runtime.config_entry, runtime.client
    if client.dialect is not DIALECT_B:
        raise RehomeError(
            "Only dialect-B heaters can be reset over the radio. Reset dialect-A "
            "heaters on their panel, then set the integration up again and "
            'choose "Pair new heaters".'
        )
    old_net = bytes.fromhex(entry.data[CONF_NETWORK_ID])
    new_net = await async_site_network_id(hass, str(runtime.dev_id))
    if new_net == old_net:
        raise RehomeError("The heaters already use this installation's own network.")
    nodes = [dict(node) for node in entry.data.get(CONF_NODES, [])]
    addrs = sorted({addr for node in nodes if (addr := _addr(node)) is not None})
    if not addrs:
        raise RehomeError("There are no heaters to move.")

    _LOGGER.info("Re-homing %d heater(s) onto the site network", len(addrs))
    identities = await _read_identities(client, addrs)
    await _reset_all(hass, runtime, addrs)

    await client.async_set_network_id(new_net)
    paired: list[Any] = []
    try:
        paired = await client.async_pair(
            timeout_s, max_heaters=len(addrs), idle_stop_s=IDLE_STOP_S
        )
    except PairingError as err:
        paired = err.paired
    except (RadioError, RadioLinkError) as err:
        _LOGGER.error("Pairing during the network move failed: %s", err)

    if len(addrs) > 1:  # one heater maps directly; more need their identities
        paired = [await _identified(client, heater) for heater in paired]
    matches = _match(addrs, identities, paired)
    names = {_addr(node): node.get("name") for node in nodes}
    moved = []
    for heater in paired:
        old = matches.get(heater.node_id)
        if old is None:
            continue
        snapshot = saved_snapshot(entry, old)
        restored = False
        if snapshot is not None:
            try:
                await client.async_restore(heater.node_id, **snapshot)
                restored = True
            except (RadioError, RadioLinkError, ValueError) as err:
                _LOGGER.error("Restoring heater %s failed: %s", heater.node_id, err)
        store_snapshot(hass, entry, old, None)
        if snapshot is not None and not restored:
            store_snapshot(hass, entry, heater.node_id, snapshot)
        moved.append({"from": old, "to": heater.node_id, "restored": restored})

    matched_old = set(matches.values())
    new_nodes = [node for node in nodes if _addr(node) not in matched_old]
    for heater in paired:
        node = radio_node(heater.node_id)
        old = matches.get(heater.node_id)
        if old is not None and names.get(old):
            node["name"] = names[old]
        new_nodes.append(node)
    hass.config_entries.async_update_entry(
        entry,
        data={
            **entry.data,
            CONF_NETWORK_ID: new_net.hex().upper(),
            CONF_NODES: new_nodes,
        },
    )
    hass.config_entries.async_schedule_reload(entry.entry_id)
    result = {
        "moved": moved,
        "unidentified": sorted(h.node_id for h in paired if h.node_id not in matches),
        "waiting": sorted(set(addrs) - matched_old),
    }
    _LOGGER.info("Network move finished: %s", result)
    return result


def rehome_summary(result: dict[str, Any]) -> str:
    """Return a plain-English summary of a finished network move."""

    lines = ["Your heaters now use this installation's own radio network."]
    for item in result["moved"]:
        outcome = (
            "Its settings were restored."
            if item["restored"]
            else "Its settings could not be restored: check them."
        )
        lines.append(f"- Heater {item['from']} is now heater {item['to']}. {outcome}")
    lines.extend(
        f"- Heater {addr} was paired, but its old heater was not recognised. "
        "Set its temperatures and program again."
        for addr in result["unidentified"]
    )
    lines.extend(
        f"- Heater {addr} was not paired. It is reset and waits. Use the "
        f"Radio pair action with heater number {addr} to pair it and "
        "restore its settings."
        for addr in result["waiting"]
    )
    if result["unidentified"] and result["waiting"]:
        lines.append(
            "A heater listed as not paired may be one of the heaters that were "
            "not recognised: check the heaters before you pair again."
        )
    if not result["moved"] and not result["unidentified"]:
        lines.insert(1, "No heater was paired in time.")
    return "\n".join(lines)


__all__ = ["RehomeError", "async_rehome", "rehome_summary"]
