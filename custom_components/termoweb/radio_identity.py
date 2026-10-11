"""Identify radio heaters by serial number instead of their pairing address.

A heater's radio id (``addr``) follows the pairing order and changes when the
heater is paired again; its serial does not. Setup reads the serial of every
stored heater that has none and stores it with the node. Device identifiers
and entity unique IDs use the serial as the node key (``identifiers.node_key``),
and :func:`async_rekey_node` moves the registry entries of an address-keyed
heater to the serial, keeping entity IDs, history and customisations.
See ``docs/radio_backend.md``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr, entity_registry as er

from .backend.radio.link import RadioLinkError
from .backend.radio_client import RadioClient, RadioError, radio_addr
from .backend.sanitize import mask_identifier
from .const import CONF_NODES, DOMAIN
from .identifiers import build_node_device_identifier, rekey_unique_id
from .inventory import Inventory
from .radio_pairing import with_serial

_LOGGER = logging.getLogger(__name__)

SERIAL_READ_TIMEOUT_S = 10.0  # one identity read per heater during setup


async def async_learn_serials(
    hass: HomeAssistant, entry: ConfigEntry, client: Any
) -> None:
    """Read and store the serial of each stored heater that has none (one try each).

    A heater that does not answer keeps its address-based identity for this
    run; the next setup tries again.
    """

    if not isinstance(client, RadioClient) or client.listen_only:
        return
    nodes = [dict(node) for node in entry.data.get(CONF_NODES, [])]
    learned = 0
    for index, node in enumerate(nodes):
        if node.get("serial"):
            continue
        try:
            addr = radio_addr(node.get("addr"))
        except ValueError:
            continue
        try:
            async with asyncio.timeout(SERIAL_READ_TIMEOUT_S):
                record = await client.async_read_identity(addr)
        except (TimeoutError, RadioError, RadioLinkError) as err:
            _LOGGER.info(
                "Heater %s did not report its serial number (%s); it keeps its "
                "address-based identity until the next start",
                addr,
                str(err) or "no answer",
            )
            continue
        nodes[index] = with_serial(node, record.serial)
        learned += 1
        _LOGGER.info("Heater %s has serial %s", addr, mask_identifier(record.serial))
    if learned:
        hass.config_entries.async_update_entry(
            entry, data={**entry.data, CONF_NODES: nodes}
        )
        client.set_nodes(nodes)


@callback
def async_rekey_node(
    hass: HomeAssistant,
    entry_id: str,
    dev_id: str,
    node_type: str,
    old_key: str,
    new_key: str,
) -> None:
    """Move a node's device and entities from ``old_key`` to ``new_key`` IDs."""

    if old_key == new_key:
        return
    dev_reg = dr.async_get(hass)
    old_identifier = build_node_device_identifier(dev_id, old_key)
    new_identifier = build_node_device_identifier(dev_id, new_key)
    device = dev_reg.async_get_device_by_identifier(old_identifier, entry_id)
    if device is not None:
        if dev_reg.async_get_device_by_identifier(new_identifier, entry_id) is not None:
            _LOGGER.warning(
                "Not moving device %s to heater %s: that heater already has a device",
                device.id,
                mask_identifier(new_key),
            )
        else:
            dev_reg.async_update_device(
                device.id,
                new_identifiers=(device.identifiers - {old_identifier})
                | {new_identifier},
            )
    ent_reg = er.async_get(hass)
    for entity in er.async_entries_for_config_entry(ent_reg, entry_id):
        new_id = rekey_unique_id(entity.unique_id, dev_id, node_type, old_key, new_key)
        if new_id is None:
            continue
        if taken := ent_reg.async_get_entity_id(entity.domain, DOMAIN, new_id):
            _LOGGER.warning(
                "Not moving %s to heater %s: %s already has its unique ID",
                entity.entity_id,
                mask_identifier(new_key),
                taken,
            )
            continue
        _LOGGER.debug("Moving %s to heater %s", entity.entity_id, new_key)
        ent_reg.async_update_entity(entity.entity_id, new_unique_id=new_id)


@callback
def async_rekey_inventory(
    hass: HomeAssistant, entry_id: str, inventory: Inventory
) -> None:
    """Move every node with a known serial from address-keyed to serial-keyed IDs."""

    for node in inventory.nodes:
        serial = getattr(node, "serial", None)
        if serial:
            async_rekey_node(
                hass, entry_id, inventory.dev_id, node.type, node.addr, serial
            )


__all__ = [
    "SERIAL_READ_TIMEOUT_S",
    "async_learn_serials",
    "async_rekey_inventory",
    "async_rekey_node",
]
