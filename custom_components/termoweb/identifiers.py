"""Identifier builders for TermoWeb entities."""

from __future__ import annotations

from typing import Any

from . import nodes as nodes_module
from .const import DOMAIN


def build_heater_unique_id(
    dev_id: Any,
    node_type: Any,
    addr: Any,
    *,
    suffix: str | None = None,
) -> str:
    """Return the canonical unique ID for a heater node."""

    dev = nodes_module.normalize_node_addr(dev_id)
    node = nodes_module.normalize_node_type(node_type)
    address = nodes_module.normalize_node_addr(addr)
    if not dev or not node or not address:
        raise ValueError("dev_id, node_type and addr must be provided")

    suffix_str = ""
    if suffix:
        suffix_clean = str(suffix)
        suffix_str = suffix_clean if suffix_clean.startswith(":") else f":{suffix_clean}"

    return f"{DOMAIN}:{dev}:{node}:{address}{suffix_str}"


def build_heater_entity_unique_id(
    dev_id: Any,
    node_type: Any,
    addr: Any,
    suffix: str | None = None,
) -> str:
    """Return the canonical unique ID for a heater entity."""

    return build_heater_unique_id(dev_id, node_type, addr, suffix=suffix)


def build_heater_energy_unique_id(dev_id: Any, node_type: Any, addr: Any) -> str:
    """Return the canonical unique ID for a heater energy sensor."""

    return build_heater_unique_id(dev_id, node_type, addr, suffix=":energy")
