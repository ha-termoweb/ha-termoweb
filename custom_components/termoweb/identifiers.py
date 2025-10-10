"""Identifier builders for TermoWeb entities."""

from __future__ import annotations

from typing import Any

from .const import DOMAIN
from .inventory import normalize_node_addr, normalize_node_type


def build_node_unique_id(
    dev_id: Any,
    node_type: Any,
    addr: Any,
    *,
    suffix: str | None = None,
) -> str:
    """Return the canonical unique ID for a TermoWeb node."""

    dev = normalize_node_addr(dev_id)
    node = normalize_node_type(node_type)
    address = normalize_node_addr(addr)
    if not dev or not node or not address:
        raise ValueError("dev_id, node_type and addr must be provided")

    suffix_str = ""
    if suffix:
        suffix_clean = str(suffix)
        suffix_str = (
            suffix_clean if suffix_clean.startswith(":") else f":{suffix_clean}"
        )

    return f"{DOMAIN}:{dev}:{node}:{address}{suffix_str}"


def build_heater_unique_id(
    dev_id: Any,
    node_type: Any,
    addr: Any,
    *,
    suffix: str | None = None,
) -> str:
    """Return the canonical unique ID for a heater node."""

    return build_node_unique_id(dev_id, node_type, addr, suffix=suffix)


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

    return build_energy_unique_id(dev_id, node_type, addr)


def build_energy_unique_id(dev_id: Any, node_type: Any, addr: Any) -> str:
    """Return the canonical unique ID for an energy sensor."""

    return build_node_unique_id(dev_id, node_type, addr, suffix=":energy")


def build_power_monitor_energy_unique_id(dev_id: Any, addr: Any) -> str:
    """Return the canonical unique ID for a power monitor energy sensor."""

    return build_energy_unique_id(dev_id, "pmo", addr)
