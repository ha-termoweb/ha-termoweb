"""Identifier builders for TermoWeb entities.

Every entity unique ID follows one scheme and is built only here:

    termoweb:<dev_id>[:<device>]:<key>

``<device>`` names the device the entity belongs to: nothing for the gateway,
``site`` for the site (installation), ``<node_type>:<node_key>`` for a node.
``<key>`` is one snake_case token. ``migrate_unique_id`` maps older formats.

A node's key (:func:`node_key`) is its address, except for a radio heater whose
serial is known: then it is the serial, so a heater that is paired again at
another address keeps its devices and entities. Node devices use the same key:
``(termoweb, <dev_id>, <node_key>)``.
"""

from __future__ import annotations

import re
from typing import Any

from .backend.factory import backend_capabilities
from .const import DOMAIN
from .nodes import normalize_node_addr, normalize_node_type

_KEY = r"[a-z0-9]+(?:_[a-z0-9]+)*"
_KEY_RE = re.compile(_KEY)
_CANONICAL_RE = re.compile(rf"{DOMAIN}:[^:]+(?::site|:[a-z]+:[^:]+)?:{_KEY}")
_DEV = r"(?P<dev>[^:]+)"
_NODE = rf"(?P<node>{DOMAIN}:[^:]+:(?!site:)[a-z]+:[^:]+)"

# (entity domain, older unique ID, canonical template) for every older format.
_LEGACY_UNIQUE_IDS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "binary_sensor",
        re.compile(rf"(?!{DOMAIN}:){_DEV}_online"),
        f"{DOMAIN}:{{dev}}:online",
    ),
    (
        "sensor",
        re.compile(rf"{DOMAIN}:{_DEV}:energy_total"),
        f"{DOMAIN}:{{dev}}:site:energy_total",
    ),
    (
        "number",
        re.compile(rf"{DOMAIN}:{_DEV}:power_limit"),
        f"{DOMAIN}:{{dev}}:site:power_limit",
    ),
    (
        "sensor",
        re.compile(rf"{DOMAIN}:{_DEV}:installation:info"),
        f"{DOMAIN}:{{dev}}:site:info",
    ),
    (
        "sensor",
        re.compile(rf"{DOMAIN}:{_DEV}:radio_frames"),
        f"{DOMAIN}:{{dev}}:frames_heard",
    ),
    (
        "sensor",
        re.compile(rf"{_NODE}:boost:(?P<key>end|minutes_remaining)"),
        "{node}:boost_{key}",
    ),
    ("climate", re.compile(_NODE), "{node}:climate"),
    (
        "sensor",
        re.compile(rf"(?!{DOMAIN}:){_DEV}:(?P<addr>[^:]+):energy"),
        f"{DOMAIN}:{{dev}}:htr:{{addr}}:energy",
    ),
)


def _key(suffix: Any) -> str:
    """Return ``suffix`` without its leading colon; it must be one snake_case token."""

    key = str(suffix).removeprefix(":")
    if not _KEY_RE.fullmatch(key):
        raise ValueError(f"unique ID key must be one snake_case token: {suffix!r}")
    return key


def _dev(dev_id: Any) -> str:
    """Return the normalised gateway id; raise ValueError when it is missing."""

    dev = normalize_node_addr(dev_id)
    if not dev:
        raise ValueError("dev_id must be provided")
    return dev


def build_heater_unique_id(
    dev_id: Any,
    node_type: Any,
    addr: Any,
    *,
    suffix: str | None = None,
) -> str:
    """Return the unique ID for a node entity; ``addr`` is its node_key, ``suffix`` its key."""

    dev = normalize_node_addr(dev_id)
    node = normalize_node_type(node_type)
    address = normalize_node_addr(addr)
    if not dev or not node or not address:
        raise ValueError("dev_id, node_type and addr must be provided")

    suffix_str = f":{_key(suffix)}" if suffix else ""
    return f"{DOMAIN}:{dev}:{node}:{address}{suffix_str}"


def node_key(inventory: Any, node_type: Any, addr: Any) -> str:
    """Return a node's ID key: its serial when the inventory knows one, else its address."""

    serial = None if inventory is None else inventory.node_serial(node_type, addr)
    return serial or normalize_node_addr(addr, use_default_when_falsey=True)


def build_node_device_identifier(dev_id: Any, key: Any) -> tuple[str, str, str]:
    """Return the device registry identifier of the node with ID key ``key``."""

    return (DOMAIN, str(dev_id), str(key))


def rekey_unique_id(
    unique_id: str, dev_id: Any, node_type: Any, old_key: Any, new_key: Any
) -> str | None:
    """Return ``unique_id`` moved from node key ``old_key`` to ``new_key``, else None."""

    old = build_heater_unique_id(dev_id, node_type, old_key)
    if unique_id != old and not unique_id.startswith(f"{old}:"):
        return None
    return build_heater_unique_id(dev_id, node_type, new_key) + unique_id[len(old) :]


def build_heater_energy_unique_id(dev_id: Any, node_type: Any, addr: Any) -> str:
    """Return the canonical unique ID for a heater energy sensor."""

    return build_heater_unique_id(dev_id, node_type, addr, suffix=":energy")


def build_power_monitor_unique_id(
    dev_id: Any,
    addr: Any,
    *,
    suffix: str | None = None,
) -> str:
    """Return the canonical unique ID for a power monitor node or entity."""

    return build_heater_unique_id(dev_id, "pmo", addr, suffix=suffix)


def build_power_monitor_energy_unique_id(dev_id: Any, addr: Any) -> str:
    """Return the canonical unique ID for a power monitor energy sensor."""

    return build_power_monitor_unique_id(dev_id, addr, suffix=":energy")


def build_power_monitor_power_unique_id(dev_id: Any, addr: Any) -> str:
    """Return the canonical unique ID for a power monitor power sensor."""

    return build_power_monitor_unique_id(dev_id, addr, suffix=":power")


def build_installation_entity_unique_id(dev_id: Any, suffix: str) -> str:
    """Return the canonical unique ID for an installation-level entity."""
    return f"{DOMAIN}:{_dev(dev_id)}:site:{_key(suffix)}"


def build_gateway_entity_unique_id(dev_id: Any, suffix: str) -> str:
    """Return the canonical unique ID for a gateway-level entity."""
    return f"{DOMAIN}:{_dev(dev_id)}:{_key(suffix)}"


def is_canonical_unique_id(unique_id: str) -> bool:
    """Return True when ``unique_id`` has the shape of the one scheme above."""

    return _CANONICAL_RE.fullmatch(unique_id) is not None


def migrate_unique_id(domain: str, unique_id: str) -> str | None:
    """Return the canonical ID for an older ``domain`` entity ID, else None."""

    for legacy_domain, pattern, template in _LEGACY_UNIQUE_IDS:
        if legacy_domain == domain and (match := pattern.fullmatch(unique_id)):
            return template.format(**match.groupdict())
    return None


def build_cloud_unique_id(brand: str, username: str) -> str:
    """Return a cloud entry's unique ID: the case-folded account, backend-scoped."""

    account = username.strip().casefold()
    scope = backend_capabilities(brand).account_scope
    return f"{scope}:{account}" if scope else account
