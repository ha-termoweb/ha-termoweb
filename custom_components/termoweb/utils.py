"""Utility helpers shared across the TermoWeb integration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
import math
from typing import TYPE_CHECKING, Any, cast

from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.loader import async_get_integration

from .const import DOMAIN

if TYPE_CHECKING:
    from .nodes import Node

HEATER_NODE_TYPES: frozenset[str] = frozenset({"htr", "acm"})


async def async_get_integration_version(hass: HomeAssistant) -> str:
    """Return the installed integration version string."""

    integration = await async_get_integration(hass, DOMAIN)
    return integration.version or "unknown"


def build_heater_energy_unique_id(
    dev_id: Any, node_type: Any, addr: Any
) -> str:
    """Return the canonical unique ID for a heater energy sensor."""

    dev = normalize_node_addr(dev_id)
    node = normalize_node_type(node_type)
    address = normalize_node_addr(addr)
    if not dev or not node or not address:
        raise ValueError("dev_id, node_type and addr must be provided")
    return f"{DOMAIN}:{dev}:{node}:{address}:energy"


def parse_heater_energy_unique_id(unique_id: str) -> tuple[str, str, str] | None:
    """Parse a heater energy sensor unique ID into its components."""

    if not isinstance(unique_id, str):
        return None
    stripped = unique_id.strip()
    if not stripped or not stripped.startswith(f"{DOMAIN}:"):
        return None
    try:
        domain, dev, node, address, metric = stripped.split(":", 4)
    except ValueError:
        return None
    if domain != DOMAIN or metric != "energy":
        return None
    if not dev or not node or not address:
        return None
    return dev, node, address


def ensure_node_inventory(
    record: Mapping[str, Any], *, nodes: Any | None = None
) -> list["Node"]:  # noqa: UP037
    """Return cached node inventory, rebuilding and caching when missing."""

    cacheable = isinstance(record, MutableMapping)
    cached = record.get("node_inventory")
    if isinstance(cached, list) and cached:
        cached_nodes: list["Node"] = []  # noqa: UP037
        for node in cached:
            if not hasattr(node, "as_dict"):
                continue
            node_type = normalize_node_type(getattr(node, "type", ""))
            addr = normalize_node_addr(getattr(node, "addr", ""))
            if not node_type or not addr:
                continue
            cached_nodes.append(cast("Node", node))
        if cached_nodes:
            if cacheable and len(cached_nodes) != len(cached):
                record["node_inventory"] = list(cached_nodes)
            return list(cached_nodes)

    payloads: list[Any] = []
    if nodes is not None:
        payloads.append(nodes)

    record_nodes = record.get("nodes")
    if record_nodes is not None and (not payloads or record_nodes is not payloads[0]):
        payloads.append(record_nodes)

    last_index = len(payloads) - 1
    for index, raw_nodes in enumerate(payloads):
        try:
            from .nodes import build_node_inventory as build_inventory  # noqa: PLC0415

            inventory = build_inventory(raw_nodes)
        except Exception:  # pragma: no cover - defensive  # noqa: BLE001
            inventory = []

        if cacheable and (inventory or index == last_index):
            record["node_inventory"] = list(inventory)

        if inventory:
            return list(inventory)

    if isinstance(cached, list):
        if cacheable and "node_inventory" not in record:
            record["node_inventory"] = []
        return []

    if cacheable and "node_inventory" not in record:
        record["node_inventory"] = []

    return []
def normalize_node_type(
    value: Any,
    *,
    default: str = "",
    use_default_when_falsey: bool = False,
) -> str:
    """Return ``value`` as a normalised node type string."""

    raw = value
    if use_default_when_falsey and not raw:
        raw = default

    try:
        normalized = str(raw).strip().lower()
    except Exception:  # pragma: no cover - defensive  # noqa: BLE001
        normalized = ""

    if normalized:
        return normalized

    if default and not use_default_when_falsey:
        try:
            return str(default).strip().lower()
        except Exception:  # pragma: no cover - defensive  # noqa: BLE001
            return ""

    return ""


def normalize_node_addr(
    value: Any,
    *,
    default: str = "",
    use_default_when_falsey: bool = False,
) -> str:
    """Return ``value`` as a normalised node address string."""

    raw = value
    if use_default_when_falsey and not raw:
        raw = default

    try:
        normalized = str(raw).strip()
    except Exception:  # pragma: no cover - defensive  # noqa: BLE001
        normalized = ""

    if normalized:
        return normalized

    if default and not use_default_when_falsey:
        try:
            return str(default).strip()
        except Exception:  # pragma: no cover - defensive  # noqa: BLE001
            return ""

    return ""


def _entry_gateway_record(
    hass: HomeAssistant | None, entry_id: str | None
) -> Mapping[str, Any] | None:
    """Return the mapping storing integration data for ``entry_id``."""

    if hass is None or entry_id is None:
        return None
    domain_data = hass.data.get(DOMAIN)
    if not isinstance(domain_data, Mapping):
        return None
    entry_data = domain_data.get(entry_id)
    if not isinstance(entry_data, Mapping):
        return None
    return entry_data


def build_gateway_device_info(
    hass: HomeAssistant | None,
    entry_id: str | None,
    dev_id: str,
    *,
    include_version: bool = True,
) -> DeviceInfo:
    """Return canonical ``DeviceInfo`` for the TermoWeb gateway."""

    identifiers = {(DOMAIN, str(dev_id))}
    info: DeviceInfo = DeviceInfo(
        identifiers=identifiers,
        manufacturer="TermoWeb",
        name="TermoWeb Gateway",
        model="Gateway/Controller",
        configuration_url="https://control.termoweb.net",
    )

    entry_data = _entry_gateway_record(hass, entry_id)
    if not entry_data:
        return info

    brand = entry_data.get("brand")
    if isinstance(brand, str) and brand.strip():
        info["manufacturer"] = brand.strip()

    version = entry_data.get("version")
    if include_version and version is not None:
        info["sw_version"] = str(version)

    coordinator = entry_data.get("coordinator")
    data: Mapping[str, Any] | None = None
    if coordinator is not None:
        data = getattr(coordinator, "data", None)
        if not isinstance(data, Mapping):
            data = None
    if data:
        gateway_data = data.get(str(dev_id))
        if isinstance(gateway_data, Mapping):
            raw = gateway_data.get("raw")
            if isinstance(raw, Mapping):
                model = raw.get("model")
                if model not in (None, ""):
                    info["model"] = str(model)

    return info


def addresses_by_node_type(
    nodes: Iterable["Node"],  # noqa: UP037
    *,
    known_types: Iterable[str] | None = None,
) -> tuple[dict[str, list[str]], set[str]]:
    """Return mapping of node type to address list, tracking unknown types."""

    known: set[str] | None = None
    if known_types is not None:
        known = {normalize_node_type(node_type) for node_type in known_types if node_type}

    result: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    unknown: set[str] = set()

    for node in nodes:
        node_type = normalize_node_type(getattr(node, "type", ""))
        if not node_type:
            continue
        addr = normalize_node_addr(getattr(node, "addr", ""))
        if not addr:
            continue
        type_seen = seen.setdefault(node_type, set())
        if addr in type_seen:
            continue
        type_seen.add(addr)
        result.setdefault(node_type, []).append(addr)
        if known is not None and node_type not in known:
            unknown.add(node_type)

    return result, unknown


def build_heater_address_map(
    nodes: Iterable[Any],
    *,
    heater_types: Iterable[str] | None = None,
) -> tuple[dict[str, list[str]], dict[str, set[str]]]:
    """Return mapping of heater node types to addresses and reverse lookup."""

    allowed_types: set[str]
    if heater_types is None:
        allowed_types = set(HEATER_NODE_TYPES)
    else:
        allowed_types = {
            normalize_node_type(node_type, use_default_when_falsey=True)
            for node_type in heater_types
            if normalize_node_type(node_type, use_default_when_falsey=True)
        }  # pragma: no cover - exercised indirectly in integration

    if not allowed_types:
        return {}, {}  # pragma: no cover - defensive

    by_type_raw, _ = addresses_by_node_type(
        nodes,
        known_types=allowed_types,
    )  # pragma: no cover - exercised via higher level integration tests

    by_type: dict[str, list[str]] = {
        node_type: list(addresses)
        for node_type, addresses in by_type_raw.items()
        if node_type in allowed_types and addresses
    }

    reverse: dict[str, set[str]] = {}
    for node_type, addresses in by_type.items():
        for address in addresses:
            reverse.setdefault(address, set()).add(node_type)

    return by_type, reverse


def float_or_none(value: Any) -> float | None:
    """Return value as ``float`` if possible, else ``None``.

    Converts integers, floats, and numeric strings to ``float`` while safely
    handling ``None`` and non-numeric inputs.
    """
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            num = float(value)
        else:
            string_val = str(value).strip()
            if not string_val:
                return None
            num = float(string_val)
        return num if math.isfinite(num) else None
    except Exception:  # noqa: BLE001
        return None


def normalize_heater_addresses(
    addrs: Iterable[Any] | Mapping[Any, Iterable[Any]] | None,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Return canonical heater addresses and compatibility aliases."""

    cleaned_map: dict[str, list[str]] = {}
    compat_aliases: dict[str, str] = {}

    if addrs is None:
        sources: Iterable[tuple[Any, Iterable[Any] | Any]] = []
    elif isinstance(addrs, Mapping):
        sources = addrs.items()
    else:
        sources = [("htr", addrs)]

    for raw_type, values in sources:
        node_type = normalize_node_type(
            raw_type,
            use_default_when_falsey=True,
        )
        if not node_type:
            continue

        alias_target: str | None = None
        if node_type in {"heater", "heaters", "htr"}:
            alias_target = "htr"
        if alias_target is not None and node_type != alias_target:
            compat_aliases[node_type] = alias_target
            node_type = alias_target

        if node_type not in HEATER_NODE_TYPES:
            continue

        if isinstance(values, str) or not isinstance(values, Iterable):
            candidates = [values]
        else:
            candidates = list(values)

        bucket = cleaned_map.setdefault(node_type, [])
        seen: set[str] = set(bucket)
        for candidate in candidates:
            addr = normalize_node_addr(
                candidate,
                use_default_when_falsey=True,
            )
            if not addr or addr in seen:
                continue
            seen.add(addr)
            bucket.append(addr)

    cleaned_map.setdefault("htr", [])
    compat_aliases["htr"] = "htr"

    return cleaned_map, compat_aliases
