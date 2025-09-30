"""Node model abstractions and helpers for TermoWeb devices."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, MutableMapping
import logging
from typing import Any, cast

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)


def _normalize_node_identifier(
    value: Any,
    *,
    default: str = "",
    use_default_when_falsey: bool = False,
    lowercase: bool,
) -> str:
    """Return ``value`` as a normalised node identifier string."""

    raw = value
    if use_default_when_falsey and not raw:
        raw = default

    try:
        normalized = str(raw).strip()
    except Exception:  # pragma: no cover - defensive  # noqa: BLE001
        normalized = ""
    else:
        if lowercase:
            normalized = normalized.lower()

    if normalized:
        return normalized

    if default and not use_default_when_falsey:
        try:
            normalized_default = str(default).strip()
        except Exception:  # pragma: no cover - defensive  # noqa: BLE001
            return ""
        if lowercase:
            normalized_default = normalized_default.lower()
        return normalized_default

    return ""


def normalize_node_type(
    value: Any,
    *,
    default: str = "",
    use_default_when_falsey: bool = False,
) -> str:
    """Return ``value`` as a normalised node type string."""

    return _normalize_node_identifier(
        value,
        default=default,
        use_default_when_falsey=use_default_when_falsey,
        lowercase=True,
    )


def normalize_node_addr(
    value: Any,
    *,
    default: str = "",
    use_default_when_falsey: bool = False,
) -> str:
    """Return ``value`` as a normalised node address string."""

    return _normalize_node_identifier(
        value,
        default=default,
        use_default_when_falsey=use_default_when_falsey,
        lowercase=False,
    )


HEATER_NODE_TYPES: frozenset[str] = frozenset({"htr", "acm"})

_NODE_SECTION_IGNORE_KEYS = frozenset(
    {"dev_id", "name", "raw", "connected", "nodes", "nodes_by_type"}
)


def _existing_nodes_map(source: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Return a mapping of node type sections extracted from ``source``."""

    if not isinstance(source, Mapping):
        return {}

    sections: dict[str, dict[str, Any]] = {}

    raw_existing = source.get("nodes_by_type")
    if isinstance(raw_existing, Mapping):
        for node_type, section in raw_existing.items():
            if isinstance(section, Mapping):
                sections[node_type] = dict(section)

    for key, value in source.items():
        if key in _NODE_SECTION_IGNORE_KEYS:
            continue
        if isinstance(value, Mapping):
            sections.setdefault(key, dict(value))

    return sections


class Node:
    """Base representation of a TermoWeb node."""

    __slots__ = ("_node_name", "addr", "type")
    NODE_TYPE = ""

    def __init__(
        self,
        *,
        name: str | None,
        addr: str | int,
        node_type: str | None = None,
    ) -> None:
        """Initialise a node with normalised metadata."""

        resolved_type = normalize_node_type(
            node_type,
            default=self.NODE_TYPE,
            use_default_when_falsey=True,
        )
        if not resolved_type:
            msg = "node_type must be provided"
            raise ValueError(msg)

        addr_str = normalize_node_addr(addr)
        if not addr_str:
            msg = "addr must be provided"
            raise ValueError(msg)

        self.addr = addr_str
        self.type = resolved_type
        self._node_name = ""
        self.name = name if name is not None else ""

    @property
    def name(self) -> str:
        """Return the friendly name for the node."""

        attr_name = getattr(self, "_attr_name", None)
        if isinstance(attr_name, str) and attr_name.strip():
            return attr_name
        return self._node_name

    @name.setter
    def name(self, value: str | None) -> None:
        """Update the stored friendly name for the node."""
        cleaned = str(value or "").strip()
        self._node_name = cleaned
        if hasattr(self, "_attr_name"):
            self._attr_name = cleaned

    def as_dict(self) -> dict[str, Any]:
        """Return a serialisable snapshot of core node metadata."""

        return {
            "name": self.name,
            "addr": self.addr,
            "type": self.type,
        }


NodeDescriptor = Node | tuple[str, str | int]


class HeaterNode(Node):
    """Heater node (type ``htr``)."""

    __slots__ = ()
    NODE_TYPE = "htr"

    def __init__(self, *, name: str | None, addr: str | int) -> None:
        """Initialise a heater node."""

        super().__init__(name=name, addr=addr)

    def supports_boost(self) -> bool:
        """Return whether the node natively exposes boost/runback control."""

        return False


class AccumulatorNode(HeaterNode):
    """Storage heater / accumulator node (type ``acm``)."""

    __slots__ = ()
    NODE_TYPE = "acm"

    def supports_boost(self) -> bool:
        """Return whether the accumulator exposes boost/runback."""

        return True


class PowerMonitorNode(Node):
    """Power monitor node (type ``pmo``)."""

    __slots__ = ()
    NODE_TYPE = "pmo"

    def __init__(self, *, name: str | None, addr: str | int) -> None:
        """Initialise a power monitor node."""

        super().__init__(name=name, addr=addr)

    def power_level(self) -> float:
        """Return the reported power level (stub)."""

        raise NotImplementedError


class ThermostatNode(Node):
    """Thermostat node (type ``thm``)."""

    __slots__ = ()
    NODE_TYPE = "thm"

    def __init__(self, *, name: str | None, addr: str | int) -> None:
        """Initialise a thermostat node."""

        super().__init__(name=name, addr=addr)

    def capabilities(self) -> dict[str, Any]:
        """Return thermostat capabilities (stub)."""

        raise NotImplementedError


NODE_CLASS_BY_TYPE: dict[str, type[Node]] = {
    HeaterNode.NODE_TYPE: HeaterNode,
    AccumulatorNode.NODE_TYPE: AccumulatorNode,
    PowerMonitorNode.NODE_TYPE: PowerMonitorNode,
    ThermostatNode.NODE_TYPE: ThermostatNode,
}


def _iter_node_payload(raw_nodes: Any) -> Iterable[dict[str, Any]]:
    """Yield node dictionaries from a payload returned by the API."""

    if isinstance(raw_nodes, dict):
        node_list = raw_nodes.get("nodes")
        if isinstance(node_list, list):
            for entry in node_list:
                if isinstance(entry, dict):
                    yield entry
        return

    if isinstance(raw_nodes, list):
        for entry in raw_nodes:
            if isinstance(entry, dict):
                yield entry


def _resolve_node_class(node_type: str) -> type[Node]:
    """Return the most appropriate node class for ``node_type``."""

    return NODE_CLASS_BY_TYPE.get(node_type, Node)


def _normalise_with_fallback(
    normalizer: Callable[..., str],
    *candidates: Any,
) -> str:
    """Return the first non-empty normalised value from ``candidates``."""

    for candidate in candidates:
        normalized = normalizer(candidate, use_default_when_falsey=True)
        if normalized:
            return normalized
    return ""


def build_node_inventory(raw_nodes: Any) -> list[Node]:
    """Return a list of :class:`Node` instances for the provided payload.

    ``raw_nodes`` is typically the JSON response from ``/mgr/nodes``.  Each
    entry is validated and normalised – unknown node types are logged at DEBUG
    level yet still represented using the base :class:`Node` class so that
    callers can account for the presence of the device.
    """

    inventory: list[Node] = []
    for index, payload in enumerate(_iter_node_payload(raw_nodes)):
        node_type = _normalise_with_fallback(
            normalize_node_type,
            payload.get("type"),
            payload.get("node_type"),
        )
        if not node_type:
            _LOGGER.debug("Skipping node with missing type at index %s: %s", index, payload)
            continue

        name = payload.get("name") or payload.get("title") or payload.get("label")
        addr = _normalise_with_fallback(
            normalize_node_addr,
            payload.get("addr"),
            payload.get("address"),
        )

        node_cls = _resolve_node_class(node_type)
        if node_cls is Node:
            _LOGGER.debug("Unsupported node type '%s' encountered", node_type)

        kwargs: dict[str, Any] = {"name": name, "addr": addr}
        if node_cls is Node:
            kwargs["node_type"] = node_type
        try:
            node = node_cls(**kwargs)
        except (TypeError, ValueError) as err:  # pragma: no cover - defensive
            _LOGGER.debug("Failed to initialise node %s at index %s: %s", payload, index, err)
            continue

        inventory.append(node)

    return inventory


def ensure_node_inventory(
    record: Mapping[str, Any], *, nodes: Any | None = None
) -> list[Node]:
    """Return cached node inventory, rebuilding and caching when missing."""

    cacheable = isinstance(record, MutableMapping)
    cached = record.get("node_inventory")
    if isinstance(cached, list) and cached:
        cached_nodes: list[Node] = []
        for node in cached:
            if not hasattr(node, "as_dict"):
                continue
            node_type = normalize_node_type(getattr(node, "type", ""))
            addr = normalize_node_addr(getattr(node, "addr", ""))
            if not node_type or not addr:
                continue
            cached_nodes.append(cast(Node, node))
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
            inventory = build_node_inventory(raw_nodes)
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


def build_heater_energy_unique_id(dev_id: Any, node_type: Any, addr: Any) -> str:
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


def addresses_by_node_type(
    nodes: Iterable[Node],
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


def collect_heater_sample_addresses(
    record: Mapping[str, Any] | None,
    *,
    coordinator: Any | None = None,
) -> tuple[list[Node], dict[str, list[str]], dict[str, str]]:
    """Return inventory and canonical heater sample subscription addresses."""

    nodes_payload: Any | None = None
    cache_record: MutableMapping[str, Any] | None = None

    if isinstance(record, MutableMapping):
        cache_record = record
        nodes_payload = record.get("nodes")
    elif isinstance(record, Mapping):
        nodes_payload = record.get("nodes")

    inventory = ensure_node_inventory(cache_record or {}, nodes=nodes_payload)

    raw_map, _ = addresses_by_node_type(
        inventory,
        known_types=HEATER_NODE_TYPES,
    )
    addr_map: dict[str, list[str]] = {
        node_type: list(addresses)
        for node_type, addresses in raw_map.items()
        if node_type in HEATER_NODE_TYPES and addresses
    }

    if not addr_map.get("htr") and coordinator is not None:
        fallback: Iterable[Any] | None = None
        if hasattr(coordinator, "_addrs"):
            try:
                fallback = coordinator._addrs()  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - defensive
                fallback = None
        if fallback:
            normalised: list[str] = []
            for candidate in fallback:
                addr = normalize_node_addr(candidate)
                if not addr:
                    continue
                normalised.append(addr)
            if normalised:
                addr_map["htr"] = normalised

    normalized_map, compat = normalize_heater_addresses(addr_map)

    return list(inventory), normalized_map, compat


def heater_sample_subscription_targets(
    addrs: Mapping[Any, Iterable[Any]] | Iterable[Any] | None,
) -> list[tuple[str, str]]:
    """Return canonical heater sample subscription target pairs."""

    normalized_map, _ = normalize_heater_addresses(addrs)
    if not any(normalized_map.values()):
        return []

    other_types = sorted(node_type for node_type in normalized_map if node_type != "htr")
    order = ["htr", *other_types]
    return [
        (node_type, addr)
        for node_type in order
        for addr in normalized_map.get(node_type, []) or []
    ]
