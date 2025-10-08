"""Inventory helpers and node model abstractions for TermoWeb."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import logging
from typing import Any

RawNodePayload = Any
PrebuiltNode = Any

_NODE_SECTION_IGNORE_KEYS = frozenset(
    {"dev_id", "name", "raw", "connected", "nodes", "nodes_by_type"}
)

_SNAPSHOT_NAME_CANDIDATE_KEYS = (
    "name",
    "label",
    "title",
    "display_name",
    "device_name",
    "friendly_name",
    "heater_name",
    "alias",
    "room",
)


__all__ = [
    "NODE_CLASS_BY_TYPE",
    "AccumulatorNode",
    "HeaterNode",
    "Inventory",
    "Node",
    "NodeDescriptor",
    "PowerMonitorNode",
    "ThermostatNode",
    "_normalize_node_identifier",
    "build_node_inventory",
    "normalize_node_addr",
    "normalize_node_type",
]


_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Inventory:
    """Represent immutable node inventory details."""

    _dev_id: str
    _payload: RawNodePayload
    _nodes: tuple[PrebuiltNode, ...]

    def __init__(
        self,
        dev_id: str,
        payload: RawNodePayload,
        nodes: Iterable[PrebuiltNode],
    ) -> None:
        """Initialize the inventory container."""

        object.__setattr__(self, "_dev_id", dev_id)
        object.__setattr__(self, "_payload", payload)
        object.__setattr__(self, "_nodes", tuple(nodes))

    @property
    def dev_id(self) -> str:
        """Get the device identifier."""

        return self._dev_id

    @property
    def payload(self) -> RawNodePayload:
        """Get the raw node payload."""

        return self._payload

    @property
    def nodes(self) -> tuple[PrebuiltNode, ...]:
        """Get the immutable tuple of node objects."""

        return self._nodes


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


def _iter_snapshot_sections(
    sections: Mapping[str, Any],
    seen: set[tuple[str, str]],
) -> Iterable[dict[str, Any]]:
    """Yield node payloads derived from snapshot-style ``sections``."""

    for node_type, payload in sections.items():
        if not isinstance(node_type, str):
            continue
        normalized_type = node_type.strip()
        if not normalized_type or not isinstance(payload, Mapping):
            continue
        for entry in _iter_snapshot_section(normalized_type, payload):
            addr = entry.get("addr")
            if not isinstance(addr, str):
                continue
            key = (normalized_type, addr)
            if key in seen:
                continue
            seen.add(key)
            yield entry


def _iter_snapshot_section(
    node_type: str, section: Mapping[str, Any]
) -> Iterable[dict[str, Any]]:
    """Yield node dictionaries for a single node type section."""

    addresses = _collect_snapshot_addresses(section)
    for addr in sorted(addresses):
        entry: dict[str, Any] = {"type": node_type, "addr": addr}
        name = _extract_snapshot_name(addresses[addr])
        if name:
            entry["name"] = name
        yield entry


def _collect_snapshot_addresses(
    section: Mapping[str, Any]
) -> dict[str, list[Mapping[str, Any]]]:
    """Return mapping of addresses to candidate payloads from ``section``."""

    addresses: dict[str, list[Mapping[str, Any]]] = {}

    addrs = section.get("addrs")
    if isinstance(addrs, (list, tuple, set)):
        for candidate in addrs:
            addr = normalize_node_addr(candidate)
            if addr:
                addresses.setdefault(addr, [])

    for value in section.values():
        if not isinstance(value, Mapping):
            continue
        for addr_key, payload in value.items():
            addr = normalize_node_addr(addr_key)
            if not addr:
                continue
            bucket = addresses.setdefault(addr, [])
            if isinstance(payload, Mapping):
                bucket.append(payload)
            elif isinstance(payload, str):
                bucket.append({"name": payload})

    return addresses


def _extract_snapshot_name(payloads: Iterable[Mapping[str, Any]]) -> str:
    """Return best candidate name extracted from ``payloads``."""

    queue = [payload for payload in payloads if isinstance(payload, Mapping)]
    seen: set[int] = set()

    while queue:
        payload = queue.pop(0)
        payload_id = id(payload)
        if payload_id in seen:
            continue
        seen.add(payload_id)

        for key in _SNAPSHOT_NAME_CANDIDATE_KEYS:
            value = payload.get(key)
            if isinstance(value, str):
                candidate = value.strip()
                if candidate:
                    return candidate

        queue.extend(
            nested for nested in payload.values() if isinstance(nested, Mapping)
        )

    return ""


def _iter_node_payload(raw_nodes: Any) -> Iterable[dict[str, Any]]:
    """Yield node dictionaries from a payload returned by the API."""

    if isinstance(raw_nodes, dict):
        node_list = raw_nodes.get("nodes")
        if isinstance(node_list, list):
            for entry in node_list:
                if isinstance(entry, dict):
                    yield entry
            return

        seen: set[tuple[str, str]] = set()

        if isinstance(node_list, dict):
            yield from _iter_snapshot_sections(node_list, seen)

        sections = _existing_nodes_map(raw_nodes)
        if sections:
            yield from _iter_snapshot_sections(sections, seen)
            if seen:
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
    """Return a list of :class:`Node` instances for the provided payload."""

    inventory: list[Node] = []
    for index, payload in enumerate(_iter_node_payload(raw_nodes)):
        node_type = _normalise_with_fallback(
            normalize_node_type,
            payload.get("type"),
            payload.get("node_type"),
        )
        if not node_type:
            _LOGGER.debug(
                "Skipping node with missing type at index %s: %s",
                index,
                payload,
            )
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

        try:
            if node_cls is Node:
                node = node_cls(name=name, addr=addr, node_type=node_type)
            else:
                node = node_cls(name=name, addr=addr)
        except (TypeError, ValueError) as err:  # pragma: no cover - defensive
            _LOGGER.debug(
                "Failed to initialise node %s at index %s: %s",
                payload,
                index,
                err,
            )
            continue

        inventory.append(node)

    return inventory
