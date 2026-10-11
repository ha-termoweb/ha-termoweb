"""Node models and node identifier normalisation for the gateway inventory."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import logging
from typing import Any

__all__ = [
    "NODE_CLASS_BY_TYPE",
    "AccumulatorNode",
    "HeaterNode",
    "Node",
    "NodeDescriptor",
    "PowerMonitorNode",
    "ThermostatNode",
    "build_node_inventory",
    "normalize_node_addr",
    "normalize_node_type",
]

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
    """Return ``value`` as a normalised node address string; ``None`` is invalid."""

    if value is None and not use_default_when_falsey:
        msg = "node address must not be None"
        raise ValueError(msg)
    return _normalize_node_identifier(
        value,
        default=default,
        use_default_when_falsey=use_default_when_falsey,
        lowercase=False,
    )


class Node:
    """Base representation of a TermoWeb node."""

    __slots__ = ("_node_name", "addr", "serial", "type")
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
        self.serial: str | None = None  # radio heaters: stable serial token
        self._node_name = ""
        self.name = name if name is not None else ""

    @property
    def name(self) -> str:
        """Return the friendly name for the node."""

        return self._node_name

    @name.setter
    def name(self, value: str | None) -> None:
        """Update the stored friendly name for the node."""
        self._node_name = str(value or "").strip()

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

    def default_name(self) -> str:
        """Return the fallback friendly name for the power monitor."""

        return self.name or f"Power Monitor {self.addr}"

    def sample_target(self) -> tuple[str, str]:
        """Return the canonical ``(node_type, addr)`` sample target."""

        return (self.type, self.addr)


class ThermostatNode(Node):
    """Thermostat node (type ``thm``)."""

    __slots__ = ()
    NODE_TYPE = "thm"

    def __init__(self, *, name: str | None, addr: str | int) -> None:
        """Initialise a thermostat node."""

        super().__init__(name=name, addr=addr)


NODE_CLASS_BY_TYPE: dict[str, type[Node]] = {
    HeaterNode.NODE_TYPE: HeaterNode,
    AccumulatorNode.NODE_TYPE: AccumulatorNode,
    PowerMonitorNode.NODE_TYPE: PowerMonitorNode,
    ThermostatNode.NODE_TYPE: ThermostatNode,
}


def _iter_node_payload(raw_nodes: Any) -> Iterable[dict[str, Any]]:
    """Yield node dictionaries from a ``{"nodes": [...]}`` payload or a node list."""

    node_list = raw_nodes.get("nodes") if isinstance(raw_nodes, Mapping) else raw_nodes
    if not isinstance(node_list, list):
        return
    yield from node_list


def _resolve_node_class(node_type: str) -> type[Node]:
    """Return the most appropriate node class for ``node_type``."""

    return NODE_CLASS_BY_TYPE.get(node_type, Node)


def build_node_inventory(raw_nodes: Any) -> list[Node]:
    """Return a list of :class:`Node` instances for the provided payload."""

    inventory: list[Node] = []
    for index, payload in enumerate(_iter_node_payload(raw_nodes)):
        node_type = normalize_node_type(
            payload.get("type"),
            use_default_when_falsey=True,
        )
        if not node_type:
            _LOGGER.debug(
                "Skipping node with missing type at index %s: %s",
                index,
                payload,
            )
            continue

        name = payload.get("name")
        addr = normalize_node_addr(
            payload.get("addr"),
            use_default_when_falsey=True,
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

        serial = payload.get("serial")
        if isinstance(serial, str) and serial.strip():
            node.serial = serial.strip()
        inventory.append(node)

    return inventory
