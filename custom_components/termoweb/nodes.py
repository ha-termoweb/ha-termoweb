"""Node model abstractions for TermoWeb devices."""

from __future__ import annotations

from collections.abc import Iterable
import logging
from typing import Any

_LOGGER = logging.getLogger(__name__)


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

        resolved_type = (node_type or self.NODE_TYPE or "").strip().lower()
        if not resolved_type:
            msg = "node_type must be provided"
            raise ValueError(msg)

        addr_str = str(addr).strip()
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

        return False


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


def build_node_inventory(raw_nodes: Any) -> list[Node]:
    """Return a list of :class:`Node` instances for the provided payload.

    ``raw_nodes`` is typically the JSON response from ``/mgr/nodes``.  Each
    entry is validated and normalised – unknown node types are logged at DEBUG
    level yet still represented using the base :class:`Node` class so that
    callers can account for the presence of the device.
    """

    inventory: list[Node] = []
    for index, payload in enumerate(_iter_node_payload(raw_nodes)):
        node_type = str(payload.get("type") or payload.get("node_type") or "").strip().lower()
        if not node_type:
            _LOGGER.debug("Skipping node with missing type at index %s: %s", index, payload)
            continue

        name = payload.get("name") or payload.get("title") or payload.get("label")
        addr = payload.get("addr") or payload.get("address")

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
