"""Node model abstractions for TermoWeb devices."""

from __future__ import annotations

from typing import Any

from .const import BRAND_DUCAHEAT


class Node:
    """Base representation of a TermoWeb node."""

    __slots__ = ("_node_name", "addr", "brand", "type")
    NODE_TYPE = ""

    def __init__(
        self,
        *,
        name: str | None,
        addr: str | int,
        brand: str,
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

        brand_str = str(brand or "").strip()
        if not brand_str:
            msg = "brand must be provided"
            raise ValueError(msg)

        self.addr = addr_str
        self.type = resolved_type
        self.brand = brand_str
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
            "brand": self.brand,
        }


class HeaterNode(Node):
    """Heater node (type ``htr``)."""

    __slots__ = ()
    NODE_TYPE = "htr"

    def __init__(self, *, name: str | None, addr: str | int, brand: str) -> None:
        """Initialise a heater node."""

        super().__init__(name=name, addr=addr, brand=brand)

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


class DucaheatAccum(AccumulatorNode):
    """Accumulator node for the Ducaheat brand supporting boost."""

    __slots__ = ()

    def __init__(
        self,
        *,
        name: str | None,
        addr: str | int,
        brand: str = BRAND_DUCAHEAT,
    ) -> None:
        """Initialise a Ducaheat accumulator node."""

        super().__init__(name=name, addr=addr, brand=brand)

    def supports_boost(self) -> bool:
        """Return ``True`` to indicate boost is available."""

        return True


class PowerMonitorNode(Node):
    """Power monitor node (type ``pmo``)."""

    __slots__ = ()
    NODE_TYPE = "pmo"

    def __init__(self, *, name: str | None, addr: str | int, brand: str) -> None:
        """Initialise a power monitor node."""

        super().__init__(name=name, addr=addr, brand=brand)

    def power_level(self) -> float:
        """Return the reported power level (stub)."""

        raise NotImplementedError


class ThermostatNode(Node):
    """Thermostat node (type ``thm``)."""

    __slots__ = ()
    NODE_TYPE = "thm"

    def __init__(self, *, name: str | None, addr: str | int, brand: str) -> None:
        """Initialise a thermostat node."""

        super().__init__(name=name, addr=addr, brand=brand)

    def capabilities(self) -> dict[str, Any]:
        """Return thermostat capabilities (stub)."""

        raise NotImplementedError
