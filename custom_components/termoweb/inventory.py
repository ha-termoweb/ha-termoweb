"""Inventory helpers for TermoWeb nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple

RawNodePayload = Any
PrebuiltNode = Any

__all__ = ["Inventory"]


@dataclass(frozen=True, slots=True)
class Inventory:
    """Represent immutable node inventory details."""

    _dev_id: str
    _payload: RawNodePayload
    _nodes: Tuple[PrebuiltNode, ...]

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
    def nodes(self) -> Tuple[PrebuiltNode, ...]:
        """Get the immutable tuple of node objects."""

        return self._nodes
