"""Backend abstractions for brand-specific behavior."""

from __future__ import annotations

from abc import ABC, abstractmethod
from asyncio import CancelledError, Task
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
import logging
import typing
from typing import Any, Protocol

from homeassistant.util import dt as dt_util

from custom_components.termoweb.backend.sanitize import mask_identifier
from custom_components.termoweb.inventory import (
    Inventory,
    NodeDescriptor,
    normalize_node_addr,
    normalize_node_type,
)
from custom_components.termoweb.utils import float_or_none

_LOGGER = logging.getLogger(__name__)

_SAMPLE_COUNTER_SCALES: dict[str, float] = {
    "htr": 1000.0,
    "acm": 1000.0,
    "pmo": 3_600_000.0,
}


class HttpClientProto(Protocol):
    """Protocol for the HTTP client used by TermoWeb entities."""

    async def list_devices(self) -> list[dict[str, Any]]:
        """Return the list of devices associated with the account."""

    async def get_nodes(self, dev_id: str) -> Any:
        """Return the node description for the given device."""

    async def get_node_settings(self, dev_id: str, node: NodeDescriptor) -> Any:
        """Return settings for the specified node."""

    async def set_node_settings(
        self,
        dev_id: str,
        node: NodeDescriptor,
        *,
        mode: str | None = None,
        stemp: float | None = None,
        prog: list[int] | None = None,
        ptemp: list[float] | None = None,
        units: str = "C",
        boost_time: int | None = None,
        cancel_boost: bool = False,
    ) -> Any:
        """Update node settings for the specified node."""

    async def set_acm_boost_state(
        self,
        dev_id: str,
        addr: str | int,
        *,
        boost: bool,
        boost_time: int | None = None,
        stemp: float | None = None,
        units: str | None = None,
    ) -> Any:
        """Toggle accumulator boost state for the specified node."""

    async def get_node_samples(
        self,
        dev_id: str,
        node: NodeDescriptor,
        start: float,
        stop: float,
    ) -> list[dict[str, str | int]]:
        """Return historical samples for the specified node."""


class WsClientProto(Protocol):
    """Protocol for websocket clients used by the integration."""

    def start(self) -> Task[Any]:
        """Start the websocket client."""

    async def stop(self) -> None:
        """Stop the websocket client."""


@dataclass(slots=True)
class BoostContext:
    """Hint data used when determining boost cancellation behavior."""

    active: bool | None = None
    mode: str | None = None


class Backend(ABC):
    """Base class for brand-specific integration backends."""

    def __init__(self, *, brand: str, client: HttpClientProto) -> None:
        """Initialize backend metadata."""

        self._brand = brand
        self._client = client

    @property
    def brand(self) -> str:
        """Return the configured brand."""

        return self._brand

    @property
    def client(self) -> HttpClientProto:
        """Return the HTTP client associated with this backend."""

        return self._client

    async def set_node_settings(
        self,
        dev_id: str,
        node: NodeDescriptor,
        *,
        mode: str | None = None,
        stemp: float | None = None,
        prog: list[int] | None = None,
        ptemp: list[float] | None = None,
        units: str = "C",
        boost_context: BoostContext | None = None,
    ) -> Any:
        """Update node settings using the backend client."""

        await self.client.set_node_settings(
            dev_id,
            node,
            mode=mode,
            stemp=stemp,
            prog=prog,
            ptemp=ptemp,
            units=units,
        )

    async def set_acm_boost_state(
        self,
        dev_id: str,
        addr: str | int,
        *,
        boost: bool,
        boost_time: int | None = None,
        stemp: float | None = None,
        units: str | None = None,
    ) -> Any:
        """Toggle accumulator boost state using the backend client."""

        await self.client.set_acm_boost_state(
            dev_id,
            addr,
            boost=boost,
            boost_time=boost_time,
            stemp=stemp,
            units=units,
        )

    @abstractmethod
    def create_ws_client(
        self,
        hass: Any,
        entry_id: str,
        dev_id: str,
        coordinator: Any,
        *,
        inventory: Inventory | None = None,
    ) -> WsClientProto:
        """Create a websocket client for the given device."""

    @abstractmethod
    async def fetch_hourly_samples(
        self,
        dev_id: str,
        nodes: Iterable[tuple[str, str]],
        start_local: datetime,
        end_local: datetime,
    ) -> dict[tuple[str, str], list[dict[str, Any]]]:
        """Return normalised hourly samples grouped by node descriptor."""

    def _resolve_node_descriptor(self, node: NodeDescriptor) -> tuple[str, str]:
        """Return canonical ``(node_type, addr)`` for ``node``."""

        if isinstance(node, tuple) and len(node) == 2:
            node_type, addr = node
        else:
            node_type = getattr(node, "type", None)
            addr = getattr(node, "addr", None)
        normalized_type = normalize_node_type(
            node_type,
            use_default_when_falsey=True,
        )
        normalized_addr = normalize_node_addr(
            addr,
            use_default_when_falsey=True,
        )
        if not normalized_type or not normalized_addr:
            msg = f"Invalid node descriptor: {node!r}"
            raise ValueError(msg)
        return normalized_type, normalized_addr


def normalise_sample_records(
    node_type: str,
    records: Iterable[Mapping[str, typing.Any] | Any],
) -> list[dict[str, Any]]:
    """Return sorted sample dictionaries containing ``ts`` and ``energy_wh``."""

    scale = float(_SAMPLE_COUNTER_SCALES.get(node_type, 1000.0) or 1000.0)
    divider = scale / 1000.0 if scale else 1.0
    samples: list[dict[str, Any]] = []

    for record in records:
        if not isinstance(record, Mapping):
            continue
        timestamp = float_or_none(record.get("t"))
        if timestamp is None:
            continue
        counter = float_or_none(record.get("counter"))
        if counter is None:
            counter = (
                float_or_none(record.get("counter_max"))
                or float_or_none(record.get("counter_min"))
                or float_or_none(record.get("value"))
            )
        if counter is None:
            continue
        energy_wh = counter / divider if divider else counter
        sample: dict[str, Any] = {
            "ts": datetime.fromtimestamp(timestamp, tz=UTC),
            "energy_wh": energy_wh,
        }
        power = float_or_none(record.get("power"))
        if power is not None:
            sample["power_w"] = power
        samples.append(sample)

    samples.sort(key=lambda item: item.get("ts", datetime.min.replace(tzinfo=UTC)))
    return samples


async def fetch_normalised_hourly_samples(
    *,
    client: HttpClientProto,
    dev_id: str,
    nodes: Iterable[tuple[str, str]],
    start_local: datetime,
    end_local: datetime,
    logger: logging.Logger = _LOGGER,
    log_prefix: str = "backend",
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Return per-node normalised samples for ``nodes`` between ``start`` and ``end``."""

    start_epoch = dt_util.as_utc(start_local).timestamp()
    end_epoch = dt_util.as_utc(end_local).timestamp()
    if end_epoch <= start_epoch:
        return {}

    results: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for node_type, addr in nodes:
        normalized_type = normalize_node_type(
            node_type,
            use_default_when_falsey=True,
        )
        normalized_addr = normalize_node_addr(
            addr,
            use_default_when_falsey=True,
        )
        if not normalized_type or not normalized_addr:
            continue
        try:
            raw_samples = await client.get_node_samples(
                dev_id,
                (normalized_type, normalized_addr),
                start_epoch,
                end_epoch,
            )
        except CancelledError:
            raise
        except Exception as err:  # noqa: BLE001 - log and continue on failure
            logger.warning(
                "%s: failed to fetch samples for %s/%s node_type=%s: %s",
                log_prefix,
                mask_identifier(dev_id),
                mask_identifier(normalized_addr),
                normalized_type,
                err,
            )
            continue
        if not isinstance(raw_samples, Iterable):
            continue
        normalised = normalise_sample_records(normalized_type, raw_samples)
        if normalised:
            results[(normalized_type, normalized_addr)] = normalised
    return results
