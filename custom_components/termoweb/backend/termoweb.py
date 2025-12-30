"""TermoWeb backend implementation."""

from __future__ import annotations

from asyncio import CancelledError
from collections.abc import Iterable
from datetime import datetime
import logging
from typing import Any, cast

from custom_components.termoweb.backend.base import (
    Backend,
    EnergySample,
    WsClientProto,
    fetch_normalised_hourly_samples,
    normalise_energy_samples,
)
from custom_components.termoweb.backend.sanitize import mask_identifier
from custom_components.termoweb.backend.ws_client import WebSocketClient
from custom_components.termoweb.inventory import Inventory

_LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - exercised via backend tests
    from custom_components.termoweb.backend.termoweb_ws import (
        TermoWebWSClient as _TermoWebWSClient,
    )
except ImportError:  # pragma: no cover - exercised via backend tests
    _TermoWebWSClient = cast(type[Any], WebSocketClient)

TermoWebWSClient = _TermoWebWSClient


class TermoWebBackend(Backend):
    """Backend for the TermoWeb brand."""

    def _resolve_ws_client_cls(self) -> type[Any]:
        """Return the websocket client class compatible with this backend."""

        if isinstance(TermoWebWSClient, type):
            return TermoWebWSClient
        return WebSocketClient

    def create_ws_client(
        self,
        hass: Any,
        entry_id: str,
        dev_id: str,
        coordinator: Any,
        *,
        inventory: Inventory | None = None,
    ) -> WsClientProto:
        """Instantiate the unified websocket client for TermoWeb."""

        ws_cls = self._resolve_ws_client_cls()
        kwargs: dict[str, Any] = {
            "entry_id": entry_id,
            "dev_id": dev_id,
            "api_client": self.client,
            "coordinator": coordinator,
            "inventory": inventory,
        }
        if issubclass(ws_cls, WebSocketClient):
            kwargs["protocol"] = "socketio09"
        return ws_cls(
            hass,
            **kwargs,
        )

    async def fetch_hourly_samples(
        self,
        dev_id: str,
        nodes: Iterable[tuple[str, str]],
        start_local: datetime,
        end_local: datetime,
    ) -> dict[tuple[str, str], list[dict[str, Any]]]:
        """Return hourly samples for ``nodes`` using the REST API."""

        return await fetch_normalised_hourly_samples(
            client=self.client,
            dev_id=dev_id,
            nodes=nodes,
            start_local=start_local,
            end_local=end_local,
            logger=_LOGGER,
            log_prefix="termoweb",  # identifies the backend in shared logs
        )

    async def get_energy_samples(
        self,
        dev_id: str,
        node: Any,
        start_s: int,
        end_s: int,
    ) -> list[EnergySample]:
        """Return energy samples normalised to epoch seconds and kWh."""

        normalized_type, normalized_addr = self._resolve_node_descriptor(node)
        if end_s <= start_s:
            return []
        start_ms = int(max(start_s, 0) * 1000)
        end_ms = int(max(end_s, 0) * 1000)
        try:
            raw_samples = await self.client.get_node_samples(
                dev_id,
                (normalized_type, normalized_addr),
                start_ms,
                end_ms,
            )
        except CancelledError:
            raise
        except Exception as err:  # noqa: BLE001 - propagate via logging
            _LOGGER.warning(
                "%s energy samples failed for %s/%s node_type=%s: %s",
                self.brand,
                mask_identifier(dev_id),
                mask_identifier(normalized_addr),
                normalized_type,
                err,
            )
            return []

        context = (
            f"{self.brand} energy {mask_identifier(dev_id)}/"
            f"{mask_identifier(normalized_addr)} ({normalized_type})"
        )
        return normalise_energy_samples(
            normalized_type,
            raw_samples or [],
            logger=_LOGGER,
            context=context,
            timestamp_divisor=1000.0,
            detect_millisecond_overflow=False,
        )
