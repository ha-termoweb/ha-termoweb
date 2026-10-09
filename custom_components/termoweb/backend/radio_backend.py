"""Radio backend: heaters reached through the local ESP32 radio gateway."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
import logging
from typing import Any

from custom_components.termoweb.backend.base import (
    Backend,
    BackendCapabilities,
    WsClientProto,
)
from custom_components.termoweb.backend.radio_client import RadioClient
from custom_components.termoweb.backend.radio_ws import RadioListener
from custom_components.termoweb.inventory import Inventory

_LOGGER = logging.getLogger(__name__)


class RadioBackend(Backend):
    """Backend that serves the integration over the local radio gateway."""

    capabilities = BackendCapabilities(
        lock=True, power_limit=False, priority=False, energy_history=False
    )

    def create_ws_client(
        self,
        hass: Any,
        entry_id: str,
        dev_id: str,
        coordinator: Any,
        *,
        inventory: Inventory | None = None,
    ) -> WsClientProto:
        """Return the radio listener that plays the websocket client's role."""

        client = self.client
        if not isinstance(client, RadioClient):
            raise TypeError("RadioBackend requires a RadioClient")
        return RadioListener(
            hass,
            entry_id=entry_id,
            dev_id=dev_id,
            client=client,
            coordinator=coordinator,
            inventory=inventory,
        )

    async def fetch_hourly_samples(
        self,
        dev_id: str,
        nodes: Iterable[tuple[str, str]],
        start_local: datetime,
        end_local: datetime,
    ) -> dict[tuple[str, str], list[dict[str, Any]]]:
        """Return no samples: radio heaters keep no energy history."""

        _LOGGER.debug("Radio backend has no hourly energy history for %s", dev_id)
        return {}


__all__ = ["RadioBackend"]
