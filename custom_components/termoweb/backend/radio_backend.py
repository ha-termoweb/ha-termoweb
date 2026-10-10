"""Radio backend: heaters reached through the local ESP32 radio gateway."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
import logging
from typing import Any

from custom_components.termoweb.backend.base import (
    Backend,
    BackendCapabilities,
    WsClientProto,
)
from custom_components.termoweb.backend.radio.link import supports_survey
from custom_components.termoweb.backend.radio_client import RadioClient
from custom_components.termoweb.backend.radio_ws import RadioListener
from custom_components.termoweb.const import CONF_RADIO_TYPE, RADIO_TYPE_ESP32
from custom_components.termoweb.inventory import Inventory

_LOGGER = logging.getLogger(__name__)


def radio_diagnostics(client: Any, entry_data: Mapping[str, Any]) -> dict[str, Any]:
    """Return radio link facts for diagnostics; no MAC or network id."""

    info = client.gateway_info
    section: dict[str, Any] = {
        "radio_type": entry_data.get(CONF_RADIO_TYPE, RADIO_TYPE_ESP32),
        "dialect": client.dialect.name,
        "connected": client.connected,
        "listen_only": client.listen_only,
        "gateway": None,
    }
    if info is not None:
        section["gateway"] = {
            "firmware": info.version,
            "freq": info.freq,
            "sync": info.sync,
            "dialect": info.dialect,
            "autoack": info.autoack,
            "station_id": info.station_id,
            "survey": supports_survey(info),
        }
    return section


class RadioBackend(Backend):
    """Backend that serves the integration over the local radio gateway."""

    capabilities = BackendCapabilities(
        lock=True,
        power_limit=True,
        priority=True,
        energy_history=False,
        energy=True,
        local_radio=True,
        options_flow=True,
        web_portal=False,
    )

    def diagnostics(self, entry_data: Mapping[str, Any]) -> dict[str, Any] | None:
        """Return the radio link facts."""

        return radio_diagnostics(self.client, entry_data)

    def create_ws_client(
        self,
        hass: Any,
        entry_id: str,
        dev_id: str,
        coordinator: Any,
        *,
        inventory: Inventory,
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


__all__ = ["RadioBackend", "radio_diagnostics"]
