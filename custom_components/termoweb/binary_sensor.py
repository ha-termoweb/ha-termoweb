"""Binary sensor entities for TermoWeb gateway connectivity and heaters."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.core import callback
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN, signal_ws_status
from .coordinator import StateCoordinator
from .entity import GatewayDispatcherEntity
from .heater import (
    HeaterNodeBase,
    iter_boostable_heater_nodes,
    log_skipped_nodes,
    prepare_heater_platform_data,
)
from .utils import build_gateway_device_info

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass, entry, async_add_entities):
    """Set up one connectivity binary sensor per TermoWeb hub (dev_id)."""
    data = hass.data[DOMAIN][entry.entry_id]
    coord: StateCoordinator = data["coordinator"]
    dev_id = data["dev_id"]
    gateway = GatewayOnlineBinarySensor(coord, entry.entry_id, dev_id)

    _, nodes_by_type, _, resolve_name = prepare_heater_platform_data(
        data,
        default_name_simple=lambda addr: f"Node {addr}",
    )

    boost_entities: list[BinarySensorEntity] = []
    for node_type, _node, addr_str, base_name in iter_boostable_heater_nodes(
        nodes_by_type, resolve_name
    ):
        unique_id = f"{DOMAIN}:{dev_id}:{node_type}:{addr_str}:boost_active"
        boost_entities.append(
            HeaterBoostActiveBinarySensor(
                coord,
                entry.entry_id,
                dev_id,
                addr_str,
                f"{base_name} Boost Active",
                unique_id,
                device_name=base_name,
                node_type=node_type,
            )
        )

    if boost_entities:
        _LOGGER.debug(
            "Adding %d TermoWeb heater boost binary sensors", len(boost_entities)
        )

    log_skipped_nodes("binary_sensor", nodes_by_type, logger=_LOGGER)
    async_add_entities([gateway, *boost_entities])


class GatewayOnlineBinarySensor(
    GatewayDispatcherEntity,
    CoordinatorEntity[StateCoordinator],
    BinarySensorEntity,
):
    """Connectivity sensor for the TermoWeb hub (gateway)."""

    _attr_device_class = BinarySensorDeviceClass.CONNECTIVITY
    _attr_should_poll = False

    def __init__(
        self, coordinator: StateCoordinator, entry_id: str, dev_id: str
    ) -> None:
        """Initialise the connectivity binary sensor."""
        super().__init__(coordinator)
        self._entry_id = entry_id
        self._dev_id = str(dev_id)
        self._attr_name = "TermoWeb Gateway Online"
        self._attr_unique_id = f"{self._dev_id}_online"

    @property
    def gateway_signal(self) -> str:
        """Return the dispatcher signal for gateway websocket status."""

        return signal_ws_status(self._entry_id)

    def _ws_state(self) -> dict[str, Any]:
        """Return the latest websocket status payload for this device."""
        rec = self.hass.data.get(DOMAIN, {}).get(self._entry_id, {}) or {}
        return (rec.get("ws_state") or {}).get(self._dev_id, {})

    @property
    def is_on(self) -> bool:
        """Return True when the integration reports the gateway is online."""
        data = (self.coordinator.data or {}).get(self._dev_id, {}) or {}
        return bool(data.get("connected"))

    @property
    def device_info(self) -> DeviceInfo:
        """Return Home Assistant device metadata for the gateway."""
        return build_gateway_device_info(self.hass, self._entry_id, self._dev_id)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return additional gateway diagnostics and websocket state."""
        data = (self.coordinator.data or {}).get(self._dev_id, {}) or {}
        ws = self._ws_state()
        return {
            "dev_id": self._dev_id,
            "name": data.get("name"),
            "connected": data.get("connected"),
            "ws_status": ws.get("status"),
            "ws_last_event_at": ws.get("last_event_at"),
            "ws_healthy_minutes": ws.get("healthy_minutes"),
            "raw": data.get("raw") or {},
        }

    @callback
    def _handle_gateway_dispatcher(self, payload: dict[str, Any]) -> None:
        """Handle websocket status broadcasts from the integration."""
        if payload.get("dev_id") != self._dev_id:
            return
        self.schedule_update_ha_state()


class HeaterBoostActiveBinarySensor(HeaterNodeBase, BinarySensorEntity):
    """Binary sensor indicating whether a heater boost is active."""

    _attr_device_class = getattr(BinarySensorDeviceClass, "HEAT", "heat")

    def __init__(
        self,
        coordinator: StateCoordinator,
        entry_id: str,
        dev_id: str,
        addr: str,
        name: str,
        unique_id: str,
        *,
        device_name: str | None = None,
        node_type: str | None = None,
    ) -> None:
        """Initialise the boost activity binary sensor."""

        super().__init__(
            coordinator,
            entry_id,
            dev_id,
            addr,
            name,
            unique_id,
            device_name=device_name,
            node_type=node_type,
        )

    @property
    def is_on(self) -> bool | None:
        """Return True when the heater boost is active."""

        state = self.boost_state()
        return state.active

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return boost metadata exposed alongside the binary state."""

        state = self.boost_state()
        return {
            "dev_id": self._dev_id,
            "addr": self._addr,
            "boost_minutes_remaining": state.minutes_remaining,
            "boost_end": state.end_iso,
            "boost_end_label": state.end_label,
        }
