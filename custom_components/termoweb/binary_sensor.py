"""Binary sensor entities for TermoWeb gateway connectivity and heaters."""

from __future__ import annotations

from collections.abc import Iterable
import logging
from typing import Any

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import CALLBACK_TYPE, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.event import async_track_point_in_utc_time
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from custom_components.termoweb.boost import supports_boost
from custom_components.termoweb.const import signal_radio_pairing_window
from custom_components.termoweb.coordinator.state import StateCoordinator
from custom_components.termoweb.domain import GatewayConnectionState
from custom_components.termoweb.domain.ids import HEATER_NODE_TYPES
from custom_components.termoweb.entity import (
    BoostState,
    SettingsResolver,
    build_settings_resolver,
    derive_boost_state_from_domain,
    log_skipped_nodes,
)
from custom_components.termoweb.identifiers import (
    build_gateway_entity_unique_id,
    build_heater_unique_id,
    node_key,
)
from custom_components.termoweb.inventory import Inventory
from custom_components.termoweb.nodes import normalize_node_addr, normalize_node_type
from custom_components.termoweb.radio_autopair import (
    pairing_supported,
    pairing_window_end,
)
from custom_components.termoweb.runtime import require_runtime
from custom_components.termoweb.utils import (
    build_gateway_device_info,
    build_node_device_info,
)

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass, entry, async_add_entities):
    """Set up connectivity and boost binary sensors for a config entry."""
    runtime = require_runtime(hass, entry.entry_id)
    coord: StateCoordinator = runtime.coordinator
    dev_id = runtime.dev_id

    gateway = GatewayOnlineBinarySensor(coord, entry.entry_id, dev_id)

    inventory = runtime.inventory

    boost_entities: list[BinarySensorEntity] = []
    for node_type, addr_str, base_name in _iter_boostable_inventory_nodes(inventory):
        unique_id = build_heater_unique_id(
            dev_id,
            node_type,
            node_key(inventory, node_type, addr_str),
            suffix=":boost_active",
        )
        settings_resolver = build_settings_resolver(
            coord,
            dev_id,
            node_type,
            addr_str,
        )
        boost_entities.append(
            HeaterBoostActiveBinarySensor(
                coord,
                entry.entry_id,
                dev_id,
                node_type,
                addr_str,
                name=None,
                unique_id=unique_id,
                inventory=inventory,
                settings_resolver=settings_resolver,
                device_name=base_name,
            )
        )

    if boost_entities:
        _LOGGER.debug(
            "Adding %d TermoWeb heater boost binary sensors", len(boost_entities)
        )

    log_skipped_nodes("binary_sensor", inventory, logger=_LOGGER)
    pairing: list[BinarySensorEntity] = []
    if pairing_supported(runtime):
        pairing.append(PairingWindowBinarySensor(entry, dev_id))
    async_add_entities([gateway, *pairing, *boost_entities])


class GatewayOnlineBinarySensor(
    CoordinatorEntity[StateCoordinator], BinarySensorEntity
):
    """Connectivity sensor for the TermoWeb hub (gateway)."""

    _attr_has_entity_name = True
    _attr_device_class = BinarySensorDeviceClass.CONNECTIVITY
    _attr_translation_key = "gateway_online"
    _attr_should_poll = False

    def __init__(
        self, coordinator: StateCoordinator, entry_id: str, dev_id: str
    ) -> None:
        """Initialise the connectivity binary sensor."""
        super().__init__(coordinator)
        self._entry_id = entry_id
        self._dev_id = str(dev_id)
        self._attr_unique_id = build_gateway_entity_unique_id(self._dev_id, "online")

    def _gateway_connection_state(self) -> GatewayConnectionState:
        """Return the gateway connection state for this device."""

        return self.coordinator.domain_view.get_gateway_connection_state()

    @property
    def is_on(self) -> bool:
        """Return True when the integration reports the gateway is online."""
        return bool(self._gateway_connection_state().connected)

    @property
    def device_info(self) -> DeviceInfo:
        """Return Home Assistant device metadata for the gateway."""
        return build_gateway_device_info(self.hass, self._entry_id, self._dev_id)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return additional gateway diagnostics and websocket state."""
        coordinator = self.coordinator
        name = getattr(coordinator, "gateway_name", None)
        model = getattr(coordinator, "gateway_model", None)
        connection_state = self._gateway_connection_state()
        return {
            "dev_id": self._dev_id,
            "name": name() if callable(name) else name,
            "connected": connection_state.connected,
            "model": model() if callable(model) else model,
            "ws_status": connection_state.status,
            "ws_last_event_at": connection_state.last_event_at,
            "ws_healthy_minutes": connection_state.healthy_minutes,
        }


class PairingWindowBinarySensor(BinarySensorEntity):
    """On while the radio gateway pairs heaters that are put into pairing mode."""

    _attr_has_entity_name = True
    _attr_translation_key = "pairing_open"
    _attr_should_poll = False

    def __init__(self, entry: ConfigEntry, dev_id: str) -> None:
        """Initialise the sensor for the gateway of ``entry``."""
        self._entry = entry
        self._dev_id = str(dev_id)
        self._attr_unique_id = build_gateway_entity_unique_id(
            self._dev_id, "pairing_open"
        )
        self._unsub_end: CALLBACK_TYPE | None = None

    async def async_added_to_hass(self) -> None:
        """Follow window changes and close the window on time."""
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                signal_radio_pairing_window(self._entry.entry_id),
                self._async_window_changed,
            )
        )
        self.async_on_remove(self._cancel_end)
        self._schedule_end()

    @callback
    def _cancel_end(self) -> None:
        """Cancel the timer that closes the window."""
        if self._unsub_end is not None:
            self._unsub_end()
            self._unsub_end = None

    @callback
    def _schedule_end(self) -> None:
        """Update the state when the open window ends."""
        self._cancel_end()
        end = pairing_window_end(self.hass, self._entry)
        if end is not None:
            self._unsub_end = async_track_point_in_utc_time(
                self.hass, self._async_window_changed, end
            )

    @callback
    def _async_window_changed(self, _now: Any = None) -> None:
        """Write the new state and follow the new window end."""
        self._unsub_end = None
        self._schedule_end()
        self.async_write_ha_state()

    @property
    def is_on(self) -> bool:
        """Return True while heaters in pairing mode are paired."""
        return pairing_window_end(self.hass, self._entry) is not None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return when the pairing window closes."""
        end = pairing_window_end(self.hass, self._entry)
        return {"ends_at": None if end is None else end.isoformat()}

    @property
    def device_info(self) -> DeviceInfo:
        """Return Home Assistant device metadata for the gateway."""
        return build_gateway_device_info(self.hass, self._entry.entry_id, self._dev_id)


class HeaterBoostActiveBinarySensor(
    CoordinatorEntity[StateCoordinator],
    BinarySensorEntity,
):
    """Binary sensor indicating whether a heater boost is active."""

    _attr_device_class = BinarySensorDeviceClass.HEAT
    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_translation_key = "boost_active"

    def __init__(
        self,
        coordinator: StateCoordinator,
        entry_id: str,
        dev_id: str,
        node_type: str,
        addr: str,
        name: str | None,
        unique_id: str,
        *,
        inventory: Inventory,
        settings_resolver: SettingsResolver,
        device_name: str | None = None,
    ) -> None:
        """Initialise the boost activity binary sensor."""

        super().__init__(coordinator)
        canonical_type = normalize_node_type(node_type, use_default_when_falsey=True)
        canonical_addr = normalize_node_addr(addr, use_default_when_falsey=True)

        self._entry_id = entry_id
        self._dev_id = str(dev_id)
        self._node_type = canonical_type
        self._addr = canonical_addr
        self._attr_unique_id = unique_id
        self._inventory = inventory
        self._settings_resolver = settings_resolver
        resolved_device_name = device_name or name
        self._device_name = resolved_device_name or ""

    @property
    def is_on(self) -> bool | None:
        """Return True when the heater boost is active."""

        return self.boost_state().active

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

    @property
    def available(self) -> bool:
        """Return True when the last update succeeded and the node is in inventory."""

        if not super().available:
            return False
        forward_map, _ = self._inventory.heater_address_map
        addresses = forward_map.get(self._node_type, [])
        return self._addr in addresses

    @property
    def device_info(self) -> DeviceInfo:
        """Expose Home Assistant device metadata for the heater."""

        return build_node_device_info(
            self.hass,
            self._entry_id,
            self._dev_id,
            node_key(self._inventory, self._node_type, self._addr),
            name=self._device_name,
            node_type=self._node_type,
        )

    def boost_state(self) -> BoostState:
        """Return derived boost metadata for this heater."""

        return derive_boost_state_from_domain(
            self._settings_resolver(), self.coordinator
        )


def _iter_boostable_inventory_nodes(
    inventory: Inventory,
) -> Iterable[tuple[str, str, str]]:
    """Yield boostable heater metadata from ``inventory``."""

    for metadata in inventory.iter_nodes_metadata(node_types=HEATER_NODE_TYPES):
        if not supports_boost(metadata.node):
            continue
        yield (metadata.node_type, metadata.addr, metadata.name)
