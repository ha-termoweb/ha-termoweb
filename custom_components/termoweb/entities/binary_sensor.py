"""Binary sensor entities for TermoWeb gateway connectivity and heaters."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import logging
from typing import Any

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from custom_components.termoweb.boost import supports_boost
from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.coordinator import StateCoordinator
from custom_components.termoweb.domain import DomainStateView, GatewayConnectionState
from custom_components.termoweb.domain.state import DomainState
from custom_components.termoweb.entities.heater import (
    BoostState,
    derive_boost_state_from_domain,
    log_skipped_nodes,
)
from custom_components.termoweb.i18n import (
    async_get_fallback_translations,
    attach_fallbacks,
)
from custom_components.termoweb.identifiers import build_heater_entity_unique_id
from custom_components.termoweb.inventory import (
    HEATER_NODE_TYPES,
    Inventory,
    normalize_node_addr,
    normalize_node_type,
)
from custom_components.termoweb.runtime import require_runtime
from custom_components.termoweb.utils import build_gateway_device_info

_LOGGER = logging.getLogger(__name__)

SettingsResolver = Callable[[], DomainState | None]


async def async_setup_entry(hass, entry, async_add_entities):
    """Set up connectivity and boost binary sensors for a config entry."""
    runtime = require_runtime(hass, entry.entry_id)
    coord: StateCoordinator = runtime.coordinator
    dev_id = runtime.dev_id

    fallbacks = await async_get_fallback_translations(hass, runtime)
    attach_fallbacks(coord, fallbacks)
    gateway = GatewayOnlineBinarySensor(coord, entry.entry_id, dev_id)

    inventory = runtime.inventory
    if not isinstance(inventory, Inventory):
        _LOGGER.error("TermoWeb heater setup missing inventory for device %s", dev_id)
        raise ValueError(  # noqa: TRY004
            "TermoWeb inventory unavailable for heater platform"
        )

    boost_entities: list[BinarySensorEntity] = []
    for node_type, addr_str, base_name in _iter_boostable_inventory_nodes(inventory):
        unique_id = build_heater_entity_unique_id(
            dev_id,
            node_type,
            addr_str,
            ":boost_active",
        )
        settings_resolver = _build_settings_resolver(
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
    async_add_entities([gateway, *boost_entities])


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
        self._attr_unique_id = f"{self._dev_id}_online"

    def _gateway_connection_state(self) -> GatewayConnectionState:
        """Return the gateway connection state for this device."""

        domain_view = getattr(self.coordinator, "domain_view", None)
        if isinstance(domain_view, DomainStateView):
            return domain_view.get_gateway_connection_state()
        return GatewayConnectionState()

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


class HeaterBoostActiveBinarySensor(
    CoordinatorEntity[StateCoordinator],
    BinarySensorEntity,
):
    """Binary sensor indicating whether a heater boost is active."""

    _attr_device_class = getattr(BinarySensorDeviceClass, "HEAT", "heat")
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
        if not canonical_type or not canonical_addr:
            msg = "node_type and addr must be provided"
            raise ValueError(msg)

        self._entry_id = entry_id
        self._dev_id = str(dev_id)
        self._node_type = canonical_type
        self._addr = canonical_addr
        self._attr_unique_id = unique_id
        self._inventory = inventory
        self._settings_resolver = settings_resolver
        resolved_device_name = device_name or name
        self._device_name = resolved_device_name if resolved_device_name else ""

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
        """Return whether the heater node exists in the inventory."""

        forward_map, _ = self._inventory.heater_address_map
        addresses = forward_map.get(self._node_type, [])
        return self._addr in addresses

    @property
    def device_info(self) -> DeviceInfo:
        """Expose Home Assistant device metadata for the heater."""

        model = "Accumulator" if self._node_type == "acm" else "Heater"
        return DeviceInfo(
            identifiers={(DOMAIN, self._dev_id, self._addr)},
            name=self._device_name,
            manufacturer="TermoWeb",
            model=model,
            via_device=(DOMAIN, self._dev_id),
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
        canonical_type = normalize_node_type(
            metadata.node_type,
            use_default_when_falsey=True,
        )
        canonical_addr = normalize_node_addr(
            metadata.addr,
            use_default_when_falsey=True,
        )
        if not canonical_type or not canonical_addr:
            continue
        yield (canonical_type, canonical_addr, metadata.name)


def _build_settings_resolver(
    coordinator: StateCoordinator,
    dev_id: str,
    node_type: str,
    addr: str,
) -> SettingsResolver:
    """Return callable resolving typed boost state for a heater node."""

    def _resolver() -> DomainState | None:
        view = getattr(coordinator, "domain_view", None)
        if isinstance(view, DomainStateView):
            return view.get_heater_state(node_type, addr)
        return None

    return _resolver
