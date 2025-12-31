"""Binary sensor entities for TermoWeb gateway connectivity and heaters."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import logging
from typing import Any

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.core import callback
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .boost import supports_boost
from .const import DOMAIN, signal_ws_data, signal_ws_status
from .coordinator import StateCoordinator
from .domain import DomainStateView
from .entity import GatewayDispatcherEntity
from .heater import (
    BoostState,
    DispatcherSubscriptionHelper,
    derive_boost_state,
    log_skipped_nodes,
)
from .i18n import async_get_fallback_translations, attach_fallbacks
from .identifiers import build_heater_entity_unique_id
from .inventory import (
    HEATER_NODE_TYPES,
    Inventory,
    normalize_node_addr,
    normalize_node_type,
)
from .utils import build_gateway_device_info

_LOGGER = logging.getLogger(__name__)

SettingsResolver = Callable[[], Mapping[str, Any] | None]


async def async_setup_entry(hass, entry, async_add_entities):
    """Set up connectivity and boost binary sensors for a config entry."""
    data = hass.data[DOMAIN][entry.entry_id]
    coord: StateCoordinator = data["coordinator"]
    dev_id = data["dev_id"]

    fallbacks = await async_get_fallback_translations(hass, data)
    attach_fallbacks(coord, fallbacks)
    gateway = GatewayOnlineBinarySensor(coord, entry.entry_id, dev_id)

    try:
        inventory = Inventory.require_from_context(container=data)
    except LookupError as err:
        _LOGGER.error("TermoWeb heater setup missing inventory for device %s", dev_id)
        raise ValueError("TermoWeb inventory unavailable for heater platform") from err

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
    GatewayDispatcherEntity,
    CoordinatorEntity[StateCoordinator],
    BinarySensorEntity,
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
            "model": data.get("model"),
            "ws_status": ws.get("status"),
            "ws_last_event_at": ws.get("last_event_at"),
            "ws_healthy_minutes": ws.get("healthy_minutes"),
        }

    @callback
    def _handle_gateway_dispatcher(self, payload: dict[str, Any]) -> None:
        """Handle websocket status broadcasts from the integration."""
        if payload.get("dev_id") != self._dev_id:
            return
        self.schedule_update_ha_state()


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
        self._ws_subscription = DispatcherSubscriptionHelper(self)

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

        settings = self._settings_resolver() or {}
        return derive_boost_state(settings, self.coordinator)

    async def async_added_to_hass(self) -> None:
        """Subscribe to websocket updates once the entity is added."""

        await super().async_added_to_hass()
        if self.hass is None:
            return
        signal = signal_ws_data(self._entry_id)
        if not signal:
            return
        self._ws_subscription.subscribe(
            self.hass,
            signal,
            self._handle_ws_message,
        )

    async def async_will_remove_from_hass(self) -> None:
        """Detach websocket listeners before removal."""

        self._ws_subscription.unsubscribe()
        await super().async_will_remove_from_hass()

    def _handle_ws_message(self, payload: Mapping[str, Any]) -> None:
        """Trigger a refresh when websocket payload targets this entity."""

        if not self._payload_targets_entity(payload):
            return
        self.schedule_update_ha_state()

    def _payload_targets_entity(self, payload: Mapping[str, Any]) -> bool:
        """Return True when ``payload`` references this heater node."""

        if payload.get("dev_id") != self._dev_id:
            return False

        candidate_type = payload.get("node_type")
        if candidate_type is not None:
            canonical = normalize_node_type(
                candidate_type, use_default_when_falsey=True
            )
            if canonical and canonical != self._node_type:
                return False

        addr = payload.get("addr")
        if addr is None:
            return True

        canonical_addr = normalize_node_addr(addr, use_default_when_falsey=True)
        if not canonical_addr:
            return False
        return canonical_addr == self._addr


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
    """Return callable resolving boost settings for a heater node."""

    def _resolver() -> Mapping[str, Any] | None:
        view = getattr(coordinator, "domain_view", None)
        if isinstance(view, DomainStateView):
            state = view.get_heater_state(node_type, addr)
            if state is not None:
                payload = state.to_legacy()
                return payload if isinstance(payload, Mapping) else None
        return None

    return _resolver
