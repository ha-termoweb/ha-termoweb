"""Button platform entities for TermoWeb gateways."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.button import ButtonEntity

try:  # pragma: no cover - fallback for stripped Home Assistant stubs in tests
    from homeassistant.exceptions import HomeAssistantError, ServiceNotFound
except ImportError:  # pragma: no cover - executed in unit test stubs

    class HomeAssistantError(Exception):
        """Fallback Home Assistant error used in unit tests."""

    class ServiceNotFound(HomeAssistantError):
        """Fallback service lookup error used in unit tests."""
from homeassistant.helpers.entity import DeviceInfo, EntityCategory
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .heater import (
    HeaterNodeBase,
    iter_heater_nodes,
    log_skipped_nodes,
    prepare_heater_platform_data,
)
from .utils import build_gateway_device_info

_LOGGER = logging.getLogger(__name__)

_SERVICE_REQUEST_ACCUMULATOR_BOOST = "request_accumulator_boost"


async def async_setup_entry(hass, entry, async_add_entities):
    """Expose hub refresh and accumulator boost helper buttons."""

    data = hass.data[DOMAIN][entry.entry_id]
    coordinator = data["coordinator"]
    dev_id = data["dev_id"]

    entities: list[ButtonEntity] = [
        StateRefreshButton(coordinator, entry.entry_id, dev_id)
    ]

    _, nodes_by_type, _, resolve_name = prepare_heater_platform_data(
        data,
        default_name_simple=lambda addr: f"Heater {addr}",
    )

    boost_entities: list[ButtonEntity] = []
    for node_type, node, addr_str, base_name in iter_heater_nodes(
        nodes_by_type,
        resolve_name,
    ):
        if node_type != "acm":
            continue
        supports_boost = getattr(node, "supports_boost", None)
        if callable(supports_boost) and not supports_boost():
            continue

        unique_prefix = f"{DOMAIN}:{dev_id}:{node_type}:{addr_str}:boost"
        boost_entities.extend(
            [
                AccumulatorBoostButton(
                    coordinator,
                    entry.entry_id,
                    dev_id,
                    addr_str,
                    base_name,
                    f"{unique_prefix}_30",
                    minutes=30,
                    node_type=node_type,
                ),
                AccumulatorBoostButton(
                    coordinator,
                    entry.entry_id,
                    dev_id,
                    addr_str,
                    base_name,
                    f"{unique_prefix}_60",
                    minutes=60,
                    node_type=node_type,
                ),
                AccumulatorBoostButton(
                    coordinator,
                    entry.entry_id,
                    dev_id,
                    addr_str,
                    base_name,
                    f"{unique_prefix}_120",
                    minutes=120,
                    node_type=node_type,
                ),
                AccumulatorBoostCancelButton(
                    coordinator,
                    entry.entry_id,
                    dev_id,
                    addr_str,
                    base_name,
                    f"{unique_prefix}_cancel",
                    node_type=node_type,
                ),
            ]
        )

    if boost_entities:
        entities.extend(boost_entities)
    log_skipped_nodes("button", nodes_by_type, logger=_LOGGER)

    async_add_entities(entities)


class StateRefreshButton(CoordinatorEntity, ButtonEntity):
    """Button that requests an immediate coordinator refresh."""

    _attr_name = "Force refresh"
    _attr_has_entity_name = True

    def __init__(self, coordinator, entry_id: str, dev_id: str) -> None:
        """Initialise the force-refresh button entity."""
        super().__init__(coordinator)
        self._entry_id = entry_id
        self._dev_id = dev_id
        self._attr_unique_id = f"{DOMAIN}:{dev_id}:refresh"

    @property
    def device_info(self) -> DeviceInfo:
        """Return the Home Assistant device metadata for this gateway."""
        return build_gateway_device_info(
            self.hass,
            getattr(self, "_entry_id", None),
            self._dev_id,
        )

    async def async_press(self) -> None:
        """Request an immediate coordinator refresh when pressed."""
        await self.coordinator.async_request_refresh()


class AccumulatorBoostButtonBase(HeaterNodeBase, ButtonEntity):
    """Base entity for TermoWeb accumulator boost helper buttons."""

    _attr_entity_category = EntityCategory.CONFIG
    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator,
        entry_id: str,
        dev_id: str,
        addr: str,
        base_name: str,
        unique_id: str,
        *,
        label: str,
        node_type: str | None = None,
    ) -> None:
        """Initialise an accumulator boost helper button."""

        HeaterNodeBase.__init__(
            self,
            coordinator,
            entry_id,
            dev_id,
            addr,
            f"{base_name} {label}",
            unique_id,
            device_name=base_name,
            node_type=node_type,
        )
        self._label = label
        self._attr_name = label

    @property
    def _service_minutes(self) -> int | None:
        """Return the minutes payload passed to the helper service."""

        return None

    async def async_press(self) -> None:
        """Invoke the helper service to update the accumulator boost state."""

        hass = self.hass
        if hass is None:
            return

        data: dict[str, Any] = {
            "entry_id": self._entry_id,
            "dev_id": self._dev_id,
            "node_type": self._node_type,
            "addr": self._addr,
        }
        minutes = self._service_minutes
        if minutes is not None:
            data["minutes"] = minutes

        try:
            await hass.services.async_call(
                DOMAIN,
                _SERVICE_REQUEST_ACCUMULATOR_BOOST,
                data,
                blocking=True,
            )
        except ServiceNotFound as err:
            _LOGGER.error(
                "Boost helper service unavailable for %s (%s): %s",
                self._addr,
                self._node_type,
                err,
            )
        except HomeAssistantError as err:  # pragma: no cover - defensive logging
            _LOGGER.error(
                "Boost helper service failed for %s (%s): %s",
                self._addr,
                self._node_type,
                err,
            )


class AccumulatorBoostButton(AccumulatorBoostButtonBase):
    """Button that starts an accumulator boost for a fixed duration."""

    _attr_icon = "mdi:timer-play"

    def __init__(
        self,
        coordinator,
        entry_id: str,
        dev_id: str,
        addr: str,
        base_name: str,
        unique_id: str,
        *,
        minutes: int,
        node_type: str | None = None,
    ) -> None:
        """Initialise the boost helper button for a fixed duration."""

        self._minutes = minutes
        super().__init__(
            coordinator,
            entry_id,
            dev_id,
            addr,
            base_name,
            unique_id,
            label=f"Boost {minutes} minutes",
            node_type=node_type,
        )

    @property
    def _service_minutes(self) -> int | None:
        """Return the hard-coded boost duration for the button."""

        return self._minutes


class AccumulatorBoostCancelButton(AccumulatorBoostButtonBase):
    """Button that stops the active accumulator boost session."""

    _attr_icon = "mdi:timer-off"

    def __init__(
        self,
        coordinator,
        entry_id: str,
        dev_id: str,
        addr: str,
        base_name: str,
        unique_id: str,
        *,
        node_type: str | None = None,
    ) -> None:
        """Initialise the helper button that cancels an active boost."""

        super().__init__(
            coordinator,
            entry_id,
            dev_id,
            addr,
            base_name,
            unique_id,
            label="Cancel boost",
            node_type=node_type,
        )

