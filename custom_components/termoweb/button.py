from __future__ import annotations

from homeassistant.components.button import ButtonEntity
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN


async def async_setup_entry(hass, entry, async_add_entities):
    """Expose only a safe 'Force refresh' hub-level button per device."""
    data = hass.data[DOMAIN][entry.entry_id]
    coordinator = data["coordinator"]
    dev_id = data["dev_id"]
    async_add_entities([StateRefreshButton(coordinator, dev_id)])


class StateRefreshButton(CoordinatorEntity, ButtonEntity):
    """Button that requests an immediate coordinator refresh."""

    _attr_name = "Force refresh"
    _attr_has_entity_name = True

    def __init__(self, coordinator, dev_id: str) -> None:
        """Initialise the force-refresh button entity."""
        super().__init__(coordinator)
        self._dev_id = dev_id
        self._attr_unique_id = f"{DOMAIN}:{dev_id}:refresh"

    @property
    def device_info(self) -> DeviceInfo:
        """Return the Home Assistant device metadata for this gateway."""
        return DeviceInfo(
            identifiers={(DOMAIN, self._dev_id)},
            name="TermoWeb Gateway",
            manufacturer="TermoWeb",
            model="Gateway/Controller",
            configuration_url="https://control.termoweb.net",
        )

    async def async_press(self) -> None:
        """Request an immediate coordinator refresh when pressed."""
        await self.coordinator.async_request_refresh()
