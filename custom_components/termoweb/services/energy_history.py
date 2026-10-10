"""The ``import_energy_history`` service."""

from __future__ import annotations

from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import ServiceValidationError
import voluptuous as vol

from custom_components.termoweb.backend.factory import backend_capabilities
from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.energy import (
    DEFAULT_MAX_HISTORY_DAYS,
    MAX_HISTORY_DAYS,
    async_import_energy_history,
)
from custom_components.termoweb.runtime import EntryRuntime

SERVICE_IMPORT_ENERGY_HISTORY = "import_energy_history"
IMPORT_ENERGY_HISTORY_SCHEMA = vol.Schema(
    {
        vol.Optional("reset_progress", default=False): bool,
        vol.Optional(
            "max_history_retrieval", default=DEFAULT_MAX_HISTORY_DAYS
        ): vol.All(vol.Coerce(int), vol.Range(min=1, max=MAX_HISTORY_DAYS)),
    }
)


async def async_register_import_energy_history_service(hass: HomeAssistant) -> None:
    """Register the import_energy_history service if it is missing."""

    if hass.services.has_service(DOMAIN, SERVICE_IMPORT_ENERGY_HISTORY):
        return

    async def _service_import_energy_history(call: ServiceCall) -> None:
        """Import energy history for every loaded entry whose backend supports it."""
        runtimes = [
            runtime
            for runtime in hass.data.get(DOMAIN, {}).values()
            if isinstance(runtime, EntryRuntime)
            and backend_capabilities(runtime.brand).energy_history
        ]
        if not runtimes:
            raise ServiceValidationError(
                "No loaded TermoWeb entry supports energy history import"
            )
        if any(runtime.energy_import_lock.locked() for runtime in runtimes):
            raise ServiceValidationError(
                "An energy history import is already running; wait for it to finish"
            )
        for runtime in runtimes:
            async with runtime.energy_import_lock:
                await async_import_energy_history(
                    hass,
                    runtime,
                    max_days=call.data["max_history_retrieval"],
                    reset_progress=call.data["reset_progress"],
                )

    hass.services.async_register(
        DOMAIN,
        SERVICE_IMPORT_ENERGY_HISTORY,
        _service_import_energy_history,
        schema=IMPORT_ENERGY_HISTORY_SCHEMA,
    )
