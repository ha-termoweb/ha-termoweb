"""Service that runs a raw radio survey to capture an unknown heater dialect."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
import voluptuous as vol

from custom_components.termoweb.backend.radio import RadioLinkError
from custom_components.termoweb.backend.radio_client import RadioClient, RadioError
from custom_components.termoweb.const import (
    CONF_DIALECT,
    CONF_RADIO_TYPE,
    DOMAIN,
    RADIO_TYPE_ESP32,
)
from custom_components.termoweb.radio_survey import (
    async_analyse,
    async_save_report,
    report_payload,
    report_summary,
)
from custom_components.termoweb.runtime import require_runtime

_LOGGER = logging.getLogger(__name__)

SERVICE_RADIO_SURVEY = "radio_survey"
DEFAULT_SURVEY_S = 120
MIN_SURVEY_S = 10
MAX_SURVEY_S = 600
RADIO_SURVEY_SCHEMA = vol.Schema(
    {
        vol.Required("entry_id"): str,
        vol.Optional("seconds", default=DEFAULT_SURVEY_S): vol.All(
            vol.Coerce(int), vol.Range(min=MIN_SURVEY_S, max=MAX_SURVEY_S)
        ),
    }
)


async def async_register_radio_survey_service(hass: HomeAssistant) -> None:
    """Register the radio_survey service once."""

    if hass.services.has_service(DOMAIN, SERVICE_RADIO_SURVEY):
        return

    async def _async_radio_survey(call: ServiceCall) -> dict[str, Any]:
        """Survey the air, save a redacted report and return its summary."""

        entry_id = call.data["entry_id"]
        seconds = int(call.data.get("seconds", DEFAULT_SURVEY_S))
        try:
            runtime = require_runtime(hass, entry_id)
        except LookupError as err:
            raise ServiceValidationError(
                f"No loaded TermoWeb entry with id {entry_id}"
            ) from err
        client = runtime.client
        if not isinstance(client, RadioClient):
            raise ServiceValidationError(
                f"TermoWeb entry {entry_id} does not use a radio gateway"
            )
        _LOGGER.info("Radio survey for %s: listening %d s", entry_id, seconds)
        try:
            bursts = await client.async_survey(seconds)
        except (RadioError, RadioLinkError) as err:
            raise HomeAssistantError(f"Radio survey failed: {err}") from err
        report = await async_analyse(hass, bursts)
        data = runtime.config_entry.data
        payload = report_payload(
            report,
            seconds=seconds,
            gateway=client.gateway_info,
            radio_type=data.get(CONF_RADIO_TYPE, RADIO_TYPE_ESP32),
            dialect=data.get(CONF_DIALECT),
            version=runtime.version or None,
        )
        path = await async_save_report(hass, entry_id, payload)
        summary = report_summary(report, path)
        runtime.last_radio_survey = summary
        _LOGGER.info(
            "Radio survey for %s: %s (%d bursts)",
            entry_id,
            report.verdict,
            len(report.bursts),
        )
        return summary

    hass.services.async_register(
        DOMAIN,
        SERVICE_RADIO_SURVEY,
        _async_radio_survey,
        schema=RADIO_SURVEY_SCHEMA,
        supports_response=SupportsResponse.OPTIONAL,
    )
