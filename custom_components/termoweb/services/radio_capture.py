"""Service that records every radio frame heard for a while and saves it as JSON."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.util import dt as dt_util
import voluptuous as vol

from custom_components.termoweb.backend.radio import RadioLinkError
from custom_components.termoweb.backend.radio.capture import redact, summarise
from custom_components.termoweb.backend.radio_client import RadioClient, RadioError
from custom_components.termoweb.const import CONF_RADIO_TYPE, DOMAIN, RADIO_TYPE_ESP32
from custom_components.termoweb.radio_survey import async_save_report
from custom_components.termoweb.runtime import require_runtime

_LOGGER = logging.getLogger(__name__)

SERVICE_RADIO_CAPTURE = "radio_capture"
CAPTURE_FORMAT = 1
CAPTURE_PREFIX = "termoweb_radio_capture"
DEFAULT_CAPTURE_S = 120
MIN_CAPTURE_S = 10
MAX_CAPTURE_S = 1800
RADIO_CAPTURE_SCHEMA = vol.Schema(
    {
        vol.Required("entry_id"): str,
        vol.Optional("seconds", default=DEFAULT_CAPTURE_S): vol.All(
            vol.Coerce(int), vol.Range(min=MIN_CAPTURE_S, max=MAX_CAPTURE_S)
        ),
        vol.Optional("redact", default=False): bool,
    }
)


def capture_payload(
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    client: RadioClient,
    radio_type: str,
    seconds: int,
    redacted: bool,
    version: str | None,
) -> dict[str, Any]:
    """Return the capture file content; the gateway MAC is never included."""

    info = client.gateway_info
    return {
        "format": CAPTURE_FORMAT,
        "created": dt_util.utcnow().isoformat(timespec="seconds"),
        "integration_version": version,
        "radio_type": radio_type,
        "listen_only": client.listen_only,
        "gateway": {
            "firmware": None if info is None else info.version,
            "freq": None if info is None else info.freq,
            "dialects": "A" if info is None or info.dialect is None else "A+B",
        },
        "capture_seconds": seconds,
        "redacted": redacted,
        "summary": summary,
        "records": records,
    }


async def async_register_radio_capture_service(hass: HomeAssistant) -> None:
    """Register the radio_capture service once."""

    if hass.services.has_service(DOMAIN, SERVICE_RADIO_CAPTURE):
        return

    async def _async_radio_capture(call: ServiceCall) -> dict[str, Any]:
        """Record the air, save the frames as JSON and return a short summary."""

        entry_id = call.data["entry_id"]
        seconds = int(call.data.get("seconds", DEFAULT_CAPTURE_S))
        redacted = bool(call.data.get("redact", False))
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
        _LOGGER.info("Radio capture for %s: recording %d s", entry_id, seconds)
        try:
            records = await client.async_capture(seconds)
        except (RadioError, RadioLinkError) as err:
            raise HomeAssistantError(f"Radio capture failed: {err}") from err
        if redacted:
            records = redact(records)
        summary = summarise(records)
        payload = capture_payload(
            records,
            summary,
            client=client,
            radio_type=runtime.config_entry.data.get(CONF_RADIO_TYPE, RADIO_TYPE_ESP32),
            seconds=seconds,
            redacted=redacted,
            version=runtime.version or None,
        )
        path = await async_save_report(hass, entry_id, payload, prefix=CAPTURE_PREFIX)
        result = {**summary, "redacted": redacted, "file": path}
        runtime.last_radio_capture = result
        _LOGGER.info(
            "Radio capture for %s: %d frames, %d undecodable",
            entry_id,
            summary["frames"],
            summary["undecodable"],
        )
        return result

    hass.services.async_register(
        DOMAIN,
        SERVICE_RADIO_CAPTURE,
        _async_radio_capture,
        schema=RADIO_CAPTURE_SCHEMA,
        supports_response=SupportsResponse.OPTIONAL,
    )
