"""Service that records every radio frame heard for a while and saves it as JSON."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
import voluptuous as vol

from custom_components.termoweb.backend.radio import RadioLinkError
from custom_components.termoweb.backend.radio.capture import (
    FrameCapture,
    mask_mac,
    redact,
    redact_line,
    summarise,
)
from custom_components.termoweb.backend.radio_client import RadioClient, RadioError
from custom_components.termoweb.backend.radio_monitor import MONITOR_DIALECTS
from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.radio_survey import async_save_report
from custom_components.termoweb.runtime import require_runtime

_LOGGER = logging.getLogger(__name__)

SERVICE_RADIO_CAPTURE = "radio_capture"
CAPTURE_VERSION = 1
CAPTURE_PREFIX = "termoweb_radio_capture"
DEFAULT_CAPTURE_S = 120
MIN_CAPTURE_S = 10
MAX_CAPTURE_S = 1800
RESPONSE_KEYS = ("frames", "networks", "nodes", "opcodes")
RADIO_CAPTURE_SCHEMA = vol.Schema(
    {
        vol.Required("entry_id"): str,
        vol.Optional("seconds", default=DEFAULT_CAPTURE_S): vol.All(
            vol.Coerce(int), vol.Range(min=MIN_CAPTURE_S, max=MAX_CAPTURE_S)
        ),
        vol.Optional("redact", default=False): bool,
        vol.Optional("note"): str,
    }
)


def dialects_listened(client: RadioClient) -> list[str]:
    """Return the dialects the capture heard: both on a listen-only ESP32 entry."""

    info = client.gateway_info
    if client.listen_only and info is not None and info.dialect is not None:
        return [dialect.name for dialect in MONITOR_DIALECTS]
    return [client.dialect.name]


def capture_payload(
    capture: FrameCapture,
    client: RadioClient,
    *,
    redacted: bool,
    note: str | None,
) -> dict[str, Any]:
    """Return the capture file content; the gateway MAC is always masked."""

    frames, raw = capture.frames, capture.raw
    if redacted:
        frames, raw = redact(frames, raw)
    info = client.gateway_info
    gateway = None if info is None else mask_mac(info.raw)
    if gateway is not None and redacted:
        gateway = redact_line(gateway)
    return {
        "version": CAPTURE_VERSION,
        "started": capture.started,
        "ended": capture.ended,
        "gateway": gateway,
        "dialects_listened": dialects_listened(client),
        "note": note,
        "redacted": redacted,
        "summary": summarise(frames),
        "frames": frames,
        "raw": raw,
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
            capture = await client.async_capture(seconds)
        except (RadioError, RadioLinkError) as err:
            raise HomeAssistantError(f"Radio capture failed: {err}") from err
        payload = capture_payload(
            capture, client, redacted=redacted, note=call.data.get("note")
        )
        path = await async_save_report(hass, None, payload, prefix=CAPTURE_PREFIX)
        summary = payload["summary"]
        result = {"file": path, **{key: summary[key] for key in RESPONSE_KEYS}}
        runtime.last_radio_capture = result
        _LOGGER.info(
            "Radio capture for %s: %d frames, %d raw lines",
            entry_id,
            summary["frames"],
            len(payload["raw"]),
        )
        return result

    hass.services.async_register(
        DOMAIN,
        SERVICE_RADIO_CAPTURE,
        _async_radio_capture,
        schema=RADIO_CAPTURE_SCHEMA,
        supports_response=SupportsResponse.OPTIONAL,
    )
