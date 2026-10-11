"""Diagnostics support for the TermoWeb integration."""

from __future__ import annotations

from collections.abc import Mapping
import logging
import platform
import typing
from typing import Any, Final

from homeassistant.components.diagnostics import REDACTED, async_redact_data
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import __version__ as HA_VERSION
from homeassistant.core import HomeAssistant

from .backend.sanitize import mask_identifier
from .backend.ws_health import WsHealthTracker
from .const import DOMAIN, get_brand_label
from .runtime import EntryRuntime, require_runtime

_LOGGER = logging.getLogger(__name__)

SENSITIVE_FIELDS: Final = {
    "access_token",
    "authorization",
    "client_secret",
    "dev_id",
    "device_id",
    "password",
    "refresh_token",
    "serial",
    "serial_id",
    "token",
    "username",
}
# Location fields of the gateway's geo_data; the time zone stays for debugging.
GEO_FIELDS: Final = {"city", "country", "state", "zip"}


def _extract_websocket_clients(runtime: EntryRuntime) -> list[dict[str, Any]]:
    """Return websocket telemetry entries for diagnostics."""

    state_map = runtime.ws_state
    tracker_map = runtime.ws_trackers

    clients: list[dict[str, Any]] = []
    for dev_id, state in state_map.items():
        client_state = dict(state)
        entry: dict[str, Any] = {
            "device": mask_identifier(dev_id),
            "state": client_state,
        }

        tracker = tracker_map.get(dev_id)
        if isinstance(tracker, WsHealthTracker):
            entry["health"] = tracker.snapshot()

        clients.append(entry)

    clients.sort(key=lambda item: str(item.get("device")))
    return clients


def _redact_serial_names(
    entries: typing.Iterable[Mapping[str, Any]], nodes: typing.Iterable[Any]
) -> list[dict[str, Any]]:
    """Return the node entries with radio heater serials masked in their names."""

    serials = [serial for node in nodes if (serial := getattr(node, "serial", None))]
    redacted = []
    for entry in entries:
        name = entry.get("name")
        for serial in serials if isinstance(name, str) else ():
            name = name.replace(serial, REDACTED)
        redacted.append({**entry, "name": name})
    return redacted


def _radio_section(runtime: EntryRuntime, entry: ConfigEntry) -> dict[str, Any] | None:
    """Return the backend's own facts plus the last radio survey and capture."""

    section = runtime.backend.diagnostics(entry.data)
    if section is None:
        return None
    if runtime.last_radio_survey is not None:
        section["last_survey"] = dict(runtime.last_radio_survey)
    if runtime.last_radio_capture is not None:
        section["last_capture"] = {
            key: value
            for key, value in runtime.last_radio_capture.items()
            if key != "networks"  # network ids stay out of diagnostics
        }
    return section


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: ConfigEntry
) -> Mapping[str, typing.Any]:
    """Return a diagnostics payload for ``entry``."""

    runtime = require_runtime(hass, entry.entry_id)
    inventory = runtime.inventory
    snapshot = inventory.snapshot()
    raw_count = snapshot.raw_count
    filtered_count = snapshot.filtered_count
    node_inventory = _redact_serial_names(snapshot.node_inventory, inventory.nodes)

    installation_section: dict[str, Any] = {
        "node_inventory": node_inventory,
    }

    coordinator = runtime.coordinator
    device_metadata = (
        getattr(coordinator, "device_metadata", None) if coordinator else None
    )
    if device_metadata is not None:
        installation_section["name"] = getattr(device_metadata, "name", None)
        installation_section["model"] = getattr(device_metadata, "model", None)
        installation_section["fw_version"] = getattr(
            device_metadata, "fw_version", None
        )
        geo_data = getattr(device_metadata, "geo_data", None)
        if geo_data is not None:
            installation_section["geo_data"] = async_redact_data(
                {
                    "country": geo_data.country,
                    "state": geo_data.state,
                    "city": geo_data.city,
                    "tz_code": geo_data.tz_code,
                    "zip": geo_data.zip,
                },
                GEO_FIELDS,
            )

    diagnostics: dict[str, Any] = {
        "integration": {
            "domain": DOMAIN,
            "version": runtime.version,
            "brand": get_brand_label(runtime.brand),
        },
        "home_assistant": {
            "version": HA_VERSION,
            "python_version": platform.python_version(),
            "time_zone": hass.config.time_zone,
        },
        "site": installation_section,
    }

    last_import = runtime.last_energy_import_summary
    if last_import is not None:
        diagnostics["energy_import"] = {"last_run": dict(last_import)}

    ws_clients = _extract_websocket_clients(runtime)
    if ws_clients:
        diagnostics["websocket"] = {"clients": ws_clients}

    radio_section = _radio_section(runtime, entry)
    if radio_section is not None:
        diagnostics["radio"] = radio_section

    _LOGGER.debug(
        "Diagnostics inventory cache for %s: raw=%d, filtered=%d",
        entry.entry_id,
        raw_count,
        filtered_count,
    )

    return async_redact_data(diagnostics, SENSITIVE_FIELDS)
