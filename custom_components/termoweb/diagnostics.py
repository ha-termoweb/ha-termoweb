"""Diagnostics support for the TermoWeb integration."""

from __future__ import annotations

from collections.abc import Mapping
import inspect
import logging
import platform
from typing import Any, Final

from homeassistant.components.diagnostics import async_redact_data
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import CONF_BRAND, DEFAULT_BRAND, DOMAIN, get_brand_label
from .energy import SUMMARY_KEY_LAST_RUN
from .inventory import Inventory
from .utils import async_get_integration_version

_LOGGER = logging.getLogger(__name__)

SENSITIVE_FIELDS: Final = {
    "access_token",
    "authorization",
    "client_secret",
    "dev_id",
    "password",
    "refresh_token",
    "token",
    "username",
}


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: ConfigEntry
) -> Mapping[str, Any]:
    """Return a diagnostics payload for ``entry``."""

    domain_data = hass.data.get(DOMAIN)
    record: Mapping[str, Any] | None = None
    if isinstance(domain_data, Mapping):
        candidate = domain_data.get(entry.entry_id)
        if isinstance(candidate, Mapping):
            record = candidate

    inventory = Inventory.require_from_record(
        record,
        context=f"diagnostics for config entry {entry.entry_id}",
    )
    raw_count = len(inventory.nodes)
    metadata = list(inventory.iter_nodes_metadata())
    filtered_count = len(metadata)

    node_inventory = [
        {
            "name": meta.name,
            "addr": meta.addr,
            "type": meta.node_type,
        }
        for meta in metadata
    ]
    node_inventory.sort(key=lambda item: (str(item["addr"]), str(item["type"])))

    assert record is not None  # Inventory.require_from_record guarantees mapping

    version = record.get("version")
    if version is None:
        version = await async_get_integration_version(hass)
    version_str = str(version)

    brand_value: str | None = None
    if isinstance(record, Mapping):
        candidate_brand = record.get("brand")
        if isinstance(candidate_brand, str) and candidate_brand.strip():
            brand_value = candidate_brand.strip()
    if not brand_value:
        entry_brand = entry.data.get(CONF_BRAND)
        if isinstance(entry_brand, str) and entry_brand.strip():
            brand_value = entry_brand.strip()
    brand_label = get_brand_label(brand_value or DEFAULT_BRAND)

    ha_version = getattr(hass, "version", None)
    ha_version_str = str(ha_version) if ha_version is not None else "unknown"

    hass_config = getattr(hass, "config", None)
    time_zone = getattr(hass_config, "time_zone", None)
    time_zone_str = str(time_zone) if time_zone not in (None, "") else None

    diagnostics: dict[str, Any] = {
        "integration": {
            "domain": DOMAIN,
            "version": version_str,
            "brand": brand_label,
        },
        "home_assistant": {
            "version": ha_version_str,
            "python_version": platform.python_version(),
        },
        "installation": {
            "node_inventory": node_inventory,
        },
    }

    if time_zone_str is not None:
        diagnostics["home_assistant"]["time_zone"] = time_zone_str

    last_import = record.get(SUMMARY_KEY_LAST_RUN)
    energy_section: dict[str, Any] = {}
    if isinstance(last_import, Mapping):
        energy_section["last_run"] = dict(last_import)
    elif last_import is not None:
        energy_section["last_run"] = last_import
    if energy_section:
        diagnostics["energy_import"] = energy_section

    _LOGGER.debug(
        "Diagnostics inventory cache for %s: raw=%d, filtered=%d",
        entry.entry_id,
        raw_count,
        filtered_count,
    )

    try:
        redacted = async_redact_data(diagnostics, SENSITIVE_FIELDS)
        if inspect.isawaitable(redacted):
            redacted = await redacted
    except Exception:  # pragma: no cover - defensive
        _LOGGER.exception("Failed to redact diagnostics payload for %s", entry.entry_id)
        raise
    return redacted
