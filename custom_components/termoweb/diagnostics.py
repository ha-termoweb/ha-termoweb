"""Diagnostics support for the TermoWeb integration."""

from __future__ import annotations

from collections.abc import Mapping
import platform
from typing import Any, Final

from homeassistant.components.diagnostics import async_redact_data
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import CONF_BRAND, DEFAULT_BRAND, DOMAIN, get_brand_label
from .installation import ensure_snapshot
from .nodes import Node, ensure_node_inventory
from .utils import async_get_integration_version

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

    snapshot = ensure_snapshot(record or {})
    if snapshot is not None:
        inventory_source = list(snapshot.inventory)
    elif record is not None:
        inventory_source = ensure_node_inventory(record)
    else:
        inventory_source = []

    node_inventory: list[dict[str, Any]] = [
        node.as_dict() for node in inventory_source if isinstance(node, Node)
    ]
    node_inventory.sort(key=lambda item: (str(item.get("addr", "")), str(item.get("type", ""))))

    version = record.get("version") if isinstance(record, Mapping) else None
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

    return await async_redact_data(diagnostics, SENSITIVE_FIELDS)
