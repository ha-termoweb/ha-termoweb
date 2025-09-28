"""Utility helpers shared across the TermoWeb integration."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.loader import async_get_integration

from .const import DOMAIN


async def async_get_integration_version(hass: HomeAssistant) -> str:
    """Return the installed integration version string."""

    integration = await async_get_integration(hass, DOMAIN)
    return integration.version or "unknown"


def _entry_gateway_record(
    hass: HomeAssistant | None, entry_id: str | None
) -> Mapping[str, Any] | None:
    """Return the mapping storing integration data for ``entry_id``."""

    if hass is None or entry_id is None:
        return None
    domain_data = hass.data.get(DOMAIN)
    if not isinstance(domain_data, Mapping):
        return None
    entry_data = domain_data.get(entry_id)
    if not isinstance(entry_data, Mapping):
        return None
    return entry_data


def build_gateway_device_info(
    hass: HomeAssistant | None,
    entry_id: str | None,
    dev_id: str,
    *,
    include_version: bool = True,
) -> DeviceInfo:
    """Return canonical ``DeviceInfo`` for the TermoWeb gateway."""

    identifiers = {(DOMAIN, str(dev_id))}
    info: DeviceInfo = DeviceInfo(
        identifiers=identifiers,
        manufacturer="TermoWeb",
        name="TermoWeb Gateway",
        model="Gateway/Controller",
        configuration_url="https://control.termoweb.net",
    )

    entry_data = _entry_gateway_record(hass, entry_id)
    if not entry_data:
        return info

    brand = entry_data.get("brand")
    if isinstance(brand, str) and brand.strip():
        info["manufacturer"] = brand.strip()

    version = entry_data.get("version")
    if include_version and version is not None:
        info["sw_version"] = str(version)

    coordinator = entry_data.get("coordinator")
    data: Mapping[str, Any] | None = None
    if coordinator is not None:
        data = getattr(coordinator, "data", None)
        if not isinstance(data, Mapping):
            data = None
    if data:
        gateway_data = data.get(str(dev_id))
        if isinstance(gateway_data, Mapping):
            raw = gateway_data.get("raw")
            if isinstance(raw, Mapping):
                model = raw.get("model")
                if model not in (None, ""):
                    info["model"] = str(model)

    return info


def float_or_none(value: Any) -> float | None:
    """Return value as ``float`` if possible, else ``None``.

    Converts integers, floats, and numeric strings to ``float`` while safely
    handling ``None`` and non-numeric inputs.
    """

    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            num = float(value)
        else:
            string_val = str(value).strip()
            if not string_val:
                return None
            num = float(string_val)
        return num if math.isfinite(num) else None
    except Exception:  # noqa: BLE001
        return None
