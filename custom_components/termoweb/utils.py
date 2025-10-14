"""Utility helpers shared across the TermoWeb integration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
import math
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.loader import async_get_integration as loader_async_get_integration
from homeassistant.util import dt as dt_util

from .const import DOMAIN
from .inventory import normalize_node_addr


async def async_get_integration(*args, **kwargs):
    """Proxy ``homeassistant.loader.async_get_integration`` for monkeypatching."""

    return await loader_async_get_integration(*args, **kwargs)


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


def apply_entry_device_overrides(
    info: DeviceInfo,
    entry_data: Mapping[str, Any] | None,
    *,
    include_version: bool = False,
) -> DeviceInfo:
    """Return device info with brand and version overrides."""

    if not isinstance(entry_data, Mapping):
        return info

    manufacturer: str | None = None

    brand = entry_data.get("brand")
    if isinstance(brand, str) and brand.strip():
        manufacturer = brand.strip()
    else:
        override = entry_data.get("manufacturer")
        if isinstance(override, str) and override.strip():
            manufacturer = override.strip()

    if manufacturer:
        info["manufacturer"] = manufacturer

    if include_version:
        version = entry_data.get("version")
        if version is not None:
            info["sw_version"] = str(version)

    return info


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
    info = apply_entry_device_overrides(
        info, entry_data, include_version=include_version
    )

    if not entry_data:
        return info

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


def build_power_monitor_device_info(
    hass: HomeAssistant | None,
    entry_id: str | None,
    dev_id: str,
    addr: str,
    *,
    name: str | None = None,
) -> DeviceInfo:
    """Return canonical ``DeviceInfo`` for a TermoWeb power monitor."""

    normalized_addr = normalize_node_addr(addr, use_default_when_falsey=True) or str(
        addr
    )
    identifier = (DOMAIN, str(dev_id), "pmo", normalized_addr)
    display_name = (name or "").strip()
    if not display_name:
        display_name = f"Power Monitor {normalized_addr}"

    info: DeviceInfo = DeviceInfo(
        identifiers={identifier},
        manufacturer="TermoWeb",
        name=display_name,
        model="Power Monitor",
        via_device=(DOMAIN, str(dev_id)),
    )

    entry_data = _entry_gateway_record(hass, entry_id)
    return apply_entry_device_overrides(info, entry_data)


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


def coerce_power_watts(value: Any) -> float | None:
    """Return a non-negative watt value parsed from ``value`` when possible."""

    if isinstance(value, Mapping):
        for key in ("value", "power", "watts", "w"):
            nested = value.get(key)
            coerced = coerce_power_watts(nested)
            if coerced is not None:
                return coerced
        return None

    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        for item in value:
            coerced = coerce_power_watts(item)
            if coerced is not None:
                return coerced
        return None

    multiplier = 1.0
    candidate = value

    if isinstance(candidate, str):
        cleaned = candidate.strip()
        if not cleaned:
            return None
        lowered = cleaned.lower()
        if lowered.endswith("kw"):
            multiplier = 1000.0
            cleaned = cleaned[:-2].strip()
        elif lowered.endswith("mw"):
            multiplier = 1_000_000.0
            cleaned = cleaned[:-2].strip()
        elif lowered.endswith("w"):
            cleaned = cleaned[:-1].strip()
        cleaned = cleaned.replace(",", "")
        candidate = cleaned

    numeric = float_or_none(candidate)
    if numeric is None or numeric < 0 or not math.isfinite(numeric):
        return None

    return numeric * multiplier


def _coerce_timestamp(value: Any) -> float | None:
    """Return a UNIX timestamp parsed from ``value`` when possible."""

    if isinstance(value, (int, float)):
        ts = float(value)
        if not math.isfinite(ts):
            return None
        if ts > 1_000_000_000_000:  # milliseconds
            ts /= 1000.0
        return ts
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        numeric = float_or_none(cleaned)
        if numeric is not None:
            return _coerce_timestamp(numeric)
        parsed = dt_util.parse_datetime(cleaned)
        if parsed is None:
            return None
        return dt_util.as_timestamp(parsed)
    return None


def extract_power_timestamp(payload: Any) -> float | None:
    """Return the newest timestamp discovered in ``payload``."""

    if isinstance(payload, Mapping):
        for key in (
            "timestamp",
            "ts",
            "t",
            "updated_at",
            "updatedAt",
            "time",
            "last_update",
            "lastUpdate",
        ):
            if key in payload:
                ts = _coerce_timestamp(payload.get(key))
                if ts is not None:
                    return ts
        for key in ("power", "status", "data", "metrics"):
            nested = payload.get(key)
            ts = extract_power_timestamp(nested)
            if ts is not None:
                return ts
        return None

    if isinstance(payload, Iterable) and not isinstance(payload, (str, bytes)):
        latest: float | None = None
        for item in payload:
            ts = extract_power_timestamp(item)
            if ts is None:
                continue
            if latest is None or ts > latest:
                latest = ts
        return latest

    return _coerce_timestamp(payload)


def extract_power_watts(payload: Any) -> float | None:
    """Return the first instantaneous power value discovered within ``payload``."""

    if isinstance(payload, Mapping):
        candidate_keys = (
            "instant_power",
            "instantPower",
            "power_w",
            "powerW",
            "power_watts",
            "watts",
            "power",
        )
        for key in candidate_keys:
            if key not in payload:
                continue
            value = payload.get(key)
            coerced = coerce_power_watts(value)
            if coerced is not None:
                return coerced
        for key in ("status", "data", "metrics"):
            nested = payload.get(key)
            coerced = extract_power_watts(nested)
            if coerced is not None:
                return coerced
        return None

    if isinstance(payload, Iterable) and not isinstance(payload, (str, bytes)):
        for item in payload:
            coerced = extract_power_watts(item)
            if coerced is not None:
                return coerced
        return None

    return coerce_power_watts(payload)


def format_timestamp_iso(timestamp: float | None) -> str | None:
    """Return an ISO timestamp string for ``timestamp`` when available."""

    if timestamp is None:
        return None
    try:
        dt = datetime.fromtimestamp(float(timestamp), tz=UTC)
    except (TypeError, ValueError, OSError, OverflowError):
        return None
    return dt.isoformat()
