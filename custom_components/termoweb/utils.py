"""Utility helpers shared across the TermoWeb integration."""

from __future__ import annotations

from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.loader import async_get_integration

from .const import CLOUD_CONFIGURATION_URL, DEVICE_BRAND_LABELS, DOMAIN, get_brand_label
from .inventory import normalize_node_addr
from .runtime import EntryRuntime, require_runtime


async def async_get_integration_version(hass: HomeAssistant) -> str:
    """Return the installed integration version string."""

    integration = await async_get_integration(hass, DOMAIN)
    return integration.version or "unknown"


def _entry_gateway_record(
    hass: HomeAssistant | None, entry_id: str | None
) -> EntryRuntime | None:
    """Return the running runtime of ``entry_id``, or None."""

    if hass is None or entry_id is None:
        return None
    try:
        return require_runtime(hass, entry_id)
    except LookupError:
        return None


def _link_via_device(
    info: DeviceInfo,
    hass: HomeAssistant | None,
    entry_id: str | None,
    parent: tuple[str, ...],
) -> None:
    """Set ``via_device_id`` to the entry's registered ``parent`` device, if any."""

    if hass is None or entry_id is None:
        return
    device = dr.async_get(hass).async_get_device_by_identifier(parent, entry_id)
    if device is not None:
        info["via_device_id"] = device.id


def _has_capability(entry_data: EntryRuntime | None, name: str) -> bool:
    """Return the running backend's capability flag; True (the cloud default) if unknown."""

    return entry_data is None or getattr(entry_data.backend.capabilities, name)


def apply_entry_device_overrides(
    info: DeviceInfo,
    entry_data: EntryRuntime | None,
    *,
    include_version: bool = False,
) -> DeviceInfo:
    """Return device info with brand and version overrides."""

    if entry_data is None:
        return info

    manufacturer: str | None = None
    brand = entry_data.brand

    if isinstance(brand, str) and brand.strip():
        manufacturer = brand.strip()
        if manufacturer in DEVICE_BRAND_LABELS:
            manufacturer = get_brand_label(manufacturer)  # brand key -> label

    if manufacturer:
        info["manufacturer"] = manufacturer

    if include_version:
        version = entry_data.version
        if version is not None:
            info["sw_version"] = str(version)

    return info


def _set_configuration_url(info: DeviceInfo, entry_data: EntryRuntime | None) -> None:
    """Link the device to the cloud web portal when the backend has one."""

    if _has_capability(entry_data, "web_portal"):
        info["configuration_url"] = CLOUD_CONFIGURATION_URL


def build_installation_device_info(
    hass: HomeAssistant | None,
    entry_id: str | None,
    dev_id: str,
) -> DeviceInfo:
    """Return canonical ``DeviceInfo`` for the installation (top-level site)."""

    identifiers = {(DOMAIN, str(dev_id), "site")}
    entry_data = _entry_gateway_record(hass, entry_id)
    info: DeviceInfo = DeviceInfo(
        identifiers=identifiers,
        manufacturer="TermoWeb",
        name="Site",
        model="Site",
    )
    _set_configuration_url(info, entry_data)
    info = apply_entry_device_overrides(info, entry_data)

    if entry_data is None:
        return info

    coordinator = entry_data.coordinator
    if coordinator is not None:
        gateway_name = getattr(coordinator, "gateway_name", None)
        if gateway_name not in (None, ""):
            info["name"] = str(gateway_name)

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
    entry_data = _entry_gateway_record(hass, entry_id)

    brand_label = "TermoWeb"
    if entry_data is not None and isinstance(entry_data.brand, str):
        brand_label = get_brand_label(entry_data.brand)

    info: DeviceInfo = DeviceInfo(
        identifiers=identifiers,
        manufacturer=brand_label,
        name=f"{brand_label} Gateway",
        model="Gateway/Controller",
    )
    if _has_capability(entry_data, "site_device"):
        _link_via_device(info, hass, entry_id, (DOMAIN, str(dev_id), "site"))
    _set_configuration_url(info, entry_data)

    info = apply_entry_device_overrides(
        info, entry_data, include_version=include_version
    )

    if entry_data is None:
        return info

    coordinator = entry_data.coordinator
    if coordinator is not None:
        model = getattr(coordinator, "gateway_model", None)
        if model not in (None, ""):
            info["model"] = str(model)

        device_metadata = getattr(coordinator, "device_metadata", None)
        if device_metadata is not None:
            fw_version = getattr(device_metadata, "fw_version", None)
            if fw_version not in (None, ""):
                info["sw_version"] = str(fw_version)
            serial_id = getattr(device_metadata, "serial_id", None)
            if serial_id not in (None, ""):
                info["serial_number"] = str(serial_id)

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
    display_name = (name or "").strip() or f"Power Monitor {normalized_addr}"
    entry_data = _entry_gateway_record(hass, entry_id)

    info: DeviceInfo = DeviceInfo(
        identifiers={identifier},
        manufacturer="TermoWeb",
        name=display_name,
        model="Power Monitor",
    )
    _link_via_device(info, hass, entry_id, (DOMAIN, str(dev_id)))
    translate_default_device_name(info, normalized_addr)

    return apply_entry_device_overrides(info, entry_data)


def build_node_device_info(
    hass: HomeAssistant | None,
    entry_id: str | None,
    dev_id: str,
    addr: str,
    *,
    name: str,
    model: str,
) -> DeviceInfo:
    """Return ``DeviceInfo`` for a heater, accumulator or thermostat node."""

    info: DeviceInfo = DeviceInfo(
        identifiers={(DOMAIN, str(dev_id), str(addr))},
        name=name,
        manufacturer="TermoWeb",
        model=model,
    )
    _link_via_device(info, hass, entry_id, (DOMAIN, str(dev_id)))
    translate_default_device_name(info, str(addr))
    return apply_entry_device_overrides(info, _entry_gateway_record(hass, entry_id))


# English default node names; keys match the ``device`` section of strings.json.
_DEFAULT_DEVICE_NAMES: dict[str, str] = {
    "heater": "Heater {addr}",
    "accumulator": "Accumulator {addr}",
    "thermostat": "Thermostat {addr}",
    "power_monitor": "Power Monitor {addr}",
    "node": "Node {addr}",
}


def translate_default_device_name(info: DeviceInfo, addr: str) -> DeviceInfo:
    """Attach a device translation key when ``info`` carries a default node name."""

    name = info.get("name")
    for key, template in _DEFAULT_DEVICE_NAMES.items():
        if name == template.format(addr=addr):
            info["translation_key"] = key
            info["translation_placeholders"] = {"addr": addr}
            break
    return info
