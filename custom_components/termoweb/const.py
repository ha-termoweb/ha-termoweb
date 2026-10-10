"""Constants for the TermoWeb integration."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
from typing import Final

# Domain
DOMAIN: Final = "termoweb"

# HTTP base & paths
API_BASE: Final = "https://control.termoweb.net"
TOKEN_PATH: Final = "/client/token"
DEVS_PATH: Final = "/api/v2/devs/"
NODES_PATH_FMT: Final = "/api/v2/devs/{dev_id}/mgr/nodes"
NODE_SAMPLES_PATH_FMT: Final = "/api/v2/devs/{dev_id}/{node_type}/{addr}/samples"
GEO_DATA_PATH_FMT: Final = "/api/v2/devs/{dev_id}/geo_data"

# Public client creds (from APK v2.5.1)
BASIC_AUTH_B64: Final = "NTIxNzJkYzg0ZjYzZDZjNzU5MDAwMDA1OmJ4djRaM3hVU2U="

# Brand handling
CONF_BRAND: Final = "brand"
BRAND_TERMOWEB: Final = "termoweb"
BRAND_DUCAHEAT: Final = "ducaheat"
BRAND_TEVOLVE: Final = "tevolve"
BRAND_RADIO: Final = "radio"  # local ESP32 radio gateway, no cloud
# Listen-only radio entry: records traffic next to a real gateway, never transmits
BRAND_RADIO_MONITOR: Final = "radio_monitor"
RADIO_BRANDS: Final = frozenset({BRAND_RADIO, BRAND_RADIO_MONITOR})
DEFAULT_BRAND: Final = BRAND_TERMOWEB

# Radio entries: the gateway address plus what discovery learned from the air
CONF_HOST: Final = "host"
CONF_PORT: Final = "port"
CONF_DIALECT: Final = "dialect"
CONF_NETWORK_ID: Final = "network_id"  # 4 hex digits
CONF_NODES: Final = "nodes"
CONF_RADIO_TYPE: Final = "radio_type"  # "esp32" (default) or "nanocul"
CONF_DEVICE: Final = "device"  # nanoCUL serial port path or pyserial URL
CONF_RADIO_DEVICE_ID: Final = "radio_device_id"  # dev_id when firmware has no MAC
RADIO_TYPE_ESP32: Final = "esp32"
RADIO_TYPE_NANOCUL: Final = "nanocul"
CONF_RADIO_POWER: Final = (
    "radio_power"  # options: power manager limit/priority/rated power
)
CONF_RADIO_RESTORE: Final = (
    "radio_restore"  # options: heater settings saved before a factory reset
)
RADIO_GATEWAY_LABEL: Final = "Radio gateway"

BRAND_LABELS: Final[Mapping[str, str]] = {
    BRAND_TERMOWEB: "TermoWeb",
    BRAND_DUCAHEAT: "Ducaheat",
    BRAND_TEVOLVE: "Tevolve",
}

BRAND_API_BASES: Final[Mapping[str, str]] = {
    BRAND_TERMOWEB: API_BASE,
    BRAND_DUCAHEAT: "https://api-tevolve.termoweb.net",
    BRAND_TEVOLVE: "https://api-tevolve.termoweb.net",
}

BRAND_BASIC_AUTH: Final[Mapping[str, str]] = {
    BRAND_TERMOWEB: BASIC_AUTH_B64,
    BRAND_DUCAHEAT: "NWM0OWRjZTk3NzUxMDM1MTUwNmM0MmRiOnRldm9sdmU=",
    BRAND_TEVOLVE: "NWM0OWRjZTk3NzUxMDM1MTUwNmM0MmRiOnRldm9sdmU=",
}

BRAND_SOCKETIO_PATHS: Final[Mapping[str, str]] = {
    BRAND_DUCAHEAT: "api/v2/socket_io",
    BRAND_TEVOLVE: "api/v2/socket_io",
}

# UA / locale (matches app loosely; helps avoid quirky WAF rules)
USER_AGENT: Final = "TermoWeb/2.5.1 (Android; HomeAssistant Integration)"
DUCAHEAT_USER_AGENT: Final = "Ducaheat/1.40.1 (Android; HomeAssistant Integration)"

TERMOWEB_REQUESTED_WITH: Final = "com.casple.termoweb.v2"
DUCAHEAT_REQUESTED_WITH: Final = "net.termoweb.ducaheat.app"
ACCEPT_LANGUAGE: Final = "en-US,en;q=0.8"

BRAND_USER_AGENTS: Final[Mapping[str, str]] = {
    BRAND_TERMOWEB: USER_AGENT,
    BRAND_DUCAHEAT: DUCAHEAT_USER_AGENT,
    BRAND_TEVOLVE: DUCAHEAT_USER_AGENT,
}

BRAND_REQUESTED_WITH: Final[Mapping[str, str]] = {
    BRAND_TERMOWEB: TERMOWEB_REQUESTED_WITH,
    BRAND_DUCAHEAT: DUCAHEAT_REQUESTED_WITH,
    BRAND_TEVOLVE: DUCAHEAT_REQUESTED_WITH,
}

DUCAHEAT_BRANDS: Final[frozenset[str]] = frozenset({BRAND_DUCAHEAT, BRAND_TEVOLVE})


def get_brand_api_base(brand: str) -> str:
    """Return API base URL for the selected brand."""

    base = BRAND_API_BASES.get(brand)
    if base:
        return base.rstrip("/")
    return API_BASE


def get_brand_basic_auth(brand: str) -> str:
    """Return Base64-encoded client credentials for the brand."""

    return BRAND_BASIC_AUTH.get(brand, BASIC_AUTH_B64)


RADIO_BRAND_LABEL: Final = "Radio"  # not in BRAND_LABELS: that feeds the login form
CLOUD_CONFIGURATION_URL: Final = "https://control.termoweb.net"


def get_brand_label(brand: str) -> str:
    """Return human-readable brand label."""

    if brand in RADIO_BRANDS:
        return RADIO_BRAND_LABEL
    return BRAND_LABELS.get(brand, BRAND_LABELS[BRAND_TERMOWEB])


def brand_has_site_device(brand: str | None) -> bool:
    """Return False for listen-only radio entries: they have no site device."""

    return brand != BRAND_RADIO_MONITOR


def get_brand_configuration_url(brand: str | None) -> str | None:
    """Return the web portal URL for a cloud brand; None for the local radio."""

    return None if brand in RADIO_BRANDS else CLOUD_CONFIGURATION_URL


def get_brand_user_agent(brand: str) -> str:
    """Return the preferred User-Agent string for the brand."""

    return BRAND_USER_AGENTS.get(brand, USER_AGENT)


def get_brand_requested_with(brand: str) -> str | None:
    """Return the X-Requested-With header value for the brand."""

    return BRAND_REQUESTED_WITH.get(brand)


def get_brand_socketio_path(brand: str) -> str:
    """Return the Socket.IO path for the selected brand."""

    path = BRAND_SOCKETIO_PATHS.get(brand)
    if path:
        return path.lstrip("/")
    return "socket.io"


def uses_ducaheat_backend(brand: str) -> bool:
    """Return True when the brand maps to the Ducaheat backend."""

    return brand in DUCAHEAT_BRANDS


# Socket.IO namespace used by the websocket client implementation
WS_NAMESPACE: Final = "/api/v2/socket_io"


def signal_ws_status(entry_id: str) -> str:
    """Signal name for WS status/health updates."""

    return f"{DOMAIN}_{entry_id}_ws_status"


def signal_radio_frames(entry_id: str) -> str:
    """Signal name for the frame count of a listen-only radio entry."""

    return f"{DOMAIN}_{entry_id}_radio_frames"


# Polling
DEFAULT_POLL_INTERVAL: Final = 1800  # seconds (30 minutes)
MIN_POLL_INTERVAL: Final = 30  # seconds
# Heater energy polling interval when relying on push updates
HTR_ENERGY_UPDATE_INTERVAL: Final = timedelta(hours=1)
