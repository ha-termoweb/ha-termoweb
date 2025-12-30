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

# Public client creds (from APK v2.5.1)
BASIC_AUTH_B64: Final = "NTIxNzJkYzg0ZjYzZDZjNzU5MDAwMDA1OmJ4djRaM3hVU2U="

# Brand handling
CONF_BRAND: Final = "brand"
BRAND_TERMOWEB: Final = "termoweb"
BRAND_DUCAHEAT: Final = "ducaheat"
DEFAULT_BRAND: Final = BRAND_TERMOWEB

BRAND_LABELS: Final[Mapping[str, str]] = {
    BRAND_TERMOWEB: "TermoWeb",
    BRAND_DUCAHEAT: "Ducaheat",
}

BRAND_API_BASES: Final[Mapping[str, str]] = {
    BRAND_TERMOWEB: API_BASE,
    BRAND_DUCAHEAT: "https://api-tevolve.termoweb.net",
}

BRAND_BASIC_AUTH: Final[Mapping[str, str]] = {
    BRAND_TERMOWEB: BASIC_AUTH_B64,
    BRAND_DUCAHEAT: "NWM0OWRjZTk3NzUxMDM1MTUwNmM0MmRiOnRldm9sdmU=",
}

BRAND_SOCKETIO_PATHS: Final[Mapping[str, str]] = {
    BRAND_DUCAHEAT: "api/v2/socket_io",
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
}

BRAND_REQUESTED_WITH: Final[Mapping[str, str]] = {
    BRAND_TERMOWEB: TERMOWEB_REQUESTED_WITH,
    BRAND_DUCAHEAT: DUCAHEAT_REQUESTED_WITH,
}


def get_brand_api_base(brand: str) -> str:
    """Return API base URL for the selected brand."""

    base = BRAND_API_BASES.get(brand)
    if base:
        return base.rstrip("/")
    return API_BASE


def get_brand_basic_auth(brand: str) -> str:
    """Return Base64-encoded client credentials for the brand."""

    return BRAND_BASIC_AUTH.get(brand, BASIC_AUTH_B64)


def get_brand_label(brand: str) -> str:
    """Return human-readable brand label."""

    return BRAND_LABELS.get(brand, BRAND_LABELS[BRAND_TERMOWEB])


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


# Socket.IO namespace used by the websocket client implementation
WS_NAMESPACE: Final = "/api/v2/socket_io"

# --- Dispatcher signal helpers (WS → entities) ---


def signal_ws_data(entry_id: str) -> str:
    """Signal name for WS ‘data’ frames dispatched to platforms."""

    return f"{DOMAIN}_{entry_id}_ws_data"


def signal_ws_status(entry_id: str) -> str:
    """Signal name for WS status/health updates."""

    return f"{DOMAIN}_{entry_id}_ws_status"


# Polling
DEFAULT_POLL_INTERVAL: Final = 1800  # seconds (30 minutes)
MIN_POLL_INTERVAL: Final = 30  # seconds
MAX_POLL_INTERVAL: Final = 3600  # seconds

# Heater energy polling interval when relying on push updates
HTR_ENERGY_UPDATE_INTERVAL: Final = timedelta(hours=1)
