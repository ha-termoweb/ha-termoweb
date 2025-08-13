from __future__ import annotations

from datetime import timedelta
from typing import Final

# Domain
DOMAIN: Final = "termoweb"

# HTTP base & paths
API_BASE: Final = "https://control.termoweb.net"
TOKEN_PATH: Final = "/client/token"
DEVS_PATH: Final = "/api/v2/devs/"
CONNECTED_PATH_FMT: Final = (
    "/api/v2/devs/{dev_id}/connected"  # some fw return 404; we tolerate
)
NODES_PATH_FMT: Final = "/api/v2/devs/{dev_id}/mgr/nodes"
HTR_SAMPLES_PATH_FMT: Final = "/api/v2/devs/{dev_id}/htr/{addr}/samples"

# Public client creds (from APK v2.5.1)
BASIC_AUTH_B64: Final = "NTIxNzJkYzg0ZjYzZDZjNzU5MDAwMDA1OmJ4djRaM3hVU2U="

# Polling
DEFAULT_POLL_INTERVAL: Final = 120  # seconds
MIN_POLL_INTERVAL: Final = 30  # seconds
MAX_POLL_INTERVAL: Final = 3600  # seconds
STRETCHED_POLL_INTERVAL: Final = 2700  # seconds (45 minutes) when WS healthy ≥5m

# Heater energy polling interval when relying on push updates
HTR_ENERGY_UPDATE_INTERVAL: Final = timedelta(hours=1)

# UA / locale (matches app loosely; helps avoid quirky WAF rules)
USER_AGENT: Final = "TermoWeb/2.5.1 (Android; HomeAssistant Integration)"
ACCEPT_LANGUAGE: Final = "en-US,en;q=0.8"

# Integration version (also shown in Device Info)
# NOTE: Other modules may read the version from the manifest at runtime (DRY),
# but we keep this constant for compatibility where needed.
INTEGRATION_VERSION: Final = "1.0.0"

# Socket.IO namespace (used by ws_client_legacy)
WS_NAMESPACE: Final = "/api/v2/socket_io"

# --- Dispatcher signal helpers (WS → entities) ---


def signal_ws_data(entry_id: str) -> str:
    """Signal name for WS ‘data’ frames dispatched to platforms."""
    return f"{DOMAIN}_{entry_id}_ws_data"


def signal_ws_status(entry_id: str) -> str:
    """Signal name for WS status/health updates."""
    return f"{DOMAIN}_{entry_id}_ws_status"
